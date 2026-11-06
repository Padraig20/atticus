"""Test distilled ATTICUS models with the bundled MetaXcan S-PrediXcan CLI."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, nullcontext
import csv
import gzip
import json
import logging
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.distillation.genotypes import TEMPLATE
from src.preprocessing.geuvadis import ROOT
from src.tracking import Tracker, add_tracking_args
from src.twas.gwas import read_gwas, write_gwas
from src.twas.metaxcan import NUMERIC, run_spredixcan, source_hash
from src.twas.models import (cell_type_indices, draw_name, export_cell, export_config, external_covariance_config,
                             initialize_cache, inspect_students, open_students, student_file_config)
from src.twas.reference import LDReference, fingerprint
from src.twas.statistics import benjamini_hochberg_log, summarize_draws


IDENTIFIERS = ["gene", "cell_type", "chrom", "tss", "mode"]
DRAW_FIELDS = [*IDENTIFIERS, "member", "bootstrap", "status", *NUMERIC, "log_pvalue", "neg_log10_pvalue"]
SUMMARY_FIELDS = ["gene", "cell_type", "zscore", "effect_size", "pvalue", "neg_log10_pvalue",
                  "qvalue", "n_draws_tested", "status"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--models", type=Path, required=True,
                        help="Native students.h5, or a model directory containing students.h5")
    parser.add_argument("--gwas", type=Path, required=True,
                        help="GWAS TSV[.gz]; defaults match data/data-for-the-paper/gwas/gwas-*.tsv.gz")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--cell-type", help="Test only this cell type; spaces/underscores are interchangeable (default: all)")
    parser.add_argument("--genotypes", type=Path,
                        help=f"Rebuild LD from {TEMPLATE}.bed/.bim/.fam in this directory instead of using saved covariance")
    parser.add_argument("--model-cache", type=Path,
                        help="Reusable PredictDB/LD exports; defaults to the distillation models/ directory when available")
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent MetaXcan draw processes")
    parser.add_argument("--fdr", type=float, default=0.05, help="BH threshold for significant-gene counts in run.json")
    for name, default in (("snp", "ID"), ("effect", "ALT"), ("other", "REF"), ("beta", "ES"), ("se", "SE")):
        parser.add_argument(f"--{name}-column", default=default)
    add_tracking_args(parser)
    args = parser.parse_args(argv)
    if args.jobs < 1 or not 0 < args.fdr < 1:
        parser.error("jobs must be positive and fdr must be in (0, 1)")
    return args


def write_json(path, value):
    partial = Path(str(path) + ".partial")
    partial.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    partial.replace(path)


def cell_results(handle, cell, name, mode, shape, plan, results, gwas_snps, draw_writer, fdr):
    summaries = []
    for gene, item in plan.items():
        identity = dict(gene=gene, cell_type=name, chrom=item["chrom"], tss=item["tss"], mode=mode)
        if item["status"] == "fitted":
            student = handle["genes"][gene][str(cell)]
            weights = student["coefficient_draws"][:]
            snps = student["snp_ids"].asstr()[:]
            matched = np.array([snp in gwas_snps for snp in snps], dtype=bool)
        rows = []
        for member in range(shape[0]):
            for bootstrap in range(shape[1]):
                result = {key: np.nan for key in (*NUMERIC, "log_pvalue", "neg_log10_pvalue")}
                result.update(n_snps_in_model=0, n_snps_in_cov=0, n_snps_used=0, status=item["status"])
                if item["status"] == "fitted":
                    count = int(np.count_nonzero(weights[member, bootstrap]))
                    used = int(np.count_nonzero(weights[member, bootstrap, matched]))
                    result.update(n_snps_in_model=count, n_snps_in_cov=count,
                                  status="zero_weights" if not count else "no_matched_snps")
                    returned = results[member, bootstrap].get(gene)
                    if returned is not None:
                        result.update(returned)
                    elif used:
                        raise RuntimeError(f"MetaXcan omitted {gene}, cell {name}, draw {member}/{bootstrap} despite matching SNPs")
                draw_writer.writerow({**identity, "member": member, "bootstrap": bootstrap, **result})
                rows.append(result)
        arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
        summary = summarize_draws(arrays)
        if item["status"] != "fitted":
            summary["status"] = item["status"]
        summaries.append({"gene": gene, "cell_type": name, **summary})
    logs = np.array([row.pop("log_pvalue") for row in summaries])
    logq = benjamini_hochberg_log(logs)
    tested = int(np.count_nonzero(~np.isnan(logs)))
    threshold = 0.05 / tested if tested else None
    for row, value in zip(summaries, logq):
        row["qvalue"] = float(np.exp(value))
    correction = {"tested_genes": tested, "significant_fdr": int(np.count_nonzero(logq < np.log(fdr))),
                  "bonferroni_threshold": threshold,
                  "significant_bonferroni": int(np.count_nonzero(logs < np.log(threshold))) if tested else 0}
    return summaries, correction


def log_summary(tracker, manifest):
    corrections = manifest["multiple_testing"]
    columns = ["cell_type", "tested_genes", "significant_bonferroni", "significant_fdr", "bonferroni_threshold"]
    tracker.table("twas/by_cell_type", columns,
                  [[name, *(values[key] for key in columns[1:])] for name, values in corrections.items()])
    hits = sum(values["significant_bonferroni"] for values in corrections.values())
    tracker.summarize({"twas/tested_gene_cell_pairs": manifest["tested_gene_cell_pairs"],
                       "twas/tested_draws": manifest["tested_draws"],
                       "twas/mean_bonferroni_genes_per_cell_type": hits / len(corrections) if corrections else None,
                       **{f"gwas/{key}": manifest["gwas_matching"][key]
                          for key in ("matched_snps", "model_snps", "allele_mismatch", "invalid_statistics")}})


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    metaxcan = ROOT / "metaxcan/software"
    if args.models.is_dir():
        args.models = args.models / "students.h5"
    with Tracker(args, "twas") as tracker, open_students(args.models, args.cell_type) as students, threadpool_limits(limits=1):
        mode, shape, samples, cells, plan, catalog = inspect_students(students, cell_type=args.cell_type)
        cell_indices = cell_type_indices(students, args.cell_type)
        tracker.configure({"distillation_mode": mode, "n_genes": len(plan), "n_cell_types": len(cells),
                           "n_ld_individuals": len(samples), "n_draws": int(np.prod(shape))})
        chromosomes = sorted({item["chrom"] for item in plan.values() if item["catalog"]}, key=int)
        stored = bool(students.attrs.get("covariances_complete", False)) and args.genotypes is None
        if stored:
            genotype_files = json.loads(students["config"].asstr()[()])["genotype_files"]
            logging.info("Using saved covariance; PLINK files are not needed")
        else:
            if args.genotypes is None:
                raise ValueError("Saved covariance is incomplete; finish distillation or pass --genotypes to rebuild LD")
            genotype_files = [fingerprint(args.genotypes / f"{TEMPLATE.format(chrom=chrom)}.{ext}")
                              for chrom in chromosomes for ext in ("bed", "bim", "fam")]
        default_cache = args.models.parent / "models" if stored else args.output / "models"
        if stored and len(cells) < len(students["cell_types"]):
            default_cache = args.models.parent / f"models-cell-{cell_indices[0]:04d}"
        args.model_cache = (args.model_cache or default_cache).resolve()
        model_config = export_config(args.models, mode, shape, samples, cells, plan, genotype_files, stored=stored)
        model_config.update(student_file_config(students, args.models, args.cell_type))
        if stored:
            model_config.update(external_covariance_config(students, args.models, cells, shape, plan))
        ordered = model_config["covariance_source"] == "external_metaxcan_text"
        engine_hash = source_hash(metaxcan, ordered=ordered)
        config = {**model_config, "gwas": fingerprint(args.gwas), "metaxcan_source_sha256": engine_hash,
                  "metaxcan_script": str(metaxcan / "SPrediXcan.py"), "python": sys.executable,
                  "model_cache": str(args.model_cache), "fdr": args.fdr,
                  "gwas_columns": {key: getattr(args, f"{key}_column") for key in ("snp", "effect", "other", "beta", "se")},
                  "aggregation": "equal-weight ACAT over available draw p-values; mean zscore and effect_size",
                  "multiple_testing": "tested genes within each cell type; BH and Bonferroni (0.05)"}
        args.output.mkdir(parents=True, exist_ok=True)
        manifest_path = args.output / "run.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["config"] != config:
                raise ValueError("TWAS inputs or settings changed; use a new output directory")
            if (previous.get("complete") and previous.get("summary_columns") == SUMMARY_FIELDS
                    and all(Path(p[0]).is_file() and fingerprint(p[0]) == p for p in previous["outputs"])):
                logging.info("Completed TWAS already exists in %s", args.output)
                log_summary(tracker, previous)
                return
        elif any((args.output / name).exists() for name in ("associations.tsv", "draws.tsv.gz", "metaxcan")):
            raise ValueError("Output files exist without run.json; use a new output directory")
        manifest = {"format": "atticus_twas_summary", "format_version": 1, "config": config,
                    "summary_columns": SUMMARY_FIELDS, "multiple_testing": {}, "complete": False}
        write_json(manifest_path, manifest)
        initialize_cache(args.model_cache, model_config)
        logging.info("Reading GWAS for %d model SNPs", len(catalog))
        gwas, diagnostics = read_gwas(args.gwas, catalog, **{f"{k}_column": v for k, v in config["gwas_columns"].items()})
        normalized_gwas = args.output / "gwas.model-snps.tsv.gz"
        write_gwas(normalized_gwas, gwas)
        manifest["gwas_matching"] = diagnostics
        write_json(manifest_path, manifest)
        logging.info("Matched %d SNPs; %s testing: %d draws, %d cell types, %d LD donors",
                     len(gwas), mode, np.prod(shape), len(cells), len(samples))
        associations = args.output / "associations.tsv"
        draws_path = args.output / "draws.tsv.gz"
        partial_summary, partial_draws = Path(str(associations) + ".partial"), Path(str(draws_path) + ".partial")
        tested_pairs, tested_draws = 0, 0
        reference_context = (nullcontext(None) if stored else
                             closing(LDReference(args.model_cache / "ld_reference.h5", args.genotypes, samples, chromosomes, args.models)))
        with (reference_context as reference,
              partial_summary.open("w", newline="") as summary_file,
              gzip.open(partial_draws, "wt", newline="") as draws_file,
              ThreadPoolExecutor(max_workers=args.jobs) as pool):
            # Read PLINK once per chromosome before exporting in MetaXcan's required gene order.
            if reference is not None:
                for i, (gene, item) in enumerate(sorted(plan.items(), key=lambda x: (int(x[1]["chrom"]), x[1]["tss"])), start=1):
                    if item["catalog"]:
                        reference.get(gene, item["chrom"], item["catalog"])
                    if i % 100 == 0:
                        logging.info("Prepared LD reference for %d / %d genes", i, len(plan))
            summary_writer = csv.DictWriter(summary_file, SUMMARY_FIELDS, delimiter="\t", lineterminator="\n")
            draw_writer = csv.DictWriter(draws_file, DRAW_FIELDS, delimiter="\t", lineterminator="\n")
            summary_writer.writeheader()
            draw_writer.writeheader()
            for completed_cells, (cell, name) in enumerate(zip(cell_indices, cells), start=1):
                logging.info("Preparing MetaXcan models for %s (%d / %d)", name, completed_cells, len(cells))
                label = f"cell_{cell:04d}"
                draws = export_cell(students, cell, shape, plan, reference, args.model_cache / label)

                def test_draw(draw):
                    if not draw["n_weights"]:
                        return {}
                    output = args.output / "metaxcan" / label / f"{draw_name(draw['member'], draw['bootstrap'])}.csv"
                    result = run_spredixcan(metaxcan, draw, normalized_gwas, output)
                    if set(result) - set(plan):
                        raise RuntimeError(f"Unexpected genes in MetaXcan results: {output}")
                    logging.info("MetaXcan finished %s, member %d bootstrap %d", name, draw["member"], draw["bootstrap"])
                    return result

                results = {(draw["member"], draw["bootstrap"]): result for draw, result in zip(draws, pool.map(test_draw, draws))}
                summary, correction = cell_results(students, cell, name, mode, shape, plan, results, gwas, draw_writer, args.fdr)
                summary_writer.writerows(summary)
                manifest["multiple_testing"][name] = correction
                tested_pairs += sum(row["n_draws_tested"] > 0 for row in summary)
                tested_draws += sum(row["n_draws_tested"] for row in summary)
                tracker.log({"progress/cell_types_completed": completed_cells, "cell_type": name,
                             "twas/tested_genes": correction["tested_genes"],
                             "twas/significant_bonferroni": correction["significant_bonferroni"],
                             "twas/significant_fdr": correction["significant_fdr"]})
        partial_summary.replace(associations)
        partial_draws.replace(draws_path)
        manifest.update(complete=bool(tested_pairs), tested_gene_cell_pairs=tested_pairs, tested_draws=tested_draws,
                        outputs=[fingerprint(path) for path in (associations, draws_path, normalized_gwas)])
        write_json(manifest_path, manifest)
        log_summary(tracker, manifest)
        if not tested_pairs:
            raise ValueError(f"MetaXcan returned no testable associations; inspect draw statuses and logs in {args.output}")
    logging.info("Saved %d gene/cell-type associations (%d draw tests) to %s", tested_pairs, tested_draws, associations)


if __name__ == "__main__":
    main()
