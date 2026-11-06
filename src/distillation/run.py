"""Distill ATTICUS into heterozygote-aware (default) or additive elastic nets."""

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import logging
import math
from pathlib import Path
import sys
import zlib

import h5py
import numpy as np
from threadpoolctl import threadpool_limits

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.distillation.genotypes import Chromosome, TEMPLATE
from src.distillation.covariance import save_covariances
from src.distillation.predict import fingerprint, strings
from src.distillation.student import fit_student, normal_targets
from src.preprocessing.geuvadis import ROOT
from src.tracking import Tracker, add_tracking_args, run_config
from src.twas.models import prepare_distillation_exports


FIT_METRICS = ("mean_heldout_spearman_full", "mean_heldout_spearman_projection",
               "mean_nonzero_dosage_per_draw", "mean_nonzero_heterozygote_per_draw",
               "mean_nonzero_projected_per_draw")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-y", "--targets", type=Path, required=True, help="Teacher predictions .h5 from distillation/predict.py")
    parser.add_argument("--genotypes", type=Path, default=ROOT / "data/geuvadis/plink",
                        help=f"Directory containing {TEMPLATE}.bed/.bim/.fam (the only supported template)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output directory for students.h5, run.json, and MetaXcan models/covariances")
    parser.add_argument("--mode", choices=["probabilistic", "deterministic"], default="probabilistic")
    parser.add_argument("--additive", action="store_true", help="Fit dosage-only students instead of dosage plus heterozygote indicators")
    parser.add_argument("--half-window", type=int, default=500_000, help="Cis-window half-width around the TSS, in bp")
    parser.add_argument("--maf", type=float, default=0.05, help="Minimum MAF in the genotype reference panel")
    parser.add_argument("--bootstraps", type=int, default=5, help="Replicates per member in probabilistic mode")
    parser.add_argument("--alpha", type=float, help="Fixed elastic-net penalty; otherwise selected by training-fold CV")
    parser.add_argument("--l1-ratio", type=float, default=0.5)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--alphas", type=int, default=15)
    parser.add_argument("--sigma-floor", type=float, default=1e-4, help="Aleatoric SD floor after inverse-normal transformation")
    parser.add_argument("--max-iter", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap seed (gene-specific deterministic offsets)")
    parser.add_argument("--split-seed", type=int, default=42, help="Fixed 80/20 donor split shared by all genes and modes")
    parser.add_argument("--jobs", type=int, default=1, help="Cell types to fit concurrently for each gene")
    add_tracking_args(parser)
    args = parser.parse_args(argv)
    if min(args.half_window, args.bootstraps, args.alphas, args.max_iter, args.jobs) < 1 or args.cv < 2:
        parser.error("Window, bootstraps, alpha count, iterations and jobs must be positive; CV must be at least 2")
    if min(args.seed, args.split_seed) < 0:
        parser.error("Seeds must be nonnegative")
    if not 0 <= args.maf <= 0.5 or not 0 < args.l1_ratio <= 1:
        parser.error("MAF must be in [0, 0.5] and l1-ratio in (0, 1]")
    if (not math.isfinite(args.sigma_floor) or args.sigma_floor <= 0
            or args.alpha is not None and (not math.isfinite(args.alpha) or args.alpha <= 0)):
        parser.error("Sigma floor and alpha must be finite and positive")
    return args


def split_samples(samples, seed):
    if len(samples) < 5 or len(set(samples)) != len(samples):
        raise ValueError("Expected at least five unique individuals in the teacher predictions")
    permutation = np.random.default_rng(seed).permutation(sorted(samples))
    n_train = 4 * len(samples) // 5
    lookup = {sample: i for i, sample in enumerate(samples)}
    return tuple(np.asarray([lookup[sample] for sample in sorted(ids)])
                 for ids in (permutation[:n_train], permutation[n_train:]))


def same_fit_config(saved, current):
    saved = json.loads(saved)
    def fit_arguments(config):
        return {k: v for k, v in config["arguments"].items() if k != "jobs" and not k.startswith("wandb_")}
    saved["arguments"] = fit_arguments(saved)
    current = {**current, "arguments": fit_arguments(current)}
    return saved == current


def bounded_results(pool, function, items, jobs):
    """Keep at most jobs fits pending while the caller saves each covariance."""
    items = iter(items)
    pending = deque()
    for _ in range(jobs):
        item = next(items, None)
        if item is None:
            break
        pending.append(pool.submit(function, item))
    while pending:
        yield pending.popleft().result()
        item = next(items, None)
        if item is not None:
            pending.append(pool.submit(function, item))


def write_student(group, result, snps):
    selected = result["snp_indices"]
    for name, values in snps.items():
        values = values[selected]
        if values.dtype.kind in "UOS":
            strings(group, name, values)
        else:
            group.create_dataset(name, data=values)
    for name, value in result.items():
        if name == "snp_indices":
            continue
        if isinstance(value, dict):
            group.attrs[name] = json.dumps(value, allow_nan=False)
        elif np.ndim(value) == 0:
            group.attrs[name] = value
        else:
            group.create_dataset(name, data=value, **({"compression": "gzip"} if np.size(value) else {}))
    group.attrs.update(coefficient_scale="raw_A1_dosage", projection="beta + (1 - 2p) * delta")


class FitProgress:
    """Cumulative diagnostics for fits computed during the current invocation."""

    def __init__(self):
        self.totals = np.zeros(len(FIT_METRICS))
        self.counts = np.zeros(len(FIT_METRICS), dtype=int)
        self.fits = 0

    def update(self, result):
        values = [result["evaluation"][f"test_{predictor}"]["spearman"]
                  for predictor in ("full", "additive_projection")]
        values.extend(np.count_nonzero(np.abs(result[key]) > 1e-12, axis=-1)
                      for key in ("dosage_draws", "heterozygote_draws", "coefficient_draws"))
        for column, value in enumerate(values):
            if value is not None:
                value = np.asarray(value, dtype=float)
                finite = np.isfinite(value)
                self.totals[column] += value[finite].sum()
                self.counts[column] += int(finite.sum())
        self.fits += 1

    def metrics(self):
        values = {name: float(total / count) if count else None
                  for name, total, count in zip(FIT_METRICS, self.totals, self.counts)}
        values.update(n_fits_this_run=self.fits, models_with_heldout_spearman_full=int(self.counts[0]),
                      models_with_heldout_spearman_projection=int(self.counts[1]))
        return {f"distillation/new_fits/{name}": value for name, value in values.items()}


def log_student_summary(path, tracker, max_iter):
    """Aggregate saved fits, including resumed work, without retaining coefficient matrices."""
    if not tracker.enabled:
        return
    metrics = [*FIT_METRICS, "fraction_draws_at_iteration_limit"]
    with h5py.File(path, "r") as output:
        cell_types = output["cell_types"].asstr()[:]
        totals = np.zeros((len(cell_types), len(metrics)))
        counts = np.zeros_like(totals, dtype=int)
        fitted = np.zeros(len(cell_types), dtype=int)
        zero_projected = np.zeros_like(fitted)
        skipped = 0
        for gene in output["genes"].values():
            if gene.attrs["status"] != "fitted":
                skipped += 1
                continue
            for cell in range(len(cell_types)):
                student = gene[str(cell)]
                evaluation = json.loads(student.attrs["evaluation"])
                nonzeros = [np.count_nonzero(np.abs(student[key][:]) > 1e-12, axis=-1)
                            for key in ("dosage_draws", "heterozygote_draws", "coefficient_draws")]
                values = [evaluation[f"test_{predictor}"]["spearman"]
                          for predictor in ("full", "additive_projection")]
                values.extend([*nonzeros, student["iterations"][:] >= max_iter])
                fitted[cell] += 1
                zero_projected[cell] += not nonzeros[-1].any()
                for column, value in enumerate(values):
                    if value is None:
                        continue
                    value = np.asarray(value, dtype=float)
                    finite = np.isfinite(value)
                    totals[cell, column] += value[finite].sum()
                    counts[cell, column] += int(finite.sum())
        completed = len(output["genes"])
    columns = ["cell_type", "fitted_genes", "skipped_genes", "zero_projected_models",
               "genes_with_heldout_spearman_full", "genes_with_heldout_spearman_projection", *metrics]
    rows = []
    for cell, name in enumerate(cell_types):
        means = [float(total / count) if count else None for total, count in zip(totals[cell], counts[cell])]
        rows.append([name, int(fitted[cell]), skipped, int(zero_projected[cell]),
                     int(counts[cell, 0]), int(counts[cell, 1]), *means])
    tracker.table("distillation/cell_types", columns, rows)
    summary = {"completed_genes": completed, "skipped_genes": skipped, "fitted_models": int(fitted.sum()),
               "zero_projected_models": int(zero_projected.sum()),
               "models_with_heldout_spearman_full": int(counts[:, 0].sum()),
               "models_with_heldout_spearman_projection": int(counts[:, 1].sum())}
    for name, total, count in zip(metrics, totals.sum(axis=0), counts.sum(axis=0)):
        summary[name] = float(total / count) if count else None
    tracker.summarize({f"distillation/{name}": value for name, value in summary.items()})


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    with Tracker(args, stage="distillation") as tracker:
        distill(args, tracker)


def distill(args, tracker):
    with h5py.File(args.targets, "r") as teacher:
        if (teacher.attrs.get("format_version") != 1 or not teacher.attrs.get("complete", False)
                or teacher.attrs.get("target_scale") != "log1p"):
            raise ValueError("Expected complete log1p predictions from distillation/predict.py")
        samples, genes, chroms, cell_types = (teacher[key].asstr()[:].tolist()
                                             for key in ("sample_ids", "genes", "chroms", "cell_types"))
        train_indices, test_indices = split_samples(samples, args.split_seed)
        expected = (len(genes), 5, len(samples), len(cell_types))
        if teacher["means"].shape != expected or teacher["variances"].shape != expected:
            raise ValueError("Teacher prediction axes must be gene x 5 members x individual x cell type")
        tracker.configure({"n_genes": len(genes), "n_cell_types": len(cell_types),
                           "n_training_individuals": len(train_indices), "n_held_out_individuals": len(test_indices),
                           "progress_scope": "new_fits metrics cover fits computed in this invocation; final summaries include all saved fits"})
        genotype_files = [fingerprint(args.genotypes / f"{TEMPLATE.format(chrom=chrom)}.{ext}")
                          for chrom in sorted(set(chroms), key=int)
                          for ext in ("bed", "bim", "fam")]
        config = {"format_version": 1,
                  "arguments": run_config(args),
                  "targets": fingerprint(args.targets), "genotype_files": genotype_files,
                  "genotype_template": TEMPLATE, "genes": sorted(genes), "cell_types": cell_types,
                  "training_individuals": [samples[i] for i in train_indices],
                  "held_out_individuals": [samples[i] for i in test_indices],
                  "target_transform": "inverse_normal_across_all_distillation_individuals",
                  "genotype_qc": "complete biallelic SNPs; reference-panel MAF",
                  "feature_screening": "none",
                  "projection_allele_frequency_source": "all_distillation_individuals_unweighted",
                  "allele_dosage": "BIM A1", "refit_on_all_individuals": False}
        config_json = json.dumps(config, sort_keys=True)
        args.output.mkdir(parents=True, exist_ok=True)
        students_path = args.output / "students.h5"
        if students_path.exists():
            with h5py.File(students_path, "r") as saved:
                if "config" in saved and not same_fit_config(saved["config"].asstr()[()], config):
                    raise ValueError("Distillation inputs or settings changed; use a new output directory")
                finished = saved.attrs.get("complete", False) and saved.attrs.get("covariances_complete", False)
            if finished:
                # Keep the student fingerprint stable when resuming only file exports.
                prepare_distillation_exports(students_path)
                log_student_summary(students_path, tracker, args.max_iter)
                logging.info("Completed distillation already exists in %s", args.output)
                return
        logging.info("Distilling %d genes, %d cell types: %s, %s; %d train / %d test donors",
                     len(genes), len(cell_types), args.mode, "additive" if args.additive else "heterozygote-aware",
                     len(train_indices), len(test_indices))
        with h5py.File(students_path, "a") as output, threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=args.jobs) as pool:
            if "config" in output and not same_fit_config(output["config"].asstr()[()], config):
                raise ValueError("Distillation inputs or settings changed; use a new output directory")
            if "config" not in output:
                output.create_dataset("config", data=config_json, dtype=h5py.string_dtype())
                output.attrs.update(format_version=1, complete=False)
                for name, values in (("sample_ids", samples), ("cell_types", cell_types)):
                    strings(output, name, values)
                output.create_dataset("train_indices", data=train_indices)
                output.create_dataset("test_indices", data=test_indices)
                output.create_group("genes")
            (args.output / "run.json").write_text(json.dumps(config, indent=2) + "\n")
            output.attrs.update(complete=False, covariances_complete=False)
            completed = sum(bool(g.attrs.get("complete", False) and g.attrs.get("covariances_complete", False))
                            for g in output["genes"].values())
            progress_every = max(1, math.ceil(len(genes) / 100))
            progress = FitProgress() if tracker.enabled else None
            chromosome, current_chrom = None, None
            for start in range(0, len(genes), 32):
                pending = [i for i in range(start, min(start + 32, len(genes)))
                           if not (genes[i] in output["genes"] and output["genes"][genes[i]].attrs.get("complete", False)
                                    and output["genes"][genes[i]].attrs.get("covariances_complete", False))]
                if not pending:
                    continue
                means = teacher["means"][pending]
                variances = teacher["variances"][pending] if args.mode == "probabilistic" else None
                for offset, i in enumerate(pending):
                    gene, chrom, tss = genes[i], chroms[i], int(teacher["tss"][i])
                    if chrom != current_chrom:
                        chromosome = Chromosome(args.genotypes, chrom, samples)
                        current_chrom = chrom
                    dosage, snps = chromosome.cis(tss, args.half_window, args.maf)
                    group = output["genes"].require_group(gene)
                    fitted_cells = {cell for cell in range(len(cell_types)) if str(cell) in group
                                    and group[str(cell)].attrs.get("fit_complete", False)}
                    group.attrs.update(chrom=chrom, tss=tss, complete=False, covariances_complete=False)
                    if not dosage.shape[1] or not np.any(dosage[train_indices].std(axis=0) > 1e-8):
                        group.attrs.update(status="no_variable_cis_snps", complete=True, covariances_complete=True)
                        logging.info("Skipping %s: no variable cis-SNPs after QC", gene)
                        output.flush()
                        completed += 1
                        if completed % progress_every == 0 or completed == len(genes):
                            tracker.log({"distillation/completed_genes": completed,
                                         **(progress.metrics() if progress else {})})
                        continue
                    gene_seed = (args.seed + zlib.crc32(gene.encode())) % (2 ** 32)

                    def fit_cell(cell):
                        if cell in fitted_cells:
                            return None
                        raw_means = means[offset, :, :, cell]
                        if args.mode == "deterministic":
                            target, variance = normal_targets(raw_means.mean(axis=0, keepdims=True))
                        else:
                            target, variance = normal_targets(raw_means, variances[offset, :, :, cell])
                        return fit_student(dosage, target, variance, train_indices, test_indices,
                                           mode=args.mode, additive=args.additive, bootstraps=args.bootstraps,
                                           alpha=args.alpha, l1_ratio=args.l1_ratio, cv=args.cv, alphas=args.alphas,
                                           max_iter=args.max_iter,
                                           sigma_floor=args.sigma_floor, seed=gene_seed)

                    snp_lookup = {snp: j for j, snp in enumerate(snps["snp_ids"])}
                    for cell, result in enumerate(bounded_results(pool, fit_cell, range(len(cell_types)), args.jobs)):
                        if result is not None:
                            if str(cell) in group:
                                del group[str(cell)]
                            cell_group = group.create_group(str(cell))
                            cell_group.attrs["cell_type"] = cell_types[cell]
                            write_student(cell_group, result, snps)
                            cell_group.attrs["fit_complete"] = True
                            output.flush()
                            if progress is not None:
                                progress.update(result)
                        else:
                            cell_group = group[str(cell)]
                        indices = [snp_lookup[snp] for snp in cell_group["snp_ids"].asstr()[:]]
                        save_covariances(cell_group, dosage[:, indices])
                    group.attrs.update(status="fitted", complete=True, covariances_complete=True, bootstrap_seed=gene_seed)
                    output.flush()
                    logging.info("Saved %s (%d cis-SNPs, %d cell types)", gene, dosage.shape[1], len(cell_types))
                    completed += 1
                    if completed % progress_every == 0 or completed == len(genes):
                        tracker.log({"distillation/completed_genes": completed,
                                     **(progress.metrics() if progress else {})})
            output.attrs.update(complete=True, covariances_complete=True)
    prepare_distillation_exports(args.output / "students.h5")
    log_student_summary(args.output / "students.h5", tracker, args.max_iter)
    logging.info("Distilled students saved to %s", args.output / "students.h5")


if __name__ == "__main__":
    main()
