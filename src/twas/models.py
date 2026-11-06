"""Export distilled students to the PredictDB and covariance formats."""

from contextlib import ExitStack, closing, contextmanager
import gzip
import json
import logging
from pathlib import Path
import sqlite3

import h5py
import numpy as np

from src.distillation.covariance import covariance_blocks, saved_triangle_blocks
from src.distillation.genotypes import TEMPLATE
from src.twas.reference import fingerprint, model_snps


SCHEMA = '''
CREATE TABLE weights (rsid TEXT, gene TEXT, weight REAL, ref_allele TEXT, eff_allele TEXT);
CREATE TABLE extra (gene TEXT, genename TEXT, "n.snps.in.model" INTEGER,
                    "pred.perf.R2" REAL, "pred.perf.pval" REAL, "pred.perf.qval" REAL);
'''


def cell_type_indices(handle, cell_type=None):
    """Keep file indices when selecting a label; accept spaces or underscores."""
    cells = handle["cell_types"].asstr()[:].tolist()
    if cell_type is None:
        return list(range(len(cells)))
    if cell_type in cells:
        return [cells.index(cell_type)]
    matches = [i for i, name in enumerate(cells) if name.replace("_", " ") == cell_type.replace("_", " ")]
    if len(matches) != 1:
        raise ValueError(f"Unknown or ambiguous cell type {cell_type!r}; choose from: {', '.join(cells)}")
    return matches


def student_file_config(handle, path, cell_type=None):
    """Describe native weight shards and scientific settings without reading weights."""
    config = json.loads(handle["config"].asstr()[()])
    result = {"model_settings": config["model_settings"]} if "model_settings" in config else {}
    if "student_files" in handle:
        if handle["student_files"].ndim != 1:
            raise ValueError("Expected a one-dimensional list of native weight shards")
        paths = handle["student_files"].asstr()[:].tolist()
        if (len(paths) != len(handle["cell_types"]) or len(set(paths)) != len(paths)
                or any(not value or Path(value).is_absolute() or ".." in Path(value).parts for value in paths)):
            raise ValueError("Expected one distinct, model-relative weight shard per cell type")
        result["student_files"] = [fingerprint(Path(path).parent / paths[i])
                                   for i in cell_type_indices(handle, cell_type)]
    return result


@contextmanager
def open_students(path, cell_type=None):
    """Keep native shards open while following the root file's HDF5 links."""
    with ExitStack() as stack:
        handle = stack.enter_context(h5py.File(path, "r"))
        for entry in student_file_config(handle, path, cell_type).get("student_files", []):
            shard = stack.enter_context(h5py.File(entry[0], "r"))
            if shard.attrs.get("format_version") != 1 or not shard.attrs.get("complete", False):
                raise ValueError(f"Incomplete or unsupported native weight shard: {entry[0]}")
        yield handle


def inspect_students(handle, require_nonzero=True, cell_type=None):
    if handle.attrs.get("format_version") != 1 or not handle.attrs.get("complete", False):
        raise ValueError("Expected completed native students.h5")
    config = json.loads(handle["config"].asstr()[()])
    mode = config["arguments"]["mode"]
    if mode not in ("probabilistic", "deterministic"):
        raise ValueError(f"Unsupported student mode: {mode}")
    shape = (5, int(config["arguments"]["bootstraps"])) if mode == "probabilistic" else (1, 1)
    if min(shape) < 1:
        raise ValueError("Expected at least one bootstrap draw")
    samples, cells = (handle[key].asstr()[:].tolist() for key in ("sample_ids", "cell_types"))
    if len(samples) < 2 or len(set(samples)) != len(samples) or not cells or len(set(cells)) != len(cells):
        raise ValueError("Student sample IDs and cell types must be unique and nonempty")
    indices = cell_type_indices(handle, cell_type)
    genes = sorted(handle["genes"])
    if not genes:
        raise ValueError("Students contain no genes")
    shards = handle["student_files"].asstr()[:].tolist() if "student_files" in handle else None
    plan, catalog = {}, {}
    for gene in genes:
        if any(c.isspace() for c in gene):
            raise ValueError(f"Gene ID contains whitespace: {gene!r}")
        group = handle["genes"][gene]
        if not group.attrs.get("complete", False):
            raise ValueError(f"Incomplete student: {gene}")
        chrom, status = str(group.attrs["chrom"]), group.attrs["status"]
        if chrom not in {str(c) for c in range(1, 23)}:
            raise ValueError(f"Expected autosomal GRCh38 models, found chromosome {chrom}")
        if status not in ("fitted", "no_variable_cis_snps"):
            raise ValueError(f"Unknown student status for {gene}: {status}")
        item = {"chrom": chrom, "tss": int(group.attrs["tss"]), "status": status, "catalog": {}}
        if status == "fitted":
            for cell in indices:
                name = cells[cell]
                if shards is not None:
                    link = group.get(str(cell), getlink=True)
                    if (not isinstance(link, h5py.ExternalLink) or link.filename != shards[cell]
                            or link.path != f"/genes/{gene}/{cell}"):
                        raise ValueError(f"Weight shard link disagrees with student_files: {gene}/{cell}")
                student = group[str(cell)]
                if student.attrs["cell_type"] != name:
                    raise ValueError(f"Cell-type order mismatch in {student.name}")
                snps = model_snps(student, chrom)
                weights = student["coefficient_draws"][:]
                if weights.shape != (*shape, len(snps)) or not np.isfinite(weights).all():
                    raise ValueError(f"Invalid member/bootstrap/projected-weight axes in {student.name}")
                active = np.any(weights != 0, axis=(0, 1))
                for (snp, metadata), keep in zip(snps.items(), active):
                    if not keep:
                        continue
                    if not snp or any(c.isspace() for c in snp):
                        raise ValueError(f"Invalid SNP ID: {snp!r}")
                    if snp in catalog and catalog[snp] != metadata:
                        raise ValueError(f"Conflicting SNP metadata for {snp}")
                    item["catalog"][snp] = catalog[snp] = metadata
        plan[gene] = item
    if require_nonzero and not catalog:
        raise ValueError("Students contain no nonzero projected SNP weights to test")
    return mode, shape, samples, [cells[i] for i in indices], plan, catalog


def draw_name(member, bootstrap):
    return f"member_{member}_bootstrap_{bootstrap}"


def export_config(path, mode, shape, samples, cells, plan, genotype_files, stored=False):
    return {"format_version": 1, "models": fingerprint(path), "genotype_files": genotype_files,
            "genotype_template": TEMPLATE, "genes": list(plan), "cell_types": cells,
            "mode": mode, "draw_shape": list(shape), "ld_individuals": samples, "covariance_ddof": 1,
            "covariance_source": "distillation" if stored else "PLINK",
            "weights": "stored coefficient_draws, raw effect-allele dosage, no rescaling"}


def external_covariance_config(handle, path, cells, shape, plan):
    """Record native MetaXcan covariance files and their streaming order."""
    if handle.attrs.get("covariance_storage") != "metaxcan_text":
        return {}
    paths = handle["covariance_paths"].asstr()[:]
    order = handle["covariance_gene_order"].asstr()[:].tolist()
    all_cells = handle["cell_types"].asstr()[:].tolist()
    if paths.shape != (len(all_cells), *shape) or len(set(order)) != len(order) or set(plan) - set(order):
        raise ValueError("External covariance paths or gene order do not match the students")
    if any(not entry or Path(entry).is_absolute() or ".." in Path(entry).parts for entry in paths.flat):
        raise ValueError("Expected model-relative covariance paths")
    paths = paths[[all_cells.index(name) for name in cells]]
    return {"covariance_source": "external_metaxcan_text",
            "covariance_files": [fingerprint(Path(path).parent / entry) for entry in paths.flat],
            "covariance_gene_order": [gene for gene in order if gene in plan]}


def initialize_cache(directory, config, refresh_model=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "config.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if previous == config:
            return
        # Covariance backfilling may change the student file's fingerprint.
        if (not refresh_model or previous["models"][0] != config["models"][0]
                or {k: v for k, v in previous.items() if k != "models"}
                != {k: v for k, v in config.items() if k != "models"}):
            raise ValueError("Model cache inputs changed; use a new --model-cache directory")
        for marker in directory.glob("cell_*/complete.json"):
            marker.unlink()
    elif any(directory.iterdir()):
        raise ValueError("Unrecognized model cache; use an empty --model-cache directory")
    partial = path.with_suffix(".json.partial")
    partial.write_text(json.dumps(config, indent=2) + "\n")
    partial.replace(path)


def prepare_distillation_exports(path):
    """Assemble saved triangles into the per-cell/draw files MetaXcan reads."""
    path = Path(path)
    with open_students(path) as handle:
        mode, shape, samples, cells, plan, _ = inspect_students(handle, require_nonzero=False)
        if not handle.attrs.get("covariances_complete", False):
            raise ValueError("Distillation covariances are incomplete")
        source = json.loads(handle["config"].asstr()[()])
        config = export_config(path, mode, shape, samples, cells, plan, source["genotype_files"], stored=True)
        config.update(student_file_config(handle, path))
        config.update(external_covariance_config(handle, path, cells, shape, plan))
        directory = path.parent / "models"
        initialize_cache(directory, config, refresh_model=True)
        for cell, name in enumerate(cells):
            export_cell(handle, cell, shape, plan, None, directory / f"cell_{cell:04d}")
            logging.info("MetaXcan model and covariance exports ready for %s", name)


def export_cell(handle, cell, shape, plan, reference, directory):
    """Export each draw in gene order, using stored raw-dosage coefficients."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory / "complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if all(Path(entry[0]).is_file() and fingerprint(entry[0]) == entry for entry in saved["files"]):
            return saved["draws"]
    marker.unlink(missing_ok=True)
    external = reference is None and handle.attrs.get("covariance_storage") == "metaxcan_text"
    order_path = directory / "gene_order.txt"
    if external:
        order = [gene for gene in handle["covariance_gene_order"].asstr()[:] if gene in plan]
        partial_order = order_path.with_suffix(".txt.partial")
        partial_order.write_text("".join(f"{gene}\n" for gene in order))
        partial_order.replace(order_path)
    draws, outputs = [], []
    with ExitStack() as stack:
        for member in range(shape[0]):
            for bootstrap in range(shape[1]):
                name = draw_name(member, bootstrap)
                db, covariance = directory / f"{name}.db", directory / f"{name}.cov.txt.gz"
                partial_db = Path(str(db) + ".partial")
                partial_db.unlink(missing_ok=True)
                connection = stack.enter_context(closing(sqlite3.connect(partial_db)))
                connection.executescript(SCHEMA)
                if external:
                    covariance = Path(handle.filename).parent / handle["covariance_paths"].asstr()[cell, member, bootstrap]
                    partial_cov, text = None, None
                else:
                    partial_cov = Path(str(covariance) + ".partial")
                    text = stack.enter_context(gzip.open(partial_cov, "wt", compresslevel=1))
                    text.write("GENE\tRSID1\tRSID2\tVALUE\n")
                draws.append({"member": member, "bootstrap": bootstrap, "model_db": str(db.resolve()),
                              "covariance": str(covariance.resolve()), "n_genes": 0, "n_weights": 0})
                if external:
                    draws[-1]["gene_order"] = str(order_path.resolve())
                outputs.append((connection, text, partial_db, db, partial_cov, covariance))
        for number, (gene, item) in enumerate(plan.items(), start=1):
            if item["status"] != "fitted" or not item["catalog"]:
                continue
            student = handle["genes"][gene][str(cell)]
            snps = student["snp_ids"].asstr()[:]
            effects, others = student["effect_alleles"].asstr()[:], student["other_alleles"].asstr()[:]
            weights = student["coefficient_draws"][:]
            if not np.any(weights):
                continue
            if reference is not None:
                reference_snps, dosage = reference.get(gene, item["chrom"], item["catalog"])
                lookup = {snp: i for i, snp in enumerate(reference_snps)}
            for draw, (connection, text, *_) in zip(draws, outputs):
                coefficients = weights[draw["member"], draw["bootstrap"]]
                selected = np.flatnonzero(coefficients != 0)
                if not len(selected):
                    continue
                ids = snps[selected].tolist()
                if external:
                    blocks = ()
                elif reference is None:
                    saved_indices, blocks = saved_triangle_blocks(student, draw["member"], draw["bootstrap"])
                    if not np.array_equal(saved_indices, selected):
                        raise ValueError(f"Stored covariance SNP support differs from weights in {student.name}")
                else:
                    blocks = covariance_blocks(dosage[:, [lookup[snp] for snp in ids]])
                connection.executemany("INSERT INTO weights VALUES (?, ?, ?, ?, ?)",
                                       [(snps[i], gene, float(coefficients[i]), others[i], effects[i]) for i in selected])
                connection.execute("INSERT INTO extra VALUES (?, ?, ?, NULL, NULL, NULL)", (gene, gene, len(ids)))
                for start, stop, triangle in blocks:
                    offset = 0
                    for i in range(start, stop):
                        text.write("".join(f"{gene}\t{ids[i]}\t{ids[j]}\t{triangle[offset + j - i]:.17g}\n"
                                           for j in range(i, len(ids))))
                        offset += len(ids) - i
                draw["n_genes"] += 1
                draw["n_weights"] += len(ids)
            if number % 100 == 0:
                logging.info("Exported cell %d: %d / %d genes", cell, number, len(plan))
        for connection, *_ in outputs:
            connection.executescript("CREATE INDEX weights_gene ON weights(gene); CREATE INDEX extra_gene ON extra(gene);")
            connection.commit()
    files = [fingerprint(order_path)] if external else []
    for _, _, partial_db, db, partial_cov, covariance in outputs:
        partial_db.replace(db)
        if partial_cov is not None:
            partial_cov.replace(covariance)
        files.extend([fingerprint(db), fingerprint(covariance)])
    partial_marker = marker.with_suffix(".json.partial")
    partial_marker.write_text(json.dumps({"draws": draws, "files": files}, indent=2) + "\n")
    partial_marker.replace(marker)
    return draws
