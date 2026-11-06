"""Extract VariantFormer protein-coding gene embeddings for one tissue."""

import argparse
from contextlib import contextmanager
import json
import logging
from pathlib import Path
import re
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.geuvadis import ROOT, read_ids, run


CORE_ASSETS = (
    "all_genes_v1_pcg_gencodeV24.csv", "pretrained_tokenizers_checkpoint.pth",
    "v4_pcg_epoch11_checkpoint.pth", "ENCFF234XEZ.bed.gz",
    "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz",
)
EMBEDDING_FIELDS = {"format_version", "features", "ensids", "gene_ids", "chroms", "tss",
                    "sample_id", "tissue", "embedding_provenance"}
PERSONALIZED_ARRAYS = ("features", "ensids", "chroms", "tss")


def personalized_path(directory, sample):
    """Prefer a native archive, falling back to a per-individual array directory."""
    directory = Path(directory)
    archive = directory / f"{sample}.npz"
    arrays = directory / sample
    return arrays if not archive.is_file() and arrays.is_dir() else archive


def personalized_files(path):
    """List every file required for a personalized embedding input."""
    path = Path(path)
    return [path / f"{name}.npy" for name in PERSONALIZED_ARRAYS] if path.is_dir() else [path]


@contextmanager
def load_personalized(path):
    """Read native NPZs or trusted legacy directories with pickled metadata arrays.

    Legacy sample identity comes from the directory name. Unrecorded tissue and
    backbone provenance remain unknown; known teacher metadata must still match.
    Features are memory-mapped without pickle, and views keep their mapping alive.
    """
    import numpy as np

    path = Path(path)
    if not path.is_dir():
        with np.load(path, allow_pickle=False) as archive:
            yield archive
        return
    missing = [file for file in personalized_files(path) if not file.is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete personalized embeddings: {missing}")
    metadata = {name: np.load(path / f"{name}.npy", allow_pickle=True)
                for name in PERSONALIZED_ARRAYS if name != "features"}
    genes = np.asarray(metadata["ensids"], dtype=str)
    yield {
        "format_version": np.asarray(1),
        "features": np.load(path / "features.npy", mmap_mode="r", allow_pickle=False),
        "ensids": genes, "gene_ids": genes,
        "chroms": np.asarray(metadata["chroms"], dtype=str),
        "tss": np.asarray(metadata["tss"], dtype=np.float64),
        "sample_id": np.asarray(path.name), "tissue": np.asarray(""),
        "embedding_provenance": np.asarray(json.dumps(
            dict.fromkeys(("tissue", "model", "variantformer", "assets")))),
    }


def asset_fingerprints(directory):
    """Record fixed model assets without hashing multi-gigabyte checkpoints."""
    result = {}
    for name in CORE_ASSETS:
        stat = (Path(directory) / name).stat()
        result[name] = [stat.st_size, stat.st_mtime_ns]
    return result


def read_provenance(archive):
    """Validate native metadata; null provenance fields explicitly mean unknown."""
    if not EMBEDDING_FIELDS <= set(archive) or archive["format_version"].item() != 1:
        raise ValueError("Expected a native format_version=1 embedding NPZ with complete metadata and provenance")
    try:
        provenance = json.loads(str(archive["embedding_provenance"].item()))
        if (not isinstance(provenance, dict) or set(provenance) != {"tissue", "model", "variantformer", "assets"}
                or any(value is not None and (not isinstance(value, str) or not value)
                       for key, value in provenance.items() if key != "assets")):
            raise ValueError
        assets = provenance["assets"]
        if assets is not None:
            if not isinstance(assets, dict) or set(assets) != set(CORE_ASSETS):
                raise ValueError
            for fingerprint in assets.values():
                if (len(fingerprint) != 2
                        or any(not isinstance(value, int) or value < 0 for value in fingerprint)):
                    raise ValueError
        if (str(archive["tissue"].item()) or None) != provenance["tissue"]:
            raise ValueError
    except (ValueError, TypeError, KeyError, AttributeError) as error:
        raise ValueError("Invalid embedding provenance in archive") from error
    return provenance


def validate_provenance(expected, actual):
    keys = {"tissue", "model", "variantformer", "assets"}
    if not isinstance(expected, dict) or set(expected) != keys or set(actual) != keys:
        raise ValueError("Expected native embedding provenance with explicit unknown fields")
    if any(actual[key] != value for key, value in expected.items() if value is not None):
        raise ValueError("Embedding provenance is missing or differs from the teacher's reference embeddings")


def warn_incomplete_provenance(provenance):
    unknown = [key for key, value in provenance.items() if value is None]
    if unknown:
        logging.warning("Embedding provenance is unknown for %s; recorded fields are checked", ", ".join(unknown))


def validate_vcf(path, sample):
    if not path.is_file() or not Path(f"{path}.tbi").is_file():
        raise ValueError(f"Missing VCF or tabix index: {path}. Run bash setup.sh --geuvadis first.")
    names = run("bcftools", "query", "-l", path, capture_output=True, text=True).stdout.split()
    if names != [sample]:
        raise ValueError(f"Expected only sample {sample} in {path}, found {names}")
    header = run("bcftools", "view", "-h", path, capture_output=True, text=True).stdout
    contigs = set(re.findall(r"##contig=<ID=([^,>]+)", header))
    if not {f"chr{i}" for i in range(1, 23)} <= contigs:
        raise ValueError(f"Expected chr1–chr22 contigs in {path}; regenerate with geuvadis.py")
    run("bcftools", "index", "-n", path, capture_output=True)


def prepare_output(output, config):
    """Resume only archives whose inputs are recorded in the same manifest."""
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "run.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != config:
            raise ValueError("Embedding inputs changed; use a new --output directory")
        return
    if any(not path.name.endswith(".partial.npz") for path in output.glob("*.npz")):
        raise ValueError("Embedding archives exist without run.json; use a new --output directory")
    temporary = output / "run.json.partial"
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tissue", required=True, help="VariantFormer tissue name, e.g. blood")
    parser.add_argument("--reference", action="store_true", help="Use the unmodified GRCh38 reference")
    parser.add_argument("--samples", type=Path, default=ROOT / "data/data-for-the-paper/1000G_EUR_503.txt")
    parser.add_argument("--vcf-dir", type=Path, default=ROOT / "data/geuvadis/individuals")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers nonnegative")
    vf = ROOT / "variantformer"
    sys.path.insert(0, str(vf))
    import numpy as np
    import torch

    if not torch.cuda.is_available():
        parser.error("VariantFormer requires a CUDA GPU")
    from processors.vcfprocessor import VCFProcessor
    for tool in ("bcftools", "samtools", "bedtools"):
        run(tool, "--version", capture_output=True)
    samples = ["REFERENCE"] if args.reference else read_ids(args.samples)
    vcfs = {} if args.reference else {
        sample: args.vcf_dir.resolve() / f"{sample}.vcf.gz" for sample in samples
    }
    for sample, path in vcfs.items():
        validate_vcf(path, sample)
    processor = VCFProcessor(model_class="v4_pcg")
    if args.tissue not in processor.get_tissues():
        parser.error(f"Unknown tissue. Choose from: {', '.join(processor.get_tissues())}")
    genes = processor.get_genes()
    genes = genes.loc[genes.chromosome.isin([f"chr{i}" for i in range(1, 23)])].copy().reset_index(drop=True)
    if genes.empty:
        raise ValueError("No autosomal protein-coding genes available")
    genes["ensid"] = genes.gene_id.str.split(".").str[0]
    query = genes[["gene_id"]].assign(tissues=args.tissue)
    output = args.output or ROOT / "data/embeddings" / (
        "reference" if args.reference else "geuvadis"
    ) / re.sub(r"[^a-zA-Z0-9_-]+", "_", args.tissue)
    provenance = {
        "tissue": args.tissue, "model": "v4_pcg",
        "variantformer": run("git", "-C", vf, "rev-parse", "HEAD", capture_output=True, text=True).stdout.strip(),
        "assets": asset_fingerprints(vf / "_artifacts"),
    }
    config = {
        **provenance, "genes": genes.gene_id.tolist(), "samples": samples,
        "vcfs": {sample: [str(path), path.stat().st_size, path.stat().st_mtime_ns]
                 for sample, path in vcfs.items()},
    }
    prepare_output(output, config)
    pending = [sample for sample in samples if not (output / f"{sample}.npz").exists()]
    if not pending:
        print(f"Embeddings already complete: {output}")
        return
    model, checkpoint, trainer = processor.load_model()
    for sample in pending:
        print(f"Extracting {sample}: {len(genes)} genes", flush=True)
        path = str(vcfs[sample]) if sample in vcfs else None
        dataset, loader = processor.create_data(
            path, query.copy(), batch_size=args.batch_size, num_workers=args.workers,
            prefetch_factor=4 if args.workers else None,
        )
        predictions = processor.predict(model, checkpoint, trainer, loader, dataset)
        if predictions.gene_id.tolist() != genes.gene_id.tolist():
            raise ValueError("VariantFormer changed the selected genes or their order")
        features = np.stack([np.asarray(row, dtype=np.float32).reshape(-1)
                             for row in predictions.embeddings])
        if not np.isfinite(features).all():
            raise ValueError(f"Non-finite embeddings for {sample}")
        temporary = output / f"{sample}.partial.npz"
        np.savez_compressed(
            temporary, format_version=1, features=features, ensids=genes.ensid.to_numpy(dtype=str),
            gene_ids=genes.gene_id.to_numpy(dtype=str),
            chroms=genes.chromosome.to_numpy(dtype=str),
            tss=np.where(genes.strand == "+", genes.start, genes.end),
            sample_id=sample, tissue=args.tissue,
            embedding_provenance=json.dumps(provenance, sort_keys=True),
        )
        temporary.replace(output / f"{sample}.npz")
    print(f"Embeddings saved to {output}")


if __name__ == "__main__":
    main()
