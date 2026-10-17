"""Extract VariantFormer protein-coding gene embeddings for one tissue."""

import argparse
import json
from pathlib import Path
import re
import sys

from geuvadis import ROOT, read_ids, run


def select_genes(genes, requested=None):
    genes = genes[genes.chromosome.isin([f"chr{i}" for i in range(1, 23)])].copy()
    genes["ensid"] = genes.gene_id.str.split(".").str[0]
    genes = genes.set_index("ensid", drop=False)
    if requested is not None:
        requested = [gene.split(".")[0] for gene in requested]
        if len(requested) != len(set(requested)):
            raise ValueError("Gene list contains duplicate Ensembl IDs")
        missing = set(requested) - set(genes.index)
        if missing:
            raise ValueError(f"Genes absent from VariantFormer's autosomal protein-coding set: {sorted(missing)}")
        genes = genes.loc[requested]
    if genes.empty:
        raise ValueError("No genes selected")
    return genes.reset_index(drop=True)


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tissue", required=True, help="VariantFormer tissue name, e.g. blood")
    parser.add_argument("--reference", action="store_true", help="Use the unmodified GRCh38 reference")
    parser.add_argument("--genes", type=Path, help="Ensembl IDs, one per line (default: all autosomal protein-coding genes)")
    parser.add_argument("--samples", type=Path, default=ROOT / "data/1000G_EUR_503.txt")
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
    genes = select_genes(processor.get_genes(), read_ids(args.genes) if args.genes else None)
    query = genes[["gene_id"]].assign(tissues=args.tissue)
    output = args.output or ROOT / "data/embeddings" / (
        "reference" if args.reference else "geuvadis"
    ) / re.sub(r"[^a-zA-Z0-9_-]+", "_", args.tissue)
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "tissue": args.tissue, "genes": genes.gene_id.tolist(), "samples": samples,
        "model": "v4_pcg",
        "variantformer": run("git", "-C", vf, "rev-parse", "HEAD", capture_output=True, text=True).stdout.strip(),
        "vcfs": {sample: [str(path), path.stat().st_size, path.stat().st_mtime_ns]
                 for sample, path in vcfs.items()},
    }
    manifest = output / "run.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        parser.error("Embedding inputs changed; use a new --output directory")
    manifest.write_text(json.dumps(config, indent=2) + "\n")
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
            temporary, features=features, ensids=genes.ensid.to_numpy(dtype=str),
            gene_ids=genes.gene_id.to_numpy(dtype=str),
            chroms=genes.chromosome.to_numpy(dtype=str),
            tss=np.where(genes.strand == "+", genes.start, genes.end),
            sample_id=sample, tissue=args.tissue,
        )
        temporary.replace(output / f"{sample}.npz")
    print(f"Embeddings saved to {output}")


if __name__ == "__main__":
    main()
