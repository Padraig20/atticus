"""Evaluate reference predictions on saved test genes after model selection."""

import argparse
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.dataset import gene_ids, load_reference
from src.preprocessing.embeddings import validate_provenance, warn_incomplete_provenance
from src.training.model import DeepEnsemble, validate_checkpoint
from src.tracking import Tracker, add_tracking_args


def correlation(left, right, *, ranked=False):
    valid = np.isfinite(left) & np.isfinite(right)
    left, right = np.asarray(left)[valid], np.asarray(right)[valid]
    if len(left) < 2 or np.ptp(left) == 0 or np.ptp(right) == 0:
        return None
    if ranked:
        left, right = rankdata(left), rankdata(right)
    return float(np.corrcoef(left, right)[0, 1])


def average(values):
    values = [value for value in values if value is not None]
    return float(np.mean(values)) if values else None


def expression_moments(path, individuals, genes, cell_types):
    """Accumulate mean log1p expression and population variance (ddof=0)."""
    sums = np.zeros((len(genes), len(cell_types)), dtype=np.float64)
    squares = np.zeros_like(sums)
    counts = np.zeros(len(cell_types), dtype=int)
    data = ad.read_h5ad(path, backed="r")
    try:
        for column in ("individual_id", "cell_type"):
            if column not in data.obs or data.obs[column].isna().any():
                raise ValueError(f"Targets require non-missing obs[{column!r}]")
        if data.obs.duplicated(["individual_id", "cell_type"]).any():
            raise ValueError("Targets contain duplicate donor/cell-type observations")
        donors = data.obs.individual_id.astype(str).to_numpy()
        labels = data.obs.cell_type.astype(str).to_numpy()
        if set(individuals) - set(donors):
            raise ValueError("Targets are missing saved training individuals")
        lookup = {gene: i for i, gene in enumerate(gene_ids(data.var_names))}
        if set(genes) - lookup.keys():
            raise ValueError("Targets are missing saved test genes")
        columns = [lookup[gene] for gene in genes]
        rows = np.flatnonzero(np.isin(donors, individuals) & np.isin(labels, cell_types))
        for start in range(0, len(rows), 256):
            selected = rows[start:start + 256]
            block = data[selected, :].X[:, columns]
            block = block.toarray() if sparse.issparse(block) else np.asarray(block)
            block = np.asarray(block, dtype=np.float64)
            if not np.isfinite(block).all() or (block < 0).any():
                raise ValueError("Targets must contain finite, nonnegative raw pseudo-bulk expression")
            block = np.log1p(block)
            for cell, label in enumerate(cell_types):
                values = block[labels[selected] == label]
                sums[:, cell] += values.sum(axis=0)
                squares[:, cell] += np.square(values).sum(axis=0)
                counts[cell] += len(values)
    finally:
        data.file.close()
    if not counts.all():
        raise ValueError("A saved cell type has no observed training individuals")
    means = sums / counts
    variances = np.maximum(0, squares / counts - np.square(means))
    return means, variances, counts


def evaluate(checkpoint, observations, targets, batch_size, device):
    validate_checkpoint(checkpoint)
    metadata = checkpoint["data"]
    genes, cell_types = metadata["genes"]["test"], metadata["cell_types"]
    features, identifiers, _, provenance = load_reference(observations)
    reference_provenance = metadata["embedding_provenance"]
    validate_provenance(reference_provenance, provenance)
    warn_incomplete_provenance(reference_provenance)
    lookup = {gene: i for i, gene in enumerate(identifiers)}
    if not genes or set(genes) - lookup.keys():
        raise ValueError("Reference embeddings are missing saved test genes")
    features = features[[lookup[gene] for gene in genes]]
    model = DeepEnsemble(**checkpoint["model_config"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    batches = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            prediction = model(torch.from_numpy(features[start:start + batch_size]).to(device))
            batches.append(tuple(value.cpu().numpy() for value in prediction))
    mean, aleatoric, epistemic = (np.concatenate(values).astype(np.float64) for values in zip(*batches))
    if any(not np.isfinite(values).all() for values in (mean, aleatoric, epistemic)):
        raise ValueError("Non-finite reference predictions")
    observed, population_variance, counts = expression_moments(
        targets, metadata["training_individuals"], genes, cell_types)
    rows = []
    for cell, label in enumerate(cell_types):
        squared_error = np.square(mean[:, cell] - observed[:, cell])
        rows.append({"cell_type": label, "n_individuals": int(counts[cell]),
                     "pearson_across_genes": correlation(mean[:, cell], observed[:, cell]),
                     "spearman_epistemic_squared_error": correlation(epistemic[:, cell], squared_error, ranked=True),
                     "pearson_aleatoric_population_variance": correlation(aleatoric[:, cell], population_variance[:, cell])})
    metrics = {name: average([row[name] for row in rows]) for name in rows[0] if name not in ("cell_type", "n_individuals")}
    return metrics, rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("-X", "--observations", type=Path, help="Reference embeddings (default: saved training path)")
    parser.add_argument("-y", "--targets", type=Path, help="Raw pseudo-bulk H5AD (default: saved training path)")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Directory for summary.json and cell_types.tsv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    add_tracking_args(parser)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint)
    args.observations = args.observations or Path(checkpoint["data"]["observations"])
    args.targets = args.targets or Path(checkpoint["data"]["targets"])
    config = {"inputs": [[str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns]
                         for path in (args.checkpoint, args.targets, args.observations)],
              "scope": "saved test genes and training individuals; donor-averaged log1p expression",
              "uncertainty_scope": "correlations across test genes per cell type; mean of defined correlations",
              "population_variance_ddof": 0}
    summary_path = args.output / "summary.json"
    if summary_path.exists() and json.loads(summary_path.read_text())["config"] != config:
        raise ValueError("Evaluation inputs changed; use a new output directory")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    with Tracker(args, stage="evaluation") as tracker:
        metrics, rows = evaluate(checkpoint, args.observations, args.targets, args.batch_size, device)
        counts = {"n_test_genes": len(checkpoint["data"]["genes"]["test"]), "n_cell_types": len(rows),
                  "n_training_individuals": len(checkpoint["data"]["training_individuals"])}
        args.output.mkdir(parents=True, exist_ok=True)
        table = pd.DataFrame(rows)
        temporary = args.output / "cell_types.tsv.partial"
        table.to_csv(temporary, sep="\t", index=False)
        temporary.replace(args.output / "cell_types.tsv")
        temporary = args.output / "summary.json.partial"
        temporary.write_text(json.dumps({"config": config, **counts, "metrics": metrics}, indent=2, allow_nan=False) + "\n")
        temporary.replace(summary_path)
        tracker.configure({**counts, "evaluation_scope": config["scope"]})
        tracker.summarize({f"evaluation/{name}": value for name, value in metrics.items()})
        tracker.table("evaluation/cell_types", list(table.columns), table.values.tolist())
        print(json.dumps(metrics, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
