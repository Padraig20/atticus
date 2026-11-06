"""Reference embeddings paired with expression from training individuals only."""

from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse
import torch
from torch.utils.data import Dataset

from src.preprocessing.embeddings import read_provenance, warn_incomplete_provenance
from src.preprocessing.geuvadis import read_ids


CHROMOSOMES = {
    "train": ["1", "2", "3", "4", "6", "8", "9", "10", "13", "15", "16", "17", "18", "19", "21", "22", "X", "Y"],
    "validation": ["7", "11", "14"],
    "test": ["5", "12", "20"],
}


def gene_ids(values):
    ids = np.asarray([str(value).split(".")[0] for value in values])
    if len(set(ids)) != len(ids):
        raise ValueError("Gene IDs must be unique, including after removing version suffixes")
    return ids


def load_reference(path):
    """Load a native reference embedding archive without pickle."""
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError("Reference embeddings must be a native REFERENCE.npz archive")
    with np.load(path, allow_pickle=False) as data:
        provenance = read_provenance(data)
        if str(data["sample_id"].item()) != "REFERENCE":
            raise ValueError("Reference training requires REFERENCE embeddings, not a donor's embeddings")
        features, genes, chroms = (data[key] for key in ("features", "ensids", "chroms"))
    if np.ndim(genes) != 1 or np.ndim(chroms) != 1:
        raise ValueError("Reference gene and chromosome arrays must be one-dimensional")
    genes = gene_ids(genes)
    chroms = np.asarray([str(c).lower().removeprefix("chr").upper() for c in chroms])
    features = np.asarray(features, dtype=np.float32)
    if (features.ndim != 2 or features.shape[1] == 0
            or features.shape[0] != len(genes) or len(genes) != len(chroms)):
        raise ValueError("Expected one feature vector and one chromosome per gene")
    if not np.isfinite(features).all():
        raise ValueError("Reference features must be finite")
    return features, genes, chroms, provenance


def split_individuals(individuals, seed=42, held_out=None):
    """Reserve supplied donors or a fixed random 20%, independent of row order."""
    ids = sorted({str(individual) for individual in individuals})
    if len(ids) < 2:
        raise ValueError("At least two individuals are required for the 80/20 donor split")
    if held_out is not None:
        available = set(ids)
        resolved = []
        for individual in held_out:
            matches = ([individual] if individual in available else
                       [value for value in ids if "_" in value and value.rsplit("_", 1)[-1] == individual])
            if len(matches) != 1:
                raise ValueError(f"Held-out individual {individual!r} is missing or has an ambiguous FID_IID suffix")
            resolved.append(matches[0])
        if not resolved or len(set(resolved)) != len(resolved):
            raise ValueError("Held-out individuals must resolve to a nonempty set of unique donors")
        training = sorted(available - set(resolved))
        if not training:
            raise ValueError("Held-out individuals leave no training donors")
        return training, sorted(resolved)
    shuffled = np.random.default_rng(seed).permutation(ids)
    n_train = 4 * len(ids) // 5
    return sorted(shuffled[:n_train].tolist()), sorted(shuffled[n_train:].tolist())


class ReferenceDataset(Dataset):
    """One gene per sample, retaining each training donor's expression."""

    def __init__(self, features, targets, observed, gene_indices):
        self.features = features
        self.targets = targets  # individual x gene x cell type, already log1p
        self.observed = observed  # individual x cell type
        self.gene_indices = np.asarray(gene_indices)

    def __len__(self):
        return len(self.gene_indices)

    def __getitem__(self, index):
        gene = self.gene_indices[index]
        return self.features[gene], self.targets[:, gene], self.observed

    def for_cell_type(self, cell):
        return ReferenceDataset(self.features, self.targets[:, :, cell:cell + 1],
                                self.observed[:, cell:cell + 1], self.gene_indices)


def load_training_data(observations, targets, split_seed=42, held_out_individuals=None, cell_type=None):
    """Exclude held-out donors and test-gene expression before loading targets.

    Reading selected rows in chunks avoids densifying the complete h5ad. The
    train/validation views share storage and keep each donor's expression.
    """
    features, genes, chroms, provenance = load_reference(observations)
    warn_incomplete_provenance(provenance)
    adata = ad.read_h5ad(targets, backed="r")
    try:
        for column in ("individual_id", "cell_type"):
            if column not in adata.obs or adata.obs[column].isna().any():
                raise ValueError(f"Targets require non-missing obs[{column!r}]")
        donors = adata.obs.individual_id.astype(str).to_numpy()
        labels = adata.obs.cell_type.astype(str).to_numpy()
        if adata.obs.duplicated(["individual_id", "cell_type"]).any():
            raise ValueError("Targets contain duplicate (individual_id, cell_type) rows")
        held_out = read_ids(held_out_individuals) if held_out_individuals is not None else None
        train_ids, held_out_ids = split_individuals(donors, split_seed, held_out)
        cell_types = sorted(set(labels))
        if cell_type is not None:
            if cell_type not in cell_types:
                raise ValueError(f"Unknown cell type {cell_type!r}; choose from: {', '.join(cell_types)}")
            cell_types = [cell_type]
        target_genes = gene_ids(adata.var_names)
        target_index = {gene: i for i, gene in enumerate(target_genes)}
        aligned = np.asarray([gene in target_index for gene in genes])
        gene_splits = {
            name: genes[aligned & np.isin(chroms, chromosomes)].tolist()
            for name, chromosomes in CHROMOSOMES.items()
        }
        for name in ("train", "validation", "test"):
            if not gene_splits[name]:
                raise ValueError(f"No aligned genes in the {name} chromosome partition")
        used = aligned & np.isin(chroms, CHROMOSOMES["train"] + CHROMOSOMES["validation"])
        features, genes, chroms = features[used], genes[used], chroms[used]
        target_columns = np.asarray([target_index[gene] for gene in genes])
        donor_index = {donor: i for i, donor in enumerate(train_ids)}
        cell_index = {label: i for i, label in enumerate(cell_types)}
        expression = np.zeros((len(train_ids), len(genes), len(cell_types)), dtype=np.float32)
        observed = np.zeros((len(train_ids), len(cell_types)), dtype=bool)
        rows = np.flatnonzero(np.isin(donors, train_ids) & np.isin(labels, cell_types))
        for start in range(0, len(rows), 256):
            selected = rows[start:start + 256]
            block = adata[selected, :].X[:, target_columns]
            block = block.toarray() if sparse.issparse(block) else np.asarray(block)
            block = np.asarray(block, dtype=np.float32)
            if not np.isfinite(block).all() or (block < 0).any():
                raise ValueError("Targets must contain finite, nonnegative raw pseudo-bulk expression")
            for row, values in zip(selected, np.log1p(block)):
                i, c = donor_index[donors[row]], cell_index[labels[row]]
                expression[i, :, c] = values
                observed[i, c] = True
        if not observed.any(axis=0).all():
            missing = np.asarray(cell_types)[~observed.any(axis=0)].tolist()
            raise ValueError(f"No training individuals observed for cell types: {missing}")
    finally:
        adata.file.close()

    features, expression, observed = map(torch.from_numpy, (features, expression, observed))
    datasets = {
        name: ReferenceDataset(features, expression, observed, np.flatnonzero(np.isin(chroms, CHROMOSOMES[name])))
        for name in ("train", "validation")
    }
    metadata = {
        "split_seed": split_seed if held_out is None else None,
        "train_fraction": 0.8 if held_out is None else len(train_ids) / len(set(donors)),
        "training_individuals": train_ids,
        "held_out_individuals": held_out_ids,
        "cell_types": cell_types,
        "chromosomes": CHROMOSOMES,
        "genes": gene_splits,
        "target_transform": "log1p",
        "missing_cell_types": "masked",
        "observed_training_individuals_per_cell_type": observed.sum(dim=0).tolist(),
        "observations": str(Path(observations).resolve()),
        "targets": str(Path(targets).resolve()),
        "embedding_provenance": provenance,
    }
    if held_out_individuals is not None:
        metadata["held_out_individuals_source"] = str(Path(held_out_individuals).resolve())
    return datasets["train"], datasets["validation"], metadata
