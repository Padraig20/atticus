"""Independent probabilistic MLP ensembles for each cell type."""

import torch
from torch import nn


class ProbabilisticMLP(nn.Module):
    def __init__(self, input_dim, n_layers=2, dropout=0.05, layer_norm=False):
        super().__init__()
        sizes = torch.linspace(input_dim, 2, n_layers + 2).to(torch.int64).tolist()
        layers = [nn.LayerNorm(input_dim)] if layer_norm else []
        for i, (left, right) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(left, right))
            if i < n_layers:
                layers.extend([nn.ReLU(), nn.Dropout(dropout)])
        self.network = nn.Sequential(*layers)

    def forward(self, features):
        mean, variance = nn.functional.softplus(self.network(features)).chunk(2, dim=-1)
        return mean, variance.clamp_min(1e-8)


class CellTypeEnsemble(nn.Module):
    """Five independently initialized mean/variance MLPs for one cell type."""

    n_members = 5

    def __init__(self, input_dim, n_layers=2, dropout=0.05, layer_norm=False, seed=0):
        super().__init__()
        if input_dim < 1 or n_layers < 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid ensemble dimensions, depth, or dropout")
        self.models = nn.ModuleList()
        for member in range(self.n_members):
            torch.manual_seed(seed + member)
            self.models.append(ProbabilisticMLP(input_dim, n_layers, dropout, layer_norm))

    def member_predictions(self, features):
        predictions = [model(features) for model in self.models]
        return tuple(torch.stack(values) for values in zip(*predictions))


class DeepEnsemble(nn.Module):
    """One independent ensemble per cell type; predictions are member × gene × cell."""

    n_members = CellTypeEnsemble.n_members

    def __init__(self, input_dim, output_dim, n_layers=2, dropout=0.05, layer_norm=False, seed=0):
        super().__init__()
        if output_dim < 1:
            raise ValueError("At least one cell type is required")
        self.ensembles = nn.ModuleList([
            CellTypeEnsemble(input_dim, n_layers, dropout, layer_norm, seed + cell * self.n_members)
            for cell in range(output_dim)
        ])

    def member_predictions(self, features):
        predictions = [ensemble.member_predictions(features) for ensemble in self.ensembles]
        return tuple(torch.cat(values, dim=-1) for values in zip(*predictions))

    def forward(self, features):
        means, variances = self.member_predictions(features)
        return means.mean(dim=0), variances.mean(dim=0), means.var(dim=0, unbiased=False)


def validate_checkpoint(checkpoint):
    """Reject shared-cell teachers and mismatched cell labels before inference."""
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 2:
        raise ValueError("Expected a per-cell-type teacher checkpoint (format_version=2). "
                         "Retrain with src/training/run.py; shared-MLP checkpoints are incompatible.")
    if not all(isinstance(checkpoint.get(key), dict)
               for key in ("model_config", "model_state_dict", "data")):
        raise ValueError("Teacher checkpoint is missing its model configuration, weights, or data metadata")
    metadata = checkpoint["data"]
    cells = metadata.get("cell_types")
    if (metadata.get("target_transform") != "log1p" or not isinstance(cells, list) or not cells
            or any(not isinstance(cell, str) or not cell for cell in cells)
            or len(set(cells)) != len(cells)
            or checkpoint["model_config"].get("output_dim") != len(cells)):
        raise ValueError("Teacher checkpoint requires log1p targets and one model output per unique cell-type label")
