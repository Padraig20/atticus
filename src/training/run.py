"""Train an independent ATTICUS(+V) reference ensemble for each cell type."""

import argparse
import json
import logging
import math
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.dataset import load_training_data
from src.training.losses import ensemble_loss
from src.training.model import DeepEnsemble
from src.tracking import Tracker, add_tracking_args, run_config


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-X", "--observations", type=Path, required=True,
                        help="Native REFERENCE.npz from embeddings.py")
    parser.add_argument("-y", "--targets", type=Path, required=True,
                        help="Raw pseudo-bulk .h5ad with obs individual_id/cell_type and Ensembl var_names")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output checkpoint (.pth)")
    parser.add_argument("--seed", type=int, default=0, help="Model/training seed; cell c, member k uses seed + 5*c + k")
    parser.add_argument("--split-seed", type=int, default=42, help="Fixed 80/20 donor split, independent of --seed")
    parser.add_argument("--held-out-individuals", type=Path,
                        help="Reserve these donor IDs instead of a random 20%%; exact IDs or unique FID_IID suffixes")
    parser.add_argument("-e", "--epochs", type=int, default=50)
    parser.add_argument("-b", "--batch-size", type=int, default=2048, help="Genes per batch, retaining all training donors")
    parser.add_argument("-lr", "--learning-rate", type=float, default=5e-4)
    parser.add_argument("-wd", "--weight-decay", type=float, default=5e-4)
    parser.add_argument("-nl", "--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--norm-layer", action="store_true", help="LayerNorm on each member's input features")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a torch device such as cuda:0")
    parser.add_argument("--pc-lambda", type=float, default=1.0, help="Weight of Pearson loss across genes per donor")
    parser.add_argument("--cossim-lambda", type=float, default=1.0)
    parser.add_argument("--pnll-lambda", type=float, default=0.1)
    parser.add_argument("--gnll-lambda", type=float, default=1.0)
    add_tracking_args(parser)
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.patience) < 1 or min(args.n_layers, args.workers, args.seed, args.split_seed) < 0:
        parser.error("Epochs, batch size and patience must be positive; depth, workers and seeds must be nonnegative")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0 or not 0 <= args.dropout < 1:
        parser.error("Weight decay must be finite and nonnegative; dropout must be in [0, 1)")
    weights = [args.pc_lambda, args.cossim_lambda, args.pnll_lambda, args.gnll_lambda]
    if any(not math.isfinite(value) or value < 0 for value in weights) or args.gnll_lambda <= 0:
        parser.error("Loss weights must be finite and nonnegative, and GNLL must have positive weight")
    if args.targets.suffix != ".h5ad":
        parser.error("Targets must be a population .h5ad, not averaged expression")
    return args


def make_scheduler(optimizer, total_steps):
    warmup_steps = int(0.1 * total_steps)

    def factor(step):
        if step < warmup_steps:
            return 0.01 + 0.99 * step / warmup_steps
        progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def run_epoch(model, loader, weights, device, optimizer=None, scheduler=None):
    model.train(optimizer is not None)
    totals = {name: 0.0 for name in ("loss", *weights)}
    n_samples = 0
    with torch.set_grad_enabled(optimizer is not None):
        for features, target, observed in loader:
            features, target, observed = (value.to(device) for value in (features, target, observed))
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            loss, metrics = ensemble_loss(model, features, target, observed, weights)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite ensemble loss; check inputs and optimization settings")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
                scheduler.step()
            size = len(features)
            n_samples += size
            totals["loss"] += loss.item() / model.n_members * size
            for name, value in metrics.items():
                totals[name] += value * size
    return {name: value / n_samples for name, value in totals.items()}


def train(model, train_loader, validation_loader, weights, args, device, cell_type, tracker=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs * len(train_loader))
    best_loss, best_epoch, best_state, bad_epochs = math.inf, 0, None, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        training = run_epoch(model, train_loader, weights, device, optimizer, scheduler)
        validation = run_epoch(model, validation_loader, weights, device)
        history.append({"cell_type": cell_type, "epoch": epoch, "train": training, "validation": validation,
                        "learning_rate": optimizer.param_groups[0]["lr"]})
        if tracker is not None:
            tracker.log({"epoch": epoch, f"train/{cell_type}/loss": training["loss"],
                         f"validation/{cell_type}/loss": validation["loss"]})
        logging.info("%s, epoch %d: train loss %.6f; validation loss %.6f",
                     cell_type, epoch, training["loss"], validation["loss"])
        if validation["loss"] < best_loss - 1e-6:
            best_loss, best_epoch, bad_epochs = validation["loss"], epoch, 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            logging.info("%s: early stopping after %d epochs without improvement", cell_type, args.patience)
            break
    model.load_state_dict(best_state)
    model.zero_grad(set_to_none=True)
    model.eval()
    return history, best_epoch, best_loss


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    with Tracker(args, stage="training") as tracker:
        train_data, validation_data, metadata = load_training_data(
            args.observations, args.targets, args.split_seed, args.held_out_individuals)
        logging.info("Using %d training individuals; reserving %d individuals. Genes: %d train / %d validation / %d held out",
                     len(metadata["training_individuals"]), len(metadata["held_out_individuals"]),
                     *(len(metadata["genes"][name]) for name in ("train", "validation", "test")))
        config = {"input_dim": train_data.features.shape[1], "output_dim": len(metadata["cell_types"]),
                  "n_layers": args.n_layers, "dropout": args.dropout, "layer_norm": args.norm_layer, "seed": args.seed}
        model = DeepEnsemble(**config)
        weights = {"pc": args.pc_lambda, "cosine": args.cossim_lambda,
                   "pnll": args.pnll_lambda, "gnll": args.gnll_lambda}
        tracker.configure({"model": config, "ensemble_members_per_cell_type": model.n_members, "loss_weights": weights,
                           "training_individuals": len(metadata["training_individuals"]),
                           "held_out_individuals": len(metadata["held_out_individuals"]),
                           "cell_types": len(metadata["cell_types"]),
                           **{f"{name}_genes": len(genes) for name, genes in metadata["genes"].items()}})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix(".splits.json").write_text(json.dumps(metadata, indent=2) + "\n")
        history, best_epochs, best_losses, summaries = [], {}, {}, []
        for cell, (cell_type, ensemble) in enumerate(zip(metadata["cell_types"], model.ensembles)):
            seed = args.seed + cell * model.n_members
            torch.manual_seed(seed)
            generator = torch.Generator().manual_seed(seed)
            train_loader = DataLoader(train_data.for_cell_type(cell), batch_size=args.batch_size, shuffle=True,
                                      generator=generator, num_workers=args.workers)
            validation_loader = DataLoader(validation_data.for_cell_type(cell), batch_size=args.batch_size,
                                           num_workers=args.workers)
            logging.info("Training cell type %d/%d: %s", cell + 1, len(model.ensembles), cell_type)
            cell_history, best_epoch, best_loss = train(
                ensemble.to(device), train_loader, validation_loader, weights, args, device, cell_type, tracker)
            ensemble.cpu()
            history.extend(cell_history)
            best_epochs[cell_type], best_losses[cell_type] = best_epoch, best_loss
            summaries.append([cell_type, best_epoch, best_loss, len(cell_history)])
        checkpoint = {
            "format_version": 2, "model_config": config,
            "model_state_dict": {name: value.cpu() for name, value in model.state_dict().items()},
            "loss_weights": weights, "data": metadata, "best_epoch": best_epochs,
            "best_validation_loss": best_losses, "history": history, "training_config": run_config(args),
        }
        temporary = args.output.with_name(args.output.name + ".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(args.output)
        args.output.with_suffix(".history.json").write_text(json.dumps(history, indent=2) + "\n")
        tracker.summarize({"mean_best_validation_loss": float(np.mean(list(best_losses.values())))})
        tracker.table("training/cell_types", ["cell_type", "best_epoch", "best_validation_loss", "epochs_completed"], summaries)
        logging.info("Saved %d independent cell-type ensembles to %s", len(model.ensembles), args.output)


if __name__ == "__main__":
    main()
