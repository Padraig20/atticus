"""Export personalized ATTICUS teacher means and variances for distillation."""

import argparse
import json
import logging
from pathlib import Path
import sys

import h5py
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.preprocessing.geuvadis import ROOT, read_ids
from src.preprocessing.embeddings import (
    load_personalized, personalized_files, personalized_path,
    read_provenance, validate_provenance, warn_incomplete_provenance,
)
from src.training.model import DeepEnsemble, validate_checkpoint
from src.tracking import Tracker, add_tracking_args


def fingerprint(path):
    path = Path(path).resolve()
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def embedding_fingerprint(path):
    return ([fingerprint(file) for file in personalized_files(path)]
            if Path(path).is_dir() else fingerprint(path))


def strings(group, name, values):
    group.create_dataset(name, data=np.asarray(values, dtype=object), dtype=h5py.string_dtype())


def embedding_metadata(data):
    genes = np.asarray([str(g).split(".")[0] for g in data["ensids"]])
    chroms = np.asarray([str(c).lower().removeprefix("chr") for c in data["chroms"]])
    tss = np.asarray(data["tss"])
    autosomes = {str(i) for i in range(1, 23)}
    if len(set(genes)) != len(genes) or not set(chroms) <= autosomes:
        raise ValueError("Expected unique autosomal gene IDs in the embeddings")
    if (genes.ndim != 1 or chroms.shape != genes.shape or tss.shape != genes.shape
            or not np.isfinite(tss).all() or (tss < 1).any() or (tss != tss.astype(np.int64)).any()):
        raise ValueError("Expected one chromosome and positive integer TSS per gene")
    return genes, chroms, tss.astype(np.int64)


def export_predictions(teacher, embeddings, samples, output, batch_size=256, device="cpu", tracker=None):
    samples = list(samples)
    if len(samples) < 5 or len(set(samples)) != len(samples) or "REFERENCE" in samples:
        raise ValueError("Distillation requires at least five unique personalized samples")
    paths = [personalized_path(embeddings, sample) for sample in samples]
    with load_personalized(paths[0]) as first:
        source_genes, source_chroms, source_tss = embedding_metadata(first)
        tissue = str(first["tissue"].item())
        provenance = read_provenance(first)
    config = {"embedding_format_version": 1, "teacher_format_version": 2,
              "teacher": fingerprint(teacher), "embeddings": [embedding_fingerprint(p) for p in paths],
              "samples": samples}
    config_json = json.dumps(config, sort_keys=True)
    output = Path(output)
    if output.exists():
        with h5py.File(output) as saved:
            if "config" not in saved or saved["config"].asstr()[()] != config_json or not saved.attrs.get("complete", False):
                raise ValueError("Prediction inputs changed or output is incomplete; use a new output path")
            if not saved.attrs["reference_provenance_verified"]:
                logging.warning("Cached predictions have unverified backbone provenance; their teacher and embedding files are unchanged")
            if tracker is not None:
                tracker.configure({**dict(zip(("n_genes", "n_ensemble_members", "n_individuals", "n_cell_types"),
                                              saved["means"].shape)), "target_scale": "log1p"})
                tracker.summarize({"prediction/individuals_completed": len(samples),
                                   "prediction/individuals_predicted_this_run": 0, "prediction/cache_reused": True})
        logging.info("Teacher predictions already complete: %s", output)
        return
    checkpoint = torch.load(teacher, map_location="cpu", weights_only=True)
    validate_checkpoint(checkpoint)
    reference_provenance = checkpoint["data"]["embedding_provenance"]
    validate_provenance(reference_provenance, provenance)
    warn_incomplete_provenance(reference_provenance)
    if provenance != reference_provenance:
        warn_incomplete_provenance(provenance)
    model = DeepEnsemble(**checkpoint["model_config"]).to(device).eval()
    model.load_state_dict(checkpoint["model_state_dict"])
    cell_types = checkpoint["data"]["cell_types"]
    if not len(source_genes):
        raise ValueError("Embeddings contain no genes")
    order = np.lexsort((source_tss, source_chroms.astype(int)))
    shape = (len(source_genes), model.n_members, len(samples), len(cell_types))
    tracking = tracker is not None and tracker.enabled
    if tracker is not None:
        tracker.configure({**dict(zip(("n_genes", "n_ensemble_members", "n_individuals", "n_cell_types"), shape)),
                           "target_scale": "log1p", "uncertainty_summary_scope": "individuals_predicted_this_run"})
    aleatoric_total = np.zeros(len(cell_types))
    epistemic_total = np.zeros(len(cell_types))
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    with h5py.File(partial, "a") as handle:
        if "config" in handle and handle["config"].asstr()[()] != config_json:
            raise ValueError("Partial prediction cache has different inputs; use a new output path")
        if "config" not in handle:
            handle.create_dataset("config", data=config_json, dtype=h5py.string_dtype())
            handle.attrs.update(format_version=1, complete=False,
                                completed_samples=0, target_scale="log1p", tissue=tissue,
                                reference_provenance_verified=all(value is not None for value in reference_provenance.values()),
                                axis_order="gene,member,individual,cell_type")
            for name, values in (("sample_ids", samples), ("genes", source_genes[order]),
                                 ("chroms", source_chroms[order]), ("cell_types", cell_types)):
                strings(handle, name, values)
            handle.create_dataset("tss", data=source_tss[order])
            for name in ("means", "variances"):
                handle.create_dataset(name, shape=shape, dtype="f4", compression="gzip", compression_opts=1,
                                      chunks=(min(32, shape[0]), shape[1], 1, shape[3]))
        completed_before = int(handle.attrs["completed_samples"])
        progress_every = max(1, (len(samples) + 99) // 100)
        console_every = max(1, (len(samples) + 24) // 25)
        for i in range(completed_before, len(samples)):
            with load_personalized(paths[i]) as data:
                metadata = embedding_metadata(data)
                if read_provenance(data) != provenance:
                    raise ValueError(f"Embedding provenance differs between personalized individuals: {paths[i]}")
                if (str(data["sample_id"].item()) != samples[i] or str(data["tissue"].item()) != tissue
                        or any(not np.array_equal(a, b) for a, b in zip(metadata, (source_genes, source_chroms, source_tss)))):
                    raise ValueError(f"Sample, tissue, or gene metadata mismatch in {paths[i]}")
                features = np.asarray(data["features"], dtype=np.float32)
            if features.shape != (len(source_genes), checkpoint["model_config"]["input_dim"]) or not np.isfinite(features).all():
                raise ValueError(f"Invalid feature shape or values in {paths[i]}")
            aleatoric_sum = np.zeros(len(cell_types))
            epistemic_sum = np.zeros(len(cell_types))
            with torch.inference_mode():
                for start in range(0, len(order), batch_size):
                    stop = min(start + batch_size, len(order))
                    batch = torch.from_numpy(features[order[start:stop]]).to(device)
                    means, variances = model.member_predictions(batch)
                    if not torch.isfinite(means).all() or not torch.isfinite(variances).all():
                        raise ValueError(f"Non-finite teacher predictions for {samples[i]}")
                    mean_values = means.cpu().numpy().transpose(1, 0, 2)
                    variance_values = variances.cpu().numpy().transpose(1, 0, 2)
                    handle["means"][start:stop, :, i, :] = mean_values
                    handle["variances"][start:stop, :, i, :] = variance_values
                    if tracking:
                        aleatoric_sum += variance_values.mean(axis=1, dtype=np.float64).sum(axis=0)
                        epistemic_sum += mean_values.var(axis=1, dtype=np.float64).sum(axis=0)
            handle.attrs["completed_samples"] = i + 1
            handle.flush()
            if i == completed_before or (i + 1) % console_every == 0 or i + 1 == len(samples):
                logging.info("Teacher predictions: %d/%d individuals", i + 1, len(samples))
            if tracking:
                aleatoric_total += aleatoric_sum
                epistemic_total += epistemic_sum
                if (i + 1) % progress_every == 0 or i + 1 == len(samples):
                    tracker.log({"prediction/individuals_completed": i + 1})
        handle.attrs["complete"] = True
    partial.replace(output)
    if tracker is not None:
        predicted = len(samples) - completed_before
        tracker.summarize({"prediction/individuals_completed": len(samples),
                           "prediction/individuals_predicted_this_run": predicted, "prediction/cache_reused": False})
        if tracking and predicted:
            aleatoric = aleatoric_total / (len(order) * predicted)
            epistemic = epistemic_total / (len(order) * predicted)
            tracker.summarize({"prediction/mean_aleatoric_variance": float(aleatoric.mean()),
                               "prediction/mean_epistemic_variance": float(epistemic.mean())})
            tracker.table("prediction/uncertainty_by_cell_type",
                          ["cell_type", "mean_aleatoric_variance", "mean_epistemic_variance"],
                          [[cell, float(aleatoric[c]), float(epistemic[c])] for c, cell in enumerate(cell_types)])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True,
                        help="Directory of personalized <sample>.npz files or trusted <sample>/*.npy directories")
    parser.add_argument("--samples", type=Path, default=ROOT / "data/geuvadis/samples.txt")
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    add_tracking_args(parser)
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    with Tracker(args, stage="prediction") as tracker:
        export_predictions(args.teacher, args.embeddings, read_ids(args.samples), args.output,
                           args.batch_size, device, tracker)


if __name__ == "__main__":
    main()
