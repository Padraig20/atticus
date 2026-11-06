"""Shared paths and preview helpers for the OneK1K tutorial notebooks."""

from pathlib import Path
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/tutorial/onek1k"
REFERENCE = ROOT / "data/onek1k/embeddings/reference/REFERENCE.npz"
PATHS = {
    "expression": ROOT / "data/onek1k/expression.h5ad",
    "reference": (REFERENCE if REFERENCE.is_file()
                  else ROOT / "data/embeddings/reference/blood/REFERENCE.npz"),
    "personalized": ROOT / "data/embeddings/geuvadis/blood",
    "samples": ROOT / "data/geuvadis/samples.txt",
    "genotypes": ROOT / "data/geuvadis/plink",
    "output": OUTPUT,
    "teacher": OUTPUT / "teacher.pth",
    "predictions": OUTPUT / "geuvadis-teacher.h5",
}


def run(command, execute=False):
    command = [str(value) for value in command]
    print(("Running: " if execute else "Preview: ") + shlex.join(command), flush=True)
    if execute:
        return subprocess.run(command, cwd=ROOT, check=True)


def uv(extra, script, *arguments):
    return ["uv", "run", "--no-sync", "--extra", extra, "python", str(ROOT / script),
            *map(str, arguments)]


def require(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(
            "Complete the preceding preparation step first. Missing files:\n" + "\n".join(missing)
        )


def file_fingerprint(path):
    path = Path(path).resolve()
    if not path.is_file():
        return None
    stat = path.stat()
    return [str(path), stat.st_size, stat.st_mtime_ns]


def load_native_checkpoint(path):
    import pickle
    import torch
    from src.preprocessing.embeddings import validate_provenance
    from src.training.model import validate_checkpoint

    message = (
        "Use a checkpoint produced by src/training/run.py. "
        "Choose a new output path to train a checkpoint."
    )
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (EOFError, RuntimeError, pickle.UnpicklingError) as error:
        raise ValueError(message) from error
    validate_checkpoint(checkpoint)
    provenance = checkpoint["data"].get("embedding_provenance")
    validate_provenance(provenance, provenance)
    return checkpoint


def student_draw_count(students):
    """Read the draw axes from a saved fit; return None before any fit is saved."""
    for gene in students["genes"].values():
        for cell in gene.values():
            if "coefficient_draws" in cell:
                members, bootstraps, _ = cell["coefficient_draws"].shape
                return members * bootstraps
    return None
