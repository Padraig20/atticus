"""Run the bundled S-PrediXcan executable and read its association results."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.stats import norm


NUMERIC = ("zscore", "effect_size", "pvalue", "var_g", "n_snps_in_model", "n_snps_in_cov", "n_snps_used")
ORDERED_LAUNCHER = Path(__file__).with_name("ordered_metaxcan.py")


def source_hash(directory, ordered=False):
    directory = Path(directory)
    if not (directory / "SPrediXcan.py").is_file():
        raise FileNotFoundError("Missing metaxcan/software/SPrediXcan.py; initialize the metaxcan submodule")
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(path.read_bytes())
    if ordered:
        digest.update(ORDERED_LAUNCHER.name.encode())
        digest.update(ORDERED_LAUNCHER.read_bytes())
    return digest.hexdigest()


def read_results(path):
    results = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"gene", *NUMERIC} <= set(reader.fieldnames or []):
            raise ValueError(f"Unexpected MetaXcan result columns in {path}")
        for row in reader:
            gene = row["gene"]
            if gene in results:
                raise ValueError(f"Duplicate MetaXcan result for {gene}")
            result = {key: np.nan if row[key] in ("", "NA", "nan") else float(row[key]) for key in NUMERIC}
            pvalue = result["pvalue"]
            if np.isfinite(pvalue) and not 0 <= pvalue <= 1:
                raise ValueError(f"Invalid MetaXcan p-value for {gene}")
            valid = np.isfinite(result["zscore"]) and np.isfinite(pvalue)
            logp = np.nan
            if valid:
                # Recover underflowed tails from MetaXcan's returned z-score.
                logp = np.log(pvalue) if pvalue > 0 else np.log(2) + norm.logsf(abs(result["zscore"]))
            result["log_pvalue"] = float(logp)
            result["neg_log10_pvalue"] = -result["log_pvalue"] / np.log(10)
            result["status"] = "tested" if valid else "undefined_metaxcan_result"
            results[gene] = result
    return results


def run_spredixcan(directory, draw, gwas, output):
    """Run a real CLI subprocess and retain the command, log, and raw CSV."""
    directory, output = Path(directory).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(directory / "SPrediXcan.py")]
    if "gene_order" in draw:
        command = [sys.executable, str(ORDERED_LAUNCHER), str(directory / "SPrediXcan.py"), draw["gene_order"]]
    command.extend(["--model_db_path", draw["model_db"], "--covariance", draw["covariance"],
               "--gwas_file", str(Path(gwas).resolve()), "--snp_column", "SNP",
               "--effect_allele_column", "A1", "--non_effect_allele_column", "A2",
               "--beta_column", "BETA", "--se_column", "SE", "--keep_non_rsid",
               "--stream_covariance", "--output_file", str(output) + ".partial",
               "--throw", "--overwrite", "--verbosity", "10"])
    command_file = output.with_suffix(".command.json")
    command_file.write_text(json.dumps(command, indent=2) + "\n")
    if output.is_file():
        return read_results(output)
    log = output.with_suffix(".log")
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    with log.open("w") as handle:
        completed = subprocess.run(command, cwd=directory, env=env, stdout=handle, stderr=subprocess.STDOUT)
    partial = Path(str(output) + ".partial")
    if completed.returncode != 0 or not partial.is_file():
        raise RuntimeError(f"S-PrediXcan failed for {output.stem} (exit {completed.returncode}); see {log}")
    results = read_results(partial)
    partial.replace(output)
    return results
