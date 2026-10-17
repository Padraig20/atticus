#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../variantformer"

if [[ "$(uname -s)-$(uname -m)" != Linux-x86_64 ]]; then
    echo "The VariantFormer installation requires Linux x86_64 (CUDA 12)." >&2
    exit 1
fi
for tool in uv samtools bedtools bcftools; do
    command -v "$tool" >/dev/null || { echo "Missing $tool. Install it or load its module so it is on PATH." >&2; exit 1; }
done
uv sync --project .. --locked --inexact --extra variantformer

uv run --project .. --no-sync python - <<'PY'
from pathlib import Path
import boto3
from botocore import UNSIGNED
from botocore.config import Config
import download_artifacts as assets

required = {
    "all_genes_v1_pcg_gencodeV24.csv",
    "pretrained_tokenizers_checkpoint.pth",
    "v4_pcg_epoch11_checkpoint.pth",
    "ENCFF234XEZ.bed.gz",
    "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz",
}
assets.ARTIFACTS = [item for item in assets.ARTIFACTS if item.local_path in required]
if {item.local_path for item in assets.ARTIFACTS} != required:
    raise RuntimeError("VariantFormer artifact list changed")
client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
errors = assets.download(client, Path("_artifacts"))
if errors:
    raise RuntimeError("\n".join(errors))
PY
samtools faidx _artifacts/GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz
