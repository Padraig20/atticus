#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

geuvadis=false
variantformer=false
for arg in "$@"; do
    case "$arg" in
        --geuvadis) geuvadis=true ;;
        --variantformer) variantformer=true ;;
        -h|--help)
            echo "Usage: bash setup.sh [--geuvadis] [--variantformer]"
            echo "Update submodules and the uv environment."
            echo "--geuvadis downloads and prepares Geuvadis BED and individual VCF files."
            echo "--variantformer installs VariantFormer and downloads its model assets."
            exit 0 ;;
        *) echo "Unknown option: $arg" >&2; exit 1 ;;
    esac
done
required_tools=(git uv)
if "$geuvadis"; then
    required_tools+=(curl bcftools plink2)
fi
for tool in "${required_tools[@]}"; do
    command -v "$tool" >/dev/null || { echo "Missing $tool. Install it or load its module so it is on PATH." >&2; exit 1; }
done
if "$geuvadis"; then
    bcftools --version >/dev/null
    plink2 --version >/dev/null
    if ! bcftools plugin -l | grep -qx split; then
        echo "bcftools +split is missing. Install bcftools 1.21 or newer with plugins." >&2
        exit 1
    fi
fi

git submodule sync --recursive
git submodule update --init --recursive
if "$variantformer"; then
    bash src/preprocessing/install_variantformer.sh
else
    uv sync --locked --inexact
fi
if "$geuvadis"; then
    uv run --no-sync python src/preprocessing/geuvadis.py
fi
