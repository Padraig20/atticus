# atticus

Cool introduction... TODO later.

## Getting Started

Installing new projects is incredibly annoying, but we'll try to make the whole process bearable. We use `uv`, which is [https://docs.astral.sh/uv/getting-started/installation/#installation-methods](pretty easy to install).

```bash
curl -LsSf https://astral.sh/uv/install.sh | less
```

Make sure that `bcftools` (1.21 or newer, with the `split` plugin), and `plink2`, `samtools` (both 1.21 or newer), and `bedtools` are available; can easily be downloaded via bioconda! Then run the following script:

```bash
bash setup.sh
```

Setup uses `uv` to set up the virtual environment and initializes the pinned `metaxcan` and `variantformer` submodules. You won't have to change repositories and can comfortably stay in this project! Data and model downloads are opt-in. Use either flag independently, or combine them:

```bash
bash setup.sh --geuvadis
bash setup.sh --variantformer
bash setup.sh --geuvadis --variantformer
```

#### Geuvadis

This will be something you'll need for model distillation. Don't worry, we'll preprocess the data for you. The script downloads and prepares the GRCh38 autosomes from [Zenodo record 16538482](https://zenodo.org/records/16538482). It selects exactly the 503 IDs in `data/1000G_EUR_503.txt`. The download is about 27 GB; you might want to allow additional disk space for the prepared data... Outputs will be saved under `data/geuvadis/`.

#### VariantFormer

Will install everything you'll need to get VariantFormer embeddings. It downloads only the protein-coding model's required checkpoint, tokenizer, annotation, regulatory regions, and GRCh38 reference.

Generate embeddings for Geuvadis individuals or the reference genome (more about that later in the tutorial):

```bash
uv run --extra variantformer python src/preprocessing/embeddings.py --tissue blood
uv run --extra variantformer python src/preprocessing/embeddings.py --tissue blood --reference
```
