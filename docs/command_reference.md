# Command reference

[Installation](../README.md#getting-started) · [Tutorials](../README.md#try-it-out)

Run from the repository root after installing the analysis dependencies in the
README. `--no-sync` keeps that environment intact. Add `--help` to any Python
command for all options. Already have linear models? [Jump straight to TWAS](#use-the-pretrained-onek1k-or-rosmap-models).

## Data and VariantFormer embeddings

See [Get the data](../README.md#get-the-data) for download availability.
Place expression at `data/onek1k/expression.h5ad` and extract ready-made embeddings
under `data/onek1k/embeddings/`: `reference/REFERENCE.npz` for training, and
`individuals/<sample>.npz` plus `individuals/samples.txt` for personalized prediction.
**Ready-made embeddings let you skip VariantFormer installation and inference.**
The distillation examples below use a separate Geuvadis genotype/embedding panel.

### Expression file

The AnnData `.h5ad` contains one row per observed individual/cell-type pair:

| Field | Contents |
| --- | --- |
| `X` | Finite, nonnegative **mean raw counts** across that individual's cells of each type; dense or sparse, with genes as columns. |
| `obs["individual_id"]`, `obs["cell_type"]` | Non-missing labels; each pair occurs once. |
| `var_names` | Ensembl gene IDs, unique after removing version suffixes. |

Leave counts unnormalized and untransformed; training applies `log1p`.
Omit unobserved individual/cell-type pairs. No extra layers are needed.

### Generate embeddings (optional)

On a Linux CUDA machine, prepare the 503-person reference panel and generate
any missing embeddings. Skip the reference command if you downloaded `REFERENCE.npz`.

```bash
bash setup.sh --geuvadis --variantformer
uv run --no-sync --extra variantformer python src/preprocessing/embeddings.py \
  --tissue blood --reference --output data/onek1k/embeddings/reference
uv run --no-sync --extra variantformer python src/preprocessing/embeddings.py \
  --tissue blood --samples data/geuvadis/samples.txt \
  --vcf-dir data/geuvadis/individuals --output data/embeddings/geuvadis/blood
```

Reference and personalized embeddings must use the same tissue and backbone.
For ROSMAP, use matching pseudobulk expression and `--tissue 'brain - frontal cortex (ba9)'`.
Generation covers autosomal protein-coding genes and may download gene-specific
assets on first use. Training aligns expression and embedding gene IDs; later
stages use the genes in their inputs. Manuscript gene lists are reference material only.

## ATTICUS(+V) reference training

Train an independent five-member MLP ensemble for each cell type:

```bash
uv run --no-sync --extra training python src/training/run.py \
  -X data/onek1k/embeddings/reference/REFERENCE.npz \
  -y data/onek1k/expression.h5ad \
  --held-out-individuals data/data-for-the-paper/test-individuals/onek1k.txt \
  -o outputs/atticus-v-blood.pth
```

This reserves the listed evaluation donors. Omit `--held-out-individuals` for
the default seeded 80/20 donor split (`--split-seed 42`). Validation uses genes
on chromosomes 7, 11 and 14; test genes on 5, 12 and 20 stay reserved. Fitting
and validation use only training donors.

Pearson and centered-cosine losses operate across genes for each donor/cell type,
alongside Poisson and Gaussian NLL. Tune their configurable weights on validation
genes using `--pc-lambda`, `--cossim-lambda`, `--pnll-lambda`, and `--gnll-lambda`.
Each ensemble restores its best validation weights. Outputs are the `.pth`
checkpoint, `.splits.json`, and `.history.json`; predictions use the `log1p` scale.

After model selection, evaluate the reserved test genes:

```bash
uv run --no-sync --extra training python src/training/evaluate.py \
  --checkpoint outputs/atticus-v-blood.pth \
  -o outputs/atticus-v-blood-reference-test
```

`summary.json` and `cell_types.tsv` report expression Pearson correlation,
epistemic variance versus squared error (Spearman), and aleatoric variance versus
population variance (Pearson). These compare genes using donor-averaged `log1p` expression
from training individuals; they do not evaluate held-out personalized predictions.
Use `-X` and `-y` to override saved input paths after moving data.

## Distillation

Export personalized teacher predictions, then fit genotype-based students:

```bash
uv run --no-sync --extra distillation python src/distillation/predict.py \
  --teacher outputs/atticus-v-blood.pth \
  --embeddings data/embeddings/geuvadis/blood \
  -o outputs/geuvadis-blood-teacher.h5

uv run --no-sync --extra distillation python src/distillation/run.py \
  -y outputs/geuvadis-blood-teacher.h5 \
  -o outputs/students-probabilistic
```

Prediction defaults to `data/geuvadis/samples.txt`. Each sample needs a matching
`<sample>.npz`; all archives must agree on gene, TSS, tissue, and backbone metadata.
For OneK1K predictions, change `--embeddings` and `--samples` to its individual
embedding directory and sample list. Distillation requires matching genotypes.

| Distillation option | Effect |
| --- | --- |
| Default | Probabilistic, heterozygote-aware: 5 bootstraps per teacher member, 25 fits per gene/cell type. |
| `--mode deterministic` | One unweighted fit to the ensemble mean. |
| `--additive` | Dosage-only models; otherwise include heterozygote indicators. Works with either mode. |
| `--half-window`, `--maf` | Cis-window half-width in bp (500,000) and minimum reference-panel MAF (0.05). |
| `--alpha` | Fixed elastic-net penalty; otherwise selected by training-fold CV. |
| `--jobs` | Concurrent cell-type fits; default 1. Higher values use more memory. |
| `--genotypes` | PLINK directory; default `data/geuvadis/plink/`. |

The only genotype filename template is `chr{chrom}.GRCh38.biallelic.{bed,bim,fam}`.
Dosages count BIM **A1**; donor IDs are aligned to the predictions. All complete,
biallelic cis-SNPs passing QC enter fitting, with **no SNP screening**. The fixed
80/20 donor split gives 402 training and 101 test donors for the 503-person panel.
Student targets use the inverse-normal scale; held-out scores measure fidelity
to teacher predictions.

Outputs are `students.h5` (weights, draws, donor split, metrics, and covariance),
`run.json`, and `models/` (PredictDB databases and MetaXcan covariance files).
Covariance is saved after each gene/cell-type fit using all reference donors.
Heterozygote-aware fits also save additive TWAS projections using full-panel
allele frequencies. **Keep the complete model directory together.**

## Association testing

```bash
uv run --no-sync --extra twas python src/twas/run.py \
  --models outputs/students-probabilistic \
  --gwas data/data-for-the-paper/gwas/gwas-sle.tsv.gz \
  -o outputs/twas-sle
```

This calls the bundled **MetaXcan S-PrediXcan**, which handles SNP matching,
allele alignment, and association statistics. Model settings are read from
`students.h5`; saved weights and covariance are reused, so PLINK files are no
longer needed. `--jobs` controls concurrent MetaXcan processes (default 1).
For another GWAS, keep `--models` and choose a new output directory.

### Use the pretrained OneK1K or ROSMAP models

Download and unpack the complete model directory under `models/onek1k/` (blood)
or `models/rosmap/` (brain); see [Get the data](../README.md#get-the-data).
These large files are not included in a fresh clone. Skip training and distillation:

```bash
uv run --no-sync --extra twas python src/twas/run.py \
  -m models/onek1k --gwas data/data-for-the-paper/gwas/gwas-sle.tsv.gz \
  -o outputs/twas-onek1k-sle
```

For ROSMAP/AD, use `-m models/rosmap`, `gwas-ad.tsv.gz`, and a separate output
directory. In [notebook 04](../notebooks/04_twas.ipynb), choose the corresponding
`STUDENT_DIR` path.

Supplied fits retain their original feature screening (up to 5,000 expanded
features), per-gene donor splits, and bootstrap-specific projection frequencies.
New distillation uses no screening and full-panel projection frequencies; TWAS
uses either model's saved weights unchanged. TWAS considers all model genes: 17,967
for OneK1K and 15,959 for ROSMAP.

### GWAS inputs and outputs

Use a TSV or gzipped TSV with GRCh38, forward-strand alleles:

| Column | Meaning |
| --- | --- |
| `ID` | SNP identifier matching the models, or an equivalent coordinate ID. |
| `ALT`, `REF` | Effect and other allele. |
| `ES`, `SE` | Signed effect for `ALT` (log odds for binary traits) and positive standard error. |

Override column names with `--snp-column`, `--effect-column`, `--other-column`,
`--beta-column`, and `--se-column`. Prepare any needed liftover or strand
complementation before testing.

| Output | Contents |
| --- | --- |
| `associations.tsv` | One row per gene/cell type: `gene`, `cell_type`, `zscore`, `effect_size`, `pvalue`, `neg_log10_pvalue`, `qvalue`, `n_draws_tested`, `status`. |
| `draws.tsv.gz` | Individual draw results, SNP counts, model variances, and test statuses. |
| `run.json` | Settings, completion status, matching diagnostics, and per-cell-type testing/significance counts. |
| `metaxcan/` | Original MetaXcan results, commands, and logs. |
| `gwas.model-snps.tsv.gz` | GWAS subset used for testing. |

Probabilistic results combine available draw p-values with equal-weight ACAT
and average Z-scores/effects; deterministic results use the single test.
**The ACAT p-value is not the p-value of the mean Z-score.** Use
`neg_log10_pvalue` for extreme tails. BH q-values and Bonferroni thresholds
(0.05 / tested genes) are calculated within each cell type, excluding undefined
tests. `--fdr` controls BH significant-gene counts (default 0.05).

## Reruns and logging

Embedding generation, prediction, distillation, and TWAS reuse matching completed
work. Changed inputs or analysis settings require a new output directory/file.
For TWAS, `--model-cache` chooses a separate export cache; `--genotypes` explicitly
rebuilds covariance from PLINK instead of using saved covariance.

W&B is optional for training, evaluation, prediction, distillation, and TWAS.
Append `--wandb-project atticus --wandb-mode offline` for local logs. For online
logging, authenticate with `uv run --no-sync wandb login` and use
`--wandb-project atticus`. `--wandb-mode disabled` turns logging off; `--wandb-group`
and `--wandb-name` label related runs. `WANDB_PROJECT` can enable a project across commands.

Logs focus on training/validation loss, evaluation scores, prediction uncertainty,
distillation fidelity and nonzero weights, and TWAS coverage/significant-gene
counts. Large predictions, weights, and association tables stay local.

## Check the installation

Run the synthetic workflow tests, including the bundled MetaXcan command:

```bash
uv run --no-sync --extra training --extra distillation --extra twas \
  python -m unittest discover -s tests -v
```
