# abyss

Deep-learning correction for **population stratification** in genome-wide association studies (GWAS) and polygenic risk scores (PRS).

abyss splits SNPs into bins by allele frequency and LD block, trains an autoencoder on each bin to learn a low-dimensional **latent representation of population structure**, and uses those latent factors as covariates in the GWAS instead of (or alongside) principal components.

## Install

```bash
pip install -e .
```

Python 3 with TensorFlow/Keras, pandas, NumPy and scikit-learn (see `requirements.txt`). PLINK 1.9 and PLINK 2 must be on your `PATH`.

## Data

Point abyss at your data folder with an environment variable:

```bash
export ABYSS_DATA=/path/to/your/data/   # contains genotype/, phenotype/, covariate/, usefull_values/
```

The pipeline was developed on the public **1000 Genomes** data. No genotype or phenotype data is included in this repository.

## Pipeline

Run the scripts in order (each reads the previous step's output):

| Step | Script | What it does |
|---|---|---|
| 0 | `00_PLINK_preprocessing.py` | QC and per-chromosome PLINK files |
| 1 | `01_AF_ids.py` | Allele frequencies per SNP |
| 2 | `02_bins.py` | Allele-frequency bins |
| 3 | `03_LD_block.py` | Split bins into LD blocks |
| 4 | `04_latent.py` | Autoencoder per block → latent population representation |
| 5 | `05_concat.py` | Concatenate latent blocks |
| 6–7 | `06_lowest_dim.py`, `07_best_bottle.py` | Pick the lowest useful dimension / best bottleneck per bin |
| 8 | `08_AF_probability.py` | Estimate MAF probabilities from the latent factors |
| 9–10 | `09_GWAS.py`, `10_PRS.py` | Run GWAS and PRS with the abyss covariates |
| 11 | `11_Keep_SNPs_from_PRS.py` | Keep the SNPs used by the PRS |

The `99_*` scripts prepare phenotypes, mapping files and LD scores. The `notebooks/` folders walk through the same pipeline step by step, with outputs cleared.

## Status

Research code, shared as is. Author: Matthieu de Hemptinne.
