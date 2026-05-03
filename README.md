# Knot Polynomial Spaces Benchmark

This repository contains code and data assets for reproducing the bulk-versus-tail evaluation protocol for rare-event evaluation in knot polynomial coefficient spaces.

## Contents

- Fixed train/validation/test splits
- Preprocessing scripts for Alexander, Jones, and HOMFLY--PT coefficient vectors
- PCA and autoencoder reconstruction baselines
- Bulk signature decoding scripts
- Rare-event tail enrichment evaluation
- Confounder controls
- Contextual outlier baselines
- Scripts to reproduce tables and figures


# Data

The raw knot polynomial data used in this benchmark come from the public Zenodo record:

- Title: dioscuri-tda/knotsBM: 0.3
- DOI: 10.5281/zenodo.10876347
- Source: Zenodo
- License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- Main files: `data.zip` and `dioscuri-tda/knotsBM-0.3.zip`

The original data are publicly available and should be cited according to the Zenodo record and the associated dataset/software papers.

This repository provides fixed train/validation/test splits, preprocessing scripts, and evaluation scripts for the benchmark. If processed benchmark files are included, they are derived from the public Zenodo data and preserve attribution to the original creators.


## Reproducing main results

```bash
conda env create -f environment.yml
conda activate knot-benchmark

python scripts/01_prepare_data.py
python scripts/02_train_autoencoders.py --invariant jones --latent_dim 16 --seed 42
python scripts/03_run_pca.py --invariant jones --latent_dim 16
python scripts/04_bulk_signature_decoding.py
python scripts/05_tail_enrichment.py
python scripts/08_make_tables_figures.py
