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
