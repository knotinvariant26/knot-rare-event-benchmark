# Knot Polynomial Spaces as a Benchmark for Rare-Event Evaluation in AI for Math

This repository contains code and reproducibility assets for the paper:

**Knot Polynomial Spaces as a Benchmark for Rare-Event Evaluation in AI for Math**

The benchmark studies rare-event evaluation in polynomial coefficient spaces for prime knots. We use coefficient-vector encodings of the Alexander, Jones, and HOMFLY--PT polynomials and evaluate unsupervised reconstruction-based scores against the knot signature, which is held out as an external probe.

The goal is not to introduce a new representation learning architecture. Instead, the repository provides a reproducible benchmark protocol for separating:

1. **Bulk accessibility**: how much signature information is accessible from raw coefficients, PCA embeddings, or autoencoder latents.
2. **Tail enrichment**: whether reconstruction residuals enrich for rare large-signature knots in fixed-mass top-k tails.

---

## Repository structure

```text
knot-rare-event-benchmark/
├── README.md
├── requirements.txt
├── environment.yml
├── scripts/
│   ├── 01_prepare_data.py
│   ├── 02_train_reconstruction_models.py
│   ├── 04_bulk_signature_decoding.py
│   ├── 05_tail_enrichment.py
│   ├── 06_make_paper_tables.py
│   ├── 07_jones_ae_ablation.py
│   ├── 08_distribution_diagnostics.py
│   ├── 09_confounder_analysis.py
│   ├── 10_tail_overlap_jones.py
│   ├── 11_make_tail_scatter.py
│   └── 12_outlier_baselines.py
├── src/
│   ├── data/
│   ├── evaluation/
│   ├── features/
│   ├── models/
│   └── plotting/
└── results/
    ├── tables/
    ├── figures/
    ├── scores/
    └── splits/
```
## Data

The raw knot polynomial data are obtained from the public Zenodo release:

> Gurnari, D. and Dłotko, P. (2024). `dioscuri-tda/knotsBM: 0.3`. Zenodo. DOI: `10.5281/zenodo.10876347`.

The Zenodo release is licensed under **Creative Commons Attribution 4.0 International**.

This repository provides preprocessing, alignment, split generation, training, and evaluation scripts for the benchmark protocol. Where redistribution of processed files is permitted, processed benchmark assets may be included; otherwise, they can be regenerated from the public raw data.

Expected raw files:

```text
Alexander_upto_17.csv
Jones_upto_15_MIRRORS.csv
HomflyPt_upto_15_MIRRORS.csv

Benchmark construction

The preprocessing pipeline aligns Alexander, Jones, and HOMFLY--PT coefficient tables by:

(number_of_crossings, is_alternating, table_number)

after removing mirror entries marked with !.

The final aligned benchmark contains:

307,110 aligned prime-knot records
Alexander dimension: 17
Jones dimension: 51
HOMFLY--PT dimension: 152

For stratified train/validation/test splitting, the singleton class sigma = 14 is removed, yielding:

307,109 records used for the fixed split protocol

The split indices are shared across all polynomial invariants.

Installation
Option 1: Conda
conda env create -f environment.yml
conda activate knot-benchmark
Option 2: pip
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

Tested with Python 3.10+.

Reproducing the main results

The main paper results are generated in stages.

1. Prepare aligned data and fixed splits
python scripts/01_prepare_data.py

Expected outputs:

results/splits/
results/tables/preprocessing_summary.csv

This step performs alignment, mirror filtering, signature consistency checks, train/validation/test split generation, and train-only standardization metadata.

2. Train reconstruction models and compute scores
python scripts/02_train_reconstruction_models.py

Expected outputs:

results/scores/
results/models/

This step trains PCA and autoencoder reconstruction models for Alexander, Jones, and HOMFLY--PT. Autoencoders are trained without signature labels.

3. Bulk signature decoding
python scripts/04_bulk_signature_decoding.py

Expected outputs:

results/tables/bulk_signature_decoding.csv

This reproduces the bulk accessibility results reported in Table 1 of the paper. The probe is a class-balanced multinomial logistic regression trained on raw coefficients, PCA embeddings, and autoencoder latents.

The knot signature is used only for this post hoc probe and is not used to train PCA or autoencoders.

4. Rare-event tail enrichment
python scripts/05_tail_enrichment.py

Expected outputs:

results/tables/tail_enrichment_main.csv
results/tables/jones_Y12_stress.csv

This reproduces the fixed-mass top-k rare-event enrichment results.

For a test set of size n and tail level tau, the tail size is:

k = ceil((1 - tau) * n)

For the held-out test set used in the paper:

tau = 0.95 -> k = 1536
tau = 0.99 -> k = 308

Rare-event targets are:

Y8  = 1[|sigma(K)| >= 8]
Y10 = 1[|sigma(K)| >= 10]
Y12 = 1[|sigma(K)| >= 12]  # qualitative stress test only
5. Generate paper tables
python scripts/06_make_paper_tables.py

Expected outputs:

results/tables/table1_bulk_decoding.tex
results/tables/table2_jones_tail_enrichment.tex
results/tables/table_y12_stress.tex
Additional analyses

The following scripts reproduce appendix diagnostics and contextual comparisons.

Jones autoencoder ablations
python scripts/07_jones_ae_ablation.py

Outputs:

results/tables/jones_ae_ablation_stability.csv

This varies latent dimension, random seed, and score type:

latent dimension: 8, 16, 32
random seeds: 42, 123, 999
score type: NRE, SSE
Distributional diagnostics
python scripts/08_distribution_diagnostics.py

Outputs:

results/tables/powerlaw_master_results.csv
results/figures/ccdf_all_invariants.png
results/figures/ccdf_all_invariants.pdf

These diagnostics are descriptive only. The paper does not claim exact power-law behavior.

Confounder analysis
python scripts/09_confounder_analysis.py

Outputs:

results/tables/confounder_spearman.csv
results/tables/confounder_models_jones.csv

This evaluates whether coefficient-level statistics such as norm, sparsity proxy, support width, and crossing number explain rare-event enrichment.

Tail-overlap analysis
python scripts/10_tail_overlap_jones.py

Outputs:

results/tables/tail_overlap_jones.csv
results/tables/ae_only_positive_examples.csv

This compares the fixed-mass AE reconstruction tail with the tail induced by a supervised confounder-only model.

Tail scatter figure
python scripts/11_make_tail_scatter.py

Outputs:

results/figures/jones_tail_scatter_full.png
results/figures/jones_tail_scatter_full.pdf
results/figures/jones_tail_scatter_tail_zoom.png
results/figures/jones_tail_scatter_tail_zoom.pdf

This generates the main scatter visualization for Jones AE-NRE against |sigma(K)|, colored by PCA-NRE.

Contextual ambient outlier baselines
python scripts/12_outlier_baselines.py

Outputs:

results/tables/outlier_baselines_test.csv
results/tables/outlier_baselines_test_table.tex

These baselines include Isolation Forest, Local Outlier Factor, One-Class SVM with RBF kernel, and RFF + PCA residual. They are included as contextual comparisons only and are not used as the main basis for the reconstruction-based claims.

Fixed-mass top-k tail convention

All enrichment and captured-count results use exact fixed-mass top-k tails.

Given scores s, test-set size n, and tail level tau:

k = ceil((1 - tau) * n)
tail = top k samples ranked by score

Ties are broken deterministically by stable sorting. This ensures that all methods are compared using identical tail sizes.

For the paper test split:

N_test = 30,711
tau = 0.95 -> tail size = 1,536
tau = 0.99 -> tail size = 308
Main expected results

Approximate values from the paper:

Bulk signature accessibility
Representation	Jones Acc.	Jones Macro-F1	Alexander Acc.	Alexander Macro-F1	HOMFLY--PT Acc.	HOMFLY--PT Macro-F1
Raw coefficients	0.873	0.721	0.838	0.722	0.960	0.913
PCA, d=16	0.856	0.678	0.838	0.723	0.399	0.503
AE latent, d=16	0.866	0.692	0.706	0.649	0.604	0.629
Jones rare-event tail enrichment for Y10
Score	Enr.@0.95	Enr.@0.99	Recall@0.99	Captured@0.99
PCA-NRE	3.49x	9.50x	9.5%	6/308
AE-NRE	4.13x	7.91x	7.9%	5/308

Small numerical differences may occur across software versions, but the fixed split and stored score files should reproduce the reported tables.

Compute requirements

The benchmark is designed to be reproducible on a standard research workstation or cloud notebook.

Recommended hardware
CPU: 4+ cores recommended.
RAM: 16 GB minimum; 32 GB recommended for the full HOMFLY--PT and outlier-baseline experiments.
GPU: optional but recommended for autoencoder training. A single NVIDIA T4, A10, V100, or similar GPU is sufficient.
Storage: approximately 2--5 GB for raw CSV files, processed arrays, intermediate scores, and result tables.
Approximate runtimes

Runtimes vary by hardware.

Step	Script	Approx. runtime	GPU needed?
Data preparation and alignment	scripts/01_prepare_data.py	minutes	No
Reconstruction models	scripts/02_train_reconstruction_models.py	tens of minutes to a few hours	Recommended
Bulk signature decoding	scripts/04_bulk_signature_decoding.py	minutes	No
Tail enrichment tables	scripts/05_tail_enrichment.py	minutes	No
Paper table generation	scripts/06_make_paper_tables.py	minutes	No
Jones AE ablation	scripts/07_jones_ae_ablation.py	several hours depending on GPU	Recommended
Distribution diagnostics	scripts/08_distribution_diagnostics.py	minutes to tens of minutes	No
Confounder analysis	scripts/09_confounder_analysis.py	minutes	No
Tail-overlap analysis	scripts/10_tail_overlap_jones.py	minutes	No
Tail scatter figure	scripts/11_make_tail_scatter.py	minutes	No
Ambient outlier baselines	scripts/12_outlier_baselines.py	tens of minutes to several hours	No, but CPU/RAM intensive

The main benchmark results in Tables 1--2 can be reproduced without running the contextual outlier baselines.

Reproducibility notes
All standardization is fit on the training split only.
Signature labels are never used to train PCA, autoencoders, reconstruction scores, or tail membership.
The same train/validation/test split indices are used across Alexander, Jones, and HOMFLY--PT.
Tail metrics use exact fixed-mass top-k tails, not threshold-based quantile inclusion.
The extreme target Y12 is reported only as a qualitative stress test because it has only two positives in the held-out test split.
License

The code in this repository is released under the license specified in LICENSE.

The raw knot polynomial data are obtained from the public knotsBM Zenodo release and are governed by the original dataset license:

Creative Commons Attribution 4.0 International

Please cite the original dataset release when using the raw or processed knot polynomial data.

Citation

Anonymous citation placeholder for review:

@misc{knot_rare_event_benchmark_2026,
  title = {Knot Polynomial Spaces as a Benchmark for Rare-Event Evaluation in AI for Math},
  author = {Anonymous},
  year = {2026},
  note = {Submitted for review}
}

Dataset citation:

@dataset{gurnari_dlotko_2024_knotsbm,
  author = {Gurnari, Davide and D{\l}otko, Pawe{\l}},
  title = {dioscuri-tda/knotsBM: 0.3},
  year = {2024},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.10876347}
}
Anonymity note

This repository is prepared for anonymous peer review. It avoids author-identifying information and uses neutral project naming.

