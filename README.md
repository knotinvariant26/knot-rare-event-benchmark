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
