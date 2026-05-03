# scripts/10_tail_overlap_jones.py

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from src.evaluation.confounders import (
    build_confounder_matrix_scaled_space,
    tail_overlap_against_confounders,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Jones AE tail with confounder-only tail.")

    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--score_dir", type=str, default="results/scores")
    parser.add_argument("--out_dir", type=str, default="results/tables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l0_tol", type=float, default=0.1)

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    score_dir = Path(args.score_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    J_train = np.load(data_dir / "X_J_train.npy").astype(np.float32)
    J_test = np.load(data_dir / "X_J_test.npy").astype(np.float32)

    meta_train = pd.read_csv(data_dir / "metadata_train.csv")
    meta_test = pd.read_csv(data_dir / "metadata_test.csv")

    sig_train = meta_train["signature"].to_numpy().astype(int)
    sig_test = meta_test["signature"].to_numpy().astype(int)

    width_train = meta_train["maximum_exponent"].to_numpy() - meta_train["minimum_exponent"].to_numpy()
    width_test = meta_test["maximum_exponent"].to_numpy() - meta_test["minimum_exponent"].to_numpy()

    crossing_train = meta_train["number_of_crossings"].to_numpy()
    crossing_test = meta_test["number_of_crossings"].to_numpy()

    C_train = build_confounder_matrix_scaled_space(
        X_scaled=J_train,
        width=width_train,
        crossing=crossing_train,
        l0_tol=args.l0_tol,
    )

    C_test = build_confounder_matrix_scaled_space(
        X_scaled=J_test,
        width=width_test,
        crossing=crossing_test,
        l0_tol=args.l0_tol,
    )

    ae_scores = np.load(score_dir / f"nre_Jones_AE_test_seed{args.seed}.npy")

    df_overlap, df_ae_only = tail_overlap_against_confounders(
        ae_scores=ae_scores,
        C_train=C_train,
        C_test=C_test,
        sig_train=sig_train,
        sig_test=sig_test,
        targets=(8, 10),
        taus=(0.95, 0.99),
        seed=args.seed,
    )

    overlap_path = out_dir / "tail_overlap_jones_test.csv"
    df_overlap.to_csv(overlap_path, index=False)

    print("Saved:", overlap_path)
    print(df_overlap)

    if len(df_ae_only) > 0:
        ae_only_path = out_dir / "ae_only_tail_positives_jones.csv"
        df_ae_only.to_csv(ae_only_path, index=False)

        print("Saved:", ae_only_path)
        print(df_ae_only.head(20))

    print("\n=== Compact: Y10 @ tau=0.99 ===")
    compact = df_overlap[
        (df_overlap["Target"] == "Y10") & (np.isclose(df_overlap["Tau"], 0.99))
    ][
        [
            "Jaccard",
            "AE_tail_covered_by_Conf",
            "Conf_tail_covered_by_AE",
            "AE_Enrichment",
            "Conf_Enrichment",
            "AE_pos_in_tail",
            "Conf_pos_in_tail",
            "AE_only_tail_size",
            "AE_only_pos_in_tail",
        ]
    ]
    print(compact)


if __name__ == "__main__":
    main()