# scripts/09_confounder_analysis.py

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from src.evaluation.confounders import (
    support_width_from_coeffs,
    compute_spearman_confounders,
    build_confounder_matrix_scaled_space,
    evaluate_confounder_models_jones,
    enrichment_sensitivity_stable,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run confounder analyses.")

    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--score_dir", type=str, default="results/scores")
    parser.add_argument("--out_dir", type=str, default="results/tables")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--l0_tol", type=float, default=0.1)

    return parser.parse_args()


def load_arrays(data_dir: Path, key: str):
    return {
        "train": np.load(data_dir / f"X_{key}_train.npy").astype(np.float32),
        "val": np.load(data_dir / f"X_{key}_val.npy").astype(np.float32),
        "test": np.load(data_dir / f"X_{key}_test.npy").astype(np.float32),
    }


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    score_dir = Path(args.score_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    A = load_arrays(data_dir, "A")
    J = load_arrays(data_dir, "J")
    H = load_arrays(data_dir, "H")

    meta_train = pd.read_csv(data_dir / "metadata_train.csv")
    meta_test = pd.read_csv(data_dir / "metadata_test.csv")

    y_train = meta_train["signature"].to_numpy().astype(int)
    y_test = meta_test["signature"].to_numpy().astype(int)

    # Test metadata
    crossing_test = meta_test["number_of_crossings"].to_numpy()
    width_test = (
        meta_test["maximum_exponent"].to_numpy() - meta_test["minimum_exponent"].to_numpy()
        if {"maximum_exponent", "minimum_exponent"}.issubset(meta_test.columns)
        else np.zeros(len(meta_test))
    )

    # Train metadata
    crossing_train = meta_train["number_of_crossings"].to_numpy()
    width_train = (
        meta_train["maximum_exponent"].to_numpy() - meta_train["minimum_exponent"].to_numpy()
        if {"maximum_exponent", "minimum_exponent"}.issubset(meta_train.columns)
        else np.zeros(len(meta_train))
    )

    # AE test scores
    nre_A = np.load(score_dir / f"nre_Alexander_AE_test_seed{args.seed}.npy")
    nre_J = np.load(score_dir / f"nre_Jones_AE_test_seed{args.seed}.npy")
    nre_H = np.load(score_dir / f"nre_HOMFLY_AE_test_seed{args.seed}.npy")

    # HOMFLY support-width proxy
    width_H_test = support_width_from_coeffs(H["test"], eps=1e-8)

    # Spearman confounder correlations
    conf_A = compute_spearman_confounders(
        X=A["test"],
        nre=nre_A,
        crossing=crossing_test,
        support_width=width_test,
    )
    conf_A.insert(0, "Invariant", "Alexander")

    conf_J = compute_spearman_confounders(
        X=J["test"],
        nre=nre_J,
        crossing=crossing_test,
        support_width=width_test,
    )
    conf_J.insert(0, "Invariant", "Jones")

    conf_H = compute_spearman_confounders(
        X=H["test"],
        nre=nre_H,
        crossing=crossing_test,
        support_width=width_H_test,
    )
    conf_H.insert(0, "Invariant", "HOMFLY")

    conf_all = pd.concat([conf_A, conf_J, conf_H], ignore_index=True)
    conf_all["Domain"] = "test"
    conf_all["N_test"] = len(meta_test)

    conf_path = out_dir / "confounders_spearman_test.csv"
    conf_all.to_csv(conf_path, index=False)

    print("Saved:", conf_path)
    print(conf_all)

    # Jones confounder-only models
    C_train = build_confounder_matrix_scaled_space(
        X_scaled=J["train"],
        width=width_train,
        crossing=crossing_train,
        l0_tol=args.l0_tol,
    )

    C_test = build_confounder_matrix_scaled_space(
        X_scaled=J["test"],
        width=width_test,
        crossing=crossing_test,
        l0_tol=args.l0_tol,
    )

    df_conf_models = evaluate_confounder_models_jones(
        C_train=C_train,
        C_test=C_test,
        sig_train=y_train,
        sig_test=y_test,
        targets=(8, 10),
        taus=(0.95, 0.99),
        seed=args.seed,
    )

    conf_model_path = out_dir / "confounder_models_jones_test_consistent.csv"
    df_conf_models.to_csv(conf_model_path, index=False)

    print("Saved:", conf_model_path)
    print(df_conf_models)

    # Tail enrichment conditioned by support-width bins
    def make_df(X, nre, support_width, support_width_type):
        df = pd.DataFrame(
            {
                "signature": y_test,
                "number_of_crossings": crossing_test,
                "support_width": support_width,
                "support_width_type": support_width_type,
                "NRE_AE": nre,
            }
        )
        return df

    dfA = make_df(A["test"], nre_A, width_test, "exponent_width")
    dfJ = make_df(J["test"], nre_J, width_test, "exponent_width")
    dfH = make_df(H["test"], nre_H, width_H_test, "coef_support_proxy")

    enrich_A = enrichment_sensitivity_stable(dfA)
    enrich_A.insert(0, "Invariant", "Alexander")
    enrich_A["support_width_type"] = "exponent_width"

    enrich_J = enrichment_sensitivity_stable(dfJ)
    enrich_J.insert(0, "Invariant", "Jones")
    enrich_J["support_width_type"] = "exponent_width"

    enrich_H = enrichment_sensitivity_stable(dfH)
    enrich_H.insert(0, "Invariant", "HOMFLY")
    enrich_H["support_width_type"] = "coef_support_proxy"

    enrich_all = pd.concat([enrich_A, enrich_J, enrich_H], ignore_index=True)
    enrich_all["Domain"] = "test"

    enrich_path = out_dir / "tail_enrichment_conditioned_test_smoothed.csv"
    enrich_all.to_csv(enrich_path, index=False)

    print("Saved:", enrich_path)
    print(enrich_all.head(30))


if __name__ == "__main__":
    main()