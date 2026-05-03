# scripts/05_tail_enrichment.py

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from src.evaluation.tail_metrics import compute_tail_table, bootstrap_cis_test


INVARIANTS = ["Alexander", "Jones", "HOMFLY"]
METHODS = ["PCA", "AE"]


def parse_args():
    parser = argparse.ArgumentParser(description="Compute rare-event tail enrichment tables.")

    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--score_dir", type=str, default="results/scores")
    parser.add_argument("--out_dir", type=str, default="results/tables")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--targets", type=int, nargs="+", default=[8, 10])
    parser.add_argument("--taus", type=float, nargs="+", default=[0.95, 0.99])

    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--bootstrap_B", type=int, default=200)
    parser.add_argument("--bootstrap_seed", type=int, default=123)

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    score_dir = Path(args.score_dir)
    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_test = pd.read_csv(data_dir / "metadata_test.csv")
    y_test = metadata_test["signature"].to_numpy().astype(int)

    rows = []

    for inv in INVARIANTS:
        for method in METHODS:
            score_path = score_dir / f"nre_{inv}_{method}_test_seed{args.seed}.npy"

            if not score_path.exists():
                raise FileNotFoundError(f"Missing score file: {score_path}")

            scores = np.load(score_path)

            df = compute_tail_table(
                scores=scores,
                y_signature=y_test,
                s_list=args.targets,
                taus=args.taus,
            )

            df["Scope"] = "MAIN"
            df["Invariant"] = inv
            df["Score"] = f"NRE_{method}"
            df["Split"] = "test"

            rows.append(df)

    df_master = pd.concat(rows, ignore_index=True)

    df_master = df_master[
        [
            "Invariant",
            "Score",
            "Split",
            "Scope",
            "Target",
            "Tau",
            "AUROC",
            "AUPRC",
            "Enrichment",
            "Positives",
            "Tail_captured",
            "Tail_size",
            "N",
            "Base_rate",
        ]
    ].sort_values(["Invariant", "Score", "Target", "Tau"])

    out_path = out_dir / f"tail_metrics_main_seed{args.seed}.csv"
    df_master.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(df_master)

    if args.bootstrap:
        ci_rows = []

        for inv in INVARIANTS:
            for method in METHODS:
                score_path = score_dir / f"nre_{inv}_{method}_test_seed{args.seed}.npy"
                scores = np.load(score_path)

                df_ci = bootstrap_cis_test(
                    y_signature=y_test,
                    scores=scores,
                    targets=args.targets,
                    B=args.bootstrap_B,
                    seed=args.bootstrap_seed,
                )

                df_ci["Invariant"] = inv
                df_ci["Score"] = f"NRE_{method}"
                df_ci["Split"] = "test"

                ci_rows.append(df_ci)

        df_ci_all = pd.concat(ci_rows, ignore_index=True)
        ci_out = out_dir / f"bootstrap_ci_main_targets_seed{args.seed}.csv"
        df_ci_all.to_csv(ci_out, index=False)

        print("Saved:", ci_out)


if __name__ == "__main__":
    main()