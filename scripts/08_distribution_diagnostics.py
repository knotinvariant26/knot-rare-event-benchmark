# scripts/08_distribution_diagnostics.py

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np

from src.evaluation.distribution_diagnostics import (
    run_distribution_diagnostics,
    plot_ccdf_row,
)


INVARIANTS = ["Alexander", "Jones", "HOMFLY"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run distributional diagnostics for AE normalized reconstruction errors."
    )

    parser.add_argument("--score_dir", type=str, default="results/scores")
    parser.add_argument("--table_dir", type=str, default="results/tables")
    parser.add_argument("--figure_dir", type=str, default="results/figures")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--method", type=str, default="AE", choices=["AE", "PCA"])
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--sample_max", type=int, default=None)
    parser.add_argument("--n_boot", type=int, default=500)

    return parser.parse_args()


def main():
    args = parse_args()

    score_dir = Path(args.score_dir)
    table_dir = Path(args.table_dir)
    figure_dir = Path(args.figure_dir)

    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    score_dict = {}

    for invariant in INVARIANTS:
        score_path = (
            score_dir
            / f"nre_{invariant}_{args.method}_{args.split}_seed{args.seed}.npy"
        )

        if not score_path.exists():
            raise FileNotFoundError(f"Missing score file: {score_path}")

        score_dict[invariant] = np.load(score_path)

    df, fits = run_distribution_diagnostics(
        score_dict=score_dict,
        sample_max=args.sample_max,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    out_csv = table_dir / f"distribution_diagnostics_{args.method}_{args.split}_seed{args.seed}.csv"
    df.to_csv(out_csv, index=False)

    out_pdf = figure_dir / f"ccdf_all_invariants_{args.method}_{args.split}_seed{args.seed}.pdf"
    out_png = figure_dir / f"ccdf_all_invariants_{args.method}_{args.split}_seed{args.seed}.png"

    plot_ccdf_row(
        fits=fits,
        out_pdf=out_pdf,
        out_png=out_png,
    )

    print("Saved:", out_csv)
    print("Saved:", out_pdf)
    print("Saved:", out_png)
    print(df)


if __name__ == "__main__":
    main()