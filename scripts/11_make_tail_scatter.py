# scripts/11_make_tail_scatter.py

from __future__ import annotations

from pathlib import Path
import argparse
import numpy as np
import pandas as pd

from src.plotting.tail_scatter import make_jones_tail_scatter


def parse_args():
    parser = argparse.ArgumentParser(description="Make Jones tail scatter figures.")

    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--score_dir", type=str, default="results/scores")
    parser.add_argument("--out_dir", type=str, default="results/figures")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    score_dir = Path(args.score_dir)

    metadata_test = pd.read_csv(data_dir / "metadata_test.csv")
    signature = metadata_test["signature"].to_numpy().astype(int)

    nre_ae = np.load(score_dir / f"nre_Jones_AE_test_seed{args.seed}.npy")
    nre_pca = np.load(score_dir / f"nre_Jones_PCA_test_seed{args.seed}.npy")

    outputs = make_jones_tail_scatter(
        nre_ae=nre_ae,
        nre_pca=nre_pca,
        signature=signature,
        out_dir=args.out_dir,
        taus=(0.95, 0.99),
        sig_lines=(8, 10),
        prefix=f"jones_tail_scatter_seed{args.seed}",
    )

    print("Saved outputs:")
    for k, v in outputs.items():
        print(k, v)


if __name__ == "__main__":
    main()