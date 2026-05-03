# scripts/07_jones_ae_ablation.py

from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from src.evaluation.ablation import run_jones_ae_ablation


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Jones autoencoder ablation over latent dimensions, seeds, and score types."
    )

    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="results/tables")

    parser.add_argument("--latent_dims", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 999])
    parser.add_argument("--score_types", type=str, nargs="+", default=["NRE", "SSE"])

    parser.add_argument("--tau", type=float, default=0.99)
    parser.add_argument("--target_s", type=int, default=10)

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pred_batch_size", type=int, default=2048)

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X_train = np.load(data_dir / "X_J_train.npy").astype(np.float32)
    X_val = np.load(data_dir / "X_J_val.npy").astype(np.float32)
    X_test = np.load(data_dir / "X_J_test.npy").astype(np.float32)

    metadata_test = pd.read_csv(data_dir / "metadata_test.csv")
    y_test = metadata_test["signature"].to_numpy().astype(int)

    df = run_jones_ae_ablation(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_test=y_test,
        widths=[128, 64, 64],
        latent_dims=args.latent_dims,
        seeds=args.seeds,
        score_types=args.score_types,
        tau=args.tau,
        target_s=args.target_s,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        pred_batch_size=args.pred_batch_size,
    )

    out_path = out_dir / "jones_ae_ablation_stability.csv"
    df.to_csv(out_path, index=False)

    print("Saved:", out_path)
    print(df)


if __name__ == "__main__":
    main()