# scripts/04_bulk_signature_decoding.py

from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd

from src.evaluation.bulk_decoding import (
    eval_multinomial_lr_split,
    majority_baseline,
    split_sanity_checks,
)
from src.features.summary_features import all_summary_features


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate bulk signature decoding.")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="results/tables")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load processed arrays.
    A_tr = np.load(data_dir / "X_A_train.npy")
    A_te = np.load(data_dir / "X_A_test.npy")

    J_tr = np.load(data_dir / "X_J_train.npy")
    J_te = np.load(data_dir / "X_J_test.npy")

    H_tr = np.load(data_dir / "X_H_train.npy")
    H_te = np.load(data_dir / "X_H_test.npy")

    metadata_train = pd.read_csv(data_dir / "metadata_train.csv")
    metadata_test = pd.read_csv(data_dir / "metadata_test.csv")

    y_train = metadata_train["signature"].to_numpy()
    y_test = metadata_test["signature"].to_numpy()

    # Sanity checks.
    sanity = {
        "train_size": int(len(y_train)),
        "test_size": int(len(y_test)),
        "train_signature_counts": metadata_train["signature"].value_counts().sort_index().to_dict(),
        "test_signature_counts": metadata_test["signature"].value_counts().sort_index().to_dict(),
        "majority_baseline": majority_baseline(y_train, y_test),
    }

    rows = []

    for name, Xtr, Xte in [
        ("Alexander_raw", A_tr, A_te),
        ("Jones_raw", J_tr, J_te),
        ("HOMFLY_raw", H_tr, H_te),
    ]:
        res = eval_multinomial_lr_split(
            Xtr,
            y_train,
            Xte,
            y_test,
            seed=args.seed,
        )
        rows.append(
            {
                "method": name,
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
                "n_iter": res["n_iter"],
            }
        )

    # Optional summary/confounder-only features.
    for name, Xtr_raw, Xte_raw in [
        ("Alexander_summary", A_tr, A_te),
        ("Jones_summary", J_tr, J_te),
        ("HOMFLY_summary", H_tr, H_te),
    ]:
        Xtr = all_summary_features(Xtr_raw)
        Xte = all_summary_features(Xte_raw)

        res = eval_multinomial_lr_split(
            Xtr,
            y_train,
            Xte,
            y_test,
            seed=args.seed,
        )
        rows.append(
            {
                "method": name,
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
                "n_iter": res["n_iter"],
            }
        )

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "bulk_signature_decoding.csv", index=False)

    with open(out_dir / "bulk_signature_decoding_sanity.json", "w") as f:
        json.dump(sanity, f, indent=2, default=str)

    print(results)
    print("Saved:", out_dir / "bulk_signature_decoding.csv")


if __name__ == "__main__":
    main()