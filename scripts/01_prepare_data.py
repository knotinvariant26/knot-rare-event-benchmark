# scripts/01_prepare_data.py

from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from src.data.preprocess import preprocess_align
from src.data.splits import prepare_splits_and_scalers


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare aligned knot polynomial benchmark data.")

    parser.add_argument(
        "--alexander",
        type=str,
        default="data/raw/Alexander_upto_17.csv",
        help="Path to Alexander CSV file.",
    )
    parser.add_argument(
        "--jones",
        type=str,
        default="data/raw/Jones_upto_15_MIRRORS.csv",
        help="Path to Jones CSV file.",
    )
    parser.add_argument(
        "--homfly",
        type=str,
        default="data/raw/HomflyPt_upto_15_MIRRORS.csv",
        help="Path to HOMFLY--PT CSV file.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/processed",
        help="Output directory for processed benchmark files.",
    )
    parser.add_argument(
        "--max_cross",
        type=int,
        default=15,
        help="Maximum crossing number to include.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified split.",
    )
    parser.add_argument(
        "--drop_mirrors",
        action="store_true",
        help="Drop mirror entries marked with '!'.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading raw CSV files...")
    A = pd.read_csv(args.alexander)
    J = pd.read_csv(args.jones)
    H = pd.read_csv(args.homfly)

    print("Aligning polynomial tables...")
    aligned = preprocess_align(
        A,
        J,
        H,
        max_cross=args.max_cross,
        drop_mirrors=args.drop_mirrors,
        drop_signature_mismatches=True,
        verbose=True,
    )

    print("Creating splits and train-only scalers...")
    prepared = prepare_splits_and_scalers(
        aligned["X_A"],
        aligned["X_J"],
        aligned["X_H"],
        aligned["metadata"],
        seed=args.seed,
        test_size=0.10,
        val_size=0.10,
    )

    # Save arrays.
    for inv in ["A", "J", "H"]:
        for split in ["train", "val", "test"]:
            np.save(out_dir / f"X_{inv}_{split}.npy", prepared[inv][split])

    # Save metadata.
    prepared["metadata"]["all"].to_csv(out_dir / "metadata_all.csv", index=False)
    for split in ["train", "val", "test"]:
        prepared["metadata"][split].to_csv(out_dir / f"metadata_{split}.csv", index=False)

    # Save split indices.
    for split, idx in prepared["indices"].items():
        pd.DataFrame({"idx": idx}).to_csv(out_dir / f"{split}_idx.csv", index=False)

    # Save scalers.
    joblib.dump(prepared["A"]["scaler"], out_dir / "scaler_A.joblib")
    joblib.dump(prepared["J"]["scaler"], out_dir / "scaler_J.joblib")
    joblib.dump(prepared["H"]["scaler"], out_dir / "scaler_H.joblib")

    # Save debug information.
    debug = {
        "args": vars(args),
        "preprocess_debug": aligned["debug"],
        "dropped_signature_classes": prepared["dropped_signature_classes"],
        "final_n_after_signature_class_filter": int(len(prepared["metadata"]["all"])),
        "split_sizes": {
            split: int(len(idx)) for split, idx in prepared["indices"].items()
        },
    }

    with open(out_dir / "preprocess_debug.json", "w") as f:
        json.dump(debug, f, indent=2, default=str)

    print("\nDone.")
    print(f"Processed files saved to: {out_dir}")
    print("Final N after class filter:", len(prepared["metadata"]["all"]))
    print("Split sizes:", debug["split_sizes"])


if __name__ == "__main__":
    main()