# scripts/02_train_reconstruction_models.py

from __future__ import annotations

from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.models.autoencoder import train_autoencoder, encode_with_model, set_global_seed
from src.evaluation.reconstruction_scores import nre_from_pca, nre_from_model
from src.evaluation.bulk_decoding import eval_multinomial_lr_split


ARCH = {
    "Alexander": {"key": "A", "latent": 16, "widths": [64, 32, 32]},
    "Jones": {"key": "J", "latent": 16, "widths": [128, 64, 64]},
    "HOMFLY": {"key": "H", "latent": 16, "widths": [256, 128, 128]},
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PCA and autoencoder reconstruction models and save NRE scores."
    )

    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--out_dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--pca_dim", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--pred_batch_size", type=int, default=2048)

    parser.add_argument("--save_models", action="store_true")

    return parser.parse_args()


def load_split_arrays(data_dir: Path, key: str):
    X_train = np.load(data_dir / f"X_{key}_train.npy").astype(np.float32)
    X_val = np.load(data_dir / f"X_{key}_val.npy").astype(np.float32)
    X_test = np.load(data_dir / f"X_{key}_test.npy").astype(np.float32)

    return X_train, X_val, X_test


def main():
    args = parse_args()
    set_global_seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)

    score_dir = out_dir / "scores"
    latent_dir = out_dir / "latents"
    table_dir = out_dir / "tables"
    model_dir = out_dir / "models"

    score_dir.mkdir(parents=True, exist_ok=True)
    latent_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    if args.save_models:
        model_dir.mkdir(parents=True, exist_ok=True)

    metadata_train = pd.read_csv(data_dir / "metadata_train.csv")
    metadata_test = pd.read_csv(data_dir / "metadata_test.csv")

    y_train = metadata_train["signature"].to_numpy().astype(int)
    y_test = metadata_test["signature"].to_numpy().astype(int)

    ae_latent_probe_rows = []
    run_debug = {
        "seed": args.seed,
        "pca_dim": args.pca_dim,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "pred_batch_size": args.pred_batch_size,
        "invariants": {},
    }

    for inv, cfg in ARCH.items():
        key = cfg["key"]

        print(f"\n=== {inv} ===")
        X_train, X_val, X_test = load_split_arrays(data_dir, key)

        run_debug["invariants"][inv] = {
            "input_dim": int(X_train.shape[1]),
            "train_n": int(X_train.shape[0]),
            "val_n": int(X_val.shape[0]),
            "test_n": int(X_test.shape[0]),
            "latent_dim": int(cfg["latent"]),
            "widths": cfg["widths"],
        }

        # PCA fit on train only.
        print("Fitting PCA...")
        pca = PCA(n_components=args.pca_dim, random_state=args.seed)
        pca.fit(X_train)

        nre_pca_test = nre_from_pca(pca, X_test)
        np.save(score_dir / f"nre_{inv}_PCA_test_seed{args.seed}.npy", nre_pca_test)

        # PCA embeddings for future Table 1 if needed.
        Z_pca_train = pca.transform(X_train).astype(np.float32)
        Z_pca_test = pca.transform(X_test).astype(np.float32)

        np.save(latent_dir / f"z_{inv}_PCA_train_seed{args.seed}.npy", Z_pca_train)
        np.save(latent_dir / f"z_{inv}_PCA_test_seed{args.seed}.npy", Z_pca_test)

        pca_probe = eval_multinomial_lr_split(
            Z_pca_train,
            y_train,
            Z_pca_test,
            y_test,
            seed=args.seed,
        )

        ae_latent_probe_rows.append(
            {
                "Invariant": inv,
                "Representation": f"PCA_d{args.pca_dim}",
                "Accuracy": pca_probe["accuracy"],
                "MacroF1": pca_probe["macro_f1"],
                "n_iter": pca_probe["n_iter"],
            }
        )

        # AE training.
        print("Training autoencoder...")
        t0 = time.time()

        ae, encoder, history = train_autoencoder(
            X_train,
            X_val,
            latent_size=cfg["latent"],
            widths=cfg["widths"],
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
            verbose=0,
        )

        train_time = time.time() - t0
        print(f"AE trained in {train_time:.1f}s")

        run_debug["invariants"][inv]["ae_train_time_seconds"] = float(train_time)
        run_debug["invariants"][inv]["ae_epochs_ran"] = int(len(history.history["loss"]))
        run_debug["invariants"][inv]["best_val_loss"] = float(np.min(history.history["val_loss"]))

        # AE NRE on test.
        print("Computing AE NRE...")
        nre_ae_test = nre_from_model(
            ae,
            X_test,
            batch_size=args.pred_batch_size,
        )
        np.save(score_dir / f"nre_{inv}_AE_test_seed{args.seed}.npy", nre_ae_test)

        # AE latent for Table 1.
        print("Encoding AE latents...")
        Z_ae_train = encode_with_model(
            encoder,
            X_train,
            batch_size=args.pred_batch_size,
        ).astype(np.float32)

        Z_ae_test = encode_with_model(
            encoder,
            X_test,
            batch_size=args.pred_batch_size,
        ).astype(np.float32)

        np.save(latent_dir / f"z_{inv}_AE_train_seed{args.seed}.npy", Z_ae_train)
        np.save(latent_dir / f"z_{inv}_AE_test_seed{args.seed}.npy", Z_ae_test)

        ae_probe = eval_multinomial_lr_split(
            Z_ae_train,
            y_train,
            Z_ae_test,
            y_test,
            seed=args.seed,
        )

        ae_latent_probe_rows.append(
            {
                "Invariant": inv,
                "Representation": f"AE_d{cfg['latent']}",
                "Accuracy": ae_probe["accuracy"],
                "MacroF1": ae_probe["macro_f1"],
                "n_iter": ae_probe["n_iter"],
            }
        )

        if args.save_models:
            ae.save(model_dir / f"ae_{inv}_seed{args.seed}.keras")
            encoder.save(model_dir / f"encoder_{inv}_seed{args.seed}.keras")

    # Save Table 1 helper rows for PCA and AE latent.
    latent_probe_df = pd.DataFrame(ae_latent_probe_rows)
    latent_probe_df.to_csv(table_dir / f"latent_probe_results_seed{args.seed}.csv", index=False)

    with open(out_dir / f"reconstruction_run_debug_seed{args.seed}.json", "w") as f:
        json.dump(run_debug, f, indent=2)

    print("\nDone.")
    print("Saved scores to:", score_dir)
    print("Saved latents to:", latent_dir)
    print("Saved latent probe table to:", table_dir / f"latent_probe_results_seed{args.seed}.csv")


if __name__ == "__main__":
    main()