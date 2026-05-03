# src/evaluation/ablation.py

from __future__ import annotations

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from src.models.autoencoder import train_autoencoder
from src.evaluation.reconstruction_scores import nre_from_model
from src.evaluation.tail_metrics import enrichment_at_tau_with_counts


def sse_from_model(model, X, batch_size: int = 2048):
    """
    Raw squared reconstruction error:
        SSE(x) = ||x - f(x)||_2^2
    """
    X = np.asarray(X, dtype=np.float32)
    X_hat = model.predict(X, batch_size=batch_size, verbose=0).astype(np.float32)

    return np.sum(
        (X.astype(np.float64) - X_hat.astype(np.float64)) ** 2,
        axis=1,
    )


def score_autoencoder(
    model,
    X,
    score_type: str = "NRE",
    batch_size: int = 2048,
    eps: float = 1e-8,
):
    """
    Compute either normalized reconstruction error (NRE)
    or raw squared reconstruction error (SSE).
    """
    score_type = score_type.upper()

    if score_type == "NRE":
        return nre_from_model(
            model,
            X,
            batch_size=batch_size,
            eps=eps,
        )

    if score_type == "SSE":
        return sse_from_model(
            model,
            X,
            batch_size=batch_size,
        )

    raise ValueError(f"Unknown score_type: {score_type}")


def run_jones_ae_ablation(
    X_train,
    X_val,
    X_test,
    y_test,
    widths: list[int],
    latent_dims: list[int] = [8, 16, 32],
    seeds: list[int] = [42, 123, 999],
    score_types: list[str] = ["NRE", "SSE"],
    tau: float = 0.99,
    target_s: int = 10,
    learning_rate: float = 5e-5,
    batch_size: int = 128,
    epochs: int = 30,
    patience: int = 5,
    pred_batch_size: int = 2048,
):
    """
    Jones-only AE ablation over latent dimension, seed, and reconstruction score.

    Returns
    -------
    pd.DataFrame
        One row per (latent_dim, seed, score_type).
    """
    rows = []

    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test).astype(int)

    ybin = (np.abs(y_test) >= target_s).astype(int)

    for latent_dim in latent_dims:
        for seed in seeds:
            tf.keras.utils.set_random_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

            ae, encoder, history = train_autoencoder(
                X_train,
                X_val,
                latent_size=latent_dim,
                widths=widths,
                learning_rate=learning_rate,
                batch_size=batch_size,
                epochs=epochs,
                patience=patience,
                seed=seed,
                verbose=0,
            )

            best_val_loss = float(np.min(history.history["val_loss"]))
            epochs_ran = int(len(history.history["loss"]))

            for score_type in score_types:
                scores = score_autoencoder(
                    ae,
                    X_test,
                    score_type=score_type,
                    batch_size=pred_batch_size,
                )

                enrichment, n_pos, n_pos_tail, tail_size, base_rate = (
                    enrichment_at_tau_with_counts(
                        scores=scores,
                        y_true=ybin,
                        tau=tau,
                    )
                )

                rows.append(
                    {
                        "invariant": "Jones",
                        "latent_dim": int(latent_dim),
                        "seed": int(seed),
                        "score": score_type.upper(),
                        "tau": float(tau),
                        "target": f"Y{target_s}",
                        "enrichment": float(enrichment),
                        "tail_captured": int(n_pos_tail),
                        "tail_size": int(tail_size),
                        "positives": int(n_pos),
                        "base_rate": float(base_rate),
                        "epochs_ran": epochs_ran,
                        "best_val_loss": best_val_loss,
                    }
                )

            print(f"[Jones | latent_dim={latent_dim} | seed={seed}] done.")

    return pd.DataFrame(rows)