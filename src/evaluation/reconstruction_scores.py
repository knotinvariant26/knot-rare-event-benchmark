# src/evaluation/reconstruction_scores.py

from __future__ import annotations

import numpy as np


def nre_from_recon(X, X_hat, eps: float = 1e-8):
    """
    Normalized reconstruction error:

        NRE(x) = ||x - x_hat||_2^2 / (||x||_2^2 + eps)
    """
    X64 = np.asarray(X, dtype=np.float64)
    Xh64 = np.asarray(X_hat, dtype=np.float64)

    numerator = np.sum((X64 - Xh64) ** 2, axis=1)
    denominator = np.sum(X64 ** 2, axis=1) + eps

    return (numerator / denominator).astype(np.float64)


def nre_from_model(model, X, batch_size: int = 2048, eps: float = 1e-8):
    """
    Compute NRE using an autoencoder model.
    """
    X_hat = model.predict(X, batch_size=batch_size, verbose=0)
    return nre_from_recon(X, X_hat, eps=eps)


def nre_from_pca(pca, X, eps: float = 1e-8):
    """
    Compute NRE using PCA reconstruction.
    """
    Z = pca.transform(X)
    X_hat = pca.inverse_transform(Z)
    return nre_from_recon(X, X_hat, eps=eps)