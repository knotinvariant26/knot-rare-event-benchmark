# src/features/summary_features.py

import numpy as np


def coeff_norm_features(X):
    X = np.asarray(X)
    l2 = np.linalg.norm(X, axis=1)
    l1 = np.sum(np.abs(X), axis=1)
    linf = np.max(np.abs(X), axis=1)
    return np.vstack([l2, l1, linf]).T


def sparsity_features(X, eps=0.0):
    X = np.asarray(X)
    nnz = np.sum(np.abs(X) > eps, axis=1)
    frac_nnz = nnz / X.shape[1]
    return np.vstack([nnz, frac_nnz]).T


def simple_stats_features(X):
    X = np.asarray(X)
    mean = X.mean(axis=1)
    std = X.std(axis=1)
    mx = X.max(axis=1)
    mn = X.min(axis=1)
    return np.vstack([mean, std, mx, mn]).T


def center_of_mass_features(X, eps=1e-12):
    X = np.asarray(X)
    w = np.abs(X)
    idx = np.arange(X.shape[1])[None, :]
    denom = w.sum(axis=1) + eps
    com = (w * idx).sum(axis=1) / denom
    spread = np.sqrt(((idx - com[:, None]) ** 2 * w).sum(axis=1) / denom)
    return np.vstack([com, spread]).T


def all_summary_features(X):
    """
    Summary/confounder-only features derived from coefficient vectors.
    These are not raw coefficients; they are low-dimensional summaries.
    """
    return np.hstack(
        [
            coeff_norm_features(X),
            sparsity_features(X, eps=0.0),
            simple_stats_features(X),
            center_of_mass_features(X),
        ]
    )