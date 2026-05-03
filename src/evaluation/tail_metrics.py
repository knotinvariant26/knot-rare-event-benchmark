# src/evaluation/tail_metrics.py

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def safe_auroc(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).ravel()

    if len(np.unique(y_true)) < 2:
        return np.nan

    return float(roc_auc_score(y_true, scores))


def safe_auprc(y_true, scores):
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).ravel()

    if y_true.sum() == 0:
        return np.nan

    return float(average_precision_score(y_true, scores))


def enrichment_at_tau_with_counts(scores, y_true, tau: float):
    """
    Fixed-mass tail enrichment.

    scores: larger means more extreme.
    y_true: binary rare-event labels.
    tau: e.g. 0.95 or 0.99.
    """
    scores = np.asarray(scores).ravel()
    y_true = np.asarray(y_true).astype(int)

    n = int(len(y_true))
    k = int(np.ceil((1.0 - tau) * n))

    base_rate = float(y_true.mean()) if n > 0 else np.nan
    n_pos = int(y_true.sum())

    if k <= 0 or base_rate == 0:
        return np.nan, n_pos, 0, k, base_rate

    idx_tail = np.argpartition(scores, -k)[-k:]
    n_pos_tail = int(y_true[idx_tail].sum())
    tail_rate = n_pos_tail / k
    enrichment = tail_rate / base_rate

    return float(enrichment), n_pos, n_pos_tail, k, base_rate


def compute_tail_table(scores, y_signature, s_list, taus):
    """
    Compute AUROC, AUPRC, enrichment, positives, captured positives, tail size,
    and base rate for targets Y_s = 1[|sigma| >= s].
    """
    rows = []

    y_signature = np.asarray(y_signature).astype(int)
    scores = np.asarray(scores).ravel().astype(np.float64)

    for s in s_list:
        ybin = (np.abs(y_signature) >= s).astype(int)

        auroc = safe_auroc(ybin, scores)
        auprc = safe_auprc(ybin, scores)

        for tau in taus:
            enrichment, n_pos, n_pos_tail, tail_size, base_rate = enrichment_at_tau_with_counts(
                scores=scores,
                y_true=ybin,
                tau=tau,
            )

            rows.append(
                {
                    "Target": f"Y{s}",
                    "Tau": float(tau),
                    "AUROC": auroc,
                    "AUPRC": auprc,
                    "Enrichment": enrichment,
                    "Positives": int(n_pos),
                    "Tail_captured": int(n_pos_tail),
                    "Tail_size": int(tail_size),
                    "N": int(len(ybin)),
                    "Base_rate": base_rate,
                }
            )

    return pd.DataFrame(rows)


def bootstrap_ci_metric(ybin, scores, metric_fn, B: int = 200, seed: int = 123):
    """
    Bootstrap confidence intervals for a scalar metric.
    """
    rng = np.random.default_rng(seed)

    ybin = np.asarray(ybin).astype(int)
    scores = np.asarray(scores).ravel()
    n = len(ybin)

    vals = []

    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        v = metric_fn(ybin[idx], scores[idx])
        if np.isfinite(v):
            vals.append(v)

    if len(vals) < 20:
        return np.nan, np.nan, np.nan

    vals = np.sort(vals)

    return (
        float(np.quantile(vals, 0.50)),
        float(np.quantile(vals, 0.025)),
        float(np.quantile(vals, 0.975)),
    )


def bootstrap_cis_test(y_signature, scores, targets=(8, 10), B: int = 200, seed: int = 123):
    """
    Bootstrap AUROC and AUPRC for Y_s targets on test set.
    """
    rows = []

    y_signature = np.asarray(y_signature).astype(int)

    for s in targets:
        ybin = (np.abs(y_signature) >= s).astype(int)

        auroc_med, auroc_lo, auroc_hi = bootstrap_ci_metric(
            ybin,
            scores,
            safe_auroc,
            B=B,
            seed=seed,
        )

        auprc_med, auprc_lo, auprc_hi = bootstrap_ci_metric(
            ybin,
            scores,
            safe_auprc,
            B=B,
            seed=seed,
        )

        rows.append(
            {
                "Target": f"Y{s}",
                "AUROC_med": auroc_med,
                "AUROC_CI_lo": auroc_lo,
                "AUROC_CI_hi": auroc_hi,
                "AUPRC_med": auprc_med,
                "AUPRC_CI_lo": auprc_lo,
                "AUPRC_CI_hi": auprc_hi,
                "Positives": int(ybin.sum()),
                "N": int(len(ybin)),
                "Base_rate": float(ybin.mean()),
            }
        )

    return pd.DataFrame(rows)