# src/evaluation/confounders.py

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score


def l2_norm(X):
    X = np.asarray(X)
    return np.sqrt(np.sum(X * X, axis=1))


def l0_sparsity(X, tol: float = 0.0):
    X = np.asarray(X)
    if tol <= 0:
        return np.sum(X != 0, axis=1)
    return np.sum(np.abs(X) > tol, axis=1)


def support_width_from_coeffs(X, eps: float = 1e-8):
    """
    Proxy support width from the span of nonzero coefficient indices.
    Used for HOMFLY--PT when exponent-width metadata is not directly comparable.
    """
    X = np.asarray(X, dtype=np.float64)
    nonzero = np.abs(X) > eps
    idx = np.arange(X.shape[1], dtype=np.float64)

    min_idx = np.where(nonzero, idx, np.inf).min(axis=1)
    max_idx = np.where(nonzero, idx, -np.inf).max(axis=1)

    span = np.where(np.isfinite(min_idx), max_idx - min_idx, 0.0)
    return span.astype(np.float64)


def build_confounder_matrix_scaled_space(
    X_scaled,
    width,
    crossing,
    l0_tol: float = 0.1,
):
    """
    Confounders computed in the same scaled coefficient space used by models:
    [L2 norm, L0 proxy, support/exponent width, crossing number].
    """
    X_scaled = np.asarray(X_scaled, dtype=np.float32)
    width = np.asarray(width).ravel().astype(np.float32)
    crossing = np.asarray(crossing).ravel().astype(np.float32)

    return np.stack(
        [
            l2_norm(X_scaled).astype(np.float32),
            l0_sparsity(X_scaled, tol=l0_tol).astype(np.float32),
            width,
            crossing,
        ],
        axis=1,
    )


def compute_spearman_confounders(
    X,
    nre,
    crossing,
    support_width,
    l0_tol: float = 1e-8,
):
    """
    Spearman correlations between NRE and simple coefficient/confounder statistics.
    """
    X = np.asarray(X, dtype=np.float64)
    nre = np.asarray(nre, dtype=np.float64).ravel()
    crossing = np.asarray(crossing, dtype=np.float64).ravel()
    support_width = np.asarray(support_width, dtype=np.float64).ravel()

    confounders = {
        "L2_norm": np.linalg.norm(X, axis=1),
        "L0_sparsity": np.sum(np.abs(X) > l0_tol, axis=1),
        "Support_width": support_width,
        "Crossings": crossing,
    }

    rows = []
    for name, values in confounders.items():
        if np.all(values == values[0]) or np.all(nre == nre[0]):
            rho, p_value = np.nan, np.nan
        else:
            rho, p_value = spearmanr(values, nre)

        rows.append(
            {
                "Confounder": name,
                "Spearman_rho": float(rho) if np.isfinite(rho) else np.nan,
                "p_value": float(p_value) if p_value is not None and np.isfinite(p_value) else np.nan,
            }
        )

    return pd.DataFrame(rows)


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

def topk_tail_mask(scores, tau: float):
    """
    Exact fixed-mass top-k tail.

    Selects exactly ceil((1 - tau) * n) samples with the largest scores.
    Ties are broken deterministically by stable sorting.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    n = int(len(scores))
    k = int(np.ceil((1.0 - tau) * n))

    if k <= 0:
        raise ValueError(f"tau={tau} gives k={k}.")
    if k > n:
        raise ValueError(f"tau={tau} gives k={k} > n={n}.")

    order = np.argsort(-scores, kind="mergesort")
    mask = np.zeros(n, dtype=bool)
    mask[order[:k]] = True
    return mask

def enrichment_at_tau_with_counts(scores, y_true, tau: float):
    """
    Exact fixed-mass top-k tail enrichment.

    Uses exactly ceil((1 - tau) * n) samples with the largest scores.
    """
    scores = np.asarray(scores, dtype=np.float64).ravel()
    y_true = np.asarray(y_true).astype(int).ravel()

    if len(scores) != len(y_true):
        raise ValueError(
            f"Length mismatch: len(scores)={len(scores)}, len(y_true)={len(y_true)}"
        )

    n = int(len(y_true))
    tail = topk_tail_mask(scores, tau)
    tail_size = int(tail.sum())

    base_rate = float(y_true.mean()) if n > 0 else np.nan
    positives = int(y_true.sum())
    tail_captured = int(y_true[tail].sum())

    if base_rate == 0 or tail_size == 0:
        return np.nan, positives, tail_captured, tail_size

    enrichment = (tail_captured / tail_size) / base_rate
    return float(enrichment), positives, tail_captured, tail_size


def lr_model(seed: int = 42):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def lr_with_interactions(seed: int = 42):
    return Pipeline(
        [
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=seed,
                ),
            ),
        ]
    )


def gbdt_model(seed: int = 42):
    return HistGradientBoostingClassifier(
        max_depth=3,
        learning_rate=0.05,
        max_iter=300,
        random_state=seed,
    )


def fit_score(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    return model.predict_proba(X_test)[:, 1]


def eval_scores(scores, ybin, taus=(0.95, 0.99)):
    out = {
        "AUROC": safe_auroc(ybin, scores),
        "AUPRC": safe_auprc(ybin, scores),
        "Positives": int(np.sum(ybin)),
        "N_test": int(len(ybin)),
        "Base_rate": float(np.mean(ybin)),
    }

    for tau in taus:
        enrichment, _, tail_captured, tail_size = enrichment_at_tau_with_counts(
            scores=scores,
            y_true=ybin,
            tau=tau,
        )
        out[f"Enr@{tau}"] = enrichment
        out[f"Tail@{tau}"] = f"{tail_captured}/{tail_size}"

    return out


def evaluate_confounder_models_jones(
    C_train,
    C_test,
    sig_train,
    sig_test,
    targets=(8, 10),
    taus=(0.95, 0.99),
    seed: int = 42,
):
    """
    Evaluate whether simple confounders alone can predict rare large-signature targets.
    """
    rows = []

    single_idxs = {
        "L2 only": 0,
        "L0 only": 1,
        "width only": 2,
        "crossings only": 3,
    }

    for s in targets:
        ybin_train = (np.abs(sig_train) >= s).astype(int)
        ybin_test = (np.abs(sig_test) >= s).astype(int)

        for name, j in single_idxs.items():
            scores = fit_score(
                lr_model(seed=seed),
                C_train[:, [j]],
                ybin_train,
                C_test[:, [j]],
            )
            rows.append(
                {
                    "Invariant": "Jones",
                    "Target": f"Y{s}",
                    "Model": name,
                    **eval_scores(scores, ybin_test, taus=taus),
                }
            )

        scores = fit_score(lr_model(seed=seed), C_train, ybin_train, C_test)
        rows.append(
            {
                "Invariant": "Jones",
                "Target": f"Y{s}",
                "Model": "LR (all confounders)",
                **eval_scores(scores, ybin_test, taus=taus),
            }
        )

        scores = fit_score(lr_with_interactions(seed=seed), C_train, ybin_train, C_test)
        rows.append(
            {
                "Invariant": "Jones",
                "Target": f"Y{s}",
                "Model": "LR + pairwise interactions",
                **eval_scores(scores, ybin_test, taus=taus),
            }
        )

        scores = fit_score(gbdt_model(seed=seed), C_train, ybin_train, C_test)
        rows.append(
            {
                "Invariant": "Jones",
                "Target": f"Y{s}",
                "Model": "GBDT (confounders only)",
                **eval_scores(scores, ybin_test, taus=taus),
            }
        )

    cols = [
        "Invariant",
        "Target",
        "Model",
        "AUROC",
        "AUPRC",
        "Enr@0.95",
        "Tail@0.95",
        "Enr@0.99",
        "Tail@0.99",
        "Positives",
        "N_test",
        "Base_rate",
    ]

    return pd.DataFrame(rows)[cols].sort_values(["Target", "Model"]).reset_index(drop=True)


def tail_mask(scores, tau: float):
    return topk_tail_mask(scores, tau)


def overlap_stats(mask_a, mask_b):
    mask_a = np.asarray(mask_a).astype(bool)
    mask_b = np.asarray(mask_b).astype(bool)

    inter = mask_a & mask_b
    union = mask_a | mask_b

    a = int(mask_a.sum())
    b = int(mask_b.sum())
    i = int(inter.sum())
    u = int(union.sum())

    return {
        "jaccard": float(i / u) if u > 0 else np.nan,
        "inter": i,
        "union": u,
        "size_a": a,
        "size_b": b,
        "cov_a": float(i / a) if a > 0 else np.nan,
        "cov_b": float(i / b) if b > 0 else np.nan,
    }


def tail_overlap_against_confounders(
    ae_scores,
    C_train,
    C_test,
    sig_train,
    sig_test,
    targets=(8, 10),
    taus=(0.95, 0.99),
    seed: int = 42,
):
    """
    Compare AE tail with a GBDT confounder-only tail.
    """
    rows = []
    ae_only_positive_rows = []

    ae_scores = np.asarray(ae_scores).ravel()
    sig_train = np.asarray(sig_train).astype(int)
    sig_test = np.asarray(sig_test).astype(int)

    for s in targets:
        y_train = (np.abs(sig_train) >= s).astype(int)
        y_test = (np.abs(sig_test) >= s).astype(int)

        conf_scores = fit_score(
            gbdt_model(seed=seed),
            C_train,
            y_train,
            C_test,
        )

        for tau in taus:
            mask_ae = tail_mask(ae_scores, tau)
            mask_cf = tail_mask(conf_scores, tau)

            ov = overlap_stats(mask_ae, mask_cf)

            enr_ae, _, pos_ae, tail_ae = enrichment_at_tau_with_counts(ae_scores, y_test, tau)
            enr_cf, _, pos_cf, tail_cf = enrichment_at_tau_with_counts(conf_scores, y_test, tau)

            ae_only = mask_ae & (~mask_cf)
            ae_only_pos = int(y_test[ae_only].sum())
            ae_only_size = int(ae_only.sum())

            rows.append(
                {
                    "Invariant": "Jones",
                    "Target": f"Y{s}",
                    "Tau": float(tau),
                    "TailSize_AE": int(tail_ae),
                    "TailSize_Conf": int(tail_cf),
                    "Tail_Intersection": ov["inter"],
                    "Jaccard": ov["jaccard"],
                    "AE_tail_covered_by_Conf": ov["cov_a"],
                    "Conf_tail_covered_by_AE": ov["cov_b"],
                    "Positives_in_test": int(y_test.sum()),
                    "AE_pos_in_tail": int(pos_ae),
                    "Conf_pos_in_tail": int(pos_cf),
                    "AE_Enrichment": float(enr_ae),
                    "Conf_Enrichment": float(enr_cf),
                    "AE_only_tail_size": ae_only_size,
                    "AE_only_pos_in_tail": ae_only_pos,
                }
            )

            if tau == 0.99 and ae_only_pos > 0:
                idxs = np.where(ae_only & (y_test == 1))[0]
                for j in idxs[:50]:
                    ae_only_positive_rows.append(
                        {
                            "Invariant": "Jones",
                            "Target": f"Y{s}",
                            "Tau": float(tau),
                            "test_position": int(j),
                            "signature": int(sig_test[j]),
                            "ae_score": float(ae_scores[j]),
                            "conf_score": float(conf_scores[j]),
                        }
                    )

    df_overlap = pd.DataFrame(rows).sort_values(["Target", "Tau"]).reset_index(drop=True)
    df_ae_only = pd.DataFrame(ae_only_positive_rows)

    return df_overlap, df_ae_only


def tail_enrichment_by_bin_stable(
    df,
    tau=0.99,
    sig_threshold=10,
    n_bins=3,
    support_col="support_width",
    nre_col="NRE_AE",
    sig_col="signature",
    smooth=1.0,
):
    """
    Smoothed enrichment by support-width bins.
    This is a sanity check, not a main claim.
    """
    df = df.copy()
    threshold = df[nre_col].quantile(tau)

    df["is_tail"] = topk_tail_mask(df[nre_col].to_numpy(), tau)
    df["is_high_sig"] = (np.abs(df[sig_col].to_numpy()) >= sig_threshold)

    df["support_bin"] = pd.qcut(df[support_col], n_bins, duplicates="drop")

    rows = []

    for b, g in df.groupby("support_bin", observed=True):
        n = int(len(g))
        n_tail = int(g["is_tail"].sum())

        k_all = int(g["is_high_sig"].sum())
        k_tail = int(g.loc[g["is_tail"], "is_high_sig"].sum()) if n_tail > 0 else 0

        p_all = (k_all + smooth) / (n + 2 * smooth)
        p_tail = (k_tail + smooth) / (n_tail + 2 * smooth) if n_tail > 0 else np.nan

        enrichment = (p_tail / p_all) if np.isfinite(p_tail) else np.nan

        rows.append(
            {
                "Support_bin": str(b),
                "N": n,
                "N_tail_in_bin": n_tail,
                "tau": float(tau),
                "sig_threshold": int(sig_threshold),
                "k_highsig_all": k_all,
                "k_highsig_tail": k_tail,
                "rate_all_smoothed": float(p_all),
                "rate_tail_smoothed": float(p_tail) if np.isfinite(p_tail) else np.nan,
                "Enrichment_smoothed": float(enrichment) if np.isfinite(enrichment) else np.nan,
            }
        )

    return pd.DataFrame(rows)


def enrichment_sensitivity_stable(
    df,
    taus=(0.98, 0.99, 0.995),
    sig_thresholds=(10, 12),
    n_bins=3,
    smooth=1.0,
):
    out = []

    for tau in taus:
        for s in sig_thresholds:
            t = tail_enrichment_by_bin_stable(
                df,
                tau=tau,
                sig_threshold=s,
                n_bins=n_bins,
                smooth=smooth,
            )
            t["Tau"] = float(tau)
            t["Target"] = f"Y{s}"
            out.append(t)

    return pd.concat(out, ignore_index=True)
