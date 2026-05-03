# src/evaluation/distribution_diagnostics.py

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_positive_values(x):
    """
    Keep finite positive values only.
    """
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    x = x[x > 0]
    return x


def maybe_subsample(x, max_n=None, seed: int = 42):
    """
    Optional random subsampling for faster fitting.
    """
    x = np.asarray(x)

    if max_n is not None and len(x) > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x), size=max_n, replace=False)
        return x[idx]

    return x


def fit_powerlaw_model(x, sample_max=None, seed: int = 42):
    """
    Fit a continuous power-law model using the powerlaw package.

    Returns
    -------
    fit : powerlaw.Fit
    x_fit : np.ndarray
    """
    import powerlaw

    x = clean_positive_values(x)
    x_fit = maybe_subsample(x, max_n=sample_max, seed=seed)

    fit = powerlaw.Fit(
        x_fit,
        discrete=False,
        verbose=False,
    )

    return fit, x_fit


def compare_powerlaw_to_alternatives(fit):
    """
    Compare fitted power law against lognormal and exponential alternatives.
    """
    out = {}

    for alternative in ["lognormal", "exponential"]:
        try:
            R, p = fit.distribution_compare("power_law", alternative)
            out[f"R_vs_{alternative}"] = float(R)
            out[f"p_vs_{alternative}"] = float(p)
        except Exception:
            out[f"R_vs_{alternative}"] = np.nan
            out[f"p_vs_{alternative}"] = np.nan

    return out


def bootstrap_ks_pvalue_fixed_xmin(
    fit,
    x_fit,
    n_boot: int = 500,
    seed: int = 42,
):
    """
    Parametric bootstrap KS diagnostic with fixed xmin.

    This is a diagnostic only; it should not be interpreted as proof
    of an exact power-law generative mechanism.
    """
    import powerlaw

    xmin = float(fit.xmin)
    tail = x_fit[x_fit >= xmin]
    n_tail = int(len(tail))

    d_emp = float(fit.power_law.D)

    rng = np.random.default_rng(seed)
    ds = []

    for _ in range(n_boot):
        # Use powerlaw's fitted generator. The rng object is not used directly
        # by powerlaw, but setting the global seed gives some reproducibility.
        np.random.seed(int(rng.integers(0, 2**31 - 1)))

        syn = fit.power_law.generate_random(n_tail)
        syn_fit = powerlaw.Fit(
            syn,
            xmin=xmin,
            discrete=False,
            verbose=False,
        )
        ds.append(float(syn_fit.power_law.D))

    ds = np.asarray(ds, dtype=float)
    p_boot = float((np.sum(ds >= d_emp) + 1) / (len(ds) + 1))

    return d_emp, p_boot, n_tail


def run_distribution_diagnostics(
    score_dict: dict[str, np.ndarray],
    sample_max=None,
    n_boot: int = 500,
    seed: int = 42,
):
    """
    Run power-law style distribution diagnostics for multiple invariants.

    Parameters
    ----------
    score_dict:
        Dictionary mapping invariant name to positive score array.

    Returns
    -------
    df : pd.DataFrame
    fits : dict
    """
    rows = []
    fits = {}

    for invariant, scores in score_dict.items():
        fit, x_fit = fit_powerlaw_model(
            scores,
            sample_max=sample_max,
            seed=seed,
        )

        fits[invariant] = fit

        comp = compare_powerlaw_to_alternatives(fit)
        d_emp, p_boot, n_tail = bootstrap_ks_pvalue_fixed_xmin(
            fit,
            x_fit,
            n_boot=n_boot,
            seed=seed,
        )

        rows.append(
            {
                "Invariant": invariant,
                "n": int(len(clean_positive_values(scores))),
                "fit_n": int(len(x_fit)),
                "xmin": float(fit.xmin),
                "alpha_mle": float(fit.alpha),
                "KS_D": float(d_emp),
                "p_bootstrap_KS": float(p_boot),
                "n_tail": int(n_tail),
                **comp,
            }
        )

    return pd.DataFrame(rows), fits


def plot_ccdf_row(fits, out_pdf, out_png=None):
    """
    Plot empirical CCDFs and fitted power-law/lognormal curves.

    Notes
    -----
    This function intentionally keeps all distributional claims diagnostic.
    """
    import matplotlib.pyplot as plt

    n = len(fits)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4), sharey=True)

    if n == 1:
        axes = [axes]

    for ax, (invariant, fit) in zip(axes, fits.items()):
        fit.plot_ccdf(
            ax=ax,
            linewidth=2,
            label="Empirical CCDF",
        )

        fit.power_law.plot_ccdf(
            ax=ax,
            linestyle="--",
            linewidth=2,
            label="Power-law fit",
        )

        try:
            fit.lognormal.plot_ccdf(
                ax=ax,
                linestyle=":",
                linewidth=2,
                label="Lognormal fit",
            )
        except Exception:
            pass

        ax.axvline(
            fit.xmin,
            linestyle="dashdot",
            linewidth=1.5,
            label=r"$x_{\min}$",
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(invariant)
        ax.set_xlabel("Normalized reconstruction error")

    axes[0].set_ylabel(r"CCDF $P(X \geq x)$")
    axes[-1].legend(frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(out_pdf, dpi=300)

    if out_png is not None:
        plt.savefig(out_png, dpi=300)

    plt.close()