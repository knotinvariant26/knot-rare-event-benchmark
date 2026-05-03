# src/plotting/tail_scatter.py

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def clip_quantile(x, q: float = 0.999):
    x = np.asarray(x, dtype=np.float64).ravel()
    finite = x[np.isfinite(x)]
    hi = np.quantile(finite, q)
    return np.clip(x, 0, hi), float(hi)


def make_jones_tail_scatter(
    nre_ae,
    nre_pca,
    signature,
    out_dir="results/figures",
    taus=(0.95, 0.99),
    sig_lines=(8, 10),
    clip_q: float = 0.999,
    prefix: str = "jones_tail_scatter",
):
    """
    Make scatter plots of Jones AE NRE versus |signature|, colored by PCA NRE.

    Outputs
    -------
    - full scatter PNG/PDF
    - tail zoom PNG/PDF
    - support CSV
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nre_ae = np.asarray(nre_ae, dtype=np.float64).ravel()
    nre_pca = np.asarray(nre_pca, dtype=np.float64).ravel()
    signature = np.asarray(signature).astype(int).ravel()
    abs_sig = np.abs(signature)

    if not (len(nre_ae) == len(nre_pca) == len(abs_sig)):
        raise ValueError(
            f"Length mismatch: len(nre_ae)={len(nre_ae)}, "
            f"len(nre_pca)={len(nre_pca)}, len(signature)={len(signature)}"
        )

    q_tau = {tau: float(np.quantile(nre_ae, tau)) for tau in taus}

    x_plot, _ = clip_quantile(nre_ae, clip_q)
    c_plot, _ = clip_quantile(nre_pca, clip_q)

    # Full scatter
    plt.figure(figsize=(6.6, 4.4))
    sc = plt.scatter(x_plot, abs_sig, c=c_plot, s=6, alpha=0.35)
    plt.xlabel(r"NRE$_{\mathrm{AE}}$ (Jones, test)")
    plt.ylabel(r"$|\sigma(K)|$")

    cb = plt.colorbar(sc)
    cb.set_label(r"NRE$_{\mathrm{PCA}}$ (Jones, test)")

    for s in sig_lines:
        plt.axhline(s, linewidth=1)

    y_text = max(abs_sig) + 0.5
    for tau in taus:
        plt.axvline(q_tau[tau], linestyle="--", linewidth=1)
        plt.text(
            q_tau[tau],
            y_text,
            rf"$\tau={tau:.2f}$",
            rotation=90,
            va="top",
            ha="left",
        )

    plt.tight_layout()

    full_png = out_dir / f"{prefix}_full.png"
    full_pdf = out_dir / f"{prefix}_full.pdf"
    plt.savefig(full_png, dpi=300)
    plt.savefig(full_pdf, dpi=300)
    plt.close()

    # Tail zoom
    tau_zoom = 0.95
    mask_tail = nre_ae >= q_tau[tau_zoom]

    x_tail, _ = clip_quantile(nre_ae[mask_tail], clip_q)
    c_tail, _ = clip_quantile(nre_pca[mask_tail], clip_q)
    abs_tail = abs_sig[mask_tail]

    plt.figure(figsize=(6.6, 4.4))
    sc_t = plt.scatter(x_tail, abs_tail, c=c_tail, s=10, alpha=0.5)

    plt.xlabel(rf"NRE$_{{\mathrm{{AE}}}}$ (Jones, test) [tail $\geq \tau={tau_zoom:.2f}$]")
    plt.ylabel(r"$|\sigma(K)|$")

    cb_t = plt.colorbar(sc_t)
    cb_t.set_label(r"NRE$_{\mathrm{PCA}}$ (Jones, test)")

    for s in sig_lines:
        plt.axhline(s, linewidth=1)

    plt.axvline(q_tau[0.99], linestyle="--", linewidth=1)
    plt.text(
        q_tau[0.99],
        max(abs_tail) + 0.5,
        r"$\tau=0.99$",
        rotation=90,
        va="top",
        ha="left",
    )

    plt.tight_layout()

    zoom_png = out_dir / f"{prefix}_tail_zoom.png"
    zoom_pdf = out_dir / f"{prefix}_tail_zoom.pdf"
    plt.savefig(zoom_png, dpi=300)
    plt.savefig(zoom_pdf, dpi=300)
    plt.close()

    # Support CSV
    df_support = pd.DataFrame(
        {
            "signature": signature,
            "abs_signature": abs_sig,
            "nre_ae": nre_ae,
            "nre_pca": nre_pca,
        }
    )

    for tau in taus:
        df_support[f"tau_{tau:.2f}_threshold_nre_ae"] = q_tau[tau]

    support_csv = out_dir / f"{prefix}_support.csv"
    df_support.to_csv(support_csv, index=False)

    return {
        "full_png": str(full_png),
        "full_pdf": str(full_pdf),
        "zoom_png": str(zoom_png),
        "zoom_pdf": str(zoom_pdf),
        "support_csv": str(support_csv),
        "thresholds": q_tau,
    }