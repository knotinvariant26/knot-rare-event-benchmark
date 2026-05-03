# src/reporting/make_tables.py

from __future__ import annotations

import numpy as np
import pandas as pd

from src.evaluation.tail_metrics import compute_tail_table


def make_jones_main_table(df_master: pd.DataFrame):
    """
    Filter the master tail table to the Jones rows used in the main paper.
    """
    return df_master[
        (df_master["Invariant"] == "Jones")
        & (df_master["Split"] == "test")
        & (df_master["Scope"] == "MAIN")
    ].copy()


def make_y12_stress_table(score_dir, metadata_test, seed: int = 42):
    """
    Compute Y12 qualitative stress test for Jones PCA and AE scores.
    """
    from pathlib import Path

    score_dir = Path(score_dir)
    y_test = metadata_test["signature"].to_numpy().astype(int)

    rows = []

    for method in ["PCA", "AE"]:
        score_path = score_dir / f"nre_Jones_{method}_test_seed{seed}.npy"

        if not score_path.exists():
            raise FileNotFoundError(f"Missing score file: {score_path}")

        scores = np.load(score_path)

        df_y12 = compute_tail_table(
            scores=scores,
            y_signature=y_test,
            s_list=[12],
            taus=[0.99],
        )

        df_y12["Invariant"] = "Jones"
        df_y12["Score"] = f"NRE_{method}"
        df_y12["Split"] = "test"

        rows.append(df_y12)

    return pd.concat(rows, ignore_index=True)