# scripts/06_make_paper_tables.py

from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd

from src.reporting.make_tables import make_jones_main_table, make_y12_stress_table


def parse_args():
    parser = argparse.ArgumentParser(description="Create paper-ready CSV tables.")

    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--table_dir", type=str, default="results/tables")
    parser.add_argument("--score_dir", type=str, default="results/scores")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    table_dir = Path(args.table_dir)
    score_dir = Path(args.score_dir)

    table_dir.mkdir(parents=True, exist_ok=True)

    df_master_path = table_dir / f"tail_metrics_main_seed{args.seed}.csv"

    if not df_master_path.exists():
        raise FileNotFoundError(
            f"Missing master table: {df_master_path}. "
            "Run scripts/05_tail_enrichment.py first."
        )

    df_master = pd.read_csv(df_master_path)

    # Main Jones table for paper.
    df_jones = make_jones_main_table(df_master)
    jones_out = table_dir / "jones_main_table.csv"
    df_jones.to_csv(jones_out, index=False)

    print("Saved:", jones_out)
    print(df_jones)

    # Y12 qualitative stress-test appendix table.
    metadata_test = pd.read_csv(data_dir / "metadata_test.csv")
    df_y12 = make_y12_stress_table(
        score_dir=score_dir,
        metadata_test=metadata_test,
        seed=args.seed,
    )

    y12_out = table_dir / "jones_Y12_appendix.csv"
    df_y12.to_csv(y12_out, index=False)

    print("Saved:", y12_out)
    print(df_y12)


if __name__ == "__main__":
    main()