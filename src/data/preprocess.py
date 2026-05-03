# src/data/preprocess.py

import re
import numpy as np
import pandas as pd


def make_knot_key(df: pd.DataFrame) -> pd.Series:
    """
    Build the alignment key used across Alexander, Jones, and HOMFLY--PT tables.

    The table_number is not unique unless is_alternating is included.
    Therefore, rows are aligned by:
        (number_of_crossings, is_alternating, table_number)
    """
    required = {"number_of_crossings", "table_number", "is_alternating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns for knot_key: {missing}")

    return (
        df["number_of_crossings"].astype(int).astype(str)
        + "_alt"
        + df["is_alternating"].astype(int).astype(str)
        + "_"
        + df["table_number"].astype(int).astype(str)
    )


def get_coef_columns(df: pd.DataFrame, invariant: str) -> list[str]:
    """
    Detect coefficient columns for each polynomial encoding.
    """
    if invariant == "alexander":
        cols = [c for c in df.columns if re.fullmatch(r"A\d+", str(c))]
    elif invariant == "jones":
        cols = [c for c in df.columns if re.fullmatch(r"J-?\d+", str(c))]
    elif invariant == "homfly":
        cols = [c for c in df.columns if re.fullmatch(r"a-?\d+_z-?\d+", str(c))]
    else:
        raise ValueError(f"Unknown invariant: {invariant}")

    if not cols:
        raise ValueError(f"No coefficient columns detected for {invariant}. Check df.columns.")

    def sort_key(c: str):
        if invariant == "alexander":
            return int(c[1:])
        if invariant == "jones":
            return int(c[1:])
        if invariant == "homfly":
            m = re.fullmatch(r"a(-?\d+)_z(-?\d+)", str(c))
            return (int(m.group(1)), int(m.group(2)))

    return sorted(cols, key=sort_key)


def drop_mirror_rows(df: pd.DataFrame, drop_mirrors: bool = True) -> pd.DataFrame:
    """
    Drop mirror entries when mirror information is encoded with '!'.

    In these files, mirrors may appear in columns such as:
      - knot_id
      - representation
    """
    if not drop_mirrors:
        return df.copy()

    out = df.copy()

    for col in ["knot_id", "representation"]:
        if col in out.columns:
            out = out[~out[col].astype(str).str.contains("!", na=False)].copy()

    return out


def preprocess_align(
    A: pd.DataFrame,
    J: pd.DataFrame,
    H: pd.DataFrame,
    max_cross: int = 15,
    drop_mirrors: bool = True,
    drop_signature_mismatches: bool = True,
    verbose: bool = True,
):
    """
    Align Alexander, Jones, and HOMFLY--PT coefficient tables.

    Alignment key:
        (number_of_crossings, is_alternating, table_number)

    The benchmark uses Alexander/Jones signature metadata as the external
    evaluation probe. Rows with inconsistent signature metadata across files
    are removed conservatively.

    Returns a dictionary with:
      - X_A, X_J, X_H
      - signature, crossing, metadata
      - coefficient column names
      - debug dictionary
    """

    A = A.copy()
    J = J.copy()
    H = H.copy()

    required_common = {"number_of_crossings", "table_number", "is_alternating", "signature"}
    for name, df in [("Alexander", A), ("Jones", J), ("HOMFLY", H)]:
        missing = required_common - set(df.columns)
        if missing:
            raise ValueError(f"{name} missing required columns: {missing}")

    # 1. Filter by crossing number.
    A = A[A["number_of_crossings"].between(3, max_cross)].copy()
    J = J[J["number_of_crossings"].between(3, max_cross)].copy()
    H = H[H["number_of_crossings"].between(3, max_cross)].copy()

    n_after_cross_filter = {"A": len(A), "J": len(J), "H": len(H)}

    # 2. Drop mirrors if encoded.
    A = drop_mirror_rows(A, drop_mirrors)
    J = drop_mirror_rows(J, drop_mirrors)
    H = drop_mirror_rows(H, drop_mirrors)

    n_after_mirror_filter = {"A": len(A), "J": len(J), "H": len(H)}

    # 3. Build alignment keys.
    A["knot_key"] = make_knot_key(A)
    J["knot_key"] = make_knot_key(J)
    H["knot_key"] = make_knot_key(H)

    # 4. Duplicate protection.
    for name, df in [("Alexander", A), ("Jones", J), ("HOMFLY", H)]:
        dup = int(df["knot_key"].duplicated().sum())
        if dup > 0:
            raise ValueError(f"{name} has duplicated knot_key values after mirror removal: {dup}")

    # 5. Detect coefficient columns.
    coef_A = get_coef_columns(A, "alexander")
    coef_J = get_coef_columns(J, "jones")
    coef_H = get_coef_columns(H, "homfly")

    # 6. Prepare metadata and feature tables.
    meta_cols = [
        "knot_key",
        "number_of_crossings",
        "table_number",
        "is_alternating",
        "signature",
        "minimum_exponent",
        "maximum_exponent",
    ]
    meta_A = [c for c in meta_cols if c in A.columns]

    A_small = A[meta_A + coef_A].copy()
    J_small = J[["knot_key"] + coef_J].copy()
    H_small = H[["knot_key"] + coef_H].copy()

    # Keep signatures from J/H only for consistency checks.
    J_sig = J[["knot_key", "signature"]].rename(columns={"signature": "signature_J"})
    H_sig = H[["knot_key", "signature"]].rename(columns={"signature": "signature_H"})

    J_small = J_small.merge(J_sig, on="knot_key", how="left", validate="one_to_one")
    H_small = H_small.merge(H_sig, on="knot_key", how="left", validate="one_to_one")

    # 7. Merge all representations.
    M = A_small.merge(J_small, on="knot_key", how="inner", validate="one_to_one")
    M = M.merge(H_small, on="knot_key", how="inner", validate="one_to_one")

    aligned_rows_before_signature_filter = len(M)

    # 8. Conservative signature consistency filter.
    mismatch_J = (M["signature"] != M["signature_J"]).to_numpy()
    mismatch_H = (M["signature"] != M["signature_H"]).to_numpy()

    n_mismatch_J = int(mismatch_J.sum())
    n_mismatch_H = int(mismatch_H.sum())

    mismatch_any = mismatch_J | mismatch_H
    n_mismatch_any = int(mismatch_any.sum())

    if n_mismatch_any > 0:
        if drop_signature_mismatches:
            print(
                f"Dropping {n_mismatch_any} rows with inconsistent signature metadata "
                f"(A-J mismatches={n_mismatch_J}, A-H mismatches={n_mismatch_H})."
            )
            M = M.loc[~mismatch_any].reset_index(drop=True)
        else:
            raise ValueError(
                f"Signature metadata mismatches detected: "
                f"A-J={n_mismatch_J}, A-H={n_mismatch_H}."
            )

    # 9. Extract arrays.
    X_A = M[coef_A].to_numpy(dtype=np.float32)
    X_J = M[coef_J].to_numpy(dtype=np.float32)
    X_H = M[coef_H].to_numpy(dtype=np.float32)

    sig = M["signature"].to_numpy()
    cross = M["number_of_crossings"].to_numpy()

    if "minimum_exponent" in M.columns and "maximum_exponent" in M.columns:
        min_exp = M["minimum_exponent"].to_numpy()
        max_exp = M["maximum_exponent"].to_numpy()
        width = max_exp - min_exp
    else:
        min_exp = None
        max_exp = None
        width = None

    metadata_cols = [
        c for c in [
            "knot_key",
            "number_of_crossings",
            "table_number",
            "is_alternating",
            "signature",
            "minimum_exponent",
            "maximum_exponent",
        ]
        if c in M.columns
    ]
    metadata = M[metadata_cols].copy()

    debug = {
        "max_cross": int(max_cross),
        "drop_mirrors": bool(drop_mirrors),
        "drop_signature_mismatches": bool(drop_signature_mismatches),
        "rows_after_cross_filter": n_after_cross_filter,
        "rows_after_mirror_filter": n_after_mirror_filter,
        "aligned_rows_before_signature_filter": int(aligned_rows_before_signature_filter),
        "signature_mismatches": {
            "A_vs_J": int(n_mismatch_J),
            "A_vs_H": int(n_mismatch_H),
            "dropped_any": int(n_mismatch_any if drop_signature_mismatches else 0),
        },
        "aligned_rows_final": int(len(M)),
        "dims": {
            "A": int(X_A.shape[1]),
            "J": int(X_J.shape[1]),
            "H": int(X_H.shape[1]),
        },
        "signature_counts": pd.Series(sig).value_counts().sort_index().to_dict(),
        "crossing_counts": pd.Series(cross).value_counts().sort_index().to_dict(),
        "has_nans": {
            "A": bool(np.isnan(X_A).any()),
            "J": bool(np.isnan(X_J).any()),
            "H": bool(np.isnan(X_H).any()),
        },
        "coef_A_first_last": (coef_A[0], coef_A[-1]),
        "coef_J_first_last": (coef_J[0], coef_J[-1]),
        "coef_H_first_last": (coef_H[0], coef_H[-1]),
    }

    if verbose:
        print("=== preprocess_align DEBUG ===")
        for k, v in debug.items():
            print(k, v)

    return {
        "X_A": X_A,
        "X_J": X_J,
        "X_H": X_H,
        "signature": sig,
        "crossing": cross,
        "min_exp": min_exp,
        "max_exp": max_exp,
        "width": width,
        "metadata": metadata,
        "coef_cols": {"A": coef_A, "J": coef_J, "H": coef_H},
        "debug": debug,
    }