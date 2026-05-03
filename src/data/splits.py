# src/data/splits.py

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler


def filter_signature_classes(
    X_A,
    X_J,
    X_H,
    metadata: pd.DataFrame,
    min_class_count: int = 2,
):
    """
    Remove signature classes with fewer than min_class_count samples.

    This is needed for stratified splitting. In the current benchmark,
    sigma=14 has only one sample and is removed before the split.
    """
    sig = metadata["signature"].to_numpy()

    vc = pd.Series(sig).value_counts()
    keep_classes = vc[vc >= min_class_count].index.to_numpy()
    mask_keep = np.isin(sig, keep_classes)

    return {
        "X_A": X_A[mask_keep],
        "X_J": X_J[mask_keep],
        "X_H": X_H[mask_keep],
        "metadata": metadata.loc[mask_keep].reset_index(drop=True),
        "dropped_signature_classes": vc[vc < min_class_count].to_dict(),
    }


def make_splits_stratified(
    y,
    test_size: float = 0.10,
    val_size: float = 0.10,
    seed: int = 42,
):
    """
    Make 80/10/10 stratified train/validation/test splits by signature.
    """
    y = np.asarray(y).ravel()
    idx = np.arange(len(y))

    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=seed,
    )
    train_val_idx, test_idx = next(sss1.split(idx, y))

    y_train_val = y[train_val_idx]

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_size / (1.0 - test_size),
        random_state=seed,
    )
    train_rel, val_rel = next(sss2.split(train_val_idx, y_train_val))

    train_idx = train_val_idx[train_rel]
    val_idx = train_val_idx[val_rel]

    return train_idx, val_idx, test_idx


def scale_train_only(X, train_idx, val_idx, test_idx):
    """
    Fit StandardScaler on train only, then transform validation and test.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_val = scaler.transform(X[val_idx])
    X_test = scaler.transform(X[test_idx])
    return X_train, X_val, X_test, scaler


def prepare_splits_and_scalers(
    X_A,
    X_J,
    X_H,
    metadata: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.10,
    val_size: float = 0.10,
):
    """
    Filter rare signature classes, create stratified splits, and scale features
    using training-set statistics only.
    """
    filtered = filter_signature_classes(
        X_A,
        X_J,
        X_H,
        metadata,
        min_class_count=2,
    )

    metadata_f = filtered["metadata"]
    y = metadata_f["signature"].to_numpy()

    train_idx, val_idx, test_idx = make_splits_stratified(
        y,
        test_size=test_size,
        val_size=val_size,
        seed=seed,
    )

    A_train, A_val, A_test, sc_A = scale_train_only(
        filtered["X_A"], train_idx, val_idx, test_idx
    )
    J_train, J_val, J_test, sc_J = scale_train_only(
        filtered["X_J"], train_idx, val_idx, test_idx
    )
    H_train, H_val, H_test, sc_H = scale_train_only(
        filtered["X_H"], train_idx, val_idx, test_idx
    )

    return {
        "A": {"train": A_train, "val": A_val, "test": A_test, "scaler": sc_A},
        "J": {"train": J_train, "val": J_val, "test": J_test, "scaler": sc_J},
        "H": {"train": H_train, "val": H_val, "test": H_test, "scaler": sc_H},
        "metadata": {
            "all": metadata_f,
            "train": metadata_f.iloc[train_idx].reset_index(drop=True),
            "val": metadata_f.iloc[val_idx].reset_index(drop=True),
            "test": metadata_f.iloc[test_idx].reset_index(drop=True),
        },
        "indices": {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        },
        "dropped_signature_classes": filtered["dropped_signature_classes"],
    }