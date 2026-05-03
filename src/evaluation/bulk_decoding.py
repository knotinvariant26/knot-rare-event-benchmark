# src/evaluation/bulk_decoding.py

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


def eval_multinomial_lr_split(
    X_train,
    y_train,
    X_test,
    y_test,
    C: float = 1.0,
    seed: int = 42,
    max_iter: int = 5000,
    class_weight: str | dict | None = "balanced",
):
    """
    Train a class-balanced logistic regression probe and evaluate
    accuracy and macro-F1 on a held-out test set.

    In recent scikit-learn versions, multinomial behavior is selected
    automatically for multiclass problems, so we do not set multi_class.
    """
    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter,
        tol=1e-4,
        C=C,
        class_weight=class_weight,
        random_state=seed,
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "n_iter": clf.n_iter_.tolist(),
        "model": clf,
    }


def majority_baseline(y_train, y_test):
    """
    Majority-class baseline accuracy.
    """
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    values, counts = np.unique(y_train, return_counts=True)
    majority_class = values[np.argmax(counts)]

    return {
        "majority_class": int(majority_class),
        "accuracy": float((y_test == majority_class).mean()),
    }


def split_sanity_checks(y, train_idx, val_idx=None, test_idx=None):
    """
    Basic sanity checks for stratified splits.
    """
    y = np.asarray(y)

    out = {}

    train_classes, train_counts = np.unique(y[train_idx], return_counts=True)
    out["n_classes_train"] = int(len(train_classes))
    out["train_class_min"] = int(train_classes.min())
    out["train_class_max"] = int(train_classes.max())
    out["train_smallest_counts"] = np.sort(train_counts)[:10].astype(int).tolist()
    out["train_size"] = int(len(train_idx))

    if val_idx is not None:
        out["val_size"] = int(len(val_idx))
        out["train_val_overlap"] = int(len(set(train_idx).intersection(set(val_idx))))

    if test_idx is not None:
        out["test_size"] = int(len(test_idx))
        out["train_test_overlap"] = int(len(set(train_idx).intersection(set(test_idx))))

    return out


def evaluate_raw_coefficients(
    A_train,
    A_test,
    J_train,
    J_test,
    H_train,
    H_test,
    y_train,
    y_test,
    seed: int = 42,
):
    """
    Evaluate bulk signature accessibility from raw coefficient vectors.
    """
    results = {
        "Alexander": eval_multinomial_lr_split(
            A_train, y_train, A_test, y_test, seed=seed
        ),
        "Jones": eval_multinomial_lr_split(
            J_train, y_train, J_test, y_test, seed=seed
        ),
        "HOMFLY": eval_multinomial_lr_split(
            H_train, y_train, H_test, y_test, seed=seed
        ),
    }

    summary = {
        name: {
            "accuracy": res["accuracy"],
            "macro_f1": res["macro_f1"],
            "n_iter": res["n_iter"],
        }
        for name, res in results.items()
    }

    return results, summary


def evaluate_representations(
    representations: dict,
    y_train,
    y_test,
    seed: int = 42,
):
    """
    Generic helper for evaluating multiple learned or linear representations.

    Parameters
    ----------
    representations:
        Dictionary of the form:
        {
            "Jones_PCA": {"train": Z_train, "test": Z_test},
            "Jones_AE": {"train": Z_train, "test": Z_test},
            ...
        }
    """
    rows = []

    for name, Z in representations.items():
        res = eval_multinomial_lr_split(
            Z["train"],
            y_train,
            Z["test"],
            y_test,
            seed=seed,
        )

        rows.append(
            {
                "representation": name,
                "accuracy": res["accuracy"],
                "macro_f1": res["macro_f1"],
                "n_iter": res["n_iter"],
            }
        )

    return rows
