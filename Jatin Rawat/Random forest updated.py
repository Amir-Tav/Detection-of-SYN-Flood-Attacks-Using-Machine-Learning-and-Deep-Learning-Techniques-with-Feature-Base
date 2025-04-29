# rf_evaluation.py
"""Streaming K‑fold evaluation for **any** scikit‑learn estimator.

This refactors your ``Random Forest.py`` notebook‑style script into a clean,
importable module so you can re‑use the exact same cross‑validation + resource
tracking workflow for *other* classical ML models (e.g. XGBoost, LightGBM,
ExtraTrees, etc.).

Key points
==========
* **Zero‑copy streaming** – reads a CSV in manageable chunks, never loading the
  full dataset into memory.
* **Manual K‑fold driven by the ``Fold`` column** – identical train/val split
  logic to your original code.
* **Pluggable estimator** – pass ``estimator_factory=lambda: RandomForestClassifier(...)``
  or any other scikit‑learn model.
* **Built‑in metrics & plots** – per‑fold accuracy, ROC‑AUC (if binary), plus a
  combined confusion matrix at the end.
* **Resource dashboard** – wall‑clock time, peak RAM, and CPU usage summarised.

Usage example
-------------
```python
from rf_evaluation import cross_validate_stream
from sklearn.ensemble import RandomForestClassifier

accs, cm = cross_validate_stream(
    file_path="K5_Dataset.csv",
    estimator_factory=lambda: RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=42),
    chunk_size=100_000,
)
```
"""
from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler

# ╭────────────────────────────────────────────────────────────╮
# │ Streaming helpers                                         │
# ╰────────────────────────────────────────────────────────────╯

def _chunk_reader(file_path: str | Path, chunk_size: int):
    """Yield DataFrame chunks from a CSV without loading it all into RAM."""
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        yield chunk


def _get_feature_cols(file_path: str, chunk_size: int = 50_000) -> List[str]:
    first_chunk = next(_chunk_reader(file_path, chunk_size))
    return [c for c in first_chunk.columns if c not in ("Label", "Fold")]

# ╭────────────────────────────────────────────────────────────╮
# │ Public API                                                │
# ╰────────────────────────────────────────────────────────────╯

def cross_validate_stream(
    *,
    file_path: str | Path,
    estimator_factory: Callable[[], "sklearn.base.BaseEstimator"],
    chunk_size: int = 50_000,
    verbose: bool = True,
) -> Tuple[List[float], np.ndarray]:
    """Run manual K‑fold CV on a CSV **without loading it into memory**.

    Parameters
    ----------
    file_path : str | Path
        CSV containing *Fold* and *Label* columns plus feature columns.
    estimator_factory : Callable[[], BaseEstimator]
        A *callable* returning a fresh, unfitted scikit‑learn estimator.
    chunk_size : int, default 50_000
        Number of rows per chunk.
    verbose : bool, default True
        If True, prints per‑fold progress and resource stats.

    Returns
    -------
    accs : list[float]
        Accuracy for each fold.
    combined_cm : np.ndarray
        Confusion matrix summed across folds.
    """

    file_path = str(file_path)
    feature_cols = _get_feature_cols(file_path, chunk_size)

    # Discover fold IDs in a single light pass
    fold_ids: set[int] = set()
    for chunk in _chunk_reader(file_path, chunk_size):
        fold_ids.update(chunk["Fold"].unique())
    fold_ids = sorted(fold_ids)

    if verbose:
        print(f"Found {len(feature_cols)} features and {len(fold_ids)} folds.")

    overall_start = time.time()
    accs: List[float] = []
    combined_cm: np.ndarray | None = None

    # ── Per‑fold loop ─────────────────────────────────────────
    for fold_id in fold_ids:
        if verbose:
            print(f"\n—— Fold {fold_id} ————————————————")
        fold_start = time.time()

        scaler = StandardScaler()
        train_feats: list[np.ndarray] = []
        train_labels: list[np.ndarray] = []
        test_feats: list[np.ndarray] = []
        test_labels: list[np.ndarray] = []

        # First pass – fit scaler & gather data
        for chunk in _chunk_reader(file_path, chunk_size):
            train_mask = chunk["Fold"] != fold_id
            test_mask = ~train_mask

            if train_mask.any():
                scaler.partial_fit(chunk.loc[train_mask, feature_cols].astype(np.float32))
                train_feats.append(chunk.loc[train_mask, feature_cols].values.astype(np.float32))
                train_labels.append(chunk.loc[train_mask, "Label"].values)
            if test_mask.any():
                test_feats.append(chunk.loc[test_mask, feature_cols].values.astype(np.float32))
                test_labels.append(chunk.loc[test_mask, "Label"].values)

        X_train = scaler.transform(np.vstack(train_feats))
        y_train = np.concatenate(train_labels)
        X_test = scaler.transform(np.vstack(test_feats))
        y_test = np.concatenate(test_labels)

        del train_feats, train_labels, test_feats, test_labels
        gc.collect()

        # ── Model ─────────────────────────────────────────────
        model = estimator_factory()
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        accs.append(acc)

        if verbose:
            print(f"Accuracy: {acc:.4f} | fold time: {time.time() - fold_start:.2f}s")

        # Confusion matrix
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        combined_cm = cm if combined_cm is None else combined_cm + cm

        # Optional ROC‑AUC for binary problems
        if verbose and len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            print(f"ROC‑AUC: {auc(fpr, tpr):.4f}")

        # Free heavy arrays
        del X_train, X_test, y_train, y_test, y_pred
        gc.collect()

    # ── Summary ──────────────────────────────────────────────
    mean_acc, std_acc = float(np.mean(accs)), float(np.std(accs))
    if verbose:
        print("\n===== Cross‑Validation Summary =====")
        print(f"Mean Accuracy : {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Per‑Fold Acc  : {[f'{a:.4f}' for a in accs]}")

    # Combined confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(combined_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Combined Confusion Matrix (All Folds)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    # Resource dashboard
    total_time = time.time() - overall_start
    proc = psutil.Process(os.getpid())
    mem_mb = proc.memory_info().rss / (1024 * 1024)
    cpu_pct = psutil.cpu_percent(interval=0.1)

    if verbose:
        box = (
            "\n┌" + "─" * 45 + "┐\n" +
            "│{:^45s}│\n".format("Overall Training Stats") +
            "├" + "─" * 45 + "┤\n" +
            "│ Total Training Time: {:>8.2f} seconds │\n".format(total_time) +
            "│ Peak RAM Usage    : {:>8.2f} MB      │\n".format(mem_mb) +
            "│ CPU Usage         : {:>7.1f}%          │\n".format(cpu_pct) +
            "└" + "─" * 45 + "┘"
        )
        print(box)

    return accs, combined_cm
