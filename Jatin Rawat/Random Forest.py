"""
Random‑Forest Cross‑Validation Pipeline (Low‑RAM Streaming Version)
------------------------------------------------------------------
* Streams the CSV – never loads whole dataset into RAM.
* Incremental `StandardScaler.partial_fit` for feature scaling.
* Tracks **execution time, RAM, and CPU usage** just like your original notebook.
* Produces the same validation artefacts (per‑fold metrics **and** combined confusion matrix).
"""

import gc
import os
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
import psutil

# ────────────────────────────────────────────────────────────────────────────────
# Helper: iterate over the CSV in chunks
# ────────────────────────────────────────────────────────────────────────────────

def chunk_reader(file_path: str | Path, chunk_size: int = 50_000):
    """Yield DataFrame chunks from a CSV without loading it all into memory."""
    for chunk in pd.read_csv(file_path, chunksize=chunk_size, low_memory=False):
        yield chunk

# ────────────────────────────────────────────────────────────────────────────────
# Quick peek – get feature columns & a rough idea of folds / labels
# ────────────────────────────────────────────────────────────────────────────────

def get_basic_info(file_path: str, sample_rows: int = 1_000) -> Tuple[List[str], List[int]]:
    first_chunk = next(chunk_reader(file_path, chunk_size=sample_rows))
    feature_cols = [c for c in first_chunk.columns if c not in ("Label", "Fold")]
    initial_folds = list(first_chunk["Fold"].unique())
    return feature_cols, initial_folds

# ────────────────────────────────────────────────────────────────────────────────
# Single scan to count rows per fold & label
# ────────────────────────────────────────────────────────────────────────────────

def scan_csv_for_stats(file_path: str) -> Tuple[dict, dict]:
    fold_counts: dict[int, int] = {}
    label_counts: dict[int, int] = {}

    for chunk in chunk_reader(file_path, chunk_size=100_000):
        for fold, cnt in chunk["Fold"].value_counts().items():
            fold_counts[fold] = fold_counts.get(fold, 0) + cnt
        for lbl, cnt in chunk["Label"].value_counts().items():
            label_counts[lbl] = label_counts.get(lbl, 0) + cnt
    return fold_counts, label_counts

# ────────────────────────────────────────────────────────────────────────────────
# Process a single fold
# ────────────────────────────────────────────────────────────────────────────────

def process_fold(
    file_path: str,
    fold_id: int,
    feature_cols: list[str],
    chunk_size: int = 50_000,
    n_estimators: int = 100,
):
    print(f"\n————  Fold {fold_id}  ————")
    fold_start = time.time()

    scaler = StandardScaler()

    train_feats: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    test_feats: list[np.ndarray] = []
    test_labels: list[np.ndarray] = []

    # First pass – fit scaler & collect raw data
    for chunk in chunk_reader(file_path, chunk_size):
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

    # Free the raw lists
    del train_feats, train_labels, test_feats, test_labels
    gc.collect()

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"Accuracy: {acc:.4f} | fold time: {time.time() - fold_start:.2f}s")

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # ROC‑AUC for binary problems
    if len(np.unique(y_test)) == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        print(f"ROC‑AUC: {roc_auc:.4f}")

    # Clean up heavy arrays
    del X_train, X_test, y_train, y_test, y_pred
    gc.collect()

    return acc, cm

# ────────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────────

def main():
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        DATA_PATH = "/content/drive/MyDrive/K5_Dataset.csv"
    except Exception:
        DATA_PATH = "D:\Coding Projects\Detection-of-SYN-Flood-Attacks-Using-Machine-Learning-and-Deep-Learning-Techniques-with-Feature-Base\Data\K5_Dataset.csv"

    DATA_PATH = str(DATA_PATH)

    overall_start = time.time()

    feat_cols, _ = get_basic_info(DATA_PATH)
    fold_counts, label_counts = scan_csv_for_stats(DATA_PATH)
    fold_ids = sorted(fold_counts)

    print("\nDataset overview:")
    print(f" • Features         : {len(feat_cols)}")
    print(f" • Rows per fold    : {fold_counts}")
    print(f" • Label distribution: {label_counts}\n")

    fold_accs: list[float] = []
    combined_cm: np.ndarray | None = None

    for fid in fold_ids:
        acc, cm = process_fold(DATA_PATH, fid, feat_cols)
        fold_accs.append(acc)
        combined_cm = cm if combined_cm is None else combined_cm + cm
        gc.collect()

    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs)

    print("\n=====  Cross‑Validation Summary  =====")
    print(f"Mean Accuracy  : {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Per‑Fold Acc   : {[f'{a:.4f}' for a in fold_accs]}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(combined_cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Combined Confusion Matrix (All Folds)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.show()

    # ── Resource usage summary ──
    total_time = time.time() - overall_start
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    cpu_pct = psutil.cpu_percent(interval=0.1)

    box = (
        "\n┌" + "─" * 45 + "┐\n" +
        "│{:^45s}│\n".format("Overall Training Stats") +
        "├" + "─" * 45 + "┤\n" +
        "│ Total Training Time: {:>8.2f} seconds │\n".format(total_time) +
        "│ Total RAM Usage   : {:>8.2f} MB      │\n".format(mem_mb) +
        "│ CPU Usage         : {:>7.1f}%          │\n".format(cpu_pct) +
        "└" + "─" * 45 + "┘"
    )
    print(box)


if __name__ == "__main__":
    main()
