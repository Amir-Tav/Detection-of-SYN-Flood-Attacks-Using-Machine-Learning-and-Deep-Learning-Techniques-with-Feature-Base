import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import psutil

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# ──────────────────────────────────────────────────────────────
# Load dataset
# ──────────────────────────────────────────────────────────────
DATA_PATH = "D:/Coding Projects/Detection-of-SYN-Flood-Attacks-Using-Machine-Learning-and-Deep-Learning-Techniques-with-Feature-Base/Data/K5_Dataset.csv"
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Label", "Fold"], errors='ignore')
y = df["Label"].values

# ──────────────────────────────────────────────────────────────
# Setup monitoring
# ──────────────────────────────────────────────────────────────
start_time = time.time()
process = psutil.Process(os.getpid())

# ──────────────────────────────────────────────────────────────
# Cross-validation setup
# ──────────────────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scaler = StandardScaler()

all_y_true = []
all_y_pred = []
all_y_scores = []
fold_accuracies = []
combined_cm = None

for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    print(f"\n—— Fold {fold_idx} ——")

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)

    acc = model.score(X_test_scaled, y_test)
    print(f"Accuracy: {acc:.4f}")
    fold_accuracies.append(acc)

    y_pred = model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred)
    combined_cm = cm if combined_cm is None else combined_cm + cm

    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

    if len(np.unique(y)) == 2:
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        all_y_scores.extend(y_proba)

# ──────────────────────────────────────────────────────────────
# Results
# ──────────────────────────────────────────────────────────────
print("\n===== Cross-Validation Summary =====")
print(f"Mean Accuracy : {np.mean(fold_accuracies):.4f}")
print(f"Per-Fold Acc  : {[f'{a:.4f}' for a in fold_accuracies]}")

# Confusion Matrix
disp_cm = ConfusionMatrixDisplay(confusion_matrix=combined_cm)
disp_cm.plot(cmap="Blues")
plt.title("Confusion Matrix (All Folds)")
plt.grid(False)
plt.show()

# ROC Curve
if all_y_scores:
    fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
    roc_auc = auc(fpr, tpr)
    RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
    plt.title(f"ROC Curve (AUC = {roc_auc:.4f}) - All Folds")
    plt.grid(True)
    plt.show()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(all_y_true, all_y_scores)
    PrecisionRecallDisplay(precision=precision, recall=recall).plot()
    plt.title("Precision-Recall Curve (All Folds)")
    plt.grid(True)
    plt.show()

# ──────────────────────────────────────────────────────────────
# Resource Usage Summary
# ──────────────────────────────────────────────────────────────
end_time = time.time()
total_time = end_time - start_time
mem_usage = process.memory_info().rss / (1024 ** 2)  # MB
cpu_percent = psutil.cpu_percent(interval=0.1)

box = (
    "\n┌" + "─" * 45 + "┐\n" +
    "│{:^45s}│\n".format("Overall Training Stats") +
    "├" + "─" * 45 + "┤\n" +
    "│ Total Training Time: {:>8.2f} seconds │\n".format(total_time) +
    "│ Total RAM Usage   : {:>8.2f} MB      │\n".format(mem_usage) +
    "│ CPU Usage         : {:>7.1f}%          │\n".format(cpu_percent) +
    "└" + "─" * 45 + "┘"
)
print(box)
