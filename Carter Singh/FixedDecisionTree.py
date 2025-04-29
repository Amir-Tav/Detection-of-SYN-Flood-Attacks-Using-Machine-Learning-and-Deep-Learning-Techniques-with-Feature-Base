import pandas as pd
import time
import psutil
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("D:\Coding Projects\Detection-of-SYN-Flood-Attacks-Using-Machine-Learning-and-Deep-Learning-Techniques-with-Feature-Base\Data\K5_Dataset.csv")

# Extract features and target
X = df.drop(columns=['Label', 'Fold'])
y = df['Label'].astype(int)

# Unique folds (assumed values are 0-4)
folds = df['Fold'].unique()

# Metrics storage
accuracies = []
conf_matrix_total = np.array([[0, 0], [0, 0]])
roc_curves = []
pr_curves = []

mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)




# === Overall Start Metrics ===



# === Start Resource Metrics ===
process = psutil.Process(os.getpid())
start_time = time.time()
start_ram = process.memory_info().rss / (1024 ** 2)  # RAM in MB
start_cpu = psutil.cpu_percent(interval=1)

# Cross-validation using Fold column
for fold in folds:
    print(f"\nTraining on Fold {fold}...")

    # Split train/test based on current fold
    train_data = df[df['Fold'] != fold]
    test_data = df[df['Fold'] == fold]

    X_train = train_data.drop(columns=['Label', 'Fold'])
    y_train = train_data['Label'].astype(int)
    X_test = test_data.drop(columns=['Label', 'Fold'])
    y_test = test_data['Label'].astype(int)

    # Initialize and train model
    clf = DecisionTreeClassifier(random_state=22)
    clf.fit(X_train, y_train)

    # Predictions and probabilities
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Accuracy
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    conf_matrix_total += cm

    # Print metrics for current fold
    print(f"Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
    print("Confusion Matrix:\n", cm)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    roc_curves.append(interp_tpr)

    # Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_pred_proba)
    interp_prec = np.interp(mean_recall, recall_vals[::-1], precision_vals[::-1])
    pr_curves.append(interp_prec)

# === After K-Fold Loop Metrics ===


# === End Resource Metrics ===
end_time = time.time()
end_ram = process.memory_info().rss / (1024 ** 2)  # RAM in MB
end_cpu = psutil.cpu_percent(interval=1)

# === Differences ===
time_taken = end_time - start_time
ram_used = end_ram - start_ram
cpu_diff = end_cpu - start_cpu

print("\n--- Resource Usage Summary ---")
print(f"Time taken: {time_taken:.2f} seconds")
print(f"RAM used: {ram_used:.2f} MB")
print(f"CPU usage change: {cpu_diff:.2f}%")

# === Plotting Results ===

# 1. Average ROC Curve
mean_tpr = np.mean(roc_curves, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='darkorange', label=f'Avg ROC (AUC = {mean_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Average ROC Curve (Fold-Based)')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# 2. Total Confusion Matrix
plt.imshow(conf_matrix_total, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Total Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Benign', 'Attack'])
plt.yticks(tick_marks, ['Benign', 'Attack'])

thresh = conf_matrix_total.max() / 2.
for i in range(conf_matrix_total.shape[0]):
    for j in range(conf_matrix_total.shape[1]):
        plt.text(j, i, str(conf_matrix_total[i, j]),
                 ha='center', va='center',
                 color='white' if conf_matrix_total[i, j] > thresh else 'black')

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# 3. Average Precision-Recall Curve
mean_prec = np.mean(pr_curves, axis=0)

plt.plot(mean_recall, mean_prec, color='blue', label='Avg PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Average Precision-Recall Curve (Fold-Based)')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# 4. Accuracy Summary Only
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")

# 5. Decision Tree Structure (last fold model)
plt.figure(figsize=(14, 7))
plot_tree(clf, filled=True, feature_names=X.columns, class_names=['Benign', 'Attack'], fontsize=8)
plt.title('Decision Tree Structure (Last Fold)')
plt.tight_layout()
plt.show()
