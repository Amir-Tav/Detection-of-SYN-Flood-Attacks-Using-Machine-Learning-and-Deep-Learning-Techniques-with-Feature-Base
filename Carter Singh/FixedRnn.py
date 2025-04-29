import pandas as pd
import time
import time
import psutil
import matplotlib.pyplot as plt
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetHandleByIndex, nvmlDeviceGetUtilizationRates
import os
import psutil
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("D:\Coding Projects\Detection-of-SYN-Flood-Attacks-Using-Machine-Learning-and-Deep-Learning-Techniques-with-Feature-Base\Data\K5_Dataset.csv")

# Separate features and target
X = df.drop(columns=['Label', 'Fold'])
y = df['Label'].astype(int)

# Use 'Fold' column for defining the folds
folds = df['Fold'].unique()

# Metrics storage
accuracies = []
conf_matrix_total = np.array([[0, 0], [0, 0]])
roc_curves = []
pr_curves = []

mean_fpr = np.linspace(0, 1, 100)
mean_recall = np.linspace(0, 1, 100)

# === Start Resource Metrics ===
process = psutil.Process(os.getpid())
start_time = time.time()
start_ram = process.memory_info().rss / (1024 ** 2)  # RAM in MB
start_cpu = psutil.cpu_percent(interval=1)


# Cross-validation loop using the 'Fold' column
for fold in folds:
    print(f"Training on fold {fold + 1}...")

    train_data = df[df['Fold'] != fold]
    test_data = df[df['Fold'] == fold]

    X_train = train_data.drop(columns=['Label', 'Fold'])
    y_train = train_data['Label'].astype(int)
    X_test = test_data.drop(columns=['Label', 'Fold'])
    y_test = test_data['Label'].astype(int)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Reshape for RNN
    X_train_scaled = np.expand_dims(X_train_scaled, axis=1)
    X_test_scaled = np.expand_dims(X_test_scaled, axis=1)

    y_train_encoded = to_categorical(y_train)
    y_test_encoded = to_categorical(y_test)

    # Build RNN model
    model = Sequential()
    model.add(SimpleRNN(50, activation='tanh', input_shape=(1, X_train_scaled.shape[2])))
    model.add(Dense(2, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train_scaled, y_train_encoded, epochs=10, batch_size=32,
              validation_data=(X_test_scaled, y_test_encoded), verbose=0)

    y_pred_probs = model.predict(X_test_scaled)
    y_pred_labels = np.argmax(y_pred_probs, axis=1)
    y_test_labels = np.argmax(y_test_encoded, axis=1)

    # Metrics
    acc = accuracy_score(y_test_labels, y_pred_labels)
    accuracies.append(acc)

    print(f"Fold {fold + 1} Accuracy: {acc:.4f}")
    print("Classification Report:\n", classification_report(y_test_labels, y_pred_labels))

    cm = confusion_matrix(y_test_labels, y_pred_labels)
    conf_matrix_total += cm
    print("Confusion Matrix:\n", cm)

    # ROC
    fpr, tpr, _ = roc_curve(y_test_labels, y_pred_probs[:, 1])
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    roc_curves.append(interp_tpr)

    # PR
    precision_vals, recall_vals, _ = precision_recall_curve(y_test_labels, y_pred_probs[:, 1])
    interp_prec = np.interp(mean_recall, recall_vals[::-1], precision_vals[::-1])
    pr_curves.append(interp_prec)


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





# ROC Curve (Average)
mean_tpr = np.mean(roc_curves, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

plt.figure(figsize=(6, 5))
plt.plot(mean_fpr, mean_tpr, label=f'Avg ROC Curve (AUC = {mean_auc:.2f})', color='darkorange')
plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.title('Average ROC Curve (5-Fold)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Confusion Matrix (Total)
plt.figure(figsize=(6, 5))
plt.imshow(conf_matrix_total, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Total Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Benign', 'Attack'])
plt.yticks(tick_marks, ['Benign', 'Attack'])

thresh = conf_matrix_total.max() / 2.
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix_total[i, j]),
                 ha='center', va='center',
                 color='white' if conf_matrix_total[i, j] > thresh else 'black')

plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.show()

# Precision-Recall Curve (Average)
mean_prec = np.mean(pr_curves, axis=0)

plt.figure(figsize=(6, 5))
plt.plot(mean_recall, mean_prec, color='blue', label='Avg Precision-Recall Curve')
plt.title('Average Precision-Recall Curve (5-Fold)')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# Accuracy Summary
print(f"\nAverage Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
