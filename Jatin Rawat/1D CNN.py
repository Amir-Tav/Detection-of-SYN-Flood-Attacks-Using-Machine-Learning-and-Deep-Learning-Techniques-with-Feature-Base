import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d  # Correct import for interpolation
from itertools import cycle
import time

# Mount Google Drive if using Colab (comment out if not using Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_PATH = "/content/drive/MyDrive/K5_Dataset.csv"
except:
    # If not using Colab, specify your local path
    DATA_PATH = "K5_Dataset.csv"

# Function to create CNN model
def create_cnn_model(input_shape, num_classes=2):
    """
    Creates a 1D CNN model suitable for network flow classification

    Args:
        input_shape: The shape of the input data (features, 1)
        num_classes: Number of output classes

    Returns:
        A compiled Keras CNN model
    """
    model = Sequential([
        # First Conv block
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),

        # Second Conv block
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(filters=128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),

        # Third Conv block
        Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        Conv1D(filters=256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.4),

        # Fully connected layers
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),

        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Function to plot training history
def plot_history(history, fold):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title(f'Model Accuracy (Fold {fold})')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')

    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title(f'Model Loss (Fold {fold})')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.show()

# Function to plot ROC curves
def plot_roc_curves(fpr, tpr, roc_auc, fold_indices):
    plt.figure(figsize=(10, 8))
    
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    
    for i, color in zip(range(len(fold_indices)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'Fold {fold_indices[i]} (AUC = {roc_auc[i]:.2f})')
    
    # Plot mean ROC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(fold_indices))]))
    mean_tpr = np.zeros_like(all_fpr)
    
    for i in range(len(fold_indices)):
        # Use interp1d for interpolation
        if len(fpr[i]) > 1:  # Make sure we have enough points to interpolate
            interp_func = interp1d(fpr[i], tpr[i], kind='linear', bounds_error=False, fill_value=(0, 1))
            mean_tpr += interp_func(all_fpr)
        else:
            # Handle edge case where we don't have enough points
            mean_tpr += np.zeros_like(all_fpr)
    
    mean_tpr /= len(fold_indices)
    mean_auc = auc(all_fpr, mean_tpr)
    
    plt.plot(all_fpr, mean_tpr, color='black', lw=2, linestyle='--',
             label=f'Mean ROC (AUC = {mean_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Function to plot precision-recall curves
def plot_precision_recall_curves(precision, recall, avg_precision, fold_indices):
    plt.figure(figsize=(10, 8))
    
    colors = cycle(['blue', 'red', 'green', 'orange', 'purple'])
    
    for i, color in zip(range(len(fold_indices)), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label=f'Fold {fold_indices[i]} (AP = {avg_precision[i]:.2f})')
    
    # Calculate mean precision-recall curve using proper interpolation
    mean_recall = np.linspace(0, 1, 100)
    mean_precision = np.zeros_like(mean_recall)
    
    for i in range(len(fold_indices)):
        if len(recall[i]) > 1:  # Make sure we have enough points to interpolate
            # Note: For PR curves, we need to flip coordinates because recall might not be monotonic
            interp_func = interp1d(recall[i][::-1], precision[i][::-1], 
                                 kind='linear', bounds_error=False, fill_value=(0, precision[i][0]))
            mean_precision += interp_func(mean_recall)
        else:
            # Handle edge case
            mean_precision += np.zeros_like(mean_recall)
    
    mean_precision /= len(fold_indices)
    mean_ap = np.mean(avg_precision)
    
    plt.plot(mean_recall, mean_precision, color='black', lw=2, linestyle='--',
             label=f'Mean PR (AP = {mean_ap:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc="lower left")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Function to plot combined confusion matrix
def plot_combined_confusion_matrix(cm_sum, classes):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_sum, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Combined Confusion Matrix (All Folds)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Main function for the entire k-fold pipeline
def main():
    print("Loading dataset...")
    # Load the dataset
    df = pd.read_csv(DATA_PATH)

    # Print basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['Label'].value_counts().sort_index()}")
    print(f"Fold distribution:\n{df['Fold'].value_counts().sort_index()}")

    # Initialize metrics collection
    all_accuracies = []
    all_losses = []
    fold_indices = sorted(df['Fold'].unique())
    
    # For ROC curves
    fpr_curves = []
    tpr_curves = []
    roc_auc_values = []
    
    # For precision-recall curves
    precision_curves = []
    recall_curves = []
    average_precision_values = []
    
    # For confusion matrix
    combined_cm = None
    class_labels = sorted(df['Label'].unique())
    num_classes = len(class_labels)
    
    # Start time for performance tracking
    start_time = time.time()
    
    print(f"\nStarting 5-fold cross-validation using predefined folds (0-4)...")
    
    # Prepare feature columns by removing non-feature columns
    feature_cols = [col for col in df.columns if col not in ['Label', 'Fold']]
    
    # Loop through each fold for cross-validation
    for fold in fold_indices:
        print(f"\n===== Processing Fold {fold} =====")
        
        # Split data based on fold
        train_df = df[df['Fold'] != fold]
        test_df = df[df['Fold'] == fold]
        
        # Extract features and labels
        X_train = train_df[feature_cols].values
        y_train = train_df['Label'].values
        X_test = test_df[feature_cols].values
        y_test = test_df['Label'].values
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Reshape data for CNN (samples, features, 1)
        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        
        print(f"Training set shape: {X_train_reshaped.shape}")
        print(f"Test set shape: {X_test_reshaped.shape}")
        print(f"Class distribution in training set: {np.bincount(y_train)}")
        print(f"Class distribution in test set: {np.bincount(y_test)}")
        
        # Create and compile the model
        model = create_cnn_model(input_shape=(X_train_scaled.shape[1], 1), num_classes=num_classes)
        
        # Define callbacks for training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001),
            ModelCheckpoint(f'best_cnn_model_fold_{fold}.h5', save_best_only=True, monitor='val_accuracy')
        ]
        
        # Train the model
        print(f"Training model for fold {fold}...")
        history = model.fit(
            X_train_reshaped, y_train,
            epochs=50,
            batch_size=64,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Plot training history for this fold
        plot_history(history, fold)
        
        # Evaluate on test set
        test_loss, test_acc = model.evaluate(X_test_reshaped, y_test)
        print(f"Test accuracy for fold {fold}: {test_acc:.4f}")
        print(f"Test loss for fold {fold}: {test_loss:.4f}")
        
        # Store metrics
        all_accuracies.append(test_acc)
        all_losses.append(test_loss)
        
        # Make predictions
        y_pred_prob = model.predict(X_test_reshaped)
        y_pred_classes = np.argmax(y_pred_prob, axis=1)
        
        # Classification report
        print(f"\nClassification Report for Fold {fold}:")
        print(classification_report(y_test, y_pred_classes))
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        if combined_cm is None:
            combined_cm = cm
        else:
            combined_cm += cm
        
        # Compute ROC curve and AUC
        if num_classes == 2:
            # For binary classification
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob[:, 1])
            roc_auc = auc(fpr, tpr)
        else:
            # For multi-class, using micro-average approach
            y_test_binary = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)
            fpr, tpr, _ = roc_curve(y_test_binary.ravel(), y_pred_prob.ravel())
            roc_auc = auc(fpr, tpr)
        
        fpr_curves.append(fpr)
        tpr_curves.append(tpr)
        roc_auc_values.append(roc_auc)
        
        # Compute precision-recall curve
        if num_classes == 2:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_prob[:, 1])
            avg_precision = average_precision_score(y_test, y_pred_prob[:, 1])
        else:
            # For multi-class, using micro-average approach
            precision, recall, _ = precision_recall_curve(y_test_binary.ravel(), y_pred_prob.ravel())
            avg_precision = average_precision_score(y_test_binary.ravel(), y_pred_prob.ravel())
        
        precision_curves.append(precision)
        recall_curves.append(recall)
        average_precision_values.append(avg_precision)
        
        # Save the model for this fold
        model.save(f'k5_cnn_model_fold_{fold}.h5')
        print(f"Model saved as 'k5_cnn_model_fold_{fold}.h5'")
    
    # Calculate and display metrics across all folds
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)
    
    print("\n===== Cross-Validation Results =====")
    print(f"Mean Accuracy: {mean_accuracy:.4f} (±{std_accuracy:.4f})")
    print(f"Mean Loss: {mean_loss:.4f} (±{std_loss:.4f})")
    print(f"Individual Fold Accuracies: {[f'{acc:.4f}' for acc in all_accuracies]}")
    print(f"Individual Fold Losses: {[f'{loss:.4f}' for loss in all_losses]}")
    
    # Plot ROC curves
    plot_roc_curves(fpr_curves, tpr_curves, roc_auc_values, fold_indices)
    
    # Plot precision-recall curves
    plot_precision_recall_curves(precision_curves, recall_curves, average_precision_values, fold_indices)
    
    # Plot combined confusion matrix
    plot_combined_confusion_matrix(combined_cm, class_labels)
    
    # Display total execution time
    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time/60:.2f} minutes")
    
    return model

# Run the pipeline
if __name__ == "__main__":
    main()