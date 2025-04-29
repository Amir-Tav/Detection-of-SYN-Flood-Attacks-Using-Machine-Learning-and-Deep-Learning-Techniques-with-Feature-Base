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
from scipy.interpolate import interp1d
from itertools import cycle
import time
import psutil  # For CPU and RAM usage monitoring
import os

# Apply basic TensorFlow optimizations without causing compatibility issues
def optimize_tensorflow():
    """Apply safer TensorFlow optimizations."""
    try:
        # Set random seed for reproducibility
        tf.random.set_seed(42)
        np.random.seed(42)
        print("Random seeds set to 42")
    except Exception as e:
        print(f"Could not set random seeds: {e}")
    
    try:
        # Configure memory growth - safer approach
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Found {len(gpus)} GPU(s) and enabled memory growth")
        else:
            print("No GPUs found, running on CPU")
    except Exception as e:
        print(f"Device optimization failed: {e}")
    
    # Avoid using advanced optimizations that might cause compatibility issues
    print("Applied safe TensorFlow optimizations")
    return

# Function to get current resource usage
def get_resource_usage():
    """Get current CPU and memory usage."""
    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
    return cpu_percent, memory_mb

# Create a more compatible CNN model
def create_dense_model(input_shape, num_classes=2):
    """
    Creates a simple Dense (fully connected) model without Conv1D layers
    to avoid the NHWC tensor format issues
    """
    # Extract the feature dimension from the input shape
    features = input_shape[0]
    
    model = Sequential([
        # Flatten the input first (no Conv1D layers)
        Flatten(input_shape=input_shape),
        
        # First dense layer
        Dense(128, activation='relu'),
        Dropout(0.3),
        
        # Second dense layer
        Dense(64, activation='relu'),
        Dropout(0.3),
        
        # Output layer
        Dense(num_classes, activation='softmax')
    ])

    # Compile with standard settings
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# Class to monitor and log resource usage
class ResourceMonitor:
    def __init__(self, log_interval=5):
        self.log_interval = log_interval  # seconds between resource checks
        self.cpu_usage = []
        self.memory_usage = []
        self.timestamps = []
        self.start_time = None
        self.is_monitoring = False
        self.peak_memory = 0
        self.avg_cpu = 0
        self.final_cpu = 0
        self.final_memory = 0
    
    def start(self):
        """Start monitoring resources."""
        self.start_time = time.time()
        self.is_monitoring = True
        self.cpu_usage = []
        self.memory_usage = []
        self.timestamps = []
        self.peak_memory = 0
        self._monitor()
    
    def stop(self):
        """Stop monitoring resources."""
        self.is_monitoring = False
        
        # Calculate statistics
        if self.memory_usage:
            self.peak_memory = max(self.memory_usage)
            self.avg_cpu = sum(self.cpu_usage) / len(self.cpu_usage) if self.cpu_usage else 0
            self.final_cpu = self.cpu_usage[-1] if self.cpu_usage else 0
            self.final_memory = self.memory_usage[-1] if self.memory_usage else 0
    
    def _monitor(self):
        """Internal monitoring function."""
        if not self.is_monitoring:
            return
        
        # Get current resource usage
        cpu, memory = get_resource_usage()
        elapsed = time.time() - self.start_time
        
        # Store data
        self.cpu_usage.append(cpu)
        self.memory_usage.append(memory)
        self.timestamps.append(elapsed)
        
        # Schedule next check
        import threading
        threading.Timer(self.log_interval, self._monitor).start()
    
    def plot_usage(self):
        """Plot resource usage over time - disabled."""
        # Resource usage plots are disabled
        pass
        
    def print_summary(self):
        """Print a summary of resource usage."""
        print("\n===== RESOURCE USAGE SUMMARY =====")
        print(f"Peak Memory Usage: {self.peak_memory:.2f} MB")
        print(f"Average CPU Usage: {self.avg_cpu:.2f}%")
        print(f"Final CPU Usage: {self.final_cpu:.2f}%")
        print(f"Final Memory Usage: {self.final_memory:.2f} MB")
        print("=================================")

# Main function for the k-fold pipeline with compatibility focus
def main():
    # Create resource monitor
    resource_monitor = ResourceMonitor(log_interval=2)  # Check every 2 seconds
    
    # Start timer for overall execution
    total_start_time = time.time()
    
    # Start resource monitoring
    resource_monitor.start()
    
    print("Loading dataset...")
    data_load_start = time.time()
    
    # Try to determine if using Colab
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        DATA_PATH = "/content/drive/MyDrive/K5_Dataset.csv"
    except:
        # If not using Colab, specify your local path
        DATA_PATH = "D:\Coding Projects\Detection-of-SYN-Flood-Attacks-Using-Machine-Learning-and-Deep-Learning-Techniques-with-Feature-Base\Data\K5_Dataset.csv"
        
    print(f"TensorFlow version: {tf.__version__}")
    print("Available devices:")
    for device in tf.config.list_physical_devices():
        print(f" - {device.name}: {device.device_type}")
    
    # Load the dataset
    df = pd.read_csv(DATA_PATH, low_memory=False)
    
    data_load_time = time.time() - data_load_start
    print(f"Dataset loaded in {data_load_time:.2f} seconds")

    # Print basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['Label'].value_counts().sort_index()}")
    print(f"Fold distribution:\n{df['Fold'].value_counts().sort_index()}")

    # Get initial resource usage
    initial_cpu, initial_memory = get_resource_usage()
    print(f"Initial CPU Usage: {initial_cpu:.2f}%")
    print(f"Initial Memory Usage: {initial_memory:.2f} MB")

    # Initialize metrics collection
    all_accuracies = []
    all_losses = []
    fold_indices = sorted(df['Fold'].unique())
    class_labels = sorted(df['Label'].unique())
    num_classes = len(class_labels)
    
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
    
    # Track fold times
    fold_times = []
    
    print(f"\nStarting {len(fold_indices)}-fold cross-validation using predefined folds...")
    
    # Prepare feature columns by removing non-feature columns
    feature_cols = [col for col in df.columns if col not in ['Label', 'Fold']]
    
    # Create a feature dataframe once instead of subsetting repeatedly
    features_df = df[feature_cols]
    
    # Loop through each fold for cross-validation
    for fold in fold_indices:
        # Check resource usage before fold processing
        fold_start_cpu, fold_start_memory = get_resource_usage()
        
        fold_start_time = time.time()
        print(f"\n===== Processing Fold {fold} =====")
        print(f"CPU Usage at start of fold: {fold_start_cpu:.2f}%")
        print(f"Memory Usage at start of fold: {fold_start_memory:.2f} MB")
        
        # Split data into train and test sets
        train_mask = df['Fold'] != fold
        test_mask = ~train_mask
        
        # Extract features and labels
        X_train = features_df[train_mask].values
        y_train = df.loc[train_mask, 'Label'].values
        X_test = features_df[test_mask].values
        y_test = df.loc[test_mask, 'Label'].values
        
        # Feature scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Reshape data for CNN (samples, features, 1)
        X_train_reshaped = X_train_scaled.reshape(X_train_scaled.shape[0], X_train_scaled.shape[1], 1)
        X_test_reshaped = X_test_scaled.reshape(X_test_scaled.shape[0], X_test_scaled.shape[1], 1)
        
        print(f"Training set shape: {X_train_reshaped.shape}")
        print(f"Test set shape: {X_test_reshaped.shape}")
        
        # Create and compile the model
        model_start_time = time.time()
        
        # Use CPU instead of GPU if there are compatibility issues
        with tf.device('/CPU:0'):
            model = create_dense_model(input_shape=(X_train_scaled.shape[1], 1), num_classes=num_classes)
            
        print(f"Model created in {time.time() - model_start_time:.2f} seconds")
        
        # Define callbacks for training
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)
        ]
        
        # Check resource usage before training
        pre_train_cpu, pre_train_memory = get_resource_usage()
        print(f"CPU Usage before training: {pre_train_cpu:.2f}%")
        print(f"Memory Usage before training: {pre_train_memory:.2f} MB")
        
        # Train the model with simpler settings
        training_start_time = time.time()
        print(f"Training model for fold {fold}...")
        
        # Use smaller validation set to speed up training
        val_size = int(0.2 * len(X_train_reshaped))
        X_val = X_train_reshaped[-val_size:]
        y_val = y_train[-val_size:]
        X_train_final = X_train_reshaped[:-val_size]
        y_train_final = y_train[:-val_size:]
        
        # Use smaller batch size and fewer epochs for compatibility
        batch_size = 64
        
        # Train using CPU to avoid compatibility issues
        with tf.device('/CPU:0'):
            history = model.fit(
                X_train_final, y_train_final,
                epochs=10,  # Reduced epochs for faster execution
                validation_data=(X_val, y_val),
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        
        training_time = time.time() - training_start_time
        print(f"Model training completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
        
        # Check resource usage after training
        post_train_cpu, post_train_memory = get_resource_usage()
        print(f"CPU Usage after training: {post_train_cpu:.2f}%")
        print(f"Memory Usage after training: {post_train_memory:.2f} MB")
        print(f"Memory increase during training: {post_train_memory - pre_train_memory:.2f} MB")
        
        # Evaluate on test set
        eval_start_time = time.time()
        
        with tf.device('/CPU:0'):
            test_loss, test_acc = model.evaluate(X_test_reshaped, y_test, batch_size=batch_size, verbose=0)
            
        print(f"Test accuracy for fold {fold}: {test_acc:.4f}")
        print(f"Test loss for fold {fold}: {test_loss:.4f}")
        
        # Store metrics
        all_accuracies.append(test_acc)
        all_losses.append(test_loss)
        
        # Make predictions
        with tf.device('/CPU:0'):
            y_pred_prob = model.predict(X_test_reshaped, verbose=0)
            
        y_pred_classes = np.argmax(y_pred_prob, axis=1)
        eval_time = time.time() - eval_start_time
        print(f"Evaluation completed in {eval_time:.2f} seconds")
        
        # Classification report
        print(f"\nClassification Report for Fold {fold}:")
        print(classification_report(y_test, y_pred_classes))
        
        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred_classes)
        print(f"Confusion Matrix for Fold {fold}:")
        print(cm)
        
        # Update combined confusion matrix
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
        
        # Check resource usage at end of fold
        fold_end_cpu, fold_end_memory = get_resource_usage()
        print(f"CPU Usage at end of fold: {fold_end_cpu:.2f}%")
        print(f"Memory Usage at end of fold: {fold_end_memory:.2f} MB")
        
        # Record total fold time
        fold_time = time.time() - fold_start_time
        fold_times.append(fold_time)
        print(f"Total time for fold {fold}: {fold_time:.2f} seconds ({fold_time/60:.2f} minutes)")
    
    # Stop resource monitoring
    resource_monitor.stop()
    
    # Calculate and display metrics across all folds
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_loss = np.mean(all_losses)
    std_loss = np.std(all_losses)
    
    print("\n===== Cross-Validation Results =====")
    print(f"Mean Accuracy: {mean_accuracy:.4f} (±{std_accuracy:.4f})")
    print(f"Mean Loss: {mean_loss:.4f} (±{std_loss:.4f})")
    print(f"Individual Fold Accuracies: {[f'{acc:.4f}' for acc in all_accuracies]}")
    
    # Plot ROC curves
    plot_start_time = time.time()
    plot_roc_curves(fpr_curves, tpr_curves, roc_auc_values, fold_indices)
    
    # Plot precision-recall curves
    plot_precision_recall_curves(precision_curves, recall_curves, average_precision_values, fold_indices)
    
    # Plot combined confusion matrix
    plot_combined_confusion_matrix(combined_cm, class_labels)
    plot_time = time.time() - plot_start_time
    
    # Get final resource usage
    final_cpu, final_memory = get_resource_usage()
    
    # Display clean time summary
    total_time = time.time() - total_start_time
    print(f"\n===== TIME SUMMARY =====")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Data loading time: {data_load_time:.2f} seconds")
    print(f"Plot generation time: {plot_time:.2f} seconds")
    
    print("\nPer-fold execution times:")
    for i, fold in enumerate(fold_indices):
        print(f"Fold {fold}: {fold_times[i]:.2f} seconds ({fold_times[i]/60:.2f} minutes)")
    
    print(f"Average fold time: {np.mean(fold_times):.2f} seconds ({np.mean(fold_times)/60:.2f} minutes)")
    
    # Print resource usage summary
    resource_monitor.print_summary()
    
    # Print overall training stats in a box (like in the image)
    print("\n┌───────────────────────────────────┐")
    print("│       Overall Training Stats       │")
    print("├───────────────────────────────────┤")
    print(f"│ Total Training Time: {total_time:.2f} seconds  │")
    print(f"│ Total RAM Usage: {resource_monitor.final_memory:.2f} MB      │")
    print(f"│ CPU Usage (at final check): {final_cpu:.1f}%   │")
    print("└───────────────────────────────────┘")
    
    return model

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

# Run the pipeline with safer settings
if __name__ == "__main__":
    # Apply safer optimizations
    optimize_tensorflow()
    
    # Run with timing
    print("==== Starting ML Pipeline with Dense Network ====")
    start_time = time.time()
    try:
        model = main()
        end_time = time.time()
        total_runtime = end_time - start_time
        print("\n==== EXECUTION COMPLETE ====")
        print(f"Total execution time: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")
    except Exception as e:
        print(f"\n==== ERROR ENCOUNTERED ====")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nDetailed tensor error information - trying to help debug the issue:")