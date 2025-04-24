import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import joblib
import time

# Mount Google Drive if using Colab (comment out if not using Colab)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_PATH = "/content/drive/MyDrive/K5_Dataset.csv"
except:
    # If not using Colab, specify your local path
    DATA_PATH = "K5_Dataset.csv"

def load_data():
    """
    Load the K5 dataset
    
    Returns:
        DataFrame with the loaded dataset
    """
    print("Loading dataset...")
    # Load the dataset
    df = pd.read_csv(DATA_PATH)
    
    # Print basic info about the dataset
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['Label'].value_counts().sort_index()}")
    print(f"Fold distribution:\n{df['Fold'].value_counts().sort_index()}")
    
    return df

def preprocess_fold_data(df, test_fold):
    """
    Preprocess data for a specific fold
    
    Args:
        df: Full dataset
        test_fold: Fold number to use as test set
        
    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # Split into training and test sets based on fold
    train_df = df[df['Fold'] != test_fold]
    test_df = df[df['Fold'] == test_fold]
    
    # Prepare features and target
    X_train = train_df.drop(['Label', 'Fold'], axis=1)
    y_train = train_df['Label']
    
    X_test = test_df.drop(['Label', 'Fold'], axis=1)
    y_test = test_df['Label']
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set shape for fold {test_fold}: {X_train.shape}")
    print(f"Test set shape for fold {test_fold}: {X_test.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_model(X_train, y_train, hyperparams=None):
    """
    Train a Random Forest model
    
    Args:
        X_train: Training features
        y_train: Training labels
        hyperparams: Optional dictionary of hyperparameters
        
    Returns:
        The trained Random Forest model
    """
    start_time = time.time()
    
    # Create and train the model
    if hyperparams:
        rf_model = RandomForestClassifier(random_state=42, n_jobs=-1, **hyperparams)
    else:
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    
    rf_model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return rf_model

def evaluate_fold(model, X_test, y_test, fold_num):
    """
    Evaluate model performance on test data for a single fold
    
    Args:
        model: Trained Random Forest model
        X_test: Test features
        y_test: Test labels
        fold_num: Current fold number
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Fold {fold_num} - Test accuracy: {accuracy:.4f}")
    
    # Classification report
    print(f"\nFold {fold_num} - Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC curve data
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall curve data
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    avg_precision = average_precision_score(y_test, y_pred_proba)
    
    # Return metrics
    return {
        'fold': fold_num,
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'fpr': fpr,
        'tpr': tpr,
        'roc_auc': roc_auc,
        'precision': precision,
        'recall': recall,
        'avg_precision': avg_precision
    }

def plot_roc_curves(fold_results):
    """
    Plot ROC curves for all folds
    
    Args:
        fold_results: List of result dictionaries from each fold
    """
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve for each fold
    for result in fold_results:
        fold = result['fold']
        fpr = result['fpr']
        tpr = result['tpr']
        roc_auc = result['roc_auc']
        
        plt.plot(
            fpr, tpr, lw=2, alpha=0.7,
            label=f'Fold {fold} (AUC = {roc_auc:.4f})'
        )
    
    # Plot chance line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for 5-Fold Cross-Validation')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    
    # Save the plot before showing it (this helps in some environments)
    plt.savefig('roc_curves.png')
    plt.show()

def plot_combined_confusion_matrix(fold_results):
    """
    Plot combined confusion matrix for all folds
    
    Args:
        fold_results: List of result dictionaries from each fold
    """
    # Combine all confusion matrices
    combined_cm = np.zeros((2, 2))
    for result in fold_results:
        combined_cm += result['confusion_matrix']
    
    # Convert to integers for better display
    combined_cm = combined_cm.astype(int)
    
    # Plot combined confusion matrix
    plt.figure(figsize=(10, 8))
    
    # Use fmt='.0f' instead of fmt='d' to avoid the formatting error
    sns.heatmap(
        combined_cm, annot=True, fmt='.0f', cmap='Blues',
        xticklabels=['Predicted 0', 'Predicted 1'],
        yticklabels=['Actual 0', 'Actual 1']
    )
    plt.title('Combined Confusion Matrix (All Folds)')
    plt.tight_layout()
    
    # Save the plot before showing it
    plt.savefig('confusion_matrix.png')
    plt.show()
    
    # Calculate and print metrics from combined confusion matrix
    tn, fp, fn, tp = combined_cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Combined metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

def plot_precision_recall_curves(fold_results):
    """
    Plot precision-recall curves for all folds
    
    Args:
        fold_results: List of result dictionaries from each fold
    """
    plt.figure(figsize=(10, 8))
    
    # Plot PR curve for each fold
    for result in fold_results:
        fold = result['fold']
        precision = result['precision']
        recall = result['recall']
        avg_precision = result['avg_precision']
        
        plt.plot(
            recall, precision, lw=2, alpha=0.7,
            label=f'Fold {fold} (AP = {avg_precision:.4f})'
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for 5-Fold Cross-Validation')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Save the plot before showing it
    plt.savefig('precision_recall_curves.png')
    plt.show()

def calculate_statistics(fold_results):
    """
    Calculate statistics across all folds
    
    Args:
        fold_results: List of result dictionaries from each fold
    """
    accuracies = [result['accuracy'] for result in fold_results]
    roc_aucs = [result['roc_auc'] for result in fold_results]
    avg_precisions = [result['avg_precision'] for result in fold_results]
    
    # Calculate mean and standard deviation
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    mean_roc_auc = np.mean(roc_aucs)
    std_roc_auc = np.std(roc_aucs)
    
    mean_avg_precision = np.mean(avg_precisions)
    std_avg_precision = np.std(avg_precisions)
    
    print("\nCross-Validation Statistics:")
    print(f"Mean Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"Mean ROC AUC: {mean_roc_auc:.4f} ± {std_roc_auc:.4f}")
    print(f"Mean Average Precision: {mean_avg_precision:.4f} ± {std_avg_precision:.4f}")
    
    print("\nIndividual fold accuracies:")
    for i, acc in enumerate(accuracies):
        print(f"Fold {i}: {acc:.4f}")

def feature_importance(model, feature_names):
    """
    Plot and return feature importance from the Random Forest model
    
    Args:
        model: Trained Random Forest model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importances
    """
    # Get feature importances
    importances = model.feature_importances_
    
    # Create a DataFrame for better visualization
    feature_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp.head(20))
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.show()
    
    return feature_imp

def main():
    """
    Main function to orchestrate the 5-fold cross-validation workflow
    """
    # Load data
    df = load_data()
    
    # Verify we have 5 folds labeled 0-4
    folds = df['Fold'].unique()
    print(f"Found {len(folds)} folds: {sorted(folds)}")
    
    # Store results for each fold
    fold_results = []
    all_models = []
    
    # Get feature names (without Label and Fold)
    feature_names = df.drop(['Label', 'Fold'], axis=1).columns.tolist()
    
    # Perform 5-fold cross-validation
    for fold in sorted(folds):
        print(f"\n{'='*50}")
        print(f"Processing Fold {fold} as test set")
        print(f"{'='*50}")
        
        # Preprocess data for this fold
        X_train, X_test, y_train, y_test, scaler = preprocess_fold_data(df, fold)
        
        # Train model
        model = train_model(X_train, y_train)
        all_models.append(model)
        
        # Evaluate on test fold
        fold_result = evaluate_fold(model, X_test, y_test, fold)
        fold_results.append(fold_result)
    
    # Plot ROC curves
    plot_roc_curves(fold_results)
    
    # Plot combined confusion matrix
    plot_combined_confusion_matrix(fold_results)
    
    # Plot precision-recall curves
    plot_precision_recall_curves(fold_results)
    
    # Calculate and display statistics
    calculate_statistics(fold_results)
    
    # Feature importance analysis
    # Use the last model for simplicity, or could average across all models
    print("\nAnalyzing feature importance...")
    feature_imp = feature_importance(all_models[-1], feature_names)
    print("Top 10 most important features:")
    print(feature_imp.head(10))
    
    # Save the final model (could also save all models)
    joblib.dump(all_models[-1], 'k5_random_forest_model.pkl')
    print("\nFinal model saved as 'k5_random_forest_model.pkl'")
    
    return all_models, fold_results

# Run the pipeline
if __name__ == "__main__":
    main()