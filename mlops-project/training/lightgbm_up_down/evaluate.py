import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

def load_model_and_info(model_dir):
    """Load trained model and associated metadata.

    Tries training/lightgbm_up_down/model first, then falls back to mlops-project/models/latest.
    Also attempts to load training_info from training/lightgbm_up_down/metrics/latest.
    """
    model_dir = Path(model_dir)

    # candidate 1: model in provided model_dir
    model_path = model_dir / "lightgbm_up_down_model.pkl"
    features_path = model_dir / "feature_names.json"

    # candidate 2: repo models/latest
    repo_root = Path(__file__).resolve().parents[2]  # mlops-project
    latest_models_dir = repo_root / "models" / "latest"
    latest_model_path = latest_models_dir / "lightgbm_up_down_model.pkl"
    latest_features_path = latest_models_dir / "feature_names.json"

    # determine which model to use
    if model_path.exists():
        model = joblib.load(model_path)
        feature_names = None
        if features_path.exists():
            with open(features_path, 'r') as f:
                feature_names = json.load(f)
    elif latest_model_path.exists():
        model = joblib.load(latest_model_path)
        feature_names = None
        if latest_features_path.exists():
            with open(latest_features_path, 'r') as f:
                feature_names = json.load(f)
    else:
        raise FileNotFoundError(f"Model not found at {model_path} or {latest_model_path}")

    # Try to load training_info from training metrics latest (preferred)
    train_info = {}
    metrics_latest = Path(__file__).resolve().parent / "metrics" / "latest" / "training_info.json"
    if metrics_latest.exists():
        with open(metrics_latest, 'r') as f:
            train_info = json.load(f)
    else:
        # fallback: look for any training_info files in metrics folder
        metrics_dir = Path(__file__).resolve().parent / "metrics"
        if metrics_dir.exists():
            candidates = sorted(metrics_dir.glob("**/training_info*.json"), reverse=True)
            if candidates:
                try:
                    with open(candidates[0], 'r') as f:
                        train_info = json.load(f)
                except Exception:
                    train_info = {}

    return model, feature_names, train_info

def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute comprehensive classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
    }
    
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Down', 'Up'],
                yticklabels=['Down', 'Up'])
    plt.title('Confusion Matrix - LightGBM Up/Down Prediction')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    return cm

def plot_feature_importance(model, feature_names=None, top_n=15, save_path=None):
    """Plot feature importance from trained model."""
    importance = model.feature_importances_
    
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importance))]
    
    # Sort by importance
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(10, 8))
    plt.bar(range(len(indices)), importance[indices])
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45)
    plt.title(f'Top {top_n} Feature Importance - LightGBM')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    
    plt.show()

def evaluate_model(model_dir, X_test, y_test, save_results=True):
    """Complete model evaluation pipeline."""
    # Load model and info
    model, feature_names, train_info = load_model_and_info(model_dir)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]  # Probability of class 1
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    # Print results
    print("=" * 50)
    print("LightGBM Up/Down Prediction - Evaluation Results")
    print("=" * 50)
    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1_score']:.4f}")
    print(f"ROC AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 50)
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))
    
    # Plots and visualizations
    results_dir = Path(model_dir) / "evaluation_results"
    # ensure parents=True so nested directories are created (fixes FileNotFoundError on Windows)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Confusion matrix
    cm = plot_confusion_matrix(y_test, y_pred, 
                             save_path=results_dir / "confusion_matrix.png")
    
    # Feature importance
    plot_feature_importance(model, feature_names, top_n=15,
                          save_path=results_dir / "feature_importance.png")
    
    # Save results
    if save_results:
        results = {
            'model_info': train_info,
            'test_metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': classification_report(y_test, y_pred, 
                                                         target_names=['Down', 'Up'], 
                                                         output_dict=True)
        }
        
        results_file = results_dir / "evaluation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to {results_file}")
    
    return metrics, y_pred, y_proba

def main():
    """Main evaluation function - can be run standalone or imported."""
    # Default model directory
    model_dir = Path(__file__).parent / "model"
    
    if not model_dir.exists():
        print("No trained model found. Please run train.py first.")
        return
    
    # For standalone evaluation, we need test data
    # In practice, you might load this from saved test set or re-generate
    print("For standalone evaluation, please ensure you have X_test and y_test available")
    print("Alternatively, run this from train.py or provide test data path")
    
    # Example of how to use if you have test data:
    # X_test, y_test = load_test_data()  # Implement this function
    # metrics, y_pred, y_proba = evaluate_model(model_dir, X_test, y_test)

if __name__ == "__main__":
    main()
