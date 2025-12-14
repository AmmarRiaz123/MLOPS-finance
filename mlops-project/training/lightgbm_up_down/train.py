import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import json
import shutil
from datetime import datetime
import importlib.util

# Load the local features.py module
_features_path = Path(__file__).resolve().parent / "features.py"
_spec = importlib.util.spec_from_file_location("lightgbm_up_down.features", str(_features_path))
_features_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_features_module)
prepare_ml_data = getattr(_features_module, "prepare_ml_data")

def time_series_split(X, y, test_size=0.2, n_splits=5):
    """Chronological split using TimeSeriesSplit, returning the last split as train/test."""
    if n_splits <= 1:
        # fallback to holdout
        split_idx = int(len(X) * (1 - test_size))
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

    tss = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tss.split(X))
    train_idx, test_idx = splits[-1]
    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]
    print(f"TimeSeriesSplit used: n_splits={n_splits}, train={len(X_train)}, test={len(X_test)}")
    return X_train, X_test, y_train, y_test

def train_lightgbm(X_train, y_train, X_valid=None, y_valid=None, random_seed: int = 42):
    """Train LightGBM classifier with tuned defaults and early stopping when validation provided."""
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'max_depth': 5,
        'learning_rate': 0.03,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': random_seed
    }

    model = lgb.LGBMClassifier(**params, n_estimators=300, n_jobs=-1)

    if X_valid is not None and y_valid is not None:
        model.fit(
            X_train, y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric='binary_logloss',
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
    else:
        model.fit(X_train, y_train)

    return model

def save_model_and_info(model, model_filename, feature_names, train_info):
    """Save trained model to models/latest, archive previous, and store metrics in training metrics folder."""
    repo_root = Path(__file__).resolve().parents[2]  # mlops-project
    models_root = repo_root / "models"
    latest_dir = models_root / "latest"
    archived_dir = models_root / "archived"
    metrics_root = Path(__file__).resolve().parent / "metrics"

    latest_dir.mkdir(parents=True, exist_ok=True)
    archived_dir.mkdir(parents=True, exist_ok=True)
    metrics_root.mkdir(parents=True, exist_ok=True)

    latest_model_path = latest_dir / model_filename

    # Archive existing latest model if present
    if latest_model_path.exists():
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        archived_name = f"{ts}_{model_filename}"
        shutil.move(str(latest_model_path), str(archived_dir / archived_name))

    # Save new model to latest
    joblib.dump(model, latest_model_path)

    # Save feature names alongside the model
    features_path = latest_dir / "feature_names.json"
    with open(features_path, 'w') as f:
        json.dump(list(feature_names), f, indent=2)

    # --- Metrics: maintain metrics/latest and metrics/archived ---
    metrics_latest_dir = metrics_root / "latest"
    metrics_archived_dir = metrics_root / "archived"
    metrics_latest_dir.mkdir(parents=True, exist_ok=True)
    metrics_archived_dir.mkdir(parents=True, exist_ok=True)

    latest_metrics_path = metrics_latest_dir / "training_info.json"

    # Archive existing latest metrics if present
    if latest_metrics_path.exists():
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        archived_metrics_name = f"training_info_{ts}.json"
        shutil.move(str(latest_metrics_path), str(metrics_archived_dir / archived_metrics_name))

    # Save newest metrics as the canonical "latest"
    with open(latest_metrics_path, 'w') as f:
        json.dump(train_info, f, indent=2)

    print(f"Model saved to {latest_model_path.resolve()}")
    print(f"Training metrics saved to {latest_metrics_path.resolve()}")
    return latest_model_path

def main(horizon: int = 1, smooth_window: int | None = None, random_seed: int = 42):
    """Main training pipeline."""
    # Find CSV data
    data_dir = Path(__file__).resolve().parents[2] / "data"
    csv_files = list(data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("No CSV files found in data directory")

    csv_path = csv_files[0]
    print(f"Training on data from: {csv_path}")

    # Prepare data (use horizon/smoothing options)
    X, y = prepare_ml_data(csv_path, horizon=horizon, smooth_window=smooth_window, scale=False)

    # Chronological split using TimeSeriesSplit (take last split)
    X_train, X_test, y_train, y_test = time_series_split(X, y, test_size=0.2, n_splits=5)

    # Train model with reproducible seed
    print("Training LightGBM model...")
    model = train_lightgbm(X_train, y_train, X_test, y_test, random_seed=random_seed)

    # Quick evaluation
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {accuracy:.4f}")

    train_info = {
        'data_file': str(csv_path),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'features_count': len(X.columns),
        'test_accuracy': accuracy,
        'feature_importance_top5': dict(zip(
            X.columns[model.feature_importances_.argsort()[-5:][::-1]],
            model.feature_importances_[model.feature_importances_.argsort()[-5:][::-1]].tolist()
        ))
    }

    # Save model and info
    model_filename = "lightgbm_up_down_model.pkl"
    model_path = save_model_and_info(model, model_filename, X.columns, train_info)

    print("\nTraining completed!")
    print(f"Model saved at: {model_path}")
    return model, X_test, y_test

if __name__ == "__main__":
    # default: 1-day horizon, no smoothing
    model, X_test, y_test = main(horizon=1, smooth_window=None, random_seed=42)
