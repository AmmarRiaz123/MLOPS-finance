"""
Train RandomForest/ExtraTrees regressor for return prediction.
Saves model to models/latest/random_forest_return_model.pkl
Saves metrics to training/random_forest_return/metrics/latest/metrics.json
"""
from pathlib import Path
import json
import shutil
from datetime import datetime
import logging
import joblib
import argparse
import numpy as np

from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error

from features import prepare_features

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
LATEST_MODEL_PATH = MODELS_DIR / "latest" / "random_forest_return_model.pkl"
ARCHIVED_MODELS = MODELS_DIR / "archived"

METRICS_DIR = Path(__file__).resolve().parent / "metrics"
METRICS_LATEST_DIR = METRICS_DIR / "latest"
METRICS_ARCHIVED_DIR = METRICS_DIR / "archived"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def _archive_if_exists(path: Path, archived_dir: Path, prefix: str):
    if path.exists():
        archived_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        dest = archived_dir / f"{ts}_{prefix}"
        shutil.move(str(path), str(dest))

def train(model_type: str = "rf",
          horizon: int = 1,
          test_size_ratio: float = 0.2,
          random_state: int = 42,
          tune: bool = False,
          n_jobs: int = -1,
          param_overrides: dict | None = None):
    """
    Train pipeline entry.
    model_type: "rf" or "et" (ExtraTrees)
    """
    logging.info("Preparing features")
    res = prepare_features(horizon, scale=True)
    X, y, dates, scaler = res

    # time-based split
    split_idx = int(len(X) * (1 - test_size_ratio))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # default hyperparameters
    base_params = {
        "n_estimators": 200,
        "max_depth": None,
        "min_samples_split": 2,
        "max_features": "sqrt",  # 'auto' is invalid in newer sklearn; use 'sqrt' or 'log2' or float
        "random_state": random_state,
        "n_jobs": n_jobs
    }
    if param_overrides:
        base_params.update(param_overrides)

    # instantiate estimator for tuning
    if model_type == "et":
        est = ExtraTreesRegressor(**base_params)
    else:
        est = RandomForestRegressor(**base_params)

    best_params = base_params.copy()
    if tune:
        logging.info("Running randomized hyperparameter search (time-series CV)")
        tss = TimeSeriesSplit(n_splits=3)
        param_dist = {
            "n_estimators": [100, 150, 200],
            "max_depth": [3,5,8, None],
            "min_samples_split": [2,5,10],
            "max_features": ["sqrt","log2", 0.6]
        }
        rs = RandomizedSearchCV(est, param_distributions=param_dist, n_iter=8, cv=tss,
                                scoring="neg_root_mean_squared_error", random_state=random_state, n_jobs=1, verbose=0)
        try:
            rs.fit(X_train, y_train)
            best_params.update(rs.best_params_)
            logging.info(f"Tuning selected params: {rs.best_params_}")
        except Exception as e:
            logging.warning(f"Tuning failed: {e}")

    # final estimator
    if model_type == "et":
        model = ExtraTreesRegressor(**best_params)
    else:
        model = RandomForestRegressor(**best_params)

    logging.info("Training final model")
    model.fit(X_train, y_train)

    logging.info("Validating")
    y_pred = model.predict(X_val)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    mae = float(mean_absolute_error(y_val, y_pred))

    # feature importance
    try:
        fi = model.feature_importances_
        feat_names = X.columns.tolist()
        if fi.sum() > 0:
            norm = (fi / fi.sum()) * 100.0
            feat_imp = dict(zip(feat_names, np.round(norm,6).tolist()))
        else:
            feat_imp = dict(zip(feat_names, [0.0]*len(feat_names)))
    except Exception:
        feat_imp = {}

    # archive & save model
    _archive_if_exists(LATEST_MODEL_PATH, ARCHIVED_MODELS, "random_forest_return_model.pkl")
    LATEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "scaler": scaler, "params": best_params}, LATEST_MODEL_PATH)
    logging.info(f"Saved model to {LATEST_MODEL_PATH.resolve()}")

    # archive & save metrics
    _archive_if_exists(METRICS_LATEST_DIR / "metrics.json", METRICS_ARCHIVED_DIR, "metrics.json")
    METRICS_LATEST_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "trained_at": datetime.now().isoformat(),
        "horizon": horizon,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "params": best_params,
        "val_rmse": rmse,
        "val_mae": mae,
        "feature_importance": feat_imp
    }
    with open(METRICS_LATEST_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved metrics to {(METRICS_LATEST_DIR / 'metrics.json').resolve()}")

    return {"model_path": str(LATEST_MODEL_PATH), "metrics_path": str(METRICS_LATEST_DIR / "metrics.json"), "rmse": rmse, "mae": mae}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["rf","et"], default="rf")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    train(model_type=args.model, horizon=args.horizon, test_size_ratio=args.test_size, random_state=args.seed, tune=args.tune)
