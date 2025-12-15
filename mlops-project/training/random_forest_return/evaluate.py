"""
Evaluate random_forest_return model on validation split.
"""
from pathlib import Path
import json
import joblib
import logging
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from features import prepare_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
LATEST_MODEL_PATH = MODELS_DIR / "latest" / "random_forest_return_model.pkl"

METRICS_DIR = Path(__file__).resolve().parent / "metrics"
METRICS_LATEST_DIR = METRICS_DIR / "latest"
METRICS_ARCHIVED_DIR = METRICS_DIR / "archived"
PLOTS_DIR = METRICS_LATEST_DIR / "plots"

def _archive_if_exists(path: Path, archived_dir: Path, prefix: str):
    if path.exists():
        archived_dir.mkdir(parents=True, exist_ok=True)
        ts = Path().absolute()  # dummy to match style (not used)
        # handled by train.py before saving metrics; keep minimal here

def evaluate(horizon: int = 1, test_size_ratio: float = 0.2):
    logging.info("Preparing features for evaluation")
    res = prepare_features(horizon, scale=True)
    X, y, dates, scaler = res

    split_idx = int(len(X) * (1 - test_size_ratio))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    if not LATEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {LATEST_MODEL_PATH}")
    obj = joblib.load(LATEST_MODEL_PATH)
    model = obj.get("model", obj) if isinstance(obj, dict) else obj
    scaler_saved = obj.get("scaler") if isinstance(obj, dict) else None

    # align cols if needed
    try:
        X_val_used = X_val[model.feature_names_in_]
    except Exception:
        X_val_used = X_val

    y_pred = model.predict(X_val_used)
    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    mae = float(mean_absolute_error(y_val, y_pred))
    r2 = float(r2_score(y_val, y_pred))

    # feature importance plot
    try:
        fi = model.feature_importances_
        feat_names = X.columns.tolist()
        idx = np.argsort(fi)[::-1]
        top = min(20, len(feat_names))
        fig, ax = plt.subplots(figsize=(8,6))
        ax.barh([feat_names[i] for i in idx[:top]][::-1], fi[idx[:top]][::-1])
        ax.set_title("Feature importance")
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)
        fp = PLOTS_DIR / "feature_importance.png"
        fig.tight_layout()
        fig.savefig(fp, dpi=150)
        plt.close(fig)
    except Exception:
        fp = None

    # save metrics (archive done in train)
    METRICS_LATEST_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "evaluated_at": datetime.now().isoformat(),
        "horizon": horizon,
        "n_eval": len(y_val),
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }
    with open(METRICS_LATEST_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Saved metrics to {(METRICS_LATEST_DIR / 'metrics.json').resolve()}")
    return metrics

if __name__ == "__main__":
    m = evaluate()
    print("Evaluation:", m)
