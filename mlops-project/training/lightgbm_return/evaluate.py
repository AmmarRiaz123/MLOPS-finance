"""
Evaluate LightGBM return regression model.
Saves metrics to training/lightgbm_return/metrics/latest/metrics.json
Saves plots to training/lightgbm_return/metrics/latest/plots/
"""
from pathlib import Path
import json
import shutil
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from features import prepare_features

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_ROOT = REPO_ROOT / "models"
# load canonical model filename
LATEST_MODEL_PATH = MODELS_ROOT / "latest" / "lightgbm_return_model.pkl"

METRICS_ROOT = Path(__file__).resolve().parent / "metrics"
METRICS_LATEST_DIR = METRICS_ROOT / "latest"
METRICS_ARCHIVED_DIR = METRICS_ROOT / "archived"
PLOTS_DIR = METRICS_LATEST_DIR / "plots"

def _archive_if_exists(path: Path, archived_dir: Path, prefix: str):
    if path.exists():
        archived_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        dest = archived_dir / f"{ts}_{prefix}"
        shutil.move(str(path), str(dest))

def evaluate(horizon: int = 3):
    print("[eval] preparing data...")
    # prepare features matching training horizon
    res = prepare_features(scale=True, horizons=[horizon])
    if len(res) == 4:
        X, y, dates, scaler = res
    else:
        X, y, dates = res
        scaler = None

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]
    dates_val = dates.iloc[split_idx:]

    if not LATEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {LATEST_MODEL_PATH}")
    model = joblib.load(LATEST_MODEL_PATH)
    print(f"[eval] loaded model from {LATEST_MODEL_PATH.resolve()}")

    # align features to model if booster has feature_name
    try:
        if callable(getattr(model, "feature_name", None)):
            feat_names = model.feature_name()
            X_val_used = X_val[feat_names]
        else:
            X_val_used = X_val
    except Exception:
        X_val_used = X_val

    try:
        y_pred = model.predict(X_val_used)
    except Exception:
        y_pred = model.predict(X_val_used.values)

    rmse = float(np.sqrt(mean_squared_error(y_val, y_pred)))
    mae = float(mean_absolute_error(y_val, y_pred))
    r2 = float(r2_score(y_val, y_pred))
    # directional accuracy (up/down)
    dir_acc = float((np.sign(y_pred) == np.sign(y_val)).mean())

    # archive previous metrics
    _archive_if_exists(METRICS_LATEST_DIR / "metrics.json", METRICS_ARCHIVED_DIR, "metrics.json")
    METRICS_LATEST_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "evaluated_at": datetime.now().isoformat(),
        "horizon": horizon,
        "n_eval": len(y_val),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "directional_accuracy": dir_acc
    }
    metrics_path = METRICS_LATEST_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] saved metrics: {metrics_path.resolve()}")

    # plots: time-series predicted vs actual
    try:
        ts_df = pd.DataFrame({"ds": dates_val.values, "y_true": y_val.values, "y_pred": y_pred})
        ts_df = ts_df.set_index("ds")
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(ts_df.index, ts_df["y_true"], label="actual", marker='o', linewidth=1)
        ax.plot(ts_df.index, ts_df["y_pred"], label="predicted", marker='x', linewidth=1)
        ax.set_title("Predicted vs Actual 1-day Return (validation)")
        ax.legend()
        fig.tight_layout()
        ts_path = PLOTS_DIR / "pred_vs_actual_ts.png"
        fig.savefig(ts_path, dpi=150)
        plt.close(fig)
        print(f"[eval] saved time-series plot: {ts_path.resolve()}")
    except Exception as e:
        print(f"[eval] time-series plot failed: {e}")

    # cumulative predicted returns (simple strategy: sum predicted returns)
    try:
        cum_df = pd.DataFrame({"ds": dates_val.values, "y_true": y_val.values, "y_pred": y_pred})
        cum_df = cum_df.set_index("ds")
        cum_df['cum_true'] = (1 + cum_df['y_true']).cumprod() - 1
        cum_df['cum_pred'] = (1 + cum_df['y_pred']).cumprod() - 1
        fig, ax = plt.subplots(figsize=(10,5))
        ax.plot(cum_df.index, cum_df['cum_true'], label='cum actual', linewidth=1)
        ax.plot(cum_df.index, cum_df['cum_pred'], label='cum predicted', linewidth=1)
        ax.set_title("Cumulative returns: actual vs predicted (validation)")
        ax.legend()
        fig.tight_layout()
        cum_path = PLOTS_DIR / "cumulative_returns.png"
        fig.savefig(cum_path, dpi=150)
        plt.close(fig)
        print(f"[eval] saved cumulative returns plot: {cum_path.resolve()}")
    except Exception as e:
        print(f"[eval] cumulative returns plot failed: {e}")

    # scatter plot
    try:
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(y_val, y_pred, alpha=0.6)
        ax.set_xlabel("Actual return_1d")
        ax.set_ylabel("Predicted return_1d")
        ax.set_title("Actual vs Predicted (scatter)")
        lim = max(np.nanmax(np.abs(y_val)), np.nanmax(np.abs(y_pred)))
        ax.plot([-lim, lim], [-lim, lim], color='red', linestyle='--')
        fig.tight_layout()
        sc_path = PLOTS_DIR / "pred_vs_actual_scatter.png"
        fig.savefig(sc_path, dpi=150)
        plt.close(fig)
        print(f"[eval] saved scatter plot: {sc_path.resolve()}")
    except Exception as e:
        print(f"[eval] scatter plot failed: {e}")

    # feature importance: handle sklearn-style models and lightgbm.Booster
    try:
        fi = None
        if hasattr(model, "feature_importances_"):
            fi = np.array(getattr(model, "feature_importances_"))
        elif hasattr(model, "feature_importance"):
            fi = np.array(model.feature_importance(importance_type="gain"))

        if fi is not None:
            feat_names = X.columns.tolist()
            idx = np.argsort(fi)[::-1]
            top_n = min(20, len(feat_names))
            fig, ax = plt.subplots(figsize=(8,6))
            ax.barh([feat_names[i] for i in idx[:top_n]][::-1], fi[idx[:top_n]][::-1])
            ax.set_title("Feature importance (LightGBM)")
            fig.tight_layout()
            fi_path = PLOTS_DIR / "feature_importance.png"
            fig.savefig(fi_path, dpi=150)
            plt.close(fig)
            print(f"[eval] saved feature importance plot: {fi_path.resolve()}")
        else:
            print("[eval] no feature importance available for model type")
    except Exception as e:
        print(f"[eval] feature importance plot failed: {e}")

    return metrics

if __name__ == "__main__":
    metrics = evaluate()
    print("Evaluation metrics:", metrics)
