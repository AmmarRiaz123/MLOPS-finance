from pathlib import Path
import json
import shutil
from datetime import datetime
import matplotlib
# use non-interactive backend so saving works in headless CI/servers
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import load

from features import prepare_prophet_df

REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
LATEST_MODEL_PATH = MODELS_DIR / "latest" / "prophet_forecast.pkl"

METRICS_ROOT = Path(__file__).resolve().parent / "metrics"
METRICS_LATEST = METRICS_ROOT / "latest" / "metrics.json"
METRICS_ARCHIVED = METRICS_ROOT / "archived"

PLOTS_DIR = METRICS_ROOT / "latest" / "plots"

def _archive_metrics_if_exists(latest_metrics_path: Path, archived_dir: Path):
    if latest_metrics_path.exists():
        archived_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        dest = archived_dir / f"metrics_{ts}.json"
        shutil.move(str(latest_metrics_path), str(dest))

def evaluate():
    # load data and regressors
    df, regressors = prepare_prophet_df()
    if df.empty:
        raise RuntimeError("No data available for evaluation")

    # split consistent with training: last 20% used as test
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # load model
    if not LATEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {LATEST_MODEL_PATH}")
    model = load(LATEST_MODEL_PATH)

    # create future for next 5 days
    periods = 5
    future = model.make_future_dataframe(periods=periods, freq='D', include_history=False)

    # include regressors: use last known normalized regressor values (no leakage)
    last_row = train_df.iloc[-1]
    for r in regressors:
        val = last_row.get(r, 0)
        future[r] = val

    forecast = model.predict(future)

    # convert log-space predictions back to price scale and keep intervals (interval_width assumed 0.8)
    forecast_small = forecast[['ds','yhat']].copy()
    if 'yhat_lower' in forecast.columns:
        forecast_small['yhat_lower'] = forecast['yhat_lower']
    if 'yhat_upper' in forecast.columns:
        forecast_small['yhat_upper'] = forecast['yhat_upper']

    forecast_small['yhat'] = np.exp(forecast_small['yhat'])
    if 'yhat_lower' in forecast_small.columns:
        forecast_small['yhat_lower'] = np.exp(forecast_small['yhat_lower'])
    if 'yhat_upper' in forecast_small.columns:
        forecast_small['yhat_upper'] = np.exp(forecast_small['yhat_upper'])

    # compute metrics for 3- and 5-day horizons relative to train end
    metrics = {}
    horizons = [3,5]
    for h in horizons:
        target_ds = train_df['ds'].max() + pd.Timedelta(days=h)
        row = forecast_small[forecast_small['ds'] == target_ds]
        actual_row = test_df[test_df['ds'] == target_ds]
        if not row.empty and not actual_row.empty and not pd.isna(actual_row['y'].values[0]):
            y_pred = row['yhat'].values
            y_true = actual_row['y'].values
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            mape = float(np.mean(np.abs((y_true - y_pred) / y_true))) if np.any(y_true!=0) else None
        else:
            mae = None; rmse = None; mape = None
        metrics[f'horizon_{h}'] = {
            "ds": str(target_ds.date()),
            "mae": mae,
            "rmse": rmse,
            "mape": mape
        }

    # include regressor coef summary if available from training metadata
    reg_coefs = {}
    training_meta_path = Path(__file__).resolve().parent / "metrics" / "latest" / "training_metadata.json"
    if training_meta_path.exists():
        try:
            with open(training_meta_path, 'r') as f:
                tmeta = json.load(f)
                reg_coefs = tmeta.get('regressor_coefs', {})
        except Exception:
            reg_coefs = {}

    # archive old metrics
    _archive_metrics_if_exists(METRICS_LATEST, METRICS_ARCHIVED)

    # ensure plot dir
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # save metrics json (include regressor coefs)
    final_metrics = {
        "evaluated_at": datetime.now().isoformat(),
        "metrics": metrics,
        "train_end": str(train_df['ds'].max().date()),
        "regressor_coefs": reg_coefs
    }
    with open(METRICS_LATEST, "w") as f:
        json.dump(final_metrics, f, indent=2)

    # build plot dataframe: recent history + test + forecast (forecast in price scale)
    hist_tail = train_df[['ds','y']].copy()
    actual_future = test_df[['ds','y']].copy()
    plot_df = pd.concat([hist_tail.tail(60), actual_future, forecast_small], sort=False).drop_duplicates(subset=['ds'], keep='first').set_index('ds')

    fig, ax = plt.subplots(figsize=(10,6))
    if 'y' in plot_df.columns:
        ax.plot(plot_df.index, plot_df['y'], label='actual', marker='o', linestyle='-')
    if 'yhat' in plot_df.columns:
        ax.plot(plot_df.index, plot_df['yhat'], label='forecast', marker='x', linestyle='--')
        lower = plot_df.get('yhat_lower', pd.Series(np.nan, index=plot_df.index)).values
        upper = plot_df.get('yhat_upper', pd.Series(np.nan, index=plot_df.index)).values
        ax.fill_between(plot_df.index, lower, upper, color='gray', alpha=0.2)
    ax.legend()
    ax.set_title("Actual vs Forecast (Prophet)")
    ax.set_xlabel("ds")
    ax.set_ylabel("Close")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "actual_vs_forecast.png", dpi=150)
    plt.close(fig)

    # components and changepoint plots
    try:
        comp_fig = model.plot_components(forecast)
        comp_fig.savefig(PLOTS_DIR / "components.png", dpi=150)
        plt.close(comp_fig)
    except Exception:
        pass

    # trend with changepoints
    try:
        trend = model.predict(train_df)[['ds','trend']].set_index('ds')
        fig2, ax2 = plt.subplots(figsize=(10,5))
        ax2.plot(trend.index, np.exp(trend['trend']), label='trend (exp)')
        if hasattr(model, 'changepoints') and model.changepoints is not None:
            for cp in model.changepoints:
                ax2.axvline(cp, color='red', alpha=0.3, linestyle='--')
        ax2.set_title("Trend with changepoints (exp scale)")
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(PLOTS_DIR / "trend_changepoints.png", dpi=150)
        plt.close(fig2)
    except Exception:
        pass

    return final_metrics

if __name__ == "__main__":
    metrics = evaluate()
    print("Evaluation metrics:", metrics)
