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
# plotly/interactive plotting is optional — don't fail if not installed
try:
    from prophet.plot import plot_plotly  # type: ignore
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False
from prophet import Prophet

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

    # Prophet model was trained on log(y). Ensure we treat predictions accordingly.
    # create future dataframe for next 5 calendar days (relative to model's history)
    periods = 5
    future = model.make_future_dataframe(periods=periods, freq='D', include_history=False)

    # include regressors by repeating last known values (simple baseline)
    last_row = train_df.iloc[-1]
    for r in regressors:
        val = last_row.get(r, 0)
        future[r] = val

    # predict (yhat is in log-space if model trained on log(y))
    forecast = model.predict(future)

    # convert forecast yhat / bounds from log-space back to price scale
    import numpy as np
    forecast_small = forecast[['ds','yhat']].copy()
    if 'yhat_lower' in forecast.columns:
        forecast_small['yhat_lower'] = forecast['yhat_lower']
    if 'yhat_upper' in forecast.columns:
        forecast_small['yhat_upper'] = forecast['yhat_upper']

    # invert predictions
    forecast_small['yhat'] = np.exp(forecast_small['yhat'])
    if 'yhat_lower' in forecast_small.columns:
        forecast_small['yhat_lower'] = np.exp(forecast_small['yhat_lower'])
    if 'yhat_upper' in forecast_small.columns:
        forecast_small['yhat_upper'] = np.exp(forecast_small['yhat_upper'])

    # Merge actuals to forecast on ds to compute metrics where actuals exist
    merged = forecast_small.merge(test_df[['ds','y']], on='ds', how='left')

    metrics = {}
    for horizon in (3,5):
        target_ds = train_df['ds'].max() + pd.Timedelta(days=horizon)
        row = merged[merged['ds'] == target_ds]
        if not row.empty and not pd.isna(row['y'].values[0]):
            y_true = row['y'].values
            y_pred = row['yhat'].values
            mae = float(mean_absolute_error(y_true, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        else:
            mae = None
            rmse = None
        metrics[f'horizon_{horizon}'] = {'mae': mae, 'rmse': rmse, 'ds': str(target_ds.date())}

    # archive previous metrics if exist
    _archive_metrics_if_exists(METRICS_LATEST, METRICS_ARCHIVED)

    # ensure directories
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # save metrics
    final_metrics = {
        "evaluated_at": datetime.now().isoformat(),
        "metrics": metrics,
        "train_end": str(train_df['ds'].max().date())
    }
    with open(METRICS_LATEST, "w") as f:
        json.dump(final_metrics, f, indent=2)

    # plotting: gather history tail + test + forecast (forecast already in price scale)
    hist_tail = train_df[['ds','y']].copy()
    actual_future = test_df[['ds','y']].copy()
    plot_df = pd.concat([hist_tail.tail(60), actual_future, forecast_small], sort=False).drop_duplicates(subset=['ds'], keep='first').set_index('ds')

    fig, ax = plt.subplots(figsize=(10,6))
    if 'y' in plot_df.columns:
        ax.plot(plot_df.index, plot_df['y'], label='actual', marker='o', linestyle='-')
    if 'yhat' in plot_df.columns:
        ax.plot(plot_df.index, plot_df['yhat'], label='forecast', marker='x', linestyle='--')
        if 'yhat_lower' in plot_df.columns and 'yhat_upper' in plot_df.columns:
            lower = plot_df['yhat_lower'].values
            upper = plot_df['yhat_upper'].values
        else:
            lower = np.full(len(plot_df), np.nan)
            upper = np.full(len(plot_df), np.nan)
        ax.fill_between(plot_df.index, lower, upper, color='gray', alpha=0.2)
    ax.legend()
    ax.set_title("Actual vs Forecast (Prophet)")
    ax.set_xlabel("ds")
    ax.set_ylabel("Close")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "actual_vs_forecast.png", dpi=150)
    plt.close(fig)

    # components plot (use forecast converted to log-space if needed, but model.plot_components accepts forecast)
    comp_fig = model.plot_components(forecast)
    comp_fig.savefig(PLOTS_DIR / "components.png", dpi=150)
    plt.close(comp_fig)

    return final_metrics

if __name__ == "__main__":
    metrics = evaluate()
    print("Evaluation metrics:", metrics)
