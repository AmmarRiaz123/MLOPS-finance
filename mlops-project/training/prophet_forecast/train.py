from pathlib import Path
import json
import shutil
from datetime import datetime
from typing import Dict, Any

# external libs
from joblib import dump
try:
    from prophet import Prophet
except Exception as e:
    raise RuntimeError(
        "Prophet is not installed. Install it (and cmdstanpy) with:\n"
        "  pip install prophet cmdstanpy\n"
        "or install all project deps:\n"
        "  pip install -r mlops-project/requirements.txt\n"
        "See https://facebook.github.io/prophet/docs/installation.html for platform-specific notes."
    ) from e

from features import prepare_prophet_df

REPO_ROOT = Path(__file__).resolve().parents[2]  # mlops-project
MODELS_DIR = REPO_ROOT / "models"
LATEST_MODEL_PATH = MODELS_DIR / "latest" / "prophet_forecast.pkl"
ARCHIVED_DIR = MODELS_DIR / "archived"

METRICS_DIR = Path(__file__).resolve().parent / "metrics"
METRICS_LATEST_DIR = METRICS_DIR / "latest"

def _archive_if_exists(path: Path, archived_dir: Path, prefix: str):
    if path.exists():
        archived_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%dT%H%M%S")
        dest = archived_dir / f"{ts}_{prefix}"
        shutil.move(str(path), str(dest))

def train(changepoint_prior_scale: float = 0.05, test_size_ratio: float = 0.2, random_seed: int = 42) -> Dict[str, Any]:
    """
    Train Prophet baseline. Returns metadata dictionary.
    """
    df, regressors = prepare_prophet_df()
    if df.empty:
        raise RuntimeError("No data available for training")

    # drop non-positive closes and keep original price for metrics
    df = df.copy()
    df = df[df['y'] > 0]
    df['y_orig'] = df['y'].copy()
    # log-transform target for Prophet training
    import numpy as np
    df['y'] = np.log(df['y'])

    # chronological split into train / holdout test
    split_idx = int(len(df) * (1 - test_size_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # further split train_df into train_sub / val_sub for hyperparameter tuning (last 20% of train as val)
    val_ratio = 0.2
    sub_split = int(len(train_df) * (1 - val_ratio))
    train_sub = train_df.iloc[:sub_split].copy()
    val_sub = train_df.iloc[sub_split:].copy()

    # candidate changepoints
    cps_list = [0.01, 0.05, 0.1, 0.2]
    best_rmse = None
    best_cps = None
    best_model = None

    from sklearn.metrics import mean_squared_error
    import math

    for cps in cps_list:
        # init model with monthly seasonality
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                    changepoint_prior_scale=cps)
        # add custom monthly seasonality
        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        for r in regressors:
            m.add_regressor(r)
        # fit on train_sub
        m.fit(train_sub)
        # predict on val_sub
        forecast_val = m.predict(val_sub)
        # invert log -> price
        y_pred = np.exp(forecast_val['yhat'].values)
        # true original prices
        y_true = val_sub['y_orig'].values
        # compute RMSE on original scale
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_cps = cps
            best_model = m

    # Refit final model on full train_df with best_cps
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                    changepoint_prior_scale=best_cps)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    for r in regressors:
        model.add_regressor(r)
    model.fit(train_df)

    # archive existing model if present
    _archive_if_exists(LATEST_MODEL_PATH, ARCHIVED_DIR, "prophet_forecast.pkl")

    # ensure latest dir
    LATEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model, LATEST_MODEL_PATH)

    # generate and save training plots (non-interactive)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR = METRICS_LATEST_DIR / "plots"
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        # Predict on training dataframe to get fitted values (no leakage)
        forecast_train = model.predict(train_df)
        # convert fitted log predictions back to price for plotting
        forecast_train['yhat'] = np.exp(forecast_train['yhat'])
        if 'yhat_lower' in forecast_train.columns:
            forecast_train['yhat_lower'] = np.exp(forecast_train['yhat_lower'])
        if 'yhat_upper' in forecast_train.columns:
            forecast_train['yhat_upper'] = np.exp(forecast_train['yhat_upper'])
        # actual vs fitted (use model.plot which expects prophet-format forecast; we use the model.plot on forecast_train)
        fig = model.plot(forecast_train)
        fig.savefig(PLOTS_DIR / "train_actual_vs_fitted.png", dpi=150)
        plt.close(fig)
        # components plot (trend/seasonality)
        comp_fig = model.plot_components(forecast_train)
        comp_fig.savefig(PLOTS_DIR / "components.png", dpi=150)
        plt.close(comp_fig)
    except Exception as _plot_err:
        METRICS_LATEST_DIR.mkdir(parents=True, exist_ok=True)
        with open(METRICS_LATEST_DIR / "plot_error.txt", "w") as _f:
            _f.write(str(_plot_err))

    # save training metadata (include selected changepoint)
    METRICS_LATEST_DIR.mkdir(parents=True, exist_ok=True)
    metadata = {
        "trained_at": datetime.now().isoformat(),
        "data_start": df['ds'].min().isoformat(),
        "data_end": df['ds'].max().isoformat(),
        "train_end": train_df['ds'].max().isoformat(),
        "n_train": len(train_df),
        "n_test": len(test_df),
        "regressors": regressors,
        "params": {
            "selected_changepoint_prior_scale": best_cps,
            "candidate_changepoint_prior_scales": cps_list
        },
        "validation_rmse": float(best_rmse)
    }
    with open(METRICS_LATEST_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata

if __name__ == "__main__":
    meta = train()
    print("Training metadata:", meta)
