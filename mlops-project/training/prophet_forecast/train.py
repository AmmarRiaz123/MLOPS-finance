import os
import sys
import traceback
from pathlib import Path
import json
import shutil
from datetime import datetime
from typing import Dict, Any


from features import prepare_prophet_df

# --- best-effort Discord alerting ---
try:
    REPO_ROOT = Path(__file__).resolve().parents[2]
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from app.core.alerting import send_discord_alert
except Exception:
    def send_discord_alert(message: str, **kwargs):  # type: ignore
        return False

def _alert_train_failure(tag: str, exc: Exception) -> None:
    try:
        send_discord_alert(f"[train][{tag}] FAILED: {exc}\n{traceback.format_exc()[:1500]}")
    except Exception:
        pass

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

def train(test_size_ratio: float = 0.2, random_seed: int = 42) -> Dict[str, Any]:
    """
    Train Prophet baseline with log-target and changepoint tuning.
    """
    df, regressors = prepare_prophet_df()
    if df.empty:
        raise RuntimeError("No data available for training")

    # keep only positive closes
    df = df.copy()
    df = df[df['y'] > 0]
    df['y_orig'] = df['y'].copy()
    import numpy as np
    df['y'] = np.log(df['y'])

    # splits
    split_idx = int(len(df) * (1 - test_size_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # internal val split for CPS tuning
    val_ratio = 0.2
    sub_split = int(len(train_df) * (1 - val_ratio))
    train_sub = train_df.iloc[:sub_split].copy()
    val_sub = train_df.iloc[sub_split:].copy()

    cps_list = [0.01, 0.05, 0.1, 0.2]
    best_rmse = None
    best_cps = None
    from sklearn.metrics import mean_squared_error
    import math

    for cps in cps_list:
        m = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                    changepoint_prior_scale=cps, interval_width=0.8)
        m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
        for r in regressors:
            m.add_regressor(r)
        m.fit(train_sub)
        forecast_val = m.predict(val_sub)
        y_pred = np.exp(forecast_val['yhat'].values)
        y_true = val_sub['y_orig'].values
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        if best_rmse is None or rmse < best_rmse:
            best_rmse = rmse
            best_cps = cps

    # final model with best_cps
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                    changepoint_prior_scale=best_cps, interval_width=0.8)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    for r in regressors:
        model.add_regressor(r)
    model.fit(train_df)

    # compute simple OLS regressor coefficients on train (log-space) for explainability
    reg_coef = {}
    try:
        if regressors:
            X = train_df[regressors].values
            y = train_df['y'].values  # log-y
            # add intercept
            X_design = np.hstack([np.ones((X.shape[0],1)), X])
            coefs, *_ = np.linalg.lstsq(X_design, y, rcond=None)
            intercept = float(coefs[0])
            for i, r in enumerate(regressors, start=1):
                reg_coef[r] = float(coefs[i])
            reg_coef = {"intercept": intercept, **reg_coef}
    except Exception:
        reg_coef = {}

    # archive and save model
    _archive_if_exists(LATEST_MODEL_PATH, ARCHIVED_DIR, "prophet_forecast.pkl")
    LATEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model, LATEST_MODEL_PATH)

    # plotting and changepoint visualization
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
        fig_path = PLOTS_DIR / "train_actual_vs_fitted.png"
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"[train] saved training actual vs fitted plot: {fig_path.resolve()}")
        # components plot (trend/seasonality)
        comp_fig = model.plot_components(forecast_train)
        comp_path = PLOTS_DIR / "components.png"
        comp_fig.savefig(comp_path, dpi=150)
        plt.close(comp_fig)
        print(f"[train] saved components plot: {comp_path.resolve()}")
    except Exception as _plot_err:
        # record plotting error for debugging
        METRICS_LATEST_DIR.mkdir(parents=True, exist_ok=True)
        err_path = METRICS_LATEST_DIR / "plot_error.txt"
        with open(err_path, "w") as _f:
            _f.write(str(_plot_err))
        print(f"[train] plotting failed, see: {err_path.resolve()}")
        pass
    # except Exception as _plot_err:
    #     METRICS_LATEST_DIR.mkdir(parents=True, exist_ok=True)
    #     with open(METRICS_LATEST_DIR / "plot_error.txt", "w") as _f:
    #         _f.write(str(_plot_err))

    # save metadata including selected changepoint and regressor coef
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
            "candidate_changepoint_prior_scales": cps_list,
            "interval_width": 0.8,
            "monthly_seasonality": {"period": 30.5, "fourier_order": 5}
        },
        "validation_rmse": float(best_rmse),
        "regressor_coefs": reg_coef
    }
    with open(METRICS_LATEST_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[train] saved training metadata: {(METRICS_LATEST_DIR / 'training_metadata.json').resolve()}")

    # also persist list of regressors / feature names for discoverability
    try:
        METRICS_LATEST_DIR.mkdir(parents=True, exist_ok=True)
        features_file = METRICS_LATEST_DIR / "training_features.json"
        # regressors is a list captured earlier in training
        with open(features_file, "w") as _f:
            json.dump({"features": regressors}, _f, indent=2)
        print(f"[train] saved training features: {features_file.resolve()}")
    except Exception as _fe:
        print(f"[train] failed to save training features: {_fe}")

    return metadata

if __name__ == "__main__":
    try:
        meta = train()
        print("Training metadata:", meta)
    except Exception as e:
        _alert_train_failure("prophet_forecast", e)
        raise
