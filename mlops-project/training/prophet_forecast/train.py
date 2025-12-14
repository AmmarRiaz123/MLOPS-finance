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

    # chronological split
    split_idx = int(len(df) * (1 - test_size_ratio))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    # init model
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True, daily_seasonality=False,
                    changepoint_prior_scale=changepoint_prior_scale)
    for r in regressors:
        model.add_regressor(r)

    # fit
    model.fit(train_df)

    # archive existing model if present
    _archive_if_exists(LATEST_MODEL_PATH, ARCHIVED_DIR, "prophet_forecast.pkl")

    # ensure latest dir
    LATEST_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    dump(model, LATEST_MODEL_PATH)

    # save training metadata
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
            "changepoint_prior_scale": changepoint_prior_scale
        }
    }
    with open(METRICS_LATEST_DIR / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata

if __name__ == "__main__":
    meta = train()
    print("Training metadata:", meta)
