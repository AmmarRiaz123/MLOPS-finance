from pathlib import Path
from typing import List, Dict, Any, Optional
from joblib import load
import pandas as pd
import numpy as np

# Try to import Prophet with graceful fallback
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

# Try to use the in-memory model registry if available
try:
    from app.core.model_loader import MODEL_REGISTRY
except Exception:
    MODEL_REGISTRY = {}

REPO_ROOT = Path(__file__).resolve().parents[2]  # mlops-project
DEFAULT_MODEL_PATH = REPO_ROOT / "models" / "latest" / "prophet_forecast.pkl"

def _load_model(model_key: str):
    if not PROPHET_AVAILABLE:
        raise RuntimeError(
            "Prophet is not installed. Install it with:\n"
            "  pip install prophet\n"
            "or install all project deps:\n"
            "  pip install -r requirements.txt\n"
            "See https://facebook.github.io/prophet/docs/installation.html"
        )
    
    # prefer in-memory registry
    model = MODEL_REGISTRY.get(model_key) if isinstance(MODEL_REGISTRY, dict) else None
    if model is not None:
        return model
    # fallback to disk
    model_path = DEFAULT_MODEL_PATH
    if not model_path.exists():
        raise RuntimeError(f"Prophet model not found at {model_path}")
    return load(model_path)

def _ensure_datetime(df: pd.DataFrame, ds_col: str = "ds") -> pd.DataFrame:
    if ds_col in df.columns:
        df[ds_col] = pd.to_datetime(df[ds_col])
    return df

def forecast_prophet(periods: int,
                     history: Optional[List[Dict[str, Any]]] = None,
                     model_key: str = "prophet_forecast") -> List[Dict[str, Any]]:
    """
    Forecast using a saved Prophet model with auto-fill for missing regressors.
    
    If Prophet is not installed, raises a RuntimeError with installation instructions.
    """
    if not PROPHET_AVAILABLE:
        raise RuntimeError(
            "Prophet forecasting is not available. Install Prophet with:\n"
            "pip install prophet\n\n"
            "Note: Prophet requires additional system dependencies. "
            "See https://facebook.github.io/prophet/docs/installation.html for details."
        )
    
    if periods is None or periods <= 0:
        raise ValueError("periods must be a positive integer")

    model = _load_model(model_key)

    # load history into DataFrame if provided
    hist_df = None
    if history:
        hist_df = pd.DataFrame(history).copy()
        hist_df = _ensure_datetime(hist_df, "ds")

    # Determine regressor names that the model expects
    reg_names = []
    try:
        extra = getattr(model, "extra_regressors", None)
        if extra and isinstance(extra, dict):
            reg_names = list(extra.keys())
    except Exception:
        reg_names = []

    # If history contains additional columns that look like regressors, include them
    if hist_df is not None:
        hist_regs = [c for c in hist_df.columns if c not in ("ds", "y", "y_orig")]
        for c in hist_regs:
            if c not in reg_names:
                reg_names.append(c)

    # Build future dataframe (dates)
    if hist_df is not None and "ds" in hist_df.columns and not hist_df.empty:
        last_ds = hist_df["ds"].max()
        start = last_ds + pd.Timedelta(days=1)
    else:
        start = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start, periods=periods, freq="D")
    future_df = pd.DataFrame({"ds": future_dates})

    # Fill regressors: use last known value from history if available, else default 0
    for r in reg_names:
        if hist_df is not None and r in hist_df.columns and not hist_df[r].dropna().empty:
            last_val = hist_df[r].dropna().iloc[-1]
            try:
                last_val = float(last_val)
            except Exception:
                last_val = 0.0
        else:
            last_val = 0.0
        future_df[r] = last_val

    # Ensure dtype consistency
    for r in reg_names:
        future_df[r] = pd.to_numeric(future_df[r], errors="coerce").fillna(0.0)

    # Run prediction
    try:
        forecast_df = model.predict(future_df)
    except Exception as e:
        raise RuntimeError(f"Prophet forecasting failed: {e}") from e

    # Extract relevant columns and convert to JSON-serializable records
    out_cols = ["ds", "yhat"]
    if "yhat_lower" in forecast_df.columns:
        out_cols.append("yhat_lower")
    if "yhat_upper" in forecast_df.columns:
        out_cols.append("yhat_upper")

    results = []
    for _, row in forecast_df[out_cols].iterrows():
        rec = {"ds": pd.Timestamp(row["ds"]).isoformat(), "yhat": float(row["yhat"])}
        if "yhat_lower" in row and not pd.isna(row.get("yhat_lower")):
            rec["yhat_lower"] = float(row["yhat_lower"])
        if "yhat_upper" in row and not pd.isna(row.get("yhat_upper")):
            rec["yhat_upper"] = float(row["yhat_upper"])
        results.append(rec)

    return results
