from pathlib import Path
import importlib.util
import json
from typing import List, Dict, Any, Optional
import joblib
import pandas as pd
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

def _load_training_features_helper() -> Any:
    path = REPO_ROOT / "training" / "prophet_forecast" / "features.py"
    if not path.exists():
        raise RuntimeError(f"Training features.py not found: {path}")
    spec = importlib.util.spec_from_file_location("training.prophet_forecast.features", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    builder = getattr(mod, "build_features_for_inference", None)
    if not callable(builder):
        raise RuntimeError("build_features_for_inference not found in training/prophet_forecast/features.py")
    return builder

def _load_training_feature_names() -> List[str]:
    tf = REPO_ROOT / "training" / "prophet_forecast" / "metrics" / "latest" / "training_features.json"
    if tf.exists():
        try:
            with open(tf, "r") as f:
                data = json.load(f)
            feats = data.get("features") or data.get("feature_names")
            if isinstance(feats, list) and feats:
                return feats
        except Exception:
            pass
    # fallback to default regressors used in training
    return ["volume", "high_low_spread", "open_close_spread"]

def _load_model_artifact():
    model_path = REPO_ROOT / "models" / "latest" / "prophet_forecast.pkl"
    if not model_path.exists():
        raise RuntimeError(f"Prophet model artifact not found: {model_path}")
    obj = joblib.load(model_path)
    # support either raw model or dict-wrapped artifact
    if isinstance(obj, dict):
        # common keys: "model", or direct Prophet inside dict
        if "model" in obj:
            return obj["model"]
    return obj

def forecast_prophet(periods: int, history: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    periods: positive int
    history: optional list of OHLCV dicts oldest->newest (must include ds if available)
    Returns list of forecast dicts with ds, yhat, optional yhat_lower/yhat_upper
    """
    if not isinstance(periods, int) or periods <= 0:
        raise RuntimeError("periods must be a positive integer")

    # 1) load training helper and canonical regressors
    builder = _load_training_features_helper()
    reg_names = _load_training_feature_names()

    # 2) build single-row regressors from training helper
    try:
        # convert history to list[dict] if Pydantic models passed; caller should pass dicts
        feat_dict = builder(history=history) if "history" in builder.__code__.co_varnames else builder(history)
    except TypeError:
        # try positional fallback
        feat_dict = builder(history)
    except Exception as e:
        raise RuntimeError(f"Feature builder failed: {e}")

    if not isinstance(feat_dict, dict):
        raise RuntimeError("Feature builder must return a dict of regressor_name->value")

    # 3) prepare future dataframe with ds and regressors
    # determine start date
    start = None
    if history:
        try:
            hist_df = pd.DataFrame(history)
            if "ds" in hist_df.columns:
                hist_df["ds"] = pd.to_datetime(hist_df["ds"], errors="coerce")
                valid = hist_df["ds"].dropna()
                if not valid.empty:
                    start = valid.max() + pd.Timedelta(days=1)
        except Exception:
            start = None
    if start is None or pd.isna(start):
        start = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)

    try:
        future_dates = pd.date_range(start=start, periods=periods, freq="D")
    except Exception as e:
        raise RuntimeError(f"Failed to create future dates: {e}")

    future_df = pd.DataFrame({"ds": future_dates})

    # fill regressors using feat_dict values aligned to reg_names (default 0.0)
    for r in reg_names:
        v = feat_dict.get(r, feat_dict.get(r.lower(), 0.0))
        try:
            v = float(v)
        except Exception:
            v = 0.0
        future_df[r] = v

    # 4) load model artifact and predict
    model = _load_model_artifact()
    try:
        forecast_df = model.predict(future_df)
    except Exception as e:
        raise RuntimeError(f"Prophet model prediction failed: {e}")

    # 5) serialize results
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
