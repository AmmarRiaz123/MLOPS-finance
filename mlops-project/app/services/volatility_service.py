from pathlib import Path
import importlib.util
import json
from typing import Dict, Any
import numpy as np
import joblib

REPO_ROOT = Path(__file__).resolve().parents[2]

def _load_training_module(folder: str):
    path = REPO_ROOT / "training" / folder / "features.py"
    if not path.exists():
        raise RuntimeError(f"Training features.py not found for {folder} at {path}")
    spec = importlib.util.spec_from_file_location(f"training.{folder}.features", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _load_training_features_list(folder: str):
    tf = REPO_ROOT / "training" / folder / "metrics" / "latest" / "training_features.json"
    if not tf.exists():
        raise RuntimeError(f"training_features.json not found for {folder}. Re-run training to generate: {tf}")
    with open(tf, "r") as f:
        data = json.load(f)
    feats = data.get("features") or data.get("feature_names")
    if not isinstance(feats, list) or not feats:
        raise RuntimeError(f"Invalid training_features.json for {folder}: {tf}")
    return feats

def _load_model_file(model_filename: str):
    model_path = REPO_ROOT / "models" / "latest" / model_filename
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")
    obj = joblib.load(model_path)
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj

def predict_volatility_from_ohlcv(payload: Dict[str, Any], model_key: str = "lightgbm_volatility_model") -> float:
    """
    Predict volatility using the model's own features.py and canonical training_features.json.
    """
    if model_key != "lightgbm_volatility_model":
        raise RuntimeError(f"Unsupported model_key: {model_key}")

    training_folder = "lightgbm_volatility"
    model_filename = "lightgbm_volatility_model.pkl"

    feature_names = _load_training_features_list(training_folder)
    mod = _load_training_module(training_folder)
    builder = getattr(mod, "build_features_for_inference", None)
    if not callable(builder):
        raise RuntimeError(f"No inference helper (build_features_for_inference) in training/{training_folder}/features.py")

    history = payload.get("history")
    try:
        feat_dict = builder(history=history, ohlcv=payload) if "history" in builder.__code__.co_varnames else builder(payload)
    except Exception as e:
        raise RuntimeError(f"Feature builder failed: {e}")

    if not isinstance(feat_dict, dict):
        raise RuntimeError("Feature builder must return a dict of feature_name -> numeric value")

    missing = [f for f in feature_names if f not in feat_dict]
    if missing:
        raise RuntimeError(f"Feature builder did not produce required features: missing {missing}")

    X_row = [float(feat_dict.get(f, 0.0)) for f in feature_names]
    X_np = np.array([X_row], dtype=float)

    model = _load_model_file(model_filename)

    try:
        pred = model.predict(X_np)
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    return float(np.asarray(pred).ravel()[0])
