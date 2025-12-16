from pathlib import Path
import importlib.util
import json
from typing import Dict, Any
import numpy as np
import joblib

REPO_ROOT = Path(__file__).resolve().parents[2]  # mlops-project

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
    # support either raw estimator or dict-wrapped artifact
    if isinstance(obj, dict) and "model" in obj:
        return obj["model"]
    return obj

def predict_with_model_from_ohlcv(payload: Dict[str, Any], model_key: str = "lightgbm_return_model") -> float:
    """
    Minimal inference: use model's own features.py and training_features.json.
    payload: OHLCV dict (open, high, low, close, volume, optional history)
    model_key: mapping to training folder / model filename expected by convention
    """
    # mapping conventions (explicit, boring)
    if model_key == "lightgbm_return_model":
        training_folder = "lightgbm_return"
        model_filename = "lightgbm_return_model.pkl"
    elif model_key == "random_forest_return_model":
        training_folder = "random_forest_return"
        model_filename = "random_forest_return_model.pkl"
    else:
        raise RuntimeError(f"Unsupported model_key for this simple service: {model_key}")

    # 1) load canonical feature order (training_features.json)
    feature_names = _load_training_features_list(training_folder)

    # 2) import training features module and call its inference helper
    mod = _load_training_module(training_folder)
    builder = getattr(mod, "build_features_for_inference", None)
    if not callable(builder):
        raise RuntimeError(
            f"No inference helper found in training/{training_folder}/features.py. "
            f"Please implement build_features_for_inference(history=..., ohlcv=...) -> dict"
        )

    # allow optional history in payload
    history = payload.get("history")
    # call builder with ohlcv payload (the training helper must accept these args)
    try:
        feat_dict = builder(history=history, ohlcv=payload) if "history" in builder.__code__.co_varnames else builder(payload)
    except Exception as e:
        raise RuntimeError(f"Feature builder failed: {e}")

    if not isinstance(feat_dict, dict):
        raise RuntimeError("Feature builder must return a dict of feature_name -> numeric value")

    # 3) align features in exact training order and detect missing names
    missing = [f for f in feature_names if f not in feat_dict]
    if missing:
        raise RuntimeError(f"Feature builder did not produce required features: missing {missing}")

    X_row = [float(feat_dict.get(f, 0.0)) for f in feature_names]
    X_np = np.array([X_row], dtype=float)

    # 4) load model and predict
    model = _load_model_file(model_filename)
    try:
        pred = model.predict(X_np)
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    # return scalar float
    return float(np.asarray(pred).ravel()[0])
