from pathlib import Path
import importlib.util
import json
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import joblib
import pandas as pd

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

def _load_model_artifact(model_filename: str):
    model_path = REPO_ROOT / "models" / "latest" / model_filename
    if not model_path.exists():
        raise RuntimeError(f"Model file not found: {model_path}")
    return joblib.load(model_path)

def _unwrap_model_and_features(artifact: Any) -> Tuple[Any, Optional[List[str]]]:
    """
    Supports either raw estimators or dict-wrapped artifacts:
      {"model": ..., "feature_names": [...]} (or similar)
    Returns: (model, saved_feature_names|None)
    """
    if isinstance(artifact, dict) and "model" in artifact:
        model = artifact.get("model")
        saved = artifact.get("feature_names") or artifact.get("features") or artifact.get("feature_list")
        return model, (saved if isinstance(saved, list) else None)
    return artifact, None

def _extract_feature_names_from_model(model: Any) -> Optional[List[str]]:
    """
    Try to get the exact feature ordering the model was trained with.
    Works for:
      - lightgbm.Booster (feature_name())
      - sklearn LGBMRegressor/LGBMClassifier (feature_name_ or booster_.feature_name())
    """
    # LightGBM Booster
    try:
        fn = getattr(model, "feature_name", None)
        if callable(fn):
            names = fn()
            if isinstance(names, list) and names:
                return names
    except Exception:
        pass

    # sklearn LightGBM wrapper sometimes stores names here
    try:
        names = getattr(model, "feature_name_", None)
        if isinstance(names, (list, tuple)) and names:
            return list(names)
    except Exception:
        pass

    # wrapper -> underlying booster
    try:
        booster = getattr(model, "booster_", None)
        if booster is not None and hasattr(booster, "feature_name"):
            names = booster.feature_name()
            if isinstance(names, list) and names:
                return names
    except Exception:
        pass

    return None

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
        # This model was trained on history-derived features (ret_3, ret_mean_5, rsi_14, vol_x_std5).
        # With no/short history those features collapse to constants -> constant prediction.
        history = payload.get("history")
        if not isinstance(history, list) or len(history) < 6:
            raise RuntimeError(
                "lightgbm_return_model requires 'history' (list of OHLCV rows oldest->newest). "
                "Provide at least 6 rows (more is better; ~15 improves RSI/rolling stability)."
            )
    elif model_key == "random_forest_return_model":
        training_folder = "random_forest_return"
        model_filename = "random_forest_return_model.pkl"
    else:
        raise RuntimeError(f"Unsupported model_key for this simple service: {model_key}")

    # 1) load artifact/model first so we can honor the model's trained feature ordering
    artifact = _load_model_artifact(model_filename)
    model, saved_feature_names = _unwrap_model_and_features(artifact)
    model_feature_names = saved_feature_names or _extract_feature_names_from_model(model)

    # 2) fallback canonical feature order (training_features.json) if model doesn't expose names
    canonical_feature_names = _load_training_features_list(training_folder)

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

    # 3) choose the feature order to send into the model
    # prefer model-trained ordering; fallback to canonical training_features.json
    feature_names = model_feature_names or canonical_feature_names

    # detect missing names
    missing = [f for f in feature_names if f not in feat_dict]
    if missing:
        raise RuntimeError(
            f"Feature builder did not produce required features: missing {missing}. "
            f"(Model expects {len(feature_names)} features; canonical list has {len(canonical_feature_names)}.)"
        )

    X_row = [float(feat_dict.get(f, 0.0)) for f in feature_names]
    # Use a DataFrame so LightGBM/sklearn can align by column names (and avoid name warnings)
    X_df = pd.DataFrame([X_row], columns=feature_names)

    try:
        pred = model.predict(X_df)
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    # return scalar float
    return float(np.asarray(pred).ravel()[0])
