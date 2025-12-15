from app.core.model_loader import get_model
from app.services.common import unwrap_model, build_features_from_ohlcv
import numpy as np

def predict_direction_from_ohlcv(ohlcv: dict, model_key: str = "lightgbm_up_down_model"):
    model_obj = get_model(model_key)
    if model_obj is None:
        raise RuntimeError(f"Model {model_key} not loaded")
    model, scaler, feat_names = unwrap_model(model_obj)
    features = build_features_from_ohlcv(ohlcv)
    if feat_names:
        x = np.array([features.get(f, 0.0) for f in feat_names], dtype=float).reshape(1, -1)
    else:
        keys = sorted(list(features.keys()))
        x = np.array([features[k] for k in keys], dtype=float).reshape(1, -1)
    if scaler is not None:
        try:
            x = scaler.transform(x)
        except Exception:
            pass
    proba = None
    try:
        proba = model.predict_proba(x)[0].max()
    except Exception:
        proba = None
    pred = model.predict(x)[0]
    return {"direction": "UP" if int(pred) == 1 else "DOWN", "probability": float(proba) if proba is not None else None}
