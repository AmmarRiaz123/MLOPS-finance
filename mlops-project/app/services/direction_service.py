from typing import Dict, Any
import numpy as np

from app.core.model_loader import MODEL_REGISTRY
from app.services.common import unwrap_model
from app.services.feature_mapper import get_training_features_for_model, build_feature_vector

def predict_direction_from_ohlcv(payload: Dict[str, Any], model_key: str = "lightgbm_up_down_model") -> Dict[str, Any]:
    model_obj = MODEL_REGISTRY.get(model_key)
    if model_obj is None:
        raise RuntimeError(f"Model '{model_key}' not loaded")
    model, scaler, saved_features = unwrap_model(model_obj)
    feature_names = saved_features or get_training_features_for_model(model_key)
    if not feature_names:
        raise RuntimeError(f"No feature names available for model '{model_key}'")

    fv = build_feature_vector(model_key, history=payload.get("history"), ohlcv=payload)
    X_row = [float(fv.get(n, 0.0)) for n in feature_names]
    X_np = np.array([X_row], dtype=float)

    if scaler is not None:
        X_np = scaler.transform(X_np)

    try:
        pred_label = model.predict(X_np)
    except Exception as e:
        raise RuntimeError(f"Model predict failed: {e}")

    # probability if available
    prob = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_np)
            # assume binary: take class probability for predicted class
            cls_idx = int(pred_label[0])
            prob = float(proba[0, cls_idx]) if proba.shape[1] > cls_idx else None
    except Exception:
        prob = None

    return {"direction": str(pred_label[0]), "probability": prob}
