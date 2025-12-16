from typing import Dict, Any
import numpy as np

from app.core.model_loader import MODEL_REGISTRY
from app.services.common import unwrap_model
from app.services.feature_mapper import get_training_features_for_model, build_feature_vector

def predict_with_model_from_ohlcv(payload: Dict[str, Any], model_key: str = "lightgbm_return_model") -> float:
    """
    Build feature vector using training feature pipeline and run model.predict.
    """
    # load model artifact from registry
    model_obj = MODEL_REGISTRY.get(model_key)
    if model_obj is None:
        raise RuntimeError(f"Model '{model_key}' not loaded")

    model, scaler, saved_features = unwrap_model(model_obj)

    # canonical feature names
    feature_names = saved_features or get_training_features_for_model(model_key)
    if not feature_names:
        raise RuntimeError(f"No feature names available for model '{model_key}'")

    # build features (requires training feature mapping implemented)
    fv = build_feature_vector(model_key, history=payload.get("history"), ohlcv=payload)

    # order features
    X_row = [float(fv.get(n, 0.0)) for n in feature_names]
    X_np = np.array([X_row], dtype=float)

    if scaler is not None:
        try:
            X_np = scaler.transform(X_np)
        except Exception as e:
            raise RuntimeError(f"Scaler transform failed: {e}")

    try:
        pred = model.predict(X_np)
    except Exception as e:
        raise RuntimeError(f"Model prediction failed: {e}")

    # return scalar
    return float(pred[0])
