from app.core.model_loader import get_model
from app.services.common import unwrap_model, build_features_from_ohlcv
import numpy as np

def _prepare_input_for_model(model_obj, features_dict):
    model, scaler, feat_names = unwrap_model(model_obj)
    if feat_names:
        x = np.array([features_dict.get(f, 0.0) for f in feat_names], dtype=float).reshape(1, -1)
    else:
        # stable ordering
        keys = sorted(list(features_dict.keys()))
        x = np.array([features_dict[k] for k in keys], dtype=float).reshape(1, -1)
    if scaler is not None:
        try:
            x = scaler.transform(x)
        except Exception:
            pass
    return model, x

def predict_with_model_from_ohlcv(ohlcv: dict, model_key: str):
    model_obj = get_model(model_key)
    if model_obj is None:
        raise RuntimeError(f"Model {model_key} not loaded")
    features = build_features_from_ohlcv(ohlcv)
    model, x = _prepare_input_for_model(model_obj, features)
    pred = model.predict(x)
    return float(pred[0])
