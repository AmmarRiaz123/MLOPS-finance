from app.core.model_loader import get_model
from app.services.common import unwrap_model, build_features_from_windows
import numpy as np

def predict_regime_from_windows(returns_window: list, volatility_window: list, model_key: str = "market_regime_hmm"):
    obj = get_model(model_key)
    if obj is None:
        raise RuntimeError(f"Model {model_key} not loaded")
    model, scaler, feat_names = unwrap_model(obj)
    features = build_features_from_windows(returns_window, volatility_window)
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
    pred = int(model.predict(x)[0])
    probs = None
    try:
        probs = model.predict_proba(x)[0].tolist()
    except Exception:
        probs = None
    # map regime id to label by sign of average returns saved in metrics (best-effort)
    # leave label determination to client; default mapping:
    label_map = {0: "Bull", 1: "Neutral", 2: "Bear"}
    label = label_map.get(pred, "Unknown")
    return {"regime_id": pred, "regime_label": label, "probabilities": probs}
