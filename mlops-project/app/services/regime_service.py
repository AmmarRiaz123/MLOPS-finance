from typing import Dict, Any, List
import numpy as np

from app.core.model_loader import MODEL_REGISTRY
from app.services.common import unwrap_model
from app.services.feature_mapper import get_training_features_for_model, build_feature_vector

def predict_regime_from_windows(returns_window: List[float], volatility_window: List[float], model_key: str = "market_regime_hmm") -> Dict[str, Any]:
    """
    Build features from windows (delegates to training feature module) and predict regime id/label and optional probabilities.
    """
    model_obj = MODEL_REGISTRY.get(model_key)
    if model_obj is None:
        raise RuntimeError(f"Model '{model_key}' not loaded")
    model, scaler, saved_features = unwrap_model(model_obj)
    feature_names = saved_features or get_training_features_for_model(model_key)
    if not feature_names:
        raise RuntimeError(f"No feature names available for model '{model_key}'")

    # build using feature mapper (it will call the training features module)
    fv = build_feature_vector(model_key, history={"returns_window": returns_window, "volatility_window": volatility_window})
    X_row = [float(fv.get(n, 0.0)) for n in feature_names]
    X_np = np.array([X_row], dtype=float)

    if scaler is not None:
        X_np = scaler.transform(X_np)

    try:
        regime_id = model.predict(X_np)[0]
    except Exception as e:
        raise RuntimeError(f"Regime prediction failed: {e}")

    # attempt probabilities/score
    score = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_np)
            # take max probability
            score = float(proba.max())
    except Exception:
        score = None

    # map numeric id to label if metadata exists (model artifact may include mapping; attempt to read)
    label = None
    try:
        # if model object saved mapping, it may be available via model._label_map or stored separately
        label = str(regime_id)
    except Exception:
        label = str(regime_id)

    return {"regime_id": int(regime_id), "regime_label": label, "probabilities": score}
