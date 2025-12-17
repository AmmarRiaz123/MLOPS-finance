from typing import Any

def unwrap_model(obj: Any):
    """
    Unwrap saved pickle object which may be {model, scaler, features} or raw estimator.
    Returns (model, scaler, feature_names or None).
    """
    if isinstance(obj, dict):
        return obj.get("model"), obj.get("scaler"), obj.get("features")
    return obj, None, None

