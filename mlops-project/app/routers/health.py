from fastapi import APIRouter
from pathlib import Path
import json

from app.core.model_loader import MODEL_REGISTRY

router = APIRouter()

# mapping of model registry keys -> training folder names
_MODEL_TO_TRAINING = {
    "lightgbm_return_model": "lightgbm_return",
    "random_forest_return_model": "random_forest_return",
    "lightgbm_up_down_model": "lightgbm_up_down",
    "lightgbm_volatility_model": "lightgbm_volatility",
    "prophet_forecast": "prophet_forecast",
    "market_regime_hmm": "market_regime_hmm",
}

REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_training_features(model_key: str):
    folder = _MODEL_TO_TRAINING.get(model_key)
    # try training metrics first
    if folder:
        metrics_file = REPO_ROOT / "training" / folder / "metrics" / "latest" / "training_features.json"
        try:
            if metrics_file.exists():
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                feats = data.get("features") or data.get("feature_names") or []
                if isinstance(feats, list):
                    return feats
        except Exception:
            pass
    # fallback: try models/latest/feature_names.json
    model_feat = REPO_ROOT / "models" / "latest" / "feature_names.json"
    try:
        if model_feat.exists():
            with open(model_feat, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
            if isinstance(data, dict):
                return data.get("features") or data.get("feature_names") or []
    except Exception:
        pass
    return []

@router.get("/")
def health():
    loaded = list(MODEL_REGISTRY.keys()) if isinstance(MODEL_REGISTRY, dict) else []
    features_map = {}
    for k in loaded:
        try:
            features_map[k] = _read_training_features(k)
        except Exception:
            features_map[k] = []
    return {"status": "ok", "loaded_models": loaded, "model_expected_features": features_map}
