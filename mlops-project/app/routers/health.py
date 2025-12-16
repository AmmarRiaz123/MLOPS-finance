from fastapi import APIRouter
from app.core.model_loader import MODEL_REGISTRY
from app.services.feature_mapper import get_training_features_for_model

router = APIRouter()

@router.get("/")
def health():
    loaded = list(MODEL_REGISTRY.keys()) if isinstance(MODEL_REGISTRY, dict) else []
    features_map = {k: get_training_features_for_model(k) for k in loaded}
    return {"status": "ok", "loaded_models": loaded, "model_expected_features": features_map}
