from fastapi import APIRouter
from app.core.model_loader import MODEL_REGISTRY

router = APIRouter()

@router.get("/")
def health():
    return {"status": "ok", "loaded_models": list(MODEL_REGISTRY.keys())}
