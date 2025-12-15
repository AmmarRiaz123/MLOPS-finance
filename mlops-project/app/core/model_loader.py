import joblib
from pathlib import Path
from typing import Dict, Any
from app.core.config import MODELS_DIR

MODEL_REGISTRY: Dict[str, Any] = {}

def load_models() -> None:
    MODEL_REGISTRY.clear()
    if not MODELS_DIR.exists():
        return
    for p in MODELS_DIR.glob("*.pkl"):
        try:
            obj = joblib.load(p)
            MODEL_REGISTRY[p.stem] = obj
        except Exception:
            continue

def get_model(name: str):
    return MODEL_REGISTRY.get(name)
