import joblib
from app.core.config import MODELS_DIR

# ensure this is assigned at module scope (services import it)
MODEL_REGISTRY = {}

def load_models():
    """Load all models into MODEL_REGISTRY at startup."""
    global MODEL_REGISTRY
    MODEL_REGISTRY = {}  # assignment makes `global MODEL_REGISTRY` meaningful (flake8 F824)
    
    if not MODELS_DIR.exists():
        return
    for p in MODELS_DIR.glob("*.pkl"):
        try:
            obj = joblib.load(p)
            MODEL_REGISTRY[p.stem] = obj
        except Exception:
            continue
    
    # Prophet model
    prophet_path = MODELS_DIR / "prophet_forecast.pkl"
    if prophet_path.exists():
        try:
            prophet_model = joblib.load(prophet_path)
            MODEL_REGISTRY["prophet_forecast"] = prophet_model
            print(f"Loaded Prophet model: {prophet_path}")
        except ImportError:
            print("Prophet not available - skipping Prophet model")
        except Exception as e:
            print(f"Failed to load Prophet model: {e}")
    else:
        print(f"Prophet model not found at {prophet_path}")

def get_model(name: str):
    return MODEL_REGISTRY.get(name)
