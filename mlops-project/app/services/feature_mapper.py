import importlib.util
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]  # mlops-project
TRAINING_DIR = REPO_ROOT / "training"

# mapping model keys (used in MODEL_REGISTRY / callers) to training subfolders
_MODEL_TO_TRAINING = {
    "lightgbm_return_model": "lightgbm_return",
    "random_forest_return_model": "random_forest_return",
    "lightgbm_up_down_model": "lightgbm_up_down",
    "lightgbm_volatility_model": "lightgbm_volatility",
    "prophet_forecast": "prophet_forecast",
    "market_regime_hmm": "market_regime_hmm",
}


def _load_module_from_file(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_training_features_for_model(model_key: str) -> List[str]:
    """
    Prefer training/<model>/metrics/latest/training_features.json if present,
    otherwise fallback to inspecting training/<model>/features.py as before.
    """
    folder = _MODEL_TO_TRAINING.get(model_key)
    if not folder:
        return []
    metrics_file = TRAINING_DIR / folder / "metrics" / "latest" / "training_features.json"
    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                data = json.load(f)
            feats = data.get("features") or data.get("feature_names") or []
            if isinstance(feats, list):
                return feats
        except Exception:
            pass

    # fallback to previous heuristic/inspection of features.py
    features_py = TRAINING_DIR / folder / "features.py"
    if not features_py.exists():
        return []
    try:
        mod = _load_module_from_file(features_py)
    except Exception:
        return []
    cand = getattr(mod, "candidate_features", None)
    if isinstance(cand, (list, tuple)) and all(isinstance(x, str) for x in cand):
        return list(cand)
    # try calling prepare_* functions as best-effort (may read CSVs)
    for fn_name in ("prepare_features", "prepare_ml_data", "prepare_prophet_df"):
        fn = getattr(mod, fn_name, None)
        if callable(fn):
            try:
                res = fn(scale=False) if fn_name == "prepare_features" else fn()
                if isinstance(res, tuple) and len(res) >= 1:
                    X = res[0]
                    try:
                        return list(X.columns)
                    except Exception:
                        continue
                if fn_name == "prepare_prophet_df" and isinstance(res, tuple) and len(res) >= 2:
                    _, regressors = res
                    return list(regressors)
            except Exception:
                continue
    return []


def build_feature_vector(model_key: str,
                         history: Optional[List[Dict[str, Any]]] = None,
                         ohlcv: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Build a feature dict keyed by canonical feature names for the given model_key.
    Strict behavior:
      - Import training/<model>/features.py and attempt to call one of:
          build_features_for_inference(history=..., ohlcv=...) |
          prepare_features_from_df(df) |
          map_input(ohlcv, history)
      - If none present, but history is a list/dict, try to call prepare_features_from_df if available.
      - Otherwise raise RuntimeError instructing how to add a mapping function.
    Returns: dict {feature_name: numeric_value}
    """
    folder = _MODEL_TO_TRAINING.get(model_key)
    if not folder:
        raise RuntimeError(f"Unknown model_key: {model_key}")

    training_features = get_training_features_for_model(model_key)
    if not training_features:
        raise RuntimeError(f"Feature list not found for model {model_key}; ensure training saved training_features.json")

    features_py = TRAINING_DIR / folder / "features.py"
    if not features_py.exists():
        raise RuntimeError(f"Training features.py not found for {folder} at {features_py}")

    mod = _load_module_from_file(features_py)

    # Preferred inference helper names
    for helper in ("build_features_for_inference", "prepare_features_from_df", "prepare_features_for_inference", "map_input"):
        fn = getattr(mod, helper, None)
        if callable(fn):
            try:
                # call with both supported signatures when possible
                if helper == "prepare_features_from_df":
                    if history is None:
                        raise RuntimeError("prepare_features_from_df requires recent history dataframe")
                    df = pd.DataFrame(history)
                    res = fn(df)
                    # expect res returns X (DataFrame) or dict
                    if isinstance(res, dict):
                        return {k: float(res.get(k, 0.0)) for k in training_features}
                    if isinstance(res, tuple) and len(res) >= 1:
                        X = res[0]
                        row = X.iloc[-1] if not X.empty else X
                        return {k: float(row.get(k, 0.0)) for k in training_features}
                else:
                    # build_features_for_inference or map_input likely accept history and ohlcv
                    res = fn(history=history, ohlcv=ohlcv) if "history" in fn.__code__.co_varnames else fn(ohlcv, history)
                    if isinstance(res, dict):
                        return {k: float(res.get(k, 0.0)) for k in training_features}
                    # sometimes returns pandas Series/DF row
                    if hasattr(res, "to_dict"):
                        d = res.to_dict()
                        return {k: float(d.get(k, 0.0)) for k in training_features}
            except Exception as e:
                raise RuntimeError(f"Feature builder '{helper}' failed: {e}") from e

    # Last-resort: if history provided and the training features module exposes prepare_features (risky)
    if history and hasattr(mod, "prepare_features_from_history"):
        try:
            df = pd.DataFrame(history)
            res = mod.prepare_features_from_history(df)
            if isinstance(res, dict):
                return {k: float(res.get(k, 0.0)) for k in training_features}
        except Exception:
            pass

    raise RuntimeError(
        f"No compatible inference feature builder found in training/{folder}/features.py.\n"
        "Please implement one of the helper functions with signature:\n"
        "  build_features_for_inference(history=[...], ohlcv={...}) -> dict\n"
        "or\n"
        "  prepare_features_from_df(df: pd.DataFrame) -> (X, ...)\n"
        "so the app can generate model-ready feature vectors."
    )
