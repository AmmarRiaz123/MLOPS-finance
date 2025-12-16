from pathlib import Path
import importlib.util
import json
from typing import Dict, Any, List
import numpy as np
import joblib

from app.core.model_loader import MODEL_REGISTRY
from app.services.common import unwrap_model

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_training_module(folder: str):
    path = REPO_ROOT / "training" / folder / "features.py"
    if not path.exists():
        raise RuntimeError(f"Training features.py not found for {folder} at {path}")
    spec = importlib.util.spec_from_file_location(f"training.{folder}.features", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_training_features_list(folder: str) -> List[str]:
    tf = REPO_ROOT / "training" / folder / "metrics" / "latest" / "training_features.json"
    if not tf.exists():
        return []
    with open(tf, "r") as f:
        data = json.load(f)
    feats = data.get("features") or data.get("feature_names") or []
    return feats if isinstance(feats, list) else []


def _load_model_artifact(model_key: str, filename_fallback: str = "market_regime_hmm.pkl"):
    # prefer MODEL_REGISTRY, else load from disk
    artifact = MODEL_REGISTRY.get(model_key) if isinstance(MODEL_REGISTRY, dict) else None
    if artifact is not None:
        return artifact
    model_path = REPO_ROOT / "models" / "latest" / filename_fallback
    if not model_path.exists():
        raise RuntimeError(f"Model artifact not found at {model_path}")
    return joblib.load(model_path)


def predict_regime_from_windows(returns_window: List[float], volatility_window: List[float], model_key: str = "market_regime_hmm") -> Dict[str, Any]:
    """
    Predict regime using the model's own features.py and training feature ordering.
    Returns dict: {regime_id: int, regime_label: str, score: float|None}
    """
    training_folder = "market_regime_hmm"
    # 1) load artifact
    artifact = _load_model_artifact(model_key, filename_fallback="market_regime_hmm.pkl")
    model, scaler, saved_features = unwrap_model(artifact)

    # 2) import training features module and call its inference helper
    mod = _load_training_module(training_folder)
    builder = getattr(mod, "build_features_for_inference", None)
    if not callable(builder):
        # accept a dict window helper if present
        if hasattr(mod, "build_features_from_windows"):
            builder = getattr(mod, "build_features_from_windows")
    if not callable(builder):
        raise RuntimeError(
            f"No inference helper found in training/{training_folder}/features.py. "
            "Please implement build_features_for_inference(history=..., ohlcv=...) -> dict "
            "or build_features_from_windows(returns_window, volatility_window) -> dict"
        )

    # call builder with windows (preferred)
    try:
        # try direct windows signature first
        feat_dict = None
        try:
            feat_dict = builder(returns_window=returns_window, volatility_window=volatility_window)
        except TypeError:
            # fallback to history dict argument
            feat_dict = builder(history={"returns_window": returns_window, "volatility_window": volatility_window}, ohlcv=None)
    except Exception as e:
        raise RuntimeError(f"Feature builder failed: {e}")

    if not isinstance(feat_dict, dict):
        raise RuntimeError("Feature builder must return a dict of feature_name -> numeric value")

    # 3) canonical feature order
    feature_names = saved_features or _load_training_features_list(training_folder)
    if not feature_names:
        # fallback: use keys from feat_dict (but prefer explicit file)
        feature_names = list(feat_dict.keys())

    missing = [f for f in feature_names if f not in feat_dict]
    if missing:
        raise RuntimeError(f"Feature builder did not produce required features: missing {missing}")

    X_row = [float(feat_dict.get(f, 0.0)) for f in feature_names]
    X_np = np.array([X_row], dtype=float)

    # apply scaler if present
    if scaler is not None:
        try:
            X_np = scaler.transform(X_np)
        except Exception as e:
            # non-fatal but surface reason
            raise RuntimeError(f"Scaler transform failed: {e}")

    # 4) predict regime id
    try:
        regime_id = int(np.asarray(model.predict(X_np)).ravel()[0])
    except Exception as e:
        raise RuntimeError(f"Model predict failed: {e}")

    # 5) compute score/confidence
    score = None
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_np)
            score = float(np.max(proba))
        else:
            # try GaussianMixture / component means fallback
            means = getattr(model, "means_", None)
            if means is not None:
                # compute distances to component means and convert to pseudo-probabilities
                dists = np.linalg.norm(means - X_np[0], axis=1)
                inv = 1.0 / (1.0 + dists)
                probs = inv / (np.sum(inv) if np.sum(inv) > 0 else 1.0)
                score = float(np.max(probs))
            else:
                # try score_samples as a weak signal (normalize across components if possible)
                try:
                    logp = None
                    if hasattr(model, "score_samples"):
                        logp = model.score_samples(X_np)
                        # convert single logp to a 0-1 via logistic(shifted)
                        score = float(1.0 / (1.0 + np.exp(-float(logp[0]))))
                except Exception:
                    score = None
    except Exception:
        score = None

    # 6) map regime id -> human label
    label = None
    # prefer mapping saved in artifact
    if isinstance(artifact, dict):
        label_map = artifact.get("regime_labels") or artifact.get("label_map") or artifact.get("regime_map")
        if label_map:
            try:
                if isinstance(label_map, dict):
                    label = label_map.get(str(regime_id)) or label_map.get(regime_id)
                elif isinstance(label_map, list):
                    label = label_map[regime_id] if regime_id < len(label_map) else None
            except Exception:
                label = None

    # fallback: check training metrics for mapping keys
    if label is None:
        metrics_paths = [
            REPO_ROOT / "training" / training_folder / "metrics" / "latest" / "metrics.json",
            REPO_ROOT / "training" / training_folder / "metrics" / "latest" / "training_metadata.json",
            REPO_ROOT / "training" / training_folder / "metrics" / "latest" / "training_info.json",
        ]
        for p in metrics_paths:
            try:
                if p.exists():
                    with open(p, "r") as f:
                        md = json.load(f)
                    # look for common keys
                    for key in ("regime_map", "regime_labels", "label_map", "labels"):
                        if key in md:
                            lm = md[key]
                            if isinstance(lm, dict):
                                label = lm.get(str(regime_id)) or lm.get(regime_id)
                            elif isinstance(lm, list):
                                label = lm[regime_id] if regime_id < len(lm) else None
                            if label:
                                break
                if label:
                    break
            except Exception:
                continue

    # final fallback: infer semantic label by comparing component means on a return-like feature
    if label is None:
        try:
            means = getattr(model, "means_", None)
            if means is not None and len(means.shape) == 2:
                # find index of a return-like feature in feature_names
                search_names = ["ret_mean_5", "ret_mean_3", "log_return", "return_lag1", "daily_return"]
                idx = next((feature_names.index(n) for n in search_names if n in feature_names), None)
                if idx is not None and idx < means.shape[1]:
                    regime_means = means[:, idx]
                    # higher mean => bull, lower => bear, middle => neutral
                    order = np.argsort(regime_means)  # ascending
                    # map regime index to semantic label
                    semantic = {}
                    if len(order) >= 1:
                        # lowest -> bear
                        semantic[int(order[0])] = "bear"
                    if len(order) >= 2:
                        semantic[int(order[-1])] = "bull"
                    if len(order) >= 3:
                        # middle ones -> neutral
                        for mid in order[1:-1]:
                            semantic[int(mid)] = "neutral"
                    label = semantic.get(int(regime_id), f"regime_{regime_id}")
            else:
                label = f"regime_{regime_id}"
        except Exception:
            label = f"regime_{regime_id}"

    return {"regime_id": int(regime_id), "regime_label": label, "score": (None if score is None else round(float(score), 4))}
