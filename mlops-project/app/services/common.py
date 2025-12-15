from typing import Dict, List, Any
import numpy as np

def unwrap_model(obj: Any):
    """
    Unwrap saved pickle object which may be {model, scaler, features} or raw estimator.
    Returns (model, scaler, feature_names or None).
    """
    if isinstance(obj, dict):
        return obj.get("model"), obj.get("scaler"), obj.get("features")
    return obj, None, None

def build_features_from_ohlcv(ohlcv: Dict[str, float]) -> Dict[str, float]:
    """
    Minimal in-memory feature builder from a single OHLCV input.
    For production, client should supply full history; this provides reasonable defaults
    so models can accept the common OHLCVInput.
    """
    open_v = float(ohlcv.get("open", 0.0))
    high = float(ohlcv.get("high", 0.0))
    low = float(ohlcv.get("low", 0.0))
    close = float(ohlcv.get("close", 0.0))
    vol = float(ohlcv.get("volume", 0.0))

    ret = (close - open_v) / open_v if open_v else 0.0
    log_ret = np.log(close) - np.log(open_v) if open_v and close>0 else 0.0
    features = {
        "return_lag1": ret,
        "return_lag2": 0.0,
        "return_lag3": 0.0,
        "return_lag5": 0.0,
        "return_lag10": 0.0,
        "ma5": close,
        "ma10": close,
        "ma20": close,
        "std5": 0.0,
        "std10": 0.0,
        "std20": 0.0,
        "momentum_8": ret,
        "vol_ma5": vol,
        "vol_ma10": vol,
        "high_low_spread": (high - low) / close if close else 0.0,
        "open_close_spread": (close - open_v) / open_v if open_v else 0.0,
        "vol_x_std5": 0.0,
        "rsi_14": 0.5,
        "macd": 0.0,
        "macd_signal": 0.0,
        "stoch_k": 0.0,
        "stoch_d": 0.0
    }
    return features

def build_features_from_windows(returns_window: List[float], vol_window: List[float]) -> Dict[str, float]:
    """
    Build features expected by HMM/regime models from recent windows.
    """
    import numpy as _np
    f = {}
    rw = _np.array(returns_window[-10:]) if returns_window else _np.zeros(1)
    vw = _np.array(vol_window[-10:]) if vol_window else _np.zeros(1)
    f["log_return"] = float(rw[-1]) if rw.size>0 else 0.0
    f["std5"] = float(_np.std(rw[-5:])) if rw.size>0 else 0.0
    f["std10"] = float(_np.std(rw[-10:])) if rw.size>0 else 0.0
    f["ret_mean_3"] = float(_np.mean(rw[-3:])) if rw.size>0 else 0.0
    f["ret_std_3"] = float(_np.std(rw[-3:])) if rw.size>0 else 0.0
    f["momentum_8"] = float(rw[-1] - _np.mean(rw[-8:])) if rw.size>0 else 0.0
    f["vol_ma5"] = float(_np.mean(vw[-5:])) if vw.size>0 else 0.0
    f["vol_ma10"] = float(_np.mean(vw[-10:])) if vw.size>0 else 0.0
    return f
