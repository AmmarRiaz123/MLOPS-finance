from app.core.model_loader import get_model
from typing import List, Dict, Any

def forecast_prophet(periods: int = 3, model_key: str = "prophet_forecast"):
    obj = get_model(model_key)
    if obj is None:
        raise RuntimeError("Prophet model not loaded")
    model = obj if not isinstance(obj, dict) else obj.get("model", obj)
    try:
        future = model.make_future_dataframe(periods=periods, include_history=False)
        forecast = model.predict(future)
        rows = []
        for _, r in forecast.iterrows():
            rows.append({
                "date": str(r["ds"]),
                "yhat": float(r.get("yhat", None)),
                "yhat_lower": float(r.get("yhat_lower", None)) if "yhat_lower" in r else None,
                "yhat_upper": float(r.get("yhat_upper", None)) if "yhat_upper" in r else None
            })
        return rows
    except Exception as e:
        raise RuntimeError(f"Prophet forecasting failed: {e}")
