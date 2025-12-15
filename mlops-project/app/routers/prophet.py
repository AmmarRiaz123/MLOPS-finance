from fastapi import APIRouter, HTTPException
from app.schemas.ohlcv import ProphetRequest
from app.schemas.prediction import ProphetResponse
from app.services.prophet_service import forecast_prophet

router = APIRouter()

@router.post("/price", response_model=ProphetResponse)
def forecast_price(req: ProphetRequest):
    try:
        # pass optional history when present in the request (service will handle None)
        history = getattr(req, "history", None)
        rows = forecast_prophet(periods=req.periods, history=history, model_key="prophet_forecast")
        return {"model": "prophet_forecast", "forecast": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
