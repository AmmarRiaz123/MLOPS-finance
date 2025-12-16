from fastapi import APIRouter, HTTPException
from typing import Optional
from app.schemas.ohlcv import ProphetRequest
from app.schemas.prediction import ProphetResponse
from app.services import prophet_service

router = APIRouter()

@router.post("/price", response_model=ProphetResponse, summary="Prophet price forecast")
def forecast_price(req: ProphetRequest):
    try:
        history = None
        if req.history:
            # ensure list of dicts
            history = [h.dict() if hasattr(h, "dict") else h for h in req.history]
        rows = prophet_service.forecast_prophet(periods=req.periods, history=history)
        return {"model": "prophet_forecast", "forecast": rows}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
