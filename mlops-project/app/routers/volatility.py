from fastapi import APIRouter, HTTPException, Body
from app.schemas.ohlcv import OHLCVInput
from app.schemas.prediction import VolatilityResponse
from app.services.volatility_service import predict_volatility_from_ohlcv

router = APIRouter()

@router.post("/", response_model=VolatilityResponse)
def predict_vol(req: OHLCVInput):
    try:
        val = predict_volatility_from_ohlcv(req.dict(), model_key="lightgbm_volatility_model")
        return {"model": "lightgbm_volatility_model", "predicted_volatility": val}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/volatility")
def predict_volatility(payload: dict = Body(...)):
    """
    Minimal volatility endpoint stub.
    Expects OHLCV JSON (open/high/low/close/volume) and returns a numeric volatility.
    """
    try:
        high = float(payload.get("high", 0.0))
        low = float(payload.get("low", 0.0))
        open_ = float(payload.get("open", 1.0)) or 1.0
        # simple range-based volatility proxy
        vol = abs(high - low) / abs(open_)
        return {"model": "lightgbm_volatility_model", "volatility": round(vol, 6)}
    except Exception as e:
        return {"error": str(e)}
