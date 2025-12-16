from fastapi import APIRouter, HTTPException
from app.schemas.ohlcv import OHLCVInput
from app.schemas.prediction import VolatilityResponse
from app.services.volatility_service import predict_volatility_from_ohlcv

router = APIRouter()

@router.post("/volatility", response_model=VolatilityResponse, summary="Predict Volatility")
def predict_volatility(req: OHLCVInput):
    try:
        val = predict_volatility_from_ohlcv(req.dict(), model_key="lightgbm_volatility_model")
        # return both fields for compatibility with frontend and schema
        return {"model": "lightgbm_volatility_model", "predicted_volatility": val, "volatility": val}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
