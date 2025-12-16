from fastapi import APIRouter, HTTPException
from app.schemas.ohlcv import OHLCVInput
from app.schemas.prediction import DirectionResponse
from app.services.direction_service import predict_direction_from_ohlcv

router = APIRouter()

@router.post("/direction", response_model=DirectionResponse, summary="Predict Direction")
def predict_direction(req: OHLCVInput):
    try:
        res = predict_direction_from_ohlcv(req.dict(), model_key="lightgbm_up_down_model")
        return {"model": "lightgbm_up_down_model", "direction": res["direction"], "probability": res.get("probability")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
