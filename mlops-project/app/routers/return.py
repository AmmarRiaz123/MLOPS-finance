from fastapi import APIRouter, HTTPException
from app.schemas.ohlcv import OHLCVInput
from app.schemas.prediction import ReturnResponse
from app.services.return_service import predict_with_model_from_ohlcv

router = APIRouter()

@router.post("/lightgbm", response_model=ReturnResponse)
def predict_lightgbm(req: OHLCVInput):
    try:
        val = predict_with_model_from_ohlcv(req.dict(), model_key="lightgbm_return_model")
        return {"model": "lightgbm_return_model", "predicted_return": val}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/random-forest", response_model=ReturnResponse)
def predict_rf(req: OHLCVInput):
    try:
        val = predict_with_model_from_ohlcv(req.dict(), model_key="random_forest_return_model")
        return {"model": "random_forest_return_model", "predicted_return": val}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
