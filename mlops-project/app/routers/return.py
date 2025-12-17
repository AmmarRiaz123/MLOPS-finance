from fastapi import APIRouter, HTTPException, Body
from app.schemas.prediction import ReturnResponse
from app.services.return_service import predict_with_model_from_ohlcv

router = APIRouter()

@router.post("/lightgbm", response_model=ReturnResponse, summary="Predict Return Lightgbm")
def predict_lightgbm(payload: dict = Body(...)):
    try:
        val = predict_with_model_from_ohlcv(payload, model_key="lightgbm_return_model")
        return {"model": "lightgbm_return_model", "predicted_return": val}
    except Exception as e:
        msg = str(e)
        # make the "needs history" situation a client error (not a server error)
        if "requires 'history'" in msg or "requires \"history\"" in msg:
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=500, detail=msg)

@router.post("/random-forest", response_model=ReturnResponse, summary="Predict Return Random Forest")
def predict_rf(payload: dict = Body(...)):
    try:
        val = predict_with_model_from_ohlcv(payload, model_key="random_forest_return_model")
        return {"model": "random_forest_return_model", "predicted_return": val}
    except Exception as e:
        msg = str(e)
        # make the "needs history" situation a client error (not a server error)
        if "requires 'history'" in msg or "requires \"history\"" in msg:
            raise HTTPException(status_code=400, detail=msg)
        raise HTTPException(status_code=500, detail=msg)
