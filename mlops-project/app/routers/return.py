from fastapi import APIRouter, HTTPException
from app.schemas.ohlcv import OHLCVInput
from app.schemas.prediction import ReturnResponse

router = APIRouter()

@router.post("/lightgbm", response_model=ReturnResponse, summary="Predict Return Lightgbm")
def predict_lightgbm(req: OHLCVInput):
    """
    Stubbed LightGBM return prediction (safe for local testing).
    Uses OHLCVInput and returns a deterministic numeric predicted_return.
    Replace with real service call when feature mapping is implemented.
    """
    try:
        close = float(req.close)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid close value") from e
    pred = round((close % 1) * 0.01, 6)
    return {"model": "lightgbm_return_model", "predicted_return": pred}

@router.post("/random-forest", response_model=ReturnResponse, summary="Predict Return Random Forest")
def predict_rf(req: OHLCVInput):
    """
    Stubbed Random Forest return prediction.
    """
    try:
        close = float(req.close)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid close value") from e
    pred = round(((close % 1) * 0.02) - 0.0005, 6)
    return {"model": "random_forest_return_model", "predicted_return": pred}
