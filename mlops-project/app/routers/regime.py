from fastapi import APIRouter, HTTPException
from app.schemas.regime import RegimeRequest, RegimeResponse
from app.services.regime_service import predict_regime_from_windows

router = APIRouter()

@router.post("/", response_model=RegimeResponse)
def predict_regime(req: RegimeRequest):
    try:
        res = predict_regime_from_windows(req.returns_window, req.volatility_window, model_key="market_regime_hmm")
        return {"model": "market_regime_hmm", "regime_id": res["regime_id"], "regime_label": res["regime_label"], "probabilities": res.get("probabilities")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
