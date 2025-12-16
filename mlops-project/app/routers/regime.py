from fastapi import APIRouter, HTTPException, Body
from app.schemas.regime import RegimeRequest, RegimeResponse
from app.services.regime_service import predict_regime_from_windows

router = APIRouter()

@router.post("/regime", response_model=RegimeResponse, summary="Predict Regime (HMM)")
def predict_regime(req: RegimeRequest = Body(...)):
    try:
        res = predict_regime_from_windows(req.returns_window, req.volatility_window, model_key="market_regime_hmm")
        # normalize service return to expected fields
        regime_label = res.get("regime_label") or res.get("regime") or res.get("label")
        score = res.get("score") or res.get("probability") or res.get("confidence")
        return {"model": "market_regime_hmm", "regime": regime_label, "score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
