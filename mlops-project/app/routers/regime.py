from fastapi import APIRouter, HTTPException, Body
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

@router.post("/regime", response_model=RegimeResponse, summary="Predict Regime with confidence score")
def predict_regime(payload: RegimeRequest = Body(...)):
    """
    Heuristic regime detector that returns a regime label and a confidence score.
    Score is derived from the magnitude of mean return (larger absolute mean => higher confidence),
    and modestly adjusted downwards when recent volatility is large.
    Replace with real HMM service call when available.
    """
    rets = payload.returns_window or []
    vols = payload.volatility_window or []

    # compute mean return and average volatility
    try:
        mean_ret = float(sum(rets) / len(rets)) if rets else 0.0
    except Exception:
        mean_ret = 0.0
    try:
        avg_vol = float(sum(vols) / len(vols)) if vols else 0.0
    except Exception:
        avg_vol = 0.0

    # label thresholds
    if mean_ret > 0.001:
        regime = "bull"
    elif mean_ret < -0.001:
        regime = "bear"
    else:
        regime = "neutral"

    # confidence score: map mean return magnitude to [0,1] using 0.001 as scaling; cap at 1.0.
    # then reduce confidence when volatility is large (simple multiplicative adjustment).
    base_score = min(1.0, abs(mean_ret) / 0.001) if rets else 0.5
    vol_adjust = 1.0 if avg_vol == 0 else max(0.2, 1.0 - avg_vol)
    score = round(min(1.0, base_score * vol_adjust), 3)

    return {"model": "hmm_regime_model", "regime": regime, "score": score}
