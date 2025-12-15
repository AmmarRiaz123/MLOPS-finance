from fastapi import FastAPI
from app.core.model_loader import load_models
from app.routers import return_router, direction, volatility, prophet, regime, health

app = FastAPI(title="MLOps Finance Inference API")

@app.on_event("startup")
def startup_event():
    load_models()

app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(return_router.router, prefix="/predict/return", tags=["return"])
app.include_router(direction.router, prefix="/predict", tags=["direction"])
app.include_router(volatility.router, prefix="/predict", tags=["volatility"])
app.include_router(prophet.router, prefix="/forecast", tags=["prophet"])
app.include_router(regime.router, prefix="/predict", tags=["regime"])
