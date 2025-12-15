from fastapi import FastAPI
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# ensure the repository package root (mlops-project) is on sys.path so
# `import app.*` works when running this file directly.
REPO_ROOT = Path(__file__).resolve().parents[1]  # mlops-project
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.model_loader import load_models

# Use lifespan handler instead of deprecated on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    load_models()
    try:
        yield
    finally:
        # optional shutdown logic can go here
        pass

app = FastAPI(title="MLOps Finance Inference API", lifespan=lifespan)

# import routers exposed by the package
from app.routers import (
    health,
    direction,
    volatility,
    prophet,
    regime,
    return_router,
)

# register routers (each variable is an APIRouter object)
app.include_router(health, prefix="/health", tags=["health"])
app.include_router(return_router, prefix="/predict/return", tags=["return"])
app.include_router(direction, prefix="/predict", tags=["direction"])
app.include_router(volatility, prefix="/predict", tags=["volatility"])
app.include_router(prophet, prefix="/forecast", tags=["prophet"])
app.include_router(regime, prefix="/predict", tags=["regime"])

# Remove any accidental route that maps exactly to the "/predict" or "/predict/" root
# (this can occur if a router defines @router.post("/") and is mounted at prefix "/predict")
_app_routes = []
for r in app.router.routes:
    path = getattr(r, "path", None)
    if path in ("/predict", "/predict/"):
        # skip accidental root-of-predict route
        continue
    _app_routes.append(r)
app.router.routes = _app_routes

# minimal root endpoint so GET / doesn't 404
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "MLOps Finance Inference API",
        "health": "/health",
        "docs": "/docs"
    }

# Run locally when executed as a script
if __name__ == "__main__":
    import uvicorn
    # Bind to localhost so it's available on local machine
    uvicorn.run(app, host="127.0.0.1", port=8000)
