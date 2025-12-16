from fastapi import FastAPI
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware  # <- added

# ensure the repository package root (mlops-project) is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]  # mlops-project
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.model_loader import load_models

# Use lifespan handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    load_models()
    try:
        yield
    finally:
        # optional shutdown logic
        pass

app = FastAPI(title="MLOps Finance Inference API", lifespan=lifespan)

# --- CORS configuration ---
origins = [
    "http://localhost:3000",  # Next.js dev server
    "http://127.0.0.1:3000",
    # add your production frontend URL here when deployed
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # allow POST, GET, OPTIONS, etc.
    allow_headers=["*"],
)

# import routers
from app.routers import (
    health,
    direction,
    volatility,
    prophet,
    regime,
    return_router,
)

# register routers
app.include_router(health, prefix="/health", tags=["health"])
app.include_router(return_router, prefix="/predict/return", tags=["return"])
app.include_router(direction, prefix="/predict", tags=["direction"])
app.include_router(volatility, prefix="/predict", tags=["volatility"])
app.include_router(prophet, prefix="/forecast", tags=["prophet"])
app.include_router(regime, prefix="/predict", tags=["regime"])

# remove accidental /predict root routes
_app_routes = []
for r in app.router.routes:
    path = getattr(r, "path", None)
    if path in ("/predict", "/predict/"):
        continue
    _app_routes.append(r)
app.router.routes = _app_routes

# minimal root endpoint
@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "MLOps Finance Inference API",
        "health": "/health",
        "docs": "/docs"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
