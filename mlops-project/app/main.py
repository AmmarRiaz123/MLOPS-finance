from fastapi import FastAPI, Request
import sys
from pathlib import Path
from contextlib import asynccontextmanager
from starlette.responses import Response

# ensure the repository package root (mlops-project) is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]  # mlops-project
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.model_loader import load_models  # noqa: E402

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

# --- Minimal CORS (only allow your Railway frontend) ---
ALLOWED_ORIGIN = "https://ml-project-production-2e4f.up.railway.app"

@app.middleware("http")
async def _simple_cors(request: Request, call_next):
    origin = request.headers.get("origin")

    is_preflight = (
        request.method == "OPTIONS"
        and origin is not None
        and request.headers.get("access-control-request-method") is not None
    )

    # --- Handle preflight requests ---
    if is_preflight:
        # only allow your Railway frontend
        if origin != ALLOWED_ORIGIN:
            return Response(status_code=400)

        return Response(
            status_code=204,
            headers={
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": request.headers.get("access-control-request-headers", "*"),
                "Access-Control-Max-Age": "86400",
                "Vary": "Origin",
            },
        )

    # --- Handle actual requests ---
    response = await call_next(request)

    if origin == ALLOWED_ORIGIN:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"

    return response

from app.routers import (  # noqa: E402
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
