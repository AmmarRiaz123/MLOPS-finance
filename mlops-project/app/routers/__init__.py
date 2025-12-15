import importlib
from typing import Any

def _load_router(name: str) -> Any:
    mod = importlib.import_module(f"app.routers.{name}")
    return getattr(mod, "router")

# load routers (ensure these modules exist)
health = _load_router("health")
direction = _load_router("direction")
volatility = _load_router("volatility")
prophet = _load_router("prophet")
regime = _load_router("regime")
# 'return' is a Python keyword — expose as return_router
return_router = _load_router("return")

__all__ = ["health", "direction", "volatility", "prophet", "regime", "return_router"]