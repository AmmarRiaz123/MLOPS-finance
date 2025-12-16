from fastapi.testclient import TestClient
from app.main import app
import pytest

client = TestClient(app)

ROUTES = [
    ("GET", "/", None),
    ("GET", "/health/", None),
    ("POST", "/predict/direction", {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1000.0}),
    ("POST", "/predict/return/lightgbm", {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1000.0}),
    ("POST", "/predict/return/random-forest", {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1000.0}),
    ("POST", "/predict/volatility", {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5, "volume": 1000.0}),
    ("POST", "/forecast/price", {"periods": 5}),
    ("POST", "/predict/regime", {"returns_window": [0.01, -0.02], "volatility_window": [0.1, 0.12]}),
]

@pytest.mark.parametrize("method,path,payload", ROUTES)
def test_route_smoke(method, path, payload):
    if method == "GET":
        r = client.get(path)
    else:
        r = client.post(path, json=payload)
    # route should exist (not 404)
    assert r.status_code != 404
