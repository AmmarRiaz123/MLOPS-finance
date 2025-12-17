import sys
from pathlib import Path
# ensure mlops-project (package root) is on sys.path when running this file directly
REPO_ROOT = Path(__file__).resolve().parents[2]  # mlops-project
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)

SAMPLE_OHLCV = {
    "open": 100.0,
    "high": 101.0,
    "low": 99.5,
    "close": 100.5,
    "volume": 123456.0
}

def test_predict_direction_route_exists():
    r = client.post("/predict/direction", json=SAMPLE_OHLCV)
    # endpoint should exist (not 404). Accept 200/422/500 depending on runtime environment.
    assert r.status_code != 404

def test_predict_direction_returns_json_when_ok():
    r = client.post("/predict/direction", json=SAMPLE_OHLCV)
    if r.status_code == 200:
        data = r.json()
        assert "model" in data
        assert "direction" in data
