from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_root_exists_and_returns_info():
    r = client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert "docs" in data

def test_health_endpoint():
    r = client.get("/health/")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    # loaded_models may be present as list
    assert "loaded_models" in data
