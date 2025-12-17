# MLOps Finance API — Endpoint JSON Schemas & Examples

Notes:
- Shared OHLCV input used by many endpoints:
  - open: float
  - high: float
  - low: float
  - close: float
  - volume: float

- All responses are JSON. Errors use standard HTTP status codes and JSON {"detail": "..."}.

---

## 1) POST /predict/return/lightgbm
Predict next-day return (regression) using LightGBM.

Input (application/json):
{
  "open": 100.0,
  "high": 101.0,
  "low": 99.5,
  "close": 100.5,
  "volume": 123456.0
}

Output (200):
{
  "model": "lightgbm_return_model",
  "predicted_return": 0.00123
}

Example request:
POST /predict/return/lightgbm
Body: see Input above

Example response:
{
  "model": "lightgbm_return_model",
  "predicted_return": -0.00045
}

---

## 2) POST /predict/return/random-forest
Predict next-day return (regression) using Random Forest.

Input: same OHLCV JSON as above.

Output (200):
{
  "model": "random_forest_return_model",
  "predicted_return": 0.00098
}

---

## 3) POST /predict/direction
Predict market direction (classification: "up" / "down").

Input (application/json):
{
  "open": 100.0,
  "high": 101.0,
  "low": 99.5,
  "close": 100.5,
  "volume": 123456.0
}

Output (200):
{
  "model": "lightgbm_up_down_model",
  "direction": "up",
  "probability": 0.78   // optional: probability/confidence for the predicted class
}

Example:
{
  "model": "lightgbm_up_down_model",
  "direction": "down",
  "probability": 0.62
}

---

## 4) POST /predict/volatility
Predict next-day volatility (regression).

Input (application/json): same OHLCV JSON.

Output (200):
{
  "model": "lightgbm_volatility_model",
  "volatility": 0.01234   // model-specific volatility metric (e.g. std-dev or proxy)
}

Example:
{
  "model": "lightgbm_volatility_model",
  "volatility": 0.0087
}

---

## 5) POST /forecast/price
Prophet time-series forecast for future Close prices.

Input (application/json):
{
  "periods": 7,           // number of future days to forecast
  // Optionally, you may include recent OHLCV history if service supports it
}

Output (200):
{
  "model": "prophet_forecast",
  "forecast": [
    {"ds": "2025-12-16", "yhat": 101.23, "yhat_lower": 99.5, "yhat_upper": 103.0},
    {"ds": "2025-12-17", "yhat": 101.80, "yhat_lower": 99.9, "yhat_upper": 103.7},
    ...
  ]
}

Example:
{
  "model": "prophet_forecast",
  "forecast": [
    {"ds": "2025-12-16", "yhat": 101.23},
    {"ds": "2025-12-17", "yhat": 101.80}
  ]
}

---

## 6) POST /predict/regime
Hidden Markov Model (HMM) regime detection (Bull / Neutral / Bear).

Input (application/json):
{
  "returns_window": [0.001, -0.002, 0.003, ...],
  "volatility_window": [0.01, 0.012, 0.011, ...]
}

Output (200):
{
  "model": "hmm_regime_model",
  "regime": "bull",         // one of: "bull", "neutral", "bear"
  "score": 0.85             // optional confidence / regime score
}

Example:
{
  "model": "hmm_regime_model",
  "regime": "neutral",
  "score": 0.47
}

---

## Common error response
Status 4xx/5xx:
{
  "detail": "Descriptive error message"
}

---

End of API schema reference.
