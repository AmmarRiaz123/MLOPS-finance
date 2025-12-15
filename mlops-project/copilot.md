You are working on a financial ML project focused on market prediction and analysis.
This prompt is the SINGLE SOURCE OF TRUTH for the entire project.

========================================================
PROJECT OVERVIEW
========================================================

This project builds an end-to-end ML system for financial market analysis using
daily OHLCV data with ~1,285 rows.

Columns:
- Date
- Open
- High
- Low
- Close
- Volume

The project is intentionally modular and production-oriented.
Raw data MUST NEVER be modified in-place because multiple models rely on it.

The goal is:
- Multiple ML models for different market tasks
- A FastAPI inference layer exposing all models
- Clean separation of data, training, inference, and serving

========================================================
EXISTING MODELS (ALREADY TRAINED)
========================================================

Models are saved as .pkl files and MUST ONLY be LOADED, not retrained.

1) lightgbm_return_model.pkl
   - Type: Regression
   - Task: Predict next-day return (continuous)

2) random_forest_return_model.pkl
   - Type: Regression
   - Task: Predict next-day return (continuous)

3) lightgbm_up_down_model.pkl
   - Type: Classification
   - Task: Predict market direction (UP / DOWN)

4) lightgbm_volatility_model.pkl
   - Type: Regression
   - Task: Predict market volatility

5) prophet_forecast.pkl
   - Type: Time-series forecast
   - Task: Predict future Close prices

6) Hidden Markov Model (HMM)
   - Type: Unsupervised classification
   - Task: Market regime detection (Bull / Neutral / Bear)
   - Uses returns + volatility windows
   - Metadata stored in JSON (regime stats, transitions)

========================================================
PIPELINE PHILOSOPHY
========================================================

- Raw data is immutable
- Feature engineering happens on-the-fly during inference
- Each model has its own service layer
- Shared schemas across models where possible
- No database (pure inference API)
- No authentication (internal / portfolio use)

========================================================
FASTAPI APPLICATION STRUCTURE
========================================================

app/
├── main.py                     # FastAPI entrypoint
├── core/
│   ├── config.py               # paths, settings
│   └── model_loader.py         # load models ONCE
├── schemas/
│   ├── ohlcv.py                # shared OHLCV input
│   ├── prediction.py           # regression/classification outputs
│   └── regime.py               # HMM regime outputs
├── services/
│   ├── return_service.py       # LGBM + RF returns
│   ├── direction_service.py    # up/down
│   ├── volatility_service.py   # volatility
│   ├── prophet_service.py      # forecasting
│   └── regime_service.py       # HMM inference
├── routers/
│   ├── return.py
│   ├── direction.py
│   ├── volatility.py
│   ├── prophet.py
│   └── regime.py
└── models/
    └── *.pkl

========================================================
API INPUT SCHEMAS
========================================================

Shared OHLCV input (used by most models):

- open: float
- high: float
- low: float
- close: float
- volume: float

Prophet input:
- periods: int (number of future days)

HMM input:
- returns_window: List[float]
- volatility_window: List[float]

========================================================
API ENDPOINTS
========================================================

POST /predict/return/lightgbm
POST /predict/return/random-forest
POST /predict/direction
POST /predict/volatility
POST /forecast/price
POST /predict/regime

Each endpoint:
- Loads model from model_loader
- Runs inference only
- Returns JSON-safe outputs
- Includes typing + docstrings

========================================================
IMPORTANT CONSTRAINTS
========================================================

- DO NOT modify raw data
- DO NOT retrain models
- DO NOT add databases or auth
- DO NOT mix training code with API code
- Focus on clean, readable, production-quality FastAPI code


========================================================
Project Structure
========================================================

## Project Architecture Overview

This repository follows a production-grade MLOps architecture:

- `app/` contains the FastAPI inference service
- `training/` contains offline model training code
- `prefect/` orchestrates retraining workflows
- `models/latest/` stores currently deployed models
- `models/archived/` stores versioned old models

FastAPI only performs inference.
Training never runs inside the API container.
Models are loaded once at startup via `model_loader.py`.

========================================================
ASSISTANT REVIEW
========================================================

- Reviewed by: GitHub Copilot (GPT-5 mini)
- Review date: 2025-12-15
- Status: I've read the project single-source copilot.md and am ready to continue working on the FastAPI inference service and related tasks as described.
