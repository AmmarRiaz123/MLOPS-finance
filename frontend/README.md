# MLOps Finance Frontend

A Next.js dashboard for interacting with the MLOps Finance API.

## Features

- **Dashboard Overview**: Combined predictions from all models
- **Return Prediction**: LightGBM and Random Forest return models
- **Direction Prediction**: Market direction (up/down) classification
- **Volatility Prediction**: Next-day volatility forecasting
- **Price Forecasting**: Prophet-based time series forecasting
- **Regime Analysis**: Market regime detection (bull/bear/neutral)

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Set environment variables:
   ```bash
   cp .env.example .env.local
   # Edit .env.local with your API URL
   ```

3. Start development server:
   ```bash
   npm run dev
   ```

4. Open http://localhost:3000

## Environment Variables

- `NEXT_PUBLIC_API_URL`: FastAPI backend URL (default: http://127.0.0.1:8000)
- `DISCORD_WEBHOOK_URL`: Discord webhook URL used by the FastAPI backend alerting module (not used by the Next.js app directly)

## Usage

1. **Dashboard**: Enter OHLCV data to get predictions from all applicable models
2. **Individual Pages**: Dedicated forms for each model type
3. **Forecasting**: Enter number of periods for Prophet forecasting
4. **Regime Analysis**: Enter returns and volatility windows for HMM regime detection

## Components

- `OHLCVForm`: Reusable form for market data input
- `PredictionCard`: Displays model predictions with confidence
- `ForecastTable`: Table for Prophet forecast results
- `RegimeIndicator`: Visual indicator for market regime

## API Integration

All API calls are handled through modular client libraries in `/lib`:
- Error handling and loading states
- Configurable base URL
- JSON serialization/deserialization
