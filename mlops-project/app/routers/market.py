from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any, Dict, List, Optional
import base64
from io import BytesIO

import pandas as pd
import numpy as np

# headless plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from app.services.return_service import predict_with_model_from_ohlcv
from app.services.direction_service import predict_direction_from_ohlcv
from app.services.volatility_service import predict_volatility_from_ohlcv
from app.services.regime_service import predict_regime_from_windows
from app.services import prophet_service

router = APIRouter()

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"

class MarketAnalyzeRequest(BaseModel):
    analysis_window: int = Field(30, ge=5, le=365)
    forecast_period: int = Field(7, ge=1, le=30)

def _find_date_col(df: pd.DataFrame) -> str:
    for c in ["Date", "date", "Datetime", "datetime", "ds", "timestamp"]:
        if c in df.columns:
            return c
    raise RuntimeError(f"No date column found. Available columns: {list(df.columns)}")

def _load_all_csvs(data_dir: Path) -> pd.DataFrame:
    files = sorted([p for p in data_dir.glob("*.csv")])
    if not files:
        raise RuntimeError(f"No CSV files found in {data_dir}")

    dfs = []
    for p in files:
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        dfs.append(df)

    full = pd.concat(dfs, ignore_index=True)
    return full

def _normalize_ohlcv_cols(df: pd.DataFrame) -> pd.DataFrame:
    # normalize common OHLCV naming to Title case used in many training scripts
    rename = {}
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc == "open": rename[c] = "Open"
        elif lc == "high": rename[c] = "High"
        elif lc == "low": rename[c] = "Low"
        elif lc == "close": rename[c] = "Close"
        elif lc in ("adj close", "adj_close"): rename[c] = "Adj Close"
        elif lc == "volume": rename[c] = "Volume"
    if rename:
        df = df.rename(columns=rename)
    return df

def _to_base64_png(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def _regime_to_numeric(label: str, regime_id: Optional[int] = None) -> float:
    if label is None:
        return float(regime_id) if regime_id is not None else 0.0
    l = str(label).lower()
    if "bear" in l:
        return -1.0
    if "bull" in l:
        return 1.0
    if "neutral" in l:
        return 0.0
    return float(regime_id) if regime_id is not None else 0.0

@router.post("/analyze", summary="Aggregate market analysis (models + plots)")
def analyze_market(req: MarketAnalyzeRequest) -> Dict[str, Any]:
    try:
        raw = _load_all_csvs(DATA_DIR)
        raw = _normalize_ohlcv_cols(raw)

        date_col = _find_date_col(raw)
        raw[date_col] = pd.to_datetime(raw[date_col], errors="coerce")
        raw = raw.dropna(subset=[date_col]).sort_values(date_col)
        raw = raw.drop_duplicates(subset=[date_col], keep="last")

        # ensure required columns exist
        needed = {"Open", "High", "Low", "Close", "Volume"}
        missing_cols = [c for c in needed if c not in raw.columns]
        if missing_cols:
            raise RuntimeError(f"Missing required OHLCV columns in data CSVs: {missing_cols}")

        window_df = raw.tail(int(req.analysis_window)).copy()
        if len(window_df) < req.analysis_window:
            raise RuntimeError(f"Not enough rows for analysis_window={req.analysis_window}. Available={len(window_df)}")

        # coerce numeric
        for c in ["Open", "High", "Low", "Close", "Volume"]:
            window_df[c] = pd.to_numeric(window_df[c], errors="coerce")
        window_df = window_df.dropna(subset=["Open", "High", "Low", "Close", "Volume"])
        if window_df.empty:
            raise RuntimeError("No valid OHLCV rows after numeric coercion")

        # latest OHLCV payload for point predictions
        last = window_df.iloc[-1]
        ohlcv_payload = {
            "open": float(last["Open"]),
            "high": float(last["High"]),
            "low": float(last["Low"]),
            "close": float(last["Close"]),
            "volume": float(last["Volume"]),
        }

        # build history payload for Prophet (and for any models that accept history)
        history: List[Dict[str, Any]] = []
        for _, r in window_df.iterrows():
            history.append({
                "date": pd.Timestamp(r[date_col]).date().isoformat(),
                "open": float(r["Open"]),
                "high": float(r["High"]),
                "low": float(r["Low"]),
                "close": float(r["Close"]),
                "volume": float(r["Volume"]),
            })

        # --- model calls (reuse existing services; no retraining; no new feature pipelines) ---
        ret_lgb = predict_with_model_from_ohlcv(dict(ohlcv_payload), model_key="lightgbm_return_model")
        ret_rf = predict_with_model_from_ohlcv(dict(ohlcv_payload), model_key="random_forest_return_model")

        direction_res = predict_direction_from_ohlcv(dict(ohlcv_payload), model_key="lightgbm_up_down_model")
        vol_val = predict_volatility_from_ohlcv(dict(ohlcv_payload), model_key="lightgbm_volatility_model")

        # regime: build returns/vol windows from aggregated close series
        closes = window_df["Close"].astype(float).values
        returns = pd.Series(closes).pct_change().fillna(0.0).values
        rolling_vol = pd.Series(returns).rolling(window=min(10, len(returns)), min_periods=1).std().fillna(0.0).values

        # "current" regime using entire available window vectors
        regime_res = predict_regime_from_windows(
            returns_window=[float(x) for x in returns.tolist()],
            volatility_window=[float(x) for x in rolling_vol.tolist()],
            model_key="market_regime_hmm",
        )

        # prophet forecast (uses its service; expects regressors derived from history)
        forecast_rows = prophet_service.forecast_prophet(periods=int(req.forecast_period), history=history)

        metrics = {
            "return": {
                "lightgbm_return_model": float(ret_lgb),
                "random_forest_return_model": float(ret_rf),
            },
            "direction": direction_res,
            "volatility": {"value": float(vol_val)},
            "regime": regime_res,
            "forecast": {"periods": int(req.forecast_period), "points": forecast_rows},
        }

        # --- plots ---
        ds = pd.to_datetime(window_df[date_col]).dt.date.astype(str).tolist()

        # 1) price trend
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(ds, window_df["Close"].astype(float).values, linewidth=2)
        ax1.set_title("Price Trend (Close)")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Close")
        ax1.tick_params(axis="x", rotation=45)
        ax1.grid(True, alpha=0.3)
        price_trend_b64 = _to_base64_png(fig1)

        # 2) returns
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(ds, (pd.Series(returns) * 100.0).values, linewidth=1.5)
        ax2.set_title("Returns (%)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Return %")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3)
        returns_b64 = _to_base64_png(fig2)

        # 3) volatility (rolling std)
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(ds, rolling_vol, linewidth=1.5)
        ax3.set_title("Rolling Volatility (std of returns)")
        ax3.set_xlabel("Date")
        ax3.set_ylabel("Volatility")
        ax3.tick_params(axis="x", rotation=45)
        ax3.grid(True, alpha=0.3)
        vol_b64 = _to_base64_png(fig3)

        # 4) regime over time (simple per-time recompute with expanding windows)
        regime_vals = []
        for i in range(len(returns)):
            w = min(10, i + 1)
            r_win = [float(x) for x in returns[max(0, i - w + 1): i + 1].tolist()]
            v_win = [float(x) for x in rolling_vol[max(0, i - w + 1): i + 1].tolist()]
            try:
                rr = predict_regime_from_windows(r_win, v_win, model_key="market_regime_hmm")
                regime_vals.append(_regime_to_numeric(rr.get("regime_label"), rr.get("regime_id")))
            except Exception:
                regime_vals.append(np.nan)

        fig4, ax4 = plt.subplots(figsize=(10, 3.5))
        ax4.plot(ds, regime_vals, marker="o", linewidth=1.0)
        ax4.set_title("Market Regime (numeric)")
        ax4.set_xlabel("Date")
        ax4.set_ylabel("Regime")
        ax4.set_yticks([-1, 0, 1])
        ax4.set_yticklabels(["bear", "neutral", "bull"])
        ax4.tick_params(axis="x", rotation=45)
        ax4.grid(True, alpha=0.3)
        regime_b64 = _to_base64_png(fig4)

        # 5) forecast plot (historical + yhat)
        f_ds = [row["ds"] for row in forecast_rows]
        f_yhat = [row["yhat"] for row in forecast_rows]

        fig5, ax5 = plt.subplots(figsize=(10, 4))
        ax5.plot(ds, window_df["Close"].astype(float).values, label="Historical Close", linewidth=2)
        ax5.plot(f_ds, f_yhat, label="Prophet Forecast (yhat)", linewidth=2)
        ax5.set_title("Historical Close + Forecast")
        ax5.set_xlabel("Date")
        ax5.set_ylabel("Price")
        ax5.tick_params(axis="x", rotation=45)
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        forecast_b64 = _to_base64_png(fig5)

        return {
            "analysis_window": int(req.analysis_window),
            "forecast_period": int(req.forecast_period),
            "metrics": metrics,
            "plots": {
                "price_trend": price_trend_b64,
                "returns": returns_b64,
                "volatility": vol_b64,
                "regime": regime_b64,
                "forecast": forecast_b64,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
