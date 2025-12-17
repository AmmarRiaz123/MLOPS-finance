from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

def _find_date_col(df: pd.DataFrame) -> str:
    for c in ['Date','date','Datetime','datetime','ds','timestamp']:
        if c in df.columns:
            return c
    raise ValueError("No date column found in CSV")

def load_all_csvs(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and concatenate all CSVs from data_dir (sorted by date)."""
    files = sorted([p for p in Path(data_dir).glob("*.csv")])
    if not files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")
    dfs = []
    for p in files:
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    return full

def prepare_prophet_df(data_dir: Path = DATA_DIR, z_window: int = 30) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load CSVs, build Prophet-compatible dataframe (ds,y) and regressors.
    Normalize regressors using rolling z-score (window=z_window).
    Returns (df_prophet, regressors_list).
    """
    raw = load_all_csvs(data_dir)
    date_col = _find_date_col(raw)
    raw[date_col] = pd.to_datetime(raw[date_col], errors='coerce')
    raw = raw.dropna(subset=[date_col])
    raw = raw.sort_values(date_col).drop_duplicates(subset=[date_col], keep='first')
    raw = raw.set_index(date_col)

    # standardize columns
    raw = raw.rename(columns=lambda c: c.strip())

    # coerce numeric
    for c in ['Open','High','Low','Close','Volume','Adj Close']:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')

    if 'Close' not in raw.columns and 'Adj Close' in raw.columns:
        raw['Close'] = raw['Adj Close']

    df = raw.copy()

    # create base regressors
    df['high_low_spread'] = np.nan
    df['open_close_spread'] = np.nan
    df['volume'] = np.nan

    if {'High','Low','Close'}.issubset(df.columns):
        df['high_low_spread'] = (df['High'] - df['Low']) / df['Close']
    if {'Open','Close'}.issubset(df.columns):
        df['open_close_spread'] = (df['Close'] - df['Open']) / df['Open']
    if 'Volume' in df.columns:
        df['volume'] = df['Volume']

    # forward-fill minimal gaps (past-only) then fill remaining with 0
    df[['volume','high_low_spread','open_close_spread']] = df[['volume','high_low_spread','open_close_spread']].ffill().fillna(0)

    # normalize regressors with rolling z-score (past window)
    reg_cols = ['volume','high_low_spread','open_close_spread']
    for r in reg_cols:
        if r in df.columns:
            roll_mean = df[r].rolling(window=z_window, min_periods=1).mean()
            roll_std = df[r].rolling(window=z_window, min_periods=1).std().replace(0, np.nan)
            z = (df[r] - roll_mean) / roll_std
            # where std is nan (constant series), set z to 0
            z = z.fillna(0)
            df[r] = z

    # build Prophet frame on Close
    prophet_df = pd.DataFrame({
        'ds': df.index,
        'y': df['Close'].values
    }, index=df.index)

    regressors = []
    for r in reg_cols:
        if r in df.columns:
            prophet_df[r] = df[r].values
            regressors.append(r)

    prophet_df = prophet_df.dropna(subset=['y'])

    # final safety: ensure regressors have no NaNs
    prophet_df[regressors] = prophet_df[regressors].fillna(0)

    return prophet_df.reset_index(drop=True), regressors

def build_features_for_inference(history=None, ohlcv=None, z_window: int = 30) -> dict:
    """
    Build regressors for Prophet inference.
    - history: optional list[dict] or list[Pydantic] rows (oldest->newest)
    - ohlcv: optional single dict with open/high/low/close/volume
    Returns: dict {regressor_name: float} aligned to training_features.json ordering.
    """
    import pandas as _pd
    import numpy as _np
    from pathlib import Path as _Path
    import json as _json

    # If no input provided, return canonical regressors filled with zeros (frontend may call with only periods)
    metrics_file = _Path(__file__).resolve().parent / "metrics" / "latest" / "training_features.json"
    canonical = None
    if metrics_file.exists():
        try:
            with open(metrics_file, "r") as f:
                data = _json.load(f)
            canonical = data.get("features") or data.get("feature_names")
            if not isinstance(canonical, list):
                canonical = None
        except Exception:
            canonical = None
    if canonical is None:
        canonical = ["volume", "high_low_spread", "open_close_spread"]

    if history is None and ohlcv is None:
        # return zeros for each canonical regressor so frontend can request forecasts without history
        return {k: 0.0 for k in canonical}

    # prepare DataFrame from inputs
    if history:
        # convert Pydantic models to dicts if needed
        if hasattr(history[0], "dict"):
            rows = [r.dict() for r in history]
        else:
            rows = history
        df = _pd.DataFrame(rows).copy()
    elif ohlcv:
        if hasattr(ohlcv, "dict"):
            row = ohlcv.dict()
        else:
            row = ohlcv
        df = _pd.DataFrame([row]).copy()

    # normalize incoming column names -> expected names
    col_map = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "adj close": "Adj Close", "adj_close": "Adj Close",
        "volume": "Volume", "vol": "Volume",
        "date": "ds", "ds": "ds", "timestamp": "ds"
    }
    rename = {}
    for c in list(df.columns):
        key = str(c).lower()
        mapped = col_map.get(key)
        if mapped:
            rename[c] = mapped
    if rename:
        df = df.rename(columns=rename)

    # coerce numerics where present
    for c in ["Open","High","Low","Close","Adj Close","Volume","volume"]:
        if c in df.columns:
            df[c] = _pd.to_numeric(df[c], errors="coerce")

    # ensure Close exists (try Adj Close)
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]

    # compute base regressors if possible
    if {"High","Low","Close"}.issubset(set(df.columns)):
        df["high_low_spread"] = (df["High"] - df["Low"]) / df["Close"]
    else:
        df["high_low_spread"] = _np.nan

    if {"Open","Close"}.issubset(set(df.columns)):
        df["open_close_spread"] = (df["Close"] - df["Open"]) / df["Open"].replace(0, _np.nan)
    else:
        df["open_close_spread"] = _np.nan

    if "Volume" in df.columns:
        df["volume"] = df["Volume"]
    elif "volume" in df.columns:
        df["volume"] = df["volume"]
    else:
        df["volume"] = _np.nan

    # forward-fill then fill remaining with 0 (past-only)
    df[["volume","high_low_spread","open_close_spread"]] = df[["volume","high_low_spread","open_close_spread"]].ffill().fillna(0)

    # apply rolling z-score normalization consistent with prepare_prophet_df
    for r in ["volume","high_low_spread","open_close_spread"]:
        roll_mean = df[r].rolling(window=z_window, min_periods=1).mean()
        roll_std = df[r].rolling(window=z_window, min_periods=1).std().replace(0, _np.nan)
        z = (df[r] - roll_mean) / roll_std
        df[r] = z.fillna(0)

    # take last row and produce regressor dict aligned to canonical order
    last = df.iloc[-1]
    out = {}
    for k in canonical:
        val = last.get(k, None)
        try:
            out[k] = float(val) if not _pd.isna(val) else 0.0
        except Exception:
            out[k] = 0.0

    return out

if __name__ == "__main__":
    df, regs = prepare_prophet_df()
    print(f"Prepared Prophet df with {len(df)} rows and regressors: {regs}")
