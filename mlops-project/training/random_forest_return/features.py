"""
Feature engineering for random_forest_return.
Provides prepare_features(horizon=1, features=None, scale=False) -> X, y, dates[, scaler]
"""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRS = [REPO_ROOT / "data", REPO_ROOT / "data" / "raw"]

def _find_csv_files():
    files = []
    for d in DATA_DIRS:
        if d.exists():
            files.extend(sorted([p for p in d.glob("*.csv")]))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIRS}")
    return files

def _load_all(files: List[Path]) -> pd.DataFrame:
    dfs = []
    for p in files:
        try:
            df = pd.read_csv(p, engine="python", on_bad_lines="skip")
        except TypeError:
            df = pd.read_csv(p, engine="python", error_bad_lines=False)  # older pandas fallback
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def prepare_features(horizon: int = 1,
                     feature_list: Optional[List[str]] = None,
                     scale: bool = False) -> Tuple[pd.DataFrame, pd.Series, pd.Series] or Tuple[pd.DataFrame, pd.Series, pd.Series, object]:
    """
    Load CSVs and prepare features and target.
    horizon: days ahead to predict (1,3,5)
    feature_list: optional list of features to keep; defaults to compact set
    scale: if True returns (X, y, dates, scaler)
    """
    files = _find_csv_files()
    raw = _load_all(files)

    # Date parsing
    date_cols = [c for c in ['Date','date','Datetime','datetime','ds','timestamp'] if c in raw.columns]
    if not date_cols:
        raise ValueError("No date column found in CSVs")
    dc = date_cols[0]
    raw[dc] = pd.to_datetime(raw[dc], errors='coerce')
    raw = raw.dropna(subset=[dc])
    raw = raw.sort_values(dc).drop_duplicates(subset=[dc], keep='first').set_index(dc)

    # Coerce numerics
    for c in ['Open','High','Low','Close','Adj Close','Volume']:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')
    if 'Close' not in raw.columns and 'Adj Close' in raw.columns:
        raw['Close'] = raw['Adj Close']

    df = raw.copy()
    # require Close
    df = df[pd.to_numeric(df['Close'], errors='coerce').notna()]

    # log/linear returns
    df['return'] = df['Close'].pct_change()  # simple returns
    df['log_return'] = np.log(df['Close']).diff()

    # Targets: future returns (horizon)
    df[f'return_{horizon}d'] = (df['Close'].shift(-horizon) - df['Close']) / df['Close']

    # Lagged returns
    df['return_lag1'] = df['return'].shift(1)
    df['return_lag2'] = df['return'].shift(2)
    df['return_lag3'] = df['return'].shift(3)
    df['return_lag5'] = df['return'].shift(5)
    df['return_lag10'] = df['return'].shift(10)

    # Moving averages
    df['ma5'] = df['Close'].rolling(5).mean()
    df['ma10'] = df['Close'].rolling(10).mean()
    df['ma20'] = df['Close'].rolling(20).mean()

    # Rolling std
    df['std5'] = df['return'].rolling(5).std()
    df['std10'] = df['return'].rolling(10).std()
    df['std20'] = df['return'].rolling(20).std()

    # Momentum
    df['ewma_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['momentum_8'] = (df['Close'] - df['ewma_8']) / df['ewma_8'].replace(0, np.nan)
    df['momentum_8'] = df['momentum_8'].fillna(0)

    # Volatility indicators / volume
    if 'Volume' in df.columns:
        df['volume'] = df['Volume'].fillna(0)
        df['vol_ma5'] = df['volume'].rolling(5).mean().fillna(0)
        df['vol_ma10'] = df['volume'].rolling(10).mean().fillna(0)
    else:
        df['volume'] = 0.0
        df['vol_ma5'] = 0.0
        df['vol_ma10'] = 0.0

    # Spreads
    if {'High','Low'}.issubset(df.columns):
        df['high_low_spread'] = (df['High'] - df['Low']) / df['Close']
    else:
        df['high_low_spread'] = 0.0
    if {'Open','Close'}.issubset(df.columns):
        df['open_close_spread'] = (df['Close'] - df['Open']) / df['Open'].replace(0, np.nan)
    else:
        df['open_close_spread'] = 0.0

    # Interactions
    df['vol_x_std5'] = df['vol_ma5'].fillna(0) * df['std5'].fillna(0)

    # Technical indicators
    # RSI 14 (0-1)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14).mean()
    roll_down = loss.rolling(14).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50) / 100.0

    # MACD & signal
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    df['macd'] = macd.fillna(0)
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean().fillna(0)

    # Stochastic %K/%D
    low_min = df['Low'].rolling(14).min() if 'Low' in df.columns else df['Close'].rolling(14).min()
    high_max = df['High'].rolling(14).max() if 'High' in df.columns else df['Close'].rolling(14).max()
    stoch_k = ((df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)).fillna(0)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = df['stoch_k'].rolling(3).mean().fillna(0)

    # Default compact features (flexible)
    default_features = [
        'return_lag1','return_lag2','return_lag3','return_lag5','return_lag10',
        'ma5','ma10','ma20','std5','std10','std20',
        'momentum_8','vol_ma5','vol_ma10','high_low_spread','open_close_spread','vol_x_std5',
        'rsi_14','macd','macd_signal','stoch_k','stoch_d'
    ]

    features = feature_list if feature_list is not None else default_features
    # intersect with available numeric columns
    features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]

    if not features:
        raise RuntimeError("No valid features available after selection")

    # forward-fill and fill remaining NaNs with 0 (past-only)
    df[features] = df[features].ffill().fillna(0)

    # drop rows where target is NaN (end of series)
    target_col = f'return_{horizon}d'
    if target_col not in df.columns:
        df[target_col] = (df['Close'].shift(-horizon) - df['Close']) / df['Close']
    df = df.dropna(subset=[target_col])

    X = df[features].loc[df.index].copy()
    y = df[target_col].loc[df.index].copy()
    dates = df.index.to_series().loc[df.index].copy()

    # align X with y (drop head rows from lags)
    X = X.loc[y.index]

    scaler = None
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        return X, y, dates, scaler

    return X, y, dates
