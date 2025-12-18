"""
Feature engineering for LightGBM return regression.

Produces:
- X (DataFrame of features)
- y (Series of 1-day forward returns)
- dates (Series of ds corresponding to each row)
"""
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRS = [REPO_ROOT / "data", REPO_ROOT / "data" / "raw"]

def _find_csv_files() -> List[Path]:
    files = []
    for d in DATA_DIRS:
        if d.exists():
            files.extend(sorted([p for p in d.glob("*.csv")]))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIRS}")
    return files

def _load_all(files: List[Path]) -> pd.DataFrame:
    """Robust CSV loading: skip malformed lines and extra headers."""
    dfs = []
    for p in files:
        try:
            df = pd.read_csv(p, engine='python', on_bad_lines='skip')
        except TypeError:
            df = pd.read_csv(p, engine='python', error_bad_lines=False)  # older pandas fallback
        df.columns = [c.strip() for c in df.columns]
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    return full

def prepare_features(scale: bool = True, horizons: list | None = None) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:

    files = _find_csv_files()
    raw = _load_all(files)

    # detect date column robustly
    date_cols = [c for c in ['Date','date','Datetime','datetime','ds','timestamp'] if c in raw.columns]
    if not date_cols:
        raise ValueError("No date column found in CSVs")
    date_col = date_cols[0]
    raw[date_col] = pd.to_datetime(raw[date_col], errors='coerce')
    raw = raw.dropna(subset=[date_col])
    raw = raw.sort_values(date_col).drop_duplicates(subset=[date_col], keep='first')
    raw = raw.set_index(date_col)

    # coerce numeric OHLCV
    for c in ['Open','High','Low','Close','Adj Close','Volume']:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')
    if 'Close' not in raw.columns and 'Adj Close' in raw.columns:
        raw['Close'] = raw['Adj Close']

    df = raw.copy()
    # require valid Close
    df = df[pd.to_numeric(df['Close'], errors='coerce').notna()]

    # daily return (past)
    df['daily_return'] = df['Close'].pct_change()

    # rolling returns (periodic pct change)
    df['ret_3'] = df['Close'].pct_change(periods=3)
    df['ret_5'] = df['Close'].pct_change(periods=5)
    df['ret_10'] = df['Close'].pct_change(periods=10)

    # rolling stats on daily returns (3/5/10)
    for w in (3,5,10):
        df[f'ret_mean_{w}'] = df['daily_return'].rolling(w, min_periods=1).mean()
        df[f'ret_std_{w}']  = df['daily_return'].rolling(w, min_periods=1).std().fillna(0)
        # skew / kurtosis may be noisy on small windows but include for signal
        df[f'ret_skew_{w}'] = df['daily_return'].rolling(w, min_periods=1).skew().fillna(0)
        df[f'ret_kurt_{w}'] = df['daily_return'].rolling(w, min_periods=1).kurt().fillna(0)

    # provide a short-name std5 for backwards compatibility 
    if 'std5' not in df.columns:
        df['std5'] = df['daily_return'].rolling(5, min_periods=1).std().fillna(0)
    # also ensure ret_std_5 exists (redundant-safe)
    if 'ret_std_5' not in df.columns:
        df['ret_std_5'] = df['std5']

    # compact engineered features (prefer non-redundant set)
    df['return_lag1'] = df['daily_return'].shift(1)
    df['return_lag2'] = df['daily_return'].shift(2)

    df['ma5'] = df['Close'].rolling(5).mean()
    df['ma10'] = df['Close'].rolling(10).mean()
 
    # EWMA momentum
    df['ewma_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    # ensure momentum_8 always exists (safe division, fillna)
    df['momentum_8'] = (df['Close'] - df['ewma_8']) / df['ewma_8'].replace(0, np.nan)
    df['momentum_8'] = df['momentum_8'].fillna(0)

    # interactions
    # create interactions using safe lookups to avoid KeyError
    vol_vals = df['Volume'].fillna(0) if 'Volume' in df.columns else pd.Series(0, index=df.index)
    std5_vals = df['std5'].fillna(0) if 'std5' in df.columns else pd.Series(0, index=df.index)
    mom8_vals = df['momentum_8'].fillna(0) if 'momentum_8' in df.columns else pd.Series(0, index=df.index)
    df['vol_x_std5'] = vol_vals * std5_vals
    df['vol_x_mom8'] = vol_vals * mom8_vals

    # RSI scaled 0-1
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14, min_periods=1).mean()
    roll_down = loss.rolling(14, min_periods=1).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    df['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50) / 100.0

    # volume/volatility interactions (normalized)
    if 'Volume' in df.columns:
        df['vol_ma5'] = df['Volume'].rolling(5).mean().fillna(0)
    else:
        df['vol_ma5'] = 0.0
    df['high_low_spread'] = ((df.get('High', df['Close']) - df.get('Low', df['Close'])) / df['Close']).fillna(0)
    df['vol_x_std5'] = df['vol_ma5'] * df['std5'].fillna(0)

    # choose a compact candidate set (8-12 features)
    candidate_features = [
        'return_lag1','return_lag2',
        'ret_3','ret_5','ret_10',
        'ret_mean_5','ret_std_5',
        'momentum_8','rsi_14',
        'vol_ma5','high_low_spread','vol_x_std5'
    ]

    # keep only features that exist and are numeric
    # intersect candidate list with actual df columns (avoid KeyError later)
    feature_cols = [c for c in candidate_features if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    if not feature_cols:
        raise RuntimeError("No engineered features available")

    # forward-fill (past-only) then fill remaining with 0 and drop constant / near-constant
    df[feature_cols] = df[feature_cols].ffill().fillna(0)
    nunique = df[feature_cols].nunique(dropna=True)
    stds = df[feature_cols].std(ddof=0)
    kept = [c for c in feature_cols if nunique.get(c,0) > 1 and not np.isclose(stds.get(c,0), 0.0)]
    if not kept:
        kept = feature_cols
    feature_cols = kept

    # multi-day targets
    if horizons is None:
        horizons = [1]
    # ensure target columns exist for all requested horizons (safe in-place creation)
    for h in horizons:
        col = f'return_{h}d'
        if col not in df.columns:
            future_close = df['Close'].shift(-h)
            df[col] = (future_close - df['Close']) / df['Close']

    # build X/y/dates aligned, drop rows where target is NaN
    X_full = df[feature_cols].copy()
    target_name = f'return_{horizons[0]}d'
    # safe drop: ensure column exists (should after creation above) then drop NaNs
    if target_name not in df.columns:
        # last-resort compute
        df[target_name] = (df['Close'].shift(-horizons[0]) - df['Close']) / df['Close']
    df = df.dropna(subset=[target_name])
    y = df[target_name].copy()
    dates = df.index.to_series()
    # align X to y index
    X = X_full.loc[y.index]

    # scaling optional
    scaler = None
    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

    X = X.fillna(0)
    y = y.fillna(0)

    if scale:
        return X, y, dates, scaler
    return X, y, dates

def build_features_for_inference(history=None, ohlcv=None):

    import pandas as _pd
    import numpy as _np

    # Build a small dataframe using history if provided, else fall back to single-row from ohlcv
    if history:
        df = _pd.DataFrame(history).copy()
    elif ohlcv:
        df = _pd.DataFrame([ohlcv]).copy()
    else:
        raise RuntimeError("Either history or ohlcv must be provided for inference.")

    # --- Normalise column names from API (lowercase keys) to training column names (Title case) ---
    col_map = {
        "open": "Open", "high": "High", "low": "Low",
        "close": "Close", "adj close": "Adj Close", "adj_close": "Adj Close",
        "volume": "Volume", "vol": "Volume",
        "date": "Date", "ds": "Date", "timestamp": "Date"
    }
    renamed = {}
    for c in df.columns:
        key = str(c).lower()
        mapped = col_map.get(key, None)
        if mapped:
            renamed[c] = mapped
    if renamed:
        df = df.rename(columns=renamed)

    # ensure date idx if present
    for c in ['Date','date','ds','timestamp']:
        if c in df.columns or (c == 'Date' and 'Date' in df.columns):
            try:
                df['Date'] = _pd.to_datetime(df.get('Date') or df.get(c), errors='coerce')
                df = df.sort_values('Date').reset_index(drop=True)
            except Exception:
                pass
            break

    # Validate required column presence early and give a clear error
    if 'Close' not in df.columns:
        raise RuntimeError(f"Feature builder requires column 'Close' (case-sensitive). Available columns: {list(df.columns)}. Provide OHLCV fields as keys 'open','high','low','close','volume' or column names matching training CSV.")

    # coerce numeric columns used by prepare_features
    for c in ['Open','High','Low','Close','Adj Close','Volume']:
        if c in df.columns:
            df[c] = _pd.to_numeric(df[c], errors='coerce')

    # replicate minimal transforms used in prepare_features to create needed features
    df['daily_return'] = df['Close'].pct_change()
    df['ret_3'] = df['Close'].pct_change(periods=3)
    df['ret_5'] = df['Close'].pct_change(periods=5)
    df['ret_10'] = df['Close'].pct_change(periods=10)
    for w in (3,5,10):
        df[f'ret_mean_{w}'] = df['daily_return'].rolling(w, min_periods=1).mean()
        df[f'ret_std_{w}']  = df['daily_return'].rolling(w, min_periods=1).std().fillna(0)
    if 'std5' not in df.columns:
        df['std5'] = df['daily_return'].rolling(5, min_periods=1).std().fillna(0)
    df['return_lag1'] = df['daily_return'].shift(1)
    df['return_lag2'] = df['daily_return'].shift(2)
    df['ewma_8'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['momentum_8'] = (df['Close'] - df['ewma_8']) / df['ewma_8'].replace(0, _np.nan)
    df['momentum_8'] = df['momentum_8'].fillna(0)
    if 'Volume' in df.columns:
        df['vol_ma5'] = df['Volume'].rolling(5).mean().fillna(0)
    else:
        df['vol_ma5'] = 0.0
    df['high_low_spread'] = ((df.get('High', df['Close']) - df.get('Low', df['Close'])) / df['Close']).fillna(0)
    df['vol_x_std5'] = df['vol_ma5'] * df['std5'].fillna(0)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    roll_up = gain.rolling(14, min_periods=1).mean()
    roll_down = loss.rolling(14, min_periods=1).mean()
    rs = roll_up / roll_down.replace(0, _np.nan)
    df['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50) / 100.0

    # candidate features (same order as training)
    candidate_features = [
        'return_lag1','return_lag2',
        'ret_3','ret_5','ret_10',
        'ret_mean_5','ret_std_5',
        'momentum_8','rsi_14',
        'vol_ma5','high_low_spread','vol_x_std5'
    ]

    # take last row as inference input
    last = df.iloc[-1]
    feat_dict = {}
    for f in candidate_features:
        val = last.get(f, None)
        feat_dict[f] = float(val) if _pd.notna(val) else 0.0

    return feat_dict

if __name__ == "__main__":
    res = prepare_features()
    # handle both signatures: (X, y, dates) or (X, y, dates, scaler)
    if isinstance(res, tuple):
        if len(res) == 4:
            X, y, dates, _scaler = res
        elif len(res) == 3:
            X, y, dates = res
        else:
            raise RuntimeError(f"prepare_features returned unexpected tuple length: {len(res)}")
    else:
        raise RuntimeError("prepare_features did not return a tuple as expected")
    print(f"Prepared features: X.shape={getattr(X, 'shape', None)}, y.shape={getattr(y, 'shape', None)}")
