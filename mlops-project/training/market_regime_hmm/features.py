from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRS = [REPO_ROOT / "data", REPO_ROOT / "data" / "raw"]

def _find_csv_files() -> List[Path]:
	paths: List[Path] = []
	for d in DATA_DIRS:
		if d.exists():
			paths.extend(sorted(d.glob("*.csv")))
	if not paths:
		raise FileNotFoundError(f"No CSV files found in {DATA_DIRS}")
	return paths

def _load_all(files: List[Path]) -> pd.DataFrame:
	dfs = []
	for p in files:
		try:
			df = pd.read_csv(p, engine="python", on_bad_lines="skip")
		except TypeError:
			df = pd.read_csv(p, engine="python", error_bad_lines=False)
		df.columns = [c.strip() for c in df.columns]
		dfs.append(df)
	return pd.concat(dfs, ignore_index=True)

def prepare_features(window_vol_short: int = 5,
                     window_vol_long: int = 10,
                     zscore: bool = True,
                     return_series: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.Series, Optional[StandardScaler]]:
	"""
	Returns: X (features DataFrame), returns (Series), vol_series (Series), scaler (if zscore True else None)
	Features are standardized (z-score) when zscore=True.
	"""
	files = _find_csv_files()
	df = _load_all(files)

	# date parse + sort
	date_cols = [c for c in ['Date','date','Datetime','datetime','ds','timestamp'] if c in df.columns]
	if not date_cols:
		raise ValueError("No date column found in CSVs")
	dc = date_cols[0]
	df[dc] = pd.to_datetime(df[dc], errors='coerce')
	df = df.dropna(subset=[dc]).sort_values(dc).drop_duplicates(subset=[dc], keep='first').set_index(dc)

	# coerce numeric
	for c in ['Open','High','Low','Close','Adj Close','Volume']:
		if c in df.columns:
			df[c] = pd.to_numeric(df[c], errors='coerce')
	if 'Close' not in df.columns and 'Adj Close' in df.columns:
		df['Close'] = df['Adj Close']

	# drop rows without Close
	df = df[pd.to_numeric(df['Close'], errors='coerce').notna()]

	# log returns
	df['log_price'] = np.log(df['Close'])
	df['log_return'] = df['log_price'].diff()

	# rolling volatility
	df[f'std{window_vol_short}'] = df['log_return'].rolling(window_vol_short, min_periods=1).std().fillna(0)
	df[f'std{window_vol_long}'] = df['log_return'].rolling(window_vol_long, min_periods=1).std().fillna(0)

	# trend features
	df['ma5'] = df['Close'].rolling(5).mean().ffill()
	df['ma10'] = df['Close'].rolling(10).mean().ffill()
	df['ma20'] = df['Close'].rolling(20).mean().ffill()

	# momentum
	df['ewma_8'] = df['Close'].ewm(span=8, adjust=False).mean()
	df['ewma_16'] = df['Close'].ewm(span=16, adjust=False).mean()
	df['momentum_8'] = ((df['Close'] - df['ewma_8']) / df['ewma_8'].replace(0, np.nan)).fillna(0)

	# spreads
	df['high_low_spread'] = ((df['High'] - df['Low']) / df['Close']).replace([np.inf, -np.inf], 0).fillna(0)
	df['open_close_spread'] = ((df['Close'] - df['Open']) / df['Open'].replace(0, np.nan)).replace([np.inf, -np.inf], 0).fillna(0)

	# volume
	if 'Volume' in df.columns:
		df['volume'] = df['Volume'].fillna(0)
		df['vol_ma5'] = df['volume'].rolling(5).mean().fillna(0)
		df['vol_ma10'] = df['volume'].rolling(10).mean().fillna(0)
	else:
		df['volume'] = 0.0
		df['vol_ma5'] = 0.0
		df['vol_ma10'] = 0.0

	# volatility spreads / interactions
	df['vol_x_std5'] = df['vol_ma5'].fillna(0) * df[f'std{window_vol_short}'].fillna(0)
	df['vol_x_mom8'] = df['vol_ma5'].fillna(0) * df['momentum_8'].fillna(0)

	# technicals: RSI, MACD, Stoch
	delta = df['Close'].diff()
	gain = delta.clip(lower=0)
	loss = -delta.clip(upper=0)
	roll_up = gain.rolling(14, min_periods=1).mean()
	roll_down = loss.rolling(14, min_periods=1).mean()
	rs = roll_up / roll_down.replace(0, np.nan)
	df['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50) / 100.0

	ema12 = df['Close'].ewm(span=12, adjust=False).mean()
	ema26 = df['Close'].ewm(span=26, adjust=False).mean()
	macd_raw = ema12 - ema26
	df['macd'] = (macd_raw / (df['ma10'].replace(0, np.nan))).fillna(0)
	df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean().fillna(0)

	low_min = df['Low'].rolling(14).min() if 'Low' in df.columns else df['Close'].rolling(14).min()
	high_max = df['High'].rolling(14).max() if 'High' in df.columns else df['Close'].rolling(14).max()
	df['stoch_k'] = ((df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)).fillna(0)
	df['stoch_d'] = df['stoch_k'].rolling(3, min_periods=1).mean().fillna(0)

	# candidate feature list
	features = [
		'log_return',
		f'std{window_vol_short}', f'std{window_vol_long}',
		'ma5','ma10','ma20',
		'momentum_8',
		'high_low_spread','open_close_spread',
		'vol_ma5','vol_ma10',
		'vol_x_std5','vol_x_mom8',
		'rsi_14','macd','macd_signal','stoch_k','stoch_d'
	]

	# keep only available numeric columns
	features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
	if not features:
		raise RuntimeError("No features available")

	# remove constant columns
	stds = df[features].std(ddof=0)
	features = [f for f in features if not np.isclose(stds.get(f, 0.0), 0.0)]

	# prepare X, returns and vol_series
	X = df[features].ffill().fillna(0).copy()
	returns = df['log_return'].fillna(0).copy()
	vol_series = df[f'std{window_vol_short}'].fillna(0).copy()
	dates = df.index.to_series()

	# standardize features
	scaler = None
	if zscore:
		scaler = StandardScaler()
		X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

	# align & return
	return X, returns, vol_series, dates, scaler

def build_features_for_inference(returns_window=None, volatility_window=None, history=None, ohlcv=None):
    """
    Build features for regime inference.
    Accepts either:
      - returns_window: list[float] and volatility_window: list[float] (preferred), OR
      - history: list[dict] of OHLCV rows (oldest->newest), OR
      - ohlcv: single OHLCV dict
    Returns dict mapped to canonical training feature names (missing -> 0.0).
    """
    import numpy as _np
    import pandas as _pd
    from pathlib import Path as _Path
    import json as _json

    # 1) compute base features from windows if provided
    out_partial = {}
    if returns_window is not None or volatility_window is not None:
        rw = _np.array(returns_window[-50:]) if returns_window else _np.array([0.0])
        vw = _np.array(volatility_window[-50:]) if volatility_window else _np.array([0.0])

        def _safe(arr, n):
            return arr[-n:] if arr.size>0 else _np.array([0.0])

        out_partial["log_return"] = float(rw[-1]) if rw.size>0 else 0.0
        out_partial["std5"] = float(_np.std(_safe(rw,5)))
        out_partial["std10"] = float(_np.std(_safe(rw,10)))
        out_partial["ret_mean_3"] = float(_np.mean(_safe(rw,3)))
        out_partial["ret_std_3"] = float(_np.std(_safe(rw,3)))
        out_partial["momentum_8"] = float(rw[-1] - _np.mean(_safe(rw,8))) if rw.size>0 else 0.0
        out_partial["vol_ma5"] = float(_np.mean(_safe(vw,5)))
        out_partial["vol_ma10"] = float(_np.mean(_safe(vw,10)))
    else:
        # build from history/ohlcv if windows not provided
        df = None
        if isinstance(history, list) and history:
            df = _pd.DataFrame(history).copy()
        elif ohlcv:
            df = _pd.DataFrame([ohlcv]).copy()

        if df is None or df.empty:
            raise RuntimeError("Provide returns_window/volatility_window or history/ohlcv for regime inference")

        # normalize column names
        col_map = {"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume","ds":"Date","date":"Date","timestamp":"Date"}
        rename = {c:col_map.get(c.lower(),c) for c in df.columns}
        df = df.rename(columns=rename)
        # ensure Close numeric
        if "Close" not in df.columns:
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]
            else:
                raise RuntimeError("history/ohlcv must include 'close' field")
        df["Close"] = _pd.to_numeric(df["Close"], errors="coerce")
        df["daily_return"] = df["Close"].pct_change().fillna(0.0)
        rw = df["daily_return"].values
        vw = (df["Volume"].fillna(0).values if "Volume" in df.columns else _np.array([0.0]))

        def _safe2(arr,n):
            return arr[-n:] if arr.size>0 else _np.array([0.0])

        out_partial["log_return"] = float(rw[-1]) if rw.size>0 else 0.0
        out_partial["std5"] = float(_np.std(_safe2(rw,5)))
        out_partial["std10"] = float(_np.std(_safe2(rw,10)))
        out_partial["ret_mean_3"] = float(_np.mean(_safe2(rw,3)))
        out_partial["ret_std_3"] = float(_np.std(_safe2(rw,3)))
        out_partial["momentum_8"] = float(rw[-1] - _np.mean(_safe2(rw,8))) if rw.size>0 else 0.0
        out_partial["vol_ma5"] = float(_np.mean(_safe2(vw,5)))
        out_partial["vol_ma10"] = float(_np.mean(_safe2(vw,10)))

    # 2) load canonical feature list saved at training time
    repo_root = _Path(__file__).resolve().parents[2]
    metrics_file = _Path(__file__).resolve().parent / "metrics" / "latest" / "training_features.json"
    model_features_file = repo_root / "models" / "latest" / "feature_names.json"
    canonical = None
    try:
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                canonical = _json.load(f).get("features") or _json.load(open(metrics_file)).get("feature_names")
        elif model_features_file.exists():
            with open(model_features_file, "r") as f:
                canonical = _json.load(f)
    except Exception:
        canonical = None

    if not isinstance(canonical, list) or not canonical:
        # fallback: pick a reasonable superset used by training
        canonical = ["log_return","std5","std10","ret_mean_3","ret_std_3","momentum_8","vol_ma5","vol_ma10"]
        # extend with plausible additional names that training may expect
        extras = ["ret_mean_5","ret_std_5","ma5","ma10","ma20","high_low_spread","open_close_spread","vol_x_std5","vol_x_mom8","rsi_14","macd","macd_signal","stoch_k","stoch_d"]
        for e in extras:
            if e not in canonical:
                canonical.append(e)

    # 3) build final dict aligned to canonical list, fill missing with 0.0
    final = {}
    for feat in canonical:
        if feat in out_partial:
            val = out_partial[feat]
        else:
            # try common aliases
            alias = feat.replace("ma","ma_") if feat.startswith("ma") and feat[2].isdigit() else feat
            val = out_partial.get(alias, 0.0)
        try:
            final[feat] = float(val) if not (_pd.isna(val) if 'pd' in globals() else False) else 0.0
        except Exception:
            try:
                final[feat] = float(val)
            except Exception:
                final[feat] = 0.0

    return final
