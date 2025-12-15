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
