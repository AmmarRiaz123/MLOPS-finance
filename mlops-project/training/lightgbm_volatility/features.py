from pathlib import Path
import pandas as pd
import numpy as np
from typing import Tuple, List

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRS = [REPO_ROOT / "data", REPO_ROOT / "data" / "raw"]

def _find_csv_files() -> List[Path]:
	paths = []
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
			df = pd.read_csv(p, engine='python', on_bad_lines='skip')
		except TypeError:
			df = pd.read_csv(p, engine='python', error_bad_lines=False)
		df.columns = [c.strip() for c in df.columns]
		dfs.append(df)
	return pd.concat(dfs, ignore_index=True)

def prepare_features(scale: bool = True, target_horizon: int = 3) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
	"""
	Returns X, y, dates.
	- y: forward-looking volatility = std(log_return.shift(-1).rolling(window=target_horizon))
	- features: compact engineered set for small data
	"""
	files = _find_csv_files()
	df = _load_all(files)

	# date parse
	date_cols = [c for c in ['Date','date','Datetime','datetime','ds','timestamp'] if c in df.columns]
	if not date_cols:
		raise ValueError("No date column found")
	dc = date_cols[0]
	df[dc] = pd.to_datetime(df[dc], errors='coerce')
	df = df.dropna(subset=[dc])
	df = df.sort_values(dc).drop_duplicates(subset=[dc], keep='first').set_index(dc)

	# numeric coercion
	for c in ['Open','High','Low','Close','Adj Close','Volume']:
		if c in df.columns:
			df[c] = pd.to_numeric(df[c], errors='coerce')
	if 'Close' not in df.columns and 'Adj Close' in df.columns:
		df['Close'] = df['Adj Close']

	# drop rows without Close
	df = df[pd.to_numeric(df['Close'], errors='coerce').notna()]

	# log returns (past)
	df['log_price'] = np.log(df['Close'])
	df['log_return'] = df['log_price'].diff()

	# target: forward-looking rolling std of log returns over next target_horizon days
	# compute future returns window: shift -1 then rolling
	df['future_log_return'] = df['log_return'].shift(-1)
	df[f'vol_{target_horizon}d'] = df['future_log_return'].rolling(window=target_horizon, min_periods=1).std().shift(-(target_horizon-1))

	# lagged returns
	df['ret_lag1'] = df['log_return'].shift(1)
	df['ret_lag2'] = df['log_return'].shift(2)
	df['ret_lag3'] = df['log_return'].shift(3)

	# rolling stats on returns (past-only)
	df['ret_mean_3'] = df['log_return'].rolling(3).mean()
	df['ret_std_3'] = df['log_return'].rolling(3).std()

	# price rolling
	df['ma5'] = df['Close'].rolling(5).mean()
	df['ma10'] = df['Close'].rolling(10).mean()
	df['std5'] = df['log_return'].rolling(5).std()
	df['std10'] = df['log_return'].rolling(10).std()

	# spreads
	if {'High','Low'}.issubset(df.columns):
		df['high_low_spread'] = (df['High'] - df['Low']) / df['Close']
	else:
		df['high_low_spread'] = 0.0
	if {'Open','Close'}.issubset(df.columns):
		df['open_close_spread'] = (df['Close'] - df['Open']) / df['Open'].replace(0, np.nan)
	else:
		df['open_close_spread'] = 0.0

	# volume
	if 'Volume' in df.columns:
		df['volume'] = df['Volume'].fillna(0)
		df['vol_ma5'] = df['volume'].rolling(5).mean().fillna(0)
		df['vol_ma10'] = df['volume'].rolling(10).mean().fillna(0)
	else:
		df['volume'] = 0.0
		df['vol_ma5'] = 0.0
		df['vol_ma10'] = 0.0

	# momentum indicators
	df['ewma_8'] = df['Close'].ewm(span=8, adjust=False).mean()
	df['ewma_16'] = df['Close'].ewm(span=16, adjust=False).mean()
	df['momentum_8'] = (df['Close'] - df['ewma_8']) / df['ewma_8'].replace(0, np.nan)
	df['momentum_8'] = df['momentum_8'].fillna(0)

	# RSI 14 scaled 0-1
	delta = df['Close'].diff()
	gain = delta.clip(lower=0)
	loss = -delta.clip(upper=0)
	roll_up = gain.rolling(14).mean()
	roll_down = loss.rolling(14).mean()
	rs = roll_up / roll_down.replace(0, np.nan)
	df['rsi_14'] = (100 - (100 / (1 + rs))).fillna(50) / 100.0

	# MACD (normalized)
	ema12 = df['Close'].ewm(span=12, adjust=False).mean()
	ema26 = df['Close'].ewm(span=26, adjust=False).mean()
	macd_raw = ema12 - ema26
	df['macd'] = (macd_raw / df['ma10'].replace(0, np.nan)).fillna(0)
	df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean().fillna(0)

	# Stochastic %K/%D (0-1)
	low_min = df['Low'].rolling(14).min() if 'Low' in df.columns else df['Close'].rolling(14).min()
	high_max = df['High'].rolling(14).max() if 'High' in df.columns else df['Close'].rolling(14).max()
	df['stoch_k'] = ((df['Close'] - low_min) / (high_max - low_min).replace(0, np.nan)).fillna(0)
	df['stoch_d'] = df['stoch_k'].rolling(3).mean().fillna(0)

	# interactions
	df['vol_x_std5'] = df['vol_ma5'].fillna(0) * df['std5'].fillna(0)
	df['vol_x_mom8'] = df['vol_ma5'].fillna(0) * df['momentum_8'].fillna(0)

	# candidate features (compact)
	candidates = [
		'ret_lag1','ret_lag2','ret_lag3',
		'ret_mean_3','ret_std_3','ma5','ma10','std5','std10',
		'high_low_spread','open_close_spread',
		'vol_ma5','vol_ma10',
		'ewma_8','ewma_16','momentum_8',
		'rsi_14','macd','macd_signal','stoch_k','stoch_d',
		'vol_x_std5','vol_x_mom8'
	]

	# ensure numeric & exist
	features = [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
	if not features:
		raise RuntimeError("No features available")

	# forward-fill (past-only), then fill zeros
	df[features] = df[features].ffill().fillna(0)

	# drop constant or near-constant
	stds = df[features].std(ddof=0)
	kept = [c for c in features if not np.isclose(stds.get(c,0.0), 0.0)]
	if not kept:
		kept = features
	features = kept

	# build X/y/dates; target is forward-looking vol
	target_col = f'vol_{target_horizon}d'
	if target_col not in df.columns:
		df[target_col] = df['future_log_return'].rolling(window=target_horizon, min_periods=1).std().shift(-(target_horizon-1))
	df = df.dropna(subset=[target_col])
	y = df[target_col].copy()
	dates = df.index.to_series()
	X = df[features].loc[y.index].copy()

	# optional scaling
	scaler = None
	if scale:
		from sklearn.preprocessing import StandardScaler
		scaler = StandardScaler()
		X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)

	return (X, y, dates, scaler) if scale else (X, y, dates)
