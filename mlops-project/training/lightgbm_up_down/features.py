import pandas as pd
import numpy as np
from pathlib import Path

def load_and_clean_data(csv_path):
    """Load CSV and clean for feature engineering."""
    df = pd.read_csv(csv_path)
    
    # normalize column names
    df.columns = df.columns.str.strip()
    
    # Handle different possible date column names from yfinance / CSV exports
    date_cols = ['Date', 'Datetime', 'timestamp']
    date_col = next((c for c in date_cols if c in df.columns), None)
    if date_col is None:
        raise ValueError(f"No date column found. Available columns: {list(df.columns)}")
    
    # Parse date and set as index
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.set_index(date_col)
    df = df.sort_index()  # Ensure chronological order
    
    # Drop rows with invalid dates
    df = df[~df.index.isna()]
    
    # Coerce common OHLCV columns to numeric to avoid string values
    numeric_candidates = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Adj_Close', 'Volume']
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # if 'Adj Close' present, also create a consistent 'Adj_Close' name
            if col == 'Adj Close' and 'Adj_Close' not in df.columns:
                df['Adj_Close'] = df[col]
    
    # Ensure we have a valid Close column (try Adj_Close fallback)
    if 'Close' not in df.columns and 'Adj_Close' in df.columns:
        df['Close'] = df['Adj_Close']
    
    # Drop rows where Close is missing (required for returns/features)
    df = df.dropna(subset=['Close'])
    
    # Drop any duplicate indices
    df = df[~df.index.duplicated(keep='first')]
    
    return df

def generate_features(df):
    """Generate ML features from OHLCV data including lagged returns, rolling stats, RSI, MACD, Bollinger Bands and volume features."""
    features_df = df.copy()

    # Basic returns and lagged returns (1,3,5)
    features_df['daily_return'] = features_df['Close'].pct_change()
    features_df['return_lag1'] = features_df['daily_return'].shift(1)
    features_df['return_lag3'] = features_df['daily_return'].shift(3)
    features_df['return_lag5'] = features_df['daily_return'].shift(5)

    # Rolling statistics on returns (windows 5,10,20)
    roll_windows = [5, 10, 20]
    for w in roll_windows:
        features_df[f'return_mean_{w}'] = features_df['daily_return'].rolling(window=w).mean()
        features_df[f'return_std_{w}'] = features_df['daily_return'].rolling(window=w).std()
        features_df[f'return_skew_{w}'] = features_df['daily_return'].rolling(window=w).skew()
        features_df[f'return_kurt_{w}'] = features_df['daily_return'].rolling(window=w).kurt()

    # Moving averages (trend)
    features_df['ma_5'] = features_df['Close'].rolling(window=5).mean()
    features_df['ma_10'] = features_df['Close'].rolling(window=10).mean()
    features_df['ma_20'] = features_df['Close'].rolling(window=20).mean()

    # Price ratios
    features_df['close_to_ma5'] = features_df['Close'] / features_df['ma_5']
    features_df['close_to_ma10'] = features_df['Close'] / features_df['ma_10']
    features_df['ma5_to_ma10'] = features_df['ma_5'] / features_df['ma_10']

    # Volatility and spreads
    features_df['high_low_spread'] = (features_df['High'] - features_df['Low']) / features_df['Close']
    features_df['open_close_spread'] = (features_df['Close'] - features_df['Open']) / features_df['Open']
    features_df['volatility_5d'] = features_df['daily_return'].rolling(window=5).std()
    features_df['volatility_10d'] = features_df['daily_return'].rolling(window=10).std()

    # RSI (14) - using Wilder's smoothing (EMA-like)
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        # Wilder's EMA
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / (avg_loss.replace(0, np.nan))
        rsi = 100 - (100 / (1 + rs))
        return rsi

    features_df['rsi_14'] = compute_rsi(features_df['Close'], period=14)

    # MACD (12,26,9)
    ema12 = features_df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = features_df['Close'].ewm(span=26, adjust=False).mean()
    features_df['macd'] = ema12 - ema26
    features_df['macd_signal'] = features_df['macd'].ewm(span=9, adjust=False).mean()
    features_df['macd_hist'] = features_df['macd'] - features_df['macd_signal']

    # Bollinger Bands (20-day)
    bb_mid = features_df['Close'].rolling(window=20).mean()
    bb_std = features_df['Close'].rolling(window=20).std()
    features_df['bb_mid_20'] = bb_mid
    features_df['bb_upper_20'] = bb_mid + 2 * bb_std
    features_df['bb_lower_20'] = bb_mid - 2 * bb_std
    features_df['bb_width_20'] = (features_df['bb_upper_20'] - features_df['bb_lower_20']) / features_df['bb_mid_20']

    # Volume-based features
    if 'Volume' in features_df.columns:
        features_df['volume_ma_5'] = features_df['Volume'].rolling(window=5).mean()
        features_df['volume_ma_20'] = features_df['Volume'].rolling(window=20).mean()
        # spike relative to recent average
        features_df['volume_spike_5'] = features_df['Volume'] / features_df['volume_ma_5']
        features_df['volume_spike_20'] = features_df['Volume'] / features_df['volume_ma_20']
        # zscore over 20
        features_df['volume_zscore_20'] = (features_df['Volume'] - features_df['volume_ma_20']) / features_df['Volume'].rolling(window=20).std()

    return features_df

def create_target(df, horizon: int = 1, smooth_window: int | None = None):
    """Create up_down target: 1 if future close (horizon days) > today close.
    If smooth_window is provided, a past-only rolling mean (no future leakage) is computed
    and used as the base series to compare future value against.
    """
    series = df['Close']
    if smooth_window and smooth_window > 1:
        # rolling uses past values only so no leakage
        series_base = series.rolling(window=smooth_window, min_periods=1).mean()
    else:
        series_base = series

    # future close at horizon
    future_close = series.shift(-horizon)
    # binary target for horizon
    df[f'up_down_h{horizon}'] = (future_close > series).astype(int)
    # alternative target using smoothed base if requested
    if smooth_window and smooth_window > 1:
        future_base = series_base.shift(-horizon)
        df[f'up_down_h{horizon}_smoothed'] = (future_base > series_base).astype(int)
    return df

def prepare_ml_data(csv_path, horizon: int = 1, smooth_window: int | None = None, scale: bool = False):
    """Complete pipeline: load -> features -> target.
    Parameters:
    - horizon: days ahead to predict (1 = next day)
    - smooth_window: optional past-only smoothing window used to create an additional smoothed target
    - scale: if True, returns (X_scaled, y, scaler)
    Returns:
    - X, y  (if scale=False)
    - (X_scaled, y, scaler) (if scale=True)
    """
    # Load and clean data
    df = load_and_clean_data(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")

    # Generate features
    df_features = generate_features(df)

    # Create target(s)
    df_final = create_target(df_features, horizon=horizon, smooth_window=smooth_window)

    # Dynamically collect feature columns (exclude raw OHLCV and targets)
    exclude_cols = {'Open', 'High', 'Low', 'Close', 'Adj Close', 'Adj_Close', 'Volume'}
    target_cols = [c for c in df_final.columns if c.startswith('up_down_h')]
    feature_cols = [c for c in df_final.columns if c not in exclude_cols.union(set(target_cols)) and c not in {'next_close'}]

    X = df_final[feature_cols].copy()
    # choose primary target: prefer smoothed horizon if requested else plain horizon
    target_name = f'up_down_h{horizon}_smoothed' if (smooth_window and smooth_window > 1) else f'up_down_h{horizon}'
    y = df_final[target_name].copy()

    # Drop rows with NaN created by lookbacks
    mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[mask]
    y_clean = y[mask]

    print(f"After feature engineering: {len(X_clean)} samples, {X_clean.shape[1]} features")
    print(f"Target distribution ({target_name}): {y_clean.value_counts().to_dict()}")

    if scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X_clean), columns=X_clean.columns, index=X_clean.index)
        return X_scaled, y_clean, scaler

    return X_clean, y_clean

if __name__ == "__main__":
    # Example usage
    data_dir = Path(__file__).resolve().parents[2] / "data"
    csv_files = list(data_dir.glob("*.csv"))
    
    if csv_files:
        csv_path = csv_files[0]  # Use first CSV found
        print(f"Processing {csv_path}")
        X, y = prepare_ml_data(csv_path)
        print("Feature engineering completed successfully!")
    else:
        print("No CSV files found in data directory")
