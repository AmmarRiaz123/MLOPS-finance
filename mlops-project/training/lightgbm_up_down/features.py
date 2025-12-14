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
    """Generate ML features from OHLCV data."""
    features_df = df.copy()
    
    # Daily returns
    features_df['daily_return'] = features_df['Close'].pct_change()
    
    # Moving averages for trend
    features_df['ma_5'] = features_df['Close'].rolling(window=5).mean()
    features_df['ma_10'] = features_df['Close'].rolling(window=10).mean()
    features_df['ma_20'] = features_df['Close'].rolling(window=20).mean()
    
    # Price ratios (trend indicators)
    features_df['close_to_ma5'] = features_df['Close'] / features_df['ma_5']
    features_df['close_to_ma10'] = features_df['Close'] / features_df['ma_10']
    features_df['ma5_to_ma10'] = features_df['ma_5'] / features_df['ma_10']
    
    # Volatility indicators
    features_df['high_low_spread'] = (features_df['High'] - features_df['Low']) / features_df['Close']
    features_df['open_close_spread'] = (features_df['Close'] - features_df['Open']) / features_df['Open']
    
    # Rolling volatility
    features_df['volatility_5d'] = features_df['daily_return'].rolling(window=5).std()
    features_df['volatility_10d'] = features_df['daily_return'].rolling(window=10).std()
    
    # Volume indicators (if available)
    if 'Volume' in features_df.columns:
        features_df['volume_ma_5'] = features_df['Volume'].rolling(window=5).mean()
        features_df['volume_ratio'] = features_df['Volume'] / features_df['volume_ma_5']
    
    # Lagged returns (momentum features)
    features_df['return_lag1'] = features_df['daily_return'].shift(1)
    features_df['return_lag2'] = features_df['daily_return'].shift(2)
    features_df['return_lag3'] = features_df['daily_return'].shift(3)
    
    return features_df

def create_target(df):
    """Create up_down target: 1 if next day close > today close, else 0."""
    df['next_close'] = df['Close'].shift(-1)
    df['up_down'] = (df['next_close'] > df['Close']).astype(int)
    return df

def prepare_ml_data(csv_path):
    """Complete pipeline: load data -> engineer features -> create target."""
    # Load and clean data
    df = load_and_clean_data(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    
    # Generate features
    df_features = generate_features(df)
    
    # Create target
    df_final = create_target(df_features)
    
    # Select feature columns (exclude OHLCV and intermediate calculations)
    feature_cols = [
        'daily_return', 'ma_5', 'ma_10', 'ma_20',
        'close_to_ma5', 'close_to_ma10', 'ma5_to_ma10',
        'high_low_spread', 'open_close_spread',
        'volatility_5d', 'volatility_10d',
        'return_lag1', 'return_lag2', 'return_lag3'
    ]
    
    # Add volume features if available
    if 'Volume' in df_final.columns:
        feature_cols.extend(['volume_ma_5', 'volume_ratio'])
    
    # Prepare final dataset
    X = df_final[feature_cols]
    y = df_final['up_down']
    
    # Drop rows with NaN (from rolling windows and lags)
    mask = ~(X.isna().any(axis=1) | y.isna())
    X_clean = X[mask]
    y_clean = y[mask]
    
    print(f"After feature engineering: {len(X_clean)} samples, {len(feature_cols)} features")
    print(f"Target distribution: {y_clean.value_counts().to_dict()}")
    
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
