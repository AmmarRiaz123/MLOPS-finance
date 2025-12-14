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

if __name__ == "__main__":
    df, regs = prepare_prophet_df()
    print(f"Prepared Prophet df with {len(df)} rows and regressors: {regs}")
