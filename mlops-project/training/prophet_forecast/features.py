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
        # normalize column names
        df.columns = df.columns.str.strip()
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    return full

def prepare_prophet_df(data_dir: Path = DATA_DIR) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load CSVs, build Prophet-compatible dataframe (ds,y) and regressors.
    Returns (df_prophet, regressors_list).
    """
    raw = load_all_csvs(data_dir)
    date_col = _find_date_col(raw)
    raw[date_col] = pd.to_datetime(raw[date_col], errors='coerce')
    raw = raw.dropna(subset=[date_col])
    raw = raw.sort_values(date_col).drop_duplicates(subset=[date_col], keep='first')
    raw = raw.set_index(date_col)

    # standardize columns
    cols = {c: c.strip() for c in raw.columns}
    raw.rename(columns=cols, inplace=True)

    # ensure numeric OHLCV present where possible
    for c in ['Open','High','Low','Close','Volume','Adj Close']:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors='coerce')

    # prefer Close, fallback to Adj Close
    if 'Close' not in raw.columns and 'Adj Close' in raw.columns:
        raw['Close'] = raw['Adj Close']

    # compute regressors (past-only stats)
    df = raw.copy()
    df['high_low_spread'] = (df['High'] - df['Low']) / df['Close'] if {'High','Low','Close'}.issubset(df.columns) else np.nan
    df['open_close_spread'] = (df['Close'] - df['Open']) / df['Open'] if {'Open','Close'}.issubset(df.columns) else np.nan
    if 'Volume' in df.columns:
        df['volume'] = df['Volume']
    else:
        df['volume'] = np.nan

    # forward-fill/backfill minimal gaps but avoid leaking future: use ffill then fillna with 0
    df[['volume','high_low_spread','open_close_spread']] = df[['volume','high_low_spread','open_close_spread']].ffill().fillna(0)

    # build Prophet frame
    prophet_df = pd.DataFrame({
        'ds': df.index,
        'y': df['Close'].values
    }, index=df.index)

    # attach regressors columns (Prophet expects them present during fit/predict)
    regressors = []
    for r in ['volume','high_low_spread','open_close_spread']:
        if r in df.columns:
            prophet_df[r] = df[r].values
            regressors.append(r)

    # drop rows where y is NA
    prophet_df = prophet_df.dropna(subset=['y'])

    return prophet_df.reset_index(drop=True), regressors

if __name__ == "__main__":
    df, regs = prepare_prophet_df()
    print(f"Prepared Prophet df with {len(df)} rows and regressors: {regs}")
