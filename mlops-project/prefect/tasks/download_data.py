from pathlib import Path
import os
from dotenv import load_dotenv
from prefect import task
import argparse

# attempt to load .env from mlops-project/.env, fallback to default load_dotenv()
REPO_ROOT = Path(__file__).resolve().parents[2]  # mlops-project
DOTENV_PATH = REPO_ROOT / ".env"
if DOTENV_PATH.exists():
    load_dotenv(dotenv_path=str(DOTENV_PATH))
    _DOTENV_LOADED = str(DOTENV_PATH.resolve())
else:
    load_dotenv()
    _DOTENV_LOADED = None

BASE_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DEFAULT_DATA_DIR)))

# yfinance + pandas
try:
    import yfinance as yf
    import pandas as pd
except Exception:
    yf = None
    pd = None

@task(retries=1, retry_delay_seconds=60)
def download_symbol(symbol: str,
                    backend: str = "yfinance",            # kept for clarity, only yfinance supported
                    interval: str | None = None,         # yfinance interval: "1d","1m","1h",...
                    start: str | None = None,            # ISO date YYYY-MM-DD
                    end: str | None = None,              # ISO date YYYY-MM-DD
                    period: str | None = None,           # yfinance period e.g. "1y","5y", "max"
                    outdir: str | None = None) -> str:
    """
    Download a symbol using yfinance and save to CSV. Returns path to saved CSV.
    """
    if backend.lower() != "yfinance":
        raise RuntimeError("Only 'yfinance' backend is supported in this repo (Alpha Vantage removed).")

    if yf is None or pd is None:
        raise RuntimeError("yfinance and pandas are required. Install with `pip install yfinance pandas`.")

    outdir_path = Path(outdir) if outdir else DEFAULT_DATA_DIR
    outdir_path.mkdir(parents=True, exist_ok=True)

    yf_interval = interval or "1d"
    try:
        if start or end:
            df = yf.download(symbol, start=start, end=end, interval=yf_interval, progress=False)
        else:
            df = yf.download(symbol, period=period or "max", interval=yf_interval, progress=False)
    except Exception as e:
        raise RuntimeError(f"yfinance download failed for {symbol}: {e}")

    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {symbol} (check ticker/interval/date range).")

    df = df.reset_index()
    csv_name = f"{symbol}_yfinance_{yf_interval}.csv"
    out_path = outdir_path / csv_name
    df.to_csv(out_path, index=False)
    return str(out_path)

# CLI / demo entry
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download data via yfinance (local demo).")
    parser.add_argument("--symbol", default="IBM", help="Ticker symbol")
    parser.add_argument("--interval", default=None, help="yfinance interval (1d,1m,1h,... )")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD")
    parser.add_argument("--period", default=None, help="yfinance period (e.g. 1y, 5y, max)")
    parser.add_argument("--outdir", default=None, help="Output directory (overrides DATA_DIR)")
    args = parser.parse_args()

    result = download_symbol(symbol=args.symbol,
                             interval=args.interval,
                             start=args.start,
                             end=args.end,
                             period=args.period,
                             outdir=args.outdir)
    print(f"Done: {result}")
