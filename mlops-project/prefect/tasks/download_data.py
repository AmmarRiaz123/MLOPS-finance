from pathlib import Path
import os
from dotenv import load_dotenv
import time
import csv
import requests
from prefect import task
import argparse

# attempt to load .env from mlops-project/.env, fallback to default load_dotenv()
REPO_ROOT = Path(__file__).resolve().parents[2]  # mlops-project (was parents[3])
DOTENV_PATH = REPO_ROOT / ".env"
if DOTENV_PATH.exists():
    load_dotenv(dotenv_path=str(DOTENV_PATH))
    _DOTENV_LOADED = str(DOTENV_PATH.resolve())
else:
    load_dotenv()  # fallback (searches CWD / parent dirs)
    _DOTENV_LOADED = None

# accept several common env var names and map the first found to ALPHAVANTAGE_API_KEY
_alt_names = ["ALPHAVANTAGE_API_KEY", "ALPHAVANTAGE_KEY", "ALPHAVANTAGE", "API_KEY", "APIKEY", "Api_Key", "ApiKey", "KEY"]
_ALPHA_ENV_SOURCE = None
for _n in _alt_names:
    _v = os.environ.get(_n)
    if _v:
        os.environ["ALPHAVANTAGE_API_KEY"] = _v
        _ALPHA_ENV_SOURCE = _n
        break

ALPHA_URL = "https://www.alphavantage.co/query"
BASE_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_DATA_DIR = Path(os.environ.get("DATA_DIR", str(BASE_DEFAULT_DATA_DIR)))

def _download_alpha_vantage_impl(symbol: str,
                                 function: str = "TIME_SERIES_DAILY_ADJUSTED",
                                 interval: str | None = None,
                                 outputsize: str = "compact",
                                 apikey: str | None = None,
                                 outdir: str | None = None,
                                 verbose: bool = True) -> str | None:
    """
    Implementation function that performs the HTTP request and saves CSV.
    Returns saved path (string) or None on non-fatal issues.
    """
    # determine source before overriding apikey
    source_by_arg = apikey is not None
    source_by_env = os.environ.get("ALPHAVANTAGE_API_KEY") is not None
    apikey = apikey or os.environ.get("ALPHAVANTAGE_API_KEY")
    if verbose:
        source = "argument" if source_by_arg else ("env" if source_by_env else "none")
        print(f"[download] using API key from {source} (loaded .env: {_DOTENV_LOADED or 'not found'})")
    if not apikey:
        raise RuntimeError("Alpha Vantage API key required (env ALPHAVANTAGE_API_KEY or apikey argument).")
    params = {"function": function, "symbol": symbol, "apikey": apikey, "outputsize": outputsize}
    if interval:
        params["interval"] = interval

    for attempt in range(1, 6):
        if verbose:
            print(f"[download] attempt {attempt} for {symbol} (function={function}{', interval='+interval if interval else ''})")
        resp = requests.get(ALPHA_URL, params=params, timeout=30)
        if resp.status_code != 200:
            if verbose:
                print(f"[download] HTTP {resp.status_code} — retrying in 5s")
            time.sleep(5)
            continue
        data = resp.json()
        # surface "Information" messages from Alpha Vantage (explains invalid key / usage limits)
        if "Information" in data:
            if verbose:
                print(f"[download] Alpha Vantage information: {data.get('Information')}")
            return None
        if "Note" in data:
            # rate limit, wait then retry
            if verbose:
                print(f"[download] Rate limit notice from Alpha Vantage: {data['Note'][:200]}... waiting 60s")
            time.sleep(60)
            continue
        if "Error Message" in data:
            raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")
        # found data
        ts_key = next((k for k in data.keys() if "Time Series" in k), None)
        if not ts_key:
            # Some responses may only contain metadata or be empty; surface that to user
            if verbose:
                print(f"[download] Unexpected response keys: {list(data.keys())}")
            return None
        series = data[ts_key]
        rows = []
        for ts, values in series.items():
            row = {"timestamp": ts}
            for k, v in values.items():
                col = k.split(". ", 1)[-1]
                row[col] = v
            rows.append(row)
        if not rows:
            if verbose:
                print("[download] No time series rows found in response.")
            return None
        rows.sort(key=lambda r: r["timestamp"])
        cols = ["timestamp"] + sorted(c for c in rows[0].keys() if c != "timestamp")
        outdir_path = Path(outdir) if outdir else DEFAULT_DATA_DIR
        outdir_path.mkdir(parents=True, exist_ok=True)
        fname_parts = [symbol, function]
        if interval:
            fname_parts.append(interval)
        out_path = outdir_path / ("_".join(fname_parts) + ".csv")
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        if verbose:
            print(f"[download] Saved {len(rows)} rows to {out_path.resolve()}")
        return str(out_path)
    raise RuntimeError("Failed to fetch data after retries.")

@task(retries=3, retry_delay_seconds=60)
def download_alpha_vantage(symbol: str,
                           function: str = "TIME_SERIES_DAILY_ADJUSTED",
                           interval: str | None = None,
                           outputsize: str = "compact",
                           apikey: str | None = None,
                           outdir: str | None = None) -> str:
    # delegate to implementation (Prefect task runs this)
    return _download_alpha_vantage_impl(symbol=symbol,
                                        function=function,
                                        interval=interval,
                                        outputsize=outputsize,
                                        apikey=apikey,
                                        outdir=outdir,
                                        verbose=True)

# CLI / demo entry so running the file writes data to DATA_DIR
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Alpha Vantage data (local demo).")
    parser.add_argument("--symbol", default="IBM", help="Ticker symbol")
    parser.add_argument("--function", default="TIME_SERIES_DAILY_ADJUSTED", help="AlphaVantage function")
    parser.add_argument("--interval", default=None, help="Interval for intraday")
    parser.add_argument("--outputsize", default="compact", choices=["compact", "full"])
    parser.add_argument("--apikey", default=None, help="Alpha Vantage API key (overrides env)")
    parser.add_argument("--outdir", default=None, help="Output directory (overrides DATA_DIR)")
    args = parser.parse_args()
    result = _download_alpha_vantage_impl(symbol=args.symbol,
                                          function=args.function,
                                          interval=args.interval,
                                          outputsize=args.outputsize,
                                          apikey=args.apikey,
                                          outdir=args.outdir,
                                          verbose=True)
    if result:
        print(f"Done: {result}")
    else:
        print("No file produced. Check response or rate limits.")
