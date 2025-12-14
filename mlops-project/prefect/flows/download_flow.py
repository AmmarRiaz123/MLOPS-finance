from prefect import flow
from typing import Iterable

# try package import; fallback to dynamic import for script execution
try:
    from ...tasks.download_data import download_symbol
except Exception:
    import importlib.util
    import sys
    from pathlib import Path
    tasks_path = Path(__file__).resolve().parents[1] / "tasks" / "download_data.py"
    spec = importlib.util.spec_from_file_location("download_data", str(tasks_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules["download_data"] = module
    spec.loader.exec_module(module)
    download_symbol = getattr(module, "download_symbol")

@flow
def download_symbols_flow(symbols: Iterable[str],
                          backend: str = "yfinance",
                          interval: str | None = None,
                          start: str | None = None,
                          end: str | None = None,
                          period: str | None = None,
                          outdir: str | None = None):
    """
    Prefect flow to download time series for multiple symbols via yfinance.
    """
    futures = []
    for s in symbols:
        fut = download_symbol.submit(symbol=s,
                                     backend=backend,
                                     interval=interval,
                                     start=start,
                                     end=end,
                                     period=period,
                                     outdir=outdir)
        futures.append(fut)
    results = [f.result() for f in futures]
    return results

if __name__ == "__main__":
    import os
    # Example local run
    download_symbols_flow(["AAPL", "MSFT"], period="5y")
