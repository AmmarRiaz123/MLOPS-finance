from prefect import flow
from prefect.tasks import task_input_hash
from typing import Iterable
from ...tasks.download_data import download_alpha_vantage  # relative import

@flow
def download_symbols_flow(symbols: Iterable[str],
                          function: str = "TIME_SERIES_DAILY_ADJUSTED",
                          interval: str | None = None,
                          outputsize: str = "compact",
                          apikey: str | None = None,
                          outdir: str | None = None):
    """
    Prefect flow to download time series for multiple symbols.
    Example: download_symbols_flow(["AAPL","MSFT"], apikey="KEY")
    """
    futures = []
    for s in symbols:
        fut = download_alpha_vantage.submit(symbol=s,
                                           function=function,
                                           interval=interval,
                                           outputsize=outputsize,
                                           apikey=apikey,
                                           outdir=outdir)
        futures.append(fut)
    results = [f.result() for f in futures]
    return results

# Simple local run entrypoint (optional)
if __name__ == "__main__":
    # Example: set symbols and run locally
    download_symbols_flow(["AAPL"], apikey=os.environ.get("ALPHAVANTAGE_API_KEY"))
