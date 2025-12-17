from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class OHLCVEntry(BaseModel):
    ds: datetime
    open: Optional[float]
    high: Optional[float]
    low: Optional[float]
    close: float
    volume: Optional[float]

class OHLCVSeries(BaseModel):
    series: List[OHLCVEntry]

class OHLCVInput(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float

class OHLCVRow(BaseModel):
    """Single OHLCV row accepted in history arrays (oldest -> newest)."""
    ds: Optional[str] = Field(None, description="ISO-8601 datetime for the row (optional)", alias="date")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price") 
    close: float = Field(..., description="Closing price")
    volume: Optional[float] = Field(None, description="Trading volume")
    
    class Config:
        allow_population_by_field_name = True

class ProphetRequest(BaseModel):
    """
    Request body for /forecast/price.
    - periods: positive integer number of future days to forecast.
    - history: optional list of recent OHLCV entries. When omitted, regressors use default values (0.0).
    """
    periods: int = Field(..., gt=0, description="Number of future days to forecast (integer > 0)")
    history: Optional[List[OHLCVRow]] = Field(
        None,
        description="Optional recent history (oldest->newest). If omitted, model regressors will use safe default values."
    )

    # (line ~55) split long line to satisfy flake8 E501
    # Example pattern:
    # raise ValueError(
    #     "your long message part 1 "
    #     "part 2"
    # )
    raise RuntimeError(
        f"Feature builder requires column 'Close' (case-sensitive). Available columns: {list(df.columns)}. "
        "Provide OHLCV fields as keys 'open','high','low','close','volume' "
        "or column names matching training CSV."
    )
