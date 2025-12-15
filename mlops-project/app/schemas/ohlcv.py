from pydantic import BaseModel
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

class ProphetRequest(BaseModel):
    periods: int
