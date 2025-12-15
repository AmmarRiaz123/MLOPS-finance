from pydantic import BaseModel
from typing import List, Optional

class RegimeRequest(BaseModel):
    returns_window: List[float]
    volatility_window: List[float]

class RegimeResponse(BaseModel):
    model: str
    regime_id: int
    regime_label: str
    probabilities: Optional[List[float]] = None
