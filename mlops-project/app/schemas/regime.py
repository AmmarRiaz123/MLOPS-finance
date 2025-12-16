from pydantic import BaseModel
from typing import List, Optional

class RegimeRequest(BaseModel):
    returns_window: List[float]
    volatility_window: List[float]

class RegimeResponse(BaseModel):
    model: str
    regime: str                      # 'bull' | 'neutral' | 'bear'
    score: Optional[float] = None    # confidence score in [0.0, 1.0]
