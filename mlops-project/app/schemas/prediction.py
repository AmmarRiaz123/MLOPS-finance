from pydantic import BaseModel
from typing import Any, List, Dict, Optional

class FeaturesRequest(BaseModel):
    features: Dict[str, float]

class BatchFeaturesRequest(BaseModel):
    records: List[Dict[str, float]]

class PredictionResponse(BaseModel):
    model: str
    prediction: Any
    raw_output: Any = None

class ReturnResponse(BaseModel):
    model: str
    predicted_return: float

class DirectionResponse(BaseModel):
    model: str
    direction: str
    probability: Optional[float] = None

class VolatilityResponse(BaseModel):
    model: str
    predicted_volatility: float

class ProphetPoint(BaseModel):
    date: str
    yhat: float
    yhat_lower: Optional[float] = None
    yhat_upper: Optional[float] = None

class ProphetResponse(BaseModel):
    model: str
    forecast: List[Dict[str, Any]]
