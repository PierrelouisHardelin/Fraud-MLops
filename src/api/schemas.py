from typing import Any, Dict, Optional

from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: Dict[str, Any]
    request_id: Optional[str] = None


class PredictResponse(BaseModel):
    request_id: Optional[str] = None
    fraud_proba: float
    fraud_pred: int
    threshold: float
