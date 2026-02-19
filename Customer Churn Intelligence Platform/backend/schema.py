from pydantic import BaseModel,Field
from typing import List,Dict,Any,Optional

class AdviceRequest(BaseModel):
    customer_id:str=Field(...,description="Unique identifier for the customer")
    
class CustomerPredictionRequest(BaseModel):
    customer_id:str=Field(...,description="Unique identifier for the customer")

class AdviceResponse(BaseModel):
    request_id:       str
    customer_id:      str
    timestamp:        str
    churn_probability: float
    risk_tier:        str
    advice:           str
    account_info:     Dict[str, Any]
    engagement_signals: Dict[str, Any]
    support_signals:  Dict[str, Any]
    billing_signals:  Dict[str, Any]
    top_churn_drivers: List[Dict[str, Any]]


class ContextResponse(BaseModel):
    request_id:        str
    customer_id:       str
    timestamp:         str
    churn_probability: float
    risk_tier:         str
    account_info:      Dict[str, Any]
    engagement_signals: Dict[str, Any]
    support_signals:   Dict[str, Any]
    billing_signals:   Dict[str, Any]
    top_churn_drivers: List[Dict[str, Any]]


class BulkAdviceRequest(BaseModel):
    customer_ids: List[str] = Field(..., description="List of customer UUIDs", max_items=20)


class BulkAdviceResponse(BaseModel):
    request_id:  str
    timestamp:   str
    total:       int
    results:     List[Dict[str, Any]]
    errors:      List[Dict[str, Any]]

class HealthResponse(BaseModel):
    status:    str
    timestamp: str
    version:   str
    models_loaded: bool