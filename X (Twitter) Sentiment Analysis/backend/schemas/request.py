from pydantic import BaseModel,Field,validator
from typing import Optional,List,Dict


class SentimentRequest(BaseModel):
    texts:List[str]=Field(
        ...,
        min_items=1,
        max_items=2,
        description="List of text to analyze"
    )

    classifier_type: Optional[str] = Field(
        None,
        description="Optional classifier to use (e.g. 'logistic_regression')"
    )

    @validator("texts")
    def validate_text(cls,texts):
        for t in texts:
            if not t.strip():
                raise ValueError("Texts must not be empty")
            if len(t) > 1000:
                raise ValueError("Each text must be <= 1000 characters")
        return texts
