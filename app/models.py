from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class AskRequest(BaseModel):
    client_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    index_name: Optional[str] = None

    # <-- important: let the UI pass conversation_id (for history)
    conversation_id: Optional[str] = None

    # allow future-safe extra keys from the UI without 422
    model_config = ConfigDict(extra="ignore")


class RefItem(BaseModel):
    title: Optional[str] = None
    breadcrumb: Optional[str] = None
    url: Optional[str] = None
    score: Optional[float] = None
    viq: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class AskResponse(BaseModel):
    answer: str
    references: List[RefItem] = []
    meta: Dict[str, Any] = {}
