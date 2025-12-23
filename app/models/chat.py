from pydantic import BaseModel, Field, AliasChoices
from typing import List, Optional, Dict

class ChatRequest(BaseModel):
    """
    Schema for the incoming chat request.
    Supports both 'message' and 'query' keys to prevent 422 errors.
    """
    message: str = Field(
        ..., 
        validation_alias=AliasChoices('message', 'query'),
        description="The user's technical question."
    )
    history: Optional[List[Dict[str, str]]] = Field(
        default=[], 
        description="Optional conversation history for context."
    )

class SourceMetadata(BaseModel):
    """
    Schema for document source citations.
    """
    title: str
    url: str
    confidence: float

class ChatResponse(BaseModel):
    """
    Schema for the API response.
    """
    answer: str
    sources: List[SourceMetadata]
    latency: Optional[float] = None