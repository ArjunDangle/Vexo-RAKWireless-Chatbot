from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    query: str
    history: Optional[List[dict]] = [] # For multi-turn conversation context

class SourceDTO(BaseModel):
    title: str
    url: str # We will construct a file path or URL here
    confidence: float

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDTO]
    latency: float