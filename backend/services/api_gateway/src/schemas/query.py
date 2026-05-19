from enum import Enum
from typing import List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict


class QueryMode(str, Enum):
    CONVERSATIONAL = "CONVERSATIONAL"
    ANALYTICAL = "ANALYTICAL"
    DIAGNOSTIC = "DIAGNOSTIC"
    PREDICTIVE = "PREDICTIVE"


class QueryRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str = Field(..., min_length=3, max_length=2000, description="The natural language query text.")
    mode: QueryMode = Field(default=QueryMode.CONVERSATIONAL, description="The operating mode of the AI.")
    machine_ids: Optional[List[str]] = Field(default=None, description="Optional list of machine IDs to scope the query.")
    time_range: Optional[str] = Field(default=None, description="Optional time range descriptor (e.g., 'last 24 hours').")
    use_agents: bool = Field(default=True, description="Whether to route through the specialized Agent orchestration or use simple RAG.")
    stream_response: bool = Field(default=False, description="If true, use SSE streaming. (Ignored for non-streaming endpoint).")
    max_tokens: int = Field(default=2000, le=8000, description="Max tokens for the response.")
    tenant_id: str = Field(..., description="Target tenant ID. In production, this is overridden by the JWT auth middleware.")


class SourceReference(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    title: str
    page: Optional[int] = None
    score: float
    excerpt: str
    source_type: str


class QueryResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: List[SourceReference] = Field(default_factory=list)
    reasoning_steps: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    failure_probability: Optional[float] = None
    related_incidents: List[str] = Field(default_factory=list)
    agent_trace: Optional[Any] = None
    latency_ms: float
