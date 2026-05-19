from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class AgentTask(BaseModel):
    """Input payload for an Agent execution."""
    model_config = ConfigDict(frozen=True)

    session_id: str
    tenant_id: str
    task_id: str
    query: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
