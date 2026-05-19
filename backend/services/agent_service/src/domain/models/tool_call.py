from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict


class ToolCall(BaseModel):
    """Represents a single execution of a tool by the Agent."""
    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    tool_name: str
    inputs: Dict[str, Any]
    output: Any
    latency_ms: float
    success: bool
    error: Optional[str] = None
