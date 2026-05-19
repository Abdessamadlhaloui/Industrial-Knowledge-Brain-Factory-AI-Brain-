import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from pydantic import BaseModel, ConfigDict


class ToolResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    success: bool
    data: Any
    error: Optional[str] = None
    latency_ms: float = 0.0


class BaseTool(ABC):
    """
    Abstract Base Class for all Agent Tools.
    """
    name: str
    description: str
    input_schema: Dict[str, Any]  # JSON Schema dictionary
    
    # Default timeout in seconds
    timeout_seconds: float = 30.0

    @abstractmethod
    async def _execute_impl(self, params: Dict[str, Any]) -> Any:
        """Internal implementation to be overridden by subclasses."""
        pass

    async def execute(self, params: Dict[str, Any]) -> ToolResult:
        """
        Executes the tool with standardized latency tracking and error handling.
        """
        start_time = time.time()
        try:
            # Here we would normally implement a timeout wrapper
            # For simplicity in this structure we just await the impl
            data = await self._execute_impl(params)
            latency = (time.time() - start_time) * 1000
            
            return ToolResult(
                success=True,
                data=data,
                latency_ms=latency
            )
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                latency_ms=latency
            )
