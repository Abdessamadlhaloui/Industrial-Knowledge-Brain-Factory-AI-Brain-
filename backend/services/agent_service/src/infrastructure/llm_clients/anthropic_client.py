import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

import anthropic
from anthropic import AsyncAnthropic, APIStatusError, APITimeoutError, APIError
from backend.shared.utils.retry import retry

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Domain-level exception for LLM API failures."""
    pass


class AnthropicClient:
    """
    Anthropic Claude client wrapper.
    Translates tool schemas to Claude's format and handles retries.
    """

    def __init__(self, model_name: str = "claude-sonnet-4-20250514"):
        self.model_name = model_name
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is required and not set. Cannot start agent_service without a valid LLM client."
            )
        self.client = AsyncAnthropic(api_key=self.api_key)

    def _format_tools(self, tools: List[Any]) -> List[Dict[str, Any]]:
        """Converts BaseTool instances into Anthropic's tool_use format."""
        formatted = []
        for tool in tools:
            formatted.append({
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema
            })
        return formatted

    @retry(max_attempts=3, exceptions=(APIStatusError, APITimeoutError), backoff_factor=2.0)
    async def complete(
        self, 
        messages: List[Dict[str, Any]], 
        system_prompt: str, 
        tools: List[Any] = None, 
        stream: bool = False,
        max_tokens: int = 2000
    ) -> Any:
        """
        Execute completion with retries.
        """
        kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
        
        if tools:
            kwargs["tools"] = self._format_tools(tools)
            
        try:
            if stream:
                return await self.client.messages.create(stream=True, **kwargs)
                
            return await self.client.messages.create(**kwargs)
        except APIError as e:
            logger.error("Anthropic API call failed: %s", str(e))
            raise LLMClientError(f"Failed to fetch LLM completion: {str(e)}") from e
