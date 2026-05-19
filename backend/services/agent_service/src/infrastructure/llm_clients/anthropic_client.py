import logging
import os
from typing import Any, AsyncIterator, Dict, List, Optional

from anthropic import AsyncAnthropic, APIStatusError, APITimeoutError
from backend.shared.utils.retry import retry

logger = logging.getLogger(__name__)


class AnthropicClient:
    """
    Anthropic Claude 3.5 Sonnet client wrapper.
    Translates tool schemas to Claude's format and handles retries.
    """

    def __init__(self, model_name: str = "claude-3-5-sonnet-20240620"):
        self.model_name = model_name
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        self.client = AsyncAnthropic(api_key=self.api_key) if self.api_key else None

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
        if not self.client:
            logger.warning("ANTHROPIC_API_KEY not set. Mocking response.")
            return self._mock_response(messages)

        kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": messages,
        }
        
        if tools:
            kwargs["tools"] = self._format_tools(tools)
            
        if stream:
            return await self.client.messages.create(stream=True, **kwargs)
            
        return await self.client.messages.create(**kwargs)

    def _mock_response(self, messages: List[Dict[str, Any]]) -> Any:
        """Mock response for testing without API key."""
        class MockMessage:
            def __init__(self):
                self.content = [{"type": "text", "text": "Mock LLM response."}]
                self.stop_reason = "end_turn"
                self.usage = type("Usage", (), {"input_tokens": 10, "output_tokens": 10})()
        return MockMessage()
