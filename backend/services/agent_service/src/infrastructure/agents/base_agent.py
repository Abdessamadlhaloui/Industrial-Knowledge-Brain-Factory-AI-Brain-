import logging
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional

from backend.services.agent_service.src.domain.models.agent_task import AgentTask
from backend.services.agent_service.src.domain.models.agent_result import AgentResult
from backend.services.agent_service.src.domain.models.tool_call import ToolCall
from backend.services.agent_service.src.infrastructure.llm_clients.anthropic_client import AnthropicClient

logger = logging.getLogger(__name__)


class BaseIndustrialAgent(ABC):
    """
    Abstract Base class for all Industrial Agents.
    Implements the ReAct (Reason -> Select Tool -> Execute Tool -> Observe) loop.
    """

    def __init__(
        self, 
        llm_client: AnthropicClient, 
        tool_registry: Dict[str, Any], 
        memory_store: Any,
        max_steps: int = 10,
        max_tokens_per_step: int = 2000
    ):
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.memory_store = memory_store
        self.max_steps = max_steps
        self.max_tokens_per_step = max_tokens_per_step

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @property
    @abstractmethod
    def allowed_tools(self) -> List[str]:
        pass

    @abstractmethod
    async def pre_process(self, input_task: AgentTask) -> AgentTask:
        pass

    @abstractmethod
    async def post_process(self, output_result: AgentResult) -> Any:
        pass

    async def run(self, input_task: AgentTask) -> AgentResult:
        """Execute the full ReAct loop."""
        start_time = time.time()
        
        # 1. Pre-process
        task = await self.pre_process(input_task)
        
        # 2. Get history from memory
        history = await self.memory_store.get_history(task.session_id)
        
        # Prepare tools
        tools_to_pass = [self.tool_registry[t] for t in self.allowed_tools if t in self.tool_registry]
        
        # Initialize conversation
        messages = history.copy()
        # Ensure alternating role format for Anthropic (user, assistant, user, etc.)
        messages.append({"role": "user", "content": task.query})
        
        tool_calls_history: List[ToolCall] = []
        total_tokens = 0
        
        for step in range(self.max_steps):
            logger.info("Agent Step %d/%d for session %s", step + 1, self.max_steps, task.session_id)
            
            # Call LLM
            response = await self.llm_client.complete(
                messages=messages,
                system_prompt=self.system_prompt,
                tools=tools_to_pass,
                max_tokens=self.max_tokens_per_step
            )
            
            # Tally tokens
            if hasattr(response, 'usage'):
                total_tokens += (response.usage.input_tokens + response.usage.output_tokens)
            
            # Parse response
            assistant_message = {"role": "assistant", "content": []}
            text_response = ""
            tool_uses = []
            
            for content_block in response.content:
                if content_block.type == "text":
                    text_response += content_block.text
                    assistant_message["content"].append({"type": "text", "text": content_block.text})
                elif content_block.type == "tool_use":
                    tool_uses.append(content_block)
                    assistant_message["content"].append(content_block.model_dump())
                    
            messages.append(assistant_message)
            
            # If no tools to call, ReAct loop terminates
            if not tool_uses:
                logger.info("No tool uses found. ReAct loop terminating.")
                break
                
            # Execute Tools
            tool_results_content = []
            for tool_use in tool_uses:
                tool_name = tool_use.name
                
                # Security: Validate tool
                if tool_name not in self.allowed_tools:
                    error_msg = f"Security Violation: Tool {tool_name} is not in allowed_tools list."
                    logger.error(error_msg)
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": error_msg,
                        "is_error": True
                    })
                    continue
                    
                tool_instance = self.tool_registry.get(tool_name)
                if not tool_instance:
                    error_msg = f"System Error: Tool {tool_name} not found in registry."
                    tool_results_content.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use.id,
                        "content": error_msg,
                        "is_error": True
                    })
                    continue
                    
                # Execute
                tool_result = await tool_instance.execute(tool_use.input)
                
                # Record
                tc = ToolCall(
                    tool_call_id=tool_use.id,
                    tool_name=tool_name,
                    inputs=tool_use.input,
                    output=tool_result.data if tool_result.success else None,
                    latency_ms=tool_result.latency_ms,
                    success=tool_result.success,
                    error=tool_result.error
                )
                tool_calls_history.append(tc)
                
                # Append result for next turn
                tool_results_content.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "content": str(tool_result.data) if tool_result.success else f"Error: {tool_result.error}",
                    "is_error": not tool_result.success
                })
                
            messages.append({"role": "user", "content": tool_results_content})

        if step == self.max_steps - 1:
            logger.warning("Max steps (%d) reached for session %s. Terminating ReAct loop.", self.max_steps, task.session_id)
            
        # Update episodic memory
        await self.memory_store.append(task.session_id, "user", task.query)
        await self.memory_store.append(task.session_id, "assistant", text_response)
        
        # 3. Post-process
        raw_result = AgentResult(
            task_id=task.task_id,
            session_id=task.session_id,
            output_text=text_response,
            tool_calls=tool_calls_history,
            total_latency_ms=(time.time() - start_time) * 1000,
            total_tokens=total_tokens
        )
        
        return await self.post_process(raw_result)

    async def stream(self, input_task: AgentTask) -> AsyncIterator[str]:
        """Streaming implementation for token-by-token output."""
        # Simplified mock stream for brevity
        yield "Starting agent...\n"
        result = await self.run(input_task)
        yield result.output_text
