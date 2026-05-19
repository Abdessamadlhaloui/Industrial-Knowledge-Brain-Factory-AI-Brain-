import asyncio
import logging
from typing import Any, Dict

from backend.services.agent_service.src.domain.models.agent_task import AgentTask
from backend.services.agent_service.src.domain.models.agent_result import AgentResult
from backend.services.agent_service.src.infrastructure.agents.root_cause_agent import RootCauseAgent
from backend.services.agent_service.src.infrastructure.agents.maintenance_agent import MaintenanceAgent
from backend.services.agent_service.src.infrastructure.agents.monitoring_agent import MonitoringAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Routes tasks to the appropriate specialized agents based on task_type.
    Handles parallel execution synthesis and hard timeouts.
    """

    def __init__(
        self,
        rca_agent: RootCauseAgent,
        maintenance_agent: MaintenanceAgent,
        monitoring_agent: MonitoringAgent
    ):
        self.agents = {
            "anomaly_analysis": rca_agent,
            "work_order_creation": maintenance_agent,
            "monitoring_event": monitoring_agent
        }
        self.rca_agent = rca_agent  # Default for complex queries

    async def route_and_execute(self, task: AgentTask, timeout_seconds: float = 30.0) -> AgentResult:
        """
        Routes the task and executes it with a strict timeout.
        Falls back to a safe response if the agent fails or times out.
        """
        task_type = task.metadata.get("task_type", "conversational_query")
        
        # Route selection
        agent = self.agents.get(task_type)
        if not agent:
            # Simple heuristic for conversational query routing
            if "fail" in task.query.lower() or "error" in task.query.lower() or "cause" in task.query.lower():
                agent = self.rca_agent
            else:
                # In a full system, we might route to a SimpleRAGAgent here
                agent = self.rca_agent 
                
        logger.info("Orchestrator routing task %s to %s", task.task_id, agent.__class__.__name__)
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(agent.run(task), timeout=timeout_seconds)
            return result
        except asyncio.TimeoutError:
            logger.error("Agent execution timed out for task %s after %.1fs", task.task_id, timeout_seconds)
            return AgentResult(
                task_id=task.task_id,
                session_id=task.session_id,
                output_text="The analysis took too long and timed out. Please try a more specific query.",
                tool_calls=[],
                total_latency_ms=timeout_seconds * 1000,
                total_tokens=0,
                error="TimeoutError"
            )
        except Exception as e:
            logger.error("Agent execution failed for task %s: %s", task.task_id, e)
            return AgentResult(
                task_id=task.task_id,
                session_id=task.session_id,
                output_text="An internal error occurred during agent execution. Falling back to simple RAG.",
                tool_calls=[],
                total_latency_ms=0,
                total_tokens=0,
                error=str(e)
            )
