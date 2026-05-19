import logging
from typing import Any, List

from backend.services.agent_service.src.infrastructure.agents.base_agent import BaseIndustrialAgent
from backend.services.agent_service.src.domain.models.agent_task import AgentTask
from backend.services.agent_service.src.domain.models.agent_result import AgentResult

logger = logging.getLogger(__name__)


class MonitoringAgent(BaseIndustrialAgent):
    """
    Continuous monitoring logic agent. Evaluates incoming anomalies to determine the response level.
    """

    @property
    def system_prompt(self) -> str:
        return (
            "You are a Level 1 Anomaly Triage Agent.\n"
            "Analyze the incoming anomaly event and decide the exact required escalation path.\n"
            "You must output ONE of the following decisions exactly:\n"
            "1. monitor_only (for minor fluctuations within tolerance)\n"
            "2. escalate_to_rca (for complex, unexplainable issues requiring deep analysis)\n"
            "3. create_work_order (for known, easily actionable wear-and-tear)\n"
            "4. page_on_call (for critical, plant-halting emergencies)\n"
            "Base your decision on the anomaly severity, frequency, and historical context."
        )

    @property
    def allowed_tools(self) -> List[str]:
        # Minimal toolset for quick triage
        return ["get_telemetry", "graph_query"]

    async def pre_process(self, input_task: AgentTask) -> AgentTask:
        return input_task

    async def post_process(self, output_result: AgentResult) -> AgentResult:
        # In a real scenario, this would parse the output text and trigger the next system via Kafka
        logger.info("Monitoring Agent output for task %s: %s", output_result.task_id, output_result.output_text)
        return output_result
