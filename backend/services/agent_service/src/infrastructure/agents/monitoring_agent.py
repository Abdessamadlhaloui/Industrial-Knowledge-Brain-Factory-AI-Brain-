from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

from backend.services.agent_service.src.infrastructure.agents.base_agent import BaseIndustrialAgent
from backend.services.agent_service.src.domain.models.agent_task import AgentTask
from backend.services.agent_service.src.domain.models.agent_result import AgentResult
from backend.shared.infrastructure.messaging.kafka_producer import KafkaMessageProducer

logger = logging.getLogger(__name__)

_ESCALATION_TOPIC: str = "ikb.anomalies.escalation"


class MonitoringAgent(BaseIndustrialAgent):
    """
    Continuous monitoring logic agent.
    Evaluates incoming anomalies to determine the required response level,
    then publishes an escalation event to Kafka for downstream pipelines.
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
        """Publish the triage decision to Kafka as a downstream escalation event.

        The Kafka publish is best-effort — a failure is logged at ERROR level
        but never swallows ``output_result``, which is always returned so the
        caller receives a complete result regardless of messaging infrastructure
        state.

        Args:
            output_result: Completed AgentResult from the ReAct loop.

        Returns:
            The unchanged ``output_result``, with or without a successful publish.
        """
        escalation_payload: Dict[str, Any] = {
            "task_id": output_result.task_id,
            "output_text": output_result.output_text,
            "confidence": output_result.confidence,
            "tenant_id": output_result.metadata.get("tenant_id"),
            "timestamp": datetime.utcnow().isoformat(),
            "agent_type": "monitoring",
        }

        producer: KafkaMessageProducer | None = None

        try:
            producer = KafkaMessageProducer()
            await producer.start()

            await producer.send(
                topic=_ESCALATION_TOPIC,
                value=escalation_payload,
                key=output_result.task_id,
            )

            logger.info(
                "Escalation event published — task_id=%s topic=%s tenant_id=%s",
                output_result.task_id,
                _ESCALATION_TOPIC,
                escalation_payload["tenant_id"],
            )

        except Exception:
            # Kafka failure must never drop the result — downstream consumers
            # can replay from DLQ; the agent result is the source of truth here.
            logger.error(
                "Failed to publish escalation event for task_id=%s topic=%s",
                output_result.task_id,
                _ESCALATION_TOPIC,
                exc_info=True,
            )

        finally:
            if producer is not None:
                await producer.stop()

        return output_result