import logging
from typing import Any, Dict, List

from pydantic import BaseModel, ConfigDict
from backend.services.agent_service.src.infrastructure.agents.base_agent import BaseIndustrialAgent
from backend.services.agent_service.src.domain.models.agent_task import AgentTask
from backend.services.agent_service.src.domain.models.agent_result import AgentResult

logger = logging.getLogger(__name__)


class RCAReport(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    symptom: str
    contributing_factors: List[str]
    root_cause: str
    recommended_actions: List[str]
    confidence: float
    data_gaps: List[str]
    failure_probability: float


class RootCauseAgent(BaseIndustrialAgent):
    """
    Expert Industrial RCA Engineer following IEC 62682 / ISA-18.2 standards.
    """

    @property
    def system_prompt(self) -> str:
        return (
            "You are an expert industrial RCA (Root Cause Analysis) engineer following IEC 62682 / ISA-18.2 standards.\n"
            "MANDATORY INSTRUCTIONS:\n"
            "1. ALWAYS retrieve sensor data from 2 hours before the event.\n"
            "2. ALWAYS check the knowledge graph for known failure modes and causality.\n"
            "3. ALWAYS search for similar historical incidents.\n"
            "4. Structure your final output strictly as: Symptom -> Contributing Factors -> Root Cause -> Actions.\n"
            "5. Provide a confidence score (0.0 to 1.0) based on data completeness.\n"
            "6. Explicitly list any data gaps that reduce your confidence.\n"
            "7. NEVER recommend unsafe actions without explicit safety checks/permits."
        )

    @property
    def allowed_tools(self) -> List[str]:
        return [
            "get_telemetry", 
            "rag_search", 
            "graph_query", 
            "causal_analysis", 
            "similar_incidents_search", 
            "report_generation"
        ]

    async def pre_process(self, input_task: AgentTask) -> AgentTask:
        """Enrich input with machine graph context (simulated 2-hop traversal)."""
        logger.info("Pre-processing RCA task for session %s", input_task.session_id)
        
        machine_id = input_task.metadata.get("machine_id")
        enrichment = ""
        if machine_id:
            # In reality, this would call the GraphTool/Service
            enrichment = f"\n[SYSTEM ENRICHMENT]: Target Machine {machine_id} context added to query."
            
        new_query = input_task.query + enrichment
        
        # We must return a new AgentTask since it's frozen
        return AgentTask(
            session_id=input_task.session_id,
            tenant_id=input_task.tenant_id,
            task_id=input_task.task_id,
            query=new_query,
            metadata=input_task.metadata
        )

    async def post_process(self, output_result: AgentResult) -> AgentResult:
        """
        Post-processing:
        - Safety validation
        - Confidence calibration based on tool usage
        """
        logger.info("Post-processing RCA task for session %s", output_result.session_id)
        
        tools_used = {tc.tool_name for tc in output_result.tool_calls}
        
        # Confidence Calibration
        calibrated_confidence = 0.0
        if "get_telemetry" in tools_used: calibrated_confidence += 0.3
        if "graph_query" in tools_used or "causal_analysis" in tools_used: calibrated_confidence += 0.3
        if "similar_incidents_search" in tools_used: calibrated_confidence += 0.2
        if "rag_search" in tools_used: calibrated_confidence += 0.2
        
        calibrated_confidence = min(calibrated_confidence, 1.0)
        
        # Safety Validation
        text = output_result.output_text
        if "replace" in text.lower() or "open" in text.lower() or "override" in text.lower():
            text += "\n\n[SYSTEM SAFETY OVERRIDE]: Recommended actions flag physical intervention. LOTO (Lockout-Tagout) and Safety Permits are MANDATORY before proceeding."

        # Re-pack the result
        return AgentResult(
            task_id=output_result.task_id,
            session_id=output_result.session_id,
            output_text=text,
            tool_calls=output_result.tool_calls,
            total_latency_ms=output_result.total_latency_ms,
            total_tokens=output_result.total_tokens,
            error=output_result.error
        )
