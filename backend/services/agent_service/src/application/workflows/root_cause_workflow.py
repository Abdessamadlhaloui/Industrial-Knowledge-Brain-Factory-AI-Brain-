from __future__ import annotations

import logging
import re
import ast
import json
from datetime import datetime, timedelta, timezone
from typing import Annotated, Dict, List, Tuple, TypedDict
from typing import Any

from langgraph.graph import StateGraph, END
from backend.services.agent_service.src.domain.models.agent_task import AgentTask
from backend.services.agent_service.src.infrastructure.tools.telemetry_tool import TelemetryTool
from backend.services.agent_service.src.infrastructure.tools.rag_tool import RagTool
from backend.services.agent_service.src.infrastructure.tools.graph_tool import GraphTool

from backend.services.agent_service.src.infrastructure.llm_clients.anthropic_client import (
    AnthropicClient,
)

logger = logging.getLogger(__name__)

UNSAFE_KEYWORDS: list[str] = [
    "override safety",
    "bypass interlock",
    "ignore threshold",
    "disable limit",
    "disable cooling",
    "exceed rated",
    "force override",
    "skip inspection",
    "remove guard",
]

# Maximum enrichment attempts before the workflow exits the retry loop gracefully.
MAX_ENRICHMENT_ATTEMPTS: int = 2

# Widened telemetry window used during enrichment (vs. 2 hours on the first pass).
ENRICHMENT_TELEMETRY_WINDOW_HOURS: int = 6

# ---------------------------------------------------------------------------
# Dependency Injection Contract
# ---------------------------------------------------------------------------
# The following keys MUST be populated in RCAWorkflowState before invoking
# rca_graph.  They are set once at workflow initialisation time (e.g. in the
# FastAPI lifespan or the AgentService factory) and carried through every node.
#
#   state["telemetry_client"]  — InfluxDBTelemetryClient
#       Required by: retrieve_telemetry, request_more_data
#
#   state["rag_client"]        — httpx.AsyncClient (pointed at rag_service)
#       Required by: retrieve_knowledge
#
#   state["graph_client"]      — Neo4jGraphClient
#       Required by: analyze_graph
#
# If any of these are absent the node that needs them will log CRITICAL, write
# an error string into the corresponding state key, and return early rather
# than falling through to a silent mock path inside the tool.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 1. Define State
# ---------------------------------------------------------------------------

class RCAWorkflowState(TypedDict):
    task: AgentTask
    # Infrastructure clients — injected before graph execution
    telemetry_client: Any   # InfluxDBTelemetryClient
    rag_client: Any         # httpx.AsyncClient
    graph_client: Any       # Neo4jGraphClient
    # Node outputs
    telemetry_data: str
    knowledge_docs: str
    graph_context: str
    synthesized_analysis: str
    confidence: float
    safety_validated: bool
    final_report: dict[str, Any]
    # Enrichment tracking — populated by request_more_data
    enrichment_attempts: int
    data_sufficient: bool


# ---------------------------------------------------------------------------
# Helper: Robust LLM JSON response parser
# ---------------------------------------------------------------------------

def _parse_llm_rca_response(response_text: str) -> Tuple[str, float]:
    """
    Robustly parse a structured JSON response from the RCA synthesis LLM.

    Handles markdown code fences, trailing commas, and Python-dict-like output.
    Never raises — degrades gracefully to (raw_text, 0.5) on total failure.

    Args:
        response_text: Raw string output from the LLM completion call.

    Returns:
        Tuple of (formatted_analysis: str, confidence: float), where
        confidence is clamped to [0.0, 1.0].

    Examples:
        >>> text_fenced = '```json\\n{"root_cause": "Bearing wear", "recommended_action": "Replace bearing", "confidence": 0.85}\\n```'
        >>> analysis, conf = _parse_llm_rca_response(text_fenced)
        >>> "Bearing wear" in analysis and "Replace bearing" in analysis
        True
        >>> conf
        0.85

        >>> text_clean = '{"root_cause": "Overheating", "recommended_action": "Improve cooling", "confidence": 1.5}'
        >>> analysis, conf = _parse_llm_rca_response(text_clean)
        >>> "Overheating" in analysis
        True
        >>> conf  # Clamped from 1.5 to 1.0
        1.0

        >>> text_malformed = 'Sorry, I cannot provide a structured response right now.'
        >>> analysis, conf = _parse_llm_rca_response(text_malformed)
        >>> analysis == text_malformed
        True
        >>> conf
        0.5
    """
    try:
        cleaned: str = re.sub(r"```json|```", "", response_text).strip()

        parsed: dict | None = None
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            try:
                candidate = ast.literal_eval(cleaned)
                if isinstance(candidate, dict):
                    parsed = candidate
            except (ValueError, SyntaxError):
                pass

        if parsed is None:
            logger.warning(
                "LLM RCA response could not be parsed as JSON or dict literal. "
                "Raw text (truncated): %.500s",
                response_text,
            )
            return (response_text, 0.5)

        root_cause: str = str(parsed.get("root_cause", "Unknown root cause."))
        recommended_action: str = str(
            parsed.get("recommended_action", "No recommended action provided.")
        )
        raw_confidence: float = parsed.get("confidence", 0.5)

        analysis: str = (
            f"Root Cause: {root_cause}\n"
            f"Recommended Action: {recommended_action}"
        )

        confidence: float = max(0.0, min(1.0, float(raw_confidence)))
        return (analysis, confidence)

    except Exception:
        logger.warning(
            "Unexpected error parsing LLM RCA response. "
            "Raw text (truncated): %.500s",
            response_text,
            exc_info=True,
        )
        return (response_text, 0.5)


# ---------------------------------------------------------------------------
# 2. Node Functions
# ---------------------------------------------------------------------------

async def route_query(state: RCAWorkflowState) -> RCAWorkflowState:
    logger.info("Workflow: Routing query")
    return state


async def retrieve_telemetry(state: RCAWorkflowState) -> RCAWorkflowState:
    logger.info("Workflow: Retrieving Telemetry")

    # ── DI guard ─────────────────────────────────────────────────────────────
    if state.get("telemetry_client") is None:
        logger.critical(
            "retrieve_telemetry: 'telemetry_client' not injected into workflow state. "
            "Populate state['telemetry_client'] with an InfluxDBTelemetryClient "
            "before invoking the graph."
        )
        state["telemetry_data"] = (
            "ERROR: TelemetryTool client not injected into workflow state."
        )
        return state

    task: AgentTask | None = state.get("task")
    if not task:
        state["telemetry_data"] = "Error: No task provided in state."
        return state

    machine_id: str | None = task.metadata.get("machine_id")
    if not machine_id:
        state["telemetry_data"] = (
            "No telemetry retrieved: 'machine_id' missing from task metadata."
        )
        return state

    now: datetime = datetime.now(timezone.utc)
    start: datetime = now - timedelta(hours=2)

    tool = TelemetryTool(telemetry_client=state.get("telemetry_client"))

    params: dict[str, Any] = {
        "machine_id": machine_id,
        "metric_names": ["temperature", "vibration"],
        "start_time": start.isoformat(),
        "end_time": now.isoformat(),
        "aggregation": "mean",
    }

    try:
        result = await tool.execute(params)
        if result.success:
            state["telemetry_data"] = str(result.data)
        else:
            logger.error("TelemetryTool returned failure: %s", result.error)
            state["telemetry_data"] = f"Error retrieving telemetry: {result.error}"
    except Exception as exc:
        logger.error("Exception during TelemetryTool execution: %s", exc)
        state["telemetry_data"] = f"Error retrieving telemetry: {exc}"

    return state


async def retrieve_knowledge(state: RCAWorkflowState) -> RCAWorkflowState:
    logger.info("Workflow: Retrieving Knowledge (RAG)")

    # ── DI guard ─────────────────────────────────────────────────────────────
    if state.get("rag_client") is None:
        logger.critical(
            "retrieve_knowledge: 'rag_client' not injected into workflow state. "
            "Populate state['rag_client'] with a configured httpx.AsyncClient "
            "before invoking the graph."
        )
        state["knowledge_docs"] = (
            "ERROR: RagTool client not injected into workflow state."
        )
        return state

    task: AgentTask | None = state.get("task")
    if not task:
        state["knowledge_docs"] = "Error: No task provided in state."
        return state

    query: str | None = task.query
    if not query:
        state["knowledge_docs"] = "Error: Task query is empty."
        return state

    tool = RagTool(rag_client=state.get("rag_client"))

    params: dict[str, Any] = {
        "query": query,
        "top_k": 3,
    }

    try:
        result = await tool.execute(params)
        if result.success:
            state["knowledge_docs"] = str(result.data)
        else:
            logger.error("RagTool returned failure: %s", result.error)
            state["knowledge_docs"] = f"Error retrieving knowledge: {result.error}"
    except Exception as exc:
        logger.error("Exception during RagTool execution: %s", exc)
        state["knowledge_docs"] = f"Error retrieving knowledge: {exc}"

    return state


async def analyze_graph(state: RCAWorkflowState) -> RCAWorkflowState:
    logger.info("Workflow: Analyzing Graph")

    # ── DI guard ─────────────────────────────────────────────────────────────
    if state.get("graph_client") is None:
        logger.critical(
            "analyze_graph: 'graph_client' not injected into workflow state. "
            "Populate state['graph_client'] with a Neo4jGraphClient "
            "before invoking the graph."
        )
        state["graph_context"] = (
            "ERROR: GraphTool client not injected into workflow state."
        )
        return state

    task: AgentTask | None = state.get("task")
    if not task:
        state["graph_context"] = "Error: No task provided in state."
        return state

    machine_id: str | None = task.metadata.get("machine_id")
    if not machine_id:
        state["graph_context"] = (
            "No graph context retrieved: 'machine_id' missing from task metadata."
        )
        return state

    tool = GraphTool(graph_client=state.get("graph_client"))

    params: dict[str, Any] = {
        "query_type": "causal_analysis",
        "machine_id": machine_id,
    }

    try:
        result = await tool.execute(params)
        if result.success:
            state["graph_context"] = str(result.data)
        else:
            logger.error("GraphTool returned failure: %s", result.error)
            state["graph_context"] = f"Error retrieving graph context: {result.error}"
    except Exception as exc:
        logger.error("Exception during GraphTool execution: %s", exc)
        state["graph_context"] = f"Error retrieving graph context: {exc}"

    return state


async def synthesize_rca(state: RCAWorkflowState) -> RCAWorkflowState:
    logger.info("Workflow: Synthesizing RCA")

    try:
        task: AgentTask | None = state.get("task")

        if task is None:
            state["synthesized_analysis"] = (
                "RCA synthesis failed: task missing from workflow state."
            )
            state["confidence"] = 0.0
            return state

        prompt = f"""
You are a principal industrial reliability engineer performing a root cause analysis.
Synthesize a diagnostic report from the following multi-source context:

Machine Query: {task.query}
Telemetry Data: {state.get('telemetry_data', 'No telemetry retrieved.')}
Document Context: {state.get('knowledge_docs', 'No documents matched.')}
Knowledge Graph Topology: {state.get('graph_context', 'No topology matched.')}

Respond ONLY with valid JSON containing exactly these keys:
- "root_cause": string (detailed failure mechanism explanation)
- "confidence": float between 0.0 and 1.0
- "recommended_action": string (step-by-step resolution)

JSON:
"""

        llm = AnthropicClient()
        response_text: str = await llm.complete(prompt)

        state["synthesized_analysis"], state["confidence"] = _parse_llm_rca_response(response_text)

    except Exception:
        logger.error("RCA synthesis workflow node failed", exc_info=True)
        state["synthesized_analysis"] = "RCA synthesis failed: unexpected error."
        state["confidence"] = 0.0

    return state


async def request_more_data(state: RCAWorkflowState) -> RCAWorkflowState:
    """
    Enrich workflow state by re-fetching telemetry over a wider time window.

    Triggered when synthesize_rca produces confidence < 0.7.  Widens the
    telemetry window from 2 to 6 hours and updates state["telemetry_data"]
    with the enriched signal so the next synthesize_rca pass has genuinely
    more data to reason over.

    ``confidence`` is deliberately NOT modified — the next synthesize_rca
    node must derive a real confidence from the enriched data.
    """
    logger.info("Workflow: Requesting More Data (wider telemetry window)")

    enrichment_attempts: int = state.get("enrichment_attempts", 0) + 1
    state["enrichment_attempts"] = enrichment_attempts

    logger.info("Enrichment attempt %d / %d", enrichment_attempts, MAX_ENRICHMENT_ATTEMPTS)

    if enrichment_attempts >= MAX_ENRICHMENT_ATTEMPTS:
        logger.warning(
            "Max enrichment attempts reached (%d). Marking data_sufficient=False "
            "so the workflow can exit the retry loop gracefully.",
            MAX_ENRICHMENT_ATTEMPTS,
        )
        state["data_sufficient"] = False
        return state

    task: AgentTask | None = state.get("task")
    if task is None:
        logger.error("request_more_data: task missing from state.")
        return state

    machine_id: str | None = task.metadata.get("machine_id")
    if not machine_id:
        logger.error("request_more_data: 'machine_id' missing from task metadata.")
        return state

    if state.get("telemetry_client") is None:
        logger.critical(
            "request_more_data: 'telemetry_client' not injected — cannot widen window."
        )
        return state

    now: datetime = datetime.now(timezone.utc)
    start: datetime = now - timedelta(hours=ENRICHMENT_TELEMETRY_WINDOW_HOURS)

    logger.info(
        "Re-fetching telemetry for machine_id=%s over %d-hour window [%s → %s]",
        machine_id,
        ENRICHMENT_TELEMETRY_WINDOW_HOURS,
        start.isoformat(),
        now.isoformat(),
    )

    tool = TelemetryTool(telemetry_client=state.get("telemetry_client"))

    params: dict[str, Any] = {
        "machine_id": machine_id,
        "metric_names": ["temperature", "vibration"],
        "start_time": start.isoformat(),
        "end_time": now.isoformat(),
        "aggregation": "mean",
    }

    try:
        result = await tool.execute(params)

        if result.success and result.data:
            enriched: str = str(result.data)
            state["telemetry_data"] = enriched
            logger.info(
                "Telemetry enriched for machine_id=%s "
                "(attempt=%d window=%dh chars=%d)",
                machine_id,
                enrichment_attempts,
                ENRICHMENT_TELEMETRY_WINDOW_HOURS,
                len(enriched),
            )
        elif result.success and not result.data:
            logger.warning(
                "Wider telemetry window returned empty data for machine_id=%s. "
                "state['telemetry_data'] unchanged.",
                machine_id,
            )
        else:
            logger.error(
                "TelemetryTool enrichment fetch failed for machine_id=%s: %s",
                machine_id,
                result.error,
            )

    except Exception:
        logger.error(
            "Exception during enrichment TelemetryTool execution for machine_id=%s",
            machine_id,
            exc_info=True,
        )

    return state


async def validate_safety(state: RCAWorkflowState) -> RCAWorkflowState:
    """
    Validate synthesized RCA output against basic industrial safety boundaries.

    First-pass keyword filter. Production hardening requires an LLM-based
    safety classifier with domain-specific rule engine integration.
    """
    logger.info("Workflow: Validating Safety")

    try:
        analysis: str = (state.get("synthesized_analysis", "") or "").lower()

        task: AgentTask | None = state.get("task")
        task_query: str = ""
        if task is not None:
            task_query = (task.query or "").lower()

        for keyword in UNSAFE_KEYWORDS:
            if keyword in analysis or keyword in task_query:
                logger.warning(
                    "Safety validation failed due to unsafe keyword match: %s",
                    keyword,
                )
                state["safety_validated"] = False
                return state

        logger.info("Safety validation passed.")
        state["safety_validated"] = True

    except Exception:
        logger.error("Unexpected error during safety validation", exc_info=True)
        state["safety_validated"] = False

    return state


async def generate_report(state: RCAWorkflowState) -> RCAWorkflowState:
    """
    Produce a structured, JSON-serialisable report dict consumed by the
    frontend dashboard.  Writes to state["final_report"] — never raises.
    """
    logger.info("Workflow: Generating Final Report")

    try:
        task: AgentTask | None = state.get("task")
        task_id: str = str(task.task_id) if task else "unknown"
        confidence: float = state.get("confidence", 0.0)
        safety_validated: bool = state.get("safety_validated", False)

        safety_label: str = (
            "PASSED" if safety_validated else "FAILED — MANUAL REVIEW REQUIRED"
        )

        report: dict[str, Any] = {
            "title": "Root Cause Analysis Report",
            "query": task.query if task else "",
            "machine_id": task.metadata.get("machine_id") if task else None,
            "tenant_id": task.metadata.get("tenant_id") if task else None,
            "timestamp": datetime.utcnow().isoformat(),
            "analysis": {
                "synthesis": state.get("synthesized_analysis", ""),
                "confidence": confidence,
                "safety_check": safety_label,
            },
            "evidence": {
                # Truncate raw telemetry to avoid bloating the report payload.
                "telemetry_summary": state.get("telemetry_data", "")[:500],
                "knowledge_docs_count": len(
                    str(state.get("knowledge_docs", "")).split("\n")
                ),
                "graph_context": state.get("graph_context", ""),
            },
            "status": "complete",
        }

        # Append data-sufficiency warning when enrichment loop was exhausted.
        if not state.get("data_sufficient", True):
            report["warning"] = (
                "Maximum enrichment attempts reached — report based on best available data."
            )

        state["final_report"] = report

        logger.info(
            "Report generated for task_id=%s confidence=%.2f safety=%s",
            task_id,
            confidence,
            safety_label,
        )

        if not safety_validated:
            logger.warning(
                "Report generated with FAILED safety check for task %s",
                task_id,
            )

    except Exception:
        logger.error("generate_report failed unexpectedly", exc_info=True)
        # Degrade to a minimal error report so the graph always terminates cleanly.
        state["final_report"] = {
            "title": "Root Cause Analysis Report",
            "status": "error",
            "error": "Report generation failed — see service logs.",
        }

    return state


# ---------------------------------------------------------------------------
# 3. Edge Logic
# ---------------------------------------------------------------------------

def check_confidence(state: RCAWorkflowState) -> str:
    """Route after synthesize_rca.

    Exits the enrichment loop early if request_more_data has flagged that
    data is insufficient (max attempts reached), preventing infinite retries.
    """
    if not state.get("data_sufficient", True):
        logger.warning(
            "Exiting enrichment loop — data_sufficient=False. "
            "Proceeding to safety validation with confidence=%.2f.",
            state.get("confidence", 0.0),
        )
        return "validate_safety"

    if state.get("confidence", 0.0) < 0.7:
        return "request_more_data"

    return "validate_safety"


# ---------------------------------------------------------------------------
# 4. Build Graph
# ---------------------------------------------------------------------------

workflow = StateGraph(RCAWorkflowState)

workflow.add_node("route_query", route_query)
workflow.add_node("retrieve_telemetry", retrieve_telemetry)
workflow.add_node("retrieve_knowledge", retrieve_knowledge)
workflow.add_node("analyze_graph", analyze_graph)
workflow.add_node("synthesize_rca", synthesize_rca)
workflow.add_node("request_more_data", request_more_data)
workflow.add_node("validate_safety", validate_safety)
workflow.add_node("generate_report", generate_report)

workflow.set_entry_point("route_query")

workflow.add_edge("route_query", "retrieve_telemetry")
workflow.add_edge("retrieve_telemetry", "retrieve_knowledge")
workflow.add_edge("retrieve_knowledge", "analyze_graph")
workflow.add_edge("analyze_graph", "synthesize_rca")

workflow.add_conditional_edges(
    "synthesize_rca",
    check_confidence,
    {
        "request_more_data": "request_more_data",
        "validate_safety": "validate_safety",
    },
)

workflow.add_edge("request_more_data", "synthesize_rca")
workflow.add_edge("validate_safety", "generate_report")
workflow.add_edge("generate_report", END)

rca_graph = workflow.compile()