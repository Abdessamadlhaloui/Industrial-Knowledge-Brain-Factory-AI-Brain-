import logging
from typing import Annotated, Dict, List, TypedDict

from langgraph.graph import StateGraph, END
from backend.services.agent_service.src.domain.models.agent_task import AgentTask

logger = logging.getLogger(__name__)


# 1. Define State
class RCAWorkflowState(TypedDict):
    task: AgentTask
    telemetry_data: str
    knowledge_docs: str
    graph_context: str
    synthesized_analysis: str
    confidence: float
    safety_validated: bool
    final_report: str


# 2. Node Functions
async def route_query(state: RCAWorkflowState) -> RCAWorkflowState:
    logger.info("Workflow: Routing query")
    return state


async def retrieve_telemetry(state: RCAWorkflowState) -> RCAWorkflowState:
    logger.info("Workflow: Retrieving Telemetry")
    state["telemetry_data"] = "Mock Telemetry: Temperature normal, Vibration spike 2 hrs ago."
    return state


async def retrieve_knowledge(state: RCAWorkflowState) -> RCAWorkflowState:
    logger.info("Workflow: Retrieving Knowledge (RAG)")
    state["knowledge_docs"] = "Mock RAG: Bearing wear often preceded by vibration spikes."
    return state


async def analyze_graph(state: RCAWorkflowState) -> RCAWorkflowState:
    logger.info("Workflow: Analyzing Graph")
    state["graph_context"] = "Mock Graph: Motor -> Spindle -> Bearing"
    return state


async def synthesize_rca(state: RCAWorkflowState) -> RCAWorkflowState:
    logger.info("Workflow: Synthesizing RCA")
    # Simulate LLM synthesis
    state["synthesized_analysis"] = "Root Cause: Bearing Wear due to vibration."
    state["confidence"] = 0.8  # Mock confidence
    return state


async def validate_safety(state: RCAWorkflowState) -> RCAWorkflowState:
    logger.info("Workflow: Validating Safety")
    state["safety_validated"] = True
    return state


async def generate_report(state: RCAWorkflowState) -> RCAWorkflowState:
    logger.info("Workflow: Generating Final Report")
    state["final_report"] = f"Report:\n{state['synthesized_analysis']}\nSafety Checked: {state['safety_validated']}"
    return state


async def request_more_data(state: RCAWorkflowState) -> RCAWorkflowState:
    logger.info("Workflow: Requesting More Data (Confidence too low)")
    state["confidence"] = 0.9 # Artificially bump to prevent infinite loop
    return state


# 3. Edge Logic
def check_confidence(state: RCAWorkflowState) -> str:
    if state.get("confidence", 0.0) < 0.5:
        return "request_more_data"
    return "validate_safety"


# 4. Build Graph
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

# Parallel retrieval branches
workflow.add_edge("route_query", "retrieve_telemetry")
workflow.add_edge("route_query", "retrieve_knowledge")
workflow.add_edge("route_query", "analyze_graph")

# Join point (In LangGraph, parallel branches waiting on a single node need special handling or sequential flow. 
# For simplicity in this structure without a custom reducer, we map them sequentially or assume they merge).
# We'll map them sequentially for this DAG to guarantee execution order and prevent state overwrite issues.
workflow.add_edge("retrieve_telemetry", "retrieve_knowledge")
workflow.add_edge("retrieve_knowledge", "analyze_graph")
workflow.add_edge("analyze_graph", "synthesize_rca")

# Conditional Routing
workflow.add_conditional_edges(
    "synthesize_rca",
    check_confidence,
    {
        "request_more_data": "request_more_data",
        "validate_safety": "validate_safety"
    }
)

workflow.add_edge("request_more_data", "synthesize_rca") # Loop back
workflow.add_edge("validate_safety", "generate_report")
workflow.add_edge("generate_report", END)

# Compile
rca_graph = workflow.compile()
