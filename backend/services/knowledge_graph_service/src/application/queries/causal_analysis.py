import logging
from typing import List, Optional
from pydantic import BaseModel, ConfigDict

from backend.services.knowledge_graph_service.src.infrastructure.graph_db.neo4j_client import Neo4jClient
from backend.services.knowledge_graph_service.src.infrastructure.graph_db.cypher_queries import CAUSAL_PATH_ANALYSIS

logger = logging.getLogger(__name__)


class CausalAnalysisQuery(BaseModel):
    model_config = ConfigDict(frozen=True)
    sensor_id: str
    anomaly_type: str
    time_window: int  # Minutes or hours


class CausalChain(BaseModel):
    root_cause_id: str
    root_cause_name: str
    detection_confidence: float
    propagation_path: List[str]
    compound_risk: float
    recommended_actions: List[str]


class CausalAnalysisHandler:
    """
    Handles CausalAnalysisQuery.
    Queries the Knowledge Graph to trace a sensor alert down to its root cause, 
    calculating the probabilistic causal chain.
    """

    def __init__(self, neo4j_client: Neo4jClient):
        self.neo4j_client = neo4j_client

    async def handle(self, query: CausalAnalysisQuery) -> List[CausalChain]:
        """
        Execute the causal path analysis query and format the results.
        Uses RAGAS-style confidence scoring conceptually through the `compound_risk`.
        """
        logger.info("Executing causal analysis for sensor: %s", query.sensor_id)
        
        # We assume min_confidence is a parameterized threshold, e.g., 0.5
        params = {
            "sensor_id": query.sensor_id,
            "min_confidence": 0.5
        }
        
        records = await self.neo4j_client.execute_query(CAUSAL_PATH_ANALYSIS, params)
        
        chains = []
        for record in records:
            # Note: The query might return multiple paths or duplicate actions depending on graph density
            recommended_action = record.get("recommended_action")
            actions = [recommended_action] if recommended_action else []
            
            chain = CausalChain(
                root_cause_id=record.get("root_cause_id", ""),
                root_cause_name=record.get("root_cause_name", "Unknown"),
                detection_confidence=float(record.get("detection_confidence", 0.0)),
                propagation_path=record.get("propagation_path", []),
                compound_risk=float(record.get("compound_risk", 0.0)),
                recommended_actions=actions
            )
            chains.append(chain)
            
        return chains
