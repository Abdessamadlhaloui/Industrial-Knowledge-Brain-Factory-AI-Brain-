import pytest
from unittest.mock import AsyncMock

from backend.services.knowledge_graph_service.src.infrastructure.graph_db.neo4j_client import Neo4jClient
from backend.services.knowledge_graph_service.src.application.queries.causal_analysis import CausalAnalysisHandler, CausalAnalysisQuery
from backend.services.knowledge_graph_service.src.infrastructure.graph_db.cypher_queries import CAUSAL_PATH_ANALYSIS


@pytest.mark.asyncio
async def test_causal_path_analysis_handler():
    mock_neo4j = AsyncMock(spec=Neo4jClient)
    
    # Mock Neo4j return record for CAUSAL_PATH_ANALYSIS
    mock_record = {
        "sensor_id": "SENS_123",
        "root_cause_id": "FM_TMP_01",
        "root_cause_name": "Motor Overheating",
        "detection_confidence": 0.85,
        "propagation_path": ["Motor Overheating", "Bearing Wear"],
        "compound_risk": 0.1275,  # 0.85 * 0.15
        "recommended_action": "Replace Bearing"
    }
    
    mock_neo4j.execute_query.return_value = [mock_record]
    
    handler = CausalAnalysisHandler(mock_neo4j)
    
    query = CausalAnalysisQuery(
        sensor_id="SENS_123",
        anomaly_type="temperature_spike",
        time_window=60
    )
    
    chains = await handler.handle(query)
    
    # Assert query execution
    mock_neo4j.execute_query.assert_called_once_with(
        CAUSAL_PATH_ANALYSIS, 
        {"sensor_id": "SENS_123", "min_confidence": 0.5}
    )
    
    # Assert DTO mapping
    assert len(chains) == 1
    chain = chains[0]
    
    assert chain.root_cause_id == "FM_TMP_01"
    assert chain.detection_confidence == 0.85
    assert "Bearing Wear" in chain.propagation_path
    assert chain.compound_risk == 0.1275
    assert "Replace Bearing" in chain.recommended_actions
