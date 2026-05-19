import pytest
from unittest.mock import AsyncMock, patch

from backend.services.knowledge_graph_service.src.infrastructure.extractors.industrial_ner import IndustrialNER
from backend.services.knowledge_graph_service.src.infrastructure.extractors.llm_extractor import LLMExtractor
from backend.services.knowledge_graph_service.src.infrastructure.extractors.relation_extractor import RelationExtractor


@pytest.mark.asyncio
@patch("backend.services.knowledge_graph_service.src.infrastructure.extractors.industrial_ner.spacy.load")
async def test_industrial_ner_extractor(mock_spacy_load):
    # Mocking SpaCy's load to prevent a massive 800MB model download during tests
    class MockEntity:
        def __init__(self, text, label_, start_char, end_char):
            self.text = text
            self.label_ = label_
            self.start_char = start_char
            self.end_char = end_char

    class MockDoc:
        def __init__(self):
            # Simulate a doc where the EntityRuler caught some terms
            self.ents = [
                MockEntity("M-1024", "MACHINE_ID", 10, 16),
                MockEntity("ERR-505", "ERROR_CODE", 20, 27)
            ]

    class MockNLP:
        def add_pipe(self, *args, **kwargs):
            mock_ruler = AsyncMock()
            mock_ruler.add_patterns = lambda x: None
            return mock_ruler
            
        def __call__(self, text):
            return MockDoc()

    mock_spacy_load.return_value = MockNLP()

    ner = IndustrialNER(confidence_threshold=0.7)
    
    text = "Machine M-1024 threw an ERR-505 warning."
    result = await ner.extract(text, {"doc_id": "doc1"})
    
    assert len(result.entities) == 2
    assert result.entities[0].text == "M-1024"
    assert result.entities[0].label == "MACHINE_ID"
    assert result.entities[1].text == "ERR-505"
    assert result.entities[1].label == "ERROR_CODE"


@pytest.mark.asyncio
async def test_llm_extractor_cache_hit():
    mock_redis = AsyncMock()
    # Simulate a cached JSON response
    cached_json = '{"relations": [{"source": "Motor", "relation_type": "CAUSED_BY", "target": "Bearing Wear", "confidence": 0.9, "sentence_span": "Motor failed due to bearing wear."}]}'
    mock_redis.get.return_value = cached_json
    
    # Do not set OPENAI_API_KEY, just use cache
    llm_extractor = LLMExtractor(mock_redis)
    # Patch the client check so it thinks it's initialized
    llm_extractor.client = True 
    
    text = "Motor failed due to bearing wear."
    result = await llm_extractor.extract(text, {"doc_id": "doc2"})
    
    # Assert
    assert len(result.relations) == 1
    rel = result.relations[0]
    assert rel.source == "Motor"
    assert rel.target == "Bearing Wear"
    assert rel.relation_type == "CAUSED_BY"
    
    # Redis should have been checked
    mock_redis.get.assert_called_once()


@pytest.mark.asyncio
async def test_relation_extractor_merging():
    mock_ner = AsyncMock()
    mock_llm = AsyncMock()
    
    # Mocks return empty results for simplicity, just testing the orchestration path
    from backend.services.knowledge_graph_service.src.infrastructure.extractors.base_extractor import ExtractionResult, IndustrialEntity, ExtractedRelation
    
    # NER finds Machine
    ner_result = ExtractionResult(
        entities=[IndustrialEntity(id="1", label="MACHINE_ID", text="M-001", confidence=0.9, start_char=0, end_char=5)]
    )
    mock_ner.extract.return_value = ner_result
    mock_ner.confidence_threshold = 0.7
    
    # LLM finds Relation
    llm_result = ExtractionResult(
        relations=[ExtractedRelation(source="M-001", target="Leak", relation_type="HAS_ISSUE", confidence=0.8, sentence_span="M-001 has leak")]
    )
    mock_llm.extract.return_value = llm_result
    
    orchestrator = RelationExtractor(mock_ner, mock_llm, relation_threshold=0.6)
    
    result = await orchestrator.extract("text", {"doc_id": "doc3", "doc_type": "maintenance_report"})
    
    # Assert merging
    assert len(result.entities) == 1
    assert result.entities[0].text == "M-001"
    
    assert len(result.relations) == 1
    assert result.relations[0].relation_type == "HAS_ISSUE"
