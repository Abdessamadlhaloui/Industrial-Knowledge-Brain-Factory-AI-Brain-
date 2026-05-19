import logging
from typing import Any, Dict

from backend.services.knowledge_graph_service.src.infrastructure.extractors.base_extractor import (
    BaseExtractor, ExtractionResult, ExtractedRelation
)
from backend.services.knowledge_graph_service.src.infrastructure.extractors.industrial_ner import IndustrialNER
from backend.services.knowledge_graph_service.src.infrastructure.extractors.llm_extractor import LLMExtractor

logger = logging.getLogger(__name__)


class RelationExtractor(BaseExtractor):
    """
    Orchestrates NER and LLM extractors, merging their outputs and mapping 
    rule-based semantic statements if necessary.
    Enforces overall confidence thresholds.
    """

    def __init__(
        self, 
        ner_extractor: IndustrialNER, 
        llm_extractor: LLMExtractor,
        relation_threshold: float = 0.6
    ):
        self.ner_extractor = ner_extractor
        self.llm_extractor = llm_extractor
        self.relation_threshold = relation_threshold

    async def extract(self, text: str, doc_metadata: Dict[str, Any]) -> ExtractionResult:
        # Run NER
        ner_result = await self.ner_extractor.extract(text, doc_metadata)
        
        # Determine if we should run LLM based on doc_type
        doc_type = doc_metadata.get("doc_type", "")
        if doc_type in ["maintenance_report", "incident_report"]:
            llm_result = await self.llm_extractor.extract(text, doc_metadata)
        else:
            llm_result = ExtractionResult()

        # Merge entities, taking care of naive deduplication by text
        seen_entities = set()
        merged_entities = []
        
        for ent in ner_result.entities + llm_result.entities:
            # Simple deduplication by exact text match
            text_lower = ent.text.lower().strip()
            if text_lower not in seen_entities and ent.confidence >= self.ner_extractor.confidence_threshold:
                merged_entities.append(ent)
                seen_entities.add(text_lower)

        # Merge relations and apply confidence threshold
        merged_relations = []
        for rel in ner_result.relations + llm_result.relations:
            if rel.confidence >= self.relation_threshold:
                merged_relations.append(rel)

        return ExtractionResult(
            entities=merged_entities,
            relations=merged_relations,
            confidence=1.0
        )
