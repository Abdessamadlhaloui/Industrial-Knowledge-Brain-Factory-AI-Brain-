import logging
import uuid
from typing import Any, Dict, Optional

import spacy
from spacy.pipeline import EntityRuler

from backend.services.knowledge_graph_service.src.infrastructure.extractors.base_extractor import (
    BaseExtractor, ExtractionResult, IndustrialEntity
)

logger = logging.getLogger(__name__)


class IndustrialNER(BaseExtractor):
    """
    Custom SpaCy pipeline designed for fast, deterministic extraction of
    Industrial metrics, part numbers, and machine IDs.
    """

    _nlp_cache: Optional[spacy.language.Language] = None

    def __init__(self, confidence_threshold: float = 0.7):
        self.confidence_threshold = confidence_threshold
        self.nlp = self._load_model()

    @classmethod
    def _load_model(cls) -> spacy.language.Language:
        if cls._nlp_cache:
            return cls._nlp_cache

        try:
            logger.info("Loading SpaCy en_core_web_lg model...")
            nlp = spacy.load("en_core_web_lg")
        except OSError:
            logger.warning("en_core_web_lg not found. Attempting to use en_core_web_sm or blank fallback.")
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                nlp = spacy.blank("en")

        # Add custom industrial entity ruler
        ruler = nlp.add_pipe("entity_ruler", before="ner" if "ner" in nlp.pipe_names else None)
        
        patterns = [
            # Machine IDs: M-1234, CNC-5, CONV-A12
            {"label": "MACHINE_ID", "pattern": [{"TEXT": {"REGEX": r"^M-\d{4}$"}}]},
            {"label": "MACHINE_ID", "pattern": [{"TEXT": {"REGEX": r"^CNC-\d+$"}}]},
            {"label": "MACHINE_ID", "pattern": [{"TEXT": {"REGEX": r"^CONV-[A-Z]\d+$"}}]},
            
            # Part Numbers: SKF codes (e.g., SKF-12345), standard alphanumeric
            {"label": "PART_NUMBER", "pattern": [{"TEXT": {"REGEX": r"^SKF-\d+$"}}]},
            {"label": "PART_NUMBER", "pattern": [{"TEXT": {"REGEX": r"^[A-Z]{2,4}-\d{4,}$"}}]},
            
            # Error Codes: ERR-123, FAULT 5, E1234
            {"label": "ERROR_CODE", "pattern": [{"TEXT": {"REGEX": r"^ERR-\d+$"}}]},
            {"label": "ERROR_CODE", "pattern": [{"LOWER": "fault"}, {"IS_DIGIT": True}]},
            {"label": "ERROR_CODE", "pattern": [{"TEXT": {"REGEX": r"^E\d{4}$"}}]},
            
            # Metrics (Regex handles token combining loosely, but SpaCy patterns prefer token-by-token)
            {"label": "TEMPERATURE", "pattern": [{"LIKE_NUM": True}, {"TEXT": {"IN": ["°C", "°F", "C", "F", "K"]}}]},
            {"label": "PRESSURE", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["bar", "psi", "kpa", "mpa"]}}]},
            {"label": "VIBRATION", "pattern": [{"LIKE_NUM": True}, {"LOWER": {"IN": ["mm/s", "g"]}}]}
        ]
        ruler.add_patterns(patterns)
        
        cls._nlp_cache = nlp
        return nlp

    async def extract(self, text: str, doc_metadata: Dict[str, Any]) -> ExtractionResult:
        """
        Run the SpaCy pipeline to extract industrial entities.
        As a deterministic pipeline, it runs synchronously but we wrap it in an async interface.
        """
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Rule-based matches from the ruler have effectively 1.0 confidence.
            # Statistical NER matches from 'ner' vary, but we default to 1.0 for ruler matches.
            confidence = 1.0
            
            # Only keep specified industrial labels or highly relevant default ones (like ORG/PRODUCT)
            if ent.label_ in ["MACHINE_ID", "PART_NUMBER", "ERROR_CODE", "TEMPERATURE", "PRESSURE", "VIBRATION", "PRODUCT"]:
                
                # Filter by threshold
                if confidence >= self.confidence_threshold:
                    entities.append(
                        IndustrialEntity(
                            id=str(uuid.uuid4()),
                            label=ent.label_,
                            text=ent.text,
                            confidence=confidence,
                            start_char=ent.start_char,
                            end_char=ent.end_char
                        )
                    )
                    
        return ExtractionResult(entities=entities, confidence=1.0)
