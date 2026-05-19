from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class IndustrialEntity:
    id: str
    label: str  # e.g., MACHINE_ID, PART_NUMBER, ERROR_CODE, TEMPERATURE
    text: str
    confidence: float
    start_char: int
    end_char: int


@dataclass
class ExtractedRelation:
    source: str
    relation_type: str  # e.g., CAUSED_BY, RESOLVED_BY, REQUIRES_PART, INDICATES
    target: str
    confidence: float
    sentence_span: str


@dataclass
class ExtractionResult:
    entities: List[IndustrialEntity] = field(default_factory=list)
    relations: List[ExtractedRelation] = field(default_factory=list)
    confidence: float = 0.0


class BaseExtractor(ABC):
    """
    Abstract base class for all entity and relation extractors.
    """

    @abstractmethod
    async def extract(self, text: str, doc_metadata: Dict[str, Any]) -> ExtractionResult:
        """
        Extract entities and relations from text.
        
        Args:
            text (str): The raw text document.
            doc_metadata (Dict[str, Any]): Associated metadata (e.g., doc_id, tenant_id).
            
        Returns:
            ExtractionResult: The structured extracted knowledge.
        """
        pass
