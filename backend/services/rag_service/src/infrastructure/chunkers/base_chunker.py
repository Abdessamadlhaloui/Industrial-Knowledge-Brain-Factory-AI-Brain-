import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List

import tiktoken


@dataclass
class Chunk:
    """Represents a standard text chunk with its position and tokens."""
    chunk_id: str
    text: str
    metadata: Dict[str, Any]
    start_idx: int
    end_idx: int
    token_count: int


@dataclass
class ChunkPair:
    """Represents a parent-child chunk relationship for hierarchical retrieval."""
    parent_id: str
    parent_text: str
    child_id: str
    child_text: str
    metadata: Dict[str, Any]


class BaseChunker(ABC):
    """
    Abstract base class for all chunking strategies.
    Provides deterministic ID generation and token counting utilities.
    """
    
    def __init__(self) -> None:
        # We use cl100k_base which is standard for OpenAI models like gpt-4
        self.encoding = tiktoken.get_encoding("cl100k_base")

    @abstractmethod
    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Any]:
        """
        Split text into chunks.
        
        Args:
            text (str): The raw text to chunk.
            metadata (Dict[str, Any]): Base metadata to attach to chunks.
            
        Returns:
            List[Any]: A list of Chunk or ChunkPair objects.
        """
        pass

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a string."""
        return len(self.encoding.encode(text))

    def generate_chunk_id(self, doc_id: str, index: int) -> str:
        """
        Generate a deterministic chunk ID using SHA256.
        Ensures idempotent ingestion and deduplication.
        """
        unique_string = f"{doc_id}_{index}"
        return hashlib.sha256(unique_string.encode("utf-8")).hexdigest()
