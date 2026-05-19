import re
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.services.rag_service.src.infrastructure.chunkers.base_chunker import BaseChunker, Chunk


class SemanticChunker(BaseChunker):
    """
    Semantic Chunker for intelligent boundary detection.
    
    Uses local CPU-efficient sentence embeddings to measure cosine similarity between
    adjacent sentences. It splits the document when the similarity drops below a given threshold,
    indicating a shift in topic.
    
    Crucially for industrial documents, it utilizes regex to detect and strictly preserve 
    header boundaries (e.g., "3.2.1 MAINTENANCE STEPS" or ALL CAPS headers).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", threshold: float = 0.3) -> None:
        super().__init__()
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        
        # Regex for common industrial headers (e.g., "1.2 Scope", "WARNING:", or all caps lines)
        self.header_pattern = re.compile(r"^(?:\d+(?:\.\d+)*\s+[A-Z]|WARNING:|CAUTION:|NOTE:|[A-Z][A-Z\s]+$)", re.MULTILINE)

    def _split_into_sentences(self, text: str) -> List[str]:
        """Naively split text into sentences while respecting newlines."""
        # Split by period-space or newlines
        raw_sentences = re.split(r"(?<=[.!?]) +|\n+", text)
        return [s.strip() for s in raw_sentences if s.strip()]

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            return 0.0
        return float(dot / norm)

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """
        Splits text based on semantic topic shifts and strict headers.
        """
        doc_id = metadata.get("doc_id", "unknown_doc")
        chunks: List[Chunk] = []
        
        sentences = self._split_into_sentences(text)
        if not sentences:
            return chunks

        # Embed all sentences
        embeddings = self.model.encode(sentences)
        
        current_chunk_sentences = [sentences[0]]
        chunk_idx = 0
        start_idx = 0
        
        # Find sentence boundaries in original text for start/end index tracking
        # For simplicity in this implementation, we approximate the indices.
        
        for i in range(1, len(sentences)):
            prev_emb = embeddings[i - 1]
            curr_emb = embeddings[i]
            
            similarity = self._cosine_similarity(prev_emb, curr_emb)
            sentence = sentences[i]
            
            # Check if this sentence looks like a header
            is_header = bool(self.header_pattern.match(sentence))
            
            # Split if similarity drops below threshold OR we hit a hard header boundary
            if similarity < self.threshold or is_header:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk_sentences)
                chunk_meta = metadata.copy()
                chunk_meta["chunk_strategy"] = "semantic"
                
                chunks.append(
                    Chunk(
                        chunk_id=self.generate_chunk_id(doc_id, chunk_idx),
                        text=chunk_text,
                        metadata=chunk_meta,
                        start_idx=start_idx,
                        end_idx=start_idx + len(chunk_text),
                        token_count=self.count_tokens(chunk_text)
                    )
                )
                
                # Start new chunk
                chunk_idx += 1
                start_idx += len(chunk_text) + 1  # approximate space
                current_chunk_sentences = [sentence]
            else:
                current_chunk_sentences.append(sentence)
                
        # Append the final chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            chunk_meta = metadata.copy()
            chunk_meta["chunk_strategy"] = "semantic"
            chunks.append(
                Chunk(
                    chunk_id=self.generate_chunk_id(doc_id, chunk_idx),
                    text=chunk_text,
                    metadata=chunk_meta,
                    start_idx=start_idx,
                    end_idx=start_idx + len(chunk_text),
                    token_count=self.count_tokens(chunk_text)
                )
            )

        return chunks
