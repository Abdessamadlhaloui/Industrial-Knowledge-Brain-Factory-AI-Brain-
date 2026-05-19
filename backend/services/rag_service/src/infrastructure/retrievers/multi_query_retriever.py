import asyncio
import logging
from typing import Any, Dict, List

from backend.services.rag_service.src.infrastructure.retrievers.hybrid_retriever import HybridRetriever
from backend.services.rag_service.src.infrastructure.vector_stores.qdrant_store import ScoredPoint

logger = logging.getLogger(__name__)


class MultiQueryRetriever:
    """
    Multi-Query Expansion Retriever.
    Takes an original query, generates N=3 alternative phrasings (via an LLM), 
    and runs the HybridRetriever for each phrasing in parallel.
    Deduplicates results by chunk_id and boosts scores based on frequency of appearance.
    
    Industrial rationale: Great for vague queries or cross-lingual terms (e.g., "pump broken" vs 
    "centrifugal pump failure").
    """

    def __init__(self, hybrid_retriever: HybridRetriever, llm_client: Any = None):
        self.hybrid_retriever = hybrid_retriever
        self.llm_client = llm_client  # Mocked/Injected LLM client for expansion

    async def _generate_phrasings(self, original_query: str) -> List[str]:
        """Generate 3 alternative phrasings via LLM. Mocked here if no LLM provided."""
        if not self.llm_client:
            # Fallback mock for testing/development
            return [
                original_query,
                f"{original_query} troubleshooting maintenance",
                f"{original_query} error codes parts"
            ]
            
        # In production, call the LLM
        prompt = (
            f"You are an industrial AI assistant. Your user is searching for: '{original_query}'. "
            "Generate 3 different ways to phrase this search query, including specific technical synonyms. "
            "Return only the 3 queries separated by newlines."
        )
        # Mocking the async LLM call
        response = "mock 1\nmock 2\nmock 3"
        return [original_query] + [line.strip() for line in response.split("\n") if line.strip()]

    async def retrieve(
        self, 
        tenant_id: str, 
        query: str, 
        filters: Dict[str, Any], 
        top_k: int = 20
    ) -> List[ScoredPoint]:
        """
        Execute multi-query expansion and retrieval.
        """
        phrasings = await self._generate_phrasings(query)
        logger.info("Generated %d phrasings for query: '%s'", len(phrasings), query)
        
        # Run hybrid retrieval for each phrasing concurrently
        tasks = [
            self.hybrid_retriever.retrieve(tenant_id, phrasing, filters, top_k)
            for phrasing in phrasings
        ]
        
        results_list = await asyncio.gather(*tasks)
        
        # Deduplicate and score by frequency
        # Score = max(score) + (frequency * 0.1)
        unique_results: Dict[str, ScoredPoint] = {}
        frequencies: Dict[str, int] = {}
        
        for results in results_list:
            for res in results:
                frequencies[res.id] = frequencies.get(res.id, 0) + 1
                
                if res.id not in unique_results or res.score > unique_results[res.id].score:
                    unique_results[res.id] = res

        # Apply frequency boost
        final_list = []
        for doc_id, res in unique_results.items():
            freq = frequencies[doc_id]
            boosted_score = res.score + (freq * 0.1)
            
            final_list.append(ScoredPoint(
                id=res.id,
                score=boosted_score,
                payload=res.payload
            ))
            
        # Sort by boosted score
        final_list.sort(key=lambda x: x.score, reverse=True)
        return final_list[:top_k]
