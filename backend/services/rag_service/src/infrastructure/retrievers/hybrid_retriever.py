import asyncio
import logging
from typing import Any, Dict, List

from rank_bm25 import BM25Okapi

from backend.services.rag_service.src.infrastructure.embedders.openai_embedder import OpenAIEmbedder
from backend.services.rag_service.src.infrastructure.vector_stores.qdrant_store import QdrantStore, ScoredPoint

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Core Hybrid Retriever executing parallel BM25 and Dense Vector search.
    Merges results using Reciprocal Rank Fusion (RRF).
    
    Industrial rationale: BM25 excels at exact keyword matches (e.g., part numbers "XYZ-1000" or error codes "E-404"),
    whereas dense vectors excel at semantic similarity ("how do I fix the pump").
    """

    def __init__(
        self, 
        qdrant_store: QdrantStore, 
        embedder: OpenAIEmbedder,
        corpus_texts: List[str],  # For in-memory BM25
        corpus_payloads: List[Dict[str, Any]]  # Metadata for BM25 docs
    ):
        self.qdrant_store = qdrant_store
        self.embedder = embedder
        
        # Initialize BM25 (Note: In production with huge corpora, this would ideally use Qdrant's sparse vectors
        # or a dedicated sparse engine like Elasticsearch. For this implementation, we use rank_bm25 in memory).
        self.corpus_texts = corpus_texts
        self.corpus_payloads = corpus_payloads
        tokenized_corpus = [doc.lower().split(" ") for doc in corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    async def retrieve(
        self, 
        tenant_id: str, 
        query: str, 
        filters: Dict[str, Any], 
        top_k: int = 20, 
        alpha: float = 0.5,
        rrf_k: int = 60
    ) -> List[ScoredPoint]:
        """
        Execute parallel hybrid search and fuse with RRF.
        
        Args:
            tenant_id (str): Tenant identifier.
            query (str): The search query.
            filters (Dict[str, Any]): Metadata filters (machine_id, doc_type, etc).
            top_k (int): Number of results to return.
            alpha (float): Weighting (0.0 = pure sparse, 1.0 = pure dense).
            rrf_k (int): RRF constant (default 60 is standard).
            
        Returns:
            List[ScoredPoint]: Fused and sorted results.
        """
        # Run BM25 and Dense search concurrently
        dense_task = asyncio.create_task(self._dense_search(tenant_id, query, filters, top_k * 2))
        sparse_task = asyncio.create_task(self._sparse_search(query, filters, top_k * 2))
        
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
        
        if alpha == 1.0:
            return dense_results[:top_k]
        if alpha == 0.0:
            return sparse_results[:top_k]
            
        # Reciprocal Rank Fusion (RRF)
        fused_scores: Dict[str, float] = {}
        payloads: Dict[str, Dict[str, Any]] = {}
        
        # Process Dense Ranks
        for rank, res in enumerate(dense_results):
            score = alpha * (1.0 / (rrf_k + rank + 1))
            fused_scores[res.id] = fused_scores.get(res.id, 0.0) + score
            payloads[res.id] = res.payload
            
        # Process Sparse Ranks
        for rank, res in enumerate(sparse_results):
            score = (1.0 - alpha) * (1.0 / (rrf_k + rank + 1))
            fused_scores[res.id] = fused_scores.get(res.id, 0.0) + score
            payloads[res.id] = res.payload
            
        # Sort by fused score
        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        final_results = [
            ScoredPoint(id=doc_id, score=score, payload=payloads[doc_id])
            for doc_id, score in sorted_results[:top_k]
        ]
        
        return final_results

    async def _dense_search(self, tenant_id: str, query: str, filters: Dict[str, Any], limit: int) -> List[ScoredPoint]:
        vectors = await self.embedder.embed([query])
        if not vectors:
            return []
            
        query_vector = vectors[0]
        return await self.qdrant_store.search(
            tenant_id=tenant_id,
            query_vector=query_vector,
            filters=filters,
            limit=limit,
            score_threshold=0.0  # Let RRF handle thresholds
        )

    async def _sparse_search(self, query: str, filters: Dict[str, Any], limit: int) -> List[ScoredPoint]:
        """In-memory BM25 search with post-filtering."""
        tokenized_query = query.lower().split(" ")
        
        # Prevent blocking
        loop = asyncio.get_running_loop()
        doc_scores = await loop.run_in_executor(None, self.bm25.get_scores, tokenized_query)
        
        # Pair with indices
        scored_docs = [(idx, float(score)) for idx, score in enumerate(doc_scores) if score > 0]
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scored_docs:
            payload = self.corpus_payloads[idx]
            
            # Apply filters
            match = True
            for k, v in filters.items():
                if v is not None and payload.get(k) != v:
                    match = False
                    break
                    
            if match:
                # Assuming payload has 'chunk_id'
                chunk_id = payload.get("chunk_id", f"bm25_mock_{idx}")
                results.append(ScoredPoint(id=chunk_id, score=score, payload=payload))
                
            if len(results) >= limit:
                break
                
        return results
