import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from pydantic import BaseModel, ConfigDict

from prometheus_client import Histogram, Counter

from backend.services.rag_service.src.infrastructure.retrievers.multi_query_retriever import MultiQueryRetriever
from backend.services.rag_service.src.infrastructure.retrievers.parent_retriever import ParentRetriever
from backend.services.rag_service.src.infrastructure.rerankers.cross_encoder_reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)

# Prometheus Metrics
RETRIEVAL_LATENCY = Histogram('retrieval_latency_ms', 'Latency of the full retrieval pipeline in milliseconds')
RERANK_LATENCY = Histogram('rerank_latency_ms', 'Latency of the reranking step in milliseconds')
EMPTY_RESULT_COUNT = Counter('empty_result_count', 'Number of queries that returned zero results')


class RetrieveContextQuery(BaseModel):
    """Query object for context retrieval."""
    model_config = ConfigDict(frozen=True)
    
    query: str
    tenant_id: str
    machine_ids: Optional[List[str]] = None
    top_k: int = 20
    rerank: bool = True
    filters: Dict[str, Any] = {}


@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
    source_doc: str


class RetrieveContextHandler:
    """
    CQRS Handler orchestrating the complete RAG retrieval pipeline:
    1. Multi-Query Expansion & Hybrid Retrieval (BM25 + Dense Qdrant + RRF)
    2. Parent Context Expansion (Redis)
    3. Cross-Encoder Reranking
    4. Formatting & Metrics Logging
    
    Hard latency budget: < 500ms p95.
    """

    def __init__(
        self,
        multi_query_retriever: MultiQueryRetriever,
        parent_retriever: ParentRetriever,
        reranker: CrossEncoderReranker
    ):
        self.multi_query_retriever = multi_query_retriever
        self.parent_retriever = parent_retriever
        self.reranker = reranker

    async def handle(self, query: RetrieveContextQuery) -> List[RetrievalResult]:
        start_time = time.time()
        
        # Build strict filters
        filters = query.filters.copy()
        if query.machine_ids:
            # Note: For production Qdrant, IN condition might be required. 
            # Simplified here to just check the first if single, or handle externally.
            if len(query.machine_ids) == 1:
                filters["machine_id"] = query.machine_ids[0]
            # else implement OR condition in QdrantStore
                
        # 1. Multi-Query Hybrid Retrieval
        retrieved_points = await self.multi_query_retriever.retrieve(
            tenant_id=query.tenant_id,
            query=query.query,
            filters=filters,
            top_k=query.top_k * 2 if query.rerank else query.top_k
        )
        
        if not retrieved_points:
            EMPTY_RESULT_COUNT.inc()
            logger.warning("No results found for query: '%s'", query.query)
            return []

        # 2. Parent Expansion
        expanded_points = await self.parent_retriever.expand_to_parents(retrieved_points)

        final_results = []
        
        # 3. Reranking
        if query.rerank:
            rerank_start = time.time()
            documents = [p.payload.get("text", "") for p in expanded_points]
            
            reranked_tuples = await self.reranker.rerank(
                query=query.query,
                documents=documents,
                top_k=query.top_k,
                min_score=0.1
            )
            
            rerank_latency = (time.time() - rerank_start) * 1000
            RERANK_LATENCY.observe(rerank_latency)
            
            for idx, score in reranked_tuples:
                point = expanded_points[idx]
                final_results.append(RetrievalResult(
                    chunk_id=point.id,
                    text=point.payload.get("text", ""),
                    score=score,
                    metadata=point.payload,
                    source_doc=point.payload.get("doc_id", "unknown")
                ))
        else:
            # No reranking, format directly
            for point in expanded_points[:query.top_k]:
                final_results.append(RetrievalResult(
                    chunk_id=point.id,
                    text=point.payload.get("text", ""),
                    score=point.score,
                    metadata=point.payload,
                    source_doc=point.payload.get("doc_id", "unknown")
                ))

        # Check Latency Budget
        total_latency = (time.time() - start_time) * 1000
        RETRIEVAL_LATENCY.observe(total_latency)
        
        if total_latency > 500:
            logger.warning("Latency budget exceeded! Total time: %.2f ms", total_latency)
        else:
            logger.info("Retrieval completed in %.2f ms", total_latency)

        return final_results
