import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.services.rag_service.src.application.queries.retrieve_context import RetrieveContextHandler, RetrieveContextQuery
from backend.services.rag_service.src.infrastructure.retrievers.multi_query_retriever import MultiQueryRetriever
from backend.services.rag_service.src.infrastructure.retrievers.hybrid_retriever import HybridRetriever
from backend.services.rag_service.src.infrastructure.retrievers.parent_retriever import ParentRetriever
from backend.services.rag_service.src.infrastructure.rerankers.cross_encoder_reranker import CrossEncoderReranker
from backend.services.rag_service.src.infrastructure.vector_stores.qdrant_store import ScoredPoint


@pytest.mark.asyncio
async def test_full_retrieval_stack():
    # Setup Mocks
    mock_qdrant = AsyncMock()
    mock_embedder = AsyncMock()
    mock_redis = AsyncMock()
    
    # 1. Mock Hybrid Retriever (returns fake BM25/Dense results)
    mock_embedder.embed.return_value = [[0.1] * 3072]
    mock_qdrant.search.return_value = [
        ScoredPoint(id="chunk_1", score=0.9, payload={"text": "Compressor XYZ maintenance", "doc_id": "doc1", "parent_id": "parent_1"}),
        ScoredPoint(id="chunk_2", score=0.8, payload={"text": "Valve ABC replacement", "doc_id": "doc2"}),
    ]
    
    # Empty BM25 mock corpus
    hybrid = HybridRetriever(mock_qdrant, mock_embedder, ["Compressor XYZ maintenance", "Valve ABC replacement"], [{"chunk_id": "chunk_1"}, {"chunk_id": "chunk_2"}])
    
    # 2. Multi-Query Retriever
    multi_query = MultiQueryRetriever(hybrid, llm_client=None)  # Uses mock LLM responses internally
    
    # 3. Parent Retriever (Mocks Redis)
    parent_retriever = ParentRetriever(mock_redis)
    # Redis will return a parent text for parent_1
    mock_redis.get.return_value = '{"text": "PARENT: Compressor XYZ maintenance requires specialized tools. Step 1...", "metadata": {}}'
    
    # 4. Reranker
    mock_reranker = AsyncMock(spec=CrossEncoderReranker)
    # Reranker returns indices mapped to scores. Document 0 gets 0.95, Document 1 gets 0.85
    mock_reranker.rerank.return_value = [(0, 0.95), (1, 0.85)]
    
    # 5. Handler
    handler = RetrieveContextHandler(multi_query, parent_retriever, mock_reranker)
    
    query = RetrieveContextQuery(
        query="compressor maintenance",
        tenant_id="tenant_123",
        top_k=2
    )
    
    # Execute
    results = await handler.handle(query)
    
    # Verify
    assert len(results) == 2
    
    # Top result should be chunk_1 which was expanded to parent_1
    top_result = results[0]
    assert top_result.chunk_id == "chunk_1"
    assert "PARENT: Compressor XYZ" in top_result.text
    assert top_result.score == 0.95  # Reranker score
    assert top_result.source_doc == "doc1"
    
    # Redis should have been called for parent_1
    mock_redis.get.assert_called_with("rag:parent:parent_1")
    
    # Qdrant search should have been called (via multi query -> hybrid)
    assert mock_qdrant.search.call_count > 0
    
    # Top-1 recall logic metric check
    assert results[0].source_doc == "doc1", "Recall failure: Expected doc1 for compressor maintenance"
