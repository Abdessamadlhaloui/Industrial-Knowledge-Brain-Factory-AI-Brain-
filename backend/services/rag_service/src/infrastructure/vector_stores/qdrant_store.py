import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


@dataclass
class ScoredPoint:
    id: str
    score: float
    payload: Dict[str, Any]


class QdrantStore:
    """
    Async Qdrant vector store management.
    Handles per-tenant collections, named vectors (dense/sparse), and HNSW configuration.
    """

    def __init__(self, host: str = "localhost", port: int = 6333, dense_dim: int = 3072):
        self.client = AsyncQdrantClient(host=host, port=port)
        self.dense_dim = dense_dim

    def _collection_name(self, tenant_id: str) -> str:
        return f"ikb_{tenant_id}"

    async def initialize_tenant(self, tenant_id: str) -> None:
        """Create a collection for a tenant if it doesn't exist, using HNSW m=16, ef_construct=200."""
        col_name = self._collection_name(tenant_id)
        
        exists = await self.client.collection_exists(col_name)
        if not exists:
            logger.info("Creating Qdrant collection for tenant %s", tenant_id)
            await self.client.create_collection(
                collection_name=col_name,
                vectors_config={
                    "dense": models.VectorParams(
                        size=self.dense_dim,
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": models.SparseVectorParams()
                },
                hnsw_config=models.HnswConfigDiff(
                    m=16,
                    ef_construct=200
                )
            )

    async def upsert(
        self, 
        tenant_id: str, 
        chunk_id: str, 
        doc_id: str, 
        text: str,
        dense_vector: List[float],
        sparse_vector: Optional[Dict[int, float]] = None,
        machine_id: Optional[str] = None,
        doc_type: Optional[str] = None,
        timestamp: Optional[float] = None,
        parent_id: Optional[str] = None
    ) -> None:
        """Upsert a single chunk into Qdrant."""
        col_name = self._collection_name(tenant_id)
        
        payload = {
            "chunk_id": chunk_id,
            "doc_id": doc_id,
            "text": text,
            "machine_id": machine_id,
            "doc_type": doc_type,
            "timestamp": timestamp,
            "parent_id": parent_id
        }
        
        vectors = {"dense": dense_vector}
        if sparse_vector:
            vectors["sparse"] = models.SparseVector(
                indices=list(sparse_vector.keys()),
                values=list(sparse_vector.values())
            )

        point = models.PointStruct(
            id=chunk_id,
            vector=vectors,
            payload={k: v for k, v in payload.items() if v is not None}
        )
        
        await self.client.upsert(
            collection_name=col_name,
            points=[point]
        )

    async def search(
        self, 
        tenant_id: str, 
        query_vector: List[float], 
        filters: Dict[str, Any], 
        limit: int = 10, 
        score_threshold: float = 0.7,
        vector_name: str = "dense"
    ) -> List[ScoredPoint]:
        """
        Search using dense vectors.
        Constructs Qdrant FieldConditions based on the provided filters dictionary.
        """
        col_name = self._collection_name(tenant_id)
        
        qdrant_filters = []
        for key, value in filters.items():
            if value is not None:
                qdrant_filters.append(
                    models.FieldCondition(
                        key=key,
                        match=models.MatchValue(value=value)
                    )
                )
                
        filter_obj = models.Filter(must=qdrant_filters) if qdrant_filters else None

        results = await self.client.search(
            collection_name=col_name,
            query_vector=(vector_name, query_vector),
            query_filter=filter_obj,
            limit=limit,
            score_threshold=score_threshold
        )
        
        return [
            ScoredPoint(id=str(r.id), score=r.score, payload=r.payload or {})
            for r in results
        ]
