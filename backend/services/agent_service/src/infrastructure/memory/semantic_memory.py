import logging
import uuid
from typing import Any, Dict, List

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    Qdrant-backed long-term memory.
    Embeds and stores derived insights to allow agents to query historical findings across sessions.
    """

    def __init__(self, qdrant_client: AsyncQdrantClient, embedder: Any = None):
        self.qdrant_client = qdrant_client
        self.embedder = embedder
        self.collection_name = "agent_insights"

    async def initialize(self) -> None:
        """Create the insights collection if it doesn't exist."""
        exists = await self.qdrant_client.collection_exists(self.collection_name)
        if not exists:
            logger.info("Creating semantic memory collection '%s'", self.collection_name)
            await self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=3072,  # Defaulting to text-embedding-3-large dims
                    distance=models.Distance.COSINE
                )
            )

    async def store_insight(self, session_id: str, tenant_id: str, insight_text: str) -> None:
        """Embeds and stores an insight to Qdrant."""
        logger.info("Storing insight for session %s: %s", session_id, insight_text[:50])
        
        if not self.embedder:
            logger.warning("No embedder configured. Mocking embedding.")
            vector = [0.1] * 3072
        else:
            vectors = await self.embedder.embed([insight_text])
            vector = vectors[0] if vectors else [0.1] * 3072

        point_id = str(uuid.uuid4())
        
        await self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "session_id": session_id,
                        "tenant_id": tenant_id,
                        "text": insight_text
                    }
                )
            ]
        )

    async def retrieve_relevant(self, query: str, tenant_id: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant past insights."""
        if not self.embedder:
            query_vector = [0.1] * 3072
        else:
            vectors = await self.embedder.embed([query])
            query_vector = vectors[0] if vectors else [0.1] * 3072

        results = await self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="tenant_id",
                        match=models.MatchValue(value=tenant_id)
                    )
                ]
            ),
            limit=top_k
        )
        
        return [
            {
                "score": r.score,
                "text": r.payload.get("text", ""),
                "session_id": r.payload.get("session_id", "")
            }
            for r in results
        ]
