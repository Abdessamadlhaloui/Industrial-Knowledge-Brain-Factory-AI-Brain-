from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


class SemanticMemory:
    """
    Qdrant-backed long-term memory.

    Embeds and stores derived insights to allow agents to query historical
    findings across sessions.

    Args:
        qdrant_client: Async Qdrant client pointed at the target cluster.
        embedder:      Required embedding service (e.g. OpenAIEmbedder).
                       Must implement ``async embed(text: str) -> List[float]``.
                       Passing ``None`` raises ``ValueError`` at construction
                       time — silent mock vectors are intentionally not supported.
        collection_name: Override the default Qdrant collection name.

    Raises:
        ValueError: If ``embedder`` is ``None``.
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        embedder: Any,
        collection_name: str = "agent_insights",
    ) -> None:
        if embedder is None:
            raise ValueError(
                "SemanticMemory requires an embedder. "
                "Configure OpenAIEmbedder or a compatible implementation that "
                "exposes `async embed(text: str) -> List[float]`."
            )

        self.qdrant_client: AsyncQdrantClient = qdrant_client
        self.embedder: Any = embedder
        self.collection_name: str = collection_name

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    async def initialize(self) -> None:
        """Create the insights collection if it does not already exist."""
        exists: bool = await self.qdrant_client.collection_exists(self.collection_name)
        if not exists:
            logger.info(
                "Creating semantic memory collection '%s'", self.collection_name
            )
            await self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=3072,  # text-embedding-3-large output dimensions
                    distance=models.Distance.COSINE,
                ),
            )

    # -------------------------------------------------------------------------
    # Write
    # -------------------------------------------------------------------------

    async def store_insight(self, insight: "Insight") -> None:
        """Embed and persist a single insight to Qdrant.

        Args:
            insight: Domain model with ``.content``, ``.session_id``,
                     ``.tenant_id``, and ``.created_at`` attributes.

        Raises:
            Exception: Re-raises any embedding or Qdrant failure after logging.
        """
        try:
            vector: list[float] = await self.embedder.embed(insight.content)

            point = models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "content": insight.content,
                    "session_id": insight.session_id,
                    "tenant_id": insight.tenant_id,
                    "created_at": insight.created_at.isoformat(),
                },
            )

            await self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=[point],
            )

            logger.info(
                "Stored insight for session_id=%s preview='%.60s'",
                insight.session_id,
                insight.content,
            )

        except Exception:
            logger.error(
                "Failed to store insight for session_id=%s preview='%.60s'",
                insight.session_id,
                insight.content,
                exc_info=True,
            )
            raise

    # -------------------------------------------------------------------------
    # Read
    # -------------------------------------------------------------------------

    async def retrieve_relevant(
        self,
        query: str,
        tenant_id: str | None = None,
        top_k: int = 5,
    ) -> list[str]:
        """Embed ``query`` and return the most similar insight content strings.

        Args:
            query:     Natural-language query to embed and search against.
            tenant_id: When provided, restricts results to a single tenant via
                       a Qdrant ``FieldCondition`` filter.  Pass ``None`` to
                       search across all tenants (admin / cross-tenant use).
            top_k:     Maximum number of results to return.

        Returns:
            Ordered list of insight content strings, most similar first.

        Raises:
            Exception: Re-raises any embedding or Qdrant failure after logging.
        """
        try:
            query_vector: list[float] = await self.embedder.embed(query)

            # Build optional tenant filter — omitting it enables cross-tenant
            # queries for admin tooling without a separate code path.
            query_filter: models.Filter | None = None
            if tenant_id is not None:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="tenant_id",
                            match=models.MatchValue(value=tenant_id),
                        )
                    ]
                )

            results = await self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=query_filter,
                limit=top_k,
            )

            contents: list[str] = [
                r.payload.get("content", "") for r in results
            ]

            logger.debug(
                "retrieve_relevant: query='%.60s' tenant_id=%s top_k=%d hits=%d",
                query,
                tenant_id,
                top_k,
                len(contents),
            )

            return contents

        except Exception:
            logger.error(
                "Failed to retrieve insights for query='%.60s' tenant_id=%s",
                query,
                tenant_id,
                exc_info=True,
            )
            raise