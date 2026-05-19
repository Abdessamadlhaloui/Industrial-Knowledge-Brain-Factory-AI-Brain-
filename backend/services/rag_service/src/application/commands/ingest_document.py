import json
import logging
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct
from redis.asyncio import Redis

from backend.shared.infrastructure.messaging.kafka_producer import KafkaMessageProducer
from backend.services.rag_service.src.infrastructure.chunkers.parent_child_chunker import ParentChildChunker
from backend.services.rag_service.src.infrastructure.chunkers.semantic_chunker import SemanticChunker
from backend.services.rag_service.src.infrastructure.chunkers.timeseries_chunker import TimeseriesChunker
from backend.services.rag_service.src.infrastructure.chunkers.recursive_chunker import RecursiveChunker

logger = logging.getLogger(__name__)


class IngestDocumentCommand(BaseModel):
    """Command to trigger document chunking and ingestion."""
    model_config = ConfigDict(frozen=True)

    doc_id: str
    tenant_id: str
    text: str
    source_type: str
    metadata: Dict[str, Any]
    chunking_strategy: str  # "parent_child", "semantic", "timeseries", "recursive"


class IngestDocumentHandler:
    """
    Orchestrates the RAG pipeline:
    1. Select chunker
    2. Chunk document
    3. Route parent chunks to Redis (48h TTL)
    4. Embed & Route child/regular chunks to Qdrant
    5. Upsert root metadata to Postgres (mocked here or handled via repository)
    6. Extract entities -> Kafka (ikb.graph.updates)
    7. Emit DocumentIndexed domain event -> Kafka
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        redis_client: Redis,
        kafka_producer: KafkaMessageProducer,
        db_pool: Any,  # asyncpg pool mock
        embedding_function: Any,  # Callable to embed texts
    ):
        self.qdrant_client = qdrant_client
        self.redis_client = redis_client
        self.kafka_producer = kafka_producer
        self.db_pool = db_pool
        self.embedding_function = embedding_function

        # Initialize chunkers
        self.chunkers = {
            "parent_child": ParentChildChunker(),
            "semantic": SemanticChunker(),
            "timeseries": TimeseriesChunker(),
            "recursive": RecursiveChunker(),
        }

    async def handle(self, cmd: IngestDocumentCommand) -> None:
        logger.info("Starting ingestion for doc_id=%s with strategy=%s", cmd.doc_id, cmd.chunking_strategy)
        
        chunker = self.chunkers.get(cmd.chunking_strategy)
        if not chunker:
            raise ValueError(f"Unknown chunking strategy: {cmd.chunking_strategy}")

        # 1 & 2. Chunk the document
        base_meta = cmd.metadata.copy()
        base_meta["doc_id"] = cmd.doc_id
        
        chunks = chunker.chunk(cmd.text, base_meta)
        if not chunks:
            logger.warning("No chunks generated for doc_id=%s", cmd.doc_id)
            return

        # 3 & 4. Route chunks to storage
        points_to_upsert = []
        
        if cmd.chunking_strategy == "parent_child":
            for pair in chunks:
                # Store parent in Redis (TTL 48h = 172800s)
                redis_key = f"rag:parent:{pair.parent_id}"
                await self.redis_client.setex(
                    redis_key, 
                    172800, 
                    json.dumps({"text": pair.parent_text, "metadata": pair.metadata})
                )
                
                # Embed child and prepare for Qdrant
                vector = await self.embedding_function(pair.child_text)
                points_to_upsert.append(
                    PointStruct(
                        id=pair.child_id,
                        vector=vector,
                        payload={"text": pair.child_text, "metadata": pair.metadata}
                    )
                )
        else:
            for chunk in chunks:
                vector = await self.embedding_function(chunk.text)
                points_to_upsert.append(
                    PointStruct(
                        id=chunk.chunk_id,
                        vector=vector,
                        payload={"text": chunk.text, "metadata": chunk.metadata}
                    )
                )
                
        # Upsert to Qdrant
        if points_to_upsert:
            await self.qdrant_client.upsert(
                collection_name="industrial_knowledge",
                points=points_to_upsert
            )
            logger.info("Upserted %d vectors to Qdrant for doc_id=%s", len(points_to_upsert), cmd.doc_id)

        # 5. Upsert metadata to PostgreSQL
        query = """
            INSERT INTO document_metadata (doc_id, source_type, metadata, chunk_count)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT (doc_id) DO UPDATE SET
                metadata = EXCLUDED.metadata,
                chunk_count = EXCLUDED.chunk_count,
                updated_at = NOW()
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute(query, cmd.doc_id, cmd.source_type, json.dumps(cmd.metadata), len(chunks))

        # 6. Send to Knowledge Graph extraction consumer
        try:
            payload = {
                "doc_id": cmd.doc_id,
                "text": cmd.text,
                "tenant_id": cmd.tenant_id,
                "doc_type": cmd.metadata.get("doc_type", "unknown")
            }
            await self.kafka_producer.send(
                topic="ikb.kg.extract",
                value=payload,
                key=cmd.doc_id
            )
            logger.info("Published to ikb.kg.extract for doc_id=%s", cmd.doc_id)
        except Exception as e:
            # We swallow this exception because the Knowledge Graph is an eventually-consistent 
            # downstream consumer. A Kafka publish failure here should not roll back the 
            # primary RAG Qdrant ingestion. Operators can replay the failure later.
            logger.error("Failed to publish to ikb.kg.extract for doc_id=%s: %s", cmd.doc_id, str(e))

        # 7. Emit DocumentIndexed event
        event_payload = {
            "event_type": "DocumentIndexed",
            "doc_id": cmd.doc_id,
            "chunking_strategy": cmd.chunking_strategy,
            "chunk_count": len(chunks)
        }
        await self.kafka_producer.send(
            topic="ikb.audit.log",
            value=event_payload,
            key=cmd.doc_id
        )
        
        logger.info("Successfully ingested doc_id=%s", cmd.doc_id)
