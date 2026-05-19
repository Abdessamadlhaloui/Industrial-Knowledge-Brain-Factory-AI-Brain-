import hashlib
import json
import logging
from typing import Any

from asyncpg.pool import Pool

from backend.services.ingestion_service.src.domain.models.ingestion_job import IngestionJob, JobSourceType, JobStatus
from backend.services.ingestion_service.src.infrastructure.parsers.pdf_parser import PDFParser
from backend.services.ingestion_service.src.infrastructure.storage.minio_client import MinIOClient
from backend.shared.infrastructure.messaging.kafka_producer import KafkaMessageProducer

logger = logging.getLogger(__name__)


class PDFIngestionHandler:
    """
    Orchestrates the ingestion of PDF documents.
    Downloads from MinIO, parses, extracts metadata, and emits to Kafka.
    Tracks job status in PostgreSQL.
    """

    def __init__(
        self,
        db_pool: Pool,
        minio_client: MinIOClient,
        kafka_producer: KafkaMessageProducer,
    ):
        self.db_pool = db_pool
        self.minio_client = minio_client
        self.kafka_producer = kafka_producer

    async def handle(
        self, 
        job_id: str, 
        bucket: str, 
        key: str, 
        tenant_id: str, 
        factory_id: str
    ) -> None:
        """
        Handle a PDF ingestion request.
        
        Args:
            job_id: External or generated job UUID.
            bucket: MinIO bucket name (e.g., 'raw-documents').
            key: MinIO object key.
            tenant_id: Tenant ID.
            factory_id: Factory ID.
        """
        job = IngestionJob(
            id=job_id,
            source_type=JobSourceType.PDF,
            source_url=f"s3://{bucket}/{key}",
            tenant_id=tenant_id,
            factory_id=factory_id,
        )
        
        # 1. Register Job as Pending (or get existing to check dedup)
        if not await self._register_or_check_job(job):
            logger.info("Job %s for %s is already processed or processing (Idempotent skip).", job.id, key)
            return

        try:
            # 2. Mark Processing
            job.mark_processing()
            await self._update_job_status(job)

            # 3. Download from MinIO
            logger.debug("Downloading %s from %s", key, bucket)
            file_bytes = await self.minio_client.get_file(bucket, key)

            # 4. Hash-based Deduplication Check
            file_hash = hashlib.sha256(file_bytes).hexdigest()
            if await self._is_duplicate_hash(file_hash, tenant_id):
                logger.info("File %s with hash %s has already been ingested. Skipping.", key, file_hash)
                job.mark_done()
                await self._update_job_status(job)
                return

            # 5. Parse PDF
            logger.debug("Parsing PDF %s", key)
            pages = PDFParser.parse(file_bytes)

            # 6. Emit to Kafka
            payload = {
                "job_id": str(job.id),
                "source_url": job.source_url,
                "tenant_id": job.tenant_id,
                "factory_id": job.factory_id,
                "file_hash": file_hash,
                "pages": [p.model_dump() for p in pages],
                "total_pages": len(pages),
            }

            await self.kafka_producer.send(
                topic="ikb.documents.ingestion",
                value=payload,
                key=str(job.id)
            )
            
            # Record the hash to prevent future duplicates
            await self._record_hash(file_hash, tenant_id, str(job.id))

            # 7. Mark Done
            job.mark_done()
            await self._update_job_status(job)
            logger.info("Successfully ingested PDF %s (Job: %s)", key, job.id)

        except Exception as e:
            logger.error("Failed to ingest PDF %s (Job: %s): %s", key, job.id, e)
            job.mark_failed(error_message=str(e))
            await self._update_job_status(job)
            
            # Route to dead-letter queue
            error_payload = {
                "job_id": str(job.id),
                "source_url": job.source_url,
                "tenant_id": job.tenant_id,
                "factory_id": job.factory_id,
                "error": str(e)
            }
            try:
                await self.kafka_producer.send(
                    topic="ikb.documents.failed",
                    value=error_payload,
                    key=str(job.id)
                )
            except Exception as kafka_err:
                logger.error("Failed to publish to dead-letter queue: %s", kafka_err)

    async def _register_or_check_job(self, job: IngestionJob) -> bool:
        """
        Insert the job into the DB if it doesn't exist.
        Returns True if we should proceed, False if it already exists and is done/processing.
        """
        query = """
            INSERT INTO ingestion_jobs (id, source_type, status, source_url, tenant_id, factory_id, created_at, updated_at)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (id) DO NOTHING
            RETURNING id
        """
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                query, 
                job.id, 
                job.source_type.value, 
                job.status.value, 
                job.source_url, 
                job.tenant_id, 
                job.factory_id, 
                job.created_at, 
                job.updated_at
            )
            
            if row is None:
                # Job already existed, check its status
                existing_status = await conn.fetchval("SELECT status FROM ingestion_jobs WHERE id = $1", job.id)
                if existing_status in (JobStatus.DONE.value, JobStatus.PROCESSING.value):
                    return False  # Skip
        return True

    async def _update_job_status(self, job: IngestionJob) -> None:
        """Update job status in DB."""
        query = """
            UPDATE ingestion_jobs 
            SET status = $1, updated_at = $2, completed_at = $3, error_message = $4
            WHERE id = $5
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute(
                query, 
                job.status.value, 
                job.updated_at, 
                job.completed_at, 
                job.error_message, 
                job.id
            )

    async def _is_duplicate_hash(self, file_hash: str, tenant_id: str) -> bool:
        """Check if this file hash was already processed for this tenant."""
        query = "SELECT 1 FROM ingested_document_hashes WHERE file_hash = $1 AND tenant_id = $2"
        async with self.db_pool.acquire() as conn:
            row = await conn.fetchrow(query, file_hash, tenant_id)
            return row is not None

    async def _record_hash(self, file_hash: str, tenant_id: str, job_id: str) -> None:
        """Record a successfully processed file hash."""
        query = """
            INSERT INTO ingested_document_hashes (file_hash, tenant_id, job_id, created_at)
            VALUES ($1, $2, $3, NOW())
            ON CONFLICT DO NOTHING
        """
        async with self.db_pool.acquire() as conn:
            await conn.execute(query, file_hash, tenant_id, job_id)
