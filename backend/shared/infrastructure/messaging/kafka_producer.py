from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

import orjson
from aiokafka import AIOKafkaProducer
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class KafkaProducerConfig(BaseModel):
    bootstrap_servers: str = "localhost:9092"
    client_id: str = "ikb-producer"
    acks: str = "all"
    max_batch_size: int = 16384
    linger_ms: int = 10
    compression_type: str = "snappy"
    retries: int = 5
    retry_backoff_ms: int = 100


class KafkaMessageProducer:
    """Async Kafka producer with automatic serialization and partition key routing.

    Partition strategy: uses machine_id as the partition key for all sensor topics
    to guarantee per-machine ordering.
    """

    def __init__(self, config: KafkaProducerConfig | None = None) -> None:
        self._config = config or KafkaProducerConfig()
        self._producer: AIOKafkaProducer | None = None
        self._started = False

    async def start(self) -> None:
        if self._started:
            return
        self._producer = AIOKafkaProducer(
            bootstrap_servers=self._config.bootstrap_servers,
            client_id=self._config.client_id,
            acks=self._config.acks,
            max_batch_size=self._config.max_batch_size,
            linger_ms=self._config.linger_ms,
            compression_type=self._config.compression_type,
            retry_backoff_ms=self._config.retry_backoff_ms,
            value_serializer=self._serialize,
            key_serializer=self._serialize_key,
        )
        await self._producer.start()
        self._started = True
        logger.info("Kafka producer started — bootstrap=%s", self._config.bootstrap_servers)

    async def stop(self) -> None:
        if self._producer and self._started:
            await self._producer.stop()
            self._started = False
            logger.info("Kafka producer stopped.")

    async def send(
        self,
        topic: str,
        value: dict[str, Any] | BaseModel,
        key: str | None = None,
        headers: dict[str, str] | None = None,
        partition: int | None = None,
    ) -> None:
        if not self._started or not self._producer:
            raise RuntimeError("Producer not started. Call start() first.")

        payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
        kafka_headers = [(k, v.encode()) for k, v in (headers or {}).items()]

        await self._producer.send_and_wait(
            topic=topic,
            value=payload,
            key=key,
            headers=kafka_headers if kafka_headers else None,
            partition=partition,
        )
        logger.debug("Produced message to %s [key=%s]", topic, key)

    async def send_sensor_event(
        self,
        topic: str,
        machine_id: str,
        value: dict[str, Any] | BaseModel,
        headers: dict[str, str] | None = None,
    ) -> None:
        """Send with machine_id as partition key — guarantees per-machine ordering."""
        await self.send(topic=topic, value=value, key=machine_id, headers=headers)

    async def send_batch(
        self,
        topic: str,
        messages: list[tuple[str | None, dict[str, Any] | BaseModel]],
    ) -> None:
        if not self._started or not self._producer:
            raise RuntimeError("Producer not started. Call start() first.")

        batch = self._producer.create_batch()
        for key, value in messages:
            payload = value.model_dump(mode="json") if isinstance(value, BaseModel) else value
            serialized = orjson.dumps(payload)
            metadata = batch.append(
                key=key.encode() if key else None,
                value=serialized,
                timestamp=None,
            )
            if metadata is None:
                partitions = await self._producer.partitions_for(topic)
                await self._producer.send_batch(batch, topic, partition=0)
                batch = self._producer.create_batch()
                batch.append(key=key.encode() if key else None, value=serialized, timestamp=None)

        partitions = await self._producer.partitions_for(topic)
        await self._producer.send_batch(batch, topic, partition=0)
        logger.debug("Produced batch of %d messages to %s", len(messages), topic)

    @staticmethod
    def _serialize(value: Any) -> bytes:
        return orjson.dumps(value)

    @staticmethod
    def _serialize_key(key: Any) -> bytes | None:
        if key is None:
            return None
        if isinstance(key, bytes):
            return key
        return str(key).encode("utf-8")

    async def __aenter__(self) -> KafkaMessageProducer:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()
