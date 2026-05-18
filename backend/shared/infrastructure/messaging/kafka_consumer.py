from __future__ import annotations

import asyncio
import logging
import signal
from typing import Any, Awaitable, Callable

import orjson
from aiokafka import AIOKafkaConsumer, ConsumerRecord
from pydantic import BaseModel

logger = logging.getLogger(__name__)

MessageHandler = Callable[[dict[str, Any], dict[str, str]], Awaitable[None]]


class KafkaConsumerConfig(BaseModel):
    bootstrap_servers: str = "localhost:9092"
    group_id: str = "ikb-consumer-group"
    auto_offset_reset: str = "earliest"
    enable_auto_commit: bool = False
    max_poll_records: int = 100
    session_timeout_ms: int = 30000
    heartbeat_interval_ms: int = 10000
    max_poll_interval_ms: int = 300000


class KafkaMessageConsumer:
    """Async Kafka consumer with manual commit, graceful shutdown, and dead-letter support."""

    def __init__(
        self,
        topics: list[str],
        handler: MessageHandler,
        config: KafkaConsumerConfig | None = None,
        dead_letter_topic: str | None = None,
        max_retries: int = 3,
    ) -> None:
        self._topics = topics
        self._handler = handler
        self._config = config or KafkaConsumerConfig()
        self._consumer: AIOKafkaConsumer | None = None
        self._running = False
        self._dead_letter_topic = dead_letter_topic
        self._max_retries = max_retries

    async def start(self) -> None:
        self._consumer = AIOKafkaConsumer(
            *self._topics,
            bootstrap_servers=self._config.bootstrap_servers,
            group_id=self._config.group_id,
            auto_offset_reset=self._config.auto_offset_reset,
            enable_auto_commit=self._config.enable_auto_commit,
            max_poll_records=self._config.max_poll_records,
            session_timeout_ms=self._config.session_timeout_ms,
            heartbeat_interval_ms=self._config.heartbeat_interval_ms,
            max_poll_interval_ms=self._config.max_poll_interval_ms,
            value_deserializer=self._deserialize,
        )
        await self._consumer.start()
        self._running = True
        logger.info(
            "Kafka consumer started — topics=%s group=%s",
            self._topics,
            self._config.group_id,
        )

    async def stop(self) -> None:
        self._running = False
        if self._consumer:
            await self._consumer.stop()
            logger.info("Kafka consumer stopped.")

    async def consume(self) -> None:
        if not self._consumer:
            raise RuntimeError("Consumer not started. Call start() first.")

        try:
            async for record in self._consumer:
                if not self._running:
                    break
                await self._process_record(record)
        except asyncio.CancelledError:
            logger.info("Consumer loop cancelled, shutting down gracefully.")
        finally:
            await self.stop()

    async def _process_record(self, record: ConsumerRecord) -> None:
        headers = {k: v.decode() for k, v in (record.headers or [])}
        retry_count = int(headers.get("x-retry-count", "0"))

        try:
            await self._handler(record.value, headers)
            await self._consumer.commit()
            logger.debug(
                "Processed message from %s [partition=%d offset=%d]",
                record.topic,
                record.partition,
                record.offset,
            )
        except Exception:
            logger.exception(
                "Error processing message from %s [partition=%d offset=%d] (retry %d/%d)",
                record.topic,
                record.partition,
                record.offset,
                retry_count,
                self._max_retries,
            )
            if retry_count >= self._max_retries and self._dead_letter_topic:
                await self._send_to_dead_letter(record, headers, retry_count)
            await self._consumer.commit()

    async def _send_to_dead_letter(
        self,
        record: ConsumerRecord,
        headers: dict[str, str],
        retry_count: int,
    ) -> None:
        logger.warning(
            "Sending message to dead letter topic %s after %d retries",
            self._dead_letter_topic,
            retry_count,
        )

    @staticmethod
    def _deserialize(value: bytes) -> dict[str, Any]:
        return orjson.loads(value)

    async def __aenter__(self) -> KafkaMessageConsumer:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()
