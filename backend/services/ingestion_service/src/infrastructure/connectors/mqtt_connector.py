import asyncio
import json
import logging
import time
from typing import Any, Dict

import aiomqtt

from backend.shared.infrastructure.messaging.kafka_producer import KafkaMessageProducer

logger = logging.getLogger(__name__)


class MQTTConnector:
    """
    MQTT Connector using aiomqtt.
    Subscribes to 'factory/+/sensors/#' pattern and publishes to Kafka.
    """

    def __init__(
        self,
        broker_url: str,
        tenant_id: str,
        factory_id: str,
        kafka_producer: KafkaMessageProducer,
        topic_pattern: str = "factory/+/sensors/#",
        port: int = 1883
    ):
        self.broker_url = broker_url
        self.port = port
        self.topic_pattern = topic_pattern
        self.tenant_id = tenant_id
        self.factory_id = factory_id
        self.kafka_producer = kafka_producer
        self._running = False
        self._loop_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._running = True
        self._loop_task = asyncio.create_task(self._monitor_loop())
        logger.info("MQTT connector started for %s", self.broker_url)

    async def stop(self) -> None:
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        logger.info("MQTT connector stopped for %s", self.broker_url)

    async def _monitor_loop(self) -> None:
        backoff = 1.0
        max_backoff = 60.0

        while self._running:
            try:
                logger.info("Connecting to MQTT broker %s:%d", self.broker_url, self.port)
                async with aiomqtt.Client(hostname=self.broker_url, port=self.port) as client:
                    backoff = 1.0  # Reset backoff
                    
                    await client.subscribe(self.topic_pattern)
                    logger.info("Subscribed to MQTT pattern: %s", self.topic_pattern)

                    async for message in client.messages:
                        if not self._running:
                            break
                        
                        await self._process_message(message)
                        
            except aiomqtt.MqttError as e:
                logger.error("MQTT connection error: %s. Reconnecting in %.1f seconds...", e, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
            except Exception as e:
                logger.error("Unexpected error in MQTT loop: %s", e)
                await asyncio.sleep(5.0)

    async def _process_message(self, message: aiomqtt.Message) -> None:
        try:
            topic = str(message.topic)
            # Expected pattern: factory/{machine_id}/sensors/{sensor_name}
            parts = topic.split("/")
            if len(parts) >= 4:
                machine_id = parts[1]
                sensor_name = parts[3]
            else:
                machine_id = "unknown"
                sensor_name = "unknown"

            payload_raw = message.payload
            if isinstance(payload_raw, bytes):
                payload_raw = payload_raw.decode("utf-8")
                
            try:
                data = json.loads(payload_raw)
                val = data.get("value", payload_raw)
            except json.JSONDecodeError:
                val = payload_raw

            kafka_payload = {
                "sensor_id": sensor_name,
                "machine_id": machine_id,
                "metric_name": sensor_name,
                "value": float(val) if isinstance(val, (int, float, str)) and str(val).replace('.','',1).isdigit() else val,
                "unit": "unknown",
                "timestamp": time.time() * 1000,
                "quality": 0,
                "tenant_id": self.tenant_id,
                "factory_id": self.factory_id
            }

            await self.kafka_producer.send(
                topic="ikb.sensors.raw",
                value=kafka_payload,
                key=sensor_name  # Partition key = sensor_id
            )
        except Exception as e:
            logger.error("Error processing MQTT message from topic %s: %s", message.topic, e)
