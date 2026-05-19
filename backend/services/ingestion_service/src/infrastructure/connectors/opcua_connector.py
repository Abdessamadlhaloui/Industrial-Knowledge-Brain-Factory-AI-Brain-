import asyncio
import logging
import time
from typing import Any, Dict, List

from asyncua import Client, Node
from asyncua.common.subscription import SubHandler

from backend.shared.infrastructure.messaging.kafka_producer import KafkaMessageProducer

logger = logging.getLogger(__name__)


class OPCUASubscriptionHandler(SubHandler):
    """
    Handler for OPC-UA subscription data changes.
    """
    def __init__(self, machine_id: str, tenant_id: str, factory_id: str, kafka_producer: KafkaMessageProducer):
        self.machine_id = machine_id
        self.tenant_id = tenant_id
        self.factory_id = factory_id
        self.kafka_producer = kafka_producer

    async def datachange_notification(self, node: Node, val: Any, data: Any) -> None:
        """
        Called when a subscribed node's value changes.
        """
        try:
            node_id = node.nodeid.Identifier
            # Assuming node_id is something like "sensor_id.metric_name"
            # For simplicity, we use node_id as sensor_id and metric_name
            sensor_id = str(node_id)
            metric_name = "opcua_value"  # Default if unable to parse

            if isinstance(sensor_id, str) and "." in sensor_id:
                parts = sensor_id.split(".")
                sensor_id = parts[0]
                metric_name = parts[1]

            # data.monitored_item.Value has ServerTimestamp, SourceTimestamp, StatusCode
            quality = 0
            if hasattr(data.monitored_item.Value, "StatusCode") and data.monitored_item.Value.StatusCode:
                quality = data.monitored_item.Value.StatusCode.value

            timestamp = time.time() * 1000
            if hasattr(data.monitored_item.Value, "SourceTimestamp") and data.monitored_item.Value.SourceTimestamp:
                timestamp = data.monitored_item.Value.SourceTimestamp.timestamp() * 1000

            payload = {
                "sensor_id": sensor_id,
                "machine_id": self.machine_id,
                "metric_name": metric_name,
                "value": float(val) if isinstance(val, (int, float)) else val,
                "unit": "unknown",
                "timestamp": timestamp,
                "quality": quality,
                "tenant_id": self.tenant_id,
                "factory_id": self.factory_id
            }

            # Partition key is sensor_id as requested
            await self.kafka_producer.send(
                topic="ikb.sensors.raw",
                value=payload,
                key=sensor_id
            )
            logger.debug("Published OPC-UA datachange for node %s", node_id)
        except Exception as e:
            logger.error("Error processing OPC-UA datachange: %s", e)

    def event_notification(self, event: Any) -> None:
        pass


class OPCUAConnector:
    """
    OPC-UA Connector using asyncua.
    Maintains subscriptions with 500ms sampling, monitors heartbeat, and handles exponential reconnects.
    """

    def __init__(
        self, 
        url: str, 
        node_ids: List[str], 
        machine_id: str, 
        tenant_id: str, 
        factory_id: str,
        kafka_producer: KafkaMessageProducer
    ):
        self.url = url
        self.node_ids = node_ids
        self.machine_id = machine_id
        self.tenant_id = tenant_id
        self.factory_id = factory_id
        self.kafka_producer = kafka_producer
        self.client = Client(url=self.url)
        self._running = False
        self._monitor_task: asyncio.Task | None = None

    async def start(self) -> None:
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("OPC-UA connector started for %s", self.url)

    async def stop(self) -> None:
        self._running = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        try:
            await self.client.disconnect()
        except Exception:
            pass
        logger.info("OPC-UA connector stopped for %s", self.url)

    async def _monitor_loop(self) -> None:
        backoff = 1.0
        max_backoff = 60.0

        while self._running:
            try:
                logger.info("Connecting to OPC-UA server %s", self.url)
                await self.client.connect()
                backoff = 1.0  # Reset backoff on successful connect

                # Setup subscriptions (500ms requested)
                handler = OPCUASubscriptionHandler(
                    self.machine_id, self.tenant_id, self.factory_id, self.kafka_producer
                )
                sub = await self.client.create_subscription(500, handler)
                
                nodes = [self.client.get_node(nid) for nid in self.node_ids]
                await sub.subscribe_data_change(nodes)
                logger.info("Subscribed to %d nodes on %s", len(nodes), self.url)

                # Heartbeat loop
                while self._running:
                    # Simple heartbeat: read ServerStatus time
                    server_time_node = self.client.get_node("i=2258")
                    await server_time_node.read_value()
                    await asyncio.sleep(5.0)

            except Exception as e:
                logger.error("OPC-UA connection error: %s. Reconnecting in %.1f seconds...", e, backoff)
                try:
                    await self.client.disconnect()
                except Exception:
                    pass
                
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, max_backoff)
