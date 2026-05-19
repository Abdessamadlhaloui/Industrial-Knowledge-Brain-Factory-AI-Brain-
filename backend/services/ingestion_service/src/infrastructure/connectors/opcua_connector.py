import asyncio
import logging
import os
from typing import Any, Callable, List, Optional, Awaitable

from asyncua import Client

logger = logging.getLogger(__name__)


class OPCUAConnector:
    """
    Async OPC-UA Connector using asyncua.
    Manages connections to PLCs, node reading, and data change subscriptions.
    """

    def __init__(self) -> None:
        self.endpoint = os.environ.get("OPCUA_ENDPOINT", "opc.tcp://localhost:4840/freeopcua/server/")
        self.username = os.environ.get("OPCUA_USERNAME")
        self.password = os.environ.get("OPCUA_PASSWORD")
        
        self.client: Optional[Client] = None
        self.subscription: Any = None

    async def connect(self) -> None:
        """Establishes an async connection to the OPC-UA server."""
        try:
            self.client = Client(url=self.endpoint)
            
            if self.username and self.password:
                self.client.set_user(self.username)
                self.client.set_password(self.password)
                
            await self.client.connect()
            logger.info("Successfully connected to OPC-UA endpoint: %s", self.endpoint)
        except Exception as e:
            logger.error("Failed to connect to OPC-UA endpoint '%s': %s", self.endpoint, str(e))
            raise e

    async def read_node(self, node_id: str) -> Any:
        """Reads the current value of a specific OPC-UA node."""
        if not self.client:
            raise RuntimeError("OPCUAConnector is not connected.")
            
        try:
            node = self.client.get_node(node_id)
            value = await node.read_value()
            return value
        except Exception as e:
            logger.error("Failed to read OPC-UA node '%s': %s", node_id, str(e))
            raise e

    async def subscribe(self, node_ids: List[str], handler: Callable[[str, Any], Awaitable[None]]) -> None:
        """
        Subscribes to data changes on the specified nodes.
        Delegates changes to the injected async handler.
        """
        if not self.client:
            raise RuntimeError("OPCUAConnector is not connected.")

        class SubHandler:
            def datachange_notification(self, node: Any, val: Any, data: Any) -> None:
                asyncio.ensure_future(handler(str(node), val))

        try:
            self.subscription = await self.client.create_subscription(500, SubHandler())
            
            for node_id in node_ids:
                node = self.client.get_node(node_id)
                await self.subscription.subscribe_data_change(node)
                
            logger.info("Successfully subscribed to %d OPC-UA nodes.", len(node_ids))
        except Exception as e:
            logger.error("Failed to establish OPC-UA subscription: %s", str(e))
            raise e

    async def disconnect(self) -> None:
        """Gracefully tears down subscriptions and the OPC-UA client connection."""
        if self.subscription:
            try:
                await self.subscription.delete()
                logger.info("OPC-UA subscription deleted.")
            except Exception as e:
                logger.warning("Error deleting OPC-UA subscription: %s", str(e))
                
        if self.client:
            try:
                await self.client.disconnect()
                logger.info("Disconnected from OPC-UA endpoint.")
            except Exception as e:
                logger.warning("Error disconnecting OPC-UA client: %s", str(e))
