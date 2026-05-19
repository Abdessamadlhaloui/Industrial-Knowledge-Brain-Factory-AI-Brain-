import logging
from typing import Any, Dict, List, Optional

from neo4j import AsyncGraphDatabase, AsyncDriver

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Async Neo4j Client wrapper managing the connection pool.
    """

    def __init__(self, uri: str, user: str, password: str, max_connection_pool_size: int = 50):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver: Optional[AsyncDriver] = None
        self._max_pool_size = max_connection_pool_size

    async def connect(self) -> None:
        """Initialize the async driver connection pool."""
        if not self._driver:
            self._driver = AsyncGraphDatabase.driver(
                self.uri, 
                auth=(self.user, self.password),
                max_connection_pool_size=self._max_pool_size
            )
            logger.info("Connected to Neo4j at %s", self.uri)

    async def close(self) -> None:
        """Close the driver and release resources."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("Closed Neo4j connection pool.")

    async def verify_connectivity(self) -> bool:
        """Health check to verify connectivity to the DB."""
        if not self._driver:
            return False
        try:
            await self._driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error("Neo4j connectivity check failed: %s", str(e))
            return False

    async def execute_query(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a read-only Cypher query.
        
        Args:
            cypher (str): The parameterized Cypher query.
            parameters (Dict[str, Any]): Dictionary of parameters.
            
        Returns:
            List[Dict[str, Any]]: List of records as dictionaries.
        """
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
            
        parameters = parameters or {}
        
        async with self._driver.session() as session:
            try:
                result = await session.run(cypher, parameters)
                records = await result.data()
                return records
            except Exception as e:
                logger.error("Failed to execute Neo4j read query: %s\nParams: %s", str(e), parameters)
                raise

    async def execute_write(self, cypher: str, parameters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Execute a write Cypher query inside an explicit transaction.
        
        Args:
            cypher (str): The parameterized Cypher write query.
            parameters (Dict[str, Any]): Dictionary of parameters.
            
        Returns:
            List[Dict[str, Any]]: Resulting records, if any.
        """
        if not self._driver:
            raise RuntimeError("Neo4j driver not initialized. Call connect() first.")
            
        parameters = parameters or {}

        async def _transaction_function(tx: Any) -> List[Dict[str, Any]]:
            result = await tx.run(cypher, parameters)
            return await result.data()

        async with self._driver.session() as session:
            try:
                records = await session.execute_write(_transaction_function)
                return records
            except Exception as e:
                logger.error("Failed to execute Neo4j write query: %s\nParams: %s", str(e), parameters)
                raise
