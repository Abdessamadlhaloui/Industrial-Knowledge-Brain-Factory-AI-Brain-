import asyncio
import logging
from typing import AsyncGenerator

import asyncpg
from asyncpg.pool import Pool

logger = logging.getLogger(__name__)


class PostgresPool:
    """
    Async PostgreSQL connection pool management using asyncpg.
    Provides context managers for acquiring connections and auto-retries transient errors.
    """

    def __init__(self, dsn: str, min_size: int = 10, max_size: int = 50) -> None:
        """
        Initialize the connection pool manager.
        
        Args:
            dsn (str): The PostgreSQL connection string (DSN).
            min_size (int): Minimum number of connections in the pool.
            max_size (int): Maximum number of connections in the pool.
        """
        self.dsn = dsn
        self.min_size = min_size
        self.max_size = max_size
        self._pool: Pool | None = None

    async def connect(self) -> None:
        """Initialize the connection pool."""
        if not self._pool:
            self._pool = await asyncpg.create_pool(
                dsn=self.dsn,
                min_size=self.min_size,
                max_size=self.max_size,
            )
            logger.info("PostgreSQL connection pool initialized.")

    async def disconnect(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None
            logger.info("PostgreSQL connection pool closed.")

    @property
    def pool(self) -> Pool:
        """Get the underlying asyncpg Pool."""
        if not self._pool:
            raise RuntimeError("Database pool is not initialized. Call connect() first.")
        return self._pool

    async def acquire(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """
        Context manager to acquire a connection from the pool.
        Includes an automatic retry mechanism for transient network/connection errors
        (3 attempts with exponential backoff).
        
        Yields:
            asyncpg.Connection: An active database connection.
        """
        max_attempts = 3
        backoff = 1.0

        for attempt in range(1, max_attempts + 1):
            try:
                async with self.pool.acquire() as conn:
                    yield conn
                    return
            except (asyncpg.PostgresConnectionError, OSError) as e:
                logger.warning("Failed to acquire connection (attempt %d/%d): %s", attempt, max_attempts, str(e))
                if attempt == max_attempts:
                    raise
                await asyncio.sleep(backoff)
                backoff *= 2.0

    async def health_check(self) -> bool:
        """
        Perform a simple health check against the database.
        
        Returns:
            bool: True if the database is reachable, False otherwise.
        """
        if not self._pool:
            return False
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error("PostgreSQL health check failed: %s", str(e))
            return False
