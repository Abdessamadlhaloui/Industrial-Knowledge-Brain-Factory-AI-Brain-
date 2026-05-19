import pytest
from unittest.mock import AsyncMock, patch

from backend.shared.infrastructure.database.postgres import PostgresPool


@pytest.mark.asyncio
async def test_postgres_pool_connect_disconnect():
    pool = PostgresPool("postgresql://dummy:dummy@localhost:5432/dummy")
    
    with patch("asyncpg.create_pool", new_callable=AsyncMock) as mock_create:
        mock_create.return_value = AsyncMock()
        await pool.connect()
        assert pool._pool is not None
        
        await pool.disconnect()
        assert pool._pool is None

@pytest.mark.asyncio
async def test_postgres_pool_health_check_success():
    pool = PostgresPool("postgresql://dummy:dummy@localhost:5432/dummy")
    pool._pool = AsyncMock()
    
    mock_conn = AsyncMock()
    pool._pool.acquire.return_value.__aenter__.return_value = mock_conn
    
    is_healthy = await pool.health_check()
    assert is_healthy is True
    mock_conn.execute.assert_called_with("SELECT 1")

@pytest.mark.asyncio
async def test_postgres_pool_health_check_failure():
    pool = PostgresPool("postgresql://dummy:dummy@localhost:5432/dummy")
    pool._pool = AsyncMock()
    
    mock_conn = AsyncMock()
    mock_conn.execute.side_effect = Exception("DB down")
    pool._pool.acquire.return_value.__aenter__.return_value = mock_conn
    
    is_healthy = await pool.health_check()
    assert is_healthy is False
