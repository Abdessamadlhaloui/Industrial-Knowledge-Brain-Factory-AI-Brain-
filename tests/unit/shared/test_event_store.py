import json
import uuid
import pytest
from unittest.mock import AsyncMock, MagicMock

from backend.shared.base.event import DomainEvent
from backend.shared.infrastructure.database.event_store import EventStore


class SampleEvent(DomainEvent):
    pass


@pytest.mark.asyncio
async def test_event_store_append():
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    mock_conn.transaction.return_value.__aenter__.return_value = AsyncMock()

    store = EventStore(mock_pool)
    aggregate_id = uuid.uuid4()
    
    events = [
        SampleEvent(
            aggregate_id=aggregate_id,
            aggregate_type="test",
            event_type="test.event",
            payload={"key": "value"}
        )
    ]
    
    await store.append(events)
    
    mock_conn.executemany.assert_called_once()
    args, _ = mock_conn.executemany.call_args
    assert "INSERT INTO events" in args[0]
    assert len(args[1]) == 1
    assert args[1][0][1] == str(aggregate_id)

@pytest.mark.asyncio
async def test_event_store_get_aggregate_history():
    mock_pool = AsyncMock()
    mock_conn = AsyncMock()
    mock_pool.acquire.return_value.__aenter__.return_value = mock_conn
    
    aggregate_id = uuid.uuid4()
    event_id = uuid.uuid4()
    
    # Mock row returned by asyncpg
    mock_row = {
        "event_id": str(event_id),
        "aggregate_id": str(aggregate_id),
        "aggregate_type": "test",
        "event_type": "test.event",
        "payload": json.dumps({"key": "value"}),
        "metadata": json.dumps({}),
        "occurred_at": "2026-05-18T12:00:00Z",
        "version": 1,
    }
    mock_conn.fetch.return_value = [mock_row]

    store = EventStore(mock_pool)
    history = await store.get_aggregate_history(aggregate_id)
    
    assert len(history) == 1
    assert history[0].aggregate_id == aggregate_id
    assert history[0].payload["key"] == "value"
