import json
import uuid
from typing import List

import asyncpg

from backend.shared.base.event import DomainEvent


class EventStore:
    """
    Append-only PostgreSQL event store using asyncpg.
    Persists DomainEvents and retrieves aggregate histories.
    """

    def __init__(self, pool: asyncpg.Pool) -> None:
        """
        Initialize the event store with an asyncpg connection pool.
        
        Args:
            pool (asyncpg.Pool): The database connection pool.
        """
        self.pool = pool

    async def append(self, events: List[DomainEvent]) -> None:
        """
        Append a batch of domain events in an atomic transaction.
        
        Args:
            events (List[DomainEvent]): The events to persist.
        """
        if not events:
            return

        query = """
            INSERT INTO events (
                event_id, aggregate_id, aggregate_type, event_type, 
                payload, metadata, occurred_at, version
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """
        
        # Prepare the list of tuples for executemany
        values = [
            (
                str(e.event_id),
                str(e.aggregate_id),
                e.aggregate_type,
                e.event_type,
                json.dumps(e.payload),
                json.dumps(e.metadata),
                e.occurred_at,
                e.version
            ) for e in events
        ]

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.executemany(query, values)

    async def get_aggregate_history(self, aggregate_id: uuid.UUID) -> List[DomainEvent]:
        """
        Retrieve the full event history for a specific aggregate, ordered by version.
        
        Args:
            aggregate_id (uuid.UUID): The aggregate to retrieve.
            
        Returns:
            List[DomainEvent]: The chronologically ordered events.
        """
        query = """
            SELECT event_id, aggregate_id, aggregate_type, event_type, 
                   payload, metadata, occurred_at, version
            FROM events
            WHERE aggregate_id = $1
            ORDER BY version ASC
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, str(aggregate_id))

        return [self._row_to_event(row) for row in rows]

    async def get_events_after(self, position: int, limit: int = 100) -> List[DomainEvent]:
        """
        Retrieve events globally after a given sequential position.
        Useful for event replay and projections.
        Assumes an implicit auto-incrementing `global_position` column in the DB.
        
        Args:
            position (int): The global sequence number to start after.
            limit (int): The maximum number of events to return.
            
        Returns:
            List[DomainEvent]: The requested events.
        """
        query = """
            SELECT event_id, aggregate_id, aggregate_type, event_type, 
                   payload, metadata, occurred_at, version
            FROM events
            WHERE global_position > $1
            ORDER BY global_position ASC
            LIMIT $2
        """
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, position, limit)

        return [self._row_to_event(row) for row in rows]

    def _row_to_event(self, row: asyncpg.Record) -> DomainEvent:
        """Map a database row back to a DomainEvent."""
        return DomainEvent(
            event_id=uuid.UUID(row["event_id"]),
            aggregate_id=uuid.UUID(row["aggregate_id"]),
            aggregate_type=row["aggregate_type"],
            event_type=row["event_type"],
            payload=json.loads(row["payload"]),
            metadata=json.loads(row["metadata"]),
            occurred_at=row["occurred_at"],
            version=row["version"],
        )
