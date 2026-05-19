from __future__ import annotations

import uuid
from abc import ABC
from datetime import datetime, timezone
from typing import Any, Dict

from pydantic import BaseModel, ConfigDict, Field


class DomainEvent(BaseModel, ABC):
    """
    Base domain event. Events are immutable records of something that happened
    in the domain.
    """

    model_config = ConfigDict(frozen=True)

    event_id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique event identifier")
    aggregate_id: uuid.UUID = Field(description="ID of the aggregate that raised this event")
    aggregate_type: str = Field(description="Type of the aggregate")
    event_type: str = Field(description="Fully qualified event type name")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Event data payload")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context (tenant, trace_id, etc.)")
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="Timestamp of the event")
    version: int = Field(default=1, description="Event schema version for evolution")

    @property
    def partition_key(self) -> str:
        """Returns a stable partition key for message brokers (Kafka/etc.)."""
        return str(self.aggregate_id)
