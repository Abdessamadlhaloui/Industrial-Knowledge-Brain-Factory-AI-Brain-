from __future__ import annotations

import uuid
from abc import ABC
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class DomainEvent(BaseModel, ABC):
    """Base domain event. Events are immutable records of something that happened."""

    model_config = ConfigDict(frozen=True)

    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    event_type: str = Field(description="Fully qualified event type name")
    aggregate_id: uuid.UUID = Field(description="ID of the aggregate that raised this event")
    correlation_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    causation_id: uuid.UUID | None = Field(default=None, description="ID of the command/event that caused this")
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = Field(default=1, description="Event schema version for evolution")
    metadata: dict = Field(default_factory=dict, description="Additional context (tenant, trace_id, etc.)")

    @property
    def partition_key(self) -> str:
        return str(self.aggregate_id)
