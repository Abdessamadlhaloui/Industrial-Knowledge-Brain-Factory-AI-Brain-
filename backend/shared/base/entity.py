from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Entity(BaseModel, ABC):
    """Base entity with identity and audit fields. All domain entities inherit from this."""

    model_config = ConfigDict(
        frozen=False,
        from_attributes=True,
        json_encoders={datetime: lambda v: v.isoformat()},
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique entity identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)
