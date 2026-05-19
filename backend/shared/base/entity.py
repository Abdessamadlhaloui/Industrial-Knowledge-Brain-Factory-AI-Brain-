from __future__ import annotations

import uuid
from abc import ABC
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class BaseEntity(BaseModel, ABC):
    """
    Base entity with identity and audit fields.
    All domain entities inherit from this.
    Supports equality by id and immutable fields via Pydantic v2.
    """

    model_config = ConfigDict(
        frozen=True,
        from_attributes=True,
        json_encoders={datetime: lambda v: v.isoformat()},
    )

    id: uuid.UUID = Field(default_factory=uuid.uuid4, description="Unique entity identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __eq__(self, other: Any) -> bool:
        """Entities are equal if they are of the same type and have the same id."""
        if not isinstance(other, BaseEntity):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        """Hash based on the entity id."""
        return hash(self.id)
