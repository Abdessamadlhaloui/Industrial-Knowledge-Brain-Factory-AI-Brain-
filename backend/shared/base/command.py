from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class Command(BaseModel, ABC):
    """Base command in CQRS. Commands represent intent to change state."""

    model_config = ConfigDict(frozen=True)

    command_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    correlation_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    issued_by: str | None = Field(default=None, description="Principal or system that issued the command")


class CommandHandler(ABC):
    """Abstract handler that processes a single command type."""

    @abstractmethod
    async def handle(self, command: Command) -> None:
        ...
