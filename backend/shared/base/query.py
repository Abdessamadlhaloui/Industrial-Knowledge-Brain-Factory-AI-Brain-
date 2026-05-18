from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

R = TypeVar("R")


class Query(BaseModel, ABC):
    """Base query in CQRS. Queries are read-only and never mutate state."""

    model_config = ConfigDict(frozen=True)

    query_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    correlation_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QueryResult(BaseModel):
    """Wrapper for query results with metadata."""

    model_config = ConfigDict(frozen=True)

    data: Any
    total_count: int | None = None
    page: int | None = None
    page_size: int | None = None


class QueryHandler(ABC, Generic[R]):
    """Abstract handler that processes a single query type and returns a result."""

    @abstractmethod
    async def handle(self, query: Query) -> R:
        ...
