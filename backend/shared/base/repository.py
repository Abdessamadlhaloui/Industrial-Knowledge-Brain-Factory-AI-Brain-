from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from backend.shared.base.aggregate import AggregateRoot

T = TypeVar("T", bound=AggregateRoot)


class Repository(ABC, Generic[T]):
    """Abstract repository for aggregate persistence. Implementations must be async."""

    @abstractmethod
    async def get_by_id(self, entity_id: uuid.UUID) -> T | None:
        ...

    @abstractmethod
    async def save(self, entity: T) -> T:
        ...

    @abstractmethod
    async def delete(self, entity_id: uuid.UUID) -> bool:
        ...

    @abstractmethod
    async def list_all(self, *, offset: int = 0, limit: int = 100) -> list[T]:
        ...

    @abstractmethod
    async def count(self) -> int:
        ...

    @abstractmethod
    async def exists(self, entity_id: uuid.UUID) -> bool:
        ...
