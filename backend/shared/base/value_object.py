from __future__ import annotations

from abc import ABC
from typing import Any

from pydantic import BaseModel, ConfigDict


class ValueObject(BaseModel, ABC):
    """Base value object. Value objects are immutable and compared by value, not identity."""

    model_config = ConfigDict(frozen=True)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.model_dump() == other.model_dump()

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.model_dump().items())))
