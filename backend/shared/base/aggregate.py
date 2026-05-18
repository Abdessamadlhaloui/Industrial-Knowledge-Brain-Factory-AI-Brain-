from __future__ import annotations

from typing import Any

from pydantic import Field

from backend.shared.base.entity import Entity
from backend.shared.base.event import DomainEvent


class AggregateRoot(Entity):
    """Base aggregate root that collects domain events for later dispatch."""

    _domain_events: list[DomainEvent] = []
    version: int = Field(default=0, description="Optimistic concurrency version")

    def model_post_init(self, __context: Any) -> None:
        self._domain_events = []

    def raise_event(self, event: DomainEvent) -> None:
        self._domain_events.append(event)

    def collect_events(self) -> list[DomainEvent]:
        events = list(self._domain_events)
        self._domain_events.clear()
        return events

    def increment_version(self) -> None:
        self.version += 1
        self.touch()
