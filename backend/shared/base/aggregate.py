from __future__ import annotations

from typing import Any

from pydantic import Field, PrivateAttr

from backend.shared.base.entity import BaseEntity
from backend.shared.base.event import DomainEvent


class AggregateRoot(BaseEntity):
    """
    Base aggregate root that collects domain events for later dispatch.
    Inherits from BaseEntity, maintaining immutability for domain fields,
    but manages an internal event list and version.
    """

    # We use PrivateAttr to allow mutability of the event list inside a frozen model
    _domain_events: list[DomainEvent] = PrivateAttr(default_factory=list)
    version: int = Field(default=0, description="Optimistic concurrency version")

    def model_post_init(self, __context: Any) -> None:
        """Initialize the private domain events list after model creation."""
        self._domain_events = []

    def add_event(self, event: DomainEvent) -> None:
        """
        Record a new domain event on this aggregate.
        
        Args:
            event (DomainEvent): The event to record.
        """
        self._domain_events.append(event)

    def pull_events(self) -> list[DomainEvent]:
        """
        Retrieve all recorded domain events and clear the internal list.
        
        Returns:
            list[DomainEvent]: The recorded events since the last pull.
        """
        events = list(self._domain_events)
        self._domain_events.clear()
        return events
