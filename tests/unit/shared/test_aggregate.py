import uuid
import pytest
from pydantic import ValidationError

from backend.shared.base.aggregate import AggregateRoot
from backend.shared.base.event import DomainEvent


class DummyEvent(DomainEvent):
    pass


class DummyAggregate(AggregateRoot):
    status: str


def test_aggregate_event_collection():
    aggregate = DummyAggregate(status="created")
    
    event = DummyEvent(
        aggregate_id=aggregate.id,
        aggregate_type="DummyAggregate",
        event_type="dummy.created",
    )
    
    aggregate.add_event(event)
    events = aggregate.pull_events()
    
    assert len(events) == 1
    assert events[0] == event
    assert len(aggregate.pull_events()) == 0  # Should be cleared after first pull

def test_aggregate_immutability():
    aggregate = DummyAggregate(status="created")
    
    with pytest.raises(ValidationError):
        aggregate.status = "updated"

def test_aggregate_version_initialization():
    aggregate = DummyAggregate(status="created")
    assert aggregate.version == 0
