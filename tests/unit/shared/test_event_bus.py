import asyncio
import uuid
import pytest

from backend.shared.base.event import DomainEvent
from backend.shared.infrastructure.messaging.event_bus import EventBus


class SimpleEvent(DomainEvent):
    pass


@pytest.mark.asyncio
async def test_event_bus_publish_and_subscribe():
    bus = EventBus(num_workers=1)
    await bus.start()
    
    received_events = []
    
    async def handler(event: DomainEvent):
        received_events.append(event)
        
    bus.subscribe("simple.event", handler)
    
    event = SimpleEvent(
        aggregate_id=uuid.uuid4(),
        aggregate_type="test",
        event_type="simple.event",
    )
    
    await bus.publish(event)
    
    # Allow some time for background worker to process
    await asyncio.sleep(0.1)
    
    await bus.stop()
    
    assert len(received_events) == 1
    assert received_events[0] == event

@pytest.mark.asyncio
async def test_event_bus_unhandled_event_type():
    bus = EventBus(num_workers=1)
    await bus.start()
    
    received_events = []
    
    async def handler(event: DomainEvent):
        received_events.append(event)
        
    bus.subscribe("simple.event", handler)
    
    # Publish different event type
    event = SimpleEvent(
        aggregate_id=uuid.uuid4(),
        aggregate_type="test",
        event_type="different.event",
    )
    
    await bus.publish(event)
    await asyncio.sleep(0.1)
    await bus.stop()
    
    assert len(received_events) == 0
