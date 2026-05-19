import asyncio
import logging
from typing import Awaitable, Callable, Dict, List

from backend.shared.base.event import DomainEvent

logger = logging.getLogger(__name__)

# Type alias for event handlers: async function taking a DomainEvent
EventHandler = Callable[[DomainEvent], Awaitable[None]]


class EventBus:
    """
    In-process async event bus for local event distribution.
    Thread-safe implementation using asyncio.Queue and background worker tasks.
    """

    def __init__(self, num_workers: int = 1) -> None:
        """
        Initialize the event bus.
        
        Args:
            num_workers (int): The number of concurrent background workers to process events.
        """
        self._subscribers: Dict[str, List[EventHandler]] = {}
        self._queue: asyncio.Queue[DomainEvent] = asyncio.Queue()
        self._workers: List[asyncio.Task] = []
        self._num_workers = num_workers
        self._running = False

    async def start(self) -> None:
        """Start the background event processing workers."""
        if self._running:
            return
        
        self._running = True
        for i in range(self._num_workers):
            task = asyncio.create_task(self._worker_loop(), name=f"EventBusWorker-{i}")
            self._workers.append(task)
        logger.info("EventBus started with %d workers.", self._num_workers)

    async def stop(self) -> None:
        """Stop the background workers and wait for the queue to drain."""
        if not self._running:
            return
        
        self._running = False
        await self._queue.join()  # Wait for remaining events to be processed

        for worker in self._workers:
            worker.cancel()
        
        await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()
        logger.info("EventBus stopped.")

    def subscribe(self, event_type: str, handler_coro: EventHandler) -> None:
        """
        Subscribe a coroutine handler to a specific event type.
        
        Args:
            event_type (str): The type of event to subscribe to.
            handler_coro (EventHandler): The async function to execute.
        """
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler_coro)
        logger.debug("Subscribed %s to %s", handler_coro.__name__, event_type)

    async def publish(self, event: DomainEvent) -> None:
        """
        Publish an event to the bus. It will be routed to all matching subscribers.
        
        Args:
            event (DomainEvent): The event to publish.
        """
        if not self._running:
            raise RuntimeError("EventBus is not running. Call start() first.")
        
        await self._queue.put(event)
        logger.debug("Published event %s to queue.", event.event_type)

    async def _worker_loop(self) -> None:
        """Background worker loop that pulls events from the queue and dispatches them."""
        while self._running:
            try:
                event = await self._queue.get()
            except asyncio.CancelledError:
                break
            
            handlers = self._subscribers.get(event.event_type, [])
            if handlers:
                # Execute all handlers for this event concurrently
                tasks = [handler(event) for handler in handlers]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log any errors from handlers
                for handler, result in zip(handlers, results):
                    if isinstance(result, Exception):
                        logger.error(
                            "Handler %s failed for event %s: %s", 
                            handler.__name__, event.event_type, str(result)
                        )
            
            self._queue.task_done()
