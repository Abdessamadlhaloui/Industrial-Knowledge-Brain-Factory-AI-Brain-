import asyncio
import logging
import time
from enum import Enum
from typing import Callable, Coroutine, Any

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    Async-compatible Circuit Breaker pattern to prevent cascading failures.
    Transitions through CLOSED -> OPEN -> HALF_OPEN based on error thresholds.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 3,
    ) -> None:
        """
        Initialize the circuit breaker.
        
        Args:
            failure_threshold (int): Number of consecutive failures before opening.
            recovery_timeout (float): Seconds to wait before transitioning to HALF_OPEN.
            half_open_max_calls (int): Number of successful calls needed in HALF_OPEN to close the circuit.
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._half_open_success_count = 0
        self._last_failure_time: float | None = None
        self._lock = asyncio.Lock()
        
        # Optional callbacks for state transitions
        self.on_open: Callable[[], None] | None = None
        self.on_half_open: Callable[[], None] | None = None
        self.on_close: Callable[[], None] | None = None

    @property
    def state(self) -> CircuitState:
        return self._state

    async def call(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """
        Execute an async function through the circuit breaker.
        
        Args:
            coro (Coroutine): The coroutine to execute.
            
        Returns:
            Any: The result of the coroutine if successful.
            
        Raises:
            Exception: Whatever the coroutine raises, or a custom exception if the circuit is open.
        """
        async with self._lock:
            self._evaluate_state()

            if self._state == CircuitState.OPEN:
                raise Exception("Circuit breaker is OPEN. Fast failing.")

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_success_count >= self.half_open_max_calls:
                    # We have already issued the allowed half_open calls
                    # New calls must wait or fail depending on implementation; 
                    # here we'll just fail them until the pending ones complete.
                    raise Exception("Circuit breaker is HALF_OPEN. Waiting for current probe requests.")

        try:
            result = await coro
            await self._record_success()
            return result
        except Exception as e:
            await self._record_failure()
            raise e

    def _evaluate_state(self) -> None:
        """Internal method to evaluate and potentially transition state based on timeouts."""
        if self._state == CircuitState.OPEN:
            if self._last_failure_time and (time.time() - self._last_failure_time) > self.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN)

    async def _record_success(self) -> None:
        """Record a successful execution."""
        async with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._half_open_success_count += 1
                if self._half_open_success_count >= self.half_open_max_calls:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                self._failure_count = 0  # Reset failures on success

    async def _record_failure(self) -> None:
        """Record a failed execution."""
        async with self._lock:
            self._last_failure_time = time.time()
            if self._state == CircuitState.HALF_OPEN:
                # Any failure in HALF_OPEN immediately opens the circuit again
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state and emit events."""
        old_state = self._state
        self._state = new_state
        logger.info("CircuitBreaker transitioned from %s to %s", old_state.name, new_state.name)

        if new_state == CircuitState.OPEN:
            if self.on_open:
                self.on_open()
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_success_count = 0
            if self.on_half_open:
                self.on_half_open()
        elif new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._half_open_success_count = 0
            if self.on_close:
                self.on_close()
