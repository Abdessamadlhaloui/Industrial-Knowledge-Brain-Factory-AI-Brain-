import asyncio
import pytest
from unittest.mock import AsyncMock

from backend.shared.utils.circuit_breaker import CircuitBreaker, CircuitState

@pytest.fixture
def breaker():
    return CircuitBreaker(failure_threshold=3, recovery_timeout=0.1, half_open_max_calls=2)


@pytest.mark.asyncio
async def test_circuit_breaker_success(breaker):
    mock_func = AsyncMock(return_value="success")
    result = await breaker.call(mock_func())
    
    assert result == "success"
    assert breaker.state == CircuitState.CLOSED


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures(breaker):
    mock_func = AsyncMock(side_effect=ValueError("Failure"))
    
    # Trigger failures up to threshold
    for _ in range(3):
        with pytest.raises(ValueError):
            await breaker.call(mock_func())
            
    assert breaker.state == CircuitState.OPEN


@pytest.mark.asyncio
async def test_circuit_breaker_prevents_calls_when_open(breaker):
    mock_func = AsyncMock(side_effect=ValueError("Failure"))
    
    # Open the circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            await breaker.call(mock_func())
            
    # Subsequence calls should fail immediately without invoking the function
    mock_success = AsyncMock(return_value="success")
    with pytest.raises(Exception, match="Circuit breaker is OPEN"):
        await breaker.call(mock_success())
        
    mock_success.assert_not_called()


@pytest.mark.asyncio
async def test_circuit_breaker_recovery(breaker):
    mock_func = AsyncMock(side_effect=ValueError("Failure"))
    
    # Open the circuit
    for _ in range(3):
        with pytest.raises(ValueError):
            await breaker.call(mock_func())
            
    assert breaker.state == CircuitState.OPEN
    
    # Wait for recovery timeout
    await asyncio.sleep(0.15)
    
    # Circuit should allow a test request (HALF_OPEN)
    mock_success = AsyncMock(return_value="success")
    result = await breaker.call(mock_success())
    
    assert result == "success"
    assert breaker.state == CircuitState.HALF_OPEN
    
    # Make enough successful calls to close the circuit
    await breaker.call(mock_success())
    
    assert breaker.state == CircuitState.CLOSED
