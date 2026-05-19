import pytest
from backend.shared.utils.retry import retry

@pytest.mark.asyncio
async def test_retry_success_on_first_attempt():
    attempts = 0

    @retry(max_attempts=3, backoff_factor=0.1)
    async def always_succeeds():
        nonlocal attempts
        attempts += 1
        return "success"

    result = await always_succeeds()
    assert result == "success"
    assert attempts == 1

@pytest.mark.asyncio
async def test_retry_success_after_failure():
    attempts = 0

    @retry(max_attempts=3, backoff_factor=0.1)
    async def eventually_succeeds():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ValueError("Failed")
        return "success"

    result = await eventually_succeeds()
    assert result == "success"
    assert attempts == 3

@pytest.mark.asyncio
async def test_retry_exhausted():
    attempts = 0

    @retry(max_attempts=3, exceptions=(ValueError,), backoff_factor=0.1)
    async def always_fails():
        nonlocal attempts
        attempts += 1
        raise ValueError("Failed")

    with pytest.raises(ValueError):
        await always_fails()
        
    assert attempts == 3

@pytest.mark.asyncio
async def test_retry_ignores_other_exceptions():
    attempts = 0

    # Only retries on ValueError, not TypeError
    @retry(max_attempts=3, exceptions=(ValueError,), backoff_factor=0.1)
    async def fails_with_type_error():
        nonlocal attempts
        attempts += 1
        raise TypeError("Failed with wrong error")

    with pytest.raises(TypeError):
        await fails_with_type_error()
        
    # Should only try once because exception is not caught by retry
    assert attempts == 1
