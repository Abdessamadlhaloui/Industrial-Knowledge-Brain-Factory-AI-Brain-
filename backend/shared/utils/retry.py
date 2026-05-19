import asyncio
import functools
import logging
from typing import Any, Callable, Tuple, Type, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def retry(
    max_attempts: int = 3,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    backoff_factor: float = 2.0,
) -> Callable:
    """
    Async-compatible decorator to retry a function upon specific exceptions.
    
    Args:
        max_attempts (int): Maximum number of times to try the function.
        exceptions (Tuple[Type[Exception], ...]): Exceptions that trigger a retry.
        backoff_factor (float): Multiplier for the delay between retries.
        
    Returns:
        Callable: The decorated async function.
    """

    def decorator(func: Callable[..., Awaitable[T]]) -> Callable[..., Awaitable[T]]:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 1
            delay = 1.0

            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt >= max_attempts:
                        logger.error(
                            "Retry exhausted for %s after %d attempts. Last error: %s",
                            func.__name__,
                            max_attempts,
                            str(e),
                        )
                        raise
                    
                    logger.warning(
                        "Attempt %d/%d for %s failed: %s. Retrying in %.2f seconds.",
                        attempt,
                        max_attempts,
                        func.__name__,
                        str(e),
                        delay,
                    )
                    
                    await asyncio.sleep(delay)
                    attempt += 1
                    delay *= backoff_factor

        return wrapper

    return decorator
