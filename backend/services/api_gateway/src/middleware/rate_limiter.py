import logging
import time
from typing import Callable, Dict
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Token bucket algorithm (Mock implementation).
    Per-user and per-tenant limits.
    """
    
    def __init__(self, app):
        super().__init__(app)
        self.default_limit = 100 # per min
        self.agent_limit = 10    # per min
        self.skip_paths = {"/health", "/api/docs"}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if request.url.path in self.skip_paths or request.method == "OPTIONS":
            return await call_next(request)

        tenant_id = getattr(request.state, "tenant_id", "anonymous")
        user_id = getattr(request.state, "user_id", "anonymous")
        
        # Determine limit
        limit = self.agent_limit if request.url.path.startswith("/api/v1/agents") else self.default_limit
        
        # Mock Redis Token Bucket check
        # In production: run a Lua script in Redis to decrement bucket and check capacity atomically
        allowed = True # Mocking success
        
        if not allowed:
            logger.warning("Rate limit exceeded for %s on %s", user_id, request.url.path)
            response = JSONResponse(
                status_code=429,
                content={
                    "error_code": "RATE_LIMIT_EXCEEDED",
                    "message": f"Rate limit of {limit} req/min exceeded.",
                    "request_id": getattr(request.state, "trace_id", "N/A"),
                    "timestamp": time.time()
                }
            )
            response.headers["Retry-After"] = "60"
            return response

        return await call_next(request)
