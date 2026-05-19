import time
import uuid
import logging
from typing import Callable, Dict
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from backend.services.api_gateway.src.middleware.auth import JWTAuthMiddleware
from backend.services.api_gateway.src.middleware.rate_limiter import RateLimiterMiddleware
from backend.services.api_gateway.src.config import get_settings

# In a real setup, we'd import these from their implementation locations
# from backend.services.api_gateway.src.routers import query, agents
# from backend.services.api_gateway.src.websocket import realtime_handler

logger = logging.getLogger(__name__)


class TracingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        trace_id = request.headers.get("X-Trace-Id", str(uuid.uuid4()))
        request.state.trace_id = trace_id
        
        # Log entry
        logger.info("Incoming Request: %s %s [Trace: %s]", request.method, request.url.path, trace_id)
        
        start_time = time.time()
        try:
            response = await call_next(request)
            response.headers["X-Trace-Id"] = trace_id
            
            # Log exit
            latency = (time.time() - start_time) * 1000
            logger.info("Response: %s %s - %d (%.2fms) [Trace: %s]", 
                        request.method, request.url.path, response.status_code, latency, trace_id)
            return response
            
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            logger.error("Request Failed: %s %s - %s (%.2fms) [Trace: %s]", 
                         request.method, request.url.path, str(e), latency, trace_id)
            
            return JSONResponse(
                status_code=500,
                content={
                    "error_code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected error occurred.",
                    "request_id": trace_id,
                    "timestamp": time.time()
                }
            )


class CircuitBreakerMiddleware(BaseHTTPMiddleware):
    """
    Mock Circuit Breaker Middleware.
    Trips to OPEN after 5 failures in 30s per downstream service.
    """
    def __init__(self, app: FastAPI):
        super().__init__(app)
        self.failures: Dict[str, int] = {}

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Simplistic demonstration of circuit breaker logic mapping routes to downstreams
        service = "unknown"
        if request.url.path.startswith("/api/v1/agents"):
            service = "agent_service"
            
        if self.failures.get(service, 0) >= 5:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"error_code": "CIRCUIT_OPEN", "message": f"Service {service} is currently unavailable.", "request_id": getattr(request.state, "trace_id", "N/A"), "timestamp": time.time()}
            )
            
        response = await call_next(request)
        
        # Record failure if 5xx
        if response.status_code >= 500:
            self.failures[service] = self.failures.get(service, 0) + 1
        elif response.status_code < 500:
            self.failures[service] = 0 # Reset on success
            
        return response


def create_app() -> FastAPI:
    app = FastAPI(
        title="Industrial Knowledge Brain API Gateway",
        description="Production API Gateway for factory-ai-brain.",
        version="1.0.0"
    )

    settings = get_settings()

    # Middlewares (Order matters: outermost first)
    cors_origins = [o.strip() for o in settings.CORS_ORIGINS.split(",")]
    app.add_middleware(CORSMiddleware, allow_origins=cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
    app.add_middleware(CircuitBreakerMiddleware)
    app.add_middleware(RateLimiterMiddleware)
    app.add_middleware(JWTAuthMiddleware, settings=settings)
    app.add_middleware(TracingMiddleware)

    # Global Exception Handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error_code": "UNHANDLED_EXCEPTION",
                "message": str(exc),
                "request_id": getattr(request.state, "trace_id", "N/A"),
                "timestamp": time.time()
            }
        )

    # Include Routers (Mock imports for now, we will add the real ones)
    from backend.services.api_gateway.src.routers import query, agents
    from backend.services.api_gateway.src.websocket import realtime_handler
    
    app.include_router(query.router)
    app.include_router(agents.router)
    app.include_router(realtime_handler.router)

    @app.get("/health")
    async def health_check():
        return {"status": "ok", "timestamp": time.time()}

    return app

app = create_app()
