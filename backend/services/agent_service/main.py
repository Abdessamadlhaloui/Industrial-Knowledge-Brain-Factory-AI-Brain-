from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI

from backend.shared.infrastructure.tracing import instrument_fastapi, setup_otel_tracing, shutdown_tracing

logger = structlog.get_logger()

SERVICE_NAME = os.getenv("SERVICE_NAME", "agent_service")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_otel_tracing(service_name=SERVICE_NAME, otlp_endpoint=OTEL_ENDPOINT, environment=ENVIRONMENT)
    logger.info("service_starting", service=SERVICE_NAME, version=SERVICE_VERSION)
    yield
    await shutdown_tracing()
    logger.info("service_stopped", service=SERVICE_NAME)


app = FastAPI(
    title=f"IKB — {SERVICE_NAME}",
    version=SERVICE_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

instrument_fastapi(app)


@app.get("/health", tags=["system"])
async def health_check() -> dict:
    return {
        "status": "healthy",
        "version": SERVICE_VERSION,
        "uptime": round(time.time() - _start_time, 2),
        "service": SERVICE_NAME,
    }


@app.get("/", tags=["system"])
async def root() -> dict:
    return {"service": SERVICE_NAME, "version": SERVICE_VERSION, "docs": "/docs"}
