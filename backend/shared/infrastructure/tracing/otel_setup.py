from __future__ import annotations

import logging
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.semconv.resource import ResourceAttributes

logger = logging.getLogger(__name__)


def setup_otel_tracing(
    service_name: str,
    otlp_endpoint: str = "http://jaeger:4317",
    environment: str = "development",
    enable_console_export: bool = False,
    sample_rate: float = 1.0,
) -> TracerProvider:
    """Configure OpenTelemetry with Jaeger OTLP exporter.

    Args:
        service_name: Name of the service (used in spans/traces).
        otlp_endpoint: OTLP gRPC endpoint for the Jaeger collector.
        environment: Deployment environment label.
        enable_console_export: Also export spans to stdout (dev only).
        sample_rate: Fraction of traces to sample (1.0 = all).

    Returns:
        Configured TracerProvider instance.
    """
    resource = Resource.create(
        {
            ResourceAttributes.SERVICE_NAME: service_name,
            ResourceAttributes.SERVICE_VERSION: "0.1.0",
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: environment,
            "service.namespace": "factory-ai-brain",
        }
    )

    provider = TracerProvider(resource=resource)

    otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    if enable_console_export:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)

    logger.info(
        "OpenTelemetry tracing initialized — service=%s endpoint=%s env=%s",
        service_name,
        otlp_endpoint,
        environment,
    )
    return provider


def instrument_fastapi(app: Any) -> None:
    """Instrument a FastAPI application with OpenTelemetry."""
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="health,healthz,ready,metrics",
    )
    logger.info("FastAPI instrumented with OpenTelemetry.")


def get_tracer(name: str) -> trace.Tracer:
    """Get a named tracer from the global provider."""
    return trace.get_tracer(name)


async def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider."""
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.shutdown()
        logger.info("OpenTelemetry tracer provider shut down.")
