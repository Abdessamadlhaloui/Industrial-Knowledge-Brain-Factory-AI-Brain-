from __future__ import annotations

import logging
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.semconv.resource import ResourceAttributes

logger = logging.getLogger(__name__)


def setup_otel_tracing(service_name: str) -> None:
    """Configure OpenTelemetry with Jaeger OTLP exporter.

    All security-sensitive and environment-specific values are driven by
    environment variables so no credentials or transport flags are baked
    into the source.

    Environment variables
    ---------------------
    OTEL_EXPORTER_OTLP_ENDPOINT   gRPC collector address
                                  (default: "http://jaeger:4317")
    OTEL_EXPORTER_OTLP_INSECURE   "true"  → plaintext (local dev)
                                  "false" → TLS (production)
                                  (default: "true")
    OTEL_SERVICE_NAME             Override the service name at deploy time.
    SERVICE_VERSION               Semantic version string emitted in spans
                                  (default: "unknown")
    DEPLOY_ENV                    Deployment environment label
                                  (default: "local")

    Args:
        service_name: Fallback service name when OTEL_SERVICE_NAME is unset.
    """
    # ------------------------------------------------------------------ #
    # 1. Resolve all configuration from the environment                   #
    # ------------------------------------------------------------------ #
    otlp_endpoint: str = os.environ.get(
        "OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317"
    )

    # Explicit string comparison keeps intent readable and avoids
    # truthy/falsy pitfalls with empty-string env vars.
    is_insecure: bool = (
        os.environ.get("OTEL_EXPORTER_OTLP_INSECURE", "true").strip().lower() == "true"
    )

    resolved_service_name: str = os.environ.get("OTEL_SERVICE_NAME", service_name)
    service_version: str = os.environ.get("SERVICE_VERSION", "unknown")
    deploy_env: str = os.environ.get("DEPLOY_ENV", "local")

    # ------------------------------------------------------------------ #
    # 2. Build the Resource (identity metadata attached to every span)    #
    # ------------------------------------------------------------------ #
    resource = Resource.create(
        {
            ResourceAttributes.SERVICE_NAME: resolved_service_name,
            ResourceAttributes.SERVICE_VERSION: service_version,
            ResourceAttributes.DEPLOYMENT_ENVIRONMENT: deploy_env,
            "service.namespace": "factory-ai-brain",
            # Explicit key mirrors the semconv name used by many backends.
            "deployment.environment": deploy_env,
        }
    )

    # ------------------------------------------------------------------ #
    # 3. Build the TracerProvider with a TLS-aware OTLP exporter          #
    # ------------------------------------------------------------------ #
    provider = TracerProvider(resource=resource)

    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        insecure=is_insecure,   # ← no longer hardcoded; ops sets OTEL_EXPORTER_OTLP_INSECURE=false in cloud
    )
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    # Console exporter: opt-in only, never enabled by default.
    if os.environ.get("OTEL_ENABLE_CONSOLE_EXPORT", "false").strip().lower() == "true":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)

    logger.info(
        "OpenTelemetry tracing initialised — service=%s version=%s "
        "endpoint=%s insecure=%s env=%s",
        resolved_service_name,
        service_version,
        otlp_endpoint,
        is_insecure,
        deploy_env,
    )


def instrument_fastapi(app: Any) -> None:
    """Instrument a FastAPI application with OpenTelemetry."""
    FastAPIInstrumentor.instrument_app(
        app,
        excluded_urls="health,healthz,ready,metrics",
    )
    logger.info("FastAPI instrumented with OpenTelemetry.")


def get_tracer(name: str) -> trace.Tracer:
    """Return a named tracer from the global provider."""
    return trace.get_tracer(name)


async def shutdown_tracing() -> None:
    """Flush and shut down the tracer provider gracefully."""
    provider = trace.get_tracer_provider()
    if isinstance(provider, TracerProvider):
        provider.shutdown()
        logger.info("OpenTelemetry tracer provider shut down.")