"""
main.py — Point d'entrée du Telemetry Service
==============================================
Ce fichier fait 3 choses :

  1. Crée l'application FastAPI
  2. Dans `lifespan` : initialise TOUS les composants au démarrage
     et les stocke dans `app.state` pour que le router puisse les utiliser
  3. Branche le router sur l'application

Pourquoi `app.state` ?
-----------------------
FastAPI ne supporte pas l'injection directe d'objets "lourds" (connexions DB,
clients HTTP...) via les dépendances sans un endroit pour les stocker.
`app.state` est un namespace partagé accessible depuis n'importe quel endpoint
via `request.app.state`. C'est le pattern standard FastAPI pour les singletons.

Cycle de vie (lifespan) :
--------------------------
  Démarrage  → tout ce qui est AVANT le `yield` (connexions, init...)
  Requêtes   → le `yield` — FastAPI sert les requêtes normalement
  Arrêt      → tout ce qui est APRÈS le `yield` (fermeture propre)
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI
from redis.asyncio import Redis

from backend.shared.infrastructure.tracing import (
    instrument_fastapi,
    setup_otel_tracing,
    shutdown_tracing,
)
from backend.services.telemetry_service.src.api.rest.router import router
from backend.services.telemetry_service.src.infrastructure.time_series_db.influxdb_client import (
    AsyncInfluxDBClient,
)
from backend.services.telemetry_service.src.infrastructure.cache.redis_cache import (
    TelemetryRedisCache,
)
from backend.services.telemetry_service.src.infrastructure.detectors.statistical_detector import (
    StatisticalDetector,
)
from backend.services.telemetry_service.src.infrastructure.detectors.rule_detector import (
    RuleDetector,
)
from backend.services.telemetry_service.src.infrastructure.detectors.ml_detector import (
    MLDetector,
)
from backend.services.telemetry_service.src.application.stream_processor import (
    TelemetryStreamProcessor,
)

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Variables d'environnement — toutes les configs viennent de l'environnement
# (fichier .env en dev, Kubernetes secrets en prod)
# ---------------------------------------------------------------------------

SERVICE_NAME    = os.getenv("SERVICE_NAME",    "telemetry_service")
SERVICE_VERSION = os.getenv("SERVICE_VERSION", "0.1.0")
ENVIRONMENT     = os.getenv("ENVIRONMENT",     "development")

# OpenTelemetry (tracing distribué via Jaeger)
OTEL_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4317")

# InfluxDB
INFLUX_URL    = os.getenv("INFLUXDB_URL",    "http://influxdb:8086")
INFLUX_TOKEN  = os.getenv("INFLUXDB_TOKEN",  "dev-token")
INFLUX_ORG    = os.getenv("INFLUXDB_ORG",    "ikb")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "telemetry_raw")

# Redis
REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB   = int(os.getenv("REDIS_DB",   "0"))

# Kafka (pour le stream processor)
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

# Détecteurs
Z_THRESHOLD       = float(os.getenv("STAT_Z_THRESHOLD",       "3.0"))
MIN_SAMPLE_COUNT  = int(os.getenv("STAT_MIN_SAMPLE_COUNT",    "30"))
ML_RECON_THRESHOLD = float(os.getenv("ML_RECONSTRUCTION_THRESHOLD", "0.05"))

_start_time = time.time()


# ---------------------------------------------------------------------------
# Lifespan — démarrage et arrêt propre du service
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Contexte de vie de l'application.
    Tout ce qui est avant `yield` s'exécute AU DÉMARRAGE.
    Tout ce qui est après `yield` s'exécute À L'ARRÊT.
    """

    # ── 1. Tracing OpenTelemetry ─────────────────────────────────────────────
    setup_otel_tracing(
        service_name=SERVICE_NAME,
        otlp_endpoint=OTEL_ENDPOINT,
        environment=ENVIRONMENT,
    )
    logger.info("service_starting", service=SERVICE_NAME, version=SERVICE_VERSION)

    # ── 2. Connexion Redis ───────────────────────────────────────────────────
    # On crée un seul client Redis partagé par tous les composants.
    # decode_responses=False → on gère nous-mêmes le décodage bytes→str
    redis_client = Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=False)
    redis_cache  = TelemetryRedisCache(redis_client)

    # Vérification que Redis répond (fail-fast au démarrage)
    try:
        await redis_client.ping()
        logger.info("redis_connected", host=REDIS_HOST, port=REDIS_PORT)
    except Exception as exc:
        logger.warning("redis_unavailable", error=str(exc))
        # On ne lève pas d'exception ici : le service peut démarrer en mode dégradé

    # ── 3. Connexion InfluxDB ────────────────────────────────────────────────
    influx_client = AsyncInfluxDBClient(
        url=INFLUX_URL,
        token=INFLUX_TOKEN,
        org=INFLUX_ORG,
        bucket=INFLUX_BUCKET,
    )
    try:
        await influx_client.connect()
        logger.info("influxdb_connected", url=INFLUX_URL, bucket=INFLUX_BUCKET)
    except Exception as exc:
        logger.warning("influxdb_unavailable", error=str(exc))

    # ── 4. Initialisation des détecteurs d'anomalies ─────────────────────────
    # Ordre d'instanciation important :
    #   StatisticalDetector a besoin de redis_cache
    #   RuleDetector        a besoin de redis_cache
    #   MLDetector          a besoin de StatisticalDetector comme fallback

    stat_detector = StatisticalDetector(
        redis_cache=redis_cache,
        z_threshold=Z_THRESHOLD,
        min_sample_count=MIN_SAMPLE_COUNT,
    )
    rule_detector = RuleDetector(redis_cache=redis_cache)
    ml_detector   = MLDetector(
        fallback_detector=stat_detector,
        reconstruction_threshold=ML_RECON_THRESHOLD,
    )

    logger.info(
        "detectors_initialized",
        z_threshold=Z_THRESHOLD,
        min_samples=MIN_SAMPLE_COUNT,
        ml_threshold=ML_RECON_THRESHOLD,
    )

    # ── 5. Démarrage du Stream Processor Kafka ───────────────────────────────
    # Le stream processor tourne en arrière-plan (asyncio task).
    # Il consomme Kafka → détecte anomalies → publie alertes.
    stream_processor = TelemetryStreamProcessor(
        kafka_bootstrap_servers=KAFKA_BOOTSTRAP,
        influx_client=influx_client,
        redis_cache=redis_cache,
        stat_detector=stat_detector,
        rule_detector=rule_detector,
        ml_detector=ml_detector,
    )
    try:
        await stream_processor.start()
        logger.info("stream_processor_started", kafka=KAFKA_BOOTSTRAP)
    except Exception as exc:
        logger.warning("stream_processor_failed", error=str(exc))
        # Démarrage en mode dégradé : l'API REST reste disponible sans Kafka

    # ── 6. Stocker tout dans app.state ───────────────────────────────────────
    # C'est ici que le router récupèrera les composants via request.app.state
    app.state.redis_cache       = redis_cache
    app.state.influx_client     = influx_client
    app.state.stat_detector     = stat_detector
    app.state.rule_detector     = rule_detector
    app.state.ml_detector       = ml_detector
    app.state.stream_processor  = stream_processor

    logger.info("service_ready", service=SERVICE_NAME)

    # ── yield : FastAPI sert les requêtes à partir d'ici ────────────────────
    yield

    # ── ARRÊT — exécuté quand le serveur reçoit SIGTERM (ex: kubectl stop) ──
    logger.info("service_stopping", service=SERVICE_NAME)

    await stream_processor.stop()
    logger.info("stream_processor_stopped")

    await influx_client.close()
    logger.info("influxdb_disconnected")

    await redis_client.aclose()
    logger.info("redis_disconnected")

    await shutdown_tracing()
    logger.info("service_stopped", service=SERVICE_NAME)


# ---------------------------------------------------------------------------
# Création de l'application FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title=f"IKB — {SERVICE_NAME}",
    version=SERVICE_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Brancher le router métier (tous les endpoints /api/v1/telemetry/...)
app.include_router(router)

# Instrumentation OpenTelemetry (tracing automatique de toutes les requêtes)
instrument_fastapi(app)


# ---------------------------------------------------------------------------
# Endpoints système (health check basique, hors router)
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
async def health_check() -> dict:
    """
    Sonde de vie minimale — répond toujours 200 si le processus tourne.
    Utilisé par Kubernetes pour les liveness probes.
    Pour un état détaillé des composants, utiliser GET /api/v1/telemetry/health/detail
    """
    return {
        "status":  "healthy",
        "version": SERVICE_VERSION,
        "uptime":  round(time.time() - _start_time, 2),
        "service": SERVICE_NAME,
    }


@app.get("/", tags=["system"])
async def root() -> dict:
    return {
        "service": SERVICE_NAME,
        "version": SERVICE_VERSION,
        "docs":    "/docs",
    }
