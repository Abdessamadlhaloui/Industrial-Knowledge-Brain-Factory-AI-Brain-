# ============================================================
# router.py  —  Router FastAPI du Telemetry Service
# ============================================================


from __future__ import annotations

import time
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Query

from backend.services.telemetry_service.src.api.rest.schemas import (
    AnomalyResponse,
    DataPointResponse,
    DetectorType,
    HealthDetailResponse,
    RecentAnomaliesResponse,
    SensorReadingRequest,
    SensorReadingResponse,
    TelemetryQueryRequest,
    TelemetryQueryResponse,
)
from backend.services.telemetry_service.src.infrastructure.time_series_db.influxdb_client import (
    AsyncInfluxDBClient,
    SensorReading,
)
from backend.services.telemetry_service.src.infrastructure.cache.redis_cache import TelemetryRedisCache
from backend.services.telemetry_service.src.infrastructure.detectors.statistical_detector import StatisticalDetector
from backend.services.telemetry_service.src.infrastructure.detectors.rule_detector import RuleDetector
from backend.services.telemetry_service.src.infrastructure.detectors.ml_detector import MLDetector

logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Création du router
# ------------------------------------------------------------

router = APIRouter(prefix="/api/v1/telemetry", tags=["Telemetry"])


# ============================================================
# Fonctions de dépendances (Dependency Injection)
# ============================================================

def get_influx(request: Request) -> AsyncInfluxDBClient:
    """Récupère le client InfluxDB initialisé dans app.state (main.py)."""
    return request.app.state.influx_client


def get_redis(request: Request) -> TelemetryRedisCache:
    """Récupère le cache Redis initialisé dans app.state (main.py)."""
    return request.app.state.redis_cache


def get_stat_detector(request: Request) -> StatisticalDetector:
    return request.app.state.stat_detector


def get_rule_detector(request: Request) -> RuleDetector:
    return request.app.state.rule_detector


def get_ml_detector(request: Request) -> MLDetector:
    return request.app.state.ml_detector


# ============================================================
# Endpoint 1 : POST /api/v1/telemetry/ingest
# ============================================================


@router.post(
    "/ingest",
    response_model=SensorReadingResponse,
    summary="Ingérer une lecture de capteur",
    description=(
        "Reçoit une lecture de capteur, l'écrit dans InfluxDB "
        "et la passe à travers la pipeline de détection d'anomalies "
        "(règles → statistique → ML)."
    ),
)
async def ingest_reading(
    body: SensorReadingRequest,                           # corps JSON validé automatiquement
    influx: AsyncInfluxDBClient = Depends(get_influx),    # client InfluxDB injecté
    redis: TelemetryRedisCache  = Depends(get_redis),     # cache Redis injecté
    rule_det: RuleDetector       = Depends(get_rule_detector),
    stat_det: StatisticalDetector = Depends(get_stat_detector),
    ml_det:   MLDetector          = Depends(get_ml_detector),
) -> SensorReadingResponse:

    # Si le client n'envoie pas de timestamp, on prend l'heure actuelle
    ts = body.timestamp or time.time()

    # --- Étape 1 : écrire dans InfluxDB ---
    reading = SensorReading(
        sensor_id=body.sensor_id,
        machine_id=body.machine_id,
        value=body.value,
        timestamp=ts,
    )
    try:
        await influx.write_batch([reading])
    except Exception as exc:
        logger.error("Échec écriture InfluxDB: %s", exc)
        raise HTTPException(status_code=503, detail="InfluxDB indisponible")

    # --- Étape 2 : pipeline de détection (règles → stat → ML) ---
   
    anomaly       = None
    detector_used : Optional[DetectorType] = None

    # 2a. Règles fixes (le plus rapide, O(1))
    anomaly = await rule_det.detect(body.sensor_id, body.machine_id, body.value, ts)
    if anomaly:
        detector_used = DetectorType.RULE

    # 2b. Détection statistique (Z-score) si pas de règle violée
    if not anomaly:
        anomaly = await stat_det.detect(body.sensor_id, body.machine_id, body.value, ts)
        if anomaly:
            detector_used = DetectorType.STATISTICAL

    # 2c. Détection ML (LSTM) — le plus coûteux, en dernier recours
    if not anomaly:
        anomaly = await ml_det.detect(body.sensor_id, body.machine_id, body.value, ts)
        if anomaly:
            detector_used = DetectorType.ML

    # --- Étape 3 : construire la réponse ---
    anomaly_resp: Optional[AnomalyResponse] = None
    if anomaly:
        anomaly_resp = AnomalyResponse(
            sensor_id=anomaly.sensor_id,
            machine_id=anomaly.machine_id,
            value=anomaly.value,
            expected_range=anomaly.expected_range,
            z_score=anomaly.z_score,
            severity=anomaly.severity,
            timestamp=anomaly.timestamp,
        )

    return SensorReadingResponse(
        status="processed",
        sensor_id=body.sensor_id,
        machine_id=body.machine_id,
        value=body.value,
        timestamp=ts,
        anomaly_detected=anomaly is not None,
        anomaly=anomaly_resp,
        detector_used=detector_used,
    )


# ============================================================
# Endpoint 2 : POST /api/v1/telemetry/query
# ============================================================

@router.post(
    "/query",
    response_model=TelemetryQueryResponse,
    summary="Interroger les données historiques d'une machine",
    description=(
        "Retourne les données de séries temporelles depuis InfluxDB "
        "pour une machine et une métrique données, sur une plage de temps."
    ),
)
async def query_telemetry(
    body: TelemetryQueryRequest,
    influx: AsyncInfluxDBClient = Depends(get_influx),
) -> TelemetryQueryResponse:

    try:
        raw_points = await influx.query_range(
            machine_id=body.machine_id,
            metric_name=body.metric_name,
            start_time=body.start_time,
            end_time=body.end_time,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    # Convertir les dicts renvoyés par influxdb_client en schémas Pydantic
    data_points = [
        DataPointResponse(
            timestamp=p["timestamp"],
            metric_name=p["metric_name"],
            value=p["value"],
        )
        for p in raw_points
    ]

    return TelemetryQueryResponse(
        machine_id=body.machine_id,
        metric_name=body.metric_name,
        start_time=body.start_time,
        end_time=body.end_time,
        count=len(data_points),
        data_points=data_points,
    )


# ============================================================
# Endpoint 3 : GET /api/v1/telemetry/anomalies/{machine_id}
# ============================================================

@router.get(
    "/anomalies/{machine_id}",
    response_model=RecentAnomaliesResponse,
    summary="Lire les anomalies récentes d'une machine",
    description=(
        "Retourne les dernières anomalies détectées pour une machine, "
        "lues depuis le cache Redis (résultat quasi-instantané)."
    ),
)
async def get_recent_anomalies(
    machine_id: str,                                         # extrait directement de l'URL
    limit: int = Query(default=20, ge=1, le=100,             # paramètre ?limit=N dans l'URL
                       description="Nombre max d'anomalies à retourner"),
    redis: TelemetryRedisCache = Depends(get_redis),
) -> RecentAnomaliesResponse:

    try:
        # get_recent_anomalies retourne une liste de dicts (stockés en JSON dans Redis)
        anomalies = await redis.get_recent_anomalies(machine_id, limit=limit)
    except Exception as exc:
        logger.error("Erreur lecture Redis pour machine %s: %s", machine_id, exc)
        raise HTTPException(status_code=503, detail="Cache Redis indisponible")

    return RecentAnomaliesResponse(
        machine_id=machine_id,
        count=len(anomalies),
        anomalies=anomalies,
    )


# ============================================================
# Endpoint 4 : GET /api/v1/telemetry/health/detail
# ============================================================

@router.get(
    "/health/detail",
    response_model=HealthDetailResponse,
    summary="État détaillé des composants du service",
    description=(
        "Retourne l'état de chaque dépendance (InfluxDB, Redis, "
        "stream processor Kafka). Utile pour le monitoring."
    ),
)
async def health_detail(
    request: Request,
    influx: AsyncInfluxDBClient = Depends(get_influx),
    redis: TelemetryRedisCache  = Depends(get_redis),
) -> HealthDetailResponse:

    # Vérifier InfluxDB — on considère qu'il est UP si le client existe
    influx_ok = influx.client is not None

    # Vérifier Redis — on tente un PING
    # Note : dans TelemetryRedisCache, le client Redis s'appelle self.redis (pas self.client)
    redis_ok = False
    try:
        await redis.redis.ping()
        redis_ok = True
    except Exception:
        pass

    # Vérifier le stream processor — stocké dans app.state par main.py
    processor = getattr(request.app.state, "stream_processor", None)
    processor_ok = processor is not None and getattr(processor, "_running", False)

    # Si au moins une dépendance est KO, le statut global est "degraded"
    overall = "healthy" if (influx_ok and redis_ok and processor_ok) else "degraded"

    return HealthDetailResponse(
        status=overall,
        influxdb=influx_ok,
        redis=redis_ok,
        processor=processor_ok,
    )
