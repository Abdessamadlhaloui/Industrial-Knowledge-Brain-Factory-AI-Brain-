# ============================================================
# schemas.py  —  Schémas Pydantic du Telemetry Service
# ============================================================


from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------
# Enums — valeurs fixes acceptées par l'API
# ------------------------------------------------------------

class SeverityLevel(str, Enum):
    """Niveaux de sévérité des anomalies, du moins grave au plus grave."""
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class DetectorType(str, Enum):
    """Quel algorithme a détecté l'anomalie."""
    RULE        = "rule"         # règles fixes (seuil absolu)
    STATISTICAL = "statistical"  # Z-score (écart à la moyenne)
    ML          = "ml"           # LSTM Autoencoder (apprentissage profond)


# ------------------------------------------------------------
# Schémas d'ENTRÉE (ce que le client envoie à l'API)
# ------------------------------------------------------------

class SensorReadingRequest(BaseModel):
    """
    Corps d'une requête POST /ingest.
    Représente une lecture de capteur envoyée manuellement à l'API
    (utile pour les tests ou les lectures hors Kafka).
    """
    sensor_id:  str   = Field(..., description="Identifiant unique du capteur (ex: 'temp_motor_A1')")
    machine_id: str   = Field(..., description="Identifiant de la machine parente (ex: 'machine_001')")
    value:      float = Field(..., description="Valeur mesurée par le capteur")
    timestamp:  Optional[float] = Field(
        default=None,
        description="Timestamp Unix (secondes). Si absent, le serveur utilise l'heure actuelle."
    )

    # Exemple affiché dans Swagger /docs
    model_config = {
        "json_schema_extra": {
            "example": {
                "sensor_id":  "temp_motor_A1",
                "machine_id": "machine_001",
                "value":      87.4,
                "timestamp":  1718700000.0
            }
        }
    }


class TelemetryQueryRequest(BaseModel):
    """
    Corps d'une requête POST /query.
    Permet d'interroger l'historique de données d'une machine.
    """
    machine_id:  str = Field(..., description="Machine à interroger")
    metric_name: str = Field(..., description="Nom du champ à lire dans InfluxDB (ex: 'value')")
    start_time:  str = Field(
        default="-1h",
        description="Début de la plage temporelle. Format Flux : '-1h', '-24h', ou RFC3339."
    )
    end_time: str = Field(
        default="now()",
        description="Fin de la plage temporelle. Format Flux : 'now()' ou RFC3339."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "machine_id":  "machine_001",
                "metric_name": "value",
                "start_time":  "-6h",
                "end_time":    "now()"
            }
        }
    }


# ------------------------------------------------------------
# Schémas de SORTIE (ce que l'API renvoie au client)
# ------------------------------------------------------------

class AnomalyResponse(BaseModel):
    """
    Représentation JSON d'une anomalie détectée.
    Correspond au dataclass `Anomaly` défini dans statistical_detector.py.
    """
    sensor_id:      str          = Field(..., description="Capteur concerné")
    machine_id:     str          = Field(..., description="Machine concernée")
    value:          float        = Field(..., description="Valeur qui a déclenché l'alerte")
    expected_range: str          = Field(..., description="Plage attendue (ex: '72.3 ± 4.1')")
    z_score:        float        = Field(..., description="Écart en nombre d'écarts-types (0 = normal)")
    severity:       SeverityLevel = Field(..., description="Niveau de sévérité")
    timestamp:      float        = Field(..., description="Timestamp Unix de la lecture")


class SensorReadingResponse(BaseModel):
    """
    Réponse renvoyée après un POST /ingest.
    Indique si une anomalie a été détectée sur la lecture ingérée.
    """
    status:          str                    = Field(..., description="'processed' si OK")
    sensor_id:       str
    machine_id:      str
    value:           float
    timestamp:       float
    anomaly_detected: bool                  = Field(..., description="True si au moins un détecteur a trouvé une anomalie")
    anomaly:         Optional[AnomalyResponse] = Field(
        default=None,
        description="Détails de l'anomalie, si détectée"
    )
    detector_used:   Optional[DetectorType] = Field(
        default=None,
        description="Quel détecteur a trouvé l'anomalie"
    )


class DataPointResponse(BaseModel):
    """Un point de données historique retourné par /query."""
    timestamp:   str   = Field(..., description="Timestamp ISO 8601")
    metric_name: str
    value:       float


class TelemetryQueryResponse(BaseModel):
    """Réponse d'un POST /query — liste de points de données historiques."""
    machine_id:  str
    metric_name: str
    start_time:  str
    end_time:    str
    count:       int                    = Field(..., description="Nombre de points retournés")
    data_points: List[DataPointResponse] = Field(default_factory=list)


class RecentAnomaliesResponse(BaseModel):
    """
    Réponse d'un GET /anomalies/{machine_id}.
    Retourne les dernières anomalies mises en cache dans Redis pour une machine.
    """
    machine_id:   str
    count:        int
    anomalies:    List[Dict[str, Any]] = Field(default_factory=list)


class HealthDetailResponse(BaseModel):
    """
    Réponse d'un GET /health/detail — état de chaque composant du service.
    Utile pour le monitoring (Grafana, Kubernetes liveness probes).
    """
    status:     str  = Field(..., description="'healthy' ou 'degraded'")
    influxdb:   bool = Field(..., description="True si la connexion InfluxDB est active")
    redis:      bool = Field(..., description="True si Redis répond")
    processor:  bool = Field(..., description="True si le stream processor Kafka tourne")
