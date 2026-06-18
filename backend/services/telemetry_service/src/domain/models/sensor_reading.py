"""
sensor_reading.py — Modèle métier "Lecture de capteur"
=========================================================
Même logique que anomaly.py : ce concept était défini dans
`infrastructure/time_series_db/influxdb_client.py`, couplé à InfluxDB
alors que "une lecture de capteur" n'a rien à voir avec InfluxDB en soi —
c'est juste une donnée métier.

`influxdb_client.py` garde sa propre classe SensorReading par simplicité
de compatibilité (pour ne pas casser le code déjà fonctionnel), mais tout
NOUVEAU code (nouveaux endpoints, nouveaux détecteurs) devrait utiliser
celle-ci comme référence métier.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorReading:
    """
    Représente une lecture brute d'un capteur industriel.

    Attributes:
        sensor_id:   Identifiant unique du capteur (ex: "temp_motor_A1")
        machine_id:  Identifiant de la machine parente (ex: "machine_001")
        value:       Valeur mesurée
        timestamp:   Timestamp Unix (secondes) de la mesure
    """
    sensor_id: str
    machine_id: str
    value: float
    timestamp: float
