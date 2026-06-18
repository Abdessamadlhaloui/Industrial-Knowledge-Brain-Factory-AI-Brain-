"""
anomaly.py — Modèle métier "Anomalie"
======================================
Couche DOMAIN = les concepts métier purs, sans aucune dépendance technique
(pas de Redis, pas d'InfluxDB, pas de FastAPI). Juste les règles du métier.

Avant : `Anomaly` était défini dans `infrastructure/detectors/statistical_detector.py`.
C'est un problème d'architecture : un concept métier ne doit pas vivre dans
une couche d'infrastructure. Si demain on change StatisticalDetector pour
autre chose, on ne doit pas perdre la définition d'Anomaly.

Règle d'architecture en couches (Clean Architecture / Hexagonal) :
    domain          → ne dépend de RIEN d'autre dans le projet
    application      → peut dépendre de domain
    infrastructure   → peut dépendre de domain et application

Tous les détecteurs (rule, statistical, ml) importent maintenant Anomaly
et Severity depuis ce fichier, au lieu de le redéfinir ou de l'importer
depuis statistical_detector.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Severity(str, Enum):
    """
    Niveaux de sévérité d'une anomalie, du moins grave au plus grave.

    Hériter de `str` permet à Severity.HIGH == "HIGH" → compatible
    avec le code existant qui compare des strings ("HIGH", "CRITICAL"...)
    sans tout casser.
    """
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass(frozen=True)
class Anomaly:
    """
    Représente une anomalie détectée sur un capteur.

    frozen=True → immuable une fois créée. Une anomalie est un fait
    qui s'est produit à un instant T ; on ne doit jamais la modifier
    après coup, seulement en créer une nouvelle si besoin.

    Attributes:
        sensor_id:      Identifiant du capteur concerné
        machine_id:      Identifiant de la machine parente
        value:           Valeur qui a déclenché l'anomalie
        expected_range:  Plage de valeurs attendue (ex: "72.3 ± 4.1" ou "above 80.0")
        z_score:         Écart en nombre d'écarts-types (0.0 si non applicable, ex: règles fixes)
        severity:        Niveau de gravité
        timestamp:       Timestamp Unix (secondes) de la lecture anormale
    """
    sensor_id: str
    machine_id: str
    value: float
    expected_range: str
    z_score: float
    severity: str  # garder str (pas Severity) pour compatibilité avec le code existant
    timestamp: float

    def is_critical(self) -> bool:
        """Règle métier : une anomalie nécessite-t-elle une escalade immédiate ?"""
        return self.severity in (Severity.HIGH.value, Severity.CRITICAL.value)
