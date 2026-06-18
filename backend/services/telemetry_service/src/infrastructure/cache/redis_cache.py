"""
redis_cache.py — Cache Redis du Telemetry Service
==================================================
Ce fichier centralise TOUTES les interactions avec Redis.
Redis est une base de données en mémoire ultra-rapide (< 1ms).
On l'utilise ici pour 4 choses :

  1. Déduplication des anomalies  → éviter de créer 1000 alertes pour le même problème
  2. Statistiques par capteur      → stocker moyenne/écart-type en temps réel (Welford)
  3. Log glissant d'anomalies      → les 100 dernières anomalies par machine
  4. Cache des règles métier       → éviter des requêtes PostgreSQL à chaque lecture

Structure des clés Redis utilisées :
  ikb:anomaly_dedup:{sensor_id}:{severity}     → déduplication (TTL 5 min)
  ikb:sensor_stats:{sensor_id}                 → stats Welford {count, mean, M2}
  ikb:anomalies:recent:{machine_id}            → ZSET trié par timestamp
  ikb:rule:{rule_id}                           → règle sérialisée en JSON
  ikb:machine_rules:{machine_id}               → SET d'IDs de règles
  ikb:baseline:ema:{sensor_id}                 → EMA baseline flottante
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any, Dict, List, Optional

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class TelemetryRedisCache:
    """
    Cache Redis pour le service de télémétrie industrielle.

    Injection :
        redis_client = Redis(host=..., port=..., decode_responses=False)
        cache = TelemetryRedisCache(redis_client)
    """

    def __init__(self, redis_client: Redis) -> None:
        # Note : l'attribut s'appelle `.redis`, pas `.client`
        # Le router et les détecteurs doivent utiliser cache.redis pour accéder au client brut.
        self.redis = redis_client

    # =========================================================================
    # 1. DÉDUPLICATION DES ANOMALIES
    # =========================================================================

    async def is_duplicate_anomaly(
        self, sensor_id: str, severity: str, window_seconds: int = 300
    ) -> bool:
        """
        Vérifie si la même anomalie (sensor + sévérité) a déjà été signalée
        dans la fenêtre temporelle donnée.

        Fonctionnement :
          - On essaie de SET une clé avec NX (Set if Not eXists) + EX (expire)
          - Si la clé existait déjà → c'est un doublon → retourne True
          - Si on vient de la créer → c'est nouvelle → retourne False

        Args:
            sensor_id:       Identifiant du capteur
            severity:        Sévérité (LOW / MEDIUM / HIGH / CRITICAL)
            window_seconds:  Durée de la fenêtre de déduplication en secondes

        Returns:
            True si l'anomalie est un doublon (à ignorer), False sinon.
        """
        key = f"ikb:anomaly_dedup:{sensor_id}:{severity}"
        is_new = await self.redis.set(key, "1", nx=True, ex=window_seconds)
        return not is_new  # is_new=True → clé créée → pas de doublon

    # =========================================================================
    # 2. STATISTIQUES WELFORD PAR CAPTEUR
    # =========================================================================
    # L'algorithme de Welford met à jour moyenne et variance de façon
    # incrémentale en O(1), sans avoir à stocker tout l'historique.
    # C'est ce qu'utilise StatisticalDetector pour calculer le Z-score.
    #
    # On stocke 3 valeurs par capteur : count, mean, M2
    # où M2 = somme cumulée des (xi - mean)² → variance = M2 / count
    # =========================================================================

    async def update_sensor_stats(self, sensor_id: str, value: float) -> None:
        """
        Met à jour les statistiques de Welford pour un capteur (O(1)).
        Appelé à chaque nouvelle lecture, AVANT get_sensor_stats.

        Args:
            sensor_id: Identifiant unique du capteur
            value:     Nouvelle valeur lue
        """
        key = f"ikb:sensor_stats:{sensor_id}"

        # Lire les stats actuelles (ou partir de zéro)
        raw = await self.redis.get(key)
        if raw:
            stats = json.loads(raw)
            count: int   = stats["count"]
            mean:  float = stats["mean"]
            M2:    float = stats["M2"]
        else:
            count = 0
            mean  = 0.0
            M2    = 0.0

        # Algorithme de Welford — une passe, stable numériquement
        count  += 1
        delta   = value - mean
        mean   += delta / count
        delta2  = value - mean
        M2     += delta * delta2

        # Réécrire dans Redis (pas de TTL → les stats persistent indéfiniment)
        await self.redis.set(key, json.dumps({"count": count, "mean": mean, "M2": M2}))

    async def get_sensor_stats(self, sensor_id: str) -> Optional[Dict[str, float]]:
        """
        Retourne les statistiques actuelles d'un capteur.
        Appelé par StatisticalDetector après update_sensor_stats.

        Returns:
            Dict {"count": int, "mean": float, "std_dev": float}
            ou None si le capteur n'a jamais été vu.
        """
        key = f"ikb:sensor_stats:{sensor_id}"
        raw = await self.redis.get(key)

        if raw is None:
            return None

        stats = json.loads(raw)
        count: int   = stats["count"]
        mean:  float = stats["mean"]
        M2:    float = stats["M2"]

        # Variance de population (on a toutes les données, pas un échantillon)
        variance = M2 / count if count > 1 else 0.0
        std_dev  = math.sqrt(variance)

        return {"count": count, "mean": mean, "std_dev": std_dev}

    # =========================================================================
    # 3. LOG GLISSANT DES ANOMALIES PAR MACHINE
    # =========================================================================

    async def add_recent_anomaly(
        self, machine_id: str, anomaly_data: Dict[str, Any], timestamp: float
    ) -> None:
        """
        Ajoute une anomalie dans le Sorted Set Redis de la machine.
        Le score est le timestamp → tri chronologique automatique.
        Garde seulement les 100 plus récentes.

        Args:
            machine_id:   Identifiant de la machine
            anomaly_data: Dict représentant l'anomalie (issu de Anomaly.__dict__)
            timestamp:    Timestamp Unix (utilisé comme score dans le ZSET)
        """
        key = f"ikb:anomalies:recent:{machine_id}"

        # zadd(key, {membre: score}) — le membre est sérialisé en JSON
        await self.redis.zadd(key, {json.dumps(anomaly_data): timestamp})

        # Garder seulement les 100 plus récentes (supprimer les plus anciennes)
        card = await self.redis.zcard(key)
        if card > 100:
            # zremrangebyrank supprime les éléments du plus ancien (index 0)
            # jusqu'à l'index card-101, gardant les 100 derniers
            await self.redis.zremrangebyrank(key, 0, card - 101)

    async def get_recent_anomalies(
        self, machine_id: str, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Retourne les `limit` anomalies les plus récentes pour une machine.
        Appelé par le router GET /anomalies/{machine_id}.

        Args:
            machine_id: Identifiant de la machine
            limit:      Nombre max d'anomalies à retourner (défaut 20, max 100)

        Returns:
            Liste de dicts, triés du plus récent au plus ancien.
        """
        key = f"ikb:anomalies:recent:{machine_id}"

        # zrange avec rev=True et limit → les N plus récents (score le plus haut)
        raw_items = await self.redis.zrange(
            key,
            start=0,
            end=limit - 1,
            desc=True,   # du plus récent au plus ancien
        )

        result: List[Dict[str, Any]] = []
        for item in raw_items:
            try:
                # item peut être bytes ou str selon decode_responses
                decoded = item.decode("utf-8") if isinstance(item, bytes) else item
                result.append(json.loads(decoded))
            except (json.JSONDecodeError, AttributeError) as exc:
                logger.warning("Entrée Redis malformée ignorée: %s", exc)

        return result

    # =========================================================================
    # 4. CACHE DES RÈGLES MÉTIER
    # =========================================================================

    async def cache_rule(
        self, rule_id: str, rule_data: Dict[str, Any], ttl_seconds: int = 300
    ) -> None:
        """
        Met en cache une règle de seuil pour évaluation O(1).
        TTL = 5 minutes par défaut (les règles changent rarement).

        Args:
            rule_id:     Identifiant unique de la règle
            rule_data:   Dict {"sensor_id", "condition", "threshold", "severity"}
            ttl_seconds: Durée de vie du cache
        """
        key = f"ikb:rule:{rule_id}"
        await self.redis.setex(key, ttl_seconds, json.dumps(rule_data))

    async def get_machine_rules(self, machine_id: str) -> List[Dict[str, Any]]:
        """
        Retourne toutes les règles actives pour une machine depuis Redis.
        Si le cache est vide, retourne une liste vide (le RuleDetector ne bloque pas).

        Args:
            machine_id: Identifiant de la machine

        Returns:
            Liste de règles, potentiellement vide.
        """
        try:
            # Le SET ikb:machine_rules:{machine_id} contient les IDs des règles
            rule_ids = await self.redis.smembers(f"ikb:machine_rules:{machine_id}")
            if not rule_ids:
                return []

            rules: List[Dict[str, Any]] = []
            for rule_id_raw in rule_ids:
                rule_id = (
                    rule_id_raw.decode("utf-8")
                    if isinstance(rule_id_raw, bytes)
                    else rule_id_raw
                )
                rule_data = await self.redis.get(f"ikb:rule:{rule_id}")
                if rule_data is not None:
                    rules.append(json.loads(rule_data))

            return rules

        except Exception as exc:
            logger.error(
                "Impossible de charger les règles pour machine_id=%s : %s",
                machine_id,
                exc,
            )
            raise

    # =========================================================================
    # 5. BASELINE EMA (Exponential Moving Average)
    # =========================================================================

    async def update_ema_baseline(
        self, sensor_id: str, value: float, alpha: float = 0.05
    ) -> float:
        """
        Met à jour et retourne la moyenne exponentielle (EMA) pour un capteur.
        Formule : EMA_new = alpha * value + (1 - alpha) * EMA_old

        Args:
            sensor_id: Identifiant du capteur
            value:     Nouvelle valeur
            alpha:     Facteur de lissage (0.05 = très lent, 0.3 = plus réactif)

        Returns:
            Nouvelle valeur EMA.
        """
        key = f"ikb:baseline:ema:{sensor_id}"
        raw = await self.redis.get(key)

        if raw is None:
            new_ema = value
        else:
            previous_ema = float(raw)
            new_ema = (value * alpha) + (previous_ema * (1.0 - alpha))

        await self.redis.set(key, str(new_ema))
        return new_ema
