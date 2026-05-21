from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from backend.services.telemetry_service.src.infrastructure.cache.redis_cache import TelemetryRedisCache

logger = logging.getLogger(__name__)


@dataclass
class Anomaly:
    sensor_id: str
    machine_id: str
    value: float
    expected_range: str
    z_score: float
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    timestamp: float


class StatisticalDetector:
    """
    Fast stateful anomaly detection using per-sensor Redis-backed Z-scores.

    Baseline statistics (mean, std_dev, count) are maintained in Redis via
    Welford's online algorithm so every sensor has its own rolling distribution
    rather than a shared hardcoded standard deviation.

    Cold-start behaviour
    --------------------
    For the first ``min_sample_count`` readings on a new sensor, ``detect``
    accumulates statistics and returns ``None`` — there is not yet enough data
    to produce a statistically valid Z-score.
    """

    def __init__(
        self,
        redis_cache: TelemetryRedisCache,
        z_threshold: float = 3.0,
        min_sample_count: int = 30,
    ) -> None:
        self.redis_cache = redis_cache
        # Minimum readings before Z-score is considered statistically valid.
        self.min_sample_count: int = min_sample_count
        # Number of standard deviations above which a reading is anomalous.
        self.z_threshold: float = z_threshold

    async def detect(
        self,
        sensor_id: str,
        machine_id: str,
        value: float,
        timestamp: float,
    ) -> Optional[Anomaly]:
        """Evaluate a single sensor reading for statistical anomalies.

        Args:
            sensor_id:  Unique sensor identifier.
            machine_id: Parent machine identifier (used for dedup and logging).
            value:      Current sensor reading.
            timestamp:  Unix epoch of the reading.

        Returns:
            An ``Anomaly`` instance if the reading exceeds ``z_threshold`` and
            is not a duplicate within the deduplication window, otherwise ``None``.
        """
        # ── Step 1: Always update the rolling distribution first ─────────────
        # This must happen before any early-return so cold-start readings
        # accumulate toward min_sample_count even when we return None.
        await self.redis_cache.update_sensor_stats(sensor_id, value)

        # ── Step 2: Fetch current per-sensor stats ────────────────────────────
        stats: Optional[Dict[str, Any]] = await self.redis_cache.get_sensor_stats(sensor_id)

        if stats is None or stats["count"] < self.min_sample_count:
            logger.debug(
                "Warming up baseline for sensor_id=%s — "
                "samples so far: %d / %d required",
                sensor_id,
                stats["count"] if stats else 0,
                self.min_sample_count,
            )
            return None

        mean: float = stats["mean"]
        std_dev: float = stats["std_dev"]

        # ── Step 3: Compute Z-score ────────────────────────────────────────────
        # Floor std_dev at 0.001 to prevent division-by-zero when a sensor
        # is stuck at a constant value (std_dev converges to 0.0).
        z_score: float = abs(value - mean) / max(std_dev, 0.001)

        # ── Step 4: Evaluate severity thresholds ──────────────────────────────
        severity: Optional[str] = None
        if z_score > 5.0:
            severity = "CRITICAL"
        elif z_score > self.z_threshold + 2.0:  # > 5.0 already caught above
            severity = "HIGH"
        elif z_score > self.z_threshold:         # > 3.0
            severity = "MEDIUM"
        elif z_score > self.z_threshold - 1.0:   # > 2.0
            severity = "LOW"

        if severity is None:
            return None

        # ── Step 5: Deduplication — suppress repeat alerts within 5 minutes ──
        is_duplicate: bool = await self.redis_cache.is_duplicate_anomaly(
            sensor_id, severity, window_seconds=300
        )
        if is_duplicate:
            logger.debug(
                "Suppressing duplicate %s anomaly for sensor_id=%s within dedup window.",
                severity,
                sensor_id,
            )
            return None

        # ── Step 6: Build Anomaly ─────────────────────────────────────────────
        expected_range: str = f"{mean:.2f} ± {std_dev:.2f}"

        anomaly = Anomaly(
            sensor_id=sensor_id,
            machine_id=machine_id,
            value=value,
            expected_range=expected_range,
            z_score=z_score,
            severity=severity,
            timestamp=timestamp,
        )

        # ── Step 7: Persist to recent anomaly log ─────────────────────────────
        await self.redis_cache.add_recent_anomaly(
            machine_id, anomaly.__dict__, timestamp
        )

        logger.warning(
            "Anomaly detected — sensor_id=%s machine_id=%s severity=%s "
            "value=%.4f z_score=%.2f expected=%s",
            sensor_id,
            machine_id,
            severity,
            value,
            z_score,
            expected_range,
        )

        return anomaly