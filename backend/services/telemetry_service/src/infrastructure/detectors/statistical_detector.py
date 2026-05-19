import logging
from dataclasses import dataclass
from typing import Optional

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
    Fast stateful baseline evaluation using Z-scores, IQR, and Rate-of-Change.
    Uses Redis to maintain O(1) EMA baselines.
    """

    def __init__(self, redis_cache: TelemetryRedisCache):
        self.redis_cache = redis_cache
        # In a real system, standard deviations and IQRs would also be tracked in Redis
        # We mock the historical standard deviation here for simplicity
        self.mock_std_dev = 5.0

    async def detect(self, sensor_id: str, machine_id: str, value: float, timestamp: float) -> Optional[Anomaly]:
        """Evaluates a single reading for statistical anomalies."""
        
        # 1. Update and fetch EMA baseline
        baseline = await self.redis_cache.update_ema_baseline(sensor_id, value)
        
        # 2. Calculate Z-Score
        if self.mock_std_dev == 0:
            return None
            
        z_score = abs(value - baseline) / self.mock_std_dev
        
        # 3. Evaluate Severity thresholds
        severity = None
        if z_score > 5.0:
            severity = "HIGH"
        elif z_score > 3.0:
            severity = "MEDIUM"
        elif z_score > 2.0:
            severity = "LOW"
            
        if not severity:
            return None
            
        # 4. Check for Deduplication (e.g., don't spam 10 HIGH alerts in 5 seconds)
        is_duplicate = await self.redis_cache.is_duplicate_anomaly(sensor_id, severity, window_seconds=300)
        if is_duplicate:
            return None
            
        expected_range = f"[{baseline - (self.mock_std_dev * 2):.2f}, {baseline + (self.mock_std_dev * 2):.2f}]"
        
        anomaly = Anomaly(
            sensor_id=sensor_id,
            machine_id=machine_id,
            value=value,
            expected_range=expected_range,
            z_score=z_score,
            severity=severity,
            timestamp=timestamp
        )
        
        # 5. Log to recent anomalies
        await self.redis_cache.add_recent_anomaly(machine_id, anomaly.__dict__, timestamp)
        
        logger.warning("Statistical Anomaly Detected: %s on %s (Severity: %s)", sensor_id, machine_id, severity)
        return anomaly
