import logging
from typing import Any, Dict, Optional

from backend.services.telemetry_service.src.infrastructure.cache.redis_cache import TelemetryRedisCache
from backend.services.telemetry_service.src.infrastructure.detectors.statistical_detector import Anomaly

logger = logging.getLogger(__name__)


class RuleDetector:
    """
    Evaluates absolute thresholds stored dynamically in PostgreSQL.
    Caches rules in Redis (TTL=5min) for O(1) evaluation per reading.
    """

    def __init__(self, redis_cache: TelemetryRedisCache):
        self.redis_cache = redis_cache

    async def detect(self, sensor_id: str, machine_id: str, value: float, timestamp: float) -> Optional[Anomaly]:
        """
        Evaluate reading against cached SQL rules.
        Rule schema: {sensor_id, condition, threshold, severity, message}
        """
        # Fetch rules from cache (mock implementation assumes cache returns empty mostly)
        rules = await self.redis_cache.get_machine_rules(machine_id)
        
        if not rules:
            return None
            
        for rule in rules:
            if rule.get("sensor_id") != sensor_id:
                continue
                
            condition = rule.get("condition")
            threshold = float(rule.get("threshold", 0.0))
            
            is_breached = False
            if condition == "above" and value > threshold:
                is_breached = True
            elif condition == "below" and value < threshold:
                is_breached = True
                
            if is_breached:
                severity = rule.get("severity", "MEDIUM")
                
                # Deduplicate
                is_duplicate = await self.redis_cache.is_duplicate_anomaly(sensor_id, severity, window_seconds=300)
                if is_duplicate:
                    return None
                    
                anomaly = Anomaly(
                    sensor_id=sensor_id,
                    machine_id=machine_id,
                    value=value,
                    expected_range=f"{condition} {threshold}",
                    z_score=0.0, # N/A for rule-based
                    severity=severity,
                    timestamp=timestamp
                )
                
                await self.redis_cache.add_recent_anomaly(machine_id, anomaly.__dict__, timestamp)
                logger.warning("Rule Anomaly Detected: %s on %s breached %s %s", sensor_id, machine_id, condition, threshold)
                
                return anomaly
                
        return None
