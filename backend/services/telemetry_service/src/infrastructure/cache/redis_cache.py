import json
import logging
from typing import Any, Dict, List, Optional

from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class TelemetryRedisCache:
    """
    High-performance Redis cache handling anomaly deduplication, EMA baselines,
    and rolling recent anomalies.
    """

    def __init__(self, redis_client: Redis):
        self.redis = redis_client

    async def is_duplicate_anomaly(self, sensor_id: str, severity: str, window_seconds: int = 300) -> bool:
        """
        Sliding window deduplication.
        Returns True if the same sensor has triggered the same severity anomaly within the window.
        """
        key = f"ikb:anomaly_dedup:{sensor_id}:{severity}"
        
        # Try to set the key if it doesn't exist. If it exists, it's a duplicate.
        # NX = Set if Not eXists, EX = Expire in X seconds
        is_new = await self.redis.set(key, "1", nx=True, ex=window_seconds)
        
        return not is_new

    async def update_ema_baseline(self, sensor_id: str, value: float, alpha: float = 0.05) -> float:
        """
        Updates and returns the Exponential Moving Average baseline for a sensor in O(1).
        Formula: EMA_today = (Value * alpha) + (EMA_yesterday * (1 - alpha))
        """
        key = f"ikb:baseline:ema:{sensor_id}"
        
        previous_ema_raw = await self.redis.get(key)
        
        if previous_ema_raw is None:
            new_ema = value
        else:
            previous_ema = float(previous_ema_raw)
            new_ema = (value * alpha) + (previous_ema * (1.0 - alpha))
            
        await self.redis.set(key, str(new_ema))
        return new_ema

    async def add_recent_anomaly(self, machine_id: str, anomaly_data: Dict[str, Any], timestamp: float) -> None:
        """
        Maintains a ZSET of the last 100 anomalies per machine.
        """
        key = f"ikb:anomalies:recent:{machine_id}"
        
        # Add to sorted set with timestamp as score
        mapping = {json.dumps(anomaly_data): timestamp}
        await self.redis.zadd(key, mapping)
        
        # Keep only the top 100 most recent (highest scores)
        # ZREMRANGEBYRANK removes elements sorted lowest to highest. 
        # To keep top 100, we remove from 0 to -(100 + 1)
        card = await self.redis.zcard(key)
        if card > 100:
            await self.redis.zremrangebyrank(key, 0, card - 101)

    async def cache_rule(self, rule_id: str, rule_data: Dict[str, Any], ttl_seconds: int = 300) -> None:
        """Cache a threshold rule for O(1) evaluation."""
        key = f"ikb:rule:{rule_id}"
        await self.redis.setex(key, ttl_seconds, json.dumps(rule_data))
        
    async def get_machine_rules(self, machine_id: str) -> List[Dict[str, Any]]:
        """Mock method: In production this would query Redis Sets of rule IDs."""
        return []
