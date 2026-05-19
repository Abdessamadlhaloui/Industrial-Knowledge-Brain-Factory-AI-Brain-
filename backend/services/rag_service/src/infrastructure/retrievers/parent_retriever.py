import json
import logging
from typing import List

from redis.asyncio import Redis

from backend.services.rag_service.src.infrastructure.vector_stores.qdrant_store import ScoredPoint

logger = logging.getLogger(__name__)


class ParentRetriever:
    """
    Parent Retriever for expanding small matched child chunks into their broad parent context.
    Fetches the parent chunk from Redis using the 'parent_id' found in the child's payload.
    Falls back to the child text if the parent is missing from the cache.
    """

    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client

    async def expand_to_parents(self, scored_points: List[ScoredPoint]) -> List[ScoredPoint]:
        """
        Takes a list of retrieved points (children) and swaps their payload text 
        for the parent text if applicable.
        """
        if not scored_points:
            return []

        expanded_points = []
        
        for point in scored_points:
            parent_id = point.payload.get("parent_id")
            
            if not parent_id:
                # Not a child chunk, keep as is
                expanded_points.append(point)
                continue
                
            # Attempt to fetch parent from Redis
            redis_key = f"rag:parent:{parent_id}"
            parent_data_str = await self.redis_client.get(redis_key)
            
            new_payload = point.payload.copy()
            
            if parent_data_str:
                try:
                    parent_data = json.loads(parent_data_str)
                    new_payload["text"] = parent_data.get("text", point.payload.get("text"))
                    logger.debug("Successfully expanded child %s to parent %s", point.id, parent_id)
                except json.JSONDecodeError:
                    logger.error("Failed to decode parent data from Redis for key %s", redis_key)
            else:
                logger.warning("Parent %s for child %s not found in Redis. Falling back to child context.", parent_id, point.id)
                
            expanded_points.append(
                ScoredPoint(
                    id=point.id,
                    score=point.score,
                    payload=new_payload
                )
            )
            
        return expanded_points
