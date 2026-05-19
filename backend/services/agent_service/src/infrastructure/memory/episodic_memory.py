import json
import logging
from typing import Any, Dict, List
from redis.asyncio import Redis

logger = logging.getLogger(__name__)


class EpisodicMemory:
    """
    Redis-backed conversational memory for agents.
    Implements a sliding window (max 50 turns) and 8-hour shift-based TTL.
    """

    def __init__(self, redis_client: Redis, max_turns: int = 50, ttl_seconds: int = 28800):
        self.redis_client = redis_client
        self.max_turns = max_turns
        self.ttl_seconds = ttl_seconds

    def _key(self, session_id: str) -> str:
        return f"agent:memory:episodic:{session_id}"

    async def append(self, session_id: str, role: str, content: str, metadata: Dict[str, Any] = None) -> None:
        """Append a message to the episodic memory."""
        key = self._key(session_id)
        msg = {
            "role": role,
            "content": content,
            "metadata": metadata or {}
        }
        
        # Append and trim
        await self.redis_client.rpush(key, json.dumps(msg))
        await self.redis_client.ltrim(key, -self.max_turns, -1)
        # Refresh TTL
        await self.redis_client.expire(key, self.ttl_seconds)

    async def get_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Retrieve the full history for a session."""
        key = self._key(session_id)
        raw_msgs = await self.redis_client.lrange(key, 0, -1)
        
        history = []
        for rm in raw_msgs:
            try:
                history.append(json.loads(rm))
            except json.JSONDecodeError:
                continue
                
        return history

    async def summarize_if_needed(self, session_id: str, current_token_count: int, threshold: int = 4096) -> None:
        """
        Auto-summarize when >80% token budget (e.g. 4096 tokens).
        Compresses old turns, keeps the last 10 verbatim.
        """
        if current_token_count < threshold:
            return
            
        logger.info("Token budget exceeded threshold (%d) for session %s. Summarizing...", current_token_count, session_id)
        history = await self.get_history(session_id)
        
        if len(history) <= 10:
            return  # Too few messages to summarize
            
        old_msgs = history[:-10]
        recent_msgs = history[-10:]
        
        # Mock summarization logic - in production, call an LLM
        summary_text = f"[SYSTEM SUMMARIZED CONTEXT]: User discussed {len(old_msgs)} prior actions..."
        
        summarized_msg = {
            "role": "system",
            "content": summary_text,
            "metadata": {"type": "summary"}
        }
        
        key = self._key(session_id)
        # Rebuild list
        await self.redis_client.delete(key)
        
        new_list = [summarized_msg] + recent_msgs
        
        for msg in new_list:
            await self.redis_client.rpush(key, json.dumps(msg))
            
        await self.redis_client.expire(key, self.ttl_seconds)
