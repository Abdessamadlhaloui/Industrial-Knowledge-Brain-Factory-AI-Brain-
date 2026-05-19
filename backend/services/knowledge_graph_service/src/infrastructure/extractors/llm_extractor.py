import hashlib
import json
import logging
import os
import uuid
from typing import Any, Dict

from openai import AsyncOpenAI
from redis.asyncio import Redis

from backend.services.knowledge_graph_service.src.infrastructure.extractors.base_extractor import (
    BaseExtractor, ExtractionResult, ExtractedRelation, IndustrialEntity
)

logger = logging.getLogger(__name__)


class LLMExtractor(BaseExtractor):
    """
    Uses GPT-4o to extract complex semantic relationships from unstructured text.
    Caches results in Redis using a text hash to prevent redundant API calls.
    """

    def __init__(self, redis_client: Redis, model: str = "gpt-4o"):
        self.redis_client = redis_client
        self.model = model
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key) if self.api_key else None
        
        self.system_prompt = (
            "You are an expert industrial maintenance engineer. Extract structured knowledge from the provided maintenance report. "
            "Identify components, failure modes, root causes, resolution actions, and spare parts used. "
            "Output strictly as a JSON object with the following schema:\n"
            "{\n"
            '  "relations": [\n'
            '    {"source": "EntityA", "relation_type": "CAUSED_BY|RESOLVED_BY|REQUIRES_PART|INDICATES", "target": "EntityB", "confidence": 0.0-1.0, "sentence_span": "Original text"}\n'
            "  ]\n"
            "}"
        )

    async def extract(self, text: str, doc_metadata: Dict[str, Any]) -> ExtractionResult:
        if not self.client:
            logger.warning("OPENAI_API_KEY not set. LLMExtractor returning empty results.")
            return ExtractionResult()

        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_key = f"ikb:extraction:{text_hash}"
        
        # Check cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            logger.debug("LLM extraction cache hit for hash %s", text_hash)
            return self._parse_json_to_result(cached_result)

        # Call LLM
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text}
                ],
                response_format={"type": "json_object"},
                temperature=0.0
            )
            
            raw_json = response.choices[0].message.content
            if not raw_json:
                return ExtractionResult()
                
            # Cache the result (TTL = 24h = 86400s)
            await self.redis_client.setex(cache_key, 86400, raw_json)
            
            return self._parse_json_to_result(raw_json)

        except Exception as e:
            logger.error("LLM extraction failed: %s", e)
            return ExtractionResult()

    def _parse_json_to_result(self, raw_json: str | bytes) -> ExtractionResult:
        try:
            data = json.loads(raw_json)
            relations_data = data.get("relations", [])
            
            relations = []
            entities = []
            
            for rel in relations_data:
                confidence = float(rel.get("confidence", 1.0))
                
                source_name = str(rel.get("source"))
                target_name = str(rel.get("target"))
                
                # We implicitly extract entities if they are part of a relation
                entities.append(IndustrialEntity(id=str(uuid.uuid4()), label="UNKNOWN", text=source_name, confidence=confidence, start_char=0, end_char=0))
                entities.append(IndustrialEntity(id=str(uuid.uuid4()), label="UNKNOWN", text=target_name, confidence=confidence, start_char=0, end_char=0))
                
                relations.append(ExtractedRelation(
                    source=source_name,
                    relation_type=str(rel.get("relation_type")),
                    target=target_name,
                    confidence=confidence,
                    sentence_span=str(rel.get("sentence_span", ""))
                ))
                
            return ExtractionResult(entities=entities, relations=relations, confidence=1.0)
            
        except json.JSONDecodeError as e:
            logger.error("Failed to parse LLM JSON: %s", e)
            return ExtractionResult()
