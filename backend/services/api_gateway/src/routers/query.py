import os
import time
import logging
import httpx
from typing import List

from fastapi import APIRouter, Request, HTTPException

from backend.services.api_gateway.src.schemas.query import QueryRequest, QueryResponse, SourceReference

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/query", tags=["Query"])

AGENT_SERVICE_URL = os.environ.get("AGENT_SERVICE_URL", "http://agent-service:8000")

@router.post("", response_model=QueryResponse)
async def execute_query(req: QueryRequest, request: Request):
    """
    Executes a natural language query asynchronously by proxying to the agent_service.
    """
    start_time = time.time()
    
    # Enforce tenant isolation securely
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Tenant ID missing from request state")

    logger.info("Executing query for tenant %s: %s", tenant_id, req.query[:50])

    payload = req.model_dump()
    payload["tenant_id"] = tenant_id  # Override to prevent spoofing

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{AGENT_SERVICE_URL}/api/v1/agents/analyze",
                json=payload
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Failed to call agent_service: %s", exc)
            raise HTTPException(status_code=502, detail="Agent service unavailable")

    data = response.json()
    
    sources_data = data.get("sources", [])
    sources = [
        SourceReference(
            doc_id=src.get("doc_id", ""),
            title=src.get("title", ""),
            score=float(src.get("score", 0.0)),
            excerpt=src.get("excerpt", ""),
            source_type=src.get("source_type", "")
        )
        for src in sources_data
    ]

    latency_ms = (time.time() - start_time) * 1000

    return QueryResponse(
        answer=data.get("answer", ""),
        confidence=float(data.get("confidence", 0.0)),
        sources=sources,
        reasoning_steps=data.get("reasoning_steps", []),
        recommended_actions=data.get("recommended_actions", []),
        latency_ms=latency_ms
    )
