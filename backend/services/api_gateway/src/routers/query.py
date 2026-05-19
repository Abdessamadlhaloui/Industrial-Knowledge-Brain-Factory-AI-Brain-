import asyncio
import json
import logging
import time
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from backend.services.api_gateway.src.schemas.query import QueryRequest, QueryResponse, SourceReference

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
async def execute_query(req: QueryRequest, request: Request):
    """
    Executes a natural language query synchronously.
    """
    start_time = time.time()
    
    # Enforce tenant isolation securely
    tenant_id = getattr(request.state, "tenant_id", req.tenant_id)
    
    logger.info("Executing query for tenant %s: %s", tenant_id, req.query[:50])
    
    # Mock routing to agent_service
    # In reality, use httpx or gRPC to call agent_service/api/v1/analyze
    
    await asyncio.sleep(1.5) # Simulate processing
    
    latency = (time.time() - start_time) * 1000
    
    return QueryResponse(
        answer=f"Simulated response to: {req.query}",
        confidence=0.92,
        sources=[
            SourceReference(
                doc_id="doc-123",
                title="CNC Maintenance Manual",
                score=0.88,
                excerpt="Always check the spindle bearing...",
                source_type="manual"
            )
        ],
        reasoning_steps=["Identified entity CNC", "Searched manuals", "Synthesized answer"],
        recommended_actions=["Check bearing", "Lubricate spindle"],
        latency_ms=latency
    )


@router.post("/stream")
async def execute_query_stream(req: QueryRequest, request: Request):
    """
    Executes a natural language query and streams the response via Server-Sent Events (SSE).
    """
    tenant_id = getattr(request.state, "tenant_id", req.tenant_id)
    logger.info("Streaming query for tenant %s: %s", tenant_id, req.query[:50])

    async def event_generator():
        # Mock streaming tokens
        tokens = ["This ", "is ", "a ", "streamed ", "response ", "from ", "the ", "agent."]
        
        for idx, token in enumerate(tokens):
            yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"
            await asyncio.sleep(0.1)
            
        # Final payload
        final_payload = {
            "type": "complete",
            "sources": [
                {
                    "doc_id": "doc-123",
                    "title": "CNC Maintenance Manual",
                    "score": 0.88,
                    "excerpt": "Always check the spindle bearing...",
                    "source_type": "manual"
                }
            ],
            "confidence": 0.87
        }
        yield f"data: {json.dumps(final_payload)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/history")
async def get_query_history(session_id: str, request: Request):
    """Fetch paginated query history."""
    return {"session_id": session_id, "history": []}
