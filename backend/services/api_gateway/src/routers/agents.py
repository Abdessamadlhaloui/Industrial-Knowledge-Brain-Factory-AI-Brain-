import os
import logging
import httpx
from typing import Dict, Any

from fastapi import APIRouter, Request, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agents", tags=["Agents"])

AGENT_SERVICE_URL = os.environ.get("AGENT_SERVICE_URL", "http://agent-service:8000")


@router.post("/analyze")
async def submit_analysis_task(request: Request) -> Dict[str, Any]:
    """
    Submit an asynchronous agent analysis task.
    Proxies to agent_service.
    """
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Tenant ID missing from request state")

    try:
        payload = await request.json()
    except Exception:
        payload = {}
        
    payload["tenant_id"] = tenant_id

    logger.info("Submitting async analysis task for tenant: %s", tenant_id)

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(
                f"{AGENT_SERVICE_URL}/api/v1/agents/analyze",
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            logger.error("Failed to submit task to agent_service: %s", exc)
            raise HTTPException(status_code=502, detail="Agent service unavailable")


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str, request: Request) -> Dict[str, Any]:
    """
    Poll task status and result.
    Proxies to agent_service.
    """
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Tenant ID missing from request state")

    logger.info("Polling task status %s for tenant %s", task_id, tenant_id)

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.get(
                f"{AGENT_SERVICE_URL}/api/v1/agents/tasks/{task_id}",
                params={"tenant_id": tenant_id}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise HTTPException(status_code=404, detail="Task not found")
            logger.error("Agent service returned error: %s", exc)
            raise HTTPException(status_code=502, detail="Agent service error")
        except httpx.HTTPError as exc:
            logger.error("Failed to get task status from agent_service: %s", exc)
            raise HTTPException(status_code=502, detail="Agent service unavailable")


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str, request: Request) -> Dict[str, Any]:
    """
    Cancel an ongoing task.
    Proxies to agent_service.
    """
    tenant_id = getattr(request.state, "tenant_id", None)
    if not tenant_id:
        raise HTTPException(status_code=401, detail="Tenant ID missing from request state")

    logger.info("Canceling task %s for tenant %s", task_id, tenant_id)

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.delete(
                f"{AGENT_SERVICE_URL}/api/v1/agents/tasks/{task_id}",
                params={"tenant_id": tenant_id}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 404:
                raise HTTPException(status_code=404, detail="Task not found")
            logger.error("Agent service returned error: %s", exc)
            raise HTTPException(status_code=502, detail="Agent service error")
        except httpx.HTTPError as exc:
            logger.error("Failed to cancel task on agent_service: %s", exc)
            raise HTTPException(status_code=502, detail="Agent service unavailable")
