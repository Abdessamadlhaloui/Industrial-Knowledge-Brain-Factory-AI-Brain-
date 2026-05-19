import logging
from fastapi import APIRouter, Request, HTTPException

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agents", tags=["Agents"])


@router.post("/analyze")
async def submit_analysis_task(request: Request):
    """
    Submit an asynchronous agent analysis task.
    Proxies to agent_service.
    """
    logger.info("Submitting async analysis task")
    # Mock proxy to agent_service
    return {"task_id": "mock-task-123", "status": "processing"}


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str, request: Request):
    """
    Poll task status and result.
    Proxies to agent_service.
    """
    return {"task_id": task_id, "status": "completed", "result": {"output": "Analysis complete."}}


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str, request: Request):
    """Cancel an ongoing task."""
    logger.info("Canceling task %s", task_id)
    return {"status": "cancelled", "task_id": task_id}
