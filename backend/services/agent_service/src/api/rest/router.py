import uuid
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from backend.services.agent_service.src.domain.models.agent_task import AgentTask

# This would normally be injected via FastAPI dependencies
# orchestrator = get_orchestrator()

router = APIRouter(prefix="/agents", tags=["agents"])

# In-memory store for mock async task polling
_task_store: Dict[str, Dict[str, Any]] = {}


class AnalyzeRequest(BaseModel):
    session_id: str
    tenant_id: str
    query: str
    task_type: str = "conversational_query"
    metadata: Dict[str, Any] = {}


@router.post("/analyze")
async def submit_analysis_task(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    task_id = str(uuid.uuid4())
    
    agent_task = AgentTask(
        session_id=req.session_id,
        tenant_id=req.tenant_id,
        task_id=task_id,
        query=req.query,
        metadata={**req.metadata, "task_type": req.task_type}
    )
    
    _task_store[task_id] = {"status": "processing", "result": None}
    
    # In a real app, this calls orchestrator.route_and_execute
    # background_tasks.add_task(process_task, agent_task)
    
    return {"task_id": task_id, "status": "processing"}


@router.get("/tasks/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in _task_store:
        raise HTTPException(status_code=404, detail="Task not found")
        
    return _task_store[task_id]


@router.get("/tasks/{task_id}/trace")
async def get_task_trace(task_id: str):
    # This would query PostgreSQL (agent_tool_calls)
    return {"task_id": task_id, "trace": []}
