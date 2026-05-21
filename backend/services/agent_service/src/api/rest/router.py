from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, Optional

from typing import Any, Dict

import asyncpg
from fastapi import Depends, HTTPException

from backend.shared.infrastructure.database.postgres import get_db_pool

import asyncpg
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from backend.services.agent_service.src.application.orchestrator import (
    get_orchestrator,
)
from backend.services.agent_service.src.domain.models.agent_task import AgentTask
from backend.shared.infrastructure.database.postgres import get_db_pool

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/agents", tags=["agents"])


class AnalyzeRequest(BaseModel):
    session_id: str
    tenant_id: str
    task_id: Optional[str] = None
    query: str
    task_type: str = "conversational_query"
    metadata: Dict[str, Any] = {}


@router.post("/analyze")
async def submit_analysis_task(
    req: AnalyzeRequest,
    background_tasks: BackgroundTasks,
    db_pool: asyncpg.Pool = Depends(get_db_pool),
) -> Dict[str, str]:
    task_id: str = req.task_id or str(uuid.uuid4())

    agent_task = AgentTask(
        session_id=req.session_id,
        tenant_id=req.tenant_id,
        task_id=task_id,
        query=req.query,
        metadata={
            **req.metadata,
            "task_type": req.task_type,
        },
    )

    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO agent_tasks (
                task_id,
                session_id,
                tenant_id,
                status,
                query,
                metadata,
                result,
                error,
                created_at,
                updated_at
            )
            VALUES (
                $1,
                $2,
                $3,
                $4,
                $5,
                $6::jsonb,
                NULL,
                NULL,
                NOW(),
                NOW()
            )
            """,
            task_id,
            req.session_id,
            req.tenant_id,
            "processing",
            req.query,
            json.dumps(agent_task.metadata),
        )

    async def process_task(task: AgentTask) -> None:
        try:
            orch = get_orchestrator()

            result = await orch.route_and_execute(task)

            result_payload: Any = (
                result.model_dump()
                if hasattr(result, "model_dump")
                else result
            )

            async with db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE agent_tasks
                    SET
                        status = $1,
                        result = $2::jsonb,
                        updated_at = NOW()
                    WHERE task_id = $3
                    """,
                    "completed",
                    json.dumps(result_payload),
                    task.task_id,
                )

        except Exception as e:
            logger.error(
                "Task processing failed for task_id=%s",
                task.task_id,
                exc_info=True,
            )

            async with db_pool.acquire() as conn:
                await conn.execute(
                    """
                    UPDATE agent_tasks
                    SET
                        status = $1,
                        error = $2,
                        updated_at = NOW()
                    WHERE task_id = $3
                    """,
                    "failed",
                    str(e),
                    task.task_id,
                )

    background_tasks.add_task(process_task, agent_task)

    return {
        "task_id": task_id,
        "status": "processing",
    }


@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    db_pool: asyncpg.Pool = Depends(get_db_pool),
) -> Dict[str, Any]:
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                task_id,
                session_id,
                tenant_id,
                status,
                query,
                metadata,
                result,
                error,
                created_at,
                updated_at
            FROM agent_tasks
            WHERE task_id = $1
            """,
            task_id,
        )

    if row is None:
        raise HTTPException(status_code=404, detail="Task not found")

    return dict(row)


@router.get("/tasks/{task_id}/trace")
async def get_task_trace(
    task_id: str,
    db_pool: asyncpg.Pool = Depends(get_db_pool),
) -> Dict[str, Any]:
    if not task_id or not task_id.strip():
        raise HTTPException(status_code=400, detail="task_id must not be empty")

    try:
        async with db_pool.acquire() as conn:
            task_exists = await conn.fetchrow(
                """
                SELECT task_id
                FROM agent_tasks
                WHERE task_id = $1
                """,
                task_id,
            )

            if task_exists is None:
                raise HTTPException(
                    status_code=404,
                    detail="Task not found",
                )

            rows = await conn.fetch(
                """
                SELECT
                    tool_name,
                    input_params,
                    output_data,
                    success,
                    error_message,
                    duration_ms,
                    created_at
                FROM agent_tool_calls
                WHERE task_id = $1
                ORDER BY created_at ASC
                """,
                task_id,
            )

        trace = [dict(row) for row in rows]

        return {
            "task_id": task_id,
            "trace": trace,
        }

    except HTTPException:
        raise

    except Exception:
        logger.error(
            "Failed to retrieve task trace for task_id=%s",
            task_id,
            exc_info=True,
        )

        raise HTTPException(
            status_code=500,
            detail="Database query failed",
        )