from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import Field, PrivateAttr

from backend.shared.base.entity import BaseEntity


class JobSourceType(str, Enum):
    PDF = "pdf"
    EXCEL = "excel"
    SCADA = "scada"
    ERP = "erp"


class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class IngestionJob(BaseEntity):
    """
    Entity representing an ingestion job for tracking data processing.
    Ensures strict status transition rules.
    """

    source_type: JobSourceType = Field(description="Type of source being ingested")
    source_url: str = Field(description="Reference or URL to the raw source data")
    tenant_id: str = Field(description="Tenant identifier")
    factory_id: str = Field(description="Factory identifier")
    
    # Internal status tracking
    _status: JobStatus = PrivateAttr(default=JobStatus.PENDING)
    
    completed_at: datetime | None = Field(default=None, description="Timestamp of completion or failure")
    error_message: str | None = Field(default=None, description="Error details if the job failed")

    def model_post_init(self, __context: Any) -> None:
        """Initialize private attributes."""
        self._status = JobStatus.PENDING

    @property
    def status(self) -> JobStatus:
        return self._status

    def mark_processing(self) -> None:
        """Transition job to PROCESSING state."""
        if self._status != JobStatus.PENDING:
            raise ValueError(f"Cannot transition from {self._status.value} to processing")
        self._status = JobStatus.PROCESSING
        self.updated_at = datetime.now(timezone.utc)

    def mark_done(self) -> None:
        """Transition job to DONE state."""
        if self._status != JobStatus.PROCESSING:
            raise ValueError(f"Cannot transition from {self._status.value} to done")
        self._status = JobStatus.DONE
        now = datetime.now(timezone.utc)
        self.updated_at = now
        self.completed_at = now

    def mark_failed(self, error_message: str) -> None:
        """Transition job to FAILED state and record the error."""
        if self._status in (JobStatus.DONE, JobStatus.FAILED):
            raise ValueError(f"Cannot transition from {self._status.value} to failed")
        self._status = JobStatus.FAILED
        self.error_message = error_message
        now = datetime.now(timezone.utc)
        self.updated_at = now
        self.completed_at = now
