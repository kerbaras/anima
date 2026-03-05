"""Data models for the hierarchical task agent system."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


class TaskPhase(str, Enum):
    PLAN = "plan"
    REVIEW = "review"
    IMPLEMENT = "implement"
    TEST = "test"
    EVALUATE = "evaluate"


class TaskStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


PHASE_ORDER = [
    TaskPhase.PLAN,
    TaskPhase.REVIEW,
    TaskPhase.IMPLEMENT,
    TaskPhase.TEST,
    TaskPhase.EVALUATE,
]


@dataclass
class ToolSpec:
    name: str = ""
    description: str = ""
    input_schema: dict = field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )


@dataclass
class AgentTask:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    parent_task_id: str | None = None
    conversation_id: str = ""
    model: str = ""  # empty = use config.default_agent_model
    description: str = ""
    tools: list[ToolSpec] = field(default_factory=list)
    phase: TaskPhase = TaskPhase.PLAN
    status: TaskStatus = TaskStatus.QUEUED
    max_subtasks: int = 5
    depth: int = 0
    max_depth: int = 3
    result: str = ""
    children: list[str] = field(default_factory=list)
    phase_results: dict[str, str] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 2
    created_at: float = field(default_factory=time.time)
