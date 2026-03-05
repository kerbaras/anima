"""Hierarchical task agent system with iterative phase execution."""

from .models import AgentTask, TaskPhase, TaskStatus, ToolSpec

__all__ = [
    "AgentTask",
    "TaskPhase",
    "TaskStatus",
    "ToolSpec",
    "TaskOrchestrator",
    "TaskRunner",
]


def __getattr__(name: str):
    if name == "TaskOrchestrator":
        from .orchestrator import TaskOrchestrator
        return TaskOrchestrator
    if name == "TaskRunner":
        from .runner import TaskRunner
        return TaskRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
