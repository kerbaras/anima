"""TaskOrchestrator — asyncio.Queue + Semaphore worker pool for AgentTasks."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from ..config import MindConfig
from ..state import SharedState
from .models import AgentTask, TaskStatus
from .runner import TaskRunner

logger = logging.getLogger(__name__)


class TaskOrchestrator:
    """
    Manages a pool of workers that pull AgentTasks from a queue,
    bounded by a semaphore for concurrency control.
    """

    def __init__(
        self,
        state: SharedState,
        config: MindConfig,
        tool_handlers: dict[str, Callable] | None = None,
    ):
        self.state = state
        self.config = config
        self._tool_handlers = tool_handlers or {}
        self._queue: asyncio.Queue[AgentTask] = asyncio.Queue()
        self._semaphore = asyncio.Semaphore(config.max_concurrent_agents)
        self._workers: list[asyncio.Task] = []
        self._running = False

    async def start(self, num_workers: int | None = None):
        """Start the worker pool. Recovers crashed tasks from DB."""
        self._running = True
        num_workers = num_workers or self.config.max_concurrent_agents

        # Crash recovery: reset RUNNING → QUEUED
        await self.state.recover_running_tasks()

        # Load queued tasks into the in-memory queue
        queued = await self.state.get_queued_agent_tasks()
        for task in queued:
            await self._queue.put(task)

        if queued:
            logger.info(
                "[ORCHESTRATOR] Recovered %d queued tasks", len(queued)
            )

        # Spawn workers
        for i in range(num_workers):
            self._workers.append(
                asyncio.create_task(self._worker(i))
            )

    async def stop(self):
        """Gracefully shut down all workers."""
        self._running = False
        for w in self._workers:
            w.cancel()
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        self._workers.clear()

    async def enqueue(self, task: AgentTask):
        """Enqueue a task for processing. Persists if not already in DB."""
        existing = await self.state.get_agent_task(task.id)
        if not existing:
            await self.state.create_agent_task(task)
        await self._queue.put(task)
        logger.info(
            "[ORCHESTRATOR] Enqueued task %s: %s (depth=%d)",
            task.id, task.description[:60], task.depth,
        )

    @property
    def pending_count(self) -> int:
        return self._queue.qsize()

    async def _worker(self, worker_id: int):
        """Worker loop: pull from queue, run with semaphore."""
        while self._running:
            try:
                task = await asyncio.wait_for(
                    self._queue.get(), timeout=5.0
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break

            async with self._semaphore:
                logger.info(
                    "[WORKER-%d] Running task %s: %s",
                    worker_id, task.id, task.description[:60],
                )
                try:
                    runner = TaskRunner(
                        task=task,
                        state=self.state,
                        config=self.config,
                        enqueue_fn=self._enqueue_child,
                        tool_handlers=self._tool_handlers,
                    )
                    completed = await runner.run()
                    logger.info(
                        "[WORKER-%d] Task %s finished: %s",
                        worker_id, completed.id, completed.status.value,
                    )
                except Exception as e:
                    logger.error(
                        "[WORKER-%d] Task %s crashed: %s",
                        worker_id, task.id, e,
                    )
                    task.status = TaskStatus.FAILED
                    task.result = f"Worker crash: {e}"
                    await self.state.update_agent_task_final(task)

                self._queue.task_done()

    async def _enqueue_child(self, child: AgentTask):
        """Callback passed to TaskRunner for spawning child tasks."""
        await self._queue.put(child)
        logger.info(
            "[ORCHESTRATOR] Child task %s enqueued (parent=%s, depth=%d)",
            child.id, child.parent_task_id, child.depth,
        )
