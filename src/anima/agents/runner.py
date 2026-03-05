"""TaskRunner — drives a single AgentTask through the Plan→Review→Implement→Test→Evaluate loop."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from typing import Any

from ..config import MindConfig
from ..llm import complete
from ..state import SharedState
from .models import AgentTask, TaskPhase, TaskStatus, ToolSpec, PHASE_ORDER

logger = logging.getLogger(__name__)

# Phase transition: PLAN → REVIEW → IMPLEMENT → TEST → EVALUATE
# On EVALUATE failure: back to PLAN (up to max_retries)

PHASE_PROMPTS = {
    TaskPhase.PLAN: (
        "You are planning how to accomplish this task.\n\n"
        "Task: {description}\n\n"
        "{context}"
        "Produce a structured plan. If this task is complex and would benefit from "
        "being broken into subtasks, include a JSON block:\n"
        '```json\n{{"subtasks": [{{"description": "...", "model": ""}}]}}\n```\n'
        "Otherwise, just describe the step-by-step approach."
    ),
    TaskPhase.REVIEW: (
        "Review this plan for the task and identify any issues or improvements.\n\n"
        "Task: {description}\n\n"
        "Plan:\n{plan_result}\n\n"
        "Is this plan sound? Flag any problems. If it looks good, say APPROVED."
    ),
    TaskPhase.IMPLEMENT: (
        "Implement the following task according to the plan.\n\n"
        "Task: {description}\n\n"
        "Plan:\n{plan_result}\n\n"
        "Review notes:\n{review_result}\n\n"
        "Produce the implementation result."
    ),
    TaskPhase.TEST: (
        "Verify the implementation result for correctness.\n\n"
        "Task: {description}\n\n"
        "Implementation:\n{implement_result}\n\n"
        "Check for errors, inconsistencies, or missing pieces. "
        "Report what you find."
    ),
    TaskPhase.EVALUATE: (
        "Evaluate whether this task has been completed successfully.\n\n"
        "Task: {description}\n\n"
        "Implementation:\n{implement_result}\n\n"
        "Test results:\n{test_result}\n\n"
        'Respond with JSON: {{"passed": true/false, "reason": "..."}}'
    ),
}


class TaskRunner:
    """Drives one AgentTask through Plan→Review→Implement→Test→Evaluate."""

    def __init__(
        self,
        task: AgentTask,
        state: SharedState,
        config: MindConfig,
        enqueue_fn: Callable[[AgentTask], Any] | None = None,
        tool_handlers: dict[str, Callable] | None = None,
    ):
        self.task = task
        self.state = state
        self.config = config
        self._enqueue_fn = enqueue_fn  # for spawning child tasks
        self._tool_handlers = tool_handlers or {}

    async def run(self) -> AgentTask:
        """Drive the task through all phases. Returns the completed/failed task."""
        self.task.status = TaskStatus.RUNNING
        await self.state.update_agent_task_status(
            self.task.id, TaskStatus.RUNNING, self.task.phase
        )

        try:
            while self.task.status == TaskStatus.RUNNING:
                result = await self._execute_phase()
                self.task.phase_results[self.task.phase.value] = result
                await self.state.update_agent_task_phase_result(
                    self.task.id, self.task.phase.value, result
                )

                if self.task.phase == TaskPhase.PLAN:
                    await self._maybe_spawn_children(result)

                if self.task.phase == TaskPhase.EVALUATE:
                    self._handle_evaluation(result)
                else:
                    self.task.phase = self._next_phase()
                    await self.state.update_agent_task_status(
                        self.task.id, TaskStatus.RUNNING, self.task.phase
                    )
        except Exception as e:
            logger.error("[RUNNER] Task %s failed: %s", self.task.id, e)
            self.task.status = TaskStatus.FAILED
            self.task.result = str(e)

        await self.state.update_agent_task_final(self.task)
        return self.task

    def _next_phase(self) -> TaskPhase:
        idx = PHASE_ORDER.index(self.task.phase)
        return PHASE_ORDER[idx + 1]

    def _handle_evaluation(self, result: str):
        passed = self._parse_evaluation(result)
        if passed:
            self.task.status = TaskStatus.COMPLETE
            self.task.result = self.task.phase_results.get("implement", result)
        elif self.task.retry_count < self.task.max_retries:
            self.task.retry_count += 1
            self.task.phase = TaskPhase.PLAN  # loop back
            logger.info(
                "[RUNNER] Task %s failed evaluation, retry %d/%d",
                self.task.id, self.task.retry_count, self.task.max_retries,
            )
        else:
            self.task.status = TaskStatus.FAILED
            self.task.result = (
                f"Failed after {self.task.max_retries} retries: {result}"
            )

    def _parse_evaluation(self, result: str) -> bool:
        try:
            # Try to extract JSON from the result
            text = result.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            data = json.loads(text)
            return bool(data.get("passed", False))
        except (json.JSONDecodeError, IndexError):
            # Fallback: check for common pass indicators
            lower = result.lower()
            return "passed" in lower and "not passed" not in lower and "failed" not in lower

    async def _execute_phase(self) -> str:
        # For IMPLEMENT with children: wait for children instead of LLM call
        if self.task.phase == TaskPhase.IMPLEMENT and self.task.children:
            return await self._wait_for_children()

        prompt = self._build_phase_prompt()
        model_raw = self.task.model or self.config.default_agent_model
        model = self.config.resolve_model(model_raw)

        # Only pass tools during IMPLEMENT phase
        tools = None
        if self.task.phase == TaskPhase.IMPLEMENT and self.task.tools:
            tools = [
                {
                    "name": t.name,
                    "description": t.description,
                    "input_schema": t.input_schema,
                }
                for t in self.task.tools
            ]

        try:
            response = await asyncio.wait_for(
                self._call_llm(model, prompt, tools),
                timeout=self.config.agent_task_timeout,
            )
            return response
        except asyncio.TimeoutError:
            return f"[TIMEOUT] Phase {self.task.phase.value} timed out after {self.config.agent_task_timeout}s"

    async def _call_llm(
        self,
        model: str,
        prompt: str,
        tools: list[dict] | None = None,
    ) -> str:
        """Call the LLM, handling tool_use loops if tools are provided."""
        messages = [{"role": "user", "content": prompt}]

        response = await complete(
            model=model,
            max_tokens=4000,
            messages=messages,
            tools=tools,
        )

        # If no tools or no tool calls in response, return text
        if not tools or response.finish_reason != "tool_use":
            return response.text

        # Tool use loop
        max_tool_rounds = 10
        for _ in range(max_tool_rounds):
            if not response.tool_calls:
                break

            # Append assistant message with tool calls
            assistant_msg: dict[str, Any] = {"role": "assistant"}
            if response.text:
                assistant_msg["content"] = response.text
            if response.raw_tool_calls:
                assistant_msg["tool_calls"] = response.raw_tool_calls
            messages.append(assistant_msg)

            # Execute tools and append results
            for tc in response.tool_calls:
                result = await self._execute_tool(tc["name"], tc["input"])
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result),
                })

            response = await complete(
                model=model,
                max_tokens=4000,
                messages=messages,
                tools=tools,
            )
            if response.finish_reason != "tool_use":
                break

        return response.text

    async def _execute_tool(self, name: str, input_data: dict) -> str:
        handler = self._tool_handlers.get(name)
        if not handler:
            return f"[ERROR] No handler for tool: {name}"
        try:
            result = handler(input_data)
            if asyncio.iscoroutine(result):
                result = await result
            return str(result)
        except Exception as e:
            return f"[ERROR] Tool {name} failed: {e}"

    def _build_phase_prompt(self) -> str:
        template = PHASE_PROMPTS[self.task.phase]
        context = ""
        if self.task.phase == TaskPhase.PLAN and self.task.depth < self.task.max_depth:
            remaining = self.task.max_subtasks - len(self.task.children)
            if remaining > 0:
                context = (
                    f"You may decompose into up to {remaining} subtasks if needed.\n"
                    f"Current depth: {self.task.depth}/{self.task.max_depth}\n\n"
                )
        return template.format(
            description=self.task.description,
            context=context,
            plan_result=self.task.phase_results.get("plan", ""),
            review_result=self.task.phase_results.get("review", ""),
            implement_result=self.task.phase_results.get("implement", ""),
            test_result=self.task.phase_results.get("test", ""),
        )

    async def _maybe_spawn_children(self, plan_result: str):
        """Parse plan for subtask decomposition and spawn child tasks."""
        if self.task.depth >= self.task.max_depth:
            return
        if len(self.task.children) >= self.task.max_subtasks:
            return

        subtasks = self._extract_subtasks(plan_result)
        if not subtasks:
            return

        remaining = self.task.max_subtasks - len(self.task.children)
        for sub in subtasks[:remaining]:
            child = AgentTask(
                parent_task_id=self.task.id,
                conversation_id=self.task.conversation_id,
                model=sub.get("model", "") or self.task.model,
                description=sub.get("description", ""),
                tools=list(self.task.tools),  # inherit parent tools
                depth=self.task.depth + 1,
                max_depth=self.task.max_depth,
                max_subtasks=self.task.max_subtasks,
                max_retries=self.task.max_retries,
            )
            await self.state.create_agent_task(child)
            await self.state.add_child_to_task(self.task.id, child.id)
            self.task.children.append(child.id)

            if self._enqueue_fn:
                await self._enqueue_fn(child)

        logger.info(
            "[RUNNER] Task %s spawned %d children at depth %d",
            self.task.id, len(subtasks[:remaining]), self.task.depth + 1,
        )

    def _extract_subtasks(self, text: str) -> list[dict]:
        try:
            # Look for JSON block with subtasks
            if "```json" in text:
                json_str = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                json_str = text.split("```")[1].split("```")[0].strip()
            else:
                # Try parsing the whole text as JSON
                json_str = text.strip()

            data = json.loads(json_str)
            return data.get("subtasks", [])
        except (json.JSONDecodeError, IndexError):
            return []

    async def _wait_for_children(self) -> str:
        """Poll children until all complete/failed, then aggregate results."""
        poll_interval = 5.0
        timeout = self.config.agent_task_timeout

        elapsed = 0.0
        while elapsed < timeout:
            children = await self.state.get_children_tasks(self.task.id)
            all_done = all(
                c.status in (TaskStatus.COMPLETE, TaskStatus.FAILED)
                for c in children
            )
            if all_done:
                break
            await asyncio.sleep(poll_interval)
            elapsed += poll_interval

        # Aggregate results
        children = await self.state.get_children_tasks(self.task.id)
        parts = []
        failed = []
        for child in children:
            if child.status == TaskStatus.COMPLETE:
                parts.append(f"[{child.description}]: {child.result}")
            else:
                failed.append(f"[{child.description}]: FAILED — {child.result}")

        result = "\n\n".join(parts)
        if failed:
            result += "\n\nFailed subtasks:\n" + "\n".join(failed)
        return result
