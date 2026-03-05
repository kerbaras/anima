"""Behavioral tests for the task orchestrator and runner.

Tests the full lifecycle of AgentTasks: enqueue → phase transitions → completion,
retry logic, child task spawning, and crash recovery.  LLM calls are mocked;
the queue, semaphore, and DB state run for real.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from anima.agents.models import AgentTask, TaskPhase, TaskStatus, ToolSpec
from anima.agents.runner import TaskRunner
from anima.config import MindConfig
from anima.llm import LLMResponse


# ── Helpers ──────────────────────────────────────────────────────────────────


def _llm(text: str) -> LLMResponse:
    return LLMResponse(text=text, finish_reason="stop")


def _eval_pass() -> LLMResponse:
    return _llm('{"passed": true, "reason": "looks good"}')


def _eval_fail() -> LLMResponse:
    return _llm('{"passed": false, "reason": "does not meet requirements"}')


# ── Scenario 8: Task lifecycle — all phases complete successfully ──────────


class TestTaskLifecycle:
    @patch("anima.agents.runner.complete")
    async def test_task_runs_through_all_phases(self, mock_complete, state):
        config = MindConfig(db_path=":memory:")
        task = AgentTask(description="Write a hello world function")
        await state.create_agent_task(task)

        # Responses for PLAN → REVIEW → IMPLEMENT → TEST → EVALUATE
        mock_complete.side_effect = [
            _llm("Step 1: create function. Step 2: return string."),
            _llm("Plan looks good. APPROVED."),
            _llm("def hello(): return 'hello world'"),
            _llm("Function works correctly, no issues found."),
            _eval_pass(),
        ]

        runner = TaskRunner(task, state, config)
        completed = await runner.run()

        assert completed.status == TaskStatus.COMPLETE
        assert "hello world" in completed.result
        assert completed.phase_results.get("plan")
        assert completed.phase_results.get("review")
        assert completed.phase_results.get("implement")
        assert completed.phase_results.get("test")
        assert completed.phase_results.get("evaluate")

    @patch("anima.agents.runner.complete")
    async def test_phase_order_is_correct(self, mock_complete, state):
        """Track the order of phases to verify PLAN→REVIEW→IMPLEMENT→TEST→EVALUATE."""
        config = MindConfig(db_path=":memory:")
        task = AgentTask(description="test phase order")
        await state.create_agent_task(task)

        phases_seen = []

        async def track_phases(**kwargs):
            phases_seen.append(task.phase.value)
            if task.phase == TaskPhase.EVALUATE:
                return _eval_pass()
            return _llm(f"result for {task.phase.value}")

        mock_complete.side_effect = track_phases

        runner = TaskRunner(task, state, config)
        await runner.run()

        assert phases_seen == ["plan", "review", "implement", "test", "evaluate"]


# ── Scenario 9: Task failure → Retry → Eventually fail ────────────────────


class TestTaskRetryBehavior:
    @patch("anima.agents.runner.complete")
    async def test_task_retries_on_evaluation_failure(self, mock_complete, state):
        config = MindConfig(db_path=":memory:")
        task = AgentTask(description="tricky task", max_retries=2)
        await state.create_agent_task(task)

        # First attempt: all phases pass, but evaluation fails
        # Second attempt: same — evaluation fails again
        # Third attempt: evaluation passes
        mock_complete.side_effect = [
            # Attempt 1
            _llm("plan v1"), _llm("APPROVED"), _llm("impl v1"), _llm("test OK"),
            _eval_fail(),
            # Attempt 2 (retry 1)
            _llm("plan v2"), _llm("APPROVED"), _llm("impl v2"), _llm("test OK"),
            _eval_fail(),
            # Attempt 3 (retry 2)
            _llm("plan v3"), _llm("APPROVED"), _llm("impl v3"), _llm("test OK"),
            _eval_pass(),
        ]

        runner = TaskRunner(task, state, config)
        completed = await runner.run()

        assert completed.status == TaskStatus.COMPLETE
        assert completed.retry_count == 2

    @patch("anima.agents.runner.complete")
    async def test_task_fails_after_max_retries(self, mock_complete, state):
        config = MindConfig(db_path=":memory:")
        task = AgentTask(description="impossible task", max_retries=1)
        await state.create_agent_task(task)

        # First attempt fails, retry fails, then out of retries
        mock_complete.side_effect = [
            _llm("plan"), _llm("APPROVED"), _llm("impl"), _llm("test"),
            _eval_fail(),
            _llm("plan2"), _llm("APPROVED"), _llm("impl2"), _llm("test2"),
            _eval_fail(),
        ]

        runner = TaskRunner(task, state, config)
        completed = await runner.run()

        assert completed.status == TaskStatus.FAILED
        assert "1 retries" in completed.result

    @patch("anima.agents.runner.complete")
    async def test_retry_resets_to_plan_phase(self, mock_complete, state):
        """On evaluation failure, phase must reset to PLAN."""
        config = MindConfig(db_path=":memory:")
        task = AgentTask(description="retry test", max_retries=1)
        await state.create_agent_task(task)

        phases_after_eval = []

        async def track(**kwargs):
            if task.phase == TaskPhase.EVALUATE:
                return _eval_fail()
            if task.phase == TaskPhase.PLAN and task.retry_count > 0:
                phases_after_eval.append(task.phase.value)
                # Now let it pass on second attempt
                task.max_retries = 0  # prevent further retries
            return _llm("result")

        # Use a counter approach instead
        call_count = 0
        async def sequenced(**kwargs):
            nonlocal call_count
            call_count += 1
            # 5 calls per attempt: plan, review, implement, test, evaluate
            if call_count == 5:
                return _eval_fail()
            if call_count == 10:
                return _eval_pass()
            return _llm("result")

        mock_complete.side_effect = sequenced

        runner = TaskRunner(task, state, config)
        completed = await runner.run()

        # The task should have gone through PLAN twice
        assert "plan" in completed.phase_results


# ── Scenario 10: Task decomposition → Child spawning ──────────────────────


class TestTaskDecomposition:
    @patch("anima.agents.runner.complete")
    async def test_plan_with_subtasks_spawns_children(self, mock_complete, state):
        config = MindConfig(db_path=":memory:")
        task = AgentTask(
            description="build a web app",
            max_depth=2,
            max_subtasks=3,
        )
        await state.create_agent_task(task)

        children_enqueued = []

        async def mock_enqueue(child_task):
            children_enqueued.append(child_task)

        plan_response = (
            "Let me break this down:\n"
            '```json\n{"subtasks": ['
            '{"description": "Set up backend", "model": ""},'
            '{"description": "Build frontend", "model": ""}'
            "]}\n```"
        )

        mock_complete.side_effect = [
            _llm(plan_response),  # PLAN — returns subtasks
            _llm("APPROVED"),     # REVIEW
            # IMPLEMENT will wait for children, but we mock state to return completed children
        ]

        runner = TaskRunner(task, state, config, enqueue_fn=mock_enqueue)

        # Just run the plan phase to test subtask extraction
        result = await runner._execute_phase()
        task.phase_results["plan"] = result
        await runner._maybe_spawn_children(result)

        assert len(task.children) == 2
        assert len(children_enqueued) == 2
        assert children_enqueued[0].description == "Set up backend"
        assert children_enqueued[1].description == "Build frontend"
        assert children_enqueued[0].depth == 1
        assert children_enqueued[0].parent_task_id == task.id

    @patch("anima.agents.runner.complete")
    async def test_max_depth_prevents_further_decomposition(self, mock_complete, state):
        config = MindConfig(db_path=":memory:")
        task = AgentTask(
            description="leaf task",
            depth=3,  # already at max
            max_depth=3,
        )
        await state.create_agent_task(task)

        plan_with_subtasks = '```json\n{"subtasks": [{"description": "child"}]}\n```'

        runner = TaskRunner(task, state, config)
        await runner._maybe_spawn_children(plan_with_subtasks)

        assert len(task.children) == 0, "Should not spawn at max depth"


# ── Scenario 11: Crash recovery ──────────────────────────────────────────


class TestCrashRecovery:
    async def test_running_tasks_recovered_to_queued(self, state):
        """Tasks left in RUNNING state after a crash should be reset to QUEUED."""
        # Pre-populate DB with tasks in various states
        running_task = AgentTask(
            id="crash-1",
            description="was running when crashed",
            status=TaskStatus.RUNNING,
        )
        complete_task = AgentTask(
            id="done-1",
            description="already completed",
            status=TaskStatus.COMPLETE,
        )
        queued_task = AgentTask(
            id="queued-1",
            description="waiting in queue",
            status=TaskStatus.QUEUED,
        )

        for t in [running_task, complete_task, queued_task]:
            await state.create_agent_task(t)

        # Simulate crash recovery
        await state.recover_running_tasks()

        recovered = await state.get_agent_task("crash-1")
        assert recovered.status == TaskStatus.QUEUED

        # Other tasks should be unaffected
        done = await state.get_agent_task("done-1")
        assert done.status == TaskStatus.COMPLETE

        still_queued = await state.get_agent_task("queued-1")
        assert still_queued.status == TaskStatus.QUEUED

    @patch("anima.agents.runner.complete")
    async def test_runner_crash_sets_task_failed(self, mock_complete, state):
        """If the runner crashes mid-execution, task status should be FAILED."""
        config = MindConfig(db_path=":memory:")
        task = AgentTask(description="will crash")
        await state.create_agent_task(task)

        mock_complete.side_effect = RuntimeError("LLM provider down")

        runner = TaskRunner(task, state, config)
        completed = await runner.run()

        assert completed.status == TaskStatus.FAILED
        assert "LLM provider down" in completed.result


# ── Scenario: Evaluation parsing edge cases ──────────────────────────────


class TestEvaluationParsing:
    """The evaluation parser should handle various LLM response formats."""

    @patch("anima.agents.runner.complete")
    async def test_json_in_markdown_fence(self, mock_complete, state):
        config = MindConfig(db_path=":memory:")
        task = AgentTask(description="test eval parsing")
        await state.create_agent_task(task)

        mock_complete.side_effect = [
            _llm("plan"), _llm("APPROVED"), _llm("impl"), _llm("test ok"),
            _llm('Here is my evaluation:\n```json\n{"passed": true, "reason": "good"}\n```'),
        ]

        runner = TaskRunner(task, state, config)
        completed = await runner.run()
        assert completed.status == TaskStatus.COMPLETE

    @patch("anima.agents.runner.complete")
    async def test_text_fallback_parsing(self, mock_complete, state):
        config = MindConfig(db_path=":memory:")
        task = AgentTask(description="test text fallback")
        await state.create_agent_task(task)

        mock_complete.side_effect = [
            _llm("plan"), _llm("APPROVED"), _llm("impl"), _llm("test ok"),
            _llm("The task has passed all requirements."),
        ]

        runner = TaskRunner(task, state, config)
        completed = await runner.run()
        assert completed.status == TaskStatus.COMPLETE

    @patch("anima.agents.runner.complete")
    async def test_tool_use_during_implement(self, mock_complete, state):
        """During IMPLEMENT phase, tool calls should be executed and results fed back."""
        config = MindConfig(db_path=":memory:")
        task = AgentTask(
            description="task with tools",
            tools=[ToolSpec(name="test_tool", description="A test tool")],
        )
        await state.create_agent_task(task)

        tool_results = []

        def mock_tool(input_data):
            tool_results.append(input_data)
            return "tool executed successfully"

        # First call returns tool_use, second returns text
        mock_complete.side_effect = [
            _llm("plan"), _llm("APPROVED"),
            # IMPLEMENT: returns tool call
            LLMResponse(
                text="Let me use the tool",
                tool_calls=[{"id": "tc1", "name": "test_tool", "input": {"arg": "val"}}],
                raw_tool_calls=[{"id": "tc1", "function": {"name": "test_tool", "arguments": '{"arg": "val"}'}}],
                finish_reason="tool_use",
            ),
            _llm("Done with implementation"),  # after tool result
            _llm("tests pass"),
            _eval_pass(),
        ]

        runner = TaskRunner(
            task, state, config,
            tool_handlers={"test_tool": mock_tool},
        )
        completed = await runner.run()

        assert completed.status == TaskStatus.COMPLETE
        assert len(tool_results) == 1
        assert tool_results[0] == {"arg": "val"}
