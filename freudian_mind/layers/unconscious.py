"""Unconscious Layer (Opus) — continuous background processor."""

from __future__ import annotations

import asyncio
import json
import math
import time

import anthropic

from ..config import MindConfig
from ..models import Impression, ImpressionType
from ..prompts.unconscious_prompts import (
    UNCONSCIOUS_SYSTEM_PROMPT,
    build_unconscious_context,
)
from ..state import SharedState
from ..systems.idea_space import IdeaSpace


class UnconsciousLayer:
    """
    Opus: single brain across all conversations. Demand-aware.
    Only does heavy processing when there's actual work to do.
    """

    def __init__(
        self,
        state: SharedState,
        idea_space: IdeaSpace,
        config: MindConfig,
    ):
        self.state = state
        self.idea_space = idea_space
        self.config = config
        self.client = anthropic.AsyncAnthropic()
        self._cycle_count = 1
        self._running = False
        self._task: asyncio.Task | None = None
        self._subagent_semaphore = asyncio.Semaphore(3)
        self._health_report: dict | None = None

    def set_health_report(self, report: dict):
        self._health_report = report

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._run_loop())

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run_loop(self):
        while self._running:
            try:
                demand = await self._assess_demand()

                if demand == "DEEP_ANALYSIS":
                    await self._deep_cycle()
                elif demand == "FEEDBACK_INTEGRATION":
                    pass  # Handled by growth engine
                elif demand == "SUBAGENT_DISPATCH":
                    await self._dispatch_tasks()
                elif demand == "GROWTH_ENGINE":
                    pass  # Triggered by mind.py
                else:
                    self._apply_urgency_decay()

                self._cycle_count += 1
                await asyncio.sleep(self.config.unconscious_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[UNCONSCIOUS] Error: {e}")
                await asyncio.sleep(self.config.unconscious_interval)

    async def _assess_demand(self) -> str:
        new_turns = await self.state.get_all_recent_turns(
            since_seconds=self.config.unconscious_interval
        )
        pending_outcomes = await self.state.get_pending_outcomes()
        pending_tasks = await self.state.get_pending_tasks()

        if new_turns:
            return "DEEP_ANALYSIS"
        elif pending_outcomes:
            return "FEEDBACK_INTEGRATION"
        elif pending_tasks:
            return "SUBAGENT_DISPATCH"
        elif self._cycle_count % self.config.growth_cycle_frequency == 0:
            return "GROWTH_ENGINE"
        else:
            return "IDLE"

    async def _deep_cycle(self):
        recent_turns = await self.state.get_all_recent_turns(since_seconds=120)
        active_impressions = await self.state.get_active_impressions()
        recent_outcomes = await self.state.get_recent_outcomes()
        pending_tasks = await self.state.get_pending_tasks()

        if not recent_turns:
            return

        context = build_unconscious_context(
            recent_turns,
            active_impressions,
            recent_outcomes,
            pending_tasks,
            self._health_report,
        )

        try:
            response = await self.client.messages.create(
                model=self.config.unconscious_model,
                max_tokens=2000,
                system=UNCONSCIOUS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": context}],
            )

            raw = response.content[0].text.strip()
            cleaned = raw.removeprefix("```json").removesuffix("```").strip()
            data = json.loads(cleaned)

            for item in data.get("impressions", []):
                await self._process_impression(item)

            for task in data.get("tasks_to_delegate", []):
                await self._delegate_task(task)

        except Exception as e:
            print(f"[UNCONSCIOUS] Deep cycle failed: {e}")

    async def _process_impression(self, item: dict):
        imp = Impression(
            type=ImpressionType(item.get("type", "pattern")),
            content=item.get("content", ""),
            payload=item.get("payload", {}),
            emotional_charge=float(item.get("emotional_charge", 0.0)),
            source_conversation=item.get("source_conversation", ""),
            source_turns=item.get("source_turns", []),
        )

        imp.embedding = self.idea_space.embed(imp.content)
        imp.payload["similarity_key"] = item.get("similarity_key", "")
        urgency = item.get("urgency", "low")

        # Critical interrupt path
        if urgency == "critical" and imp.type == ImpressionType.CORRECTION:
            correction = imp.payload.get("correction", imp.content)
            conv_id = imp.source_conversation
            if conv_id:
                await self.state.create_interrupt(
                    conv_id=conv_id,
                    content=correction,
                    urgency=0.95,
                    reason=imp.payload.get("what_went_wrong", "unconscious correction"),
                )
                return

        # Normal path: store or reinforce
        await self._store_or_reinforce(imp, urgency)

    async def _store_or_reinforce(self, imp: Impression, urgency: str):
        sim_key = imp.payload.get("similarity_key", "")
        existing = await self.state.get_active_impressions()

        # Check by similarity key
        for ex in existing:
            ex_payload = (
                json.loads(ex["payload"])
                if isinstance(ex["payload"], str)
                else ex["payload"]
            )
            if sim_key and ex_payload.get("similarity_key") == sim_key:
                boost = self._calculate_pressure(imp, urgency)
                await self.state.reinforce_impression(ex["id"], boost)
                return

        # Check by embedding proximity
        if imp.embedding:
            similar = await self.idea_space.find_similar(
                imp.embedding, threshold=self.config.convergence_threshold
            )
            if similar:
                closest, similarity = similar[0]
                if not closest.get("promoted") and not closest.get("repressed"):
                    boost = self._calculate_pressure(imp, urgency) * similarity
                    await self.state.reinforce_impression(closest["id"], boost)
                    return

        # New unique impression
        imp.pressure = self._calculate_pressure(imp, urgency)
        imp.pressure = max(
            self.config.initial_pressure_range[0],
            min(self.config.initial_pressure_range[1], imp.pressure),
        )
        await self.state.store_impression(imp)

    def _calculate_pressure(self, imp: Impression, urgency: str) -> float:
        base = {"low": 0.15, "medium": 0.3, "high": 0.5, "critical": 0.9}
        return base.get(urgency, 0.15) + abs(imp.emotional_charge) * 0.3

    def _apply_urgency_decay(self):
        """Logarithmic urgency decay for stale impressions."""
        # This runs synchronously — we schedule the async work
        asyncio.ensure_future(self._async_urgency_decay())

    async def _async_urgency_decay(self):
        now = time.time()
        for imp in await self.state.get_active_impressions():
            age = now - imp.get("last_reinforced_at", imp.get("created_at", now))
            decay = self.config.urgency_decay_rate * math.log(
                1 + age / self.config.urgency_decay_tau
            )
            new_pressure = max(0.0, imp["pressure"] * (1 - decay))
            if abs(new_pressure - imp["pressure"]) > 0.001:
                await self.state.update_impression_pressure(imp["id"], new_pressure)

    async def _delegate_task(self, task_data: dict):
        task_id = await self.state.create_task(
            conv_id=task_data.get("conversation_id", ""),
            description=task_data.get("description", ""),
            prompt=task_data.get("prompt", ""),
        )

    async def _dispatch_tasks(self):
        pending = await self.state.get_pending_tasks()
        for task in pending[:3]:
            asyncio.create_task(self._run_subagent(task))

    async def _run_subagent(self, task: dict):
        async with self._subagent_semaphore:
            task_id = task["id"]
            await self.state.update_task(task_id, "running")
            try:
                response = await asyncio.wait_for(
                    self.client.messages.create(
                        model=self.config.subagent_model,
                        max_tokens=2000,
                        messages=[{"role": "user", "content": task["prompt"]}],
                    ),
                    timeout=120.0,
                )
                result = response.content[0].text
                await self.state.update_task(task_id, "complete", result)
            except Exception as e:
                await self.state.update_task(task_id, "failed", str(e))
