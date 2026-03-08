"""Preconscious Layer (Sonnet) — defense selection and impression gating."""

from __future__ import annotations

import asyncio
import json
import math

from ..config import MindConfig
from ..llm import complete
from ..models import (
    DefenseLevel,
    DefenseMechanism,
    DEFENSE_LEVELS,
)
from ..prompts.preconscious_prompts import DEFENSE_SELECTOR_PROMPT
from ..state import SharedState
from ..systems.defense import DefenseProfile
from ..systems.idea_space import IdeaSpace


class PreconsciousLayer:
    """
    Sonnet gatekeeper. Evaluates above-threshold impressions using
    Freudian defense mechanisms. 8-second async loop.
    """

    def __init__(
        self,
        state: SharedState,
        idea_space: IdeaSpace,
        defense_profile: DefenseProfile,
        config: MindConfig,
    ):
        self.state = state
        self.idea_space = idea_space
        self.defense_profile = defense_profile
        self.config = config
        self._running = False
        self._task: asyncio.Task | None = None

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
                # 1. Check convergence clusters
                await self._check_convergence()

                # 2. Get above-threshold candidates
                candidates = [
                    imp
                    for imp in await self.state.get_active_impressions()
                    if imp["pressure"] >= self.config.pressure_threshold
                ]

                if candidates:
                    # 3. Evaluate each candidate with defense selection
                    for candidate in candidates[:8]:
                        await self._evaluate_candidate(candidate)

                await asyncio.sleep(self.config.preconscious_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"[PRECONSCIOUS] Error: {e}")
                await asyncio.sleep(self.config.preconscious_interval)

    async def _check_convergence(self):
        clusters = await self.idea_space.find_clusters(
            threshold=self.config.convergence_threshold,
            min_cluster_size=self.config.min_cluster_size,
        )
        for cluster in clusters:
            boost = self.idea_space.check_convergence(
                cluster, self.config.convergence_coefficient
            )
            if boost > 0:
                for imp in cluster:
                    if not imp.get("promoted") and not imp.get("repressed"):
                        await self.state.reinforce_impression(imp["id"], boost)

    async def _evaluate_candidate(self, candidate: dict):
        """Run defense selection for a single candidate impression."""
        health = self.defense_profile.get_health_report()
        recent_defense_events = await self.state.get_recent_defense_events(10)
        active_patterns = await self.state.get_active_patterns()

        is_moral_tension = candidate.get("type") == "moral_tension"

        context_parts = [
            "=== IMPRESSION TO EVALUATE ===",
            f"ID: {candidate['id']}",
            f"Type: {candidate['type']}",
            f"Content: {candidate['content']}",
            f"Pressure: {candidate['pressure']:.2f}",
            f"Emotional charge: {candidate.get('emotional_charge', 0):+.2f}",
            f"Reinforced: {candidate.get('times_reinforced', 0)}x",
        ]

        if is_moral_tension:
            context_parts.append(
                "\n⚡ SUPEREGO WEIGHT: This is a MORAL_TENSION impression. "
                "It carries superego authority. NEVER repress or deny moral tension. "
                "Prefer mature defenses: sublimation (→ directive), anticipation, altruism. "
                "The goal is to restore value alignment, not suppress the signal."
            )

        context_parts.append("\n=== SYSTEM HEALTH ===")
        context_parts.append(json.dumps(health, indent=2))

        if recent_defense_events:
            context_parts.append("\n=== RECENT DEFENSE LOG ===")
            for e in recent_defense_events:
                outcome = "positive" if e.get("led_to_positive") else "negative/unknown"
                context_parts.append(f"- {e['defense']}: {e.get('action_taken', '')} → {outcome}")

        if active_patterns:
            context_parts.append("\n=== ACTIVE NEUROTIC PATTERNS ===")
            for p in active_patterns:
                context_parts.append(
                    f"- [{p.get('severity', '?')}] {p.get('description', '')[:80]}"
                )

        try:
            response = await complete(
                model=self.config.preconscious_model,
                max_tokens=800,
                system=DEFENSE_SELECTOR_PROMPT,
                messages=[{"role": "user", "content": "\n".join(context_parts)}],
            )

            raw = response.text.strip()
            cleaned = raw.removeprefix("```json").removesuffix("```").strip()
            data = json.loads(cleaned)

            await self._apply_defense_decision(candidate, data)

        except Exception as e:
            print(f"[PRECONSCIOUS] Defense selection failed: {e}")

    async def _apply_defense_decision(self, candidate: dict, decision: dict):
        """Apply the defense mechanism's action mapping per ARCHITECTURE.md §6."""
        defense_name = decision.get("selected_defense", "repression")
        try:
            defense = DefenseMechanism(defense_name)
        except ValueError:
            defense = DefenseMechanism.REPRESSION

        level = DEFENSE_LEVELS.get(defense, DefenseLevel.NEUROTIC)
        action = decision.get("action", "")
        promotion_data = decision.get("promotion")

        # Log defense event
        await self.state.log_defense_event(
            impression_id=candidate["id"],
            defense=defense.value,
            level=level.value,
            action_taken=action,
        )

        # Apply the defense → action mapping
        if defense == DefenseMechanism.SUBLIMATION:
            # Promote as TOOL
            if promotion_data:
                await self.state.upsert_promotion(
                    ptype="tool",
                    key=promotion_data.get("key", f"tool_{candidate['id']}"),
                    content=promotion_data.get("content", {}),
                    confidence=promotion_data.get("confidence", 0.8),
                    impression_ids=[candidate["id"]],
                )
                await self.state.mark_promoted(candidate["id"])

        elif defense == DefenseMechanism.ANTICIPATION:
            # Promote as DIRECTIVE (proactive)
            if promotion_data:
                await self.state.upsert_promotion(
                    ptype="directive",
                    key=promotion_data.get("key", f"anticipate_{candidate['id']}"),
                    content=promotion_data.get("content", {"instruction": action}),
                    confidence=promotion_data.get("confidence", 0.8),
                    impression_ids=[candidate["id"]],
                )
                await self.state.mark_promoted(candidate["id"])

        elif defense == DefenseMechanism.HUMOR:
            # Promote as DIRECTIVE (self-aware)
            if promotion_data:
                await self.state.upsert_promotion(
                    ptype="directive",
                    key=promotion_data.get("key", f"humor_{candidate['id']}"),
                    content=promotion_data.get("content", {"instruction": action}),
                    confidence=promotion_data.get("confidence", 0.7),
                    impression_ids=[candidate["id"]],
                )
                await self.state.mark_promoted(candidate["id"])

        elif defense == DefenseMechanism.SUPPRESSION:
            # Hold with timer — just leave it, pressure continues, re-evaluate later
            pass

        elif defense == DefenseMechanism.ALTRUISM:
            # Promote as DIRECTIVE (user-benefit)
            if promotion_data:
                await self.state.upsert_promotion(
                    ptype="directive",
                    key=promotion_data.get("key", f"altruism_{candidate['id']}"),
                    content=promotion_data.get("content", {"instruction": action}),
                    confidence=promotion_data.get("confidence", 0.8),
                    impression_ids=[candidate["id"]],
                )
                await self.state.mark_promoted(candidate["id"])

        elif defense == DefenseMechanism.REPRESSION:
            # Push back to buffer, pressure continues
            await self.state.mark_repressed(candidate["id"])

        elif defense == DefenseMechanism.DISPLACEMENT:
            # Transfer pressure to idea space neighbor
            await self.idea_space.displace(
                candidate, alpha=self.config.displacement_alpha
            )

        elif defense == DefenseMechanism.RATIONALIZATION:
            # Log justification, reduce pressure 30%
            new_pressure = candidate["pressure"] * 0.7
            await self.state.update_impression_pressure(candidate["id"], new_pressure)

        elif defense == DefenseMechanism.REACTION_FORMATION:
            # Promote OPPOSITE (flag for monitoring)
            if promotion_data:
                await self.state.upsert_promotion(
                    ptype="directive",
                    key=promotion_data.get("key", f"reaction_{candidate['id']}"),
                    content=promotion_data.get("content", {"instruction": action}),
                    confidence=promotion_data.get("confidence", 0.6),
                    impression_ids=[candidate["id"]],
                )
                await self.state.mark_promoted(candidate["id"])

        elif defense == DefenseMechanism.INTELLECTUALIZATION:
            # Store as MEMORY without behavioral change
            content = candidate.get("content", "")
            await self.state.upsert_promotion(
                ptype="memory",
                key=f"fact_{candidate['id']}",
                content={"fact": content, "category": "context"},
                confidence=0.6,
                impression_ids=[candidate["id"]],
            )
            await self.state.mark_promoted(candidate["id"])

        elif defense == DefenseMechanism.PROJECTION:
            # Log (growth engine should upgrade)
            pass

        elif defense == DefenseMechanism.SPLITTING:
            # Binary accept/reject (growth should upgrade)
            pass

        elif defense == DefenseMechanism.PASSIVE_AGGRESSION:
            # Log (actively discouraged)
            pass

        elif defense == DefenseMechanism.DENIAL:
            # Drop entirely
            await self.state.mark_repressed(candidate["id"])

        elif defense == DefenseMechanism.DISTORTION:
            # Transform meaning (growth should prevent)
            await self.state.mark_repressed(candidate["id"])
