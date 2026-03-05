"""FreudianMind — the complete three-layer orchestrator."""

from __future__ import annotations

import asyncio

from .config import MindConfig
from .layers.conscious import ConsciousLayer
from .layers.preconscious import PreconsciousLayer
from .layers.unconscious import UnconsciousLayer
from .models import MessageBurst, ResponseOutcome
from .state import SharedState
from .systems.defense import DefenseProfile
from .systems.growth import GrowthEngine
from .systems.idea_space import IdeaSpace
from .systems.neurosis import RepetitionDetector
from .systems.outcome import OutcomeClassifier


class FreudianMind:
    """
    The complete three-layer mind.

    One Opus brain. One Sonnet filter. N Haiku voices (one per conversation).
    """

    def __init__(self, config: MindConfig | None = None):
        self.config = config or MindConfig()
        self.state = SharedState(self.config.db_path)
        self.idea_space = IdeaSpace(self.state)
        self.defense_profile = DefenseProfile()
        self.repetition_detector = RepetitionDetector(
            correction_threshold=self.config.correction_loop_threshold,
            repression_threshold=self.config.repression_loop_threshold,
            escalation_window=self.config.escalation_window,
        )
        self.growth_engine = GrowthEngine(
            self.defense_profile, self.repetition_detector
        )
        self.outcome_classifier = OutcomeClassifier(self.config.classifier_model)

        self.unconscious = UnconsciousLayer(
            self.state, self.idea_space, self.config
        )
        self.preconscious = PreconsciousLayer(
            self.state, self.idea_space, self.defense_profile, self.config
        )
        self.conscious = ConsciousLayer(self.state, self.config)
        self.conversations: dict[str, int] = {}

    async def start(self):
        await self.state.initialize()
        await asyncio.gather(
            self.unconscious.start(),
            self.preconscious.start(),
        )

    async def stop(self):
        await self.unconscious.stop()
        await self.preconscious.stop()
        await self.state.close()

    async def new_conversation(self) -> str:
        conv_id = await self.state.create_conversation()
        self.conversations[conv_id] = 0
        return conv_id

    async def chat(self, conv_id: str, user_message: str, bridge_context: str = "") -> MessageBurst:
        if conv_id not in self.conversations:
            self.conversations[conv_id] = await self.state.get_turn_count(conv_id)

        self.conversations[conv_id] += 1
        turn = self.conversations[conv_id]

        # Log user turn
        await self.state.log_turn(conv_id, turn, "user", user_message)

        # Classify outcome of PREVIOUS exchange
        if turn > 1:
            prev = await self.state.get_last_assistant_message(conv_id)
            if prev:
                outcome = await self.outcome_classifier.classify(
                    prev, user_message, conv_id, turn
                )
                await self.state.log_outcome(outcome)
                self.repetition_detector.record_outcome(outcome)
                self._update_last_defense_outcome(outcome)

        # Generate response
        burst = await self.conscious.respond(conv_id, user_message, bridge_context=bridge_context)
        burst.turn_number = turn

        # Log assistant messages
        for i, msg in enumerate(burst.messages):
            await self.state.log_turn(conv_id, turn, "assistant", msg, burst_index=i)

        # Update health report for unconscious
        self.unconscious.set_health_report(self.defense_profile.get_health_report())

        return burst

    def _update_last_defense_outcome(self, outcome: ResponseOutcome):
        """Update the most recent defense event with the outcome signal."""
        is_positive = outcome.signal in ("positive", "delight", "neutral")
        # This would update the defense profile based on outcomes
        # For now, tracked through the defense_events table
