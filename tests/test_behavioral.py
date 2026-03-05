"""Behavioral tests — cross-component flows through the real system.

These tests validate that the Freudian Mind *behaves* correctly under
realistic multi-step scenarios.  LLM calls are mocked; everything else
(state, defense profile, neurosis detector, growth engine, idea space)
runs for real.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from anima.config import MindConfig
from anima.models import (
    DefenseMechanism,
    GrowthMechanism,
    Impression,
    ImpressionType,
    OutcomeSignal,
    ResponseOutcome,
)
from anima.systems.defense import DefenseProfile
from anima.systems.growth import GrowthEngine
from anima.systems.neurosis import RepetitionDetector

from conftest import make_classifier_response, make_llm_response


# ── Scenario 1: Correction loop → Neurosis detection → Growth intervention ──


class TestCorrectionLoopTriggersGrowth:
    """When a user repeatedly corrects the system, the neurosis detector must
    fire a correction_loop pattern and the growth engine must recommend
    integration / working-through to break the loop."""

    @patch("anima.systems.outcome.complete")
    @patch("anima.layers.conscious.complete")
    async def test_repeated_corrections_trigger_neurosis_and_growth(
        self, mock_conscious, mock_outcome, mind
    ):
        mock_conscious.return_value = make_llm_response("Got it. [DONE]")
        mock_outcome.return_value = make_classifier_response("correction", user_had_to_repeat=True)

        conv_id = await mind.new_conversation()

        # Turn 1 — no classification (first turn)
        await mind.chat(conv_id, "What is X?")

        # Turns 2-7: user keeps correcting → outcome = correction each time
        for i in range(6):
            mock_conscious.return_value = make_llm_response(f"Sorry, fixed. [DONE]")
            await mind.chat(conv_id, f"No, that's wrong again (attempt {i+2})")

        # Neurosis detector should have recorded 6 correction outcomes
        assert len(mind.repetition_detector.outcome_history) == 6
        patterns = mind.repetition_detector.detect_patterns()
        correction_patterns = [p for p in patterns if p["pattern_type"] == "correction_loop"]
        assert len(correction_patterns) == 1
        assert correction_patterns[0]["severity"] == "HIGH"

        # Growth engine should recommend actions to break the loop
        actions = mind.growth_engine.run_therapeutic_cycle([], [])
        integration_or_wt = [
            a for a in actions
            if a["type"] in ("integration", "working_through")
        ]
        assert len(integration_or_wt) >= 1

    @patch("anima.systems.outcome.complete")
    @patch("anima.layers.conscious.complete")
    async def test_severity_escalates_from_medium_to_high(
        self, mock_conscious, mock_outcome, mind
    ):
        """More corrections → severity escalates to HIGH."""
        mock_conscious.return_value = make_llm_response("OK [DONE]")
        mock_outcome.return_value = make_classifier_response("correction")

        conv_id = await mind.new_conversation()
        await mind.chat(conv_id, "Start")

        # 6 correction turns → should be HIGH severity (threshold is 5)
        for i in range(6):
            await mind.chat(conv_id, f"Wrong again #{i}")

        patterns = mind.repetition_detector.detect_patterns()
        correction_patterns = [p for p in patterns if p["pattern_type"] == "correction_loop"]
        assert correction_patterns[0]["severity"] == "HIGH"


# ── Scenario 2: Full chat flow with outcome feedback ──


class TestChatOutcomeFeedbackLoop:
    """Outcome classification feeds into the repetition detector and defense
    profile, forming a feedback loop that drives system adaptation."""

    @patch("anima.systems.outcome.complete")
    @patch("anima.layers.conscious.complete")
    async def test_positive_outcomes_dont_trigger_neurosis(
        self, mock_conscious, mock_outcome, mind
    ):
        mock_conscious.return_value = make_llm_response("Here you go! [DONE]")
        mock_outcome.return_value = make_classifier_response("positive")

        conv_id = await mind.new_conversation()
        await mind.chat(conv_id, "Hello")

        for _ in range(5):
            await mind.chat(conv_id, "Thanks, that's great")

        patterns = mind.repetition_detector.detect_patterns()
        assert len(patterns) == 0, "Positive outcomes should NOT trigger neurosis"

    @patch("anima.systems.outcome.complete")
    @patch("anima.layers.conscious.complete")
    async def test_mixed_outcomes_below_threshold_no_neurosis(
        self, mock_conscious, mock_outcome, mind
    ):
        """Fewer than threshold corrections should NOT fire neurosis."""
        mock_conscious.return_value = make_llm_response("Response [DONE]")
        conv_id = await mind.new_conversation()
        await mind.chat(conv_id, "Start")

        # 2 corrections (below threshold of 3) interleaved with positives
        signals = ["correction", "positive", "correction", "positive"]
        for sig in signals:
            mock_outcome.return_value = make_classifier_response(sig)
            await mind.chat(conv_id, "next turn")

        patterns = mind.repetition_detector.detect_patterns()
        correction_patterns = [p for p in patterns if p["pattern_type"] == "correction_loop"]
        assert len(correction_patterns) == 0


# ── Scenario 3: Critical correction → Interrupt → Conscious injection ──


class TestCriticalCorrectionCreatesInterrupt:
    """When the unconscious processes a critical correction impression,
    it must create an interrupt that the conscious layer consumes."""

    @patch("anima.layers.conscious.complete")
    async def test_interrupt_consumed_by_conscious(self, mock_conscious, mind):
        mock_conscious.side_effect = [
            make_llm_response("Let me address that."),
            make_llm_response("[DONE]"),
        ]

        conv_id = await mind.new_conversation()

        # Simulate the unconscious creating an interrupt (as it would for a
        # critical correction impression)
        await mind.state.create_interrupt(
            conv_id=conv_id,
            content="You gave wrong information about Python GIL",
            urgency=0.95,
            reason="unconscious correction",
        )

        burst = await mind.chat(conv_id, "Tell me more about threading")

        assert len(burst.interrupts_applied) == 1
        # The interrupt should be consumed (uses_remaining decremented)
        remaining = await mind.state.consume_interrupts(conv_id)
        assert len(remaining) == 0, "Interrupt should be fully consumed"

    @patch("anima.layers.conscious.complete")
    async def test_interrupt_appears_in_system_prompt(self, mock_conscious, mind):
        """The conscious layer should include interrupt content in its system prompt."""
        calls = []

        async def capture_call(**kwargs):
            calls.append(kwargs)
            return make_llm_response("Understood. [DONE]")

        mock_conscious.side_effect = capture_call

        conv_id = await mind.new_conversation()
        await mind.state.create_interrupt(
            conv_id=conv_id,
            content="CRITICAL: stop using deprecated API",
            urgency=0.95,
        )

        await mind.chat(conv_id, "Help me with the API")

        # The system prompt sent to the LLM should contain the interrupt
        assert len(calls) >= 1
        system_prompt = calls[0].get("system", "")
        assert "CRITICAL: stop using deprecated API" in system_prompt


# ── Scenario 4: Defense profile degradation → Growth insight ──


class TestDefenseProfileDegradation:
    """When the system exclusively uses pathological defenses (denial) and
    they consistently fail, the growth engine must recommend upgrades
    and rigidity insights."""

    def test_denial_spiral_triggers_upgrade_and_insight(self):
        profile = DefenseProfile()
        detector = RepetitionDetector()

        # Simulate heavy denial usage with negative outcomes
        for _ in range(15):
            profile.record_defense_use(DefenseMechanism.DENIAL, False)

        health = profile.get_health_report()
        assert health["maturity_label"] == "pathological"
        assert health["flexibility_score"] < 0.25

        engine = GrowthEngine(profile, detector)
        actions = engine.run_therapeutic_cycle([], [])

        # Should recommend upgrading denial
        upgrades = [
            a for a in actions
            if a["type"] == "defense_upgrade" and a.get("old_defense") == "denial"
        ]
        assert len(upgrades) >= 1
        assert upgrades[0]["new_defense"] == "suppression"

        # Should flag rigidity
        insights = [a for a in actions if a["type"] == "insight"]
        assert len(insights) >= 1

    def test_mixed_mature_defenses_no_upgrade_needed(self):
        """A healthy profile should not trigger upgrades or insights."""
        profile = DefenseProfile()
        detector = RepetitionDetector()

        # Use diverse mature defenses with good outcomes
        mature = [
            DefenseMechanism.SUBLIMATION,
            DefenseMechanism.ANTICIPATION,
            DefenseMechanism.HUMOR,
            DefenseMechanism.SUPPRESSION,
            DefenseMechanism.ALTRUISM,
        ]
        for d in mature:
            for _ in range(4):
                profile.record_defense_use(d, True)

        health = profile.get_health_report()
        assert health["maturity_label"] == "mature"

        engine = GrowthEngine(profile, detector)
        actions = engine.run_therapeutic_cycle([], [])

        upgrades = [a for a in actions if a["type"] == "defense_upgrade"]
        insights = [a for a in actions if a["type"] == "insight"]
        assert len(upgrades) == 0
        assert len(insights) == 0


# ── Scenario 5: Convergence cluster → Pressure boost ──


class TestConvergenceCluster:
    """Multiple similar impressions should form a cluster, and convergence
    should boost their pressure."""

    async def test_cluster_detection_boosts_pressure(self, state):
        from anima.systems.idea_space import IdeaSpace

        idea_space = IdeaSpace(state)

        # Store 3 impressions with identical embeddings (same content = same hash)
        base_content = "user keeps asking about error handling"
        embedding = idea_space.embed(base_content)

        for i in range(3):
            imp = Impression(
                id=f"imp{i}",
                type=ImpressionType.PATTERN,
                content=base_content,
                embedding=embedding,
                pressure=0.3,
                emotional_charge=0.5,
            )
            await state.store_impression(imp)

        clusters = await idea_space.find_clusters(threshold=0.99, min_cluster_size=3)
        assert len(clusters) >= 1
        assert len(clusters[0]) == 3

        boost = idea_space.check_convergence(clusters[0], coefficient=0.15)
        assert boost > 0, "Convergence boost should be positive for emotional cluster"


# ── Scenario 6: Escalation spiral detection ──


class TestEscalationSpiralDetection:
    """A sequence of negative signals must trigger escalation spiral detection
    with correct severity."""

    def test_frustration_escalation_abandonment_spiral(self):
        detector = RepetitionDetector(escalation_window=5)

        # Simulate an escalation spiral: frustration → correction → escalation → frustration → correction
        spiral_signals = [
            OutcomeSignal.FRUSTRATION,
            OutcomeSignal.CORRECTION,
            OutcomeSignal.ESCALATION,
            OutcomeSignal.FRUSTRATION,
            OutcomeSignal.CORRECTION,
        ]
        for sig in spiral_signals:
            detector.record_outcome(ResponseOutcome(signal=sig))

        patterns = detector.detect_patterns()
        spirals = [p for p in patterns if p["pattern_type"] == "escalation_spiral"]
        assert len(spirals) == 1
        assert spirals[0]["severity"] == "HIGH"

    def test_critical_severity_at_7_consecutive(self):
        detector = RepetitionDetector(escalation_window=5)

        for _ in range(7):
            detector.record_outcome(ResponseOutcome(signal=OutcomeSignal.FRUSTRATION))

        patterns = detector.detect_patterns()
        spirals = [p for p in patterns if p["pattern_type"] == "escalation_spiral"]
        assert len(spirals) == 1
        assert spirals[0]["severity"] == "CRITICAL"

    def test_positive_breaks_spiral(self):
        """A positive outcome in the middle should break the streak."""
        detector = RepetitionDetector(escalation_window=5)

        for _ in range(3):
            detector.record_outcome(ResponseOutcome(signal=OutcomeSignal.FRUSTRATION))
        # Positive break
        detector.record_outcome(ResponseOutcome(signal=OutcomeSignal.POSITIVE))
        for _ in range(3):
            detector.record_outcome(ResponseOutcome(signal=OutcomeSignal.CORRECTION))

        patterns = detector.detect_patterns()
        spirals = [p for p in patterns if p["pattern_type"] == "escalation_spiral"]
        assert len(spirals) == 0, "Positive outcome should break the streak"

    def test_simultaneous_avoidance_and_correction_patterns(self):
        """Both avoidance_loop and correction_loop can fire simultaneously."""
        detector = RepetitionDetector(correction_threshold=3)

        for _ in range(4):
            detector.record_outcome(ResponseOutcome(signal=OutcomeSignal.CORRECTION))
        for _ in range(4):
            detector.record_outcome(ResponseOutcome(signal=OutcomeSignal.ABANDONMENT))

        patterns = detector.detect_patterns()
        pattern_types = {p["pattern_type"] for p in patterns}
        assert "correction_loop" in pattern_types
        assert "avoidance_loop" in pattern_types


# ── Scenario 7: Outcome → Defense feedback integration ──


class TestOutcomeDefenseFeedback:
    """Outcomes with defense_applied should feed back into the defense profile,
    affecting maturity scores over time."""

    def test_positive_mature_defense_outcomes_improve_maturity(self):
        profile = DefenseProfile()

        # Start with some immature defense usage to lower baseline
        for _ in range(3):
            profile.record_defense_use(DefenseMechanism.DENIAL, False)

        baseline_maturity = profile.maturity_score

        # Now successful mature defenses should improve score
        for _ in range(10):
            profile.record_defense_use(DefenseMechanism.SUBLIMATION, True)

        assert profile.maturity_score > baseline_maturity

    def test_negative_immature_defense_outcomes_degrade_maturity(self):
        profile = DefenseProfile()

        # Start with mature baseline
        for _ in range(5):
            profile.record_defense_use(DefenseMechanism.SUBLIMATION, True)

        baseline_maturity = profile.maturity_score

        # Flood with failing immature defenses
        for _ in range(20):
            profile.record_defense_use(DefenseMechanism.DENIAL, False)

        assert profile.maturity_score < baseline_maturity
        assert profile.get_health_report()["maturity_label"] in ("pathological", "immature")

    def test_warning_signs_detect_over_reliance(self):
        """Over-reliance on a single primitive defense should produce warnings."""
        profile = DefenseProfile()

        for _ in range(10):
            profile.record_defense_use(DefenseMechanism.DENIAL, False)

        health = profile.get_health_report()
        assert len(health["warning_signs"]) > 0
        has_overreliance = any("denial" in w.lower() for w in health["warning_signs"])
        assert has_overreliance


# ── Scenario: Repression loop detection via defense log ──


class TestRepressionLoopDetection:
    """When the same impression is repressed 3+ times, neurosis detector
    should fire a repression_loop pattern."""

    def test_repression_loop_detected_from_defense_log(self):
        detector = RepetitionDetector(repression_threshold=3)

        defense_log = [
            {"defense": "repression", "impression_id": "imp-A"},
            {"defense": "repression", "impression_id": "imp-A"},
            {"defense": "repression", "impression_id": "imp-A"},
            {"defense": "sublimation", "impression_id": "imp-B"},
        ]

        patterns = detector.detect_patterns(defense_log)
        repression_patterns = [p for p in patterns if p["pattern_type"] == "repression_loop"]
        assert len(repression_patterns) == 1
        assert repression_patterns[0]["maintaining_defense"] == "repression"

    def test_no_repression_loop_below_threshold(self):
        detector = RepetitionDetector(repression_threshold=3)

        defense_log = [
            {"defense": "repression", "impression_id": "imp-A"},
            {"defense": "repression", "impression_id": "imp-A"},
        ]

        patterns = detector.detect_patterns(defense_log)
        repression_patterns = [p for p in patterns if p["pattern_type"] == "repression_loop"]
        assert len(repression_patterns) == 0


# ── Scenario: Growth engine handles combined neurosis + repressions ──


class TestGrowthEngineComplex:
    """Growth engine should produce correct therapeutic actions when faced
    with both active neurotic patterns AND repressed skill/correction impressions."""

    def test_combined_neurosis_and_repressions(self):
        detector = RepetitionDetector(correction_threshold=3)
        profile = DefenseProfile()

        # Create neurosis: many corrections
        for _ in range(5):
            detector.record_outcome(ResponseOutcome(signal=OutcomeSignal.CORRECTION))

        # Create defense rigidity
        for _ in range(12):
            profile.record_defense_use(DefenseMechanism.REPRESSION, False)

        engine = GrowthEngine(profile, detector)

        repressions = [
            {"id": "imp1", "type": "skill", "content": "user wants code formatting"},
            {"id": "imp2", "type": "correction", "content": "wrong API endpoint"},
        ]

        actions = engine.run_therapeutic_cycle([], repressions)

        action_types = {a["type"] for a in actions}
        # Should have integration (for correction loop), sublimation (for skill repression),
        # integration (for correction repression), defense_upgrade, insight, and working_through
        assert "integration" in action_types
        assert "sublimation" in action_types
        assert "defense_upgrade" in action_types
        assert "insight" in action_types
        assert "working_through" in action_types
