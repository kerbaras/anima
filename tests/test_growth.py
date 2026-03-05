"""Tests for the Growth Engine."""

from __future__ import annotations

from anima.models import (
    DefenseMechanism,
    GrowthMechanism,
    OutcomeSignal,
    ResponseOutcome,
)
from anima.systems.defense import DefenseProfile
from anima.systems.growth import GrowthEngine
from anima.systems.neurosis import RepetitionDetector


def _make_engine(
    profile: DefenseProfile | None = None,
    detector: RepetitionDetector | None = None,
) -> GrowthEngine:
    return GrowthEngine(
        defense_profile=profile or DefenseProfile(),
        repetition_detector=detector or RepetitionDetector(),
    )


class TestRepetitionAddressing:
    def test_repression_loop_triggers_integration(self):
        detector = RepetitionDetector(correction_threshold=3)
        for _ in range(6):
            detector.record_outcome(
                ResponseOutcome(signal=OutcomeSignal.CORRECTION)
            )
        engine = _make_engine(detector=detector)
        defense_log = []
        repressions = []
        actions = engine.run_therapeutic_cycle(defense_log, repressions)
        integration = [a for a in actions if a["type"] == "integration"]
        assert len(integration) >= 1

    def test_denial_loop_triggers_upgrade(self):
        detector = RepetitionDetector(correction_threshold=3)
        for _ in range(6):
            detector.record_outcome(
                ResponseOutcome(signal=OutcomeSignal.ABANDONMENT)
            )
        engine = _make_engine(detector=detector)
        actions = engine.run_therapeutic_cycle([], [])
        upgrades = [a for a in actions if a.get("old_defense") == "denial"]
        assert len(upgrades) >= 1


class TestDefenseUpgrades:
    def test_recommends_upgrade_for_failing_defense(self):
        profile = DefenseProfile()
        for _ in range(5):
            profile.record_defense_use(DefenseMechanism.DENIAL, False)
        engine = _make_engine(profile=profile)
        actions = engine.run_therapeutic_cycle([], [])
        upgrades = [a for a in actions if a["type"] == "defense_upgrade"]
        assert len(upgrades) >= 1

    def test_no_upgrade_for_successful_defense(self):
        profile = DefenseProfile()
        for _ in range(5):
            profile.record_defense_use(DefenseMechanism.SUBLIMATION, True)
        engine = _make_engine(profile=profile)
        actions = engine.run_therapeutic_cycle([], [])
        upgrades = [
            a
            for a in actions
            if a["type"] == "defense_upgrade"
            and a.get("old_defense") == "sublimation"
        ]
        assert len(upgrades) == 0


class TestRigidityInsight:
    def test_rigid_defense_triggers_insight(self):
        profile = DefenseProfile()
        for _ in range(10):
            profile.record_defense_use(DefenseMechanism.REPRESSION, False)
        engine = _make_engine(profile=profile)
        actions = engine.run_therapeutic_cycle([], [])
        insights = [a for a in actions if a["type"] == "insight"]
        assert len(insights) >= 1


class TestSublimation:
    def test_skill_repression_triggers_sublimation(self):
        engine = _make_engine()
        repressions = [
            {"id": "imp1", "type": "skill", "content": "user wants code formatting"},
        ]
        actions = engine.run_therapeutic_cycle([], repressions)
        sublimations = [a for a in actions if a["type"] == "sublimation"]
        assert len(sublimations) >= 1

    def test_correction_repression_triggers_integration(self):
        engine = _make_engine()
        repressions = [
            {"id": "imp2", "type": "correction", "content": "wrong answer about X"},
        ]
        actions = engine.run_therapeutic_cycle([], repressions)
        integrations = [a for a in actions if a["type"] == "integration"]
        assert len(integrations) >= 1


class TestWorkingThrough:
    def test_working_through_with_severe_pattern(self):
        detector = RepetitionDetector(correction_threshold=3)
        for _ in range(5):
            detector.record_outcome(
                ResponseOutcome(signal=OutcomeSignal.CORRECTION)
            )
        engine = _make_engine(detector=detector)
        repressions = [
            {"id": "imp1", "type": "pattern", "content": "keeps making same error"},
        ]
        actions = engine.run_therapeutic_cycle([], repressions)
        wt = [a for a in actions if a["type"] == "working_through"]
        assert len(wt) >= 1
