"""Tests for the Defense Profile system."""

from __future__ import annotations

import pytest

from anima.models import DefenseLevel, DefenseMechanism
from anima.systems.defense import DefenseProfile


class TestDefenseProfile:
    def test_initial_state(self):
        profile = DefenseProfile()
        assert profile.maturity_score == 3.0
        assert profile.flexibility_score == 0.5
        assert profile.growth_velocity == 0.0

    def test_record_single_defense(self):
        profile = DefenseProfile()
        profile.record_defense_use(DefenseMechanism.SUBLIMATION, True)
        assert profile.usage_counts["sublimation"] == 1
        assert profile.maturity_score == 4.0  # Only mature defense used

    def test_record_multiple_defenses(self):
        profile = DefenseProfile()
        profile.record_defense_use(DefenseMechanism.DENIAL, False)
        profile.record_defense_use(DefenseMechanism.SUBLIMATION, True)
        # Average of pathological (1) and mature (4) = 2.5
        assert profile.maturity_score == pytest.approx(2.5)

    def test_flexibility_increases_with_variety(self):
        profile = DefenseProfile()
        # Use many different defenses
        for d in list(DefenseMechanism)[:5]:
            profile.record_defense_use(d, True)
        flex_diverse = profile.flexibility_score

        # Compare with using only one defense
        profile2 = DefenseProfile()
        for _ in range(5):
            profile2.record_defense_use(DefenseMechanism.REPRESSION, True)
        flex_rigid = profile2.flexibility_score

        assert flex_diverse > flex_rigid

    def test_flexibility_zero_for_single_defense(self):
        profile = DefenseProfile()
        for _ in range(3):
            profile.record_defense_use(DefenseMechanism.REPRESSION, True)
        assert profile.flexibility_score == 0.0

    def test_defense_success_rate(self):
        profile = DefenseProfile()
        profile.record_defense_use(DefenseMechanism.HUMOR, True)
        profile.record_defense_use(DefenseMechanism.HUMOR, True)
        profile.record_defense_use(DefenseMechanism.HUMOR, False)
        rate = profile.get_defense_success_rate(DefenseMechanism.HUMOR)
        assert rate == pytest.approx(2 / 3)

    def test_unknown_defense_success_rate(self):
        profile = DefenseProfile()
        rate = profile.get_defense_success_rate(DefenseMechanism.ALTRUISM)
        assert rate == 0.5  # Unknown = neutral

    def test_outcomes_capped_at_50(self):
        profile = DefenseProfile()
        for _ in range(60):
            profile.record_defense_use(DefenseMechanism.REPRESSION, True)
        assert len(profile.defense_outcomes["repression"]) <= 50


class TestHealthReport:
    def test_health_report_structure(self):
        profile = DefenseProfile()
        profile.record_defense_use(DefenseMechanism.SUBLIMATION, True)
        report = profile.get_health_report()
        assert "maturity_score" in report
        assert "maturity_label" in report
        assert "flexibility_score" in report
        assert "growth_velocity" in report
        assert "growth_direction" in report
        assert "most_used_defenses" in report
        assert "most_effective_defenses" in report
        assert "warning_signs" in report
        assert "total_events" in report
        assert "is_neurotic" in report

    def test_maturity_labels(self):
        profile = DefenseProfile()
        # Only mature defenses
        for _ in range(5):
            profile.record_defense_use(DefenseMechanism.SUBLIMATION, True)
        assert profile.get_health_report()["maturity_label"] == "mature"

        # Only pathological defenses
        profile2 = DefenseProfile()
        for _ in range(5):
            profile2.record_defense_use(DefenseMechanism.DENIAL, False)
        assert profile2.get_health_report()["maturity_label"] == "pathological"

    def test_growth_direction(self):
        profile = DefenseProfile()
        # Start with low-level defenses, then high-level
        for _ in range(5):
            profile.record_defense_use(DefenseMechanism.DENIAL, False)
        for _ in range(5):
            profile.record_defense_use(DefenseMechanism.SUBLIMATION, True)
        report = profile.get_health_report()
        assert report["growth_velocity"] > 0

    def test_warning_signs_low_maturity(self):
        profile = DefenseProfile()
        for _ in range(5):
            profile.record_defense_use(DefenseMechanism.DENIAL, False)
        warnings = profile.get_health_report()["warning_signs"]
        assert any("primitive" in w.lower() for w in warnings)

    def test_warning_signs_overreliance(self):
        profile = DefenseProfile()
        for _ in range(10):
            profile.record_defense_use(DefenseMechanism.DENIAL, False)
        for _ in range(2):
            profile.record_defense_use(DefenseMechanism.SUBLIMATION, True)
        warnings = profile.get_health_report()["warning_signs"]
        assert any("over-reliance" in w.lower() for w in warnings)

    def test_is_neurotic(self):
        profile = DefenseProfile()
        for _ in range(10):
            profile.record_defense_use(DefenseMechanism.REPRESSION, False)
        report = profile.get_health_report()
        assert report["is_neurotic"] is True

    def test_most_used_defenses(self):
        profile = DefenseProfile()
        for _ in range(5):
            profile.record_defense_use(DefenseMechanism.REPRESSION, True)
        for _ in range(2):
            profile.record_defense_use(DefenseMechanism.HUMOR, True)
        top = profile.get_health_report()["most_used_defenses"]
        assert top[0]["defense"] == "repression"
        assert top[0]["count"] == 5

    def test_most_effective_defenses(self):
        profile = DefenseProfile()
        for _ in range(5):
            profile.record_defense_use(DefenseMechanism.HUMOR, True)
        for _ in range(5):
            profile.record_defense_use(DefenseMechanism.DENIAL, False)
        effective = profile.get_health_report()["most_effective_defenses"]
        if effective:
            assert effective[0]["defense"] == "humor"
