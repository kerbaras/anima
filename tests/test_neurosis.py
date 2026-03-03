"""Tests for Neurosis Detection."""

from __future__ import annotations

from freudian_mind.models import OutcomeSignal, ResponseOutcome
from freudian_mind.systems.neurosis import RepetitionDetector


class TestCorrectionLoops:
    def test_detects_correction_loop(self):
        detector = RepetitionDetector(correction_threshold=3)
        for i in range(4):
            detector.record_outcome(
                ResponseOutcome(signal=OutcomeSignal.CORRECTION)
            )
        patterns = detector.detect_patterns()
        correction_loops = [
            p for p in patterns if p["pattern_type"] == "correction_loop"
        ]
        assert len(correction_loops) >= 1

    def test_no_loop_below_threshold(self):
        detector = RepetitionDetector(correction_threshold=3)
        for i in range(2):
            detector.record_outcome(
                ResponseOutcome(signal=OutcomeSignal.CORRECTION)
            )
        patterns = detector.detect_patterns()
        correction_loops = [
            p for p in patterns if p["pattern_type"] == "correction_loop"
        ]
        assert len(correction_loops) == 0


class TestRepressionLoops:
    def test_detects_repression_loop(self):
        detector = RepetitionDetector(repression_threshold=3)
        defense_log = [
            {"defense": "repression", "impression_id": "imp1"},
            {"defense": "repression", "impression_id": "imp1"},
            {"defense": "repression", "impression_id": "imp1"},
        ]
        patterns = detector.detect_patterns(defense_log)
        repression_loops = [
            p for p in patterns if p["pattern_type"] == "repression_loop"
        ]
        assert len(repression_loops) >= 1

    def test_no_loop_different_impressions(self):
        detector = RepetitionDetector(repression_threshold=3)
        defense_log = [
            {"defense": "repression", "impression_id": "imp1"},
            {"defense": "repression", "impression_id": "imp2"},
            {"defense": "repression", "impression_id": "imp3"},
        ]
        patterns = detector.detect_patterns(defense_log)
        repression_loops = [
            p for p in patterns if p["pattern_type"] == "repression_loop"
        ]
        assert len(repression_loops) == 0


class TestAvoidanceLoops:
    def test_detects_avoidance(self):
        detector = RepetitionDetector(correction_threshold=3)
        for i in range(4):
            detector.record_outcome(
                ResponseOutcome(signal=OutcomeSignal.ABANDONMENT)
            )
        patterns = detector.detect_patterns()
        avoidance = [
            p for p in patterns if p["pattern_type"] == "avoidance_loop"
        ]
        assert len(avoidance) >= 1


class TestEscalationSpirals:
    def test_detects_escalation(self):
        detector = RepetitionDetector(escalation_window=5)
        # 6 consecutive negative signals
        for _ in range(6):
            detector.record_outcome(
                ResponseOutcome(signal=OutcomeSignal.FRUSTRATION)
            )
        patterns = detector.detect_patterns()
        spirals = [
            p for p in patterns if p["pattern_type"] == "escalation_spiral"
        ]
        assert len(spirals) >= 1

    def test_no_spiral_with_breaks(self):
        detector = RepetitionDetector(escalation_window=5)
        for _ in range(3):
            detector.record_outcome(
                ResponseOutcome(signal=OutcomeSignal.FRUSTRATION)
            )
        detector.record_outcome(
            ResponseOutcome(signal=OutcomeSignal.POSITIVE)
        )
        for _ in range(3):
            detector.record_outcome(
                ResponseOutcome(signal=OutcomeSignal.CORRECTION)
            )
        patterns = detector.detect_patterns()
        spirals = [
            p for p in patterns if p["pattern_type"] == "escalation_spiral"
        ]
        assert len(spirals) == 0

    def test_severity_increases_with_length(self):
        detector = RepetitionDetector(escalation_window=5)
        for _ in range(8):
            detector.record_outcome(
                ResponseOutcome(signal=OutcomeSignal.ESCALATION)
            )
        patterns = detector.detect_patterns()
        spirals = [
            p for p in patterns if p["pattern_type"] == "escalation_spiral"
        ]
        assert spirals[0]["severity"] == "CRITICAL"
