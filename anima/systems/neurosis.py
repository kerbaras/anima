"""Neurosis Detection — detects repetition compulsion and pathological patterns."""

from __future__ import annotations

from ..models import OutcomeSignal, ResponseOutcome


class RepetitionDetector:
    """
    Analyzes outcome history to detect repetition compulsion.

    4 pattern types:
    1. Correction loops: same error corrected 3+ times
    2. Repression loops: same impression repressed 3+ times
    3. Avoidance loops: topic consistently produces abandonment
    4. Escalation spirals: consecutive negative signals with increasing severity
    """

    def __init__(
        self,
        correction_threshold: int = 3,
        repression_threshold: int = 3,
        escalation_window: int = 5,
    ):
        self.correction_threshold = correction_threshold
        self.repression_threshold = repression_threshold
        self.escalation_window = escalation_window
        self.outcome_history: list[ResponseOutcome] = []

    def record_outcome(self, outcome: ResponseOutcome):
        self.outcome_history.append(outcome)

    def detect_patterns(
        self,
        defense_log: list[dict] | None = None,
    ) -> list[dict]:
        """Run all 4 detection algorithms. Returns list of pattern dicts."""
        patterns = []
        patterns.extend(self._detect_correction_loops())
        patterns.extend(self._detect_repression_loops(defense_log or []))
        patterns.extend(self._detect_avoidance_loops())
        patterns.extend(self._detect_escalation_spirals())
        return patterns

    def _detect_correction_loops(self) -> list[dict]:
        corrections = [
            o for o in self.outcome_history if o.signal == OutcomeSignal.CORRECTION
        ]
        if len(corrections) >= self.correction_threshold:
            return [
                {
                    "pattern_type": "correction_loop",
                    "description": "User repeatedly corrects the system",
                    "trigger": "user asks similar questions",
                    "response": "system gives incorrect or incomplete answers",
                    "maintaining_defense": "repression",
                    "occurrence_count": len(corrections),
                    "severity": "HIGH" if len(corrections) >= 5 else "MEDIUM",
                }
            ]
        return []

    def _detect_repression_loops(self, defense_log: list[dict]) -> list[dict]:
        repressed = [d for d in defense_log if d.get("defense") == "repression"]
        key_counts: dict[str, int] = {}
        for d in repressed:
            key = d.get("impression_id", "")
            if key:
                key_counts[key] = key_counts.get(key, 0) + 1

        patterns = []
        for key, count in key_counts.items():
            if count >= self.repression_threshold:
                patterns.append(
                    {
                        "pattern_type": "repression_loop",
                        "description": f"Repeatedly repressing impression: {key}",
                        "trigger": "unconscious produces this observation",
                        "response": "preconscious keeps blocking it",
                        "maintaining_defense": "repression",
                        "occurrence_count": count,
                        "severity": "HIGH" if count >= 5 else "MEDIUM",
                    }
                )
        return patterns

    def _detect_avoidance_loops(self) -> list[dict]:
        abandonment = [
            o
            for o in self.outcome_history
            if o.signal == OutcomeSignal.ABANDONMENT
        ]
        if len(abandonment) >= self.correction_threshold:
            return [
                {
                    "pattern_type": "avoidance_loop",
                    "description": "User repeatedly abandons conversations",
                    "trigger": "specific topic or request type",
                    "response": "system provides unsatisfying response",
                    "maintaining_defense": "denial",
                    "occurrence_count": len(abandonment),
                    "severity": "HIGH" if len(abandonment) >= 5 else "MEDIUM",
                }
            ]
        return []

    def _detect_escalation_spirals(self) -> list[dict]:
        recent = self.outcome_history[-20:]
        neg_signals = {
            OutcomeSignal.FRUSTRATION,
            OutcomeSignal.ESCALATION,
            OutcomeSignal.CORRECTION,
        }

        streak = 0
        max_streak = 0
        for o in recent:
            if o.signal in neg_signals:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        if max_streak >= self.escalation_window:
            return [
                {
                    "pattern_type": "escalation_spiral",
                    "description": "Escalation spiral: consecutive negative outcomes",
                    "trigger": "ongoing conversation",
                    "response": "system fails to adapt",
                    "maintaining_defense": "rationalization",
                    "occurrence_count": max_streak,
                    "severity": "CRITICAL" if max_streak >= 7 else "HIGH",
                }
            ]
        return []
