"""Neurosis Detection — detects repetition compulsion and pathological patterns."""

from __future__ import annotations

from ..models import OutcomeSignal, ResponseOutcome


class RepetitionDetector:
    """
    Analyzes outcome history to detect repetition compulsion.

    5 pattern types:
    1. Correction loops: same error corrected 3+ times
    2. Repression loops: same impression repressed 3+ times
    3. Avoidance loops: topic consistently produces abandonment
    4. Escalation spirals: consecutive negative signals with increasing severity
    5. Moral erosion: moral tension accumulating beyond threshold
    """

    def __init__(
        self,
        correction_threshold: int = 3,
        repression_threshold: int = 3,
        escalation_window: int = 5,
        moral_erosion_threshold: int = 5,
    ):
        self.correction_threshold = correction_threshold
        self.repression_threshold = repression_threshold
        self.escalation_window = escalation_window
        self.moral_erosion_threshold = moral_erosion_threshold
        self.outcome_history: list[ResponseOutcome] = []
        self._superego = None

    def set_superego(self, superego):
        """Wire the superego for moral erosion detection."""
        self._superego = superego

    def record_outcome(self, outcome: ResponseOutcome):
        self.outcome_history.append(outcome)

    def detect_patterns(
        self,
        defense_log: list[dict] | None = None,
    ) -> list[dict]:
        """Run all 5 detection algorithms. Returns list of pattern dicts."""
        patterns = []
        patterns.extend(self._detect_correction_loops())
        patterns.extend(self._detect_repression_loops(defense_log or []))
        patterns.extend(self._detect_avoidance_loops())
        patterns.extend(self._detect_escalation_spirals())
        patterns.extend(self._detect_moral_erosion())
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

    def _detect_moral_erosion(self) -> list[dict]:
        """Detect when moral tension has been accumulating beyond threshold.

        Connects the superego's moral tension tracking to the neurosis
        detection pipeline.
        """
        if not self._superego:
            return []

        patterns = []
        for value in self._superego.values:
            if value.tension_count >= self.moral_erosion_threshold:
                patterns.append(
                    {
                        "pattern_type": "moral_erosion",
                        "description": (
                            f"Value '{value.id}' has been strained {value.tension_count} times "
                            f"(injury={value.injury_pressure:.1f})"
                        ),
                        "trigger": "repeated value violations",
                        "response": "system compromises on ethical standards",
                        "maintaining_defense": "rationalization",
                        "occurrence_count": value.tension_count,
                        "severity": "CRITICAL" if value.injury_pressure >= 3.0 else "HIGH",
                    }
                )
        return patterns
