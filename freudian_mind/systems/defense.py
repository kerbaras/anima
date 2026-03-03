"""Defense Mechanism System — tracks defense usage and computes health metrics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

from ..models import (
    DefenseEvent,
    DefenseLevel,
    DefenseMechanism,
    DEFENSE_LEVELS,
)


@dataclass
class DefenseProfile:
    """
    The system's current defense profile. Tracks which defenses are
    being used and their outcomes. This IS the system's mental health.
    """

    usage_counts: dict[str, int] = field(
        default_factory=lambda: {d.value: 0 for d in DefenseMechanism}
    )
    defense_outcomes: dict[str, list[bool]] = field(
        default_factory=lambda: {d.value: [] for d in DefenseMechanism}
    )
    maturity_score: float = 3.0
    flexibility_score: float = 0.5
    growth_velocity: float = 0.0
    maturity_history: list[tuple[float, float]] = field(default_factory=list)

    def record_defense_use(self, defense: DefenseMechanism, led_to_positive: bool):
        self.usage_counts[defense.value] = self.usage_counts.get(defense.value, 0) + 1
        outcomes = self.defense_outcomes.get(defense.value, [])
        outcomes.append(led_to_positive)
        self.defense_outcomes[defense.value] = outcomes[-50:]
        self._recalculate_scores()

    def _recalculate_scores(self):
        total_uses = sum(self.usage_counts.values())
        if total_uses == 0:
            return

        # Maturity: weighted average of defense levels
        weighted_sum = 0.0
        for defense, count in self.usage_counts.items():
            try:
                level = DEFENSE_LEVELS[DefenseMechanism(defense)].value
                weighted_sum += level * count
            except (ValueError, KeyError):
                continue
        self.maturity_score = weighted_sum / total_uses

        # Flexibility: normalized entropy of usage distribution
        probs = [c / total_uses for c in self.usage_counts.values() if c > 0]
        max_entropy = math.log(len(DefenseMechanism))
        if max_entropy > 0 and probs:
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            self.flexibility_score = entropy / max_entropy
        else:
            self.flexibility_score = 0.0

        # Growth velocity: trend in maturity
        self.maturity_history.append((time.time(), self.maturity_score))
        if len(self.maturity_history) >= 5:
            recent = self.maturity_history[-10:]
            if len(recent) >= 2:
                self.growth_velocity = recent[-1][1] - recent[0][1]

    def get_defense_success_rate(self, defense: DefenseMechanism) -> float:
        outcomes = self.defense_outcomes.get(defense.value, [])
        if not outcomes:
            return 0.5
        return sum(1 for o in outcomes if o) / len(outcomes)

    def get_health_report(self) -> dict:
        return {
            "maturity_score": round(self.maturity_score, 2),
            "maturity_label": self._maturity_label(),
            "flexibility_score": round(self.flexibility_score, 2),
            "growth_velocity": round(self.growth_velocity, 3),
            "growth_direction": (
                "improving"
                if self.growth_velocity > 0.05
                else "declining" if self.growth_velocity < -0.05 else "stable"
            ),
            "total_events": sum(self.usage_counts.values()),
            "is_neurotic": self.flexibility_score < 0.3 and self.growth_velocity <= 0,
            "most_used_defenses": self._top_defenses(5),
            "most_effective_defenses": self._most_effective(5),
            "warning_signs": self._detect_warning_signs(),
        }

    def _maturity_label(self) -> str:
        if self.maturity_score >= 3.5:
            return "mature"
        elif self.maturity_score >= 2.5:
            return "neurotic"
        elif self.maturity_score >= 1.5:
            return "immature"
        else:
            return "pathological"

    def _top_defenses(self, n: int) -> list[dict]:
        sorted_d = sorted(
            self.usage_counts.items(), key=lambda x: x[1], reverse=True
        )
        return [
            {
                "defense": d,
                "count": c,
                "level": DEFENSE_LEVELS.get(
                    DefenseMechanism(d), DefenseLevel.NEUROTIC
                ).name,
            }
            for d, c in sorted_d[:n]
            if c > 0
        ]

    def _most_effective(self, n: int) -> list[dict]:
        rates = []
        for d in DefenseMechanism:
            rate = self.get_defense_success_rate(d)
            count = self.usage_counts.get(d.value, 0)
            if count >= 3:
                rates.append(
                    {"defense": d.value, "success_rate": round(rate, 2), "count": count}
                )
        return sorted(rates, key=lambda x: x["success_rate"], reverse=True)[:n]

    def _detect_warning_signs(self) -> list[str]:
        warnings = []
        if self.maturity_score < 2.0:
            warnings.append("System is using predominantly primitive defenses")
        if self.flexibility_score < 0.2:
            warnings.append("Defense selection is highly rigid — possible neurosis")
        if self.growth_velocity < -0.1:
            warnings.append("System mental health is declining")

        total = sum(self.usage_counts.values()) or 1
        for d, count in self.usage_counts.items():
            if count / total > 0.4:
                try:
                    level = DEFENSE_LEVELS.get(
                        DefenseMechanism(d), DefenseLevel.NEUROTIC
                    )
                    if level.value <= 2:
                        warnings.append(f"Over-reliance on {d} (primitive defense)")
                except ValueError:
                    pass
        return warnings
