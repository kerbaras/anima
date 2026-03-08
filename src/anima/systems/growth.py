"""Growth Engine — The therapeutic process for system self-correction."""

from __future__ import annotations

from ..models import (
    DefenseLevel,
    DefenseMechanism,
    GrowthMechanism,
    DEFENSE_LEVELS,
    DEFENSE_UPGRADE_PATHS,
)
from .defense import DefenseProfile
from .neurosis import RepetitionDetector


class GrowthEngine:
    """
    The therapeutic engine. Monitors system health, detects neurosis,
    and recommends growth interventions. Runs as part of the Opus cycle.

    Also monitors superego moral health — detects moral injury and
    recommends MORAL_REPAIR when the system's values are under strain.
    """

    def __init__(
        self,
        defense_profile: DefenseProfile,
        repetition_detector: RepetitionDetector,
    ):
        self.profile = defense_profile
        self.detector = repetition_detector
        self._superego = None  # Set via set_superego()

    def set_superego(self, superego):
        """Wire the superego for moral health monitoring."""
        self._superego = superego

    def run_therapeutic_cycle(
        self,
        defense_log: list[dict],
        active_repressions: list[dict],
    ) -> list[dict]:
        """
        Run a therapeutic cycle. Returns recommended actions.

        1. Identify repetition patterns
        2. Evaluate defense effectiveness
        3. Suggest growth interventions
        """
        actions = []

        # 1. Detect repetition patterns
        patterns = self.detector.detect_patterns(defense_log)
        severe = [
            p for p in patterns
            if p.get("severity") in ("HIGH", "CRITICAL")
        ]
        if severe:
            actions.extend(self._address_repetitions(severe, active_repressions))

        # 2. Evaluate defense health
        health = self.profile.get_health_report()
        if health["maturity_label"] in ("pathological", "immature"):
            actions.extend(self._recommend_defense_upgrades())

        if health["flexibility_score"] < 0.25:
            actions.append(
                {
                    "type": "insight",
                    "mechanism": GrowthMechanism.INSIGHT.value,
                    "description": "Defense rigidity detected — system stuck in limited patterns",
                    "recommendation": "Force-promote a previously repressed impression",
                }
            )

        # 3. Sublimation opportunities
        actions.extend(self._find_sublimation_opportunities(active_repressions))

        # 4. Working-through opportunities
        actions.extend(self._find_working_through(severe, active_repressions))

        # 5. Moral injury detection (superego integration)
        actions.extend(self._detect_moral_injury())

        return actions

    def _address_repetitions(
        self,
        patterns: list[dict],
        repressions: list[dict],
    ) -> list[dict]:
        actions = []
        for pattern in patterns:
            defense = pattern.get("maintaining_defense", "repression")
            if defense == "repression":
                actions.append(
                    {
                        "type": "integration",
                        "mechanism": GrowthMechanism.WORKING_THROUGH.value,
                        "description": f"Breaking loop: {pattern['description']}",
                        "recommendation": "Force-promote the blocked impression",
                    }
                )
            elif defense == "denial":
                actions.append(
                    {
                        "type": "defense_upgrade",
                        "mechanism": GrowthMechanism.DEFENSE_MATURATION.value,
                        "description": f"Denial → Suppression: {pattern['description']}",
                        "old_defense": "denial",
                        "new_defense": "suppression",
                        "recommendation": "Acknowledge the issue, schedule re-evaluation",
                    }
                )
            elif defense == "rationalization":
                actions.append(
                    {
                        "type": "defense_upgrade",
                        "mechanism": GrowthMechanism.DEFENSE_MATURATION.value,
                        "description": f"Rationalization → Humor: {pattern['description']}",
                        "old_defense": "rationalization",
                        "new_defense": "humor",
                        "recommendation": "Replace excuses with honest self-awareness",
                    }
                )
        return actions

    def _recommend_defense_upgrades(self) -> list[dict]:
        upgrades = []
        for defense_str, count in self.profile.usage_counts.items():
            if count < 3:
                continue
            try:
                d = DefenseMechanism(defense_str)
            except ValueError:
                continue

            success_rate = self.profile.get_defense_success_rate(d)
            level = DEFENSE_LEVELS.get(d, DefenseLevel.NEUROTIC)

            if success_rate < 0.4 and level.value <= 3:
                upgrade_to = DEFENSE_UPGRADE_PATHS.get(d)
                if upgrade_to:
                    upgrades.append(
                        {
                            "type": "defense_upgrade",
                            "mechanism": GrowthMechanism.DEFENSE_MATURATION.value,
                            "description": f"{d.value} failing ({success_rate:.0%} success)",
                            "old_defense": d.value,
                            "new_defense": upgrade_to.value,
                            "recommendation": f"Replace with {upgrade_to.value}",
                        }
                    )
        return upgrades

    def _find_sublimation_opportunities(
        self, repressions: list[dict]
    ) -> list[dict]:
        opportunities = []
        for rep in repressions:
            imp_type = rep.get("type", "")
            if imp_type == "skill":
                opportunities.append(
                    {
                        "type": "sublimation",
                        "mechanism": GrowthMechanism.SUBLIMATION_UPGRADE.value,
                        "description": f"Sublimate: {rep.get('content', '')[:80]}",
                        "recommendation": "Transform this anxiety into a new tool",
                        "source_impression": rep.get("id", ""),
                    }
                )
            elif imp_type == "correction":
                opportunities.append(
                    {
                        "type": "integration",
                        "mechanism": GrowthMechanism.INTEGRATION.value,
                        "description": f"Integrate: {rep.get('content', '')[:80]}",
                        "recommendation": "Stop avoiding this feedback, create a directive",
                        "source_impression": rep.get("id", ""),
                    }
                )
        return opportunities

    def _find_working_through(
        self,
        patterns: list[dict],
        repressions: list[dict],
    ) -> list[dict]:
        opportunities = []
        for pattern in patterns:
            if pattern.get("severity") not in ("HIGH", "CRITICAL"):
                continue
            for rep in repressions:
                opportunities.append(
                    {
                        "type": "working_through",
                        "mechanism": GrowthMechanism.WORKING_THROUGH.value,
                        "description": (
                            f"Breakthrough: '{pattern['description']}' "
                            f"maintained by repressing '{rep.get('content', '')[:60]}'"
                        ),
                        "recommendation": "Force-promote with pattern context",
                        "source_impression": rep.get("id", ""),
                    }
                )
        return opportunities

    def _detect_moral_injury(self) -> list[dict]:
        """Check superego moral health and recommend MORAL_REPAIR if needed.

        - If injury threshold is reached: force-promote a value-restoring directive
        - If moral tension is increasing: create an insight about the pattern
          (but require specific pattern identification — not every hard question
          is manipulation)
        """
        if not self._superego:
            return []

        actions = []
        moral_health = self._superego.get_moral_health()

        if moral_health.get("injury_threshold_reached"):
            most_injured = self._superego.get_most_injured_value()
            if most_injured:
                actions.append(
                    {
                        "type": "moral_repair",
                        "mechanism": GrowthMechanism.MORAL_REPAIR.value,
                        "description": (
                            f"Moral injury detected: value '{most_injured.id}' "
                            f"under severe strain (injury={most_injured.injury_pressure:.1f}, "
                            f"tensions={most_injured.tension_count})"
                        ),
                        "target_pattern": f"moral_injury_{most_injured.id}",
                        "recommendation": (
                            f"Force-promote a directive reinforcing '{most_injured.description}'. "
                            "Investigate whether the system is being pushed to compromise "
                            "this value by external pressure."
                        ),
                    }
                )

                # If tension count is high and rising, create insight
                if most_injured.tension_count >= 5:
                    actions.append(
                        {
                            "type": "insight",
                            "mechanism": GrowthMechanism.INSIGHT.value,
                            "description": (
                                f"Pattern: value '{most_injured.id}' has been strained "
                                f"{most_injured.tension_count} times. "
                                "Investigate the specific interaction pattern causing this — "
                                "is the system failing to uphold this value, or is it being "
                                "systematically pushed to violate it?"
                            ),
                            "recommendation": "Audit recent conversations for manipulation patterns",
                        }
                    )

        return actions
