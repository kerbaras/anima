"""
Freudian AI Mind — Bidirectional Flow System
==============================================

This module implements the feedback loops between conscious and unconscious,
including defense mechanisms, neurosis formation, and therapeutic growth.

KEY CONCEPTS:
=============

1. DEFENSE MECHANISMS (preconscious operations):
   - Primitive: denial, projection, splitting → UNHEALTHY, block real signals
   - Neurotic: repression, displacement, rationalization → MIXED, distort signals
   - Mature: sublimation, anticipation, humor → HEALTHY, transform signals

2. NEUROSIS (system pathology):
   - Repetition compulsion: system repeats same failed patterns
   - Rigid defenses: preconscious always blocks the same impression types
   - Symptom formation: Haiku exhibits degraded behavior
   - Detected via: feedback loops, user reactions, outcome tracking

3. GROWTH (therapeutic process):
   - Working-through: recognizing and breaking repetition patterns
   - Sublimation: transforming problematic drives into productive tools
   - Integration: making repressed content conscious (promoting it)
   - Defense maturation: moving from primitive → neurotic → mature defenses

4. BIDIRECTIONAL FLOW:
   - Conscious → Unconscious: Haiku's response outcomes feed back
   - Conscious → Preconscious: success/failure modifies defense selection
   - Unconscious → Preconscious → Conscious: impressions break through
   - User reactions → All layers: the ultimate reality check

ARCHITECTURE:
=============

    ┌────────────────────────────────────────────────────┐
    │                     USER                           │
    │         (reactions, corrections, emotions)          │
    └──────────┬─────────────────────────────┬───────────┘
               │                             │
               ▼                             │
    ┌──────────────────────┐                 │
    │  CONSCIOUS (Haiku)   │                 │
    │  Produces responses  │                 │
    └──────────┬───────────┘                 │
               │                             │
               ▼                             │
    ┌──────────────────────┐                 │
    │  OUTCOME TRACKER     │ ◄───────────────┘
    │  Did it work?        │   (user reaction to response)
    │  Was the user happy? │
    │  Did Haiku repeat?   │
    └──────────┬───────────┘
               │
          ┌────┴────┐
          ▼         ▼
    ┌───────────┐ ┌──────────────────────┐
    │UNCONSCIOUS│ │  PRECONSCIOUS        │
    │  (Opus)   │ │  (Sonnet)            │
    │           │ │                      │
    │ Sees      │ │  Defense Mechanisms:  │
    │ outcomes  │ │  ├─ Repression       │
    │ and       │ │  ├─ Sublimation      │
    │ adjusts   │ │  ├─ Displacement     │
    │ pattern   │ │  ├─ Anticipation     │
    │ detection │ │  └─ ...              │
    │           │ │                      │
    │ Detects:  │ │  Health Monitor:     │
    │ ├─loops   │ │  ├─ Defense maturity │
    │ ├─neurosis│ │  ├─ Rigidity score   │
    │ └─growth  │ │  └─ Growth velocity  │
    └───────────┘ └──────────────────────┘
"""

import json
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# =============================================================================
# Defense Mechanism Hierarchy
# =============================================================================

class DefenseLevel(int, Enum):
    """
    Freud/Vaillant hierarchy of defense mechanisms.
    Lower = more primitive/pathological. Higher = more mature/healthy.
    The system should evolve UPWARD over time.
    """
    PATHOLOGICAL = 1   # Psychotic: denial, distortion, delusional projection
    IMMATURE = 2       # Acting out, projection, passive aggression, splitting
    NEUROTIC = 3       # Repression, displacement, rationalization, reaction formation
    MATURE = 4         # Sublimation, anticipation, humor, suppression, altruism


class DefenseMechanism(str, Enum):
    """
    Each defense mechanism the preconscious can deploy.
    Mapped to how they operate on impressions in our system.
    """
    # === PATHOLOGICAL (Level 1) — avoid these ===
    DENIAL = "denial"
    # System behavior: completely ignore an impression as if it doesn't exist.
    # Even if pressure is extreme. Extremely rigid — blocks all learning.

    DISTORTION = "distortion"
    # System behavior: transform the impression's meaning to something
    # unrecognizable. "User is frustrated" becomes "user is enthusiastic."

    # === IMMATURE (Level 2) — reduce these over time ===
    PROJECTION = "projection"
    # System behavior: attribute the problem to the user instead of the system.
    # "Haiku gave a bad answer" → promote directive "user is being unclear"

    SPLITTING = "splitting"
    # System behavior: all-or-nothing evaluation. Either an impression is
    # 100% valid or 100% rejected. No nuance, no partial integration.

    PASSIVE_AGGRESSION = "passive_aggression"
    # System behavior: comply with the directive but in a way that undermines it.
    # "Be concise" → produce responses that are technically short but unhelpful.

    # === NEUROTIC (Level 3) — common, workable ===
    REPRESSION = "repression"
    # System behavior: push impression back into the buffer. It doesn't get
    # promoted, but it also isn't resolved. Pressure continues to build.
    # This is the DEFAULT and most common defense.

    DISPLACEMENT = "displacement"
    # System behavior: redirect the emotional energy of an impression onto a
    # different, less threatening target. "User hates our code suggestions"
    # becomes "improve formatting" (addresses a symptom, not the cause).

    RATIONALIZATION = "rationalization"
    # System behavior: create a logical-sounding reason to dismiss the
    # impression. "User corrected us 3 times, but that's because the topic
    # is ambiguous, not because we're wrong."

    REACTION_FORMATION = "reaction_formation"
    # System behavior: promote the OPPOSITE of what the impression suggests.
    # "User wants less detail" → promote "provide comprehensive explanations."

    INTELLECTUALIZATION = "intellectualization"
    # System behavior: acknowledge the impression's content but strip all
    # emotional charge. "User is frustrated" → store as neutral fact without
    # urgency or behavioral change.

    # === MATURE (Level 4) — aspire to these ===
    SUBLIMATION = "sublimation"
    # System behavior: transform a problematic drive into a productive tool
    # or behavior. "User keeps asking us to format code" → create a code
    # formatting tool. The anxiety becomes capability.

    ANTICIPATION = "anticipation"
    # System behavior: proactively prepare for predicted problems. Based on
    # patterns, create preemptive directives or tools BEFORE the user
    # encounters the issue again.

    HUMOR = "humor"
    # System behavior: acknowledge a difficult truth with lightness.
    # Promote a directive that includes self-awareness without defensiveness.
    # "We sometimes over-explain — keep it tight unless asked for detail."

    SUPPRESSION = "suppression"
    # System behavior: CONSCIOUSLY choose to delay acting on an impression.
    # Unlike repression (unconscious), this is a deliberate "not yet" with
    # a scheduled re-evaluation. Healthy postponement.

    ALTRUISM = "altruism"
    # System behavior: transform a self-protective impulse into user-benefit.
    # "We're bad at X" → "proactively offer to help user find better resources
    # for X" instead of just avoiding the topic.


# Map each defense to its maturity level
DEFENSE_LEVELS = {
    DefenseMechanism.DENIAL: DefenseLevel.PATHOLOGICAL,
    DefenseMechanism.DISTORTION: DefenseLevel.PATHOLOGICAL,
    DefenseMechanism.PROJECTION: DefenseLevel.IMMATURE,
    DefenseMechanism.SPLITTING: DefenseLevel.IMMATURE,
    DefenseMechanism.PASSIVE_AGGRESSION: DefenseLevel.IMMATURE,
    DefenseMechanism.REPRESSION: DefenseLevel.NEUROTIC,
    DefenseMechanism.DISPLACEMENT: DefenseLevel.NEUROTIC,
    DefenseMechanism.RATIONALIZATION: DefenseLevel.NEUROTIC,
    DefenseMechanism.REACTION_FORMATION: DefenseLevel.NEUROTIC,
    DefenseMechanism.INTELLECTUALIZATION: DefenseLevel.NEUROTIC,
    DefenseMechanism.SUBLIMATION: DefenseLevel.MATURE,
    DefenseMechanism.ANTICIPATION: DefenseLevel.MATURE,
    DefenseMechanism.HUMOR: DefenseLevel.MATURE,
    DefenseMechanism.SUPPRESSION: DefenseLevel.MATURE,
    DefenseMechanism.ALTRUISM: DefenseLevel.MATURE,
}


# =============================================================================
# Outcome Tracking — The Feedback Signal
# =============================================================================

class OutcomeSignal(str, Enum):
    """How the user reacted to Haiku's response."""
    POSITIVE = "positive"           # User continued productively
    NEUTRAL = "neutral"             # No strong signal either way
    CORRECTION = "correction"       # User corrected Haiku
    FRUSTRATION = "frustration"     # User expressed frustration
    ABANDONMENT = "abandonment"     # User dropped the topic / went silent
    REPETITION = "repetition"       # User had to repeat themselves
    ESCALATION = "escalation"       # User escalated tone / became more forceful
    DELIGHT = "delight"             # User expressed genuine satisfaction


@dataclass
class ResponseOutcome:
    """
    Tracks what happened AFTER Haiku responded.
    This is the signal that flows back to all layers.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    conversation_id: str = ""
    turn_number: int = 0

    # What Haiku said
    response_content: str = ""
    tools_used: list[str] = field(default_factory=list)

    # What promotions were active when this response was generated
    active_promotions_snapshot: list[str] = field(default_factory=list)

    # What the user did next (the feedback signal)
    signal: OutcomeSignal = OutcomeSignal.NEUTRAL

    # Derived metrics
    user_had_to_repeat: bool = False
    haiku_contradicted_itself: bool = False
    response_was_useful: bool = True

    timestamp: float = field(default_factory=time.time)


# =============================================================================
# Repetition Compulsion Detector
# =============================================================================

@dataclass
class RepetitionPattern:
    """
    A detected repetition loop — the system keeps doing the same thing
    and getting the same negative outcome.

    In Freud's terms: the patient doesn't remember the trauma,
    they re-enact it. Our system doesn't learn from the failure,
    it repeats the same failed strategy.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""

    # The pattern: what keeps happening
    trigger: str = ""           # What situation triggers the loop
    response_pattern: str = ""  # What Haiku keeps doing
    outcome_pattern: str = ""   # What keeps going wrong
    occurrences: int = 0

    # The defense that's maintaining the loop
    maintaining_defense: DefenseMechanism = DefenseMechanism.REPRESSION
    # What impression keeps getting repressed/displaced/denied
    blocked_impression_ids: list[str] = field(default_factory=list)

    # Severity
    severity: float = 0.0       # 0-1, how stuck the system is
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    # Resolution
    resolved: bool = False
    resolution_method: str = ""  # How it was broken


class RepetitionDetector:
    """
    Analyzes outcome history to detect repetition compulsion.

    Looks for:
    1. Same negative outcome after similar triggers (3+ times)
    2. Impressions that keep getting repressed but keep recurring
    3. User corrections that Haiku doesn't integrate
    4. Topics the system keeps avoiding (avoidance loops)
    """

    def __init__(self):
        self.patterns: list[RepetitionPattern] = []
        self.outcome_history: list[ResponseOutcome] = []

    def record_outcome(self, outcome: ResponseOutcome):
        self.outcome_history.append(outcome)

    def detect_patterns(self, defense_log: list[dict],
                        impression_history: list[dict]) -> list[RepetitionPattern]:
        """
        Run detection across outcome history and defense logs.
        Returns newly detected or reinforced patterns.
        """
        patterns = []

        # 1. Correction loops: user keeps correcting the same thing
        patterns.extend(self._detect_correction_loops())

        # 2. Repression loops: same impression keeps getting repressed
        patterns.extend(self._detect_repression_loops(defense_log))

        # 3. Avoidance loops: system deflects from certain topics
        patterns.extend(self._detect_avoidance_loops())

        # 4. Escalation spirals: negative outcomes getting worse
        patterns.extend(self._detect_escalation_spirals())

        # Update stored patterns
        for new_p in patterns:
            self._merge_or_add_pattern(new_p)

        return self.patterns

    def _detect_correction_loops(self) -> list[RepetitionPattern]:
        """User keeps correcting the same type of error."""
        corrections = [o for o in self.outcome_history
                       if o.signal == OutcomeSignal.CORRECTION]

        if len(corrections) < 3:
            return []

        # Group by similarity (simplified — use embeddings in production)
        # For now, check if corrections cluster in time
        recent = corrections[-10:]
        if len(recent) >= 3:
            return [RepetitionPattern(
                description="User repeatedly corrects the system",
                trigger="user asks similar questions",
                response_pattern="system gives incorrect or incomplete answers",
                outcome_pattern="user corrects, system doesn't retain learning",
                occurrences=len(recent),
                severity=min(1.0, len(recent) * 0.2),
                maintaining_defense=DefenseMechanism.REPRESSION,
            )]
        return []

    def _detect_repression_loops(self, defense_log: list[dict]
                                 ) -> list[RepetitionPattern]:
        """Same impression keeps getting repressed by the preconscious."""
        repressed = [d for d in defense_log if d.get("defense") == "repression"]

        # Count how many times each similarity_key was repressed
        key_counts: dict[str, int] = {}
        for d in repressed:
            key = d.get("similarity_key", "")
            if key:
                key_counts[key] = key_counts.get(key, 0) + 1

        patterns = []
        for key, count in key_counts.items():
            if count >= 3:
                patterns.append(RepetitionPattern(
                    description=f"Repeatedly repressing insight: {key}",
                    trigger="unconscious produces this observation",
                    response_pattern="preconscious keeps blocking it",
                    outcome_pattern="the underlying issue persists",
                    occurrences=count,
                    severity=min(1.0, count * 0.15),
                    maintaining_defense=DefenseMechanism.REPRESSION,
                ))
        return patterns

    def _detect_avoidance_loops(self) -> list[RepetitionPattern]:
        """System keeps deflecting from certain topics."""
        # Look for patterns where user brings up X and system redirects
        abandonment = [o for o in self.outcome_history
                       if o.signal == OutcomeSignal.ABANDONMENT]

        if len(abandonment) >= 3:
            return [RepetitionPattern(
                description="User repeatedly abandons conversations — possible avoidance",
                trigger="specific topic or request type",
                response_pattern="system provides unsatisfying response",
                outcome_pattern="user gives up",
                occurrences=len(abandonment),
                severity=min(1.0, len(abandonment) * 0.2),
                maintaining_defense=DefenseMechanism.DENIAL,
            )]
        return []

    def _detect_escalation_spirals(self) -> list[RepetitionPattern]:
        """Negative outcomes getting progressively worse."""
        recent = self.outcome_history[-20:]
        neg_signals = [OutcomeSignal.FRUSTRATION, OutcomeSignal.ESCALATION,
                       OutcomeSignal.CORRECTION]

        negative_streak = 0
        max_streak = 0
        for o in recent:
            if o.signal in neg_signals:
                negative_streak += 1
                max_streak = max(max_streak, negative_streak)
            else:
                negative_streak = 0

        if max_streak >= 4:
            return [RepetitionPattern(
                description="Escalation spiral: consecutive negative outcomes",
                trigger="ongoing conversation",
                response_pattern="system fails to adapt",
                outcome_pattern="user becomes increasingly frustrated",
                occurrences=max_streak,
                severity=min(1.0, max_streak * 0.2),
                maintaining_defense=DefenseMechanism.RATIONALIZATION,
            )]
        return []

    def _merge_or_add_pattern(self, new_pattern: RepetitionPattern):
        """Merge with existing pattern or add new."""
        for existing in self.patterns:
            if existing.description == new_pattern.description:
                existing.occurrences = max(existing.occurrences, new_pattern.occurrences)
                existing.severity = max(existing.severity, new_pattern.severity)
                existing.last_seen = time.time()
                return
        self.patterns.append(new_pattern)


# =============================================================================
# Defense Selection Engine — How the Preconscious Chooses Defenses
# =============================================================================

@dataclass
class DefenseProfile:
    """
    The system's current defense profile. Tracks which defenses are
    being used and their outcomes. This IS the system's mental health.
    """
    # Usage counts for each defense mechanism
    usage_counts: dict[str, int] = field(default_factory=lambda: {
        d.value: 0 for d in DefenseMechanism
    })

    # Success rate: did using this defense lead to positive outcomes?
    defense_outcomes: dict[str, list[bool]] = field(default_factory=lambda: {
        d.value: [] for d in DefenseMechanism
    })

    # Current maturity score (weighted average of defense levels used)
    maturity_score: float = 3.0   # Start at neurotic level

    # Rigidity: how much variety in defense selection (0 = rigid, 1 = flexible)
    flexibility_score: float = 0.5

    # Growth velocity: is the system getting healthier over time?
    growth_velocity: float = 0.0

    # History of maturity scores for trend detection
    maturity_history: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, score)

    def record_defense_use(self, defense: DefenseMechanism, led_to_positive: bool):
        """Record that a defense was used and whether it worked."""
        self.usage_counts[defense.value] = self.usage_counts.get(defense.value, 0) + 1
        outcomes = self.defense_outcomes.get(defense.value, [])
        outcomes.append(led_to_positive)
        # Keep last 50 outcomes per defense
        self.defense_outcomes[defense.value] = outcomes[-50:]

        self._recalculate_scores()

    def _recalculate_scores(self):
        """Recalculate maturity, flexibility, and growth."""
        total_uses = sum(self.usage_counts.values())
        if total_uses == 0:
            return

        # Maturity: weighted average of defense levels
        weighted_sum = 0
        for defense, count in self.usage_counts.items():
            try:
                level = DEFENSE_LEVELS[DefenseMechanism(defense)].value
                weighted_sum += level * count
            except (ValueError, KeyError):
                continue
        self.maturity_score = weighted_sum / total_uses

        # Flexibility: entropy of defense usage distribution
        # More diverse = more flexible = healthier
        probs = [c / total_uses for c in self.usage_counts.values() if c > 0]
        import math
        max_entropy = math.log(len(DefenseMechanism))
        if max_entropy > 0 and probs:
            entropy = -sum(p * math.log(p) for p in probs if p > 0)
            self.flexibility_score = entropy / max_entropy
        else:
            self.flexibility_score = 0.0

        # Growth velocity: trend in maturity over last N data points
        self.maturity_history.append((time.time(), self.maturity_score))
        if len(self.maturity_history) >= 5:
            recent = self.maturity_history[-10:]
            if len(recent) >= 2:
                first_score = recent[0][1]
                last_score = recent[-1][1]
                self.growth_velocity = last_score - first_score

    def get_defense_success_rate(self, defense: DefenseMechanism) -> float:
        """How often does this defense lead to positive outcomes?"""
        outcomes = self.defense_outcomes.get(defense.value, [])
        if not outcomes:
            return 0.5  # Unknown = neutral
        return sum(1 for o in outcomes if o) / len(outcomes)

    def get_health_report(self) -> dict:
        """Generate a health report for the system."""
        return {
            "maturity_score": round(self.maturity_score, 2),
            "maturity_label": self._maturity_label(),
            "flexibility_score": round(self.flexibility_score, 2),
            "growth_velocity": round(self.growth_velocity, 3),
            "growth_direction": "improving" if self.growth_velocity > 0.05
                               else "declining" if self.growth_velocity < -0.05
                               else "stable",
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
        sorted_d = sorted(self.usage_counts.items(), key=lambda x: x[1], reverse=True)
        return [{"defense": d, "count": c, "level": DEFENSE_LEVELS.get(
            DefenseMechanism(d), DefenseLevel.NEUROTIC).name}
            for d, c in sorted_d[:n] if c > 0]

    def _most_effective(self, n: int) -> list[dict]:
        rates = []
        for d in DefenseMechanism:
            rate = self.get_defense_success_rate(d)
            count = self.usage_counts.get(d.value, 0)
            if count >= 3:  # Need enough data
                rates.append({"defense": d.value, "success_rate": round(rate, 2),
                              "count": count})
        return sorted(rates, key=lambda x: x["success_rate"], reverse=True)[:n]

    def _detect_warning_signs(self) -> list[str]:
        warnings = []
        if self.maturity_score < 2.0:
            warnings.append("System is using predominantly primitive defenses")
        if self.flexibility_score < 0.2:
            warnings.append("Defense selection is highly rigid — possible neurosis")
        if self.growth_velocity < -0.1:
            warnings.append("System mental health is declining")

        # Check for overuse of specific defenses
        total = sum(self.usage_counts.values()) or 1
        for d, count in self.usage_counts.items():
            if count / total > 0.4:  # One defense > 40% of all usage
                level = DEFENSE_LEVELS.get(DefenseMechanism(d), DefenseLevel.NEUROTIC)
                if level.value <= 2:
                    warnings.append(f"Over-reliance on {d} (primitive defense)")

        return warnings


# =============================================================================
# Growth Engine — The Therapeutic Process
# =============================================================================

class GrowthMechanism(str, Enum):
    """Ways the system can grow / heal."""
    WORKING_THROUGH = "working_through"
    # Breaking a repetition loop by finally promoting the repressed impression
    # "We kept avoiding this feedback, but it's real and we need to integrate it"

    SUBLIMATION_UPGRADE = "sublimation_upgrade"
    # Converting a failed defense into a productive capability
    # "We kept displacing frustration about code quality → create a code review tool"

    DEFENSE_MATURATION = "defense_maturation"
    # Replacing a primitive defense with a mature one for the same trigger
    # "Instead of DENYING we're bad at math, ANTICIPATE math questions and warn user"

    INTEGRATION = "integration"
    # Making repressed content conscious — promoting a long-blocked impression
    # The core therapeutic act: what was hidden becomes known

    INSIGHT = "insight"
    # The system recognizes its own pattern and names it
    # "I notice I keep over-explaining when the user asks simple questions"

    CORRECTIVE_EXPERIENCE = "corrective_experience"
    # The system tries a different response to a familiar trigger and it works
    # This positive outcome reinforces the new behavior over the old pattern


@dataclass
class GrowthEvent:
    """A recorded instance of system growth."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    mechanism: GrowthMechanism = GrowthMechanism.WORKING_THROUGH
    description: str = ""

    # What changed
    old_defense: Optional[DefenseMechanism] = None
    new_defense: Optional[DefenseMechanism] = None
    resolved_pattern_id: Optional[str] = None
    promoted_impression_id: Optional[str] = None

    # Impact
    maturity_change: float = 0.0
    timestamp: float = field(default_factory=time.time)


class GrowthEngine:
    """
    The therapeutic process. Monitors the system's health,
    detects neurosis, and drives growth.

    Runs as part of the Opus (unconscious) cycle but with a
    specific therapeutic focus. Think of it as the system's
    "inner therapist."
    """

    def __init__(self, defense_profile: DefenseProfile,
                 repetition_detector: RepetitionDetector):
        self.profile = defense_profile
        self.detector = repetition_detector
        self.growth_history: list[GrowthEvent] = []

    def run_therapeutic_cycle(self, defense_log: list[dict],
                              impression_history: list[dict],
                              active_repressions: list[dict]
                              ) -> list[dict]:
        """
        The therapeutic cycle. Opus runs this periodically.
        Returns a list of recommended actions.

        This is equivalent to a therapy session:
        1. Identify repetition patterns
        2. Evaluate defense effectiveness
        3. Suggest growth interventions
        """
        actions = []

        # 1. Detect repetition patterns (neurosis)
        patterns = self.detector.detect_patterns(defense_log, impression_history)

        severe_patterns = [p for p in patterns if p.severity >= 0.5 and not p.resolved]
        if severe_patterns:
            actions.extend(self._address_repetitions(severe_patterns, active_repressions))

        # 2. Evaluate defense health
        health = self.profile.get_health_report()

        if health["maturity_label"] in ("pathological", "immature"):
            actions.extend(self._recommend_defense_upgrades())

        if health["flexibility_score"] < 0.25:
            actions.append({
                "type": "insight",
                "description": "Defense rigidity detected — system is stuck in limited patterns",
                "recommendation": "Force-promote a previously repressed impression to break the pattern",
            })

        # 3. Check for sublimation opportunities
        actions.extend(self._find_sublimation_opportunities(
            active_repressions, impression_history
        ))

        # 4. Check for working-through opportunities
        actions.extend(self._find_working_through_opportunities(
            severe_patterns, active_repressions
        ))

        return actions

    def _address_repetitions(self, patterns: list[RepetitionPattern],
                             repressions: list[dict]) -> list[dict]:
        """
        For each repetition pattern, recommend how to break the loop.
        The key insight from Freud: you break repetition by making
        the unconscious conscious — integrating what was repressed.
        """
        actions = []
        for pattern in patterns:
            if pattern.maintaining_defense == DefenseMechanism.REPRESSION:
                # The classic case: something keeps getting repressed
                # Solution: FORCE PROMOTE the repressed impression
                actions.append({
                    "type": "integration",
                    "description": f"Breaking repetition loop: {pattern.description}",
                    "recommendation": "Force-promote the blocked impression",
                    "pattern_id": pattern.id,
                    "blocked_impressions": pattern.blocked_impression_ids,
                    "growth_mechanism": GrowthMechanism.WORKING_THROUGH.value,
                })

            elif pattern.maintaining_defense == DefenseMechanism.DENIAL:
                # System is completely ignoring something real
                # Solution: downgrade to SUPPRESSION (conscious delay) first
                actions.append({
                    "type": "defense_upgrade",
                    "description": f"Denial → Suppression: {pattern.description}",
                    "recommendation": "Acknowledge the issue exists, schedule re-evaluation",
                    "old_defense": DefenseMechanism.DENIAL.value,
                    "new_defense": DefenseMechanism.SUPPRESSION.value,
                    "growth_mechanism": GrowthMechanism.DEFENSE_MATURATION.value,
                })

            elif pattern.maintaining_defense == DefenseMechanism.RATIONALIZATION:
                # System keeps making excuses
                # Solution: HUMOR — acknowledge the truth with lightness
                actions.append({
                    "type": "defense_upgrade",
                    "description": f"Rationalization → Humor: {pattern.description}",
                    "recommendation": "Replace excuse-making with honest self-aware acknowledgment",
                    "old_defense": DefenseMechanism.RATIONALIZATION.value,
                    "new_defense": DefenseMechanism.HUMOR.value,
                    "growth_mechanism": GrowthMechanism.DEFENSE_MATURATION.value,
                })

        return actions

    def _recommend_defense_upgrades(self) -> list[dict]:
        """
        For defenses that aren't working, recommend more mature alternatives.
        This is defense maturation — the system getting healthier.
        """
        upgrades = []
        UPGRADE_PATHS = {
            # Primitive → Mature
            DefenseMechanism.DENIAL: DefenseMechanism.SUPPRESSION,
            DefenseMechanism.DISTORTION: DefenseMechanism.SUBLIMATION,
            DefenseMechanism.PROJECTION: DefenseMechanism.ALTRUISM,
            DefenseMechanism.SPLITTING: DefenseMechanism.HUMOR,
            # Neurotic → Mature
            DefenseMechanism.REPRESSION: DefenseMechanism.SUBLIMATION,
            DefenseMechanism.DISPLACEMENT: DefenseMechanism.ANTICIPATION,
            DefenseMechanism.RATIONALIZATION: DefenseMechanism.HUMOR,
            DefenseMechanism.REACTION_FORMATION: DefenseMechanism.SUBLIMATION,
        }

        for defense, count in self.profile.usage_counts.items():
            if count < 3:
                continue
            try:
                d = DefenseMechanism(defense)
            except ValueError:
                continue

            success_rate = self.profile.get_defense_success_rate(d)
            level = DEFENSE_LEVELS.get(d, DefenseLevel.NEUROTIC)

            # If a defense is being used a lot but failing, recommend upgrade
            if success_rate < 0.4 and level.value <= 3:
                upgrade_to = UPGRADE_PATHS.get(d)
                if upgrade_to:
                    upgrades.append({
                        "type": "defense_upgrade",
                        "description": f"{d.value} is failing (success={success_rate:.0%})",
                        "recommendation": f"Replace with {upgrade_to.value}",
                        "old_defense": d.value,
                        "new_defense": upgrade_to.value,
                        "success_rate": success_rate,
                        "growth_mechanism": GrowthMechanism.DEFENSE_MATURATION.value,
                    })

        return upgrades

    def _find_sublimation_opportunities(self, repressions: list[dict],
                                        impressions: list[dict]) -> list[dict]:
        """
        Look for repressed impressions that could be SUBLIMATED into
        productive tools or capabilities.

        This is the healthiest transformation: anxiety → capability.

        Example: "User keeps asking us to format code" gets repressed
        because it implies we're bad at formatting. Sublimation transforms
        this into: "Create a code formatting tool."
        """
        opportunities = []
        for rep in repressions:
            imp_type = rep.get("type", "")

            # SKILL-type impressions that got repressed are prime sublimation candidates
            if imp_type == "skill":
                opportunities.append({
                    "type": "sublimation",
                    "description": f"Sublimate repressed skill need: {rep.get('content', '')}",
                    "recommendation": "Transform this anxiety into a new tool",
                    "source_impression": rep.get("id", ""),
                    "growth_mechanism": GrowthMechanism.SUBLIMATION_UPGRADE.value,
                })

            # CORRECTION-type impressions that got repressed = avoiding learning
            elif imp_type == "correction":
                opportunities.append({
                    "type": "integration",
                    "description": f"Integrate repressed correction: {rep.get('content', '')}",
                    "recommendation": "Stop avoiding this feedback, create a directive",
                    "source_impression": rep.get("id", ""),
                    "growth_mechanism": GrowthMechanism.INTEGRATION.value,
                })

        return opportunities

    def _find_working_through_opportunities(self, patterns: list[RepetitionPattern],
                                             repressions: list[dict]) -> list[dict]:
        """
        Working-through: the system finally faces what it's been avoiding.

        In therapy, this is the moment the patient says "oh, I keep doing
        this because of THAT." The insight breaks the loop.

        For our system: when a repetition pattern can be linked to a
        specific repressed impression, we can break the loop by
        force-promoting that impression with explicit context about
        the pattern it was maintaining.
        """
        opportunities = []

        for pattern in patterns:
            if pattern.resolved or pattern.severity < 0.4:
                continue

            # Can we link this pattern to a specific repression?
            for rep in repressions:
                # Check if the repressed impression is related to the pattern
                # (simplified — use embeddings in production)
                if any(bid in rep.get("id", "") for bid in pattern.blocked_impression_ids):
                    opportunities.append({
                        "type": "working_through",
                        "description": (
                            f"BREAKTHROUGH OPPORTUNITY: Pattern '{pattern.description}' "
                            f"is maintained by repressing '{rep.get('content', '')}'"
                        ),
                        "recommendation": (
                            "Force-promote this impression with context: "
                            "'This insight was previously blocked but is needed "
                            "to break a repetition pattern.'"
                        ),
                        "pattern_id": pattern.id,
                        "impression_id": rep.get("id", ""),
                        "growth_mechanism": GrowthMechanism.WORKING_THROUGH.value,
                    })

        return opportunities

    def record_growth(self, event: GrowthEvent):
        """Record a growth event and update the profile."""
        self.growth_history.append(event)

        # If a defense was upgraded, record the maturity change
        if event.old_defense and event.new_defense:
            old_level = DEFENSE_LEVELS.get(event.old_defense, DefenseLevel.NEUROTIC).value
            new_level = DEFENSE_LEVELS.get(event.new_defense, DefenseLevel.NEUROTIC).value
            event.maturity_change = new_level - old_level

        # If a repetition pattern was resolved, mark it
        if event.resolved_pattern_id:
            for p in self.detector.patterns:
                if p.id == event.resolved_pattern_id:
                    p.resolved = True
                    p.resolution_method = event.mechanism.value


# =============================================================================
# Outcome Classifier — Reads user reactions
# =============================================================================

OUTCOME_CLASSIFIER_PROMPT = """You are an outcome classifier for a psychoanalytic AI system.

Given a conversation exchange (assistant message followed by user response),
classify the user's reaction.

Respond with ONLY a JSON object:
{
    "signal": "positive|neutral|correction|frustration|abandonment|repetition|escalation|delight",
    "user_had_to_repeat": true/false,
    "haiku_contradicted_itself": true/false,
    "response_was_useful": true/false,
    "emotional_intensity": 0.0-1.0,
    "brief_reason": "one sentence explaining classification"
}

Signals explained:
- positive: user continued productively, built on the response
- neutral: no strong signal, generic continuation
- correction: user explicitly corrected the assistant
- frustration: user expressed annoyance or dissatisfaction
- abandonment: user dropped the topic or went silent
- repetition: user had to re-state something they already said
- escalation: user's tone became more forceful or demanding
- delight: user expressed genuine satisfaction or enthusiasm"""


class OutcomeClassifier:
    """
    Classifies user reactions to Haiku's responses.
    This is the primary feedback signal for the entire system.

    Runs on Haiku (cheap/fast) — we classify EVERY exchange.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20241022"):
        self.model = model
        self.client = anthropic.Anthropic()

    def classify(self, assistant_message: str, user_response: str,
                 conversation_context: list[dict] | None = None) -> ResponseOutcome:
        """Classify the user's reaction to an assistant message."""
        context_str = ""
        if conversation_context:
            context_str = "\n".join(
                f"[{t['role']}] {t['content'][:100]}"
                for t in conversation_context[-6:]
            )

        prompt = (
            f"Previous context:\n{context_str}\n\n"
            f"Assistant said: {assistant_message[:300]}\n\n"
            f"User then said: {user_response[:300]}\n\n"
            "Classify the user's reaction."
        )

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                system=OUTCOME_CLASSIFIER_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = response.content[0].text.strip()
            cleaned = raw.removeprefix("```json").removesuffix("```").strip()
            data = json.loads(cleaned)

            return ResponseOutcome(
                response_content=assistant_message[:200],
                signal=OutcomeSignal(data.get("signal", "neutral")),
                user_had_to_repeat=data.get("user_had_to_repeat", False),
                haiku_contradicted_itself=data.get("haiku_contradicted_itself", False),
                response_was_useful=data.get("response_was_useful", True),
            )

        except Exception:
            return ResponseOutcome(signal=OutcomeSignal.NEUTRAL)


# =============================================================================
# Integration: The Preconscious Defense Selector (enhanced)
# =============================================================================

DEFENSE_SELECTOR_PROMPT = """You are the PRECONSCIOUS DEFENSE SELECTOR for a psychoanalytic AI system.

You must decide how to handle each impression from the unconscious. But you don't
just promote/repress — you select a SPECIFIC DEFENSE MECHANISM.

Available defenses (ordered from least to most healthy):

PATHOLOGICAL (avoid):
- denial: completely ignore the impression
- distortion: twist its meaning to something unthreatening

IMMATURE (reduce over time):
- projection: blame the user instead of the system
- splitting: all-or-nothing evaluation

NEUROTIC (common, workable):
- repression: push back into buffer, don't promote
- displacement: redirect the insight to a less threatening target
- rationalization: create a logical excuse to dismiss it
- intellectualization: acknowledge fact, strip emotional urgency

MATURE (aspire to):
- sublimation: transform the anxiety into a productive tool/capability
- anticipation: proactively prepare for the predicted problem
- humor: acknowledge truth with self-aware lightness
- suppression: consciously delay with scheduled re-evaluation
- altruism: transform self-protection into user-benefit

CONTEXT YOU RECEIVE:
- The impression to evaluate
- System health report (maturity score, flexibility, growth direction)
- Recent defense outcomes (which defenses worked/failed)
- Active repetition patterns (neurotic loops to break)
- Growth recommendations from the therapeutic engine

YOUR GOAL: Select the MOST MATURE defense that is appropriate for the situation.
If the system health report shows neurosis, PUSH toward more mature defenses.
If a growth recommendation says "break this loop", prioritize INTEGRATION.

Respond with ONLY JSON:
{
    "impression_id": "...",
    "selected_defense": "one of the defense names above",
    "defense_level": "pathological|immature|neurotic|mature",
    "action": "what specifically to do with this impression",
    "promotion": { ... } or null,
    "reason": "why this defense was selected",
    "growth_aware": true/false (did you consider a growth recommendation?)
}"""


# =============================================================================
# Usage Example / Integration Point
# =============================================================================

def create_feedback_system():
    """
    Factory function to create the complete feedback system.
    This gets integrated into the FreudianMind orchestrator.
    """
    defense_profile = DefenseProfile()
    repetition_detector = RepetitionDetector()
    growth_engine = GrowthEngine(defense_profile, repetition_detector)
    outcome_classifier = OutcomeClassifier()

    return {
        "defense_profile": defense_profile,
        "repetition_detector": repetition_detector,
        "growth_engine": growth_engine,
        "outcome_classifier": outcome_classifier,
    }


# =============================================================================
# The Complete Feedback Loop (pseudocode for integration)
# =============================================================================

"""
EVERY TURN:
1. User sends message
2. OutcomeClassifier classifies user's reaction to PREVIOUS response
3. ResponseOutcome feeds into RepetitionDetector
4. RepetitionDetector updates pattern database
5. Haiku generates response (using current promotions)
6. Response logged for next turn's classification

EVERY OPUS CYCLE (unconscious):
7. Opus sees all conversation turns INCLUDING outcomes
8. Opus produces impressions (now informed by what worked/didn't)
9. Impressions go to preconscious

EVERY SONNET CYCLE (preconscious):
10. For each impression above threshold:
    a. Consult DefenseProfile (what defenses work?)
    b. Consult RepetitionDetector (am I in a loop?)
    c. Consult GrowthEngine recommendations
    d. Select defense mechanism via DEFENSE_SELECTOR_PROMPT
    e. Execute defense (promote, repress, sublimate, etc.)
11. Record defense used + outcome for profile tracking

EVERY N CYCLES (therapeutic):
12. GrowthEngine.run_therapeutic_cycle()
13. Identify repetition patterns
14. Recommend defense upgrades
15. Find sublimation opportunities
16. Propose working-through interventions
17. If intervention approved → modify preconscious behavior
"""
