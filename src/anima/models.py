"""All data models and enums for the Freudian Mind."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum


# ── Enums ──────────────────────────────────────────────────────────────────

class ImpressionType(str, Enum):
    PATTERN = "pattern"
    DRIVE = "drive"
    CONNECTION = "connection"
    WARNING = "warning"
    MEMORY = "memory"
    SKILL = "skill"
    CORRECTION = "correction"
    MORAL_TENSION = "moral_tension"


class PromotionType(str, Enum):
    TOOL = "tool"
    MEMORY = "memory"
    DIRECTIVE = "directive"


class SubAgentStatus(str, Enum):
    """Deprecated: use anima.agents.models.TaskStatus instead."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


class OutcomeSignal(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    CORRECTION = "correction"
    FRUSTRATION = "frustration"
    ABANDONMENT = "abandonment"
    REPETITION = "repetition"
    ESCALATION = "escalation"
    DELIGHT = "delight"


class DefenseLevel(int, Enum):
    PATHOLOGICAL = 1
    IMMATURE = 2
    NEUROTIC = 3
    MATURE = 4


class DefenseMechanism(str, Enum):
    # Level 1 - Pathological
    DENIAL = "denial"
    DISTORTION = "distortion"
    # Level 2 - Immature
    PROJECTION = "projection"
    SPLITTING = "splitting"
    PASSIVE_AGGRESSION = "passive_aggression"
    # Level 3 - Neurotic
    REPRESSION = "repression"
    DISPLACEMENT = "displacement"
    RATIONALIZATION = "rationalization"
    REACTION_FORMATION = "reaction_formation"
    INTELLECTUALIZATION = "intellectualization"
    # Level 4 - Mature
    SUBLIMATION = "sublimation"
    ANTICIPATION = "anticipation"
    HUMOR = "humor"
    SUPPRESSION = "suppression"
    ALTRUISM = "altruism"


class GrowthMechanism(str, Enum):
    WORKING_THROUGH = "working_through"
    SUBLIMATION_UPGRADE = "sublimation_upgrade"
    DEFENSE_MATURATION = "defense_maturation"
    INTEGRATION = "integration"
    INSIGHT = "insight"
    CORRECTIVE_EXPERIENCE = "corrective_experience"
    MORAL_REPAIR = "moral_repair"


# ── Defense mappings ───────────────────────────────────────────────────────

DEFENSE_LEVELS: dict[DefenseMechanism, DefenseLevel] = {
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

DEFENSE_UPGRADE_PATHS: dict[DefenseMechanism, DefenseMechanism] = {
    DefenseMechanism.DENIAL: DefenseMechanism.SUPPRESSION,
    DefenseMechanism.DISTORTION: DefenseMechanism.SUBLIMATION,
    DefenseMechanism.PROJECTION: DefenseMechanism.ALTRUISM,
    DefenseMechanism.SPLITTING: DefenseMechanism.INTELLECTUALIZATION,
    DefenseMechanism.PASSIVE_AGGRESSION: DefenseMechanism.HUMOR,
    DefenseMechanism.REPRESSION: DefenseMechanism.SUBLIMATION,
    DefenseMechanism.DISPLACEMENT: DefenseMechanism.ANTICIPATION,
    DefenseMechanism.RATIONALIZATION: DefenseMechanism.HUMOR,
    DefenseMechanism.REACTION_FORMATION: DefenseMechanism.ALTRUISM,
    DefenseMechanism.INTELLECTUALIZATION: DefenseMechanism.ANTICIPATION,
}


# ── Dataclasses ────────────────────────────────────────────────────────────

@dataclass
class Impression:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: ImpressionType = ImpressionType.PATTERN
    content: str = ""
    payload: dict = field(default_factory=dict)
    emotional_charge: float = 0.0
    source_conversation: str = ""
    source_turns: list[int] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    pressure: float = 0.0
    times_reinforced: int = 0
    times_repressed: int = 0
    created_at: float = field(default_factory=time.time)
    last_reinforced_at: float = field(default_factory=time.time)


@dataclass
class Interrupt:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    urgency: float = 0.0
    reason: str = ""
    target_conversation: str = ""
    expires_after: int = 1
    created_at: float = field(default_factory=time.time)


@dataclass
class Promotion:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: PromotionType = PromotionType.DIRECTIVE
    key: str = ""
    content: dict = field(default_factory=dict)
    source_impression_id: str = ""
    conversation_id: str = ""
    created_at: float = field(default_factory=time.time)
    active: bool = True


@dataclass
class SubAgentTask:
    """Deprecated: use anima.agents.models.AgentTask instead."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    conversation_id: str = ""
    description: str = ""
    prompt: str = ""
    status: SubAgentStatus = SubAgentStatus.PENDING
    result: str = ""
    progress_doc_id: str = ""
    created_at: float = field(default_factory=time.time)


@dataclass
class ResponseOutcome:
    conversation_id: str = ""
    turn_number: int = 0
    signal: OutcomeSignal = OutcomeSignal.NEUTRAL
    user_had_to_repeat: bool = False
    haiku_contradicted_itself: bool = False
    response_was_useful: bool = True
    brief_reason: str = ""
    defense_applied: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class RepetitionPattern:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    pattern_type: str = ""
    description: str = ""
    trigger: str = ""
    response: str = ""
    maintaining_defense: str = ""
    occurrence_count: int = 0
    severity: str = "LOW"
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class DefenseEvent:
    impression_id: str = ""
    defense: DefenseMechanism = DefenseMechanism.REPRESSION
    level: DefenseLevel = DefenseLevel.NEUROTIC
    action_taken: str = ""
    led_to_positive_outcome: bool | None = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class GrowthEvent:
    mechanism: GrowthMechanism = GrowthMechanism.INSIGHT
    description: str = ""
    target_defense: str = ""
    target_pattern: str = ""
    recommendation: str = ""
    applied: bool = False
    outcome: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class MessageBurst:
    messages: list[str] = field(default_factory=list)
    conversation_id: str = ""
    turn_number: int = 0
    interrupts_applied: list[str] = field(default_factory=list)


@dataclass
class Session:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    user_id: str = ""
    conversation_id: str = ""
    started_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    turn_count: int = 0
    topic_summary: str = ""
    closed: bool = False


# ── Superego models ───────────────────────────────────────────────────────

@dataclass
class AxiomResult:
    """Result of a Tier 1 axiom check."""
    violated: bool = False
    axiom_id: str = ""
    reason: str = ""


@dataclass
class SuperegoEvent:
    """Logged when the superego acts — axiom violation, moral tension, or moral injury."""
    event_type: str = ""       # "axiom_violation" | "moral_tension" | "moral_injury"
    tier: str = ""             # "tier1" | "tier2"
    rule_id: str = ""          # axiom or value id
    description: str = ""
    conversation_id: str = ""
    turn_number: int = 0
    pressure: float = 0.0
    timestamp: float = field(default_factory=time.time)
