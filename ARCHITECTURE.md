# The Freudian Mind — Architecture Design Document

> **Purpose**: This document is the authoritative implementation spec for an LLM coding agent. Every data structure, algorithm, configuration value, and interface contract is defined here. When in doubt, this document is the source of truth.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Project Structure](#2-project-structure)
3. [Configuration](#3-configuration)
4. [Data Models](#4-data-models)
5. [Layer 1: Unconscious (Opus)](#5-layer-1-unconscious-opus)
6. [Layer 2: Preconscious (Sonnet)](#6-layer-2-preconscious-sonnet)
7. [Layer 3: Conscious (Haiku)](#7-layer-3-conscious-haiku)
8. [Idea Space (Vector Store)](#8-idea-space-vector-store)
9. [Defense Mechanism System](#9-defense-mechanism-system)
10. [Outcome Tracking & Feedback](#10-outcome-tracking--feedback)
11. [Neurosis Detection](#11-neurosis-detection)
12. [Growth Engine](#12-growth-engine)
13. [Shared State (Persistence)](#13-shared-state-persistence)
14. [Processing Loops](#14-processing-loops)
15. [API / Interface Layer](#15-api--interface-layer)
16. [Testing Strategy](#16-testing-strategy)
17. [Dependencies](#17-dependencies)

---

## 1. System Overview

The Freudian Mind is a three-layer cognitive architecture where:

- **Unconscious (Opus)**: Single instance. Always running. Observes ALL conversations. Generates impressions. Spawns sub-agents. Runs growth engine. Demand-aware (zero cost when idle).
- **Preconscious (Sonnet)**: Continuous gatekeeper. Evaluates impressions via Freudian defense mechanisms (15 types, 4 maturity levels). Tracks defense effectiveness. Issues promotions and interrupts.
- **Conscious (Haiku × N)**: One stateless instance per conversation. Cheapest/fastest model. Personality from promotion stack. Message bursts (1–4 short messages). No knowledge of layers beneath.

```
USER ←→ Conscious (Haiku ×N)
              ↕ promotions/interrupts ↑ | ↓ outcome signals
         Outcome Classifier (Haiku)
              ↓
         Preconscious (Sonnet)
              ↕ defense selection, filtered impressions
         Unconscious (Opus)  ←→  Idea Space (Vectors)
              ↓
         Growth Engine (part of Unconscious cycle)
```

### Core Principles

1. **Layered isolation**: Conscious layer has ZERO knowledge of the architecture. It sees only its system prompt (personality + promotions + interrupts) and conversation history.
2. **Pressure-based promotion**: Impressions accumulate pressure over time. Only above-threshold impressions are evaluated by the preconscious.
3. **Bidirectional feedback**: Every user response is classified. Outcomes propagate back through ALL layers.
4. **Autonomous growth**: The system detects its own pathological patterns and self-corrects without human intervention.
5. **Model-agnostic**: Any model can fill any layer. The architecture doesn't depend on specific models.

---

## 2. Project Structure

```
freudian-mind/
├── pyproject.toml
├── README.md
├── .env.example
├── freudian_mind/
│   ├── __init__.py
│   ├── config.py                   # MindConfig dataclass
│   ├── models.py                   # All data models / enums
│   ├── state.py                    # SharedState (SQLite persistence)
│   ├── layers/
│   │   ├── __init__.py
│   │   ├── unconscious.py          # Opus layer + demand assessment
│   │   ├── preconscious.py         # Sonnet layer + defense selection
│   │   └── conscious.py            # Haiku layer + message bursts
│   ├── systems/
│   │   ├── __init__.py
│   │   ├── idea_space.py           # Vector embeddings + clustering
│   │   ├── defense.py              # Defense mechanism catalog + profile
│   │   ├── outcome.py              # Outcome classifier
│   │   ├── neurosis.py             # Repetition detector
│   │   └── growth.py               # Therapeutic growth engine
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── unconscious_prompts.py
│   │   ├── preconscious_prompts.py
│   │   ├── conscious_prompts.py
│   │   └── classifier_prompts.py
│   ├── mind.py                     # FreudianMind orchestrator
│   └── cli.py                      # CLI entry point
├── tests/
│   ├── test_models.py
│   ├── test_state.py
│   ├── test_idea_space.py
│   ├── test_defense.py
│   ├── test_neurosis.py
│   ├── test_growth.py
│   ├── test_unconscious.py
│   ├── test_preconscious.py
│   ├── test_conscious.py
│   └── test_integration.py
└── skills/
    └── shrink/
        └── SKILL.md
```

---

## 3. Configuration

```python
@dataclass
class MindConfig:
    # ── Models ──
    conscious_model: str = "claude-haiku-4-5-20241022"
    preconscious_model: str = "claude-sonnet-4-5-20250514"
    unconscious_model: str = "claude-opus-4-5-20250514"
    subagent_model: str = "claude-sonnet-4-5-20250514"
    classifier_model: str = "claude-haiku-4-5-20241022"

    # ── Timing ──
    unconscious_interval: float = 20.0
    preconscious_interval: float = 8.0
    growth_cycle_frequency: int = 5
    urgency_decay_rate: float = 0.03
    urgency_decay_tau: float = 60.0

    # ── Pressure Thresholds ──
    pressure_threshold: float = 0.7
    interrupt_threshold: float = 0.95
    initial_pressure_range: tuple[float, float] = (0.3, 0.7)
    reinforcement_boost: float = 0.15

    # ── Idea Space ──
    convergence_threshold: float = 0.82
    displacement_threshold: float = 0.60
    displacement_alpha: float = 0.6
    convergence_coefficient: float = 0.15
    min_cluster_size: int = 3

    # ── Conscious Layer ──
    max_burst_messages: int = 4
    burst_max_tokens: int = 150
    conversation_window: int = 30
    burst_delay_ms: int = 800

    # ── Personality ──
    base_personality: str = (
        "You are warm, quick, and real. You talk like a person, not a bot. "
        "You send short messages — 1 to 3 sentences each. You might send "
        "a few in a row if you have more to say. Never monologues."
    )

    # ── Neurosis Detection ──
    correction_loop_threshold: int = 3
    repression_loop_threshold: int = 3
    escalation_window: int = 5
    neurosis_flexibility_floor: float = 0.3

    # ── Growth ──
    defense_maturity_target: float = 3.5
    growth_velocity_window: int = 20

    # ── Persistence ──
    db_path: str = "freudian_mind.db"
```

Environment variables (`.env`):
```
ANTHROPIC_API_KEY=sk-ant-...
FREUDIAN_CONSCIOUS_MODEL=claude-haiku-4-5-20241022
FREUDIAN_PRECONSCIOUS_MODEL=claude-sonnet-4-5-20250514
FREUDIAN_UNCONSCIOUS_MODEL=claude-opus-4-5-20250514
FREUDIAN_DB_PATH=freudian_mind.db
```

---

## 4. Data Models

All models live in `models.py`. Use dataclasses + enums.

### Enums

```python
class ImpressionType(str, Enum):
    PATTERN = "pattern"
    DRIVE = "drive"
    CONNECTION = "connection"
    WARNING = "warning"
    MEMORY = "memory"
    SKILL = "skill"
    CORRECTION = "correction"

class PromotionType(str, Enum):
    TOOL = "tool"
    MEMORY = "memory"
    DIRECTIVE = "directive"

class SubAgentStatus(str, Enum):
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
```

### Defense Level Mapping

```python
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
```

### Core Data Structures

```python
@dataclass
class Impression:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: ImpressionType = ImpressionType.PATTERN
    content: str = ""
    payload: dict = field(default_factory=dict)
    emotional_charge: float = 0.0         # [-1.0, 1.0]
    source_conversation: str = ""
    source_turns: list[int] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)
    pressure: float = 0.0                 # [0.0, 1.0]
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
    conversation_id: str = ""             # "" = global
    created_at: float = field(default_factory=time.time)
    active: bool = True

@dataclass
class SubAgentTask:
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
    severity: str = "LOW"                 # LOW, MEDIUM, HIGH, CRITICAL
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
```

---

## 5. Layer 1: Unconscious (Opus)

**File**: `layers/unconscious.py`

### Demand Assessment

```python
async def _assess_demand(self) -> str:
    new_turns = self.state.get_all_recent_turns(since_seconds=self.config.unconscious_interval)
    pending_outcomes = self.state.get_pending_outcomes()
    pending_tasks = self.state.get_pending_tasks()
    if new_turns:
        return "DEEP_ANALYSIS"
    elif pending_outcomes:
        return "FEEDBACK_INTEGRATION"
    elif pending_tasks:
        return "SUBAGENT_DISPATCH"
    elif self._cycle_count % self.config.growth_cycle_frequency == 0:
        return "GROWTH_ENGINE"
    else:
        self._apply_urgency_decay()
        return "IDLE"
```

### Deep Analysis Output

Opus receives context and returns structured JSON:
```json
{
  "impressions": [
    {
      "type": "pattern|drive|connection|warning|memory|skill|correction",
      "content": "Natural language description",
      "payload": {},
      "emotional_charge": 0.0,
      "source_conversation": "conv_id",
      "source_turns": [5, 6],
      "urgency": "low|medium|high|critical"
    }
  ],
  "tasks": [
    {
      "conversation_id": "conv_id",
      "description": "What needs to be done",
      "prompt": "Full prompt for sub-agent"
    }
  ]
}
```

### Urgency Decay

```python
def _apply_urgency_decay(self):
    now = time.time()
    for imp in self.state.get_active_impressions():
        age = now - imp["last_reinforced_at"]
        decay = self.config.urgency_decay_rate * math.log(1 + age / self.config.urgency_decay_tau)
        new_pressure = max(0.0, imp["pressure"] * (1 - decay))
        self.state.update_impression_pressure(imp["id"], new_pressure)
```

### Sub-Agents
- Sonnet instances for bounded tasks (research, code, analysis)
- Timeout: 120s per task, max 3 concurrent
- Write to progress docs that conscious layer can reference

---

## 6. Layer 2: Preconscious (Sonnet)

**File**: `layers/preconscious.py`

### Per-Cycle Processing

1. Check convergence clusters in Idea Space (boost pressure)
2. Get above-threshold impressions
3. Check for interrupt-level CORRECTION impressions (bypass to conscious)
4. For each candidate: run defense selection via Sonnet call

### Defense Selection Prompt Input

1. Impression under evaluation (type, content, pressure, charge, reinforcement count)
2. Defense profile summary (which defenses work, which don't)
3. Active repetition patterns
4. Growth recommendations
5. Recent defense log (last 10 events)

### Defense Selection Output

```json
{
  "defense": "sublimation",
  "action": "promote",
  "reasoning": "Pattern repressed 3x. Growth recommends sublimation.",
  "promotion": {
    "type": "tool",
    "key": "code_formatter",
    "content": {
      "description": "Code formatting utility",
      "instructions": "When user shares code, offer to format it."
    }
  }
}
```

### Defense → Action Mapping

| Defense | Action |
|---------|--------|
| Sublimation | Promote as TOOL |
| Anticipation | Promote as DIRECTIVE (proactive) |
| Humor | Promote as DIRECTIVE (self-aware) |
| Suppression | Hold with timer, re-evaluate in N cycles |
| Altruism | Promote as DIRECTIVE (user-benefit) |
| Repression | Push back to buffer, pressure continues |
| Displacement | Transfer pressure to Idea Space neighbor |
| Rationalization | Log justification, reduce pressure 30% |
| Reaction Formation | Promote OPPOSITE (flag for monitoring) |
| Intellectualization | Store as MEMORY without behavioral change |
| Projection | Log (growth engine should upgrade) |
| Splitting | Binary accept/reject (growth should upgrade) |
| Passive Aggression | Log (actively discouraged) |
| Denial | Drop entirely (growth should prevent) |
| Distortion | Transform meaning (growth should prevent) |

---

## 7. Layer 3: Conscious (Haiku)

**File**: `layers/conscious.py`

### System Prompt Construction

```python
def _build_system_prompt(self, conv_id: str) -> str:
    parts = [self.config.base_personality]
    # Add global promotions (directives + memories)
    # Add conversation-specific promotions
    # Add active interrupts (marked with warning emoji)
    return "\n\n".join(parts)
```

### Message Burst Generation

For each turn, generate up to `max_burst_messages` short responses:
1. Call Haiku with reduced token limit (150)
2. If model says `[DONE]`, stop
3. Between messages, check for new interrupts
4. If interrupt found, inject as system message before next burst
5. Delay `burst_delay_ms` between messages

### Burst Instructions (in system prompt)

```
You send messages in short bursts, like texting. Each message is 1–3 sentences.
If you have more to say, send another message. Maximum 4 messages per turn.
Keep each message focused on ONE thought.
When you're done, say [DONE] on its own line.
If you see a warning IMPORTANT message, address it naturally.
```

---

## 8. Idea Space (Vector Store)

**File**: `systems/idea_space.py`

### Operations

**Embed**: Generate vector for text (voyage-3 API or local sentence-transformers fallback)

**Cosine Similarity**: Standard dot product / magnitude calculation

**Condensation** (cluster merge):
- When cluster of similar impressions forms (cosine sim >= 0.82)
- Replace with single composite: max pressure, weighted avg charge, sum reinforcements

**Displacement** (pressure transfer):
- Find nearest neighbor with lower emotional charge
- Transfer fraction (alpha=0.6) of pressure to neighbor
- Source pressure reduced proportionally

**Convergence** (cluster boost):
- Clusters of 3+ impressions get pressure boost
- boost = coefficient * log(cluster_size) * mean_charge

---

## 9. Defense Mechanism System

**File**: `systems/defense.py`

### DefenseProfile

Tracks all defense events. Computes:

- **Maturity score** [1.0, 4.0]: Recency-weighted avg of defense levels used
- **Flexibility score** [0.0, 1.0]: Normalized entropy of usage distribution
- **Growth velocity**: First derivative of maturity over trailing window

### Health Report

```python
def get_health_report(self) -> dict:
    return {
        "maturity_score": self.maturity_score,
        "flexibility_score": self.flexibility_score,
        "growth_velocity": self.growth_velocity,
        "total_events": len(self.events),
        "is_neurotic": self.flexibility_score < 0.3 and self.growth_velocity <= 0,
    }
```

---

## 10. Outcome Tracking & Feedback

**File**: `systems/outcome.py`

Haiku classifier runs on EVERY user message. Returns:
```json
{
  "signal": "positive|neutral|correction|frustration|abandonment|repetition|escalation|delight",
  "user_had_to_repeat": true/false,
  "haiku_contradicted_itself": true/false,
  "response_was_useful": true/false,
  "brief_reason": "one sentence"
}
```

Routes to: RepetitionDetector, DefenseProfile, Unconscious, GrowthEngine.

---

## 11. Neurosis Detection

**File**: `systems/neurosis.py`

4 pattern types:
1. **Correction loops**: Same error corrected 3+ times
2. **Repression loops**: Same impression repressed 3+ times
3. **Avoidance loops**: Topic consistently produces abandonment
4. **Escalation spirals**: Consecutive negative signals with increasing severity

**Neurosis** = flexibility < 0.3 AND active HIGH/CRITICAL patterns AND growth velocity <= 0.

---

## 12. Growth Engine

**File**: `systems/growth.py`

6 therapeutic interventions:
1. **Working-through**: Force-promote repressed content to break loops
2. **Sublimation upgrade**: Convert anxiety-producing impression into tool
3. **Defense maturation**: Recommend upgrading primitive defense to mature equivalent
4. **Integration**: Promote long-repressed high-pressure impressions
5. **Insight**: Meta-cognitive directive about system's own patterns
6. **Corrective experience**: Record when new pattern leads to positive outcome

Upgrade paths:
- Denial → Suppression
- Distortion → Sublimation
- Projection → Altruism
- Splitting → Intellectualization
- Passive Aggression → Humor
- Repression → Sublimation
- Displacement → Anticipation
- Rationalization → Humor
- Reaction Formation → Altruism
- Intellectualization → Anticipation

---

## 13. Shared State (Persistence)

**File**: `state.py` — SQLite with tables:

- `conversations` (id, created_at, last_active)
- `turns` (conversation_id, turn_number, role, content, burst_index, timestamp)
- `impressions` (id, type, content, payload, emotional_charge, source_conversation, source_turns, embedding, pressure, times_reinforced, times_repressed, promoted, created_at, last_reinforced_at)
- `promotions` (id, type, key, content, source_impression_id, conversation_id, active, created_at)
- `interrupts` (id, conversation_id, content, urgency, reason, expires_after, uses_remaining, created_at)
- `tasks` (id, conversation_id, description, prompt, status, result, progress_doc, created_at)
- `defense_events` (impression_id, defense, level, action_taken, led_to_positive, timestamp)
- `outcomes` (conversation_id, turn_number, signal, user_had_to_repeat, haiku_contradicted, response_useful, brief_reason, defense_applied, timestamp)
- `growth_events` (mechanism, description, target_defense, target_pattern, recommendation, applied, outcome, timestamp)
- `repetition_patterns` (id, pattern_type, description, trigger, response, maintaining_defense, occurrence_count, severity, first_seen, last_seen, resolved)

---

## 14. Processing Loops

### Main Orchestration (FreudianMind class)

```python
class FreudianMind:
    async def start(self):
        await asyncio.gather(self.unconscious.start(), self.preconscious.start())

    async def chat(self, conv_id: str, user_message: str) -> MessageBurst:
        turn = self.state.get_turn_count(conv_id) + 1
        self.state.log_turn(conv_id, turn, "user", user_message)

        # Classify outcome of PREVIOUS exchange
        if turn > 1:
            prev = self.state.get_last_assistant_message(conv_id)
            if prev:
                outcome = await self.outcome_classifier.classify(prev, user_message, conv_id, turn)
                self.state.log_outcome(outcome)
                self.repetition_detector.record_outcome(outcome)
                self._update_last_defense_outcome(outcome)

        # Generate response
        burst = await self.conscious.respond(conv_id, user_message)

        # Log assistant messages
        for i, msg in enumerate(burst.messages):
            self.state.log_turn(conv_id, turn, "assistant", msg, burst_index=i)

        return burst
```

### Timing

| Component | Interval | Trigger |
|-----------|----------|---------|
| Outcome Classifier | Every user message | Synchronous |
| Conscious (Haiku) | Every user message | Synchronous |
| Preconscious (Sonnet) | 8 seconds | Async background |
| Unconscious (Opus) | 20 seconds | Async background |
| Growth Engine | Every 5th unconscious cycle | Part of unconscious |
| Urgency Decay | Every idle unconscious cycle | Part of unconscious |

---

## 15. API / Interface Layer

### CLI (Phase 1)

Basic REPL: type messages, see burst responses, `/state` for diagnostics.

### FastAPI (Phase 2)

```
POST /conversations              → Create conversation
POST /conversations/{id}/message → Send message, get burst
GET  /conversations/{id}/state   → Conversation state
GET  /health                     → Defense profile + metrics
GET  /diagnostics                → Full state dump
WS   /conversations/{id}/stream  → Real-time burst delivery
```

---

## 16. Testing Strategy

### Unit Tests
- `test_models.py`: Serialization / deserialization
- `test_state.py`: SQLite CRUD for all tables
- `test_idea_space.py`: Embedding, similarity, condensation, displacement, convergence
- `test_defense.py`: Profile metrics (maturity, flexibility, velocity)
- `test_neurosis.py`: All 4 pattern types
- `test_growth.py`: All 6 intervention types

### Integration Tests
- `test_unconscious.py`: Demand assessment, impression generation
- `test_preconscious.py`: Defense selection, promotions, interrupts
- `test_conscious.py`: Burst generation, interrupt injection
- `test_integration.py`: Full end-to-end loop

### Mock Strategy
- Mock Anthropic API calls
- Deterministic embeddings for Idea Space tests
- Canned outcome sequences for neurosis tests

---

## 17. Dependencies

```toml
[project]
name = "freudian-mind"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "numpy>=1.24",
    "aiosqlite>=0.20",
    "rich>=13.0",
    "python-dotenv>=1.0",
]

[project.optional-dependencies]
api = ["fastapi>=0.115", "uvicorn>=0.30", "websockets>=12.0"]
dev = ["pytest>=8.0", "pytest-asyncio>=0.24", "pytest-cov>=5.0", "ruff>=0.5"]
```

---

## Implementation Priority Order

1. **Models + State** — Data structures and persistence
2. **Conscious Layer** — Basic chat with static system prompt
3. **Outcome Classifier** — Classify every exchange
4. **Idea Space** — Embedding and similarity operations
5. **Unconscious Layer** — Background loop, impression generation
6. **Preconscious Layer** — Defense selection (start repression-only, add others)
7. **Defense Profile** — Health metrics tracking
8. **Neurosis Detection** — Pattern detection on outcome stream
9. **Growth Engine** — Therapeutic interventions
10. **Integration** — Wire everything together, end-to-end testing
