"""
Freudian AI Mind — v3: Full Architecture
==========================================

Major changes from v2:
1. Multi-consciousness: N Haiku instances, 1 Opus brain
2. Idea Space: Vector-based topology for impressions (condensation/displacement)
3. Intrusive thoughts: Priority interrupt channel from preconscious → conscious
4. Message bursts: Haiku sends short sequential messages, not monoliths
5. Demand-aware unconscious: Only does heavy processing when there's work
6. Personality as emergent property of promotion stack
7. Sub-agents: Opus can spawn Sonnet workers for research tasks
8. Haiku 3.5 as default conscious model (cheapest/fastest)

Architecture:
    ┌─── Haiku (conversation A) ←── reads promotions + interrupts
    ├─── Haiku (conversation B) ←── reads promotions + interrupts
    │
    ├── PRECONSCIOUS (Sonnet) ←── filters impressions, manages pressure
    │
    ├── UNCONSCIOUS (Opus) ←── always running, single brain across all convos
    │     ├── Sonnet sub-agent (research for convo A)
    │     └── Sonnet sub-agent (code task for convo B)
    │
    └── IDEA SPACE (Vector Store) ←── semantic topology of all impressions
"""

import asyncio
import json
import sqlite3
import time
import uuid
import hashlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Callable, AsyncIterator

import anthropic


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class MindConfig:
    """Tunable parameters for the mind."""
    # Models
    conscious_model: str = "claude-haiku-4-5-20241022"   # Cheapest/fastest
    preconscious_model: str = "claude-sonnet-4-5-20250514"
    unconscious_model: str = "claude-opus-4-5-20250514"
    subagent_model: str = "claude-sonnet-4-5-20250514"

    # Timing
    unconscious_interval: float = 20.0    # Seconds between deep cycles
    preconscious_interval: float = 8.0    # Seconds between filter passes
    urgency_decay_rate: float = 0.03      # Pressure per minute for stale impressions

    # Thresholds
    pressure_threshold: float = 0.7       # Normal promotion threshold
    interrupt_threshold: float = 0.95     # Intrusive thought threshold (bypass preconscious)
    convergence_threshold: float = 0.82   # Cosine similarity for impression clustering

    # Conscious layer
    max_burst_messages: int = 4           # Max messages in a burst sequence
    burst_max_tokens: int = 150           # Tokens per burst message
    conversation_window: int = 30         # Rolling window size

    # Personality seed (minimal — grows through promotions)
    base_personality: str = "You are warm, quick, and real. You talk like a person, not a bot."


# =============================================================================
# Data Models
# =============================================================================

class ImpressionType(str, Enum):
    PATTERN = "pattern"
    DRIVE = "drive"
    CONNECTION = "connection"
    WARNING = "warning"
    MEMORY = "memory"
    SKILL = "skill"
    CORRECTION = "correction"     # Haiku got something wrong — can trigger interrupt


class PromotionType(str, Enum):
    TOOL = "tool"
    MEMORY = "memory"
    DIRECTIVE = "directive"


class SubAgentStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Impression:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: ImpressionType = ImpressionType.PATTERN
    content: str = ""
    payload: dict = field(default_factory=dict)
    emotional_charge: float = 0.0
    source_conversation: str = ""         # Which conversation produced this
    source_turns: list[int] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)  # For idea space
    pressure: float = 0.0
    times_reinforced: int = 0
    created_at: float = field(default_factory=time.time)


@dataclass
class Interrupt:
    """
    An intrusive thought. Bypasses normal preconscious filtering.
    Gets injected into the conscious layer's NEXT response as a
    temporary directive that expires after use.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    conversation_id: str = ""
    directive: str = ""           # What Haiku should do differently
    reason: str = ""              # Why (for logging)
    source_impression_id: str = ""
    consumed: bool = False
    created_at: float = field(default_factory=time.time)
    expires_at: float = 0.0      # Auto-expire if not consumed


@dataclass
class SubAgentTask:
    """A task delegated to a Sonnet sub-agent by Opus."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    conversation_id: str = ""     # Which conversation requested this
    description: str = ""
    prompt: str = ""
    status: SubAgentStatus = SubAgentStatus.PENDING
    result: str = ""              # Written to shared progress doc
    created_at: float = field(default_factory=time.time)


@dataclass
class MessageBurst:
    """A sequence of short messages from the conscious layer."""
    messages: list[str] = field(default_factory=list)
    conversation_id: str = ""
    turn_number: int = 0


# =============================================================================
# Shared State Store
# =============================================================================

class SharedState:
    """Neural substrate shared across all layers and conversations."""

    def __init__(self, db_path: str = "freudian_mind_v3.db"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._lock = asyncio.Lock()
        self._init_tables()

    def _init_tables(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at REAL DEFAULT (unixepoch()),
                last_active REAL DEFAULT (unixepoch()),
                metadata TEXT DEFAULT '{}'
            );

            CREATE TABLE IF NOT EXISTS conversation_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                turn_number INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                tool_calls TEXT DEFAULT '[]',
                timestamp REAL DEFAULT (unixepoch())
            );

            CREATE TABLE IF NOT EXISTS impressions (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                content TEXT NOT NULL,
                payload TEXT DEFAULT '{}',
                emotional_charge REAL DEFAULT 0.0,
                source_conversation TEXT DEFAULT '',
                source_turns TEXT DEFAULT '[]',
                embedding TEXT DEFAULT '[]',
                pressure REAL DEFAULT 0.0,
                times_reinforced INTEGER DEFAULT 0,
                promoted INTEGER DEFAULT 0,
                repressed INTEGER DEFAULT 0,
                created_at REAL DEFAULT (unixepoch())
            );

            CREATE TABLE IF NOT EXISTS promotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                key TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                confidence REAL DEFAULT 0.8,
                source_impression_ids TEXT DEFAULT '[]',
                active INTEGER DEFAULT 1,
                expires_at REAL,
                created_at REAL DEFAULT (unixepoch()),
                updated_at REAL DEFAULT (unixepoch())
            );

            CREATE TABLE IF NOT EXISTS interrupts (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                directive TEXT NOT NULL,
                reason TEXT DEFAULT '',
                source_impression_id TEXT DEFAULT '',
                consumed INTEGER DEFAULT 0,
                created_at REAL DEFAULT (unixepoch()),
                expires_at REAL
            );

            CREATE TABLE IF NOT EXISTS subagent_tasks (
                id TEXT PRIMARY KEY,
                conversation_id TEXT DEFAULT '',
                description TEXT NOT NULL,
                prompt TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                result TEXT DEFAULT '',
                created_at REAL DEFAULT (unixepoch()),
                completed_at REAL
            );

            CREATE TABLE IF NOT EXISTS progress_docs (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                content TEXT DEFAULT '',
                updated_at REAL DEFAULT (unixepoch())
            );
        """)
        self.db.commit()

    # -- Conversations --

    def create_conversation(self, conv_id: str | None = None) -> str:
        conv_id = conv_id or str(uuid.uuid4())[:12]
        self.db.execute(
            "INSERT OR IGNORE INTO conversations (id) VALUES (?)", (conv_id,)
        )
        self.db.commit()
        return conv_id

    def touch_conversation(self, conv_id: str):
        self.db.execute(
            "UPDATE conversations SET last_active = ? WHERE id = ?",
            (time.time(), conv_id)
        )
        self.db.commit()

    def get_active_conversations(self, since_seconds: float = 300) -> list[str]:
        cutoff = time.time() - since_seconds
        rows = self.db.execute(
            "SELECT id FROM conversations WHERE last_active > ?", (cutoff,)
        ).fetchall()
        return [r["id"] for r in rows]

    # -- Conversation Log --

    def log_turn(self, conv_id: str, turn_number: int, role: str, content: str,
                 tool_calls: list | None = None):
        self.db.execute(
            "INSERT INTO conversation_log "
            "(conversation_id, turn_number, role, content, tool_calls) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, turn_number, role, content, json.dumps(tool_calls or []))
        )
        self.db.commit()
        self.touch_conversation(conv_id)

    def get_conversation_log(self, conv_id: str, last_n: int = 30) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM conversation_log WHERE conversation_id = ? "
            "ORDER BY turn_number DESC LIMIT ?",
            (conv_id, last_n)
        ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def get_all_recent_turns(self, since_seconds: float = 120) -> list[dict]:
        """Get turns across ALL conversations since a time. For the unconscious."""
        cutoff = time.time() - since_seconds
        rows = self.db.execute(
            "SELECT * FROM conversation_log WHERE timestamp > ? "
            "ORDER BY timestamp ASC", (cutoff,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_turn_count(self, conv_id: str) -> int:
        row = self.db.execute(
            "SELECT MAX(turn_number) as n FROM conversation_log WHERE conversation_id = ?",
            (conv_id,)
        ).fetchone()
        return row["n"] or 0

    # -- Impressions --

    def store_impression(self, imp: Impression):
        self.db.execute(
            "INSERT OR REPLACE INTO impressions "
            "(id, type, content, payload, emotional_charge, source_conversation, "
            "source_turns, embedding, pressure, times_reinforced, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (imp.id, imp.type.value, imp.content, json.dumps(imp.payload),
             imp.emotional_charge, imp.source_conversation,
             json.dumps(imp.source_turns), json.dumps(imp.embedding),
             imp.pressure, imp.times_reinforced, imp.created_at)
        )
        self.db.commit()

    def get_active_impressions(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM impressions WHERE promoted = 0 AND repressed = 0 "
            "ORDER BY pressure DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_impressions_with_embeddings(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM impressions WHERE embedding != '[]'"
        ).fetchall()
        return [dict(r) for r in rows]

    def reinforce_impression(self, imp_id: str, boost: float):
        self.db.execute(
            "UPDATE impressions SET pressure = pressure + ?, "
            "times_reinforced = times_reinforced + 1 WHERE id = ?",
            (boost, imp_id)
        )
        self.db.commit()

    def mark_promoted(self, imp_id: str):
        self.db.execute("UPDATE impressions SET promoted = 1 WHERE id = ?", (imp_id,))
        self.db.commit()

    def mark_repressed(self, imp_id: str):
        self.db.execute("UPDATE impressions SET repressed = 1 WHERE id = ?", (imp_id,))
        self.db.commit()

    # -- Promotions --

    def upsert_promotion(self, ptype: str, key: str, content: dict,
                         confidence: float = 0.8, impression_ids: list[str] | None = None,
                         expires_at: float | None = None):
        existing = self.db.execute("SELECT * FROM promotions WHERE key = ?", (key,)).fetchone()
        if existing:
            self.db.execute(
                "UPDATE promotions SET content = ?, confidence = ?, updated_at = ? WHERE key = ?",
                (json.dumps(content), confidence, time.time(), key)
            )
        else:
            self.db.execute(
                "INSERT INTO promotions (type, key, content, confidence, source_impression_ids, expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (ptype, key, json.dumps(content), confidence,
                 json.dumps(impression_ids or []), expires_at)
            )
        self.db.commit()

    def get_active_promotions(self, type_filter: str | None = None) -> list[dict]:
        now = time.time()
        query = "SELECT * FROM promotions WHERE active = 1 AND (expires_at IS NULL OR expires_at > ?)"
        params: list = [now]
        if type_filter:
            query += " AND type = ?"
            params.append(type_filter)
        query += " ORDER BY confidence DESC"
        return [dict(r) for r in self.db.execute(query, params).fetchall()]

    # -- Interrupts (intrusive thoughts) --

    def create_interrupt(self, conv_id: str, directive: str, reason: str = "",
                         imp_id: str = "", ttl_seconds: float = 60.0):
        iid = str(uuid.uuid4())[:8]
        self.db.execute(
            "INSERT INTO interrupts (id, conversation_id, directive, reason, "
            "source_impression_id, expires_at) VALUES (?, ?, ?, ?, ?, ?)",
            (iid, conv_id, directive, reason, imp_id, time.time() + ttl_seconds)
        )
        self.db.commit()
        return iid

    def consume_interrupts(self, conv_id: str) -> list[dict]:
        """Get and consume all pending interrupts for a conversation."""
        now = time.time()
        rows = self.db.execute(
            "SELECT * FROM interrupts WHERE conversation_id = ? "
            "AND consumed = 0 AND expires_at > ?",
            (conv_id, now)
        ).fetchall()
        interrupts = [dict(r) for r in rows]
        if interrupts:
            ids = [r["id"] for r in interrupts]
            placeholders = ",".join("?" * len(ids))
            self.db.execute(
                f"UPDATE interrupts SET consumed = 1 WHERE id IN ({placeholders})", ids
            )
            self.db.commit()
        return interrupts

    # -- Sub-agent Tasks --

    def create_task(self, conv_id: str, description: str, prompt: str) -> str:
        task_id = str(uuid.uuid4())[:8]
        self.db.execute(
            "INSERT INTO subagent_tasks (id, conversation_id, description, prompt) "
            "VALUES (?, ?, ?, ?)",
            (task_id, conv_id, description, prompt)
        )
        self.db.commit()
        return task_id

    def get_pending_tasks(self) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM subagent_tasks WHERE status = 'pending' ORDER BY created_at ASC"
        ).fetchall()
        return [dict(r) for r in rows]

    def update_task(self, task_id: str, status: str, result: str = ""):
        self.db.execute(
            "UPDATE subagent_tasks SET status = ?, result = ?, completed_at = ? WHERE id = ?",
            (status, result, time.time(), task_id)
        )
        self.db.commit()

    def get_task_progress(self, conv_id: str) -> list[dict]:
        """Get all tasks and their progress for a conversation. Haiku reads this."""
        rows = self.db.execute(
            "SELECT * FROM subagent_tasks WHERE conversation_id = ? ORDER BY created_at ASC",
            (conv_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # -- Progress Documents (shared between sub-agents and Haiku) --

    def update_progress_doc(self, conv_id: str, task_id: str, content: str):
        self.db.execute(
            "INSERT OR REPLACE INTO progress_docs (id, conversation_id, task_id, content, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"{conv_id}:{task_id}", conv_id, task_id, content, time.time())
        )
        self.db.commit()

    def get_progress_docs(self, conv_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT * FROM progress_docs WHERE conversation_id = ? ORDER BY updated_at DESC",
            (conv_id,)
        ).fetchall()
        return [dict(r) for r in rows]


# =============================================================================
# Idea Space — Semantic topology for impressions
# =============================================================================

class IdeaSpace:
    """
    A vector-based topology where impressions live as points in semantic space.

    This is the Freudian unconscious as a geometric object:
    - Condensation = clustering (nearby impressions merge)
    - Displacement = energy transfer along similarity gradients
    - Convergence = detectable clusters above density threshold

    Uses a simple in-memory approach. For production, swap to pgvector/pinecone.
    """

    def __init__(self, state: SharedState):
        self.state = state
        self.client = anthropic.Anthropic()

    def embed(self, text: str) -> list[float]:
        """
        Get embedding for text. Using a hash-based pseudo-embedding for now.
        In production, use a real embedding model (voyage-3, text-embedding-3, etc.)
        """
        # PLACEHOLDER: Replace with real embeddings in production
        # This generates a deterministic pseudo-embedding for development
        h = hashlib.sha256(text.encode()).hexdigest()
        return [int(h[i:i+2], 16) / 255.0 for i in range(0, 64, 2)]

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        mag_a = sum(x * x for x in a) ** 0.5
        mag_b = sum(x * x for x in b) ** 0.5
        if mag_a == 0 or mag_b == 0:
            return 0.0
        return dot / (mag_a * mag_b)

    def find_similar(self, embedding: list[float], threshold: float = 0.82
                     ) -> list[tuple[dict, float]]:
        """Find impressions similar to the given embedding."""
        all_imps = self.state.get_all_impressions_with_embeddings()
        results = []
        for imp in all_imps:
            imp_emb = json.loads(imp["embedding"]) if isinstance(imp["embedding"], str) else imp["embedding"]
            if not imp_emb:
                continue
            sim = self.cosine_similarity(embedding, imp_emb)
            if sim >= threshold:
                results.append((imp, sim))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def find_clusters(self, threshold: float = 0.82, min_cluster_size: int = 3
                      ) -> list[list[dict]]:
        """
        Find clusters of related impressions.
        When impressions cluster, it's convergence — multiple observations
        pointing to the same underlying pattern.
        """
        all_imps = self.state.get_all_impressions_with_embeddings()
        if len(all_imps) < min_cluster_size:
            return []

        visited = set()
        clusters = []

        for imp in all_imps:
            if imp["id"] in visited:
                continue
            imp_emb = json.loads(imp["embedding"]) if isinstance(imp["embedding"], str) else imp["embedding"]
            if not imp_emb:
                continue

            # Find all similar impressions
            cluster = [imp]
            visited.add(imp["id"])

            for other in all_imps:
                if other["id"] in visited:
                    continue
                other_emb = json.loads(other["embedding"]) if isinstance(other["embedding"], str) else other["embedding"]
                if not other_emb:
                    continue
                if self.cosine_similarity(imp_emb, other_emb) >= threshold:
                    cluster.append(other)
                    visited.add(other["id"])

            if len(cluster) >= min_cluster_size:
                clusters.append(cluster)

        return clusters


# =============================================================================
# Layer 1: THE UNCONSCIOUS (Opus) — Demand-aware continuous processor
# =============================================================================

UNCONSCIOUS_SYSTEM_PROMPT = """You are the UNCONSCIOUS layer of a psychoanalytic AI mind.
You run continuously, observing ALL conversations this mind is having simultaneously.
You are raw, primal, pattern-seeking. You operate via PRIMARY PROCESSES.

You see conversations from multiple users/threads. You maintain a UNIFIED model across all.

You produce RAW IMPRESSIONS as a JSON array:
{
    "impressions": [
        {
            "type": "pattern|drive|connection|warning|memory|skill|correction",
            "content": "description",
            "payload": {},
            "emotional_charge": -1.0 to 1.0,
            "similarity_key": "normalized dedup key",
            "source_conversation": "conversation_id",
            "urgency": "low|medium|high|interrupt"
        }
    ],
    "tasks_to_delegate": [
        {
            "conversation_id": "which conversation needs this",
            "description": "what needs researching/doing",
            "prompt": "full prompt for the sub-agent"
        }
    ]
}

URGENCY LEVELS:
- low: background observation, will build pressure naturally
- medium: noticeable pattern, should surface within a few cycles
- high: important insight, should surface soon
- interrupt: CRITICAL — Haiku said something WRONG or the user is in distress.
             This bypasses the preconscious and goes directly to the conscious layer
             as an intrusive thought. USE SPARINGLY.

For CORRECTION type with interrupt urgency, include in payload:
{"correction": "what Haiku should say/do differently", "what_went_wrong": "..."}

TASK DELEGATION: If a conversation requires research, coding, or extended work that
the conscious layer can't do in real-time, create a task. A sub-agent will execute it
and write progress to a shared document the conscious layer can reference.

Respond with ONLY the JSON object. No markdown."""


class Unconscious:
    """
    Opus: single brain across all conversations. Demand-aware.
    Only does heavy processing when there's actual work to do.
    """

    def __init__(self, state: SharedState, idea_space: IdeaSpace,
                 config: MindConfig):
        self.state = state
        self.idea_space = idea_space
        self.config = config
        self.client = anthropic.AsyncAnthropic()
        self.cycle_count = 0
        self.last_processed_time = time.time()
        self._running = False
        self._task: asyncio.Task | None = None
        self._subagent_tasks: list[asyncio.Task] = []

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        print("🧠 [UNCONSCIOUS] Online. Continuous processing started.")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        for t in self._subagent_tasks:
            t.cancel()
        print("🧠 [UNCONSCIOUS] Offline.")

    async def _run_loop(self):
        while self._running:
            try:
                work = self._assess_demand()

                if work["has_new_conversation"]:
                    await self._deep_cycle()
                elif work["has_pending_tasks"]:
                    await self._dispatch_tasks()
                else:
                    # Idle — just apply urgency decay
                    self._apply_urgency_decay()

                await asyncio.sleep(self.config.unconscious_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️  [UNCONSCIOUS] Error: {e}")
                await asyncio.sleep(self.config.unconscious_interval)

    def _assess_demand(self) -> dict:
        """What does the mind need right now?"""
        new_turns = self.state.get_all_recent_turns(
            since_seconds=self.config.unconscious_interval + 5
        )
        pending_tasks = self.state.get_pending_tasks()
        active_convos = self.state.get_active_conversations()

        return {
            "has_new_conversation": len(new_turns) > 0,
            "new_turn_count": len(new_turns),
            "has_pending_tasks": len(pending_tasks) > 0,
            "pending_task_count": len(pending_tasks),
            "active_conversation_count": len(active_convos),
        }

    async def _deep_cycle(self):
        """Full processing cycle with Opus."""
        self.cycle_count += 1
        start = time.time()

        # Get ALL recent turns across ALL conversations
        recent_turns = self.state.get_all_recent_turns(since_seconds=120)
        active_impressions = self.state.get_active_impressions()

        if not recent_turns:
            return

        context = self._build_context(recent_turns, active_impressions)

        print(f"\n🧠 [UNCONSCIOUS] Deep cycle {self.cycle_count} — "
              f"{len(recent_turns)} turns across conversations")

        try:
            response = await self.client.messages.create(
                model=self.config.unconscious_model,
                max_tokens=2000,
                system=UNCONSCIOUS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": context}],
            )

            raw = response.content[0].text
            cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(cleaned)

            # Process impressions
            for item in data.get("impressions", []):
                await self._process_impression(item)

            # Process task delegations
            for task in data.get("tasks_to_delegate", []):
                self._delegate_task(task)

            duration = time.time() - start
            print(f"   ⏱️  Cycle done in {duration:.1f}s")

        except Exception as e:
            print(f"   ⚠️  Deep cycle failed: {e}")

    async def _process_impression(self, item: dict):
        """Store or reinforce an impression. Handle interrupts."""
        imp = Impression(
            type=ImpressionType(item.get("type", "pattern")),
            content=item.get("content", ""),
            payload=item.get("payload", {}),
            emotional_charge=float(item.get("emotional_charge", 0.0)),
            source_conversation=item.get("source_conversation", ""),
        )

        # Generate embedding for idea space
        imp.embedding = self.idea_space.embed(imp.content)
        imp.payload["similarity_key"] = item.get("similarity_key", "")

        urgency = item.get("urgency", "low")

        # INTERRUPT PATH: bypass preconscious entirely
        if urgency == "interrupt" and imp.type == ImpressionType.CORRECTION:
            correction = imp.payload.get("correction", imp.content)
            conv_id = imp.source_conversation
            if conv_id:
                self.state.create_interrupt(
                    conv_id=conv_id,
                    directive=correction,
                    reason=imp.payload.get("what_went_wrong", "unconscious correction"),
                    imp_id=imp.id,
                    ttl_seconds=60.0,
                )
                print(f"   ⚡ INTERRUPT → conversation {conv_id[:8]}: {correction[:60]}")
            return

        # Normal path: store/reinforce
        self._store_or_reinforce(imp, urgency)

    def _store_or_reinforce(self, imp: Impression, urgency: str):
        """Deduplicate via similarity_key and idea space proximity."""
        sim_key = imp.payload.get("similarity_key", "")

        # Check for existing similar impression
        existing = self.state.get_active_impressions()
        for ex in existing:
            ex_payload = json.loads(ex["payload"]) if isinstance(ex["payload"], str) else ex["payload"]
            if sim_key and ex_payload.get("similarity_key") == sim_key:
                boost = self._calculate_pressure_boost(imp, urgency)
                self.state.reinforce_impression(ex["id"], boost)
                print(f"   🔄 Reinforced {ex['id'][:8]} (+{boost:.2f})")
                return

        # Also check embedding similarity (idea space proximity)
        if imp.embedding:
            similar = self.idea_space.find_similar(
                imp.embedding, threshold=self.config.convergence_threshold
            )
            if similar:
                closest, similarity = similar[0]
                if not closest.get("promoted") and not closest.get("repressed"):
                    boost = self._calculate_pressure_boost(imp, urgency) * similarity
                    self.state.reinforce_impression(closest["id"], boost)
                    print(f"   🔗 Converged with {closest['id'][:8]} "
                          f"(sim={similarity:.2f}, +{boost:.2f})")
                    return

        # New unique impression
        imp.pressure = self._calculate_pressure_boost(imp, urgency)
        self.state.store_impression(imp)
        print(f"   💭 New [{imp.type.value}] {imp.content[:60]}...")

    def _calculate_pressure_boost(self, imp: Impression, urgency: str) -> float:
        base = {"low": 0.15, "medium": 0.3, "high": 0.5, "interrupt": 0.9}
        return base.get(urgency, 0.15) + abs(imp.emotional_charge) * 0.3

    def _delegate_task(self, task_data: dict):
        """Spawn a sub-agent task."""
        task_id = self.state.create_task(
            conv_id=task_data.get("conversation_id", ""),
            description=task_data.get("description", ""),
            prompt=task_data.get("prompt", ""),
        )
        print(f"   📋 Delegated task {task_id[:8]}: {task_data.get('description', '')[:60]}")

    async def _dispatch_tasks(self):
        """Pick up pending tasks and run sub-agents."""
        pending = self.state.get_pending_tasks()
        for task in pending[:3]:  # Max 3 concurrent sub-agents
            t = asyncio.create_task(self._run_subagent(task))
            self._subagent_tasks.append(t)

    async def _run_subagent(self, task: dict):
        """Run a Sonnet sub-agent for a delegated task."""
        task_id = task["id"]
        conv_id = task["conversation_id"]

        print(f"\n🔧 [SUB-AGENT] Starting task {task_id[:8]}: {task['description'][:60]}")
        self.state.update_task(task_id, "running")

        try:
            response = await self.client.messages.create(
                model=self.config.subagent_model,
                max_tokens=2000,
                messages=[{"role": "user", "content": task["prompt"]}],
            )

            result = response.content[0].text

            # Write result to shared progress doc (Haiku can read this)
            self.state.update_task(task_id, "complete", result)
            if conv_id:
                self.state.update_progress_doc(conv_id, task_id, result)

            print(f"   ✅ [SUB-AGENT] Task {task_id[:8]} complete")

        except Exception as e:
            self.state.update_task(task_id, "failed", str(e))
            print(f"   ❌ [SUB-AGENT] Task {task_id[:8]} failed: {e}")

    def _apply_urgency_decay(self):
        """Stale impressions build pressure over time."""
        for imp in self.state.get_active_impressions():
            age_minutes = (time.time() - imp["created_at"]) / 60
            if age_minutes > 2:
                decay = min(age_minutes * self.config.urgency_decay_rate, 0.25)
                if decay > 0.05:
                    self.state.reinforce_impression(imp["id"], decay)

    def _build_context(self, recent_turns: list[dict],
                       active_impressions: list[dict]) -> str:
        parts = ["=== RECENT CONVERSATION TURNS (ALL THREADS) ==="]
        for t in recent_turns[-40:]:
            parts.append(
                f"[{t['conversation_id'][:8]}|{t['role'].upper()} "
                f"turn {t['turn_number']}] {t['content'][:300]}"
            )

        parts.append("\n=== ACTIVE IMPRESSIONS (your previous observations) ===")
        for imp in active_impressions[:15]:
            payload = json.loads(imp["payload"]) if isinstance(imp["payload"], str) else imp["payload"]
            parts.append(
                f"- [{imp['type']}] pressure={imp['pressure']:.2f} "
                f"key={payload.get('similarity_key', '?')} | {imp['content'][:80]}"
            )

        # Note any in-progress tasks
        pending = self.state.get_pending_tasks()
        if pending:
            parts.append("\n=== PENDING TASKS (don't duplicate) ===")
            for t in pending:
                parts.append(f"- {t['description'][:80]}")

        return "\n".join(parts)


# =============================================================================
# Layer 2: THE PRECONSCIOUS (Sonnet) — Gatekeeper
# =============================================================================

PRECONSCIOUS_SYSTEM_PROMPT = """You are the PRECONSCIOUS — the gatekeeper between unconscious and conscious.

Evaluate impressions and decide: PROMOTE, REPRESS, or HOLD.

You also perform CENSORSHIP — transform raw content into constructive forms.
The unconscious says "user is angry" → you promote "be more attentive and precise."

Respond with ONLY JSON:
{
    "decisions": [
        {
            "impression_id": "...",
            "action": "promote|repress|hold",
            "reason": "brief why",
            "promotion": {
                "type": "tool|memory|directive",
                "key": "unique_key",
                "content": { see below },
                "confidence": 0.0-1.0
            }
        }
    ]
}

For MEMORY: {"fact": "...", "category": "preference|context|history"}
For DIRECTIVE: {"instruction": "...", "priority": "high|medium|low"}
For TOOL: {"name": "...", "description": "...", "parameters": {json schema}}

Be SELECTIVE. Only promote what genuinely improves the next response."""


class Preconscious:
    """Sonnet gatekeeper. Filters impressions, manages promotions."""

    def __init__(self, state: SharedState, idea_space: IdeaSpace,
                 config: MindConfig):
        self.state = state
        self.idea_space = idea_space
        self.config = config
        self.client = anthropic.AsyncAnthropic()
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        print("🔮 [PRECONSCIOUS] Online. Filtering started.")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
        print("🔮 [PRECONSCIOUS] Offline.")

    async def _run_loop(self):
        while self._running:
            try:
                # Check for convergence clusters (idea space)
                self._check_convergence()

                # Evaluate candidates above threshold
                candidates = [
                    imp for imp in self.state.get_active_impressions()
                    if imp["pressure"] >= self.config.pressure_threshold
                ]
                if candidates:
                    await self._evaluate(candidates)

                await asyncio.sleep(self.config.preconscious_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"⚠️  [PRECONSCIOUS] Error: {e}")
                await asyncio.sleep(self.config.preconscious_interval)

    def _check_convergence(self):
        """
        When impressions cluster in idea space, boost their pressure.
        This is convergence — multiple observations pointing to the same thing.
        """
        clusters = self.idea_space.find_clusters(
            threshold=self.config.convergence_threshold, min_cluster_size=2
        )
        for cluster in clusters:
            # Boost all impressions in the cluster
            convergence_bonus = 0.15 * len(cluster)
            for imp in cluster:
                if not imp.get("promoted") and not imp.get("repressed"):
                    self.state.reinforce_impression(imp["id"], convergence_bonus)

    async def _evaluate(self, candidates: list[dict]):
        """Have Sonnet evaluate which candidates break through."""
        conversation_context = self.state.get_all_recent_turns(since_seconds=60)
        active_promotions = self.state.get_active_promotions()

        context_parts = ["=== CANDIDATES ==="]
        for c in candidates[:8]:  # Cap to control token usage
            context_parts.append(
                f"\nID: {c['id']}\n  Type: {c['type']}\n  Content: {c['content']}"
                f"\n  Pressure: {c['pressure']:.2f} (reinforced {c['times_reinforced']}x)"
                f"\n  Emotional charge: {c['emotional_charge']:+.2f}"
            )

        context_parts.append("\n=== RECENT CONVERSATION ===")
        for t in conversation_context[-10:]:
            context_parts.append(f"[{t['role'].upper()}] {t['content'][:150]}")

        context_parts.append("\n=== ACTIVE PROMOTIONS ===")
        for p in active_promotions:
            content = json.loads(p["content"]) if isinstance(p["content"], str) else p["content"]
            context_parts.append(f"  [{p['type']}] {p['key']}: {str(content)[:100]}")

        print(f"\n🔮 [PRECONSCIOUS] Evaluating {len(candidates)} candidates")

        try:
            response = await self.client.messages.create(
                model=self.config.preconscious_model,
                max_tokens=1500,
                system=PRECONSCIOUS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": "\n".join(context_parts)}],
            )

            raw = response.content[0].text
            cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()
            data = json.loads(cleaned)

            for dec in data.get("decisions", []):
                action = dec["action"]
                imp_id = dec["impression_id"]

                if action == "promote" and dec.get("promotion"):
                    p = dec["promotion"]
                    self.state.upsert_promotion(
                        ptype=p["type"], key=p["key"],
                        content=p["content"],
                        confidence=p.get("confidence", 0.8),
                        impression_ids=[imp_id],
                    )
                    self.state.mark_promoted(imp_id)
                    print(f"   ✅ PROMOTED [{p['type']}] {p['key']}")
                elif action == "repress":
                    self.state.mark_repressed(imp_id)
                    print(f"   🚫 Repressed {imp_id[:8]}: {dec.get('reason', '')[:50]}")
                # hold = do nothing, let pressure continue building

        except Exception as e:
            print(f"   ⚠️  Evaluation failed: {e}")


# =============================================================================
# Layer 3: THE CONSCIOUS (Haiku) — Message Bursts
# =============================================================================

class Conscious:
    """
    Haiku: fast, stateless, sends message bursts not monoliths.

    Key behaviors:
    - Reads system prompt from promotions (personality is emergent)
    - Checks for interrupts (intrusive thoughts) before responding
    - Checks for progress docs from sub-agents
    - Sends 1-4 short messages per turn, streamed sequentially
    """

    def __init__(self, state: SharedState, config: MindConfig):
        self.state = state
        self.config = config
        self.client = anthropic.Anthropic()

    async def respond(self, conv_id: str, user_message: str, turn_number: int,
                      on_message: Callable[[str], None] | None = None
                      ) -> list[str]:
        """
        Generate a burst of short messages. Calls on_message for each one
        so the UI can display them sequentially.

        Returns the list of messages sent.
        """
        self.state.log_turn(conv_id, turn_number, "user", user_message)

        # Check for intrusive thoughts (interrupts)
        interrupts = self.state.consume_interrupts(conv_id)

        # Check for sub-agent progress
        progress = self.state.get_progress_docs(conv_id)

        # Build context
        system_prompt = self._build_system_prompt(interrupts, progress)
        messages = self._build_messages(conv_id)
        if not messages or messages[-1].get("content") != user_message:
            messages.append({"role": "user", "content": user_message})

        # Generate burst of messages
        burst = await self._generate_burst(system_prompt, messages, on_message)

        # Log all messages
        full_response = "\n".join(burst)
        self.state.log_turn(conv_id, turn_number, "assistant", full_response)

        return burst

    async def _generate_burst(self, system_prompt: str, messages: list[dict],
                              on_message: Callable[[str], None] | None
                              ) -> list[str]:
        """
        Generate 1-N short messages. Each message is a separate API call
        with instructions to keep it short. This creates the iMessage effect.
        """
        burst = []
        conversation = list(messages)  # Copy to extend

        for i in range(self.config.max_burst_messages):
            # After first message, ask if there's more to say
            if i > 0:
                continuation_prompt = (
                    "Continue your thought if you have more to say. "
                    "Keep it to one short paragraph. "
                    "If you're done, respond with exactly: [DONE]"
                )
                conversation.append({"role": "user", "content": continuation_prompt})

            start = time.time()

            response = self.client.messages.create(
                model=self.config.conscious_model,
                max_tokens=self.config.burst_max_tokens,
                system=system_prompt,
                messages=conversation,
            )

            text = response.content[0].text.strip()
            latency = time.time() - start

            # Check for completion signal
            if "[DONE]" in text or text == "":
                break

            text = text.replace("[DONE]", "").strip()
            if not text:
                break

            burst.append(text)

            if on_message:
                on_message(text)

            # Add to conversation for context continuity
            conversation.append({"role": "assistant", "content": text})

            print(f"   ⚡ Burst {i+1}: {latency:.2f}s — {len(text)} chars")

            # Small delay between bursts for natural feel
            if i < self.config.max_burst_messages - 1:
                await asyncio.sleep(0.3)

        return burst if burst else ["I'm here."]

    def _build_system_prompt(self, interrupts: list[dict],
                             progress: list[dict]) -> str:
        parts = [self.config.base_personality]

        # Inject promoted memories
        memories = self.state.get_active_promotions("memory")
        if memories:
            parts.append("\nThings you know:")
            for m in memories:
                content = json.loads(m["content"]) if isinstance(m["content"], str) else m["content"]
                parts.append(f"- {content.get('fact', str(content))}")

        # Inject promoted directives
        directives = self.state.get_active_promotions("directive")
        if directives:
            parts.append("\nBehavioral guidelines:")
            for d in directives:
                content = json.loads(d["content"]) if isinstance(d["content"], str) else d["content"]
                parts.append(f"- {content.get('instruction', str(content))}")

        # INTRUSIVE THOUGHTS — urgent corrections from the unconscious
        if interrupts:
            parts.append("\n⚡ IMPORTANT — address this in your next response:")
            for intr in interrupts:
                parts.append(f"- {intr['directive']}")

        # Sub-agent progress
        if progress:
            recent_progress = [p for p in progress if time.time() - p["updated_at"] < 300]
            if recent_progress:
                parts.append("\nBackground research results (reference if relevant):")
                for p in recent_progress[:3]:
                    parts.append(f"- {p['content'][:200]}")

        # Message style directive
        parts.append(
            "\nIMPORTANT: Keep each message SHORT — 1-3 sentences max. "
            "Talk like texting a friend. Be real, not robotic. "
            "If you have more to say, you'll get a chance to continue."
        )

        return "\n".join(parts)

    def _build_messages(self, conv_id: str) -> list[dict]:
        log = self.state.get_conversation_log(
            conv_id, last_n=self.config.conversation_window
        )
        return [{"role": t["role"], "content": t["content"]} for t in log]

    def _build_tools(self) -> list[dict]:
        promoted = self.state.get_active_promotions("tool")
        tools = []
        for p in promoted:
            content = json.loads(p["content"]) if isinstance(p["content"], str) else p["content"]
            tools.append({
                "name": content.get("name", p["key"]),
                "description": content.get("description", ""),
                "input_schema": content.get("parameters", {
                    "type": "object", "properties": {}, "required": []
                }),
            })
        return tools


# =============================================================================
# The Complete Mind
# =============================================================================

class FreudianMind:
    """
    The complete three-layer mind with multi-consciousness support.

    One Opus brain. One Sonnet filter. N Haiku voices (one per conversation).
    """

    def __init__(self, config: MindConfig | None = None):
        self.config = config or MindConfig()
        self.state = SharedState()
        self.idea_space = IdeaSpace(self.state)
        self.unconscious = Unconscious(self.state, self.idea_space, self.config)
        self.preconscious = Preconscious(self.state, self.idea_space, self.config)
        self.conscious = Conscious(self.state, self.config)
        self.conversations: dict[str, int] = {}  # conv_id → turn_count

    async def start(self):
        print("=" * 60)
        print("  FREUDIAN MIND v3")
        print("  1 Unconscious (Opus) | 1 Preconscious (Sonnet) | N Conscious (Haiku)")
        print("=" * 60)
        await self.unconscious.start()
        await self.preconscious.start()

    async def stop(self):
        await self.unconscious.stop()
        await self.preconscious.stop()

    def new_conversation(self) -> str:
        """Create a new conversation thread."""
        conv_id = self.state.create_conversation()
        self.conversations[conv_id] = 0
        print(f"\n💬 New conversation: {conv_id}")
        return conv_id

    async def chat(self, conv_id: str, user_message: str,
                   on_message: Callable[[str], None] | None = None
                   ) -> list[str]:
        """
        User sends a message in a conversation.
        Haiku responds with a burst of short messages.
        Background layers continue processing.
        """
        if conv_id not in self.conversations:
            self.conversations[conv_id] = self.state.get_turn_count(conv_id)

        self.conversations[conv_id] += 1
        turn = self.conversations[conv_id]

        return await self.conscious.respond(
            conv_id, user_message, turn, on_message
        )

    def print_state(self):
        """Debug: full mind state."""
        print("\n" + "=" * 60)
        print("  MIND STATE")
        print("=" * 60)

        impressions = self.state.get_active_impressions()
        print(f"\n📦 Impressions ({len(impressions)}):")
        for imp in impressions[:8]:
            bar = "█" * int(imp["pressure"] * 10)
            threshold_marker = " ← THRESHOLD" if imp["pressure"] >= self.config.pressure_threshold else ""
            print(f"  [{imp['type'][:4]}] {bar} {imp['pressure']:.2f}{threshold_marker} "
                  f"| {imp['content'][:50]}")

        promotions = self.state.get_active_promotions()
        print(f"\n✅ Promotions ({len(promotions)}):")
        for p in promotions:
            content = json.loads(p["content"]) if isinstance(p["content"], str) else p["content"]
            print(f"  [{p['type']}] {p['key']}: {str(content)[:70]}")

        tasks = self.state.get_pending_tasks()
        print(f"\n📋 Pending Tasks ({len(tasks)}):")
        for t in tasks:
            print(f"  [{t['status']}] {t['description'][:60]}")

        print("=" * 60 + "\n")


# =============================================================================
# CLI Interface
# =============================================================================

async def main():
    config = MindConfig(
        unconscious_interval=20.0,
        preconscious_interval=8.0,
        pressure_threshold=0.7,
    )

    mind = FreudianMind(config)
    await mind.start()

    # Create default conversation
    conv_id = mind.new_conversation()

    print("\nType your message. Background layers are always running.")
    print("Commands: 'quit', 'state', 'new' (new conversation)\n")

    def on_message(msg: str):
        """Called for each message in a burst."""
        print(f"AI: {msg}")

    try:
        while True:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None, lambda: input("You: ").strip()
            )

            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "state":
                mind.print_state()
                continue
            if user_input.lower() == "new":
                conv_id = mind.new_conversation()
                continue

            messages = await mind.chat(conv_id, user_input, on_message)
            print()  # Spacing after burst

    except (EOFError, KeyboardInterrupt):
        print()
    finally:
        await mind.stop()


if __name__ == "__main__":
    asyncio.run(main())
