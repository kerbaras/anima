"""SharedState — async SQLite persistence layer for all layers."""

from __future__ import annotations

import json
import time
import uuid

import aiosqlite

from .agents.models import AgentTask, TaskPhase, TaskStatus, ToolSpec
from .models import Impression, ResponseOutcome


class SharedState:
    """Neural substrate shared across all layers and conversations."""

    def __init__(self, db_path: str = "anima.db"):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None

    async def initialize(self):
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._init_tables()

    async def close(self):
        if self._db:
            await self._db.close()

    async def _init_tables(self):
        await self._db.executescript("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at REAL,
                last_active REAL
            );

            CREATE TABLE IF NOT EXISTS turns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                turn_number INTEGER,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                burst_index INTEGER DEFAULT 0,
                timestamp REAL
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
                times_repressed INTEGER DEFAULT 0,
                promoted INTEGER DEFAULT 0,
                repressed INTEGER DEFAULT 0,
                created_at REAL,
                last_reinforced_at REAL
            );

            CREATE TABLE IF NOT EXISTS promotions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                key TEXT NOT NULL UNIQUE,
                content TEXT NOT NULL,
                confidence REAL DEFAULT 0.8,
                source_impression_ids TEXT DEFAULT '[]',
                conversation_id TEXT DEFAULT '',
                active INTEGER DEFAULT 1,
                created_at REAL,
                updated_at REAL
            );

            CREATE TABLE IF NOT EXISTS interrupts (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                content TEXT NOT NULL,
                urgency REAL DEFAULT 0.0,
                reason TEXT DEFAULT '',
                expires_after INTEGER DEFAULT 1,
                uses_remaining INTEGER DEFAULT 1,
                created_at REAL
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                conversation_id TEXT DEFAULT '',
                description TEXT NOT NULL,
                prompt TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                result TEXT DEFAULT '',
                progress_doc TEXT DEFAULT '',
                created_at REAL
            );

            CREATE TABLE IF NOT EXISTS defense_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                impression_id TEXT,
                defense TEXT NOT NULL,
                level INTEGER,
                action_taken TEXT DEFAULT '',
                led_to_positive INTEGER,
                timestamp REAL
            );

            CREATE TABLE IF NOT EXISTS outcomes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT,
                turn_number INTEGER,
                signal TEXT NOT NULL,
                user_had_to_repeat INTEGER DEFAULT 0,
                haiku_contradicted INTEGER DEFAULT 0,
                response_useful INTEGER DEFAULT 1,
                brief_reason TEXT DEFAULT '',
                defense_applied TEXT DEFAULT '',
                timestamp REAL
            );

            CREATE TABLE IF NOT EXISTS growth_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                mechanism TEXT NOT NULL,
                description TEXT DEFAULT '',
                target_defense TEXT DEFAULT '',
                target_pattern TEXT DEFAULT '',
                recommendation TEXT DEFAULT '',
                applied INTEGER DEFAULT 0,
                outcome TEXT DEFAULT '',
                timestamp REAL
            );

            CREATE TABLE IF NOT EXISTS repetition_patterns (
                id TEXT PRIMARY KEY,
                pattern_type TEXT DEFAULT '',
                description TEXT DEFAULT '',
                trigger TEXT DEFAULT '',
                response TEXT DEFAULT '',
                maintaining_defense TEXT DEFAULT '',
                occurrence_count INTEGER DEFAULT 0,
                severity TEXT DEFAULT 'LOW',
                first_seen REAL,
                last_seen REAL,
                resolved INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                started_at REAL,
                last_activity REAL,
                turn_count INTEGER DEFAULT 0,
                topic_summary TEXT DEFAULT '',
                closed INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS user_conversations (
                user_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                created_at REAL
            );

            CREATE TABLE IF NOT EXISTS agent_tasks (
                id TEXT PRIMARY KEY,
                parent_task_id TEXT,
                conversation_id TEXT DEFAULT '',
                model TEXT DEFAULT '',
                description TEXT NOT NULL,
                tools TEXT DEFAULT '[]',
                phase TEXT DEFAULT 'plan',
                status TEXT DEFAULT 'queued',
                max_subtasks INTEGER DEFAULT 5,
                depth INTEGER DEFAULT 0,
                max_depth INTEGER DEFAULT 3,
                result TEXT DEFAULT '',
                children TEXT DEFAULT '[]',
                phase_results TEXT DEFAULT '{}',
                retry_count INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 2,
                created_at REAL,
                updated_at REAL
            );

            CREATE INDEX IF NOT EXISTS idx_agent_tasks_status
                ON agent_tasks(status);
            CREATE INDEX IF NOT EXISTS idx_agent_tasks_parent
                ON agent_tasks(parent_task_id);

            CREATE TABLE IF NOT EXISTS superego_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                tier TEXT NOT NULL,
                rule_id TEXT NOT NULL,
                description TEXT DEFAULT '',
                conversation_id TEXT DEFAULT '',
                turn_number INTEGER DEFAULT 0,
                pressure REAL DEFAULT 0.0,
                timestamp REAL
            );

            CREATE INDEX IF NOT EXISTS idx_superego_events_type
                ON superego_events(event_type);
            CREATE INDEX IF NOT EXISTS idx_superego_events_rule
                ON superego_events(rule_id);
        """)
        await self._db.commit()

    # ── Conversations ──────────────────────────────────────────────────────

    async def create_conversation(self, conv_id: str | None = None) -> str:
        conv_id = conv_id or str(uuid.uuid4())[:12]
        now = time.time()
        await self._db.execute(
            "INSERT OR IGNORE INTO conversations (id, created_at, last_active) "
            "VALUES (?, ?, ?)",
            (conv_id, now, now),
        )
        await self._db.commit()
        return conv_id

    async def touch_conversation(self, conv_id: str):
        await self._db.execute(
            "UPDATE conversations SET last_active = ? WHERE id = ?",
            (time.time(), conv_id),
        )
        await self._db.commit()

    # ── Turns ──────────────────────────────────────────────────────────────

    async def log_turn(
        self,
        conv_id: str,
        turn_number: int,
        role: str,
        content: str,
        burst_index: int = 0,
    ):
        await self._db.execute(
            "INSERT INTO turns (conversation_id, turn_number, role, content, "
            "burst_index, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (conv_id, turn_number, role, content, burst_index, time.time()),
        )
        await self._db.commit()
        await self.touch_conversation(conv_id)

    async def get_conversation_log(
        self, conv_id: str, last_n: int = 30
    ) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM turns WHERE conversation_id = ? "
            "ORDER BY id DESC LIMIT ?",
            (conv_id, last_n),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in reversed(rows)]

    async def get_all_recent_turns(self, since_seconds: float = 120) -> list[dict]:
        cutoff = time.time() - since_seconds
        cursor = await self._db.execute(
            "SELECT * FROM turns WHERE timestamp > ? ORDER BY timestamp ASC",
            (cutoff,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_turn_count(self, conv_id: str) -> int:
        cursor = await self._db.execute(
            "SELECT MAX(turn_number) as n FROM turns WHERE conversation_id = ?",
            (conv_id,),
        )
        row = await cursor.fetchone()
        return (row["n"] or 0) if row else 0

    async def get_last_assistant_message(self, conv_id: str) -> str | None:
        cursor = await self._db.execute(
            "SELECT content FROM turns WHERE conversation_id = ? AND role = 'assistant' "
            "ORDER BY turn_number DESC, burst_index DESC LIMIT 1",
            (conv_id,),
        )
        row = await cursor.fetchone()
        return row["content"] if row else None

    # ── Impressions ────────────────────────────────────────────────────────

    async def store_impression(self, imp: Impression):
        await self._db.execute(
            "INSERT OR REPLACE INTO impressions "
            "(id, type, content, payload, emotional_charge, source_conversation, "
            "source_turns, embedding, pressure, times_reinforced, times_repressed, "
            "created_at, last_reinforced_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                imp.id,
                imp.type.value,
                imp.content,
                json.dumps(imp.payload),
                imp.emotional_charge,
                imp.source_conversation,
                json.dumps(imp.source_turns),
                json.dumps(imp.embedding),
                imp.pressure,
                imp.times_reinforced,
                imp.times_repressed,
                imp.created_at,
                imp.last_reinforced_at,
            ),
        )
        await self._db.commit()

    async def get_active_impressions(self) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM impressions WHERE promoted = 0 AND repressed = 0 "
            "ORDER BY pressure DESC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_all_impressions_with_embeddings(self) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM impressions WHERE embedding != '[]'"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def reinforce_impression(self, imp_id: str, boost: float):
        await self._db.execute(
            "UPDATE impressions SET pressure = MIN(1.0, pressure + ?), "
            "times_reinforced = times_reinforced + 1, last_reinforced_at = ? "
            "WHERE id = ?",
            (boost, time.time(), imp_id),
        )
        await self._db.commit()

    async def update_impression_pressure(self, imp_id: str, new_pressure: float):
        await self._db.execute(
            "UPDATE impressions SET pressure = ? WHERE id = ?",
            (new_pressure, imp_id),
        )
        await self._db.commit()

    async def mark_promoted(self, imp_id: str):
        await self._db.execute(
            "UPDATE impressions SET promoted = 1 WHERE id = ?", (imp_id,)
        )
        await self._db.commit()

    async def mark_repressed(self, imp_id: str):
        await self._db.execute(
            "UPDATE impressions SET repressed = 1, times_repressed = times_repressed + 1 "
            "WHERE id = ?",
            (imp_id,),
        )
        await self._db.commit()

    # ── Promotions ─────────────────────────────────────────────────────────

    async def upsert_promotion(
        self,
        ptype: str,
        key: str,
        content: dict,
        confidence: float = 0.8,
        impression_ids: list[str] | None = None,
        conversation_id: str = "",
    ):
        now = time.time()
        cursor = await self._db.execute(
            "SELECT * FROM promotions WHERE key = ?", (key,)
        )
        existing = await cursor.fetchone()
        if existing:
            await self._db.execute(
                "UPDATE promotions SET content = ?, confidence = ?, updated_at = ? "
                "WHERE key = ?",
                (json.dumps(content), confidence, now, key),
            )
        else:
            await self._db.execute(
                "INSERT INTO promotions (type, key, content, confidence, "
                "source_impression_ids, conversation_id, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ptype,
                    key,
                    json.dumps(content),
                    confidence,
                    json.dumps(impression_ids or []),
                    conversation_id,
                    now,
                    now,
                ),
            )
        await self._db.commit()

    async def get_active_promotions(
        self, type_filter: str | None = None
    ) -> list[dict]:
        query = "SELECT * FROM promotions WHERE active = 1"
        params: list = []
        if type_filter:
            query += " AND type = ?"
            params.append(type_filter)
        query += " ORDER BY confidence DESC"
        cursor = await self._db.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Interrupts ─────────────────────────────────────────────────────────

    async def create_interrupt(
        self,
        conv_id: str,
        content: str,
        urgency: float = 0.9,
        reason: str = "",
        expires_after: int = 1,
    ) -> str:
        iid = str(uuid.uuid4())[:8]
        await self._db.execute(
            "INSERT INTO interrupts (id, conversation_id, content, urgency, "
            "reason, expires_after, uses_remaining, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (iid, conv_id, content, urgency, reason, expires_after, expires_after, time.time()),
        )
        await self._db.commit()
        return iid

    async def consume_interrupts(self, conv_id: str) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM interrupts WHERE conversation_id = ? AND uses_remaining > 0",
            (conv_id,),
        )
        rows = await cursor.fetchall()
        interrupts = [dict(r) for r in rows]
        if interrupts:
            ids = [r["id"] for r in interrupts]
            placeholders = ",".join("?" * len(ids))
            await self._db.execute(
                f"UPDATE interrupts SET uses_remaining = uses_remaining - 1 "
                f"WHERE id IN ({placeholders})",
                ids,
            )
            await self._db.commit()
        return interrupts

    # ── Tasks ──────────────────────────────────────────────────────────────

    async def create_task(
        self, conv_id: str, description: str, prompt: str
    ) -> str:
        task_id = str(uuid.uuid4())[:8]
        await self._db.execute(
            "INSERT INTO tasks (id, conversation_id, description, prompt, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (task_id, conv_id, description, prompt, time.time()),
        )
        await self._db.commit()
        return task_id

    async def get_pending_tasks(self) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM tasks WHERE status = 'pending' ORDER BY created_at ASC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def update_task(self, task_id: str, status: str, result: str = ""):
        await self._db.execute(
            "UPDATE tasks SET status = ?, result = ? WHERE id = ?",
            (status, result, task_id),
        )
        await self._db.commit()

    # ── Defense Events ─────────────────────────────────────────────────────

    async def log_defense_event(
        self,
        impression_id: str,
        defense: str,
        level: int,
        action_taken: str = "",
        led_to_positive: bool | None = None,
    ):
        await self._db.execute(
            "INSERT INTO defense_events (impression_id, defense, level, "
            "action_taken, led_to_positive, timestamp) VALUES (?, ?, ?, ?, ?, ?)",
            (
                impression_id,
                defense,
                level,
                action_taken,
                1 if led_to_positive else (0 if led_to_positive is False else None),
                time.time(),
            ),
        )
        await self._db.commit()

    async def get_recent_defense_events(self, limit: int = 10) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM defense_events ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Outcomes ───────────────────────────────────────────────────────────

    async def log_outcome(self, outcome: ResponseOutcome):
        await self._db.execute(
            "INSERT INTO outcomes (conversation_id, turn_number, signal, "
            "user_had_to_repeat, haiku_contradicted, response_useful, "
            "brief_reason, defense_applied, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                outcome.conversation_id,
                outcome.turn_number,
                outcome.signal.value,
                int(outcome.user_had_to_repeat),
                int(outcome.haiku_contradicted_itself),
                int(outcome.response_was_useful),
                outcome.brief_reason,
                outcome.defense_applied,
                outcome.timestamp,
            ),
        )
        await self._db.commit()

    async def get_recent_outcomes(self, limit: int = 20) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM outcomes ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def get_pending_outcomes(self) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM outcomes WHERE defense_applied = '' ORDER BY timestamp ASC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ── Growth Events ──────────────────────────────────────────────────────

    async def log_growth_event(
        self,
        mechanism: str,
        description: str = "",
        target_defense: str = "",
        target_pattern: str = "",
        recommendation: str = "",
        applied: bool = False,
        outcome: str = "",
    ):
        await self._db.execute(
            "INSERT INTO growth_events (mechanism, description, target_defense, "
            "target_pattern, recommendation, applied, outcome, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (mechanism, description, target_defense, target_pattern,
             recommendation, int(applied), outcome, time.time()),
        )
        await self._db.commit()

    # ── Repetition Patterns ────────────────────────────────────────────────

    async def upsert_repetition_pattern(
        self,
        pattern_id: str,
        pattern_type: str,
        description: str,
        trigger: str = "",
        response: str = "",
        maintaining_defense: str = "",
        occurrence_count: int = 1,
        severity: str = "LOW",
    ):
        now = time.time()
        cursor = await self._db.execute(
            "SELECT * FROM repetition_patterns WHERE id = ?", (pattern_id,)
        )
        existing = await cursor.fetchone()
        if existing:
            await self._db.execute(
                "UPDATE repetition_patterns SET occurrence_count = ?, severity = ?, "
                "last_seen = ? WHERE id = ?",
                (occurrence_count, severity, now, pattern_id),
            )
        else:
            await self._db.execute(
                "INSERT INTO repetition_patterns (id, pattern_type, description, "
                "trigger, response, maintaining_defense, occurrence_count, severity, "
                "first_seen, last_seen) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (pattern_id, pattern_type, description, trigger, response,
                 maintaining_defense, occurrence_count, severity, now, now),
            )
        await self._db.commit()

    async def get_active_patterns(self) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM repetition_patterns WHERE resolved = 0 "
            "ORDER BY last_seen DESC"
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def resolve_pattern(self, pattern_id: str):
        await self._db.execute(
            "UPDATE repetition_patterns SET resolved = 1 WHERE id = ?",
            (pattern_id,),
        )
        await self._db.commit()

    # ── User-Conversation Mapping ─────────────────────────────────────────

    async def get_or_create_user_conversation(self, user_id: str) -> str:
        cursor = await self._db.execute(
            "SELECT conversation_id FROM user_conversations WHERE user_id = ?",
            (user_id,),
        )
        row = await cursor.fetchone()
        if row:
            return row["conversation_id"]
        conv_id = await self.create_conversation()
        await self._db.execute(
            "INSERT INTO user_conversations (user_id, conversation_id, created_at) "
            "VALUES (?, ?, ?)",
            (user_id, conv_id, time.time()),
        )
        await self._db.commit()
        return conv_id

    # ── Sessions ──────────────────────────────────────────────────────────

    async def create_session(self, user_id: str, conversation_id: str) -> str:
        session_id = str(uuid.uuid4())[:12]
        now = time.time()
        await self._db.execute(
            "INSERT INTO sessions (id, user_id, conversation_id, started_at, "
            "last_activity, turn_count, topic_summary, closed) "
            "VALUES (?, ?, ?, ?, ?, 0, '', 0)",
            (session_id, user_id, conversation_id, now, now),
        )
        await self._db.commit()
        return session_id

    async def get_active_session(self, user_id: str) -> dict | None:
        cursor = await self._db.execute(
            "SELECT * FROM sessions WHERE user_id = ? AND closed = 0 "
            "ORDER BY started_at DESC LIMIT 1",
            (user_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def close_session(self, session_id: str):
        await self._db.execute(
            "UPDATE sessions SET closed = 1 WHERE id = ?", (session_id,)
        )
        await self._db.commit()

    async def touch_session(self, session_id: str):
        await self._db.execute(
            "UPDATE sessions SET last_activity = ?, turn_count = turn_count + 1 "
            "WHERE id = ?",
            (time.time(), session_id),
        )
        await self._db.commit()

    async def update_session_summary(self, session_id: str, summary: str):
        await self._db.execute(
            "UPDATE sessions SET topic_summary = ? WHERE id = ?",
            (summary, session_id),
        )
        await self._db.commit()

    async def get_previous_session(
        self, user_id: str, before_session_id: str
    ) -> dict | None:
        cursor = await self._db.execute(
            "SELECT started_at FROM sessions WHERE id = ?",
            (before_session_id,),
        )
        current = await cursor.fetchone()
        if not current:
            return None
        cursor = await self._db.execute(
            "SELECT * FROM sessions WHERE user_id = ? AND started_at < ? "
            "ORDER BY started_at DESC LIMIT 1",
            (user_id, current["started_at"]),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def get_session_closing_turns(
        self,
        conversation_id: str,
        session_started_at: float,
        session_last_activity: float,
        n: int = 5,
    ) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM turns WHERE conversation_id = ? "
            "AND timestamp >= ? AND timestamp <= ? "
            "ORDER BY id DESC LIMIT ?",
            (conversation_id, session_started_at, session_last_activity, n),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in reversed(rows)]

    # ── Agent Tasks ─────────────────────────────────────────────────────────

    def _task_to_row(self, task: AgentTask) -> tuple:
        return (
            task.id,
            task.parent_task_id,
            task.conversation_id,
            task.model,
            task.description,
            json.dumps([vars(t) for t in task.tools]),
            task.phase.value,
            task.status.value,
            task.max_subtasks,
            task.depth,
            task.max_depth,
            task.result,
            json.dumps(task.children),
            json.dumps(task.phase_results),
            task.retry_count,
            task.max_retries,
            task.created_at,
            time.time(),
        )

    def _row_to_task(self, row: dict) -> AgentTask:
        tools_raw = json.loads(row.get("tools", "[]"))
        return AgentTask(
            id=row["id"],
            parent_task_id=row.get("parent_task_id"),
            conversation_id=row.get("conversation_id", ""),
            model=row.get("model", ""),
            description=row["description"],
            tools=[ToolSpec(**t) for t in tools_raw],
            phase=TaskPhase(row.get("phase", "plan")),
            status=TaskStatus(row.get("status", "queued")),
            max_subtasks=row.get("max_subtasks", 5),
            depth=row.get("depth", 0),
            max_depth=row.get("max_depth", 3),
            result=row.get("result", ""),
            children=json.loads(row.get("children", "[]")),
            phase_results=json.loads(row.get("phase_results", "{}")),
            retry_count=row.get("retry_count", 0),
            max_retries=row.get("max_retries", 2),
            created_at=row.get("created_at", 0.0),
        )

    async def create_agent_task(self, task: AgentTask) -> str:
        await self._db.execute(
            "INSERT INTO agent_tasks (id, parent_task_id, conversation_id, model, "
            "description, tools, phase, status, max_subtasks, depth, max_depth, "
            "result, children, phase_results, retry_count, max_retries, "
            "created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            self._task_to_row(task),
        )
        await self._db.commit()
        return task.id

    async def get_agent_task(self, task_id: str) -> AgentTask | None:
        cursor = await self._db.execute(
            "SELECT * FROM agent_tasks WHERE id = ?", (task_id,)
        )
        row = await cursor.fetchone()
        return self._row_to_task(dict(row)) if row else None

    async def get_queued_agent_tasks(self) -> list[AgentTask]:
        cursor = await self._db.execute(
            "SELECT * FROM agent_tasks WHERE status = 'queued' "
            "ORDER BY created_at ASC"
        )
        rows = await cursor.fetchall()
        return [self._row_to_task(dict(r)) for r in rows]

    async def get_children_tasks(self, parent_id: str) -> list[AgentTask]:
        cursor = await self._db.execute(
            "SELECT * FROM agent_tasks WHERE parent_task_id = ?",
            (parent_id,),
        )
        rows = await cursor.fetchall()
        return [self._row_to_task(dict(r)) for r in rows]

    async def update_agent_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        phase: TaskPhase | None = None,
    ):
        if phase is not None:
            await self._db.execute(
                "UPDATE agent_tasks SET status = ?, phase = ?, updated_at = ? "
                "WHERE id = ?",
                (status.value, phase.value, time.time(), task_id),
            )
        else:
            await self._db.execute(
                "UPDATE agent_tasks SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, time.time(), task_id),
            )
        await self._db.commit()

    async def update_agent_task_phase_result(
        self, task_id: str, phase: str, result: str
    ):
        task = await self.get_agent_task(task_id)
        if not task:
            return
        task.phase_results[phase] = result
        await self._db.execute(
            "UPDATE agent_tasks SET phase_results = ?, updated_at = ? WHERE id = ?",
            (json.dumps(task.phase_results), time.time(), task_id),
        )
        await self._db.commit()

    async def update_agent_task_final(self, task: AgentTask):
        await self._db.execute(
            "UPDATE agent_tasks SET status = ?, phase = ?, result = ?, "
            "children = ?, phase_results = ?, retry_count = ?, updated_at = ? "
            "WHERE id = ?",
            (
                task.status.value,
                task.phase.value,
                task.result,
                json.dumps(task.children),
                json.dumps(task.phase_results),
                task.retry_count,
                time.time(),
                task.id,
            ),
        )
        await self._db.commit()

    async def add_child_to_task(self, parent_id: str, child_id: str):
        task = await self.get_agent_task(parent_id)
        if not task:
            return
        task.children.append(child_id)
        await self._db.execute(
            "UPDATE agent_tasks SET children = ?, updated_at = ? WHERE id = ?",
            (json.dumps(task.children), time.time(), parent_id),
        )
        await self._db.commit()

    async def recover_running_tasks(self):
        """Reset RUNNING tasks to QUEUED on startup (crash recovery)."""
        await self._db.execute(
            "UPDATE agent_tasks SET status = 'queued', updated_at = ? "
            "WHERE status = 'running'",
            (time.time(),),
        )
        await self._db.commit()

    # ── Superego Events ──────────────────────────────────────────────────

    async def log_superego_event(
        self,
        event_type: str,
        tier: str,
        rule_id: str,
        description: str = "",
        conversation_id: str = "",
        turn_number: int = 0,
        pressure: float = 0.0,
    ):
        await self._db.execute(
            "INSERT INTO superego_events (event_type, tier, rule_id, description, "
            "conversation_id, turn_number, pressure, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (event_type, tier, rule_id, description, conversation_id,
             turn_number, pressure, time.time()),
        )
        await self._db.commit()

    async def get_moral_tension_count(self, rule_id: str | None = None) -> int:
        if rule_id:
            cursor = await self._db.execute(
                "SELECT COUNT(*) as n FROM superego_events "
                "WHERE event_type = 'moral_tension' AND rule_id = ?",
                (rule_id,),
            )
        else:
            cursor = await self._db.execute(
                "SELECT COUNT(*) as n FROM superego_events "
                "WHERE event_type = 'moral_tension'"
            )
        row = await cursor.fetchone()
        return row["n"] if row else 0

    async def get_recent_superego_events(self, limit: int = 20) -> list[dict]:
        cursor = await self._db.execute(
            "SELECT * FROM superego_events ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]
