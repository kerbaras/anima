"""Tests for Telegram bot session management and bridge context."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freudian_mind.config import MindConfig
from freudian_mind.state import SharedState


# ── Session state methods ─────────────────────────────────────────────────


class TestUserConversationMapping:
    async def test_creates_new_conversation(self, state):
        conv_id = await state.get_or_create_user_conversation("user-1")
        assert len(conv_id) > 0

    async def test_returns_same_conv_for_same_user(self, state):
        first = await state.get_or_create_user_conversation("user-1")
        second = await state.get_or_create_user_conversation("user-1")
        assert first == second

    async def test_different_users_get_different_convs(self, state):
        c1 = await state.get_or_create_user_conversation("user-1")
        c2 = await state.get_or_create_user_conversation("user-2")
        assert c1 != c2


class TestSessions:
    async def test_create_session(self, state):
        conv_id = await state.create_conversation("c1")
        session_id = await state.create_session("user-1", conv_id)
        assert len(session_id) > 0

    async def test_get_active_session(self, state):
        conv_id = await state.create_conversation("c1")
        sid = await state.create_session("user-1", conv_id)
        session = await state.get_active_session("user-1")
        assert session is not None
        assert session["id"] == sid
        assert session["closed"] == 0

    async def test_no_active_session_initially(self, state):
        session = await state.get_active_session("user-1")
        assert session is None

    async def test_close_session(self, state):
        conv_id = await state.create_conversation("c1")
        sid = await state.create_session("user-1", conv_id)
        await state.close_session(sid)
        session = await state.get_active_session("user-1")
        assert session is None

    async def test_touch_session(self, state):
        conv_id = await state.create_conversation("c1")
        sid = await state.create_session("user-1", conv_id)
        before = (await state.get_active_session("user-1"))["turn_count"]
        await state.touch_session(sid)
        after = (await state.get_active_session("user-1"))["turn_count"]
        assert after == before + 1

    async def test_update_session_summary(self, state):
        conv_id = await state.create_conversation("c1")
        sid = await state.create_session("user-1", conv_id)
        await state.update_session_summary(sid, "talked about kubernetes")
        session = await state.get_active_session("user-1")
        assert session["topic_summary"] == "talked about kubernetes"

    async def test_get_previous_session(self, state):
        conv_id = await state.create_conversation("c1")
        sid1 = await state.create_session("user-1", conv_id)
        await state.close_session(sid1)
        sid2 = await state.create_session("user-1", conv_id)

        prev = await state.get_previous_session("user-1", sid2)
        assert prev is not None
        assert prev["id"] == sid1

    async def test_no_previous_session_when_first(self, state):
        conv_id = await state.create_conversation("c1")
        sid = await state.create_session("user-1", conv_id)
        prev = await state.get_previous_session("user-1", sid)
        assert prev is None


class TestSessionClosingTurns:
    async def test_get_turns_from_session_window(self, state):
        conv_id = await state.create_conversation("c1")
        now = time.time()

        # Log turns with timestamps in the session window
        await state.log_turn("c1", 1, "user", "hello")
        await state.log_turn("c1", 1, "assistant", "hi there")
        await state.log_turn("c1", 2, "user", "how are you?")
        await state.log_turn("c1", 2, "assistant", "doing well")

        turns = await state.get_session_closing_turns(
            "c1", now - 1, time.time() + 1, n=3
        )
        assert len(turns) <= 3
        assert len(turns) > 0

    async def test_respects_limit(self, state):
        conv_id = await state.create_conversation("c1")
        now = time.time()

        for i in range(10):
            await state.log_turn("c1", i + 1, "user", f"msg {i}")

        turns = await state.get_session_closing_turns(
            "c1", now - 1, time.time() + 1, n=3
        )
        assert len(turns) == 3


# ── Telegram bot logic ────────────────────────────────────────────────────


class TestTelegramBotSessionLogic:
    """Test session management logic without requiring python-telegram-bot."""

    @pytest.fixture
    async def bot_deps(self, state, config):
        """Provide state and config for bot-like testing."""
        return state, config

    async def test_session_timeout_detection(self, state, config):
        """Sessions older than timeout should be considered expired."""
        conv_id = await state.create_conversation("c1")
        sid = await state.create_session("user-1", conv_id)

        # Manually set last_activity to be past the timeout
        past = time.time() - (config.session_timeout_minutes * 60) - 10
        await state._db.execute(
            "UPDATE sessions SET last_activity = ? WHERE id = ?",
            (past, sid),
        )
        await state._db.commit()

        session = await state.get_active_session("user-1")
        assert session is not None
        # The session exists but is expired
        assert (time.time() - session["last_activity"]) > (
            config.session_timeout_minutes * 60
        )

    async def test_bridge_context_with_two_sessions(self, state, config):
        """Bridge context should pull turns from the previous session."""
        conv_id = await state.get_or_create_user_conversation("user-1")

        # Session 1: log some turns
        sid1 = await state.create_session("user-1", conv_id)
        await state.log_turn(conv_id, 1, "user", "tell me about kubernetes")
        await state.log_turn(conv_id, 1, "assistant", "what do you want to know?")
        await state.log_turn(conv_id, 2, "user", "I'm stressed about the deadline")
        await state.log_turn(conv_id, 2, "assistant", "take a break, you've earned it")
        await state.touch_session(sid1)
        await state.close_session(sid1)

        # Session 2
        sid2 = await state.create_session("user-1", conv_id)

        # Get previous session and its closing turns
        prev = await state.get_previous_session("user-1", sid2)
        assert prev is not None
        assert prev["id"] == sid1

        turns = await state.get_session_closing_turns(
            prev["conversation_id"],
            prev["started_at"],
            prev["last_activity"],
            config.cross_session_turns,
        )
        assert len(turns) > 0
        # Should contain the conversation content
        contents = [t["content"] for t in turns]
        assert any("kubernetes" in c or "deadline" in c or "break" in c for c in contents)


class TestBridgeContextFormat:
    """Test the bridge context formatting logic (extracted from TelegramBot)."""

    def _format_bridge(self, prev_session: dict, turns: list[dict]) -> str:
        """Replicate the formatting logic from TelegramBot._build_bridge_context."""
        lines = []
        if prev_session.get("topic_summary"):
            lines.append(f"Topic: {prev_session['topic_summary']}")

        gap = time.time() - prev_session["last_activity"]
        if gap < 3600:
            ago = f"{int(gap / 60)} minutes ago"
        elif gap < 86400:
            ago = f"{gap / 3600:.1f} hours ago"
        else:
            ago = f"{gap / 86400:.1f} days ago"

        lines.append(f"Last time you spoke ({ago}), the conversation went:")
        for t in turns:
            role_label = "User" if t["role"] == "user" else "You"
            lines.append(f"[{role_label}] {t['content']}")

        return "\n".join(lines)

    def test_format_with_summary(self):
        prev = {
            "topic_summary": "kubernetes migration",
            "last_activity": time.time() - 7200,  # 2 hours ago
        }
        turns = [
            {"role": "user", "content": "I'm stressed"},
            {"role": "assistant", "content": "Take a break"},
        ]
        result = self._format_bridge(prev, turns)
        assert "Topic: kubernetes migration" in result
        assert "hours ago" in result
        assert "[User] I'm stressed" in result
        assert "[You] Take a break" in result

    def test_format_without_summary(self):
        prev = {
            "topic_summary": "",
            "last_activity": time.time() - 300,  # 5 min ago
        }
        turns = [{"role": "user", "content": "hey"}]
        result = self._format_bridge(prev, turns)
        assert "Topic:" not in result
        assert "minutes ago" in result

    def test_format_days_ago(self):
        prev = {
            "topic_summary": "",
            "last_activity": time.time() - 172800,  # 2 days ago
        }
        turns = [{"role": "user", "content": "hello"}]
        result = self._format_bridge(prev, turns)
        assert "days ago" in result
