"""Telegram entry point for the Freudian Mind."""

from __future__ import annotations

import asyncio
import logging
import signal
import time

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from .config import MindConfig
from .mind import FreudianMind

logger = logging.getLogger(__name__)


class TelegramBot:
    """Telegram bot wrapping a single shared FreudianMind instance.

    One Telegram chat = one ongoing relationship (one conv_id per user).
    Sessions are soft boundaries within that relationship, tracked by
    inactivity gaps.
    """

    def __init__(self, config: MindConfig):
        self.config = config
        self.mind = FreudianMind(config)
        self._session_timeout = config.session_timeout_minutes * 60
        self._user_locks: dict[str, asyncio.Lock] = {}

    def _lock_for(self, user_id: str) -> asyncio.Lock:
        if user_id not in self._user_locks:
            self._user_locks[user_id] = asyncio.Lock()
        return self._user_locks[user_id]

    # ── Session management ────────────────────────────────────────────────

    async def _ensure_session(self, user_id: str) -> tuple[str, str, bool]:
        """Return (conv_id, session_id, is_new_session)."""
        conv_id = await self.mind.state.get_or_create_user_conversation(user_id)

        # Ensure conv_id is tracked by the mind
        if conv_id not in self.mind.conversations:
            self.mind.conversations[conv_id] = await self.mind.state.get_turn_count(
                conv_id
            )

        session = await self.mind.state.get_active_session(user_id)

        if session and (time.time() - session["last_activity"]) < self._session_timeout:
            return conv_id, session["id"], False

        # Close expired session if any
        if session:
            await self.mind.state.close_session(session["id"])

        # Start new session
        session_id = await self.mind.state.create_session(user_id, conv_id)
        return conv_id, session_id, True

    async def _build_bridge_context(
        self, user_id: str, current_session_id: str
    ) -> str:
        """Build cross-session bridge context from the previous session."""
        prev = await self.mind.state.get_previous_session(
            user_id, current_session_id
        )
        if not prev:
            return ""

        turns = await self.mind.state.get_session_closing_turns(
            prev["conversation_id"],
            prev["started_at"],
            prev["last_activity"],
            self.config.cross_session_turns,
        )
        if not turns:
            return ""

        lines = []
        if prev.get("topic_summary"):
            lines.append(f"Topic: {prev['topic_summary']}")

        # Calculate how long ago the previous session was
        gap = time.time() - prev["last_activity"]
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

    # ── Handlers ──────────────────────────────────────────────────────────

    async def _handle_start(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        user_id = str(update.effective_user.id)
        async with self._lock_for(user_id):
            await self._ensure_session(user_id)
        await update.message.reply_text(
            "Hey. I'm here whenever you want to talk."
        )

    async def _handle_new(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Explicitly close the current session (same conv_id continues)."""
        user_id = str(update.effective_user.id)
        async with self._lock_for(user_id):
            session = await self.mind.state.get_active_session(user_id)
            if session:
                await self.mind.state.close_session(session["id"])
        await update.message.reply_text("Fresh start. What's on your mind?")

    async def _handle_state(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show mind state — mirrors CLI /state command."""
        impressions = await self.mind.state.get_active_impressions()
        promotions = await self.mind.state.get_active_promotions()
        health = self.mind.defense_profile.get_health_report()

        lines = [
            f"Impressions: {len(impressions)} active",
            f"Promotions: {len(promotions)} active",
            f"Maturity: {health.get('maturity_score', 0):.2f} ({health.get('maturity_label', 'n/a')})",
            f"Flexibility: {health.get('flexibility_score', 0):.2f}",
            f"Growth: {health.get('growth_velocity', 0):+.3f} ({health.get('growth_direction', 'stable')})",
        ]

        if health.get("is_neurotic"):
            lines.append("⚠ Neurotic pattern detected")

        if health.get("warning_signs"):
            for w in health["warning_signs"]:
                lines.append(f"  · {w}")

        await update.message.reply_text("\n".join(lines))

    async def _handle_message(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Main message handler — the core loop."""
        if not update.message or not update.message.text:
            return

        user_id = str(update.effective_user.id)
        text = update.message.text

        async with self._lock_for(user_id):
            conv_id, session_id, is_new_session = await self._ensure_session(user_id)

            # Build bridge context only on first message of a new session
            bridge_context = ""
            if is_new_session:
                bridge_context = await self._build_bridge_context(
                    user_id, session_id
                )

            burst = await self.mind.chat(
                conv_id, text, bridge_context=bridge_context
            )

            await self.mind.state.touch_session(session_id)

        # Send burst messages with delay (outside the lock)
        for i, msg in enumerate(burst.messages):
            if i > 0:
                await asyncio.sleep(self.config.burst_delay_ms / 1000.0)
            await update.message.reply_text(msg)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    async def run(self) -> None:
        await self.mind.start()
        logger.info("Freudian Mind started")

        app = Application.builder().token(self.config.telegram_bot_token).build()
        app.add_handler(CommandHandler("start", self._handle_start))
        app.add_handler(CommandHandler("new", self._handle_new))
        app.add_handler(CommandHandler("state", self._handle_state))
        app.add_handler(
            MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
        )

        # Graceful shutdown via signals
        loop = asyncio.get_event_loop()
        stop_event = asyncio.Event()

        def _signal_handler() -> None:
            logger.info("Shutdown signal received")
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _signal_handler)

        async with app:
            await app.start()
            await app.updater.start_polling()
            logger.info("Telegram bot polling — ready")
            await stop_event.wait()
            logger.info("Stopping Telegram bot...")
            await app.updater.stop()
            await app.stop()

        await self.mind.stop()
        logger.info("Freudian Mind stopped")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    config = MindConfig()
    if not config.telegram_bot_token:
        print("Error: TELEGRAM_BOT_TOKEN not set in environment or .env")
        return

    bot = TelegramBot(config)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
