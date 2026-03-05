"""MindConfig — all tunable parameters for the Freudian Mind."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


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

    # ── Agent Tasks ──
    max_concurrent_agents: int = 3
    default_agent_model: str = "claude-sonnet-4-5-20250514"
    max_task_depth: int = 3
    max_subtasks_per_task: int = 5
    agent_task_timeout: float = 300.0  # per-phase timeout in seconds
    max_task_retries: int = 2

    # ── Session ──
    session_timeout_minutes: int = 45
    session_greeting: bool = True
    cross_session_turns: int = 5

    # ── Telegram ──
    telegram_bot_token: str = ""

    # ── Persistence ──
    db_path: str = "freudian_mind.db"

    def __post_init__(self):
        self.conscious_model = os.getenv(
            "FREUDIAN_CONSCIOUS_MODEL", self.conscious_model
        )
        self.preconscious_model = os.getenv(
            "FREUDIAN_PRECONSCIOUS_MODEL", self.preconscious_model
        )
        self.unconscious_model = os.getenv(
            "FREUDIAN_UNCONSCIOUS_MODEL", self.unconscious_model
        )
        self.db_path = os.getenv("FREUDIAN_DB_PATH", self.db_path)
        self.default_agent_model = os.getenv(
            "FREUDIAN_AGENT_MODEL", self.default_agent_model
        )
        self.telegram_bot_token = os.getenv(
            "TELEGRAM_BOT_TOKEN", self.telegram_bot_token
        )
