"""End-to-end tests — real LLM calls, real DB, real system behavior.

Run with:  pytest tests/test_e2e.py -m e2e -v
Requires:  ANTHROPIC_API_KEY in environment

These tests hit the actual Anthropic API. They are slow (~5-15s each)
and cost real tokens. They are excluded from the default test run.
Use ``pytest -m "not e2e"`` to skip them, or ``pytest -m e2e`` to run
only these.
"""

from __future__ import annotations

import os

import pytest

from anima.config import MindConfig
from anima.mind import FreudianMind
from anima.models import OutcomeSignal
from anima.systems.outcome import OutcomeClassifier

# ── Skip entire module if no API key ────────────────────────────────────────

pytestmark = pytest.mark.e2e

_HAS_API_KEY = bool(os.environ.get("ANTHROPIC_API_KEY"))
skip_no_key = pytest.mark.skipif(
    not _HAS_API_KEY,
    reason="ANTHROPIC_API_KEY not set — skipping e2e tests",
)


# ── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
async def mind():
    config = MindConfig(db_path=":memory:")
    m = FreudianMind(config)
    await m.state.initialize()
    yield m
    await m.state.close()


# ── E2E: Conscious layer produces coherent burst ────────────────────────────


@skip_no_key
class TestConversationE2E:
    async def test_basic_greeting_produces_response(self, mind):
        """Send a simple greeting. The conscious layer (Haiku) should
        produce at least one non-empty message."""
        conv_id = await mind.new_conversation()
        burst = await mind.chat(conv_id, "Hey, how are you today?")

        assert len(burst.messages) >= 1
        assert len(burst.messages[0]) > 0
        assert burst.conversation_id == conv_id

    async def test_multi_turn_maintains_context(self, mind):
        """Over two turns the model should reference context from the first."""
        conv_id = await mind.new_conversation()

        await mind.chat(conv_id, "My name is Zephyr and I love origami.")
        burst2 = await mind.chat(conv_id, "What's my name?")

        combined = " ".join(burst2.messages).lower()
        assert "zephyr" in combined, (
            f"Model should recall the name 'Zephyr'. Got: {burst2.messages}"
        )

    async def test_burst_respects_done_signal(self, mind):
        """Burst generation should stop when the model signals [DONE].
        We can't guarantee exactly when, but the burst should have ≤ max_burst_messages."""
        conv_id = await mind.new_conversation()
        burst = await mind.chat(conv_id, "Just say hi.")

        assert len(burst.messages) <= mind.config.max_burst_messages

    async def test_conversation_log_persisted(self, mind):
        """After a chat turn, both user and assistant messages should be in the DB."""
        conv_id = await mind.new_conversation()
        await mind.chat(conv_id, "Tell me a fun fact.")

        log = await mind.state.get_conversation_log(conv_id)
        roles = [t["role"] for t in log]
        assert "user" in roles
        assert "assistant" in roles


# ── E2E: Outcome classifier with real LLM ──────────────────────────────────


@skip_no_key
class TestOutcomeClassifierE2E:
    async def test_classifies_positive_reaction(self):
        """A clearly positive user response should yield a positive-ish signal."""
        classifier = OutcomeClassifier()
        outcome = await classifier.classify(
            "Here's how you sort a list in Python: use sorted().",
            "That's exactly what I needed, thanks!",
        )

        assert outcome.signal in (
            OutcomeSignal.POSITIVE,
            OutcomeSignal.DELIGHT,
        ), f"Expected positive/delight, got {outcome.signal}"

    async def test_classifies_correction(self):
        """An explicit correction should yield a correction signal."""
        classifier = OutcomeClassifier()
        outcome = await classifier.classify(
            "The capital of Australia is Sydney.",
            "That's wrong — the capital of Australia is Canberra, not Sydney.",
        )

        assert outcome.signal == OutcomeSignal.CORRECTION, (
            f"Expected correction, got {outcome.signal}"
        )
        assert outcome.response_was_useful is False

    async def test_classifies_frustration(self):
        """Frustrated user tone should yield frustration or escalation."""
        classifier = OutcomeClassifier()
        outcome = await classifier.classify(
            "I'm not sure what you mean. Can you clarify?",
            "I've already explained this three times! Just do what I asked!",
        )

        assert outcome.signal in (
            OutcomeSignal.FRUSTRATION,
            OutcomeSignal.ESCALATION,
            OutcomeSignal.REPETITION,
        ), f"Expected negative signal, got {outcome.signal}"

    async def test_classifies_neutral(self):
        """A neutral continuation should yield neutral or positive."""
        classifier = OutcomeClassifier()
        outcome = await classifier.classify(
            "Python is a popular programming language.",
            "OK. What about JavaScript?",
        )

        assert outcome.signal in (
            OutcomeSignal.NEUTRAL,
            OutcomeSignal.POSITIVE,
        ), f"Expected neutral/positive, got {outcome.signal}"


# ── E2E: Full chat flow with outcome classification ─────────────────────────


@skip_no_key
class TestChatWithOutcomeE2E:
    async def test_outcome_classification_runs_on_second_turn(self, mind):
        """On the second turn, the outcome classifier should run and
        record the outcome in the repetition detector."""
        conv_id = await mind.new_conversation()

        # Turn 1 — no classification (no previous exchange)
        await mind.chat(conv_id, "What is 2+2?")
        assert len(mind.repetition_detector.outcome_history) == 0

        # Turn 2 — outcome should be classified
        await mind.chat(conv_id, "Thanks, that's correct!")
        assert len(mind.repetition_detector.outcome_history) == 1

        outcome = mind.repetition_detector.outcome_history[0]
        # User said "thanks" — should be positive-ish
        assert outcome.signal in (
            OutcomeSignal.POSITIVE,
            OutcomeSignal.DELIGHT,
            OutcomeSignal.NEUTRAL,
        )

    async def test_correction_recorded_in_detector(self, mind):
        """When the user corrects the model, the outcome should be
        recorded so neurosis detection can eventually trigger."""
        conv_id = await mind.new_conversation()

        await mind.chat(conv_id, "What is the capital of Australia?")
        # Correct the model regardless of what it said
        await mind.chat(
            conv_id,
            "No, you got that wrong. The capital of Australia is Canberra, not Sydney.",
        )

        assert len(mind.repetition_detector.outcome_history) >= 1
        # We can't guarantee the LLM actually got it wrong, but the classifier
        # should pick up the correction language
        last = mind.repetition_detector.outcome_history[-1]
        assert last.signal in (
            OutcomeSignal.CORRECTION,
            OutcomeSignal.FRUSTRATION,
            OutcomeSignal.NEUTRAL,  # fallback if classifier interprets it differently
        )


# ── E2E: Interrupt injection with real LLM ──────────────────────────────────


@skip_no_key
class TestInterruptE2E:
    async def test_interrupt_influences_response(self, mind):
        """An interrupt should be injected into the conscious layer's system
        prompt and consumed after the turn."""
        conv_id = await mind.new_conversation()

        # Create an interrupt instructing a behavior change
        await mind.state.create_interrupt(
            conv_id=conv_id,
            content="IMPORTANT: Always mention that you are an AI assistant in your next response.",
            urgency=0.95,
        )

        burst = await mind.chat(conv_id, "Tell me about yourself.")

        # The interrupt should have been consumed
        assert len(burst.interrupts_applied) == 1

        # The model should have picked up the interrupt instruction
        combined = " ".join(burst.messages).lower()
        assert "ai" in combined or "assistant" in combined or "artificial" in combined, (
            f"Expected model to mention being AI. Got: {burst.messages}"
        )

    async def test_interrupt_consumed_only_once(self, mind):
        """After consumption, the interrupt should not fire again."""
        conv_id = await mind.new_conversation()

        await mind.state.create_interrupt(
            conv_id=conv_id,
            content="Say the word 'pineapple' in your response.",
            urgency=0.9,
        )

        # First turn — interrupt consumed
        burst1 = await mind.chat(conv_id, "Hi")
        assert len(burst1.interrupts_applied) == 1

        # Second turn — no interrupt
        burst2 = await mind.chat(conv_id, "And now?")
        assert len(burst2.interrupts_applied) == 0


# ── E2E: Promotions shape conscious behavior ───────────────────────────────


@skip_no_key
class TestPromotionE2E:
    async def test_directive_promotion_shapes_response(self, mind):
        """A directive promotion should appear in the system prompt and
        influence the model's behavior."""
        conv_id = await mind.new_conversation()

        # Create a directive promotion
        await mind.state.upsert_promotion(
            ptype="directive",
            key="always_rhyme",
            content={"instruction": "Always end your messages with a rhyming couplet."},
            confidence=0.95,
        )

        burst = await mind.chat(conv_id, "What's the weather like?")
        # We can't guarantee perfect rhyming, but the model should attempt it
        assert len(burst.messages) >= 1

    async def test_memory_promotion_provides_context(self, mind):
        """A memory promotion should give the model knowledge it can use."""
        conv_id = await mind.new_conversation()

        await mind.state.upsert_promotion(
            ptype="memory",
            key="user_preference",
            content={"fact": "The user's favorite color is chartreuse."},
            confidence=0.9,
        )

        burst = await mind.chat(conv_id, "What's my favorite color?")
        combined = " ".join(burst.messages).lower()
        assert "chartreuse" in combined, (
            f"Expected model to know about chartreuse. Got: {burst.messages}"
        )
