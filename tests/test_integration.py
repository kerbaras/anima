"""Integration tests for the full FreudianMind."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anima.config import MindConfig
from anima.mind import FreudianMind


def _mock_haiku_response(text: str):
    mock = MagicMock()
    mock.content = [MagicMock(text=text)]
    return mock


def _mock_classifier_response(signal: str = "neutral"):
    data = {
        "signal": signal,
        "user_had_to_repeat": False,
        "haiku_contradicted_itself": False,
        "response_was_useful": True,
        "brief_reason": "test",
    }
    mock = MagicMock()
    mock.content = [MagicMock(text=json.dumps(data))]
    return mock


class TestFreudianMind:
    @patch("anima.systems.outcome.anthropic.AsyncAnthropic")
    @patch("anima.layers.conscious.anthropic.AsyncAnthropic")
    @patch("anima.layers.unconscious.anthropic.AsyncAnthropic")
    @patch("anima.layers.preconscious.anthropic.AsyncAnthropic")
    async def test_basic_chat_flow(
        self, mock_pre_cls, mock_unc_cls, mock_con_cls, mock_out_cls
    ):
        # Set up mock clients
        mock_conscious = AsyncMock()
        mock_con_cls.return_value = mock_conscious
        mock_conscious.messages.create = AsyncMock(
            side_effect=[
                _mock_haiku_response("Hey there!"),
                _mock_haiku_response("[DONE]"),
            ]
        )

        mock_classifier = AsyncMock()
        mock_out_cls.return_value = mock_classifier
        mock_classifier.messages.create = AsyncMock(
            return_value=_mock_classifier_response("positive")
        )

        config = MindConfig(db_path=":memory:")
        mind = FreudianMind(config)
        await mind.state.initialize()

        # Override clients directly
        mind.conscious.client = mock_conscious
        mind.outcome_classifier.client = mock_classifier

        conv_id = await mind.new_conversation()
        burst = await mind.chat(conv_id, "Hello!")

        assert len(burst.messages) >= 1
        assert burst.messages[0] == "Hey there!"
        assert burst.conversation_id == conv_id

        await mind.state.close()

    @patch("anima.systems.outcome.anthropic.AsyncAnthropic")
    @patch("anima.layers.conscious.anthropic.AsyncAnthropic")
    @patch("anima.layers.unconscious.anthropic.AsyncAnthropic")
    @patch("anima.layers.preconscious.anthropic.AsyncAnthropic")
    async def test_outcome_classified_on_second_turn(
        self, mock_pre_cls, mock_unc_cls, mock_con_cls, mock_out_cls
    ):
        mock_conscious = AsyncMock()
        mock_con_cls.return_value = mock_conscious
        mock_conscious.messages.create = AsyncMock(
            return_value=_mock_haiku_response("Response [DONE]")
        )

        mock_classifier = AsyncMock()
        mock_out_cls.return_value = mock_classifier
        mock_classifier.messages.create = AsyncMock(
            return_value=_mock_classifier_response("correction")
        )

        config = MindConfig(db_path=":memory:")
        mind = FreudianMind(config)
        await mind.state.initialize()

        mind.conscious.client = mock_conscious
        mind.outcome_classifier.client = mock_classifier

        conv_id = await mind.new_conversation()

        # First turn — no classification
        await mind.chat(conv_id, "What's 2+2?")
        assert mock_classifier.messages.create.call_count == 0

        # Second turn — should classify
        mock_conscious.messages.create.reset_mock()
        mock_conscious.messages.create = AsyncMock(
            return_value=_mock_haiku_response("4 [DONE]")
        )
        await mind.chat(conv_id, "Actually it's 4, not 5")
        assert mock_classifier.messages.create.call_count == 1

        await mind.state.close()

    @patch("anima.systems.outcome.anthropic.AsyncAnthropic")
    @patch("anima.layers.conscious.anthropic.AsyncAnthropic")
    @patch("anima.layers.unconscious.anthropic.AsyncAnthropic")
    @patch("anima.layers.preconscious.anthropic.AsyncAnthropic")
    async def test_turns_logged(
        self, mock_pre_cls, mock_unc_cls, mock_con_cls, mock_out_cls
    ):
        mock_conscious = AsyncMock()
        mock_con_cls.return_value = mock_conscious
        mock_conscious.messages.create = AsyncMock(
            side_effect=[
                _mock_haiku_response("Hi!"),
                _mock_haiku_response("[DONE]"),
            ]
        )

        mock_classifier = AsyncMock()
        mock_out_cls.return_value = mock_classifier

        config = MindConfig(db_path=":memory:")
        mind = FreudianMind(config)
        await mind.state.initialize()

        mind.conscious.client = mock_conscious
        mind.outcome_classifier.client = mock_classifier

        conv_id = await mind.new_conversation()
        await mind.chat(conv_id, "Hello")

        log = await mind.state.get_conversation_log(conv_id)
        assert len(log) >= 2  # user + assistant
        roles = [t["role"] for t in log]
        assert "user" in roles
        assert "assistant" in roles

        await mind.state.close()

    @patch("anima.systems.outcome.anthropic.AsyncAnthropic")
    @patch("anima.layers.conscious.anthropic.AsyncAnthropic")
    @patch("anima.layers.unconscious.anthropic.AsyncAnthropic")
    @patch("anima.layers.preconscious.anthropic.AsyncAnthropic")
    async def test_interrupt_injection(
        self, mock_pre_cls, mock_unc_cls, mock_con_cls, mock_out_cls
    ):
        mock_conscious = AsyncMock()
        mock_con_cls.return_value = mock_conscious
        mock_conscious.messages.create = AsyncMock(
            side_effect=[
                _mock_haiku_response("Let me correct that."),
                _mock_haiku_response("[DONE]"),
            ]
        )

        mock_classifier = AsyncMock()
        mock_out_cls.return_value = mock_classifier

        config = MindConfig(db_path=":memory:")
        mind = FreudianMind(config)
        await mind.state.initialize()

        mind.conscious.client = mock_conscious
        mind.outcome_classifier.client = mock_classifier

        conv_id = await mind.new_conversation()

        # Create an interrupt before chatting
        await mind.state.create_interrupt(conv_id, "You made an error earlier!")

        burst = await mind.chat(conv_id, "Tell me about X")
        assert len(burst.interrupts_applied) == 1

        await mind.state.close()
