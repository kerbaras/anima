"""Integration tests for the full FreudianMind."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from anima.config import MindConfig
from anima.llm import LLMResponse
from anima.mind import FreudianMind


def _mock_llm_response(text: str) -> LLMResponse:
    return LLMResponse(text=text, finish_reason="stop")


def _mock_classifier_response(signal: str = "neutral") -> LLMResponse:
    data = {
        "signal": signal,
        "user_had_to_repeat": False,
        "haiku_contradicted_itself": False,
        "response_was_useful": True,
        "brief_reason": "test",
    }
    return LLMResponse(text=json.dumps(data), finish_reason="stop")


class TestFreudianMind:
    @patch("anima.systems.outcome.complete")
    @patch("anima.layers.conscious.complete")
    async def test_basic_chat_flow(self, mock_conscious_complete, mock_outcome_complete):
        mock_conscious_complete.side_effect = [
            _mock_llm_response("Hey there!"),
            _mock_llm_response("[DONE]"),
        ]
        mock_outcome_complete.return_value = _mock_classifier_response("positive")

        config = MindConfig(db_path=":memory:")
        mind = FreudianMind(config)
        await mind.state.initialize()

        conv_id = await mind.new_conversation()
        burst = await mind.chat(conv_id, "Hello!")

        assert len(burst.messages) >= 1
        assert burst.messages[0] == "Hey there!"
        assert burst.conversation_id == conv_id

        await mind.state.close()

    @patch("anima.systems.outcome.complete")
    @patch("anima.layers.conscious.complete")
    async def test_outcome_classified_on_second_turn(
        self, mock_conscious_complete, mock_outcome_complete
    ):
        mock_conscious_complete.return_value = _mock_llm_response("Response [DONE]")
        mock_outcome_complete.return_value = _mock_classifier_response("correction")

        config = MindConfig(db_path=":memory:")
        mind = FreudianMind(config)
        await mind.state.initialize()

        conv_id = await mind.new_conversation()

        # First turn — no classification
        await mind.chat(conv_id, "What's 2+2?")
        assert mock_outcome_complete.call_count == 0

        # Second turn — should classify
        mock_conscious_complete.reset_mock()
        mock_conscious_complete.return_value = _mock_llm_response("4 [DONE]")
        await mind.chat(conv_id, "Actually it's 4, not 5")
        assert mock_outcome_complete.call_count == 1

        await mind.state.close()

    @patch("anima.systems.outcome.complete")
    @patch("anima.layers.conscious.complete")
    async def test_turns_logged(self, mock_conscious_complete, mock_outcome_complete):
        mock_conscious_complete.side_effect = [
            _mock_llm_response("Hi!"),
            _mock_llm_response("[DONE]"),
        ]

        config = MindConfig(db_path=":memory:")
        mind = FreudianMind(config)
        await mind.state.initialize()

        conv_id = await mind.new_conversation()
        await mind.chat(conv_id, "Hello")

        log = await mind.state.get_conversation_log(conv_id)
        assert len(log) >= 2  # user + assistant
        roles = [t["role"] for t in log]
        assert "user" in roles
        assert "assistant" in roles

        await mind.state.close()

    @patch("anima.systems.outcome.complete")
    @patch("anima.layers.conscious.complete")
    async def test_interrupt_injection(self, mock_conscious_complete, mock_outcome_complete):
        mock_conscious_complete.side_effect = [
            _mock_llm_response("Let me correct that."),
            _mock_llm_response("[DONE]"),
        ]

        config = MindConfig(db_path=":memory:")
        mind = FreudianMind(config)
        await mind.state.initialize()

        conv_id = await mind.new_conversation()

        # Create an interrupt before chatting
        await mind.state.create_interrupt(conv_id, "You made an error earlier!")

        burst = await mind.chat(conv_id, "Tell me about X")
        assert len(burst.interrupts_applied) == 1

        await mind.state.close()
