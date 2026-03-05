"""Tests for the Conscious Layer."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anima.config import MindConfig
from anima.layers.conscious import ConsciousLayer
from anima.prompts.conscious_prompts import build_system_prompt
from anima.state import SharedState


def _mock_api_response(text: str):
    """Create a mock Anthropic API response."""
    mock = MagicMock()
    mock.content = [MagicMock(text=text)]
    return mock


class TestSystemPrompt:
    def test_base_personality_only(self):
        prompt = build_system_prompt("Be nice.", [], [])
        assert "Be nice." in prompt
        assert "⚡ IMPORTANT" not in prompt

    def test_with_memory_promotions(self):
        promos = [
            {"type": "memory", "content": '{"fact": "User likes Python"}'},
        ]
        prompt = build_system_prompt("Be nice.", promos, [])
        assert "User likes Python" in prompt
        assert "Things you know" in prompt

    def test_with_directive_promotions(self):
        promos = [
            {"type": "directive", "content": '{"instruction": "Be concise"}'},
        ]
        prompt = build_system_prompt("Be nice.", promos, [])
        assert "Be concise" in prompt
        assert "Behavioral guidelines" in prompt

    def test_with_interrupts(self):
        interrupts = [{"content": "Stop giving wrong math answers"}]
        prompt = build_system_prompt("Be nice.", [], interrupts)
        assert "⚡ IMPORTANT" in prompt
        assert "Stop giving wrong math answers" in prompt

    def test_burst_instructions_included(self):
        prompt = build_system_prompt("Be nice.", [], [])
        assert "1-3 sentences" in prompt
        assert "[DONE]" in prompt

    def test_no_promotions_or_interrupts(self):
        prompt = build_system_prompt("personality", [], [])
        assert "personality" in prompt
        assert "⚡ IMPORTANT" not in prompt
        assert "Things you know" not in prompt


class TestConsciousLayer:
    @patch("anima.layers.conscious.anthropic.AsyncAnthropic")
    async def test_single_message_burst(self, mock_anthropic_cls, state, config):
        mock_client = AsyncMock()
        mock_anthropic_cls.return_value = mock_client

        # First message, then [DONE]
        mock_client.messages.create = AsyncMock(
            side_effect=[
                _mock_api_response("Hello there!"),
                _mock_api_response("[DONE]"),
            ]
        )

        layer = ConsciousLayer(state, config)
        layer.client = mock_client
        await state.create_conversation("c1")

        burst = await layer.respond("c1", "hi")
        assert len(burst.messages) == 1
        assert burst.messages[0] == "Hello there!"

    @patch("anima.layers.conscious.anthropic.AsyncAnthropic")
    async def test_multi_message_burst(self, mock_anthropic_cls, state, config):
        mock_client = AsyncMock()
        mock_anthropic_cls.return_value = mock_client

        mock_client.messages.create = AsyncMock(
            side_effect=[
                _mock_api_response("First thought."),
                _mock_api_response("Second thought."),
                _mock_api_response("[DONE]"),
            ]
        )

        layer = ConsciousLayer(state, config)
        layer.client = mock_client
        await state.create_conversation("c1")

        burst = await layer.respond("c1", "tell me more")
        assert len(burst.messages) == 2

    @patch("anima.layers.conscious.anthropic.AsyncAnthropic")
    async def test_max_burst_limit(self, mock_anthropic_cls, state, config):
        mock_client = AsyncMock()
        mock_anthropic_cls.return_value = mock_client
        config.max_burst_messages = 3

        mock_client.messages.create = AsyncMock(
            return_value=_mock_api_response("More words.")
        )

        layer = ConsciousLayer(state, config)
        layer.client = mock_client
        await state.create_conversation("c1")

        burst = await layer.respond("c1", "go on")
        assert len(burst.messages) <= 3

    @patch("anima.layers.conscious.anthropic.AsyncAnthropic")
    async def test_empty_response_fallback(self, mock_anthropic_cls, state, config):
        mock_client = AsyncMock()
        mock_anthropic_cls.return_value = mock_client

        mock_client.messages.create = AsyncMock(
            return_value=_mock_api_response("")
        )

        layer = ConsciousLayer(state, config)
        layer.client = mock_client
        await state.create_conversation("c1")

        burst = await layer.respond("c1", "hello")
        assert len(burst.messages) >= 1
        assert burst.messages[0] == "I'm here."

    @patch("anima.layers.conscious.anthropic.AsyncAnthropic")
    async def test_done_stripped_from_message(self, mock_anthropic_cls, state, config):
        mock_client = AsyncMock()
        mock_anthropic_cls.return_value = mock_client

        mock_client.messages.create = AsyncMock(
            side_effect=[
                _mock_api_response("Here's my answer. [DONE]"),
                _mock_api_response("[DONE]"),
            ]
        )

        layer = ConsciousLayer(state, config)
        layer.client = mock_client
        await state.create_conversation("c1")

        burst = await layer.respond("c1", "question")
        assert "[DONE]" not in burst.messages[0]

    @patch("anima.layers.conscious.anthropic.AsyncAnthropic")
    async def test_interrupts_consumed(self, mock_anthropic_cls, state, config):
        mock_client = AsyncMock()
        mock_anthropic_cls.return_value = mock_client

        mock_client.messages.create = AsyncMock(
            side_effect=[
                _mock_api_response("OK fixing that."),
                _mock_api_response("[DONE]"),
            ]
        )

        layer = ConsciousLayer(state, config)
        layer.client = mock_client
        await state.create_conversation("c1")
        await state.create_interrupt("c1", "You gave wrong math!")

        burst = await layer.respond("c1", "what's 2+2?")
        assert len(burst.interrupts_applied) == 1
