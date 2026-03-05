"""Tests for the Unconscious Layer."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from freudian_mind.agents.orchestrator import TaskOrchestrator
from freudian_mind.config import MindConfig
from freudian_mind.layers.unconscious import UnconsciousLayer
from freudian_mind.models import Impression, ImpressionType
from freudian_mind.state import SharedState
from freudian_mind.systems.defense import DefenseProfile
from freudian_mind.systems.growth import GrowthEngine
from freudian_mind.systems.idea_space import IdeaSpace
from freudian_mind.systems.neurosis import RepetitionDetector


def _mock_api_response(data: dict):
    mock = MagicMock()
    mock.content = [MagicMock(text=json.dumps(data))]
    return mock


def _make_layer(state, config):
    """Create an UnconsciousLayer with all required dependencies."""
    idea_space = IdeaSpace(state)
    defense_profile = DefenseProfile()
    repetition_detector = RepetitionDetector()
    growth_engine = GrowthEngine(defense_profile, repetition_detector)
    orchestrator = TaskOrchestrator(state, config)
    return UnconsciousLayer(
        state, idea_space, config, orchestrator, defense_profile, growth_engine
    )


class TestProcessImpression:
    @patch("freudian_mind.layers.unconscious.anthropic.AsyncAnthropic")
    async def test_stores_new_impression(self, mock_cls, state, config):
        layer = _make_layer(state, config)

        item = {
            "type": "pattern",
            "content": "user prefers concise answers",
            "emotional_charge": 0.3,
            "source_conversation": "c1",
            "urgency": "medium",
        }
        await layer._process_impression(item)

        imps = await state.get_active_impressions()
        assert len(imps) == 1
        assert imps[0]["content"] == "user prefers concise answers"

    @patch("freudian_mind.layers.unconscious.anthropic.AsyncAnthropic")
    async def test_critical_correction_creates_interrupt(self, mock_cls, state, config):
        layer = _make_layer(state, config)
        await state.create_conversation("c1")

        item = {
            "type": "correction",
            "content": "wrong answer",
            "source_conversation": "c1",
            "urgency": "critical",
            "payload": {"correction": "2+2=4 not 5"},
        }
        await layer._process_impression(item)

        # Should create interrupt, not impression
        imps = await state.get_active_impressions()
        assert len(imps) == 0

        interrupts = await state.consume_interrupts("c1")
        assert len(interrupts) == 1
        assert "2+2=4" in interrupts[0]["content"]

    @patch("freudian_mind.layers.unconscious.anthropic.AsyncAnthropic")
    async def test_reinforces_similar_impression(self, mock_cls, state, config):
        layer = _make_layer(state, config)

        # Store existing impression with similarity key
        imp = Impression(
            id="existing",
            content="user likes code",
            payload={"similarity_key": "code_preference"},
            pressure=0.3,
        )
        await state.store_impression(imp)

        # Process similar impression
        item = {
            "type": "pattern",
            "content": "user prefers coding examples",
            "urgency": "medium",
            "similarity_key": "code_preference",
        }
        await layer._process_impression(item)

        imps = await state.get_active_impressions()
        assert len(imps) == 1
        assert imps[0]["id"] == "existing"
        assert imps[0]["pressure"] > 0.3


class TestPressureCalculation:
    async def test_pressure_levels(self, state, config):
        layer = _make_layer(state, config)

        imp = Impression(emotional_charge=0.0)
        assert layer._calculate_pressure(imp, "low") == pytest.approx(0.15)
        assert layer._calculate_pressure(imp, "medium") == pytest.approx(0.3)
        assert layer._calculate_pressure(imp, "high") == pytest.approx(0.5)
        assert layer._calculate_pressure(imp, "critical") == pytest.approx(0.9)

    async def test_emotional_charge_adds_pressure(self, state, config):
        layer = _make_layer(state, config)

        imp = Impression(emotional_charge=0.5)
        base = layer._calculate_pressure(imp, "low")
        assert base > 0.15  # More than base because of charge

    async def test_initial_pressure_clamped(self, state, config):
        layer = _make_layer(state, config)

        # High urgency should still be clamped to initial_pressure_range
        item = {
            "type": "pattern",
            "content": "test",
            "urgency": "high",
            "emotional_charge": 1.0,
        }
        await layer._process_impression(item)
        imps = await state.get_active_impressions()
        assert imps[0]["pressure"] <= config.initial_pressure_range[1]


class TestDeepCycle:
    @patch("freudian_mind.layers.unconscious.anthropic.AsyncAnthropic")
    async def test_deep_cycle_processes_response(self, mock_cls, state, config):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create = AsyncMock(
            return_value=_mock_api_response(
                {
                    "impressions": [
                        {
                            "type": "pattern",
                            "content": "user is learning Python",
                            "emotional_charge": 0.2,
                            "source_conversation": "c1",
                            "urgency": "low",
                            "similarity_key": "python_learning",
                        }
                    ],
                    "tasks_to_delegate": [],
                }
            )
        )

        await state.create_conversation("c1")
        await state.log_turn("c1", 1, "user", "I'm learning Python")

        layer = _make_layer(state, config)
        layer.client = mock_client

        await layer._deep_cycle()

        imps = await state.get_active_impressions()
        assert len(imps) >= 1
