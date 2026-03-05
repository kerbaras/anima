"""Tests for the Preconscious Layer."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from anima.config import MindConfig
from anima.layers.preconscious import PreconsciousLayer
from anima.models import Impression, ImpressionType
from anima.state import SharedState
from anima.systems.defense import DefenseProfile
from anima.systems.idea_space import IdeaSpace


def _mock_defense_response(data: dict):
    mock = MagicMock()
    mock.content = [MagicMock(text=json.dumps(data))]
    return mock


class TestDefenseApplication:
    @patch("anima.layers.preconscious.anthropic.AsyncAnthropic")
    async def test_sublimation_promotes_as_tool(self, mock_cls, state, config):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create = AsyncMock(
            return_value=_mock_defense_response(
                {
                    "impression_id": "imp1",
                    "selected_defense": "sublimation",
                    "defense_level": "mature",
                    "action": "Transform into code formatting tool",
                    "promotion": {
                        "type": "tool",
                        "key": "code_formatter",
                        "content": {"name": "format_code", "description": "Formats code"},
                        "confidence": 0.9,
                    },
                    "reason": "Productive transformation",
                    "growth_aware": False,
                }
            )
        )

        idea_space = IdeaSpace(state)
        profile = DefenseProfile()
        layer = PreconsciousLayer(state, idea_space, profile, config)
        layer.client = mock_client

        # Store an impression
        imp = Impression(
            id="imp1",
            type=ImpressionType.SKILL,
            content="user wants code formatting",
            pressure=0.8,
            emotional_charge=0.3,
            embedding=idea_space.embed("code formatting"),
        )
        await state.store_impression(imp)

        candidate = (await state.get_active_impressions())[0]
        await layer._evaluate_candidate(candidate)

        # Should have created a tool promotion
        promos = await state.get_active_promotions(type_filter="tool")
        assert len(promos) >= 1

        # Should have marked the impression as promoted
        active = await state.get_active_impressions()
        assert len(active) == 0

    @patch("anima.layers.preconscious.anthropic.AsyncAnthropic")
    async def test_repression_marks_repressed(self, mock_cls, state, config):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create = AsyncMock(
            return_value=_mock_defense_response(
                {
                    "impression_id": "imp1",
                    "selected_defense": "repression",
                    "defense_level": "neurotic",
                    "action": "Push back to buffer",
                    "promotion": None,
                    "reason": "Not ready to surface",
                    "growth_aware": False,
                }
            )
        )

        idea_space = IdeaSpace(state)
        profile = DefenseProfile()
        layer = PreconsciousLayer(state, idea_space, profile, config)
        layer.client = mock_client

        imp = Impression(id="imp1", content="test", pressure=0.8)
        await state.store_impression(imp)

        candidate = (await state.get_active_impressions())[0]
        await layer._evaluate_candidate(candidate)

        active = await state.get_active_impressions()
        assert len(active) == 0  # Repressed = filtered out

    @patch("anima.layers.preconscious.anthropic.AsyncAnthropic")
    async def test_rationalization_reduces_pressure(self, mock_cls, state, config):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create = AsyncMock(
            return_value=_mock_defense_response(
                {
                    "impression_id": "imp1",
                    "selected_defense": "rationalization",
                    "defense_level": "neurotic",
                    "action": "Topic is ambiguous",
                    "promotion": None,
                    "reason": "Justified dismissal",
                    "growth_aware": False,
                }
            )
        )

        idea_space = IdeaSpace(state)
        profile = DefenseProfile()
        layer = PreconsciousLayer(state, idea_space, profile, config)
        layer.client = mock_client

        imp = Impression(id="imp1", content="test", pressure=0.8)
        await state.store_impression(imp)

        candidate = (await state.get_active_impressions())[0]
        await layer._evaluate_candidate(candidate)

        active = await state.get_active_impressions()
        assert len(active) == 1
        assert active[0]["pressure"] < 0.8  # Reduced by 30%

    @patch("anima.layers.preconscious.anthropic.AsyncAnthropic")
    async def test_intellectualization_stores_as_memory(self, mock_cls, state, config):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create = AsyncMock(
            return_value=_mock_defense_response(
                {
                    "impression_id": "imp1",
                    "selected_defense": "intellectualization",
                    "defense_level": "neurotic",
                    "action": "Store as neutral fact",
                    "promotion": None,
                    "reason": "Acknowledge without urgency",
                    "growth_aware": False,
                }
            )
        )

        idea_space = IdeaSpace(state)
        profile = DefenseProfile()
        layer = PreconsciousLayer(state, idea_space, profile, config)
        layer.client = mock_client

        imp = Impression(id="imp1", content="user is frustrated", pressure=0.8)
        await state.store_impression(imp)

        candidate = (await state.get_active_impressions())[0]
        await layer._evaluate_candidate(candidate)

        # Should have created a memory promotion
        memories = await state.get_active_promotions(type_filter="memory")
        assert len(memories) >= 1

    @patch("anima.layers.preconscious.anthropic.AsyncAnthropic")
    async def test_denial_drops_impression(self, mock_cls, state, config):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create = AsyncMock(
            return_value=_mock_defense_response(
                {
                    "impression_id": "imp1",
                    "selected_defense": "denial",
                    "defense_level": "pathological",
                    "action": "Completely ignore",
                    "promotion": None,
                    "reason": "Denial",
                    "growth_aware": False,
                }
            )
        )

        idea_space = IdeaSpace(state)
        profile = DefenseProfile()
        layer = PreconsciousLayer(state, idea_space, profile, config)
        layer.client = mock_client

        imp = Impression(id="imp1", content="test", pressure=0.8)
        await state.store_impression(imp)

        candidate = (await state.get_active_impressions())[0]
        await layer._evaluate_candidate(candidate)

        active = await state.get_active_impressions()
        assert len(active) == 0  # Denied = dropped

    @patch("anima.layers.preconscious.anthropic.AsyncAnthropic")
    async def test_defense_event_logged(self, mock_cls, state, config):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client

        mock_client.messages.create = AsyncMock(
            return_value=_mock_defense_response(
                {
                    "impression_id": "imp1",
                    "selected_defense": "humor",
                    "defense_level": "mature",
                    "action": "Acknowledge with lightness",
                    "promotion": {
                        "key": "humor_imp1",
                        "content": {"instruction": "Be self-aware about this"},
                        "confidence": 0.7,
                    },
                    "reason": "Humor is appropriate",
                    "growth_aware": True,
                }
            )
        )

        idea_space = IdeaSpace(state)
        profile = DefenseProfile()
        layer = PreconsciousLayer(state, idea_space, profile, config)
        layer.client = mock_client

        imp = Impression(id="imp1", content="test", pressure=0.8)
        await state.store_impression(imp)

        candidate = (await state.get_active_impressions())[0]
        await layer._evaluate_candidate(candidate)

        events = await state.get_recent_defense_events()
        assert len(events) >= 1
        assert events[0]["defense"] == "humor"


class TestConvergence:
    async def test_convergence_boosts_pressure(self, state, config):
        idea_space = IdeaSpace(state)
        profile = DefenseProfile()
        layer = PreconsciousLayer(state, idea_space, profile, config)

        # Create cluster of identical embeddings
        emb = idea_space.embed("same topic")
        for i in range(4):
            imp = Impression(
                id=f"imp{i}",
                content=f"observation {i}",
                pressure=0.3,
                emotional_charge=0.5,
                embedding=emb,
            )
            await state.store_impression(imp)

        config.min_cluster_size = 3
        await layer._check_convergence()

        imps = await state.get_active_impressions()
        # At least some should have boosted pressure
        boosted = [i for i in imps if i["pressure"] > 0.3]
        assert len(boosted) > 0
