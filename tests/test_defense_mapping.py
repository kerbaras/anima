"""Behavioral tests for defense mechanism action mapping.

Tests that each of the 15 defense mechanisms produces the correct side effect
when applied by the preconscious layer.  Uses the real DB (in-memory) and
real state methods; only the LLM defense selector call is mocked.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from anima.config import MindConfig
from anima.layers.preconscious import PreconsciousLayer
from anima.models import DefenseMechanism, ImpressionType
from anima.systems.defense import DefenseProfile
from anima.systems.idea_space import IdeaSpace

from conftest import make_defense_response


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture
async def preconscious_env(state):
    """Set up a PreconsciousLayer with real state, idea_space, and defense profile."""
    config = MindConfig(db_path=":memory:")
    idea_space = IdeaSpace(state)
    defense_profile = DefenseProfile()
    layer = PreconsciousLayer(state, idea_space, defense_profile, config)
    return layer, state, idea_space, defense_profile


def _make_candidate(imp_id: str = "test-imp", pressure: float = 0.8, content: str = "test") -> dict:
    return {
        "id": imp_id,
        "type": "pattern",
        "content": content,
        "pressure": pressure,
        "emotional_charge": 0.3,
        "times_reinforced": 2,
        "embedding": json.dumps([0.5] * 32),
        "payload": "{}",
    }


# ── Scenario 12: Parameterized defense action mapping ──────────────────────


class TestDefenseActionMapping:
    """For each defense mechanism, verify the correct side effect occurs."""

    @pytest.mark.parametrize(
        "defense,expected_promotion_type",
        [
            ("sublimation", "tool"),
            ("anticipation", "directive"),
            ("humor", "directive"),
            ("altruism", "directive"),
            ("reaction_formation", "directive"),
        ],
    )
    async def test_mature_defenses_create_promotions(
        self, preconscious_env, defense, expected_promotion_type
    ):
        layer, state, _, _ = preconscious_env
        candidate = _make_candidate()

        # Store the impression in DB first so state operations work
        from anima.models import Impression

        imp = Impression(id=candidate["id"], type=ImpressionType.PATTERN, content="test", pressure=0.8)
        await state.store_impression(imp)

        decision = {
            "selected_defense": defense,
            "action": "promote this",
            "promotion": {
                "key": f"{defense}_test",
                "content": {"instruction": "do something useful"},
                "confidence": 0.8,
            },
        }

        await layer._apply_defense_decision(candidate, decision)

        # Verify promotion was created
        promotions = await state.get_active_promotions()
        matching = [p for p in promotions if p["key"] == f"{defense}_test"]
        assert len(matching) == 1
        assert matching[0]["type"] == expected_promotion_type

        # Verify impression was marked as promoted
        active = await state.get_active_impressions()
        imp_ids = [i["id"] for i in active]
        assert candidate["id"] not in imp_ids, f"{defense} should mark impression as promoted"

    @pytest.mark.parametrize(
        "defense",
        ["repression", "denial", "distortion"],
    )
    async def test_blocking_defenses_repress_impression(
        self, preconscious_env, defense
    ):
        layer, state, _, _ = preconscious_env
        candidate = _make_candidate(imp_id=f"imp-{defense}")

        from anima.models import Impression

        imp = Impression(id=candidate["id"], type=ImpressionType.PATTERN, content="blocked", pressure=0.8)
        await state.store_impression(imp)

        decision = {"selected_defense": defense, "action": "block"}
        await layer._apply_defense_decision(candidate, decision)

        active = await state.get_active_impressions()
        imp_ids = [i["id"] for i in active]
        assert candidate["id"] not in imp_ids, f"{defense} should repress the impression"

    async def test_rationalization_reduces_pressure(self, preconscious_env):
        layer, state, _, _ = preconscious_env
        candidate = _make_candidate(pressure=1.0)

        from anima.models import Impression

        imp = Impression(id=candidate["id"], type=ImpressionType.PATTERN, content="test", pressure=1.0)
        await state.store_impression(imp)

        decision = {"selected_defense": "rationalization", "action": "justify"}
        await layer._apply_defense_decision(candidate, decision)

        # Pressure should be reduced by 30%
        active = await state.get_active_impressions()
        updated = [i for i in active if i["id"] == candidate["id"]]
        assert len(updated) == 1
        assert abs(updated[0]["pressure"] - 0.7) < 0.01

    async def test_intellectualization_creates_memory_promotion(self, preconscious_env):
        layer, state, _, _ = preconscious_env
        candidate = _make_candidate(content="Python GIL prevents true parallelism")

        from anima.models import Impression

        imp = Impression(
            id=candidate["id"], type=ImpressionType.PATTERN,
            content=candidate["content"], pressure=0.8,
        )
        await state.store_impression(imp)

        decision = {"selected_defense": "intellectualization", "action": "store as fact"}
        await layer._apply_defense_decision(candidate, decision)

        promotions = await state.get_active_promotions()
        memory_proms = [p for p in promotions if p["type"] == "memory"]
        assert len(memory_proms) == 1
        content = json.loads(memory_proms[0]["content"])
        assert content["category"] == "context"

    async def test_displacement_transfers_pressure(self, preconscious_env):
        layer, state, idea_space, _ = preconscious_env

        from anima.models import Impression

        # Create source impression with high charge
        source = Impression(
            id="displace-src",
            type=ImpressionType.PATTERN,
            content="anxiety about deadlines",
            pressure=0.9,
            emotional_charge=0.8,
        )
        source.embedding = idea_space.embed(source.content)
        await state.store_impression(source)

        # Create neighbor with lower charge and similar embedding
        neighbor = Impression(
            id="displace-tgt",
            type=ImpressionType.PATTERN,
            content="anxiety about deadlines tomorrow",
            pressure=0.3,
            emotional_charge=0.2,
        )
        neighbor.embedding = idea_space.embed(neighbor.content)
        await state.store_impression(neighbor)

        candidate = {
            "id": source.id,
            "type": "pattern",
            "content": source.content,
            "pressure": source.pressure,
            "emotional_charge": source.emotional_charge,
            "embedding": json.dumps(source.embedding),
        }

        decision = {"selected_defense": "displacement", "action": "transfer"}
        await layer._apply_defense_decision(candidate, decision)

        # Source pressure should be reduced, neighbor should be boosted
        all_imps = await state.get_active_impressions()
        src_after = [i for i in all_imps if i["id"] == "displace-src"]
        tgt_after = [i for i in all_imps if i["id"] == "displace-tgt"]

        if src_after and tgt_after:
            # If displacement found a similar enough neighbor
            assert src_after[0]["pressure"] < source.pressure
            assert tgt_after[0]["pressure"] > neighbor.pressure

    @pytest.mark.parametrize(
        "defense",
        ["suppression", "projection", "splitting", "passive_aggression"],
    )
    async def test_passthrough_defenses_preserve_state(
        self, preconscious_env, defense
    ):
        """Passthrough defenses should not modify the impression."""
        layer, state, _, _ = preconscious_env
        candidate = _make_candidate(imp_id=f"pass-{defense}", pressure=0.85)

        from anima.models import Impression

        imp = Impression(
            id=candidate["id"], type=ImpressionType.PATTERN,
            content="test", pressure=0.85,
        )
        await state.store_impression(imp)

        decision = {"selected_defense": defense, "action": "pass"}
        await layer._apply_defense_decision(candidate, decision)

        # Impression should still be active with same pressure
        active = await state.get_active_impressions()
        found = [i for i in active if i["id"] == candidate["id"]]
        assert len(found) == 1
        assert abs(found[0]["pressure"] - 0.85) < 0.01

    async def test_defense_event_logged_for_every_decision(self, preconscious_env):
        """Every defense decision should create a defense_events record."""
        layer, state, _, _ = preconscious_env
        candidate = _make_candidate()

        from anima.models import Impression

        imp = Impression(id=candidate["id"], type=ImpressionType.PATTERN, content="test", pressure=0.8)
        await state.store_impression(imp)

        decision = {"selected_defense": "sublimation", "action": "promote", "promotion": {
            "key": "event_log_test", "content": {"instruction": "test"}, "confidence": 0.8,
        }}
        await layer._apply_defense_decision(candidate, decision)

        events = await state.get_recent_defense_events(10)
        assert len(events) >= 1
        assert events[0]["defense"] == "sublimation"
        assert events[0]["impression_id"] == candidate["id"]


# ── Scenario 13: Growth-aware defense context ────────────────────────────


class TestGrowthAwareDefenseContext:
    """When active neurotic patterns exist, the defense evaluation prompt
    sent to the LLM must include them so Sonnet can make informed decisions."""

    @patch("anima.layers.preconscious.complete")
    async def test_active_patterns_included_in_context(self, mock_complete, preconscious_env):
        layer, state, _, _ = preconscious_env

        # Create an active neurotic pattern in the DB
        await state.upsert_repetition_pattern(
            pattern_id="pat-1",
            pattern_type="correction_loop",
            description="User keeps correcting Python answers",
            maintaining_defense="repression",
            severity="HIGH",
        )

        # Store an impression to evaluate
        from anima.models import Impression

        imp = Impression(
            id="eval-imp",
            type=ImpressionType.CORRECTION,
            content="wrong answer about list comprehensions",
            pressure=0.85,
        )
        await state.store_impression(imp)

        candidate = {
            "id": "eval-imp",
            "type": "correction",
            "content": "wrong answer about list comprehensions",
            "pressure": 0.85,
            "emotional_charge": -0.4,
            "times_reinforced": 1,
        }

        mock_complete.return_value = make_defense_response(
            "sublimation",
            promotion={"key": "ctx_test", "content": {"instruction": "fix"}, "confidence": 0.8},
        )

        await layer._evaluate_candidate(candidate)

        # Verify the LLM was called with context containing the active pattern
        assert mock_complete.called
        call_kwargs = mock_complete.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages", [])
        user_content = messages[0]["content"] if messages else ""

        assert "ACTIVE NEUROTIC PATTERNS" in user_content
        assert "correction_loop" in user_content or "User keeps correcting" in user_content

    @patch("anima.layers.preconscious.complete")
    async def test_health_report_included_in_context(self, mock_complete, preconscious_env):
        """Defense evaluation should include the defense profile health report."""
        layer, state, _, defense_profile = preconscious_env

        # Record some defense usage so health report has data
        defense_profile.record_defense_use(DefenseMechanism.DENIAL, False)
        defense_profile.record_defense_use(DefenseMechanism.DENIAL, False)

        from anima.models import Impression

        imp = Impression(id="health-imp", type=ImpressionType.PATTERN, content="test", pressure=0.8)
        await state.store_impression(imp)

        candidate = {
            "id": "health-imp", "type": "pattern", "content": "test",
            "pressure": 0.8, "emotional_charge": 0.0, "times_reinforced": 0,
        }

        mock_complete.return_value = make_defense_response("suppression")
        await layer._evaluate_candidate(candidate)

        call_kwargs = mock_complete.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs[1].get("messages", [])
        user_content = messages[0]["content"] if messages else ""

        assert "SYSTEM HEALTH" in user_content
        assert "maturity_score" in user_content
