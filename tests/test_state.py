"""Tests for SharedState persistence layer."""

from __future__ import annotations

import pytest

from anima.models import (
    Impression,
    ImpressionType,
    OutcomeSignal,
    ResponseOutcome,
)
from anima.state import SharedState


class TestConversations:
    async def test_create_conversation(self, state):
        conv_id = await state.create_conversation()
        assert len(conv_id) > 0

    async def test_create_conversation_with_id(self, state):
        conv_id = await state.create_conversation("test-conv")
        assert conv_id == "test-conv"

    async def test_create_duplicate_ignored(self, state):
        await state.create_conversation("dup")
        await state.create_conversation("dup")  # Should not raise


class TestTurns:
    async def test_log_and_get_turns(self, state):
        await state.create_conversation("c1")
        await state.log_turn("c1", 1, "user", "hello")
        await state.log_turn("c1", 1, "assistant", "hi there")
        log = await state.get_conversation_log("c1")
        assert len(log) == 2
        assert log[0]["role"] == "user"
        assert log[1]["role"] == "assistant"

    async def test_turn_count(self, state):
        await state.create_conversation("c1")
        assert await state.get_turn_count("c1") == 0
        await state.log_turn("c1", 1, "user", "hello")
        assert await state.get_turn_count("c1") == 1
        await state.log_turn("c1", 2, "user", "more")
        assert await state.get_turn_count("c1") == 2

    async def test_get_recent_turns(self, state):
        await state.create_conversation("c1")
        await state.log_turn("c1", 1, "user", "hello")
        turns = await state.get_all_recent_turns(since_seconds=10)
        assert len(turns) >= 1

    async def test_get_last_assistant_message(self, state):
        await state.create_conversation("c1")
        await state.log_turn("c1", 1, "user", "hello")
        await state.log_turn("c1", 1, "assistant", "response A")
        await state.log_turn("c1", 1, "assistant", "response B", burst_index=1)
        msg = await state.get_last_assistant_message("c1")
        assert msg == "response B"

    async def test_conversation_window(self, state):
        await state.create_conversation("c1")
        for i in range(10):
            await state.log_turn("c1", i + 1, "user", f"msg {i}")
        log = await state.get_conversation_log("c1", last_n=5)
        assert len(log) == 5


class TestImpressions:
    async def test_store_and_get(self, state):
        imp = Impression(
            id="imp1",
            type=ImpressionType.PATTERN,
            content="test pattern",
            pressure=0.5,
        )
        await state.store_impression(imp)
        active = await state.get_active_impressions()
        assert len(active) == 1
        assert active[0]["id"] == "imp1"
        assert active[0]["pressure"] == 0.5

    async def test_reinforce_impression(self, state):
        imp = Impression(id="imp1", pressure=0.3)
        await state.store_impression(imp)
        await state.reinforce_impression("imp1", 0.2)
        active = await state.get_active_impressions()
        assert active[0]["pressure"] == pytest.approx(0.5, abs=0.01)
        assert active[0]["times_reinforced"] == 1

    async def test_pressure_capped_at_1(self, state):
        imp = Impression(id="imp1", pressure=0.9)
        await state.store_impression(imp)
        await state.reinforce_impression("imp1", 0.5)
        active = await state.get_active_impressions()
        assert active[0]["pressure"] <= 1.0

    async def test_mark_promoted(self, state):
        imp = Impression(id="imp1")
        await state.store_impression(imp)
        await state.mark_promoted("imp1")
        active = await state.get_active_impressions()
        assert len(active) == 0  # Promoted impressions are filtered out

    async def test_mark_repressed(self, state):
        imp = Impression(id="imp1")
        await state.store_impression(imp)
        await state.mark_repressed("imp1")
        active = await state.get_active_impressions()
        assert len(active) == 0

    async def test_update_pressure(self, state):
        imp = Impression(id="imp1", pressure=0.8)
        await state.store_impression(imp)
        await state.update_impression_pressure("imp1", 0.3)
        active = await state.get_active_impressions()
        assert active[0]["pressure"] == pytest.approx(0.3)

    async def test_get_with_embeddings(self, state):
        imp1 = Impression(id="imp1", embedding=[0.1, 0.2, 0.3])
        imp2 = Impression(id="imp2", embedding=[])
        await state.store_impression(imp1)
        await state.store_impression(imp2)
        with_emb = await state.get_all_impressions_with_embeddings()
        assert len(with_emb) == 1
        assert with_emb[0]["id"] == "imp1"


class TestPromotions:
    async def test_upsert_new(self, state):
        await state.upsert_promotion(
            ptype="directive",
            key="be_nice",
            content={"instruction": "be kind"},
        )
        promos = await state.get_active_promotions()
        assert len(promos) == 1
        assert promos[0]["key"] == "be_nice"

    async def test_upsert_existing(self, state):
        await state.upsert_promotion("directive", "k1", {"v": "old"})
        await state.upsert_promotion("directive", "k1", {"v": "new"})
        promos = await state.get_active_promotions()
        assert len(promos) == 1

    async def test_filter_by_type(self, state):
        await state.upsert_promotion("directive", "d1", {"a": 1})
        await state.upsert_promotion("memory", "m1", {"b": 2})
        directives = await state.get_active_promotions(type_filter="directive")
        assert len(directives) == 1
        assert directives[0]["type"] == "directive"


class TestInterrupts:
    async def test_create_and_consume(self, state):
        await state.create_conversation("c1")
        iid = await state.create_interrupt("c1", "fix this!", urgency=0.9)
        interrupts = await state.consume_interrupts("c1")
        assert len(interrupts) == 1
        assert interrupts[0]["content"] == "fix this!"
        # Should be consumed now
        again = await state.consume_interrupts("c1")
        assert len(again) == 0

    async def test_no_interrupts_for_other_conv(self, state):
        await state.create_interrupt("c1", "for c1 only")
        interrupts = await state.consume_interrupts("c2")
        assert len(interrupts) == 0


class TestTasks:
    async def test_create_and_get_pending(self, state):
        tid = await state.create_task("c1", "research X", "prompt for X")
        pending = await state.get_pending_tasks()
        assert len(pending) == 1
        assert pending[0]["description"] == "research X"

    async def test_update_task(self, state):
        tid = await state.create_task("c1", "task", "prompt")
        await state.update_task(tid, "complete", "result here")
        pending = await state.get_pending_tasks()
        assert len(pending) == 0


class TestDefenseEvents:
    async def test_log_and_get(self, state):
        await state.log_defense_event(
            impression_id="imp1",
            defense="sublimation",
            level=4,
            action_taken="promoted as tool",
            led_to_positive=True,
        )
        events = await state.get_recent_defense_events()
        assert len(events) == 1
        assert events[0]["defense"] == "sublimation"


class TestOutcomes:
    async def test_log_and_get(self, state):
        outcome = ResponseOutcome(
            conversation_id="c1",
            turn_number=1,
            signal=OutcomeSignal.CORRECTION,
            brief_reason="user corrected a fact",
        )
        await state.log_outcome(outcome)
        recent = await state.get_recent_outcomes()
        assert len(recent) == 1
        assert recent[0]["signal"] == "correction"

    async def test_pending_outcomes(self, state):
        outcome = ResponseOutcome(
            conversation_id="c1",
            signal=OutcomeSignal.POSITIVE,
        )
        await state.log_outcome(outcome)
        pending = await state.get_pending_outcomes()
        assert len(pending) == 1


class TestGrowthEvents:
    async def test_log(self, state):
        await state.log_growth_event(
            mechanism="defense_maturation",
            description="upgraded denial to suppression",
            target_defense="denial",
        )
        # No get method needed — just verify no error


class TestRepetitionPatterns:
    async def test_upsert_and_get(self, state):
        await state.upsert_repetition_pattern(
            pattern_id="p1",
            pattern_type="correction_loop",
            description="keeps correcting math",
            severity="HIGH",
        )
        patterns = await state.get_active_patterns()
        assert len(patterns) == 1
        assert patterns[0]["severity"] == "HIGH"

    async def test_resolve_pattern(self, state):
        await state.upsert_repetition_pattern(
            pattern_id="p1",
            pattern_type="correction_loop",
            description="test",
        )
        await state.resolve_pattern("p1")
        active = await state.get_active_patterns()
        assert len(active) == 0

    async def test_upsert_updates_existing(self, state):
        await state.upsert_repetition_pattern(
            "p1", "correction_loop", "test", occurrence_count=1
        )
        await state.upsert_repetition_pattern(
            "p1", "correction_loop", "test", occurrence_count=5, severity="CRITICAL"
        )
        patterns = await state.get_active_patterns()
        assert len(patterns) == 1
        assert patterns[0]["occurrence_count"] == 5
        assert patterns[0]["severity"] == "CRITICAL"
