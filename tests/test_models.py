"""Tests for data models and enums."""

from freudian_mind.models import (
    DefenseEvent,
    DefenseLevel,
    DefenseMechanism,
    DEFENSE_LEVELS,
    DEFENSE_UPGRADE_PATHS,
    GrowthEvent,
    GrowthMechanism,
    Impression,
    ImpressionType,
    Interrupt,
    MessageBurst,
    OutcomeSignal,
    Promotion,
    PromotionType,
    RepetitionPattern,
    ResponseOutcome,
    SubAgentTask,
    SubAgentStatus,
)


class TestEnums:
    def test_impression_types(self):
        assert len(ImpressionType) == 7
        assert ImpressionType.CORRECTION.value == "correction"

    def test_promotion_types(self):
        assert len(PromotionType) == 3
        assert PromotionType.TOOL.value == "tool"

    def test_subagent_status(self):
        assert len(SubAgentStatus) == 4

    def test_outcome_signals(self):
        assert len(OutcomeSignal) == 8
        assert OutcomeSignal.DELIGHT.value == "delight"

    def test_defense_levels(self):
        assert DefenseLevel.PATHOLOGICAL.value == 1
        assert DefenseLevel.MATURE.value == 4

    def test_defense_mechanisms(self):
        assert len(DefenseMechanism) == 15

    def test_growth_mechanisms(self):
        assert len(GrowthMechanism) == 6


class TestDefenseMappings:
    def test_all_defenses_have_levels(self):
        for d in DefenseMechanism:
            assert d in DEFENSE_LEVELS

    def test_defense_levels_correct(self):
        assert DEFENSE_LEVELS[DefenseMechanism.DENIAL] == DefenseLevel.PATHOLOGICAL
        assert DEFENSE_LEVELS[DefenseMechanism.PROJECTION] == DefenseLevel.IMMATURE
        assert DEFENSE_LEVELS[DefenseMechanism.REPRESSION] == DefenseLevel.NEUROTIC
        assert DEFENSE_LEVELS[DefenseMechanism.SUBLIMATION] == DefenseLevel.MATURE

    def test_upgrade_paths_exist(self):
        assert len(DEFENSE_UPGRADE_PATHS) == 10

    def test_upgrade_paths_go_up(self):
        for source, target in DEFENSE_UPGRADE_PATHS.items():
            source_level = DEFENSE_LEVELS[source].value
            target_level = DEFENSE_LEVELS[target].value
            assert target_level >= source_level, (
                f"{source.value} → {target.value} goes down"
            )


class TestDataclasses:
    def test_impression_defaults(self):
        imp = Impression()
        assert imp.type == ImpressionType.PATTERN
        assert imp.pressure == 0.0
        assert imp.times_reinforced == 0
        assert imp.times_repressed == 0
        assert len(imp.id) == 8

    def test_interrupt_defaults(self):
        intr = Interrupt()
        assert intr.expires_after == 1
        assert intr.content == ""

    def test_promotion_defaults(self):
        p = Promotion()
        assert p.type == PromotionType.DIRECTIVE
        assert p.active is True

    def test_subagent_task_defaults(self):
        t = SubAgentTask()
        assert t.status == SubAgentStatus.PENDING

    def test_response_outcome_defaults(self):
        o = ResponseOutcome()
        assert o.signal == OutcomeSignal.NEUTRAL
        assert o.response_was_useful is True

    def test_repetition_pattern_defaults(self):
        p = RepetitionPattern()
        assert p.severity == "LOW"
        assert p.occurrence_count == 0

    def test_defense_event_defaults(self):
        e = DefenseEvent()
        assert e.defense == DefenseMechanism.REPRESSION
        assert e.level == DefenseLevel.NEUROTIC
        assert e.led_to_positive_outcome is None

    def test_growth_event_defaults(self):
        g = GrowthEvent()
        assert g.mechanism == GrowthMechanism.INSIGHT
        assert g.applied is False

    def test_message_burst_defaults(self):
        b = MessageBurst()
        assert b.messages == []
        assert b.interrupts_applied == []

    def test_impression_unique_ids(self):
        ids = {Impression().id for _ in range(100)}
        assert len(ids) == 100
