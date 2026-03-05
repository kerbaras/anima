"""Shared fixtures for all tests."""

from __future__ import annotations

import json

import pytest

from anima.config import MindConfig
from anima.llm import LLMResponse
from anima.mind import FreudianMind
from anima.state import SharedState


@pytest.fixture
def config():
    return MindConfig(db_path=":memory:")


@pytest.fixture
async def state(config):
    s = SharedState(config.db_path)
    await s.initialize()
    yield s
    await s.close()


@pytest.fixture
async def mind():
    """A fully wired FreudianMind with in-memory DB, initialized but no background loops."""
    config = MindConfig(db_path=":memory:")
    m = FreudianMind(config)
    await m.state.initialize()
    yield m
    await m.state.close()


def make_llm_response(text: str) -> LLMResponse:
    """Create a simple LLMResponse with given text."""
    return LLMResponse(text=text, finish_reason="stop")


def make_classifier_response(
    signal: str = "neutral",
    user_had_to_repeat: bool = False,
    response_was_useful: bool = True,
) -> LLMResponse:
    """Create a mock outcome classifier JSON response."""
    data = {
        "signal": signal,
        "user_had_to_repeat": user_had_to_repeat,
        "haiku_contradicted_itself": False,
        "response_was_useful": response_was_useful,
        "brief_reason": "test",
    }
    return LLMResponse(text=json.dumps(data), finish_reason="stop")


def make_defense_response(
    defense: str = "sublimation",
    action: str = "promote",
    promotion: dict | None = None,
) -> LLMResponse:
    """Create a mock defense selector JSON response."""
    data = {
        "selected_defense": defense,
        "action": action,
        "reasoning": "test",
    }
    if promotion is not None:
        data["promotion"] = promotion
    return LLMResponse(text=json.dumps(data), finish_reason="stop")
