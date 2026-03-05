"""Shared fixtures for all tests."""

from __future__ import annotations

import pytest

from anima.config import MindConfig
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
