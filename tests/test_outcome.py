"""Tests for the Outcome Classifier."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from freudian_mind.models import OutcomeSignal
from freudian_mind.systems.outcome import OutcomeClassifier


def _mock_api_response(data: dict):
    mock = MagicMock()
    mock.content = [MagicMock(text=json.dumps(data))]
    return mock


class TestOutcomeClassifier:
    @patch("freudian_mind.systems.outcome.anthropic.AsyncAnthropic")
    async def test_classifies_positive(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(
            return_value=_mock_api_response(
                {
                    "signal": "positive",
                    "user_had_to_repeat": False,
                    "haiku_contradicted_itself": False,
                    "response_was_useful": True,
                    "brief_reason": "User continued productively",
                }
            )
        )

        classifier = OutcomeClassifier()
        classifier.client = mock_client
        outcome = await classifier.classify("I can help with that", "Great, thanks!")
        assert outcome.signal == OutcomeSignal.POSITIVE
        assert outcome.response_was_useful is True

    @patch("freudian_mind.systems.outcome.anthropic.AsyncAnthropic")
    async def test_classifies_correction(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(
            return_value=_mock_api_response(
                {
                    "signal": "correction",
                    "user_had_to_repeat": True,
                    "haiku_contradicted_itself": False,
                    "response_was_useful": False,
                    "brief_reason": "User corrected the answer",
                }
            )
        )

        classifier = OutcomeClassifier()
        classifier.client = mock_client
        outcome = await classifier.classify("2+2=5", "No, 2+2=4")
        assert outcome.signal == OutcomeSignal.CORRECTION
        assert outcome.user_had_to_repeat is True

    @patch("freudian_mind.systems.outcome.anthropic.AsyncAnthropic")
    async def test_handles_markdown_fence(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(
            return_value=MagicMock(
                content=[
                    MagicMock(
                        text='```json\n{"signal": "delight", "user_had_to_repeat": false, '
                        '"haiku_contradicted_itself": false, "response_was_useful": true, '
                        '"brief_reason": "happy"}\n```'
                    )
                ]
            )
        )

        classifier = OutcomeClassifier()
        classifier.client = mock_client
        outcome = await classifier.classify("answer", "wow amazing!")
        assert outcome.signal == OutcomeSignal.DELIGHT

    @patch("freudian_mind.systems.outcome.anthropic.AsyncAnthropic")
    async def test_falls_back_to_neutral(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(side_effect=Exception("API error"))

        classifier = OutcomeClassifier()
        classifier.client = mock_client
        outcome = await classifier.classify("msg", "response")
        assert outcome.signal == OutcomeSignal.NEUTRAL

    @patch("freudian_mind.systems.outcome.anthropic.AsyncAnthropic")
    async def test_stores_conversation_metadata(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        mock_client.messages.create = AsyncMock(
            return_value=_mock_api_response({"signal": "neutral"})
        )

        classifier = OutcomeClassifier()
        classifier.client = mock_client
        outcome = await classifier.classify(
            "msg", "resp", conversation_id="c1", turn_number=3
        )
        assert outcome.conversation_id == "c1"
        assert outcome.turn_number == 3
