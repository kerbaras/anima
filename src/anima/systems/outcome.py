"""Outcome Classifier — classifies user reactions to every exchange."""

from __future__ import annotations

import json

from ..llm import complete
from ..models import OutcomeSignal, ResponseOutcome
from ..prompts.classifier_prompts import OUTCOME_CLASSIFIER_PROMPT


class OutcomeClassifier:
    """Haiku-based classifier that runs on every user message."""

    def __init__(self, model: str = "claude-haiku-4-5-20241022"):
        self.model = model

    async def classify(
        self,
        assistant_message: str,
        user_response: str,
        conversation_id: str = "",
        turn_number: int = 0,
    ) -> ResponseOutcome:
        prompt = (
            f"Assistant said: {assistant_message[:300]}\n\n"
            f"User then said: {user_response[:300]}\n\n"
            "Classify the user's reaction."
        )

        try:
            response = await complete(
                model=self.model,
                max_tokens=200,
                system=OUTCOME_CLASSIFIER_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )

            raw = response.text.strip()
            cleaned = raw.removeprefix("```json").removesuffix("```").strip()
            data = json.loads(cleaned)

            return ResponseOutcome(
                conversation_id=conversation_id,
                turn_number=turn_number,
                signal=OutcomeSignal(data.get("signal", "neutral")),
                user_had_to_repeat=data.get("user_had_to_repeat", False),
                haiku_contradicted_itself=data.get("haiku_contradicted_itself", False),
                response_was_useful=data.get("response_was_useful", True),
                brief_reason=data.get("brief_reason", ""),
            )
        except Exception:
            return ResponseOutcome(
                conversation_id=conversation_id,
                turn_number=turn_number,
                signal=OutcomeSignal.NEUTRAL,
            )
