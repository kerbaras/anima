"""Conscious Layer (Haiku) — fast, stateless, message bursts."""

from __future__ import annotations

import json

from ..config import MindConfig
from ..llm import complete
from ..models import MessageBurst
from ..prompts.conscious_prompts import (
    CONTINUATION_PROMPT,
    build_system_prompt,
)
from ..state import SharedState


class ConsciousLayer:
    """
    Haiku: fast, stateless, sends message bursts not monoliths.

    - Reads system prompt from promotions (personality is emergent)
    - Checks for interrupts (intrusive thoughts) before responding
    - Sends 1-4 short messages per turn
    """

    def __init__(self, state: SharedState, config: MindConfig):
        self.state = state
        self.config = config

    async def respond(self, conv_id: str, user_message: str, bridge_context: str = "") -> MessageBurst:
        # Consume interrupts
        interrupts = await self.state.consume_interrupts(conv_id)

        # Get promotions for system prompt
        promotions = await self.state.get_active_promotions()

        # Build system prompt
        system_prompt = build_system_prompt(
            self.config.base_personality,
            promotions,
            interrupts,
            bridge_context=bridge_context,
        )

        # Build conversation messages
        messages = await self._build_messages(conv_id)
        messages.append({"role": "user", "content": user_message})

        # Generate burst
        burst_messages = await self._generate_burst(system_prompt, messages, conv_id)

        # Build tools if any tool promotions exist
        # (tools handled in system prompt for now)

        return MessageBurst(
            messages=burst_messages,
            conversation_id=conv_id,
            interrupts_applied=[i["id"] for i in interrupts],
        )

    async def _generate_burst(
        self,
        system_prompt: str,
        messages: list[dict],
        conv_id: str,
    ) -> list[str]:
        burst = []
        conversation = list(messages)

        for i in range(self.config.max_burst_messages):
            if i > 0:
                # Check for mid-burst interrupts
                new_interrupts = await self.state.consume_interrupts(conv_id)
                if new_interrupts:
                    interrupt_text = " ".join(
                        intr["content"] for intr in new_interrupts
                    )
                    conversation.append(
                        {
                            "role": "user",
                            "content": f"[System note: {interrupt_text}]\n{CONTINUATION_PROMPT}",
                        }
                    )
                else:
                    conversation.append(
                        {"role": "user", "content": CONTINUATION_PROMPT}
                    )

            response = await complete(
                model=self.config.conscious_model,
                max_tokens=self.config.burst_max_tokens,
                system=system_prompt,
                messages=conversation,
            )

            text = response.text.strip()

            if "[DONE]" in text or text == "":
                break

            text = text.replace("[DONE]", "").strip()
            if not text:
                break

            burst.append(text)
            conversation.append({"role": "assistant", "content": text})

        return burst if burst else ["I'm here."]

    async def _build_messages(self, conv_id: str) -> list[dict]:
        log = await self.state.get_conversation_log(
            conv_id, last_n=self.config.conversation_window
        )
        return [{"role": t["role"], "content": t["content"]} for t in log]

    def _build_tools(self, promotions: list[dict]) -> list[dict]:
        tool_promotions = [p for p in promotions if p.get("type") == "tool"]
        tools = []
        for p in tool_promotions:
            content = (
                json.loads(p["content"])
                if isinstance(p["content"], str)
                else p["content"]
            )
            tools.append(
                {
                    "name": content.get("name", p["key"]),
                    "description": content.get("description", ""),
                    "input_schema": content.get(
                        "parameters",
                        {"type": "object", "properties": {}, "required": []},
                    ),
                }
            )
        return tools
