"""Thin LiteLLM wrapper — single entry point for all LLM calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import litellm


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""

    text: str
    tool_calls: list[dict] | None = None      # normalized: [{id, name, input}]
    raw_tool_calls: list | None = None         # original objects for message history
    finish_reason: str = "stop"


def _resolve_model(model: str) -> str:
    """Auto-prefix Claude model IDs with 'anthropic/' for LiteLLM routing."""
    if "/" in model:
        return model
    if model.startswith("claude-"):
        return f"anthropic/{model}"
    return model


def _convert_tools(tools: list[dict]) -> list[dict]:
    """Convert Anthropic-style tool defs to OpenAI-style for LiteLLM."""
    openai_tools = []
    for t in tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", t.get("parameters", {})),
            },
        })
    return openai_tools


def _parse_tool_calls(message: Any) -> tuple[list[dict] | None, list | None]:
    """Extract and normalize tool calls from an OpenAI-style message."""
    raw = getattr(message, "tool_calls", None)
    if not raw:
        return None, None

    normalized = []
    for tc in raw:
        args = tc.function.arguments
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {"raw": args}
        normalized.append({
            "id": tc.id,
            "name": tc.function.name,
            "input": args,
        })
    return normalized, raw


async def complete(
    *,
    model: str,
    messages: list[dict],
    system: str | None = None,
    max_tokens: int = 4000,
    tools: list[dict] | None = None,
) -> LLMResponse:
    """Call any LLM via LiteLLM and return a normalized response."""
    resolved = _resolve_model(model)

    full_messages = list(messages)
    if system:
        full_messages.insert(0, {"role": "system", "content": system})

    kwargs: dict[str, Any] = {
        "model": resolved,
        "messages": full_messages,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = _convert_tools(tools)

    response = await litellm.acompletion(**kwargs)

    choice = response.choices[0]
    message = choice.message
    text = message.content or ""
    tool_calls, raw_tool_calls = _parse_tool_calls(message)

    finish = choice.finish_reason or "stop"
    if finish == "tool_calls":
        finish = "tool_use"

    return LLMResponse(
        text=text,
        tool_calls=tool_calls,
        raw_tool_calls=raw_tool_calls,
        finish_reason=finish,
    )
