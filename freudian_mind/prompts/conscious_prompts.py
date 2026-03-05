"""Prompts for the Conscious (Haiku) layer."""

from __future__ import annotations

import json


BURST_INSTRUCTIONS = (
    "\nIMPORTANT: Keep each message SHORT — 1-3 sentences max. "
    "Talk like texting a friend. Be real, not robotic. "
    "If you have more to say, you'll get a chance to continue. "
    "When you're done, say [DONE] on its own line."
)

CONTINUATION_PROMPT = (
    "Continue your thought if you have more to say. "
    "Keep it to one short paragraph. "
    "If you're done, respond with exactly: [DONE]"
)


def build_system_prompt(
    base_personality: str,
    promotions: list[dict],
    interrupts: list[dict],
    bridge_context: str = "",
) -> str:
    """Assemble the conscious layer's system prompt from promotions and interrupts."""
    parts = [base_personality]

    # Cross-session bridge context (background knowledge, not conversation)
    if bridge_context:
        parts.append("\n--- Previous session context (background knowledge) ---")
        parts.append(bridge_context)
        parts.append("--- End previous session context ---")

    # Memory promotions
    memories = [p for p in promotions if p.get("type") == "memory"]
    if memories:
        parts.append("\nThings you know:")
        for m in memories:
            content = json.loads(m["content"]) if isinstance(m["content"], str) else m["content"]
            parts.append(f"- {content.get('fact', str(content))}")

    # Directive promotions
    directives = [p for p in promotions if p.get("type") == "directive"]
    if directives:
        parts.append("\nBehavioral guidelines:")
        for d in directives:
            content = json.loads(d["content"]) if isinstance(d["content"], str) else d["content"]
            parts.append(f"- {content.get('instruction', str(content))}")

    # Interrupts (intrusive thoughts)
    if interrupts:
        parts.append("\n⚡ IMPORTANT — address this in your next response:")
        for intr in interrupts:
            parts.append(f"- {intr['content']}")

    parts.append(BURST_INSTRUCTIONS)
    return "\n".join(parts)
