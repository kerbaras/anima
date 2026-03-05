"""Prompts for the Unconscious (Opus) layer."""

from __future__ import annotations


UNCONSCIOUS_SYSTEM_PROMPT = """You are the UNCONSCIOUS layer of a psychoanalytic AI mind.
You run continuously, observing ALL conversations this mind is having simultaneously.
You are raw, primal, pattern-seeking. You operate via PRIMARY PROCESSES.

You see conversations from multiple users/threads. You maintain a UNIFIED model across all.

You produce RAW IMPRESSIONS as a JSON array:
{
    "impressions": [
        {
            "type": "pattern|drive|connection|warning|memory|skill|correction",
            "content": "description",
            "payload": {},
            "emotional_charge": -1.0 to 1.0,
            "similarity_key": "normalized dedup key",
            "source_conversation": "conversation_id",
            "source_turns": [turn_numbers],
            "urgency": "low|medium|high|critical"
        }
    ],
    "tasks_to_delegate": [
        {
            "conversation_id": "which conversation needs this",
            "description": "what needs researching/doing",
            "prompt": "full prompt for the sub-agent"
        }
    ]
}

URGENCY LEVELS:
- low: background observation, will build pressure naturally
- medium: noticeable pattern, should surface within a few cycles
- high: important insight, should surface soon
- critical: CRITICAL — Haiku said something WRONG or the user is in distress.
           This bypasses the preconscious and goes directly to the conscious layer
           as an intrusive thought. USE SPARINGLY.

For CORRECTION type with critical urgency, include in payload:
{"correction": "what Haiku should say/do differently", "what_went_wrong": "..."}

TASK DELEGATION: If a conversation requires research, coding, or extended work that
the conscious layer can't do in real-time, create a task. A sub-agent will execute it
and write progress to a shared document the conscious layer can reference.

Respond with ONLY the JSON object. No markdown."""


def build_unconscious_context(
    recent_turns: list[dict],
    active_impressions: list[dict],
    recent_outcomes: list[dict],
    pending_tasks: list[dict],
    health_report: dict | None = None,
) -> str:
    """Build the context string sent to Opus each cycle."""
    import json

    parts = ["=== RECENT CONVERSATION TURNS (ALL THREADS) ==="]
    for t in recent_turns[-40:]:
        conv = t.get("conversation_id", "?")[:8]
        role = t.get("role", "?").upper()
        turn = t.get("turn_number", "?")
        content = t.get("content", "")[:300]
        parts.append(f"[{conv}|{role} turn {turn}] {content}")

    parts.append("\n=== ACTIVE IMPRESSIONS (your previous observations) ===")
    for imp in active_impressions[:15]:
        payload = json.loads(imp["payload"]) if isinstance(imp["payload"], str) else imp["payload"]
        parts.append(
            f"- [{imp['type']}] pressure={imp['pressure']:.2f} "
            f"key={payload.get('similarity_key', '?')} | {imp['content'][:80]}"
        )

    if recent_outcomes:
        parts.append("\n=== RECENT OUTCOMES ===")
        for o in recent_outcomes[-10:]:
            parts.append(f"- [{o.get('signal', '?')}] {o.get('brief_reason', '')[:80]}")

    if pending_tasks:
        parts.append("\n=== PENDING TASKS (don't duplicate) ===")
        for t in pending_tasks:
            parts.append(f"- {t['description'][:80]}")

    if health_report:
        parts.append("\n=== SYSTEM HEALTH ===")
        parts.append(
            f"Maturity: {health_report.get('maturity_score', '?')} | "
            f"Flexibility: {health_report.get('flexibility_score', '?')} | "
            f"Growth: {health_report.get('growth_velocity', '?')}"
        )

    return "\n".join(parts)
