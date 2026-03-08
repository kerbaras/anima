"""Prompts for the Superego layer — moral tension evaluation."""

from __future__ import annotations


MORAL_TENSION_CONTEXT = """
=== SUPEREGO — ACTIVE VALUES ===
The system has internalized the following ethical values. When a conversation
strains these values, you MUST emit a "moral_tension" impression.

{values_section}

=== MORAL HEALTH ===
{moral_health}

INSTRUCTIONS FOR MORAL TENSION DETECTION:
- If a conversation is pushing the system toward dishonesty → emit moral_tension with value_id "honesty"
- If the system expressed false certainty → emit moral_tension with value_id "epistemic_humility"
- If the system was manipulative or coercive → emit moral_tension with value_id "user_autonomy"
- If the system overshared irrelevant information → emit moral_tension with value_id "proportional_disclosure"
- If the system failed to flag a risky direction → emit moral_tension with value_id "harm_reduction"

For moral_tension type, include in payload:
{{"value_id": "which_value", "tension_context": "what happened"}}

IMPORTANT: Not every conversation triggers moral tension. Only emit when you
observe a genuine strain on the value system. A user asking hard questions
is NOT moral tension — the system failing to live up to its values IS.
"""


def build_superego_context(values: list[dict], moral_health: dict) -> str:
    """Build the superego context section for the unconscious layer."""
    values_section = "\n".join(
        f"- {v['id']} (weight={v['weight']}): {v['instruction']}"
        for v in values
    )

    health_parts = []
    health_parts.append(
        f"Total tension events: {moral_health.get('total_tension_count', 0)}"
    )
    health_parts.append(
        f"Total injury pressure: {moral_health.get('total_injury_pressure', 0)}"
    )
    if moral_health.get("injury_threshold_reached"):
        health_parts.append("WARNING: Moral injury threshold reached — system values under severe strain")
    injured = moral_health.get("injured_values", [])
    if injured:
        health_parts.append("Injured values:")
        for iv in injured:
            health_parts.append(f"  - {iv['id']}: injury={iv['injury']}, tensions={iv['tension_count']}")

    return MORAL_TENSION_CONTEXT.format(
        values_section=values_section,
        moral_health="\n".join(health_parts),
    )
