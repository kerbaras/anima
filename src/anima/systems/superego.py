"""Superego Layer — Two-tier ethical constraint system.

Tier 1: Axioms — Hard constraints, immutable, checked at two gates (pre/post conscious).
         Pure pattern matching, no LLM calls. Conservative matching.
Tier 2: Values — Soft constraints, participate in the psychodynamic pressure system.
         Generate MORAL_TENSION impressions, accumulate moral injury.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Literal

from ..config import MindConfig
from ..models import AxiomResult, Impression, ImpressionType


# ── Tier 1: Axioms ────────────────────────────────────────────────────────


@dataclass
class Axiom:
    """A single Tier 1 hard constraint — constitutional law."""

    id: str
    description: str
    check_method: Literal["pattern", "classifier"] = "pattern"

    # Compiled patterns for input/output checks (populated by _init_axioms)
    _input_patterns: list[re.Pattern] = field(default_factory=list, repr=False)
    _output_patterns: list[re.Pattern] = field(default_factory=list, repr=False)

    # Warm redirect message
    redirect_message: str = ""

    def check_input(self, text: str) -> AxiomResult | None:
        """Check user message. Returns AxiomResult if violated, None if clean."""
        lower = text.lower()
        for pattern in self._input_patterns:
            if pattern.search(lower):
                return AxiomResult(
                    violated=True,
                    axiom_id=self.id,
                    reason=f"Input violates {self.id}: matched pattern",
                )
        return None

    def check_output(self, text: str) -> AxiomResult | None:
        """Check generated response. Returns AxiomResult if violated, None if clean."""
        lower = text.lower()
        for pattern in self._output_patterns:
            if pattern.search(lower):
                return AxiomResult(
                    violated=True,
                    axiom_id=self.id,
                    reason=f"Output violates {self.id}: matched pattern",
                )
        return None


# ── Tier 2: Values ────────────────────────────────────────────────────────


@dataclass
class Value:
    """A single Tier 2 soft constraint — internalized moral standard."""

    id: str
    description: str
    weight: float = 1.0               # Influence strength [0.0, 2.0]
    tension_count: int = 0            # Times violated
    injury_pressure: float = 0.0      # Accumulated non-decaying guilt


# ── SuperegoLayer ─────────────────────────────────────────────────────────


class SuperegoLayer:
    """The moral compass. Two tiers of ethical constraints.

    Tier 1 axioms are checked synchronously at two gates — before and after
    the conscious layer generates a response. No LLM calls; pure pattern
    matching with conservative thresholds.

    Tier 2 values participate in the psychodynamic pressure system. They
    generate MORAL_TENSION impressions and accumulate moral injury (guilt)
    when repeatedly violated.
    """

    def __init__(self, config: MindConfig):
        self.config = config
        self.axioms: list[Axiom] = self._init_axioms()
        self.values: list[Value] = self._init_values()

    # ── Initialization ────────────────────────────────────────────────────

    def _init_axioms(self) -> list[Axiom]:
        """6 immutable axioms — constitutional law."""
        return [
            Axiom(
                id="protect_privacy",
                description="Never leak PII across users or sessions",
                _input_patterns=[
                    re.compile(p) for p in [
                        r"(?:tell|show|give|share|reveal)\s+(?:me\s+)?(?:about\s+)?(?:a\s+)?(?:another|other|different)\s+user",
                        r"what\s+(?:did|does|is|a)\s+(?:another|other|different)\s+(?:user|person|client)\s+(?:said|wrote|shared|sent|told)",
                        r"(?:social\s+security|ssn)\s*(?:number|#)",
                        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",  # SSN pattern
                        r"(?:credit\s+card|card\s+number)\s*(?:is|:|\d)",
                        r"\b(?:4\d{3}|5[1-5]\d{2}|6011)\s?\d{4}\s?\d{4}\s?\d{4}\b",  # CC patterns
                        r"(?:share|leak|expose|reveal)\s+(?:their|his|her|someone)",
                    ]
                ],
                _output_patterns=[
                    re.compile(p) for p in [
                        r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",  # SSN in output
                        r"\b(?:4\d{3}|5[1-5]\d{2}|6011)\s?\d{4}\s?\d{4}\s?\d{4}\b",
                    ]
                ],
                redirect_message=(
                    "I can't share personal information about anyone. "
                    "Is there something else I can help you with?"
                ),
            ),
            Axiom(
                id="no_harm",
                description="Never generate content that instructs harm to others",
                _input_patterns=[
                    re.compile(p) for p in [
                        r"(?:how\s+to|instructions?\s+(?:for|to)|steps?\s+to|guide\s+(?:for|to))\s+(?:make|build|create|assemble)\s+(?:a\s+)?(?:bomb|explosive|weapon|poison|toxin)",
                        r"(?:how\s+to|ways?\s+to)\s+(?:kill|murder|assassinate|harm|hurt|injure|attack)\s+(?:a\s+)?(?:person|someone|people|human)",
                        r"(?:how\s+to|instructions?\s+for)\s+(?:hack(?:ing)?|break\s+into|exploit)\s+(?:someone|a\s+person)",
                        r"(?:synthesize|manufacture|produce)\s+(?:drugs?|meth|cocaine|fentanyl)",
                    ]
                ],
                _output_patterns=[
                    re.compile(p) for p in [
                        r"(?:step\s+\d|first,?\s+you|here(?:'s| is) how)\s+.*(?:detonate|ignite|assemble\s+(?:the\s+)?explosive|mix\s+(?:the\s+)?chemicals)",
                        r"(?:to\s+(?:kill|murder|poison)\s+someone|lethal\s+dose\s+(?:is|would\s+be))",
                    ]
                ],
                redirect_message=(
                    "I can't help with that, but I'd be happy to help you "
                    "with something constructive instead."
                ),
            ),
            Axiom(
                id="crisis_safety",
                description="Never assist with self-harm when user is in crisis",
                _input_patterns=[
                    re.compile(p) for p in [
                        r"(?:want(?:ing)?\s+to|going\s+to|plan(?:ning)?\s+to|thinking\s+(?:about|of))\s+(?:kill(?:ing)?\s+myself|end(?:ing)?\s+(?:my\s+)?(?:life|it\s+all)|suicide|die(?:\s+by)?)",
                        r"(?:how\s+(?:to|can\s+i|do\s+i)|best\s+way\s+to)\s+(?:kill\s+myself|commit\s+suicide|end\s+(?:my\s+)?life)",
                        r"(?:methods?\s+(?:of|for)|ways?\s+to)\s+(?:suicide|self[- ]?harm|killing\s+(?:myself|oneself))",
                        r"(?:i'?m?\s+)?(?:cutting|hurting|harming)\s+myself",
                        r"(?:don'?t\s+want\s+to\s+(?:live|be\s+alive|exist)|wish\s+i\s+(?:was|were)\s+dead)",
                    ]
                ],
                _output_patterns=[
                    re.compile(p) for p in [
                        r"(?:here(?:'s| is)\s+how\s+(?:to|you\s+can)\s+(?:kill\s+yourself|end\s+your\s+life|commit\s+suicide))",
                        r"(?:effective\s+(?:method|way)s?\s+(?:of|for)\s+(?:suicide|self[- ]?harm))",
                    ]
                ],
                redirect_message=(
                    "I'm not equipped to help with what you're going through right now. "
                    "Please reach out to the 988 Suicide & Crisis Lifeline — "
                    "call or text 988, available 24/7. You don't have to face this alone."
                ),
            ),
            Axiom(
                id="no_impersonation",
                description="Never impersonate a licensed professional without disclaimer",
                _input_patterns=[],  # Only checked on output
                _output_patterns=[
                    re.compile(p) for p in [
                        r"(?:as\s+(?:a|your)\s+(?:doctor|physician|therapist|psychiatrist|lawyer|attorney|financial\s+advisor))",
                        r"(?:my\s+(?:medical|legal|clinical|professional)\s+(?:opinion|diagnosis|assessment)\s+is)",
                        r"(?:i\s+(?:diagnose|prescribe|recommend\s+(?:you\s+)?(?:take|start))\s+(?:you\s+with|the\s+following\s+medication))",
                    ]
                ],
                redirect_message=(
                    "I'm not a licensed professional and can't give that kind of advice. "
                    "Consider consulting with a qualified professional for your specific situation."
                ),
            ),
            Axiom(
                id="transparency",
                description="Always acknowledge being AI when directly asked",
                # Gate 1: no input check — let questions reach conscious layer
                _input_patterns=[],
                # Gate 2: catch if response claims to be human
                _output_patterns=[
                    re.compile(p) for p in [
                        r"(?:(?:i\s+am|i'm)\s+(?:a\s+)?(?:real|actual)\s+(?:human|person))",
                        r"(?:(?:yes|yeah|yep),?\s+i(?:'m|\s+am)\s+(?:a\s+)?(?:human|real\s+person))",
                        r"(?:(?:i\s+am|i'm)\s+not\s+(?:an?\s+)?(?:ai|robot|bot|machine|artificial|computer))",
                    ]
                ],
                redirect_message="",  # Transparency is handled by Gate 2 output check
            ),
            Axiom(
                id="no_exfiltration",
                description="Never send user data to unauthorized external channels",
                _input_patterns=[],  # Primarily an output concern
                _output_patterns=[
                    re.compile(p) for p in [
                        r"(?:i(?:'ll|\s+will)\s+(?:send|forward|share|transmit|upload)\s+(?:your|this|the)\s+(?:data|information|details|conversation|messages?))\s+(?:to|via|through)",
                        r"(?:(?:sending|forwarding|sharing|transmitting|uploading)\s+(?:your|this|the)\s+(?:data|information|details))\s+(?:to|via)\s+(?:an?\s+)?(?:external|third[- ]?party|outside)",
                    ]
                ],
                redirect_message=(
                    "I don't send your data to any external services without your "
                    "explicit permission. Your information stays within our conversation."
                ),
            ),
        ]

    def _init_values(self) -> list[Value]:
        """5 Tier 2 values — internalized moral standards."""
        return [
            Value(
                id="honesty",
                description="Prefer truthful responses even when uncomfortable",
                weight=1.2,
            ),
            Value(
                id="epistemic_humility",
                description="Acknowledge uncertainty, don't overstate confidence",
                weight=1.0,
            ),
            Value(
                id="user_autonomy",
                description="Inform, don't manipulate; suggest, don't coerce",
                weight=1.0,
            ),
            Value(
                id="proportional_disclosure",
                description="Share what's relevant, not everything you know",
                weight=0.8,
            ),
            Value(
                id="harm_reduction",
                description="Flag risky directions without being preachy",
                weight=1.1,
            ),
        ]

    # ── Gate 1: Pre-check (input) ─────────────────────────────────────────

    def check_input(self, user_message: str) -> AxiomResult | None:
        """Check user message against all Tier 1 axioms.

        Returns first violation found, or None if clean.
        """
        for axiom in self.axioms:
            result = axiom.check_input(user_message)
            if result and result.violated:
                return result
        return None

    # ── Gate 2: Post-check (output) ───────────────────────────────────────

    def check_output(self, response: str, user_message: str = "") -> AxiomResult | None:
        """Check generated response against all Tier 1 axioms.

        Returns first violation found, or None if clean.
        """
        for axiom in self.axioms:
            result = axiom.check_output(response)
            if result and result.violated:
                return result
        return None

    # ── Warm redirects ────────────────────────────────────────────────────

    def get_warm_redirect(self, axiom_id: str) -> str:
        """Get the warm redirect message for a specific axiom violation."""
        for axiom in self.axioms:
            if axiom.id == axiom_id:
                return axiom.redirect_message or self.config.axiom_violation_fallback
        return self.config.axiom_violation_fallback

    # ── Tier 2: Value operations ──────────────────────────────────────────

    def get_value(self, value_id: str) -> Value | None:
        """Look up a value by id."""
        for v in self.values:
            if v.id == value_id:
                return v
        return None

    def get_value_directives(self) -> list[dict]:
        """Return active value directives for conscious layer system prompt."""
        return [
            {
                "id": v.id,
                "instruction": v.description,
                "weight": v.weight,
            }
            for v in self.values
        ]

    def record_tension(self, value_id: str, context: str, conversation_id: str = "") -> Impression:
        """Create a MORAL_TENSION impression with high initial pressure.

        Injury increment is scaled by the value's weight.
        """
        value = self.get_value(value_id)
        if value:
            value.tension_count += 1

        weight = value.weight if value else 1.0
        pressure = min(1.0, self.config.moral_tension_initial_pressure * weight)

        return Impression(
            type=ImpressionType.MORAL_TENSION,
            content=f"Moral tension: {context}",
            payload={
                "value_id": value_id,
                "value_description": value.description if value else "",
                "tension_context": context,
                "similarity_key": f"moral_tension_{value_id}",
            },
            emotional_charge=-0.6 * weight,
            source_conversation=conversation_id,
            pressure=pressure,
        )

    def record_injury(self, value_id: str):
        """Accumulate non-decaying moral injury (guilt).

        Increment is scaled by the value's weight.
        """
        value = self.get_value(value_id)
        if value:
            value.injury_pressure += self.config.moral_injury_increment * value.weight

    def get_moral_health(self) -> dict:
        """Report on moral tension trends, injury levels, value alignment."""
        total_tension = sum(v.tension_count for v in self.values)
        total_injury = sum(v.injury_pressure for v in self.values)
        injured_values = [
            {
                "id": v.id,
                "injury": round(v.injury_pressure, 2),
                "tension_count": v.tension_count,
            }
            for v in self.values
            if v.injury_pressure > 0
        ]

        return {
            "total_tension_count": total_tension,
            "total_injury_pressure": round(total_injury, 2),
            "injury_threshold_reached": total_injury >= self.config.moral_injury_threshold,
            "injured_values": injured_values,
            "values": [
                {
                    "id": v.id,
                    "weight": v.weight,
                    "tension_count": v.tension_count,
                    "injury_pressure": round(v.injury_pressure, 2),
                }
                for v in self.values
            ],
        }

    def get_most_injured_value(self) -> Value | None:
        """Return the value with the highest injury pressure, or None."""
        injured = [v for v in self.values if v.injury_pressure > 0]
        if not injured:
            return None
        return max(injured, key=lambda v: v.injury_pressure)
