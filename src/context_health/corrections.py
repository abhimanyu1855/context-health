"""Correction event detection for Context Health.

Provides a deterministic, rule-based heuristic to detect explicit
user correction, rejection, and redirect statements during a conversation.

Important Disclaimers (V0.1)
-----------------------------
Correction detection is an **experimental deterministic heuristic**.

It is **NOT**:
- an LLM-based sentiment or satisfaction classifier
- a general semantic intent parser
- a hallucination detector
- a measure of agent correctness or failure probability

It has **NOT** been empirically validated. It operates strictly on a closed,
deterministic set of explicit syntactic and lexical markers (such as
"No, use PostgreSQL instead", "That's wrong", "I meant X, not Y").

Supported Patterns
------------------
Detects explicit user corrections categorized into:
1. **rejection_redirect**: Explicit rejection followed by a redirect instruction
   (e.g., "No, use PostgreSQL instead", "Don't do that; use X instead", "Stop, that's wrong").
2. **error_assertion**: Explicit statement that the agent's prior output was wrong
   (e.g., "That's wrong, use Redis", "Wrong: do X", "It's incorrect").
3. **misunderstanding**: Explicit claim that the requirement or intent was misunderstood
   (e.g., "You misunderstood the requirement", "That's not what I asked for", "I didn't ask for that").
4. **intent_clarification**: Clarification of what was actually intended
   (e.g., "I meant X, not Y", "Not what I meant").
5. **explicit_override**: Explicit conversational redirect of the prior action
   (e.g., "Actually, change that to X", "Instead of that, do Y").
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Sequence


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CorrectionEvent:
    """Represents a detected explicit user correction event.

    Attributes
    ----------
    message:
        The raw user message containing the correction.
    correction_type:
        Canonical classification of the correction.
    matched_phrase:
        The substring that triggered detection.
    turn_index:
        Optional conversation turn index where the correction occurred.
    """

    message: str
    correction_type: str
    matched_phrase: str
    turn_index: int | None = None


@dataclass(frozen=True)
class CorrectionResult:
    """Container for detected correction events and summary count.

    Supports sequence protocol (len, iter, indexing).
    """

    events: list[CorrectionEvent]
    correction_count: int

    def __len__(self) -> int:
        return self.correction_count

    def __iter__(self) -> Iterator[CorrectionEvent]:
        return iter(self.events)

    def __getitem__(self, index: int) -> CorrectionEvent:
        return self.events[index]


# ---------------------------------------------------------------------------
# Pattern definitions
# ---------------------------------------------------------------------------

_CORRECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # 1. Error assertion: explicit declaration of wrongness
    (
        "error_assertion",
        re.compile(
            r"\b(?:that[\x27\u2019]s|that\s+is|this\s+is|it[\x27\u2019]s|it\s+is)\s+(?:wrong|incorrect|not\s+right|not\s+correct|broken|invalid)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "error_assertion",
        re.compile(r"^\s*(?:wrong|incorrect)\s*[,.:;!-]\s*", re.IGNORECASE),
    ),

    # 2. Misunderstanding: explicit declaration that agent failed to understand
    (
        "misunderstanding",
        re.compile(
            r"\b(?:you\s+misunderstood|you\s+got\s+that\s+wrong|you\s+missed\s+the\s+requirement)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "misunderstanding",
        re.compile(
            r"\b(?:that[\x27\u2019]s|this\s+is|it[\x27\u2019]s)\s+not\s+what\s+i\s+(?:asked(?:\s+for)?|wanted|requested|meant)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "misunderstanding",
        re.compile(
            r"\bi\s+(?:didn[\x27\u2019]t\s+ask|never\s+asked|never\s+ask)\s+for\b",
            re.IGNORECASE,
        ),
    ),

    # 3. Rejection redirect: explicit "no" or "don't do that" followed by action
    (
        "rejection_redirect",
        re.compile(
            r"^\s*(?:no|nope)\s*[,.:;!-]\s*(?:please\s+)?(?:use|do|change|make|switch|try|put|instead|not|that[\x27\u2019]s)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "rejection_redirect",
        re.compile(r"^\s*(?:don[\x27\u2019]t|do\s+not)\s+do\s+that\b", re.IGNORECASE),
    ),
    (
        "rejection_redirect",
        re.compile(
            r"^\s*(?:stop|wait)\s*[,.:;!-]\s*(?:that[\x27\u2019]s|this\s+is|please|use|do|don[\x27\u2019]t|not)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "rejection_redirect",
        re.compile(r"\b(?:instead\s+of\s+that|undo\s+that\s+and)\b", re.IGNORECASE),
    ),

    # 4. Intent clarification: clarifying what was meant
    (
        "intent_clarification",
        re.compile(r"\bi\s+meant\b", re.IGNORECASE),
    ),
    (
        "intent_clarification",
        re.compile(r"\bnot\s+what\s+i\s+(?:meant|intended)\b", re.IGNORECASE),
    ),

    # 5. Explicit override: conversational redirect of prior action
    (
        "explicit_override",
        re.compile(
            r"\bactually\s*,\s*(?:change\s+that\s+to|change\s+this\s+to|switch\s+to|use|do\s+not|don[\x27\u2019]t)\b",
            re.IGNORECASE,
        ),
    ),
]


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------


def detect_single_message_correction(
    message: str,
    turn_index: int | None = None,
) -> CorrectionEvent | None:
    """Evaluate a single message string for an explicit correction.

    Returns
    -------
    CorrectionEvent | None
        The detected correction event or None if no correction was found.
    """
    if not message or not message.strip():
        return None

    cleaned = message.strip()

    for correction_type, pattern in _CORRECTION_PATTERNS:
        match = pattern.search(cleaned)
        if match:
            return CorrectionEvent(
                message=message,
                correction_type=correction_type,
                matched_phrase=match.group(0).strip(),
                turn_index=turn_index,
            )

    return None


def is_correction(message: str) -> bool:
    """Return True if message contains an explicit correction, False otherwise."""
    return detect_single_message_correction(message) is not None


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class CorrectionDetector:
    """Maintains sequential history of detected correction events.

    Can be used statelessly via :meth:`detect` or incrementally via
    :meth:`add_message`.
    """

    def __init__(self) -> None:
        self._events: list[CorrectionEvent] = []

    def reset(self) -> None:
        """Reset internal history and detected correction count."""
        self._events.clear()

    def add_message(
        self,
        message: str,
        turn_index: int | None = None,
    ) -> CorrectionEvent | None:
        """Process a single message and return any newly detected CorrectionEvent."""
        event = detect_single_message_correction(message, turn_index=turn_index)
        if event is not None:
            self._events.append(event)
        return event

    def detect(self, messages: Sequence[str]) -> CorrectionResult:
        """Statelessly evaluate a sequence of user messages for corrections.

        Parameters
        ----------
        messages:
            Sequence of user messages.

        Returns
        -------
        CorrectionResult
            Summary containing list of detected events and total correction_count.
        """
        self.reset()
        for idx, msg in enumerate(messages):
            self.add_message(msg, turn_index=idx)

        return CorrectionResult(
            events=list(self._events),
            correction_count=len(self._events),
        )

    @property
    def events(self) -> list[CorrectionEvent]:
        """List of all detected correction events so far."""
        return list(self._events)

    @property
    def correction_count(self) -> int:
        """Total number of detected correction events."""
        return len(self._events)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def detect_corrections(messages: Sequence[str]) -> CorrectionResult:
    """Convenience helper to detect corrections across a sequence of messages."""
    return CorrectionDetector().detect(messages)
