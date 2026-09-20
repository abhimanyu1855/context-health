"""Contradiction detection for Context Health.

Provides a deterministic, rule-based heuristic to detect explicit
instruction and requirement reversals across a conversation.

Important Disclaimers (V0.1)
-----------------------------
Contradiction detection is an **experimental deterministic heuristic**.

It is **NOT**:
- a general semantic reasoning engine
- an LLM-based evaluator
- a hallucination detector
- a reasoning verification tool
- a correctness verification or reliability prediction system

It has **NOT** been empirically validated as a general contradiction
detector. It operates strictly on a closed, explicit set of normalized
syntactic instruction patterns (e.g. "use X" vs "do not use X").

Supported Patterns
------------------
Detects explicit polar reversals on identical normalized targets for:
- USE: "use X" vs "do not use X" / "don't use X" / "never use X"
- ENABLE: "enable X" vs "disable X" / "do not enable X" / "don't enable X"
- ALLOW: "allow X" / "permit X" vs "disallow X" / "do not allow X" / "don't allow X"
- INCLUDE: "include X" vs "exclude X" / "do not include X" / "don't include X"
- ADD: "add X" / "insert X" vs "do not add X" / "don't add X" / "never add X"
- MODIFY: "modify X" / "edit X" / "change X" / "update X" vs "do not modify X" / "don't modify X"
- REMOVE: "remove X" / "delete X" vs "do not remove X" / "don't remove X" / "never remove X"
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterator, Sequence


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Contradiction:
    """Represents a detected contradiction between two statements.

    Attributes
    ----------
    first_statement:
        The earlier statement that established the instruction.
    second_statement:
        The subsequent statement that contradicted the earlier instruction.
    contradiction_type:
        Canonical classification of the contradiction (e.g. 'use_reversal').
    severity:
        Qualitative severity label (defaults to 'high').
    confidence:
        Confidence score in [0.0, 1.0] (deterministic rule match: 1.0).
    """

    first_statement: str
    second_statement: str
    contradiction_type: str
    severity: str = "high"
    confidence: float = 1.0


@dataclass(frozen=True)
class ContradictionResult:
    """Container for detected contradictions and summary count.

    Supports sequence protocol (len, iter, indexing).
    """

    contradictions: list[Contradiction]
    contradiction_count: int

    def __len__(self) -> int:
        return self.contradiction_count

    def __iter__(self) -> Iterator[Contradiction]:
        return iter(self.contradictions)

    def __getitem__(self, index: int) -> Contradiction:
        return self.contradictions[index]


@dataclass(frozen=True)
class _Instruction:
    """Internal normalized representation of an instruction clause."""

    action: str
    target: str
    polarity: bool  # True = affirmative/enable, False = negative/disable
    raw_statement: str


# ---------------------------------------------------------------------------
# Rule definitions & Normalization
# ---------------------------------------------------------------------------

# Ordered list of (canonical_action, polarity, regex_pattern).
# Negative patterns are listed BEFORE affirmative patterns to prevent
# affirmative patterns from greedily matching negated clauses (e.g. "do not use").
_INSTRUCTION_PATTERNS: list[tuple[str, bool, re.Pattern[str]]] = [
    # 1. USE
    ("use", False, re.compile(r"\b(?:do\s+not|don['’]t|never|cannot|can['’]t|should\s+not|shouldn['’]t)\s+use\s+(.+)", re.IGNORECASE)),
    ("use", False, re.compile(r"\b(?:stop\s+using|avoid\s+using|avoid)\s+(.+)", re.IGNORECASE)),
    ("use", True, re.compile(r"\b(?:must\s+use|please\s+use|should\s+use|always\s+use|use)\s+(.+)", re.IGNORECASE)),

    # 2. ENABLE
    ("enable", False, re.compile(r"\b(?:do\s+not|don['’]t|never)\s+enable\s+(.+)", re.IGNORECASE)),
    ("enable", False, re.compile(r"\b(?:disable|deactivate|turn\s+off)\s+(.+)", re.IGNORECASE)),
    ("enable", True, re.compile(r"\b(?:please\s+enable|must\s+enable|enable|activate|turn\s+on)\s+(.+)", re.IGNORECASE)),

    # 3. ALLOW
    ("allow", False, re.compile(r"\b(?:do\s+not|don['’]t|never)\s+allow\s+(.+)", re.IGNORECASE)),
    ("allow", False, re.compile(r"\b(?:disallow|prohibit|forbid)\s+(.+)", re.IGNORECASE)),
    ("allow", True, re.compile(r"\b(?:please\s+allow|must\s+allow|allow|permit)\s+(.+)", re.IGNORECASE)),

    # 4. INCLUDE
    ("include", False, re.compile(r"\b(?:do\s+not|don['’]t|never)\s+include\s+(.+)", re.IGNORECASE)),
    ("include", False, re.compile(r"\bexclude\s+(.+)", re.IGNORECASE)),
    ("include", True, re.compile(r"\b(?:please\s+include|must\s+include|include)\s+(.+)", re.IGNORECASE)),

    # 5. ADD
    ("add", False, re.compile(r"\b(?:do\s+not|don['’]t|never)\s+add\s+(.+)", re.IGNORECASE)),
    ("add", False, re.compile(r"\b(?:do\s+not|don['’]t|never)\s+insert\s+(.+)", re.IGNORECASE)),
    ("add", True, re.compile(r"\b(?:please\s+add|must\s+add|add|insert)\s+(.+)", re.IGNORECASE)),

    # 6. MODIFY
    ("modify", False, re.compile(r"\b(?:do\s+not|don['’]t|never)\s+(?:modify|edit|change|update)\s+(.+)", re.IGNORECASE)),
    ("modify", True, re.compile(r"\b(?:please\s+(?:modify|edit|change|update)|must\s+(?:modify|edit|change|update)|modify|edit|change|update)\s+(.+)", re.IGNORECASE)),

    # 7. REMOVE
    ("remove", False, re.compile(r"\b(?:do\s+not|don['’]t|never)\s+(?:remove|delete)\s+(.+)", re.IGNORECASE)),
    ("remove", True, re.compile(r"\b(?:please\s+(?:remove|delete)|must\s+(?:remove|delete)|remove|delete)\s+(.+)", re.IGNORECASE)),
]


def _normalize_target(raw: str) -> str:
    """Normalize target string: strip sentence punctuation and whitespace."""
    text = raw.strip()
    text = text.strip(".,!?;:'\"`()[]{}")
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def _extract_instructions_from_statement(statement: str) -> list[_Instruction]:
    """Extract canonical instructions from a single statement string."""
    if not statement or not statement.strip():
        return []

    # Split into candidate sentences or clauses
    raw_clauses = re.split(r"(?:[\n\r]+|(?<=[.!?])\s+)", statement)
    instructions: list[_Instruction] = []

    for clause in raw_clauses:
        clause_str = clause.strip()
        if not clause_str:
            continue

        for action, polarity, pattern in _INSTRUCTION_PATTERNS:
            match = pattern.search(clause_str)
            if match:
                raw_target = match.group(1)
                normalized_target = _normalize_target(raw_target)
                if normalized_target:
                    instructions.append(
                        _Instruction(
                            action=action,
                            target=normalized_target,
                            polarity=polarity,
                            raw_statement=statement,
                        )
                    )
                # Take the first rule that matches this clause
                break

    return instructions


# ---------------------------------------------------------------------------
# Contradiction Detector
# ---------------------------------------------------------------------------


class ContradictionDetector:
    """Deterministic, rule-based detector for observable instruction reversals.

    Maintains sequential instruction history to detect when a subsequent
    statement directly reverses an active instruction on the same target.
    """

    def __init__(self) -> None:
        self._history: list[_Instruction] = []
        self._contradictions: list[Contradiction] = []

    def reset(self) -> None:
        """Reset internal history and detected contradictions."""
        self._history.clear()
        self._contradictions.clear()

    def add_statement(self, statement: str) -> list[Contradiction]:
        """Process a single statement incrementally.

        Returns any newly detected contradictions caused by this statement.
        """
        new_contradictions: list[Contradiction] = []
        extracted = _extract_instructions_from_statement(statement)

        for current_inst in extracted:
            # Check backwards through history for the most recent statement on this action/target
            for prior_inst in reversed(self._history):
                if (
                    prior_inst.action == current_inst.action
                    and prior_inst.target == current_inst.target
                ):
                    if prior_inst.polarity != current_inst.polarity:
                        contradiction = Contradiction(
                            first_statement=prior_inst.raw_statement,
                            second_statement=current_inst.raw_statement,
                            contradiction_type=f"{current_inst.action}_reversal",
                            severity="high",
                            confidence=1.0,
                        )
                        new_contradictions.append(contradiction)
                        self._contradictions.append(contradiction)
                    # Stop checking once the most recent relevant instruction is found
                    break

            self._history.append(current_inst)

        return new_contradictions

    def detect(self, statements: Sequence[str]) -> ContradictionResult:
        """Statelessly evaluate a sequence of statements for contradictions.

        Parameters
        ----------
        statements:
            Sequence of string statements or conversation messages.

        Returns
        -------
        ContradictionResult
            Summary containing list of contradictions and contradiction_count.
        """
        self.reset()
        for statement in statements:
            self.add_statement(statement)

        return ContradictionResult(
            contradictions=list(self._contradictions),
            contradiction_count=len(self._contradictions),
        )

    @property
    def contradictions(self) -> list[Contradiction]:
        """List of all detected contradictions so far."""
        return list(self._contradictions)

    @property
    def contradiction_count(self) -> int:
        """Total number of detected contradictions."""
        return len(self._contradictions)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def detect_contradictions(statements: Sequence[str]) -> ContradictionResult:
    """Convenience helper to detect contradictions across a sequence of statements.

    Parameters
    ----------
    statements:
        A sequence of statement strings.

    Returns
    -------
    ContradictionResult
        The detected contradictions and contradiction_count.
    """
    return ContradictionDetector().detect(statements)
