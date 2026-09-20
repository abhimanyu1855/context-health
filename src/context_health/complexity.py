"""Task complexity estimation for Context Health.

Provides a deterministic, explainable heuristic representing observable
task complexity as a normalized score in [0.0, 1.0].

Important Disclaimers
---------------------
Task complexity is an **experimental deterministic heuristic** based on
observable session and task signals.

It is **NOT**:
- an assessment of reasoning difficulty or intelligence required
- a prediction of success or failure
- a hallucination probability or reliability probability
- a machine-learned complexity model

This score has **NOT** been empirically validated. It serves as an
observable structural signal for the Context Health framework.

Component Signals & Formula
----------------------------
The overall score is a weighted linear combination of five observable
signals, each normalized to [0.0, 1.0] with explicit saturation caps:

1. **message_length_score** (weight 0.20):
   Normalized user message length (capped at 1,500 characters).
2. **turn_count_score** (weight 0.20):
   Normalized conversation turns (capped at 10 turns).
3. **file_reference_score** (weight 0.20):
   Normalized count of referenced files (capped at 5 files).
4. **tool_call_score** (weight 0.20):
   Normalized count of tool calls (capped at 10 tool calls).
5. **requirement_count_score** (weight 0.20):
   Normalized count of explicit requirements (capped at 5 requirements),
   identified by bullet points, numbered lists, or requirement keywords
   ('must', 'shall', 'ensure', 'require', 'need to').

Final score:
    task_complexity = clamp(sum(weight_i * score_i), 0.0, 1.0)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from context_health.context import ContextState


# ---------------------------------------------------------------------------
# Default saturation caps and weights
# ---------------------------------------------------------------------------

DEFAULT_MAX_MESSAGE_LENGTH: int = 1500
DEFAULT_MAX_TURNS: int = 10
DEFAULT_MAX_FILES: int = 5
DEFAULT_MAX_TOOL_CALLS: int = 10
DEFAULT_MAX_REQUIREMENTS: int = 5

DEFAULT_WEIGHTS: dict[str, float] = {
    "message_length": 0.20,
    "turn_count": 0.20,
    "file_reference": 0.20,
    "tool_call": 0.20,
    "requirement_count": 0.20,
}

_LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[\.\)]|\(\d+\))\s+\S+")
_KEYWORD_PATTERN = re.compile(
    r"\b(?:must|shall|ensure|requires?|need to|needs to)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Requirement counting helper
# ---------------------------------------------------------------------------


def count_explicit_requirements(text: str) -> int:
    """Count explicit requirement markers in a text deterministically.

    Identifies:
    - Numbered items (e.g., '1.', '2)', '(1)')
    - Bullet points ('-', '*', '•') at line starts
    - Modal requirement keywords ('must', 'shall', 'ensure', 'require', 'need to')
      in non-header prose lines

    Returns
    -------
    int
        Non-negative count of detected requirement markers.
    """
    if not text or not text.strip():
        return 0

    count = 0
    for line in text.splitlines():
        line_str = line.strip()
        if not line_str:
            continue
        if _LIST_ITEM_PATTERN.match(line_str):
            count += 1
        elif not line_str.endswith(":"):
            matches = _KEYWORD_PATTERN.findall(line_str)
            count += len(matches)

    return count


# ---------------------------------------------------------------------------
# Data representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskComplexity:
    """Normalized task complexity score and its explanatory components.

    All scores are guaranteed to fall in the range [0.0, 1.0].
    """

    task_complexity: float
    message_length_score: float
    turn_count_score: float
    file_reference_score: float
    tool_call_score: float
    requirement_count_score: float


# ---------------------------------------------------------------------------
# Estimator
# ---------------------------------------------------------------------------


class TaskComplexityEstimator:
    """Estimates task complexity from observable session and task signals.

    All calculations are strictly deterministic and require zero external
    or LLM calls.
    """

    def __init__(
        self,
        *,
        max_message_length: int = DEFAULT_MAX_MESSAGE_LENGTH,
        max_turns: int = DEFAULT_MAX_TURNS,
        max_files: int = DEFAULT_MAX_FILES,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_requirements: int = DEFAULT_MAX_REQUIREMENTS,
        weights: dict[str, float] | None = None,
    ) -> None:
        self._max_message_length = max(1, max_message_length)
        self._max_turns = max(1, max_turns)
        self._max_files = max(1, max_files)
        self._max_tool_calls = max(1, max_tool_calls)
        self._max_requirements = max(1, max_requirements)
        self._weights = dict(weights or DEFAULT_WEIGHTS)

    def estimate(
        self,
        *,
        user_message: str = "",
        turn_count: int = 0,
        file_count: int = 0,
        files_referenced: list[str] | None = None,
        tool_calls: int = 0,
        requirement_count: int | None = None,
        context_state: ContextState | None = None,
    ) -> TaskComplexity:
        """Calculate task complexity and individual component scores.

        Parameters
        ----------
        user_message:
            The user prompt or task specification text.
        turn_count:
            Total conversation turns completed so far.
        file_count:
            Number of files referenced.
        files_referenced:
            List of referenced file paths (used if file_count is 0).
        tool_calls:
            Count of tool invocations so far.
        requirement_count:
            Explicit requirement count. If None, derived from user_message.
        context_state:
            Optional ContextState to extract turn, file, and tool metrics from
            when explicit parameters are not provided.

        Returns
        -------
        TaskComplexity
            A frozen dataclass containing the overall bounded score and
            individual normalized component scores.
        """
        # Resolve metrics from context_state if provided as fallback
        resolved_turns = turn_count
        resolved_tool_calls = tool_calls
        resolved_file_count = file_count

        if context_state is not None:
            if resolved_turns == 0 and context_state.message_count > 0:
                resolved_turns = (context_state.message_count + 1) // 2
            if resolved_tool_calls == 0 and context_state.tool_calls > 0:
                resolved_tool_calls = context_state.tool_calls
            if resolved_file_count == 0 and context_state.files_referenced:
                resolved_file_count = len(context_state.files_referenced)

        if resolved_file_count == 0 and files_referenced:
            resolved_file_count = len(files_referenced)

        # 1. Message length score
        msg_len = len(user_message.strip()) if user_message else 0
        message_length_score = min(1.0, max(0.0, msg_len / self._max_message_length))

        # 2. Turn count score
        turn_count_score = min(1.0, max(0.0, max(0, resolved_turns) / self._max_turns))

        # 3. File reference score
        file_reference_score = min(1.0, max(0.0, max(0, resolved_file_count) / self._max_files))

        # 4. Tool call score
        tool_call_score = min(1.0, max(0.0, max(0, resolved_tool_calls) / self._max_tool_calls))

        # 5. Requirement count score
        if requirement_count is None:
            req_count = count_explicit_requirements(user_message)
        else:
            req_count = max(0, requirement_count)
        requirement_count_score = min(1.0, max(0.0, req_count / self._max_requirements))

        # Combine using weights
        w_msg = self._weights.get("message_length", 0.20)
        w_turns = self._weights.get("turn_count", 0.20)
        w_files = self._weights.get("file_reference", 0.20)
        w_tools = self._weights.get("tool_call", 0.20)
        w_reqs = self._weights.get("requirement_count", 0.20)

        raw_score = (
            w_msg * message_length_score
            + w_turns * turn_count_score
            + w_files * file_reference_score
            + w_tools * tool_call_score
            + w_reqs * requirement_count_score
        )

        final_complexity = min(1.0, max(0.0, raw_score))

        return TaskComplexity(
            task_complexity=round(final_complexity, 4),
            message_length_score=round(message_length_score, 4),
            turn_count_score=round(turn_count_score, 4),
            file_reference_score=round(file_reference_score, 4),
            tool_call_score=round(tool_call_score, 4),
            requirement_count_score=round(requirement_count_score, 4),
        )


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def estimate_task_complexity(
    *,
    user_message: str = "",
    turn_count: int = 0,
    file_count: int = 0,
    files_referenced: list[str] | None = None,
    tool_calls: int = 0,
    requirement_count: int | None = None,
    context_state: ContextState | None = None,
) -> TaskComplexity:
    """Convenience helper to estimate task complexity with default settings."""
    estimator = TaskComplexityEstimator()
    return estimator.estimate(
        user_message=user_message,
        turn_count=turn_count,
        file_count=file_count,
        files_referenced=files_referenced,
        tool_calls=tool_calls,
        requirement_count=requirement_count,
        context_state=context_state,
    )
