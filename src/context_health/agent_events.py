"""Provider-neutral agent event model for Context Health telemetry.

Defines a closed set of frozen dataclass events that any coding-agent adapter
can emit.  Events contain ONLY structural identifiers and numeric measurements.

Privacy Boundary
-----------------
Events do NOT store:

- Raw prompts, responses, or conversation transcripts
- Source code or file contents
- API keys, secrets, or authorization headers
- Raw error messages or stack traces

Error categorization uses a closed ``ToolErrorCategory`` enum.  Adapters must
map provider-specific errors into one of the defined categories.

Externally Supplied Signals
----------------------------
``relevant_context_ratio``, ``task_complexity``, and ``correction_detected``
are externally supplied measurements.  The event model does not infer them —
it accepts them from the caller when available.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Tool error categories (closed enum)
# ---------------------------------------------------------------------------


class ToolErrorCategory(enum.Enum):
    """Closed set of normalized tool error categories.

    Adapters must map provider-specific errors into one of these categories.
    No arbitrary strings or raw error text may enter the normalized event.
    """

    TIMEOUT = "timeout"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"
    SYNTAX_ERROR = "syntax_error"
    RATE_LIMITED = "rate_limited"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Session events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionStarted:
    """Marks the start of a coding-agent session.

    Attributes
    ----------
    session_id:
        Non-empty unique session identifier.
    context_window:
        Model context window size in tokens.  Must be > 0.
    model:
        Optional model identifier (e.g. ``"claude-sonnet-4-20250514"``).
    """

    session_id: str
    context_window: int = 200_000
    model: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if self.context_window <= 0:
            raise ValueError(
                f"context_window must be > 0, got {self.context_window}"
            )


@dataclass(frozen=True)
class SessionFinished:
    """Marks the end of a coding-agent session.

    Attributes
    ----------
    session_id:
        Non-empty session identifier.
    """

    session_id: str

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string")


# ---------------------------------------------------------------------------
# Turn events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TurnStarted:
    """Marks the beginning of a user turn.

    Attributes
    ----------
    session_id:
        Non-empty session identifier.
    turn_index:
        Zero-indexed turn number.  Must be >= 0.
    """

    session_id: str
    turn_index: int

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if self.turn_index < 0:
            raise ValueError(
                f"turn_index must be >= 0, got {self.turn_index}"
            )


@dataclass(frozen=True)
class TurnFinished:
    """Marks the end of a turn with externally supplied measurements.

    ``relevant_context_ratio``, ``task_complexity``, and ``correction_detected``
    are externally supplied measurements/signals.  The recorder does not infer
    them — it accepts them from the caller if available.

    Attributes
    ----------
    session_id:
        Non-empty session identifier.
    turn_index:
        Zero-indexed turn number.
    correction_detected:
        Externally supplied signal indicating whether a user correction was
        detected in this turn.  ``None`` means the signal is unavailable.
    relevant_context_ratio:
        Externally supplied relevant context ratio in ``[0.0, 1.0]``.
        ``None`` means the measurement is unavailable.
    task_complexity:
        Externally supplied task complexity score in ``[0.0, 1.0]``.
        ``None`` means the measurement is unavailable.
    """

    session_id: str
    turn_index: int
    correction_detected: bool | None = None
    relevant_context_ratio: float | None = None
    task_complexity: float | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if self.turn_index < 0:
            raise ValueError(
                f"turn_index must be >= 0, got {self.turn_index}"
            )
        if self.relevant_context_ratio is not None:
            if not math.isfinite(self.relevant_context_ratio) or not (
                0.0 <= self.relevant_context_ratio <= 1.0
            ):
                raise ValueError(
                    f"relevant_context_ratio must be in [0.0, 1.0], "
                    f"got {self.relevant_context_ratio}"
                )
        if self.task_complexity is not None:
            if not math.isfinite(self.task_complexity) or not (
                0.0 <= self.task_complexity <= 1.0
            ):
                raise ValueError(
                    f"task_complexity must be in [0.0, 1.0], "
                    f"got {self.task_complexity}"
                )


# ---------------------------------------------------------------------------
# Context measurement event
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextUpdated:
    """Reports provider-reported token measurements for the current turn.

    These are raw provider-reported values.  The recorder passes them through
    to the existing ``ContextTracker`` without reinterpretation.

    ``input_tokens`` and ``output_tokens`` preserve the exact semantics of
    ``ContextTracker.record_assistant_response``:

    - ``input_tokens`` represents the provider-reported input token count
      for this turn (used as context occupancy proxy AND accumulated into
      cumulative input tokens).
    - ``output_tokens`` represents the provider-reported output token count
      for this turn.

    Attributes
    ----------
    session_id:
        Non-empty session identifier.
    turn_index:
        Zero-indexed turn number.
    input_tokens:
        Provider-reported input token count for this turn.  Must be >= 0.
    output_tokens:
        Provider-reported output token count for this turn.  Must be >= 0.
    context_utilization:
        Provider-reported or computed context utilization in ``[0.0, 1.0]``.
    """

    session_id: str
    turn_index: int
    input_tokens: int
    output_tokens: int
    context_utilization: float

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if self.turn_index < 0:
            raise ValueError(
                f"turn_index must be >= 0, got {self.turn_index}"
            )
        if self.input_tokens < 0:
            raise ValueError(
                f"input_tokens must be >= 0, got {self.input_tokens}"
            )
        if self.output_tokens < 0:
            raise ValueError(
                f"output_tokens must be >= 0, got {self.output_tokens}"
            )
        if not math.isfinite(self.context_utilization) or not (
            0.0 <= self.context_utilization <= 1.0
        ):
            raise ValueError(
                f"context_utilization must be in [0.0, 1.0], "
                f"got {self.context_utilization}"
            )


# ---------------------------------------------------------------------------
# Tool events
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCallStarted:
    """Records the start of a tool invocation.

    Attributes
    ----------
    session_id:
        Non-empty session identifier.
    turn_index:
        Zero-indexed turn number.
    tool_name:
        Structural tool identifier (e.g. ``"run_command"``, ``"read_file"``).
    operation_id:
        Optional specific operation identifier (e.g. file path).
    """

    session_id: str
    turn_index: int
    tool_name: str
    operation_id: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if self.turn_index < 0:
            raise ValueError(
                f"turn_index must be >= 0, got {self.turn_index}"
            )
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")


@dataclass(frozen=True)
class ToolCallFinished:
    """Records the result of a tool invocation.

    Error categorization uses the closed ``ToolErrorCategory`` enum.
    No raw error messages or stack traces are stored.

    Attributes
    ----------
    session_id:
        Non-empty session identifier.
    turn_index:
        Zero-indexed turn number.
    tool_name:
        Structural tool identifier.
    success:
        Whether the tool invocation succeeded.
    is_retry:
        Explicit flag indicating this attempt is a retry.
    operation_id:
        Optional specific operation identifier.
    error_category:
        Optional normalized error category from the closed
        ``ToolErrorCategory`` enum.
    """

    session_id: str
    turn_index: int
    tool_name: str
    success: bool = True
    is_retry: bool = False
    operation_id: str | None = None
    error_category: ToolErrorCategory | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if self.turn_index < 0:
            raise ValueError(
                f"turn_index must be >= 0, got {self.turn_index}"
            )
        if not self.tool_name or not self.tool_name.strip():
            raise ValueError("tool_name must be a non-empty string")
        if self.error_category is not None and not isinstance(
            self.error_category, ToolErrorCategory
        ):
            raise ValueError(
                f"error_category must be an instance of ToolErrorCategory or None, "
                f"got {type(self.error_category).__name__}: {self.error_category!r}"
            )


# ---------------------------------------------------------------------------
# Union type for dispatching
# ---------------------------------------------------------------------------

AgentEvent = (
    SessionStarted
    | TurnStarted
    | ContextUpdated
    | ToolCallStarted
    | ToolCallFinished
    | TurnFinished
    | SessionFinished
)
