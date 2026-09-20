"""Tool retry tracking for Context Health.

Provides a deterministic, rule-based mechanism for tracking tool-call retries
and repeated attempts during a session.

Important Disclaimers (V0.1)
-----------------------------
Tool retry tracking is an **experimental deterministic observation signal**.

It is **NOT**:
- an assessment of whether the agent was "wrong"
- a measure of context degradation
- an LLM-based or semantic similarity evaluator
- an automatic retry execution or intervention system
- a hallucination detector

It has **NOT** been empirically validated. It tracks observable repeated
tool attempts under explicit conditions (e.g. repeated operation after failure
or explicit retry metadata).

Definition
----------
A tool retry occurs when a tool operation is attempted again because a previous
attempt failed, was invalid, or was explicitly marked as a retry by the caller.

Ordinary repeated calls to the same tool with different arguments (e.g.
``read_file("a.py")`` followed by ``read_file("b.py")``) are **NOT** retries.
Independent successful calls with the same arguments are also **NOT**
automatically treated as retries unless explicitly marked as such.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Sequence


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolAttempt:
    """Represents a single tool invocation attempt.

    Attributes
    ----------
    tool_name:
        The name/identifier of the tool (e.g. "run_command", "read_file").
    operation_id:
        Optional specific operation/target identifier (e.g. command string, file path).
    success:
        Whether this tool execution attempt succeeded.
    is_retry:
        Explicit caller annotation indicating this attempt is a retry.
    error_message:
        Optional error details if the attempt failed.
    turn_index:
        Optional conversation turn index.
    """

    tool_name: str
    operation_id: str | None = None
    success: bool = True
    is_retry: bool = False
    error_message: str | None = None
    turn_index: int | None = None


@dataclass(frozen=True)
class ToolRetryEvent:
    """Represents a detected tool retry event.

    Attributes
    ----------
    tool_name:
        The name of the retried tool.
    operation_id:
        The operation identifier for the retried tool call.
    attempt_number:
        The ordinal attempt number (e.g. 2 for the first retry).
    reason:
        Explanation for retry classification (e.g. "failed_previous_attempt", "explicit_retry").
    turn_index:
        Optional conversation turn index where the retry occurred.
    error_message:
        Optional error message from previous/current attempt.
    """

    tool_name: str
    operation_id: str | None = None
    attempt_number: int = 2
    reason: str = "failed_previous_attempt"
    turn_index: int | None = None
    error_message: str | None = None


@dataclass(frozen=True)
class ToolRetryResult:
    """Container for detected tool retry events and aggregate retry count.

    Supports sequence protocol (len, iter, indexing).
    """

    events: list[ToolRetryEvent]
    retry_count: int

    def __len__(self) -> int:
        return self.retry_count

    def __iter__(self) -> Iterator[ToolRetryEvent]:
        return iter(self.events)

    def __getitem__(self, index: int) -> ToolRetryEvent:
        return self.events[index]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------


class ToolRetryDetector:
    """Maintains sequential history of tool attempts and detects retries.

    A retry is detected when:
    1. An attempt is explicitly marked with ``is_retry=True``.
    2. An attempt targets the same ``(tool_name, operation_id)`` as a previous
       attempt that ended in failure (``success=False``).

    Independent successful calls to the same ``(tool_name, operation_id)`` are
    not treated as retries unless ``is_retry=True``. Calls with different
    ``operation_id`` values are treated as distinct operations.
    """

    def __init__(self) -> None:
        self._events: list[ToolRetryEvent] = []
        self._last_status: dict[tuple[str, str | None], bool] = {}
        self._attempt_counts: dict[tuple[str, str | None], int] = {}

    def reset(self) -> None:
        """Reset internal history and detected retry counts."""
        self._events.clear()
        self._last_status.clear()
        self._attempt_counts.clear()

    def record_attempt(
        self,
        tool_name: str,
        *,
        operation_id: str | None = None,
        success: bool = True,
        is_retry: bool = False,
        error_message: str | None = None,
        turn_index: int | None = None,
    ) -> ToolRetryEvent | None:
        """Record a tool attempt and return a ToolRetryEvent if it is a retry.

        Parameters
        ----------
        tool_name:
            The name of the tool invoked.
        operation_id:
            Optional identifier for the specific operation (e.g. command string or file path).
        success:
            Whether this attempt succeeded. Defaults to True.
        is_retry:
            Explicit caller flag indicating this attempt is a retry. Defaults to False.
        error_message:
            Optional error message.
        turn_index:
            Optional turn index.

        Returns
        -------
        ToolRetryEvent | None
            A ToolRetryEvent if this attempt is classified as a retry, or None otherwise.
        """
        key = (tool_name, operation_id)
        prev_status = self._last_status.get(key)
        current_count = self._attempt_counts.get(key, 0) + 1
        self._attempt_counts[key] = current_count

        event: ToolRetryEvent | None = None

        if is_retry:
            event = ToolRetryEvent(
                tool_name=tool_name,
                operation_id=operation_id,
                attempt_number=current_count,
                reason="explicit_retry",
                turn_index=turn_index,
                error_message=error_message,
            )
        elif operation_id is not None and prev_status is False:
            event = ToolRetryEvent(
                tool_name=tool_name,
                operation_id=operation_id,
                attempt_number=current_count,
                reason="failed_previous_attempt",
                turn_index=turn_index,
                error_message=error_message,
            )

        # Update last known status for this tool/operation
        self._last_status[key] = success

        if event is not None:
            self._events.append(event)

        return event

    def add_attempt(self, attempt: ToolAttempt) -> ToolRetryEvent | None:
        """Record a ToolAttempt instance and return any detected ToolRetryEvent."""
        return self.record_attempt(
            tool_name=attempt.tool_name,
            operation_id=attempt.operation_id,
            success=attempt.success,
            is_retry=attempt.is_retry,
            error_message=attempt.error_message,
            turn_index=attempt.turn_index,
        )

    def detect(self, attempts: Sequence[ToolAttempt]) -> ToolRetryResult:
        """Statelessly evaluate a sequence of ToolAttempt objects for retries.

        Parameters
        ----------
        attempts:
            Sequence of tool attempts to evaluate.

        Returns
        -------
        ToolRetryResult
            Container with detected retry events and total retry count.
        """
        self.reset()
        for attempt in attempts:
            self.add_attempt(attempt)

        return ToolRetryResult(
            events=list(self._events),
            retry_count=len(self._events),
        )

    @property
    def events(self) -> list[ToolRetryEvent]:
        """List of all detected retry events so far."""
        return list(self._events)

    @property
    def retry_count(self) -> int:
        """Total number of detected retry events."""
        return len(self._events)


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


def detect_tool_retries(attempts: Sequence[ToolAttempt]) -> ToolRetryResult:
    """Convenience helper to detect retries across a sequence of tool attempts."""
    return ToolRetryDetector().detect(attempts)
