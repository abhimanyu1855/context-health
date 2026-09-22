"""Context Health recorder for real agent telemetry.

Bridges provider-neutral agent events to the existing ``ContextTracker``.

The recorder consumes events emitted by any coding-agent adapter and maps them
to ``ContextTracker`` method calls.  It enforces lightweight session/turn
lifecycle rules and stores externally supplied measurements (relevant context
ratio, task complexity, correction detection).

There is exactly one Context Health scoring implementation
(``context_health.health``).  The recorder does NOT duplicate or modify it.

The recorder does NOT infer ``relevant_context_ratio``, ``task_complexity``,
or ``correction_detected``.  These are externally supplied measurements that
the caller provides when available.
"""

from __future__ import annotations

from context_health.agent_events import (
    AgentEvent,
    ContextUpdated,
    SessionFinished,
    SessionStarted,
    ToolCallFinished,
    ToolCallStarted,
    TurnFinished,
    TurnStarted,
)
from context_health.context import ContextState, ContextTracker


class ContextHealthRecorder:
    """Consumes provider-neutral agent events and updates a ContextTracker.

    Lifecycle rules
    ----------------
    - ``SessionStarted`` must come first; a second raises ``ValueError``.
    - ``TurnStarted`` requires prior ``SessionStarted``.
    - ``ContextUpdated``, ``ToolCallStarted``, ``ToolCallFinished`` require
      an active turn.
    - ``TurnFinished`` requires a matching active turn index.
    - ``SessionFinished`` requires prior ``SessionStarted``; after it, no
      more events are accepted.
    - A second ``SessionFinished`` raises ``ValueError``.

    Duplicate / idempotency
    ------------------------
    - Duplicate ``SessionStarted`` raises ``ValueError``.
    - Duplicate ``TurnStarted`` for the same index while a turn is active
      raises ``ValueError``.
    - Duplicate ``ToolCallFinished`` events are accepted (each counts as an
      attempt).
    - Duplicate ``SessionFinished`` raises ``ValueError``.
    """

    def __init__(self) -> None:
        self._tracker: ContextTracker | None = None
        self._session_id: str | None = None
        self._session_started: bool = False
        self._session_finished: bool = False
        self._active_turn_index: int | None = None
        self._relevant_context_ratio: float | None = None
        self._task_complexity: float | None = None
        self._model: str | None = None

    # -- public API ----------------------------------------------------------

    def record(self, event: AgentEvent) -> None:
        """Consume an agent event and update internal state.

        Parameters
        ----------
        event:
            A provider-neutral agent event.

        Raises
        ------
        ValueError
            If the event violates lifecycle rules.
        """
        if isinstance(event, SessionStarted):
            self._handle_session_started(event)
        elif isinstance(event, TurnStarted):
            self._handle_turn_started(event)
        elif isinstance(event, ContextUpdated):
            self._handle_context_updated(event)
        elif isinstance(event, ToolCallStarted):
            self._handle_tool_call_started(event)
        elif isinstance(event, ToolCallFinished):
            self._handle_tool_call_finished(event)
        elif isinstance(event, TurnFinished):
            self._handle_turn_finished(event)
        elif isinstance(event, SessionFinished):
            self._handle_session_finished(event)
        else:
            raise ValueError(f"Unknown event type: {type(event).__name__}")

    # -- properties ----------------------------------------------------------

    @property
    def tracker(self) -> ContextTracker:
        """The underlying ContextTracker.

        Raises
        ------
        ValueError
            If the session has not been started.
        """
        if self._tracker is None:
            raise ValueError("Session has not been started")
        return self._tracker

    @property
    def state(self) -> ContextState:
        """Current context state snapshot.

        Raises
        ------
        ValueError
            If the session has not been started.
        """
        return self.tracker.state

    @property
    def session_id(self) -> str | None:
        """Session identifier, or ``None`` if not yet started."""
        return self._session_id

    @property
    def is_active(self) -> bool:
        """Whether the session is active (started and not finished)."""
        return self._session_started and not self._session_finished

    @property
    def turn_index(self) -> int | None:
        """Current active turn index, or ``None`` if no turn is active."""
        return self._active_turn_index

    @property
    def relevant_context_ratio(self) -> float | None:
        """Latest externally supplied relevant context ratio, or ``None``."""
        return self._relevant_context_ratio

    @property
    def task_complexity(self) -> float | None:
        """Latest externally supplied task complexity, or ``None``."""
        return self._task_complexity

    @property
    def model(self) -> str | None:
        """Model identifier from ``SessionStarted``, or ``None``."""
        return self._model

    # -- private lifecycle helpers -------------------------------------------

    def _require_session_active(self, event: AgentEvent) -> None:
        """Raise ``ValueError`` if session is not active or session_id mismatches."""
        event_name = type(event).__name__
        if self._session_finished:
            raise ValueError(
                f"Cannot record {event_name}: session has already finished"
            )
        if not self._session_started:
            raise ValueError(
                f"Cannot record {event_name}: session has not been started"
            )
        if event.session_id != self._session_id:
            raise ValueError(
                f"Cannot record {event_name}: event session_id '{event.session_id}' "
                f"does not match recorder session_id '{self._session_id}'"
            )

    def _require_turn_active(self, event: AgentEvent) -> None:
        """Raise ``ValueError`` if no turn is active or turn_index mismatches."""
        self._require_session_active(event)
        event_name = type(event).__name__
        if self._active_turn_index is None:
            raise ValueError(
                f"Cannot record {event_name}: no turn is currently active"
            )
        if hasattr(event, "turn_index") and event.turn_index != self._active_turn_index:
            raise ValueError(
                f"Cannot record {event_name}: event turn_index {event.turn_index} "
                f"does not match active turn (active turn is {self._active_turn_index})"
            )

    # -- private event handlers ----------------------------------------------

    def _handle_session_started(self, event: SessionStarted) -> None:
        if self._session_started:
            raise ValueError(
                "Cannot start session: session has already been started"
            )
        self._session_id = event.session_id
        self._session_started = True
        self._model = event.model
        self._tracker = ContextTracker(
            context_window=event.context_window,
            session_id=event.session_id,
        )

    def _handle_turn_started(self, event: TurnStarted) -> None:
        self._require_session_active(event)
        if self._active_turn_index is not None:
            raise ValueError(
                f"Cannot start turn {event.turn_index}: "
                f"turn {self._active_turn_index} is still active"
            )
        self._active_turn_index = event.turn_index
        assert self._tracker is not None
        # Record user message without content (privacy boundary).
        # Correction detection is handled separately via the externally
        # supplied correction_detected flag on TurnFinished.
        self._tracker.record_user_message()

    def _handle_context_updated(self, event: ContextUpdated) -> None:
        self._require_turn_active(event)
        assert self._tracker is not None
        # Pass through provider-reported token measurements to the tracker.
        # input_tokens and output_tokens preserve the exact semantics of
        # ContextTracker.record_assistant_response: input_tokens serves as
        # the context occupancy proxy AND is accumulated into cumulative
        # input tokens; output_tokens is accumulated into cumulative output
        # tokens.
        self._tracker.record_assistant_response(
            input_tokens=event.input_tokens,
            output_tokens=event.output_tokens,
        )

    def _handle_tool_call_started(self, event: ToolCallStarted) -> None:
        self._require_turn_active(event)
        # ToolCallStarted is validated for lifecycle compliance.
        # The actual tool attempt is recorded on ToolCallFinished.

    def _handle_tool_call_finished(self, event: ToolCallFinished) -> None:
        self._require_turn_active(event)
        assert self._tracker is not None
        # Map error_category enum value to error_message string for the
        # tracker's ToolRetryDetector.  The enum value is a controlled
        # vocabulary, not raw error text.
        error_msg: str | None = None
        if event.error_category is not None:
            error_msg = event.error_category.value
        self._tracker.record_tool_attempt(
            tool_name=event.tool_name,
            operation_id=event.operation_id,
            success=event.success,
            is_retry=event.is_retry,
            error_message=error_msg,
            turn_index=event.turn_index,
        )

    def _handle_turn_finished(self, event: TurnFinished) -> None:
        self._require_turn_active(event)
        assert self._tracker is not None

        # Store externally supplied measurements.
        if event.relevant_context_ratio is not None:
            self._relevant_context_ratio = event.relevant_context_ratio
        if event.task_complexity is not None:
            self._task_complexity = event.task_complexity

        # Record correction if externally detected.
        if event.correction_detected is True:
            self._tracker.record_correction_event()

        self._active_turn_index = None

    def _handle_session_finished(self, event: SessionFinished) -> None:
        self._require_session_active(event)
        self._session_finished = True
        self._active_turn_index = None
