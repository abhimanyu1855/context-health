"""Context tracking — token counts and context utilization.

Records the observable state of a multi-turn conversation so that later
milestones can compute Context Health scores from reliable measurements.

Token semantics (V0.1)
-----------------------
The tracker distinguishes between **cumulative telemetry** and **estimated
current context occupancy**:

``input_tokens`` / ``output_tokens``
    Cumulative sums across all turns.  Useful for billing and throughput
    telemetry, but NOT a measure of current context-window occupancy
    (because ``input_tokens`` for each turn already includes the full
    conversation history the model re-read).

``estimated_context_tokens``
    The *latest* provider-reported ``input_tokens`` value.  In the
    Anthropic Messages API this represents the total tokens the model read
    for the most recent request — which is the closest observable proxy for
    current context-window occupancy.

    **This is an estimate/proxy and is NOT guaranteed to equal the
    provider's internal context occupancy.**  System-prompt tokens,
    caching, and internal overhead may cause the true occupancy to differ.

``context_utilization`` is computed deterministically as::

    estimated_context_tokens / context_window

where ``context_window`` is an explicitly configured value.  The result is
clamped to ``[0.0, 1.0]``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Context state snapshot
# ---------------------------------------------------------------------------


@dataclass
class ContextState:
    """Snapshot of the current conversation context.

    All fields use simple Python types so that no provider-specific SDK
    objects leak into the rest of the application.
    """

    session_id: str = ""
    message_count: int = 0

    # -- cumulative telemetry ------------------------------------------------
    input_tokens: int = 0
    """Cumulative input tokens across all turns (telemetry, not occupancy)."""

    output_tokens: int = 0
    """Cumulative output tokens across all turns."""

    # -- estimated context occupancy -----------------------------------------
    estimated_context_tokens: int = 0
    """Latest provider-reported input token count — proxy for current
    context-window occupancy.  NOT guaranteed to equal the provider's
    internal context occupancy."""

    context_window: int = 200_000
    context_utilization: float = 0.0
    """estimated_context_tokens / context_window, clamped to [0.0, 1.0]."""

    files_referenced: list[str] = field(default_factory=list)
    tool_calls: int = 0


# ---------------------------------------------------------------------------
# Context tracker
# ---------------------------------------------------------------------------


class ContextTracker:
    """Maintains the observable state of a multi-turn conversation.

    The tracker is provider-agnostic: it accepts plain integers for token
    counts (as returned by :class:`~context_health.client.CompletionResponse`)
    rather than raw SDK objects.

    Parameters
    ----------
    context_window:
        The configured model context-window size in tokens.
    session_id:
        Optional explicit session ID.  If ``None``, a UUID4 is generated.
    """

    def __init__(
        self,
        *,
        context_window: int = 200_000,
        session_id: str | None = None,
    ) -> None:
        self._session_id: str = session_id or uuid.uuid4().hex
        self._context_window: int = context_window
        self._message_count: int = 0
        self._input_tokens: int = 0
        self._output_tokens: int = 0
        self._latest_input_tokens: int = 0
        self._files_referenced: list[str] = []
        self._tool_calls: int = 0

    # -- recording -----------------------------------------------------------

    def record_user_message(self) -> None:
        """Record that a user message was added to the conversation."""
        self._message_count += 1

    def record_assistant_response(
        self,
        *,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record an assistant response with its associated token usage.

        Parameters
        ----------
        input_tokens:
            The number of input tokens reported by the provider for this turn.
            This typically represents *all* tokens the model read for this
            request, including the full conversation history.  It is used as
            the ``estimated_context_tokens`` (latest snapshot of context
            occupancy) AND accumulated into cumulative ``input_tokens``.
        output_tokens:
            The number of output tokens the model produced for this turn.
        """
        self._message_count += 1
        self._input_tokens += input_tokens
        self._output_tokens += output_tokens
        self._latest_input_tokens = input_tokens

    def record_file_reference(self, path: str) -> None:
        """Record that a file was referenced during the session.

        Duplicate paths are ignored.
        """
        if path not in self._files_referenced:
            self._files_referenced.append(path)

    def record_tool_call(self) -> None:
        """Record that a tool call occurred."""
        self._tool_calls += 1

    # -- state retrieval -----------------------------------------------------

    @property
    def state(self) -> ContextState:
        """Return the current context state as a snapshot."""
        estimated = self._latest_input_tokens
        utilization = (
            estimated / self._context_window
            if self._context_window > 0
            else 0.0
        )
        # Clamp to [0.0, 1.0].
        utilization = max(0.0, min(1.0, utilization))

        return ContextState(
            session_id=self._session_id,
            message_count=self._message_count,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            estimated_context_tokens=estimated,
            context_window=self._context_window,
            context_utilization=utilization,
            files_referenced=list(self._files_referenced),
            tool_calls=self._tool_calls,
        )
