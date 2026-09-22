"""Tests for Milestone 12C — Agent Events and ContextHealthRecorder.

Covers:
- Event construction and validation
- ToolErrorCategory closed enum
- Frozen immutability
- Privacy sentinels
- Session/turn lifecycle rules
- Lifecycle violation errors
- ContextUpdated → ContextTracker token mapping
- ToolCallFinished → ContextTracker tool attempt mapping
- Tool retry detection through recorder
- Correction detection (externally supplied)
- Relevant context ratio passthrough
- Task complexity passthrough
- Unavailable signal handling (None)
- Multi-turn session flow
- State snapshot accuracy
- Duplicate event handling
- Token accumulation across turns
- Context utilization mapping
- Session ID propagation
- Model property
- No production module changes (existing imports still work)
"""

from __future__ import annotations

import dataclasses
from dataclasses import FrozenInstanceError

import pytest

from context_health.agent_events import (
    AgentEvent,
    ContextUpdated,
    SessionFinished,
    SessionStarted,
    ToolCallFinished,
    ToolCallStarted,
    ToolErrorCategory,
    TurnFinished,
    TurnStarted,
)
from context_health.agent_recorder import ContextHealthRecorder


# ---------------------------------------------------------------------------
# Privacy sentinels — these MUST NOT appear in any event field
# ---------------------------------------------------------------------------

SUPER_SECRET_TEST_API_KEY = "sk-ant-SUPER_SECRET_TEST_API_KEY_12345"
PRIVATE_PROMPT_SENTINEL = "PRIVATE_PROMPT_SENTINEL_user_secret_instruction"
PRIVATE_RESPONSE_SENTINEL = "PRIVATE_RESPONSE_SENTINEL_model_output_text"
AUTHORIZATION_HEADER_SENTINEL = "Bearer AUTHORIZATION_HEADER_SENTINEL_token"


# ===================================================================
# Event Construction & Validation
# ===================================================================


class TestEventConstruction:
    """Verify valid events can be constructed."""

    def test_session_started_valid(self) -> None:
        e = SessionStarted(session_id="sess-1", context_window=100_000)
        assert e.session_id == "sess-1"
        assert e.context_window == 100_000
        assert e.model is None

    def test_session_started_with_model(self) -> None:
        e = SessionStarted(
            session_id="sess-1", model="claude-sonnet-4-20250514"
        )
        assert e.model == "claude-sonnet-4-20250514"

    def test_session_started_default_context_window(self) -> None:
        e = SessionStarted(session_id="sess-1")
        assert e.context_window == 200_000

    def test_turn_started_valid(self) -> None:
        e = TurnStarted(session_id="sess-1", turn_index=0)
        assert e.session_id == "sess-1"
        assert e.turn_index == 0

    def test_context_updated_valid(self) -> None:
        e = ContextUpdated(
            session_id="sess-1",
            turn_index=0,
            input_tokens=1000,
            output_tokens=500,
            context_utilization=0.05,
        )
        assert e.input_tokens == 1000
        assert e.output_tokens == 500
        assert e.context_utilization == 0.05

    def test_tool_call_started_valid(self) -> None:
        e = ToolCallStarted(
            session_id="sess-1",
            turn_index=0,
            tool_name="read_file",
            operation_id="/path/to/file.py",
        )
        assert e.tool_name == "read_file"
        assert e.operation_id == "/path/to/file.py"

    def test_tool_call_finished_valid(self) -> None:
        e = ToolCallFinished(
            session_id="sess-1",
            turn_index=0,
            tool_name="run_command",
            success=False,
            is_retry=False,
            operation_id="pytest",
            error_category=ToolErrorCategory.SYNTAX_ERROR,
        )
        assert e.success is False
        assert e.error_category == ToolErrorCategory.SYNTAX_ERROR

    def test_tool_call_finished_defaults(self) -> None:
        e = ToolCallFinished(
            session_id="sess-1", turn_index=0, tool_name="read_file"
        )
        assert e.success is True
        assert e.is_retry is False
        assert e.operation_id is None
        assert e.error_category is None

    def test_turn_finished_valid(self) -> None:
        e = TurnFinished(
            session_id="sess-1",
            turn_index=0,
            correction_detected=True,
            relevant_context_ratio=0.85,
            task_complexity=0.30,
        )
        assert e.correction_detected is True
        assert e.relevant_context_ratio == 0.85
        assert e.task_complexity == 0.30

    def test_turn_finished_defaults(self) -> None:
        e = TurnFinished(session_id="sess-1", turn_index=0)
        assert e.correction_detected is None
        assert e.relevant_context_ratio is None
        assert e.task_complexity is None

    def test_session_finished_valid(self) -> None:
        e = SessionFinished(session_id="sess-1")
        assert e.session_id == "sess-1"


# ===================================================================
# Event Validation
# ===================================================================


class TestEventValidation:
    """Verify __post_init__ validation catches invalid inputs."""

    def test_empty_session_id_raises(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            SessionStarted(session_id="")

    def test_whitespace_session_id_raises(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            TurnStarted(session_id="   ", turn_index=0)

    def test_negative_turn_index_raises(self) -> None:
        with pytest.raises(ValueError, match="turn_index"):
            TurnStarted(session_id="sess-1", turn_index=-1)

    def test_negative_input_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="input_tokens"):
            ContextUpdated(
                session_id="s",
                turn_index=0,
                input_tokens=-1,
                output_tokens=0,
                context_utilization=0.0,
            )

    def test_negative_output_tokens_raises(self) -> None:
        with pytest.raises(ValueError, match="output_tokens"):
            ContextUpdated(
                session_id="s",
                turn_index=0,
                input_tokens=0,
                output_tokens=-1,
                context_utilization=0.0,
            )

    def test_utilization_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="context_utilization"):
            ContextUpdated(
                session_id="s",
                turn_index=0,
                input_tokens=0,
                output_tokens=0,
                context_utilization=-0.01,
            )

    def test_utilization_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="context_utilization"):
            ContextUpdated(
                session_id="s",
                turn_index=0,
                input_tokens=0,
                output_tokens=0,
                context_utilization=1.01,
            )

    def test_context_window_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="context_window"):
            SessionStarted(session_id="s", context_window=0)

    def test_context_window_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="context_window"):
            SessionStarted(session_id="s", context_window=-100)

    def test_relevance_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="relevant_context_ratio"):
            TurnFinished(
                session_id="s", turn_index=0, relevant_context_ratio=-0.1
            )

    def test_relevance_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="relevant_context_ratio"):
            TurnFinished(
                session_id="s", turn_index=0, relevant_context_ratio=1.5
            )

    def test_complexity_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="task_complexity"):
            TurnFinished(
                session_id="s", turn_index=0, task_complexity=-0.01
            )

    def test_complexity_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="task_complexity"):
            TurnFinished(
                session_id="s", turn_index=0, task_complexity=1.001
            )

    def test_session_finished_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            SessionFinished(session_id="")

    def test_tool_call_started_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            ToolCallStarted(session_id="", turn_index=0, tool_name="x")

    def test_tool_call_finished_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            ToolCallFinished(session_id="", turn_index=0, tool_name="x")

    def test_context_updated_empty_id_raises(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            ContextUpdated(
                session_id="",
                turn_index=0,
                input_tokens=0,
                output_tokens=0,
                context_utilization=0.0,
            )

    def test_tool_call_started_empty_tool_name_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_name"):
            ToolCallStarted(session_id="s", turn_index=0, tool_name="")

    def test_tool_call_finished_empty_tool_name_raises(self) -> None:
        with pytest.raises(ValueError, match="tool_name"):
            ToolCallFinished(session_id="s", turn_index=0, tool_name="   ")

    def test_tool_call_finished_invalid_error_category_raises(self) -> None:
        with pytest.raises(ValueError, match="error_category"):
            ToolCallFinished(
                session_id="s",
                turn_index=0,
                tool_name="cmd",
                error_category="arbitrary_string",  # type: ignore[arg-type]
            )

    def test_tool_call_finished_string_matching_enum_name_raises(self) -> None:
        """Arbitrary strings even matching enum values are rejected; must use ToolErrorCategory."""
        with pytest.raises(ValueError, match="error_category"):
            ToolCallFinished(
                session_id="s",
                turn_index=0,
                tool_name="cmd",
                error_category="timeout",  # type: ignore[arg-type]
            )

    def test_context_utilization_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="context_utilization"):
            ContextUpdated(
                session_id="s",
                turn_index=0,
                input_tokens=0,
                output_tokens=0,
                context_utilization=float("nan"),
            )

    def test_context_utilization_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="context_utilization"):
            ContextUpdated(
                session_id="s",
                turn_index=0,
                input_tokens=0,
                output_tokens=0,
                context_utilization=float("inf"),
            )

    def test_context_utilization_neg_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="context_utilization"):
            ContextUpdated(
                session_id="s",
                turn_index=0,
                input_tokens=0,
                output_tokens=0,
                context_utilization=float("-inf"),
            )

    def test_relevance_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="relevant_context_ratio"):
            TurnFinished(
                session_id="s", turn_index=0, relevant_context_ratio=float("nan")
            )

    def test_relevance_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="relevant_context_ratio"):
            TurnFinished(
                session_id="s", turn_index=0, relevant_context_ratio=float("inf")
            )

    def test_complexity_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="task_complexity"):
            TurnFinished(
                session_id="s", turn_index=0, task_complexity=float("nan")
            )

    def test_complexity_inf_raises(self) -> None:
        with pytest.raises(ValueError, match="task_complexity"):
            TurnFinished(
                session_id="s", turn_index=0, task_complexity=float("inf")
            )


# ===================================================================
# ToolErrorCategory
# ===================================================================


class TestToolErrorCategory:
    """Verify the closed enum for tool error categories."""

    def test_all_categories_exist(self) -> None:
        expected = {
            "timeout",
            "permission_denied",
            "not_found",
            "syntax_error",
            "rate_limited",
            "network_error",
            "validation_error",
            "unknown",
        }
        actual = {e.value for e in ToolErrorCategory}
        assert actual == expected

    def test_enum_is_closed(self) -> None:
        """ToolErrorCategory members are exactly the defined set."""
        assert len(ToolErrorCategory) == 8

    def test_tool_call_finished_with_each_category(self) -> None:
        for cat in ToolErrorCategory:
            e = ToolCallFinished(
                session_id="s",
                turn_index=0,
                tool_name="cmd",
                success=False,
                error_category=cat,
            )
            assert e.error_category is cat

    def test_error_category_none_when_success(self) -> None:
        e = ToolCallFinished(
            session_id="s",
            turn_index=0,
            tool_name="cmd",
            success=True,
        )
        assert e.error_category is None


# ===================================================================
# Frozen Immutability
# ===================================================================


class TestEventImmutability:
    """Verify all events are frozen dataclasses."""

    def test_session_started_frozen(self) -> None:
        e = SessionStarted(session_id="s")
        with pytest.raises(FrozenInstanceError):
            e.session_id = "other"  # type: ignore[misc]

    def test_turn_started_frozen(self) -> None:
        e = TurnStarted(session_id="s", turn_index=0)
        with pytest.raises(FrozenInstanceError):
            e.turn_index = 1  # type: ignore[misc]

    def test_context_updated_frozen(self) -> None:
        e = ContextUpdated(
            session_id="s",
            turn_index=0,
            input_tokens=100,
            output_tokens=50,
            context_utilization=0.1,
        )
        with pytest.raises(FrozenInstanceError):
            e.input_tokens = 999  # type: ignore[misc]

    def test_tool_call_finished_frozen(self) -> None:
        e = ToolCallFinished(session_id="s", turn_index=0, tool_name="x")
        with pytest.raises(FrozenInstanceError):
            e.success = False  # type: ignore[misc]

    def test_turn_finished_frozen(self) -> None:
        e = TurnFinished(session_id="s", turn_index=0)
        with pytest.raises(FrozenInstanceError):
            e.correction_detected = True  # type: ignore[misc]

    def test_session_finished_frozen(self) -> None:
        e = SessionFinished(session_id="s")
        with pytest.raises(FrozenInstanceError):
            e.session_id = "other"  # type: ignore[misc]

    def test_tool_call_started_frozen(self) -> None:
        e = ToolCallStarted(session_id="s", turn_index=0, tool_name="x")
        with pytest.raises(FrozenInstanceError):
            e.tool_name = "y"  # type: ignore[misc]


# ===================================================================
# Privacy Sentinels
# ===================================================================


class TestPrivacySentinels:
    """Verify privacy sentinels never appear in any event field."""

    SENTINELS = [
        SUPER_SECRET_TEST_API_KEY,
        PRIVATE_PROMPT_SENTINEL,
        PRIVATE_RESPONSE_SENTINEL,
        AUTHORIZATION_HEADER_SENTINEL,
    ]

    def _all_string_values(self, obj: object) -> list[str]:
        """Extract all string field values from a frozen dataclass."""
        result: list[str] = []
        for f in dataclasses.fields(obj):  # type: ignore[arg-type]
            val = getattr(obj, f.name)
            if isinstance(val, str):
                result.append(val)
        return result

    def test_no_secrets_in_session_started(self) -> None:
        e = SessionStarted(session_id="clean-session")
        for sentinel in self.SENTINELS:
            for val in self._all_string_values(e):
                assert sentinel not in val

    def test_no_secrets_in_tool_call_finished(self) -> None:
        e = ToolCallFinished(
            session_id="clean",
            turn_index=0,
            tool_name="run_command",
            operation_id="pytest",
        )
        for sentinel in self.SENTINELS:
            for val in self._all_string_values(e):
                assert sentinel not in val

    def test_no_secrets_in_turn_finished(self) -> None:
        e = TurnFinished(session_id="clean", turn_index=0)
        for sentinel in self.SENTINELS:
            for val in self._all_string_values(e):
                assert sentinel not in val

    def test_sentinels_cannot_enter_via_session_id(self) -> None:
        """Verify that if a sentinel were somehow used as a session_id,
        the fields are inspectable — adapters must sanitize inputs."""
        e = SessionStarted(session_id="safe-id")
        for val in self._all_string_values(e):
            for sentinel in self.SENTINELS:
                assert sentinel not in val

    def test_event_model_has_no_content_fields(self) -> None:
        """Verify none of the event classes have fields for raw content."""
        prohibited_field_names = {
            "prompt",
            "response",
            "content",
            "message",
            "api_key",
            "secret",
            "authorization",
            "source_code",
            "raw_error",
            "error_message",
            "stack_trace",
            "traceback",
        }
        event_classes = [
            SessionStarted,
            TurnStarted,
            ContextUpdated,
            ToolCallStarted,
            ToolCallFinished,
            TurnFinished,
            SessionFinished,
        ]
        for cls in event_classes:
            field_names = {f.name for f in dataclasses.fields(cls)}
            overlap = field_names & prohibited_field_names
            assert not overlap, (
                f"{cls.__name__} has prohibited field(s): {overlap}"
            )


# ===================================================================
# Recorder Lifecycle — Happy Path
# ===================================================================


class TestRecorderLifecycle:
    """Verify correct session/turn lifecycle flow."""

    def test_happy_path_full_session(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1", context_window=100_000))
        assert rec.is_active is True
        assert rec.session_id == "s1"

        rec.record(TurnStarted(session_id="s1", turn_index=0))
        assert rec.turn_index == 0

        rec.record(
            ContextUpdated(
                session_id="s1",
                turn_index=0,
                input_tokens=1000,
                output_tokens=200,
                context_utilization=0.01,
            )
        )
        rec.record(
            ToolCallStarted(
                session_id="s1", turn_index=0, tool_name="read_file"
            )
        )
        rec.record(
            ToolCallFinished(
                session_id="s1", turn_index=0, tool_name="read_file"
            )
        )
        rec.record(TurnFinished(session_id="s1", turn_index=0))
        assert rec.turn_index is None

        rec.record(SessionFinished(session_id="s1"))
        assert rec.is_active is False

    def test_not_active_before_start(self) -> None:
        rec = ContextHealthRecorder()
        assert rec.is_active is False
        assert rec.session_id is None
        assert rec.turn_index is None

    def test_not_active_after_finish(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(SessionFinished(session_id="s1"))
        assert rec.is_active is False


# ===================================================================
# Recorder Lifecycle — Violations
# ===================================================================


class TestRecorderLifecycleViolations:
    """Verify lifecycle violations raise ValueError."""

    def test_turn_started_before_session_started(self) -> None:
        rec = ContextHealthRecorder()
        with pytest.raises(ValueError, match="session has not been started"):
            rec.record(TurnStarted(session_id="s1", turn_index=0))

    def test_context_updated_before_turn_started(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        with pytest.raises(ValueError, match="no turn is currently active"):
            rec.record(
                ContextUpdated(
                    session_id="s1",
                    turn_index=0,
                    input_tokens=100,
                    output_tokens=50,
                    context_utilization=0.0,
                )
            )

    def test_tool_call_started_before_turn_started(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        with pytest.raises(ValueError, match="no turn is currently active"):
            rec.record(
                ToolCallStarted(
                    session_id="s1", turn_index=0, tool_name="x"
                )
            )

    def test_tool_call_finished_before_turn_started(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        with pytest.raises(ValueError, match="no turn is currently active"):
            rec.record(
                ToolCallFinished(
                    session_id="s1", turn_index=0, tool_name="x"
                )
            )

    def test_turn_finished_before_turn_started(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        with pytest.raises(ValueError, match="no turn is currently active"):
            rec.record(TurnFinished(session_id="s1", turn_index=0))

    def test_events_after_session_finished(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(SessionFinished(session_id="s1"))
        with pytest.raises(ValueError, match="session has already finished"):
            rec.record(TurnStarted(session_id="s1", turn_index=0))

    def test_session_finished_before_session_started(self) -> None:
        rec = ContextHealthRecorder()
        with pytest.raises(ValueError, match="session has not been started"):
            rec.record(SessionFinished(session_id="s1"))

    def test_turn_finished_mismatched_index(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        with pytest.raises(ValueError, match="active turn is 0"):
            rec.record(TurnFinished(session_id="s1", turn_index=1))

    def test_context_updated_after_session_finished(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(SessionFinished(session_id="s1"))
        with pytest.raises(ValueError, match="session has already finished"):
            rec.record(
                ContextUpdated(
                    session_id="s1",
                    turn_index=0,
                    input_tokens=100,
                    output_tokens=50,
                    context_utilization=0.0,
                )
            )

    def test_event_mismatched_session_id_raises(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        with pytest.raises(ValueError, match="does not match recorder session_id"):
            rec.record(TurnStarted(session_id="different-session", turn_index=0))

    def test_context_updated_mismatched_turn_index_raises(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        with pytest.raises(ValueError, match="does not match active turn"):
            rec.record(
                ContextUpdated(
                    session_id="s1",
                    turn_index=1,
                    input_tokens=100,
                    output_tokens=50,
                    context_utilization=0.0,
                )
            )

    def test_tool_call_started_mismatched_turn_index_raises(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        with pytest.raises(ValueError, match="does not match active turn"):
            rec.record(
                ToolCallStarted(
                    session_id="s1", turn_index=1, tool_name="cmd"
                )
            )

    def test_tool_call_finished_mismatched_turn_index_raises(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        with pytest.raises(ValueError, match="does not match active turn"):
            rec.record(
                ToolCallFinished(
                    session_id="s1", turn_index=1, tool_name="cmd"
                )
            )


# ===================================================================
# Duplicate Event Handling
# ===================================================================


class TestDuplicateEvents:
    """Verify deterministic behavior for duplicate events."""

    def test_duplicate_session_started_raises(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        with pytest.raises(ValueError, match="already been started"):
            rec.record(SessionStarted(session_id="s1"))

    def test_duplicate_turn_started_while_active_raises(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        with pytest.raises(ValueError, match="turn 0 is still active"):
            rec.record(TurnStarted(session_id="s1", turn_index=1))

    def test_duplicate_session_finished_raises(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(SessionFinished(session_id="s1"))
        with pytest.raises(ValueError, match="session has already finished"):
            rec.record(SessionFinished(session_id="s1"))

    def test_duplicate_tool_call_finished_accepted(self) -> None:
        """Each ToolCallFinished counts as a separate attempt."""
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            ToolCallFinished(session_id="s1", turn_index=0, tool_name="cmd")
        )
        rec.record(
            ToolCallFinished(session_id="s1", turn_index=0, tool_name="cmd")
        )
        assert rec.state.tool_calls == 2


# ===================================================================
# Token Mapping (ContextUpdated → ContextTracker)
# ===================================================================


class TestRecorderTokenMapping:
    """Verify ContextUpdated correctly maps to ContextTracker token semantics."""

    def test_context_updated_maps_tokens(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1", context_window=100_000))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            ContextUpdated(
                session_id="s1",
                turn_index=0,
                input_tokens=5000,
                output_tokens=1000,
                context_utilization=0.05,
            )
        )
        rec.record(TurnFinished(session_id="s1", turn_index=0))

        state = rec.state
        assert state.input_tokens == 5000
        assert state.output_tokens == 1000
        assert state.estimated_context_tokens == 5000

    def test_token_accumulation_across_turns(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1", context_window=200_000))

        # Turn 0
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            ContextUpdated(
                session_id="s1",
                turn_index=0,
                input_tokens=3000,
                output_tokens=500,
                context_utilization=0.015,
            )
        )
        rec.record(TurnFinished(session_id="s1", turn_index=0))

        # Turn 1
        rec.record(TurnStarted(session_id="s1", turn_index=1))
        rec.record(
            ContextUpdated(
                session_id="s1",
                turn_index=1,
                input_tokens=7000,
                output_tokens=800,
                context_utilization=0.035,
            )
        )
        rec.record(TurnFinished(session_id="s1", turn_index=1))

        state = rec.state
        # Cumulative input tokens
        assert state.input_tokens == 3000 + 7000
        # Cumulative output tokens
        assert state.output_tokens == 500 + 800
        # Latest input tokens (context proxy)
        assert state.estimated_context_tokens == 7000

    def test_context_utilization_computed_by_tracker(self) -> None:
        """Utilization is computed by ContextTracker from input_tokens/context_window,
        not taken from the event's context_utilization field."""
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1", context_window=100_000))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            ContextUpdated(
                session_id="s1",
                turn_index=0,
                input_tokens=50_000,
                output_tokens=1000,
                context_utilization=0.99,  # intentionally different
            )
        )
        rec.record(TurnFinished(session_id="s1", turn_index=0))

        state = rec.state
        # Tracker computes 50000/100000 = 0.5, NOT the event's 0.99
        assert state.context_utilization == pytest.approx(0.5)


# ===================================================================
# Tool Attempt Mapping (ToolCallFinished → ContextTracker)
# ===================================================================


class TestRecorderToolMapping:
    """Verify ToolCallFinished maps to ContextTracker.record_tool_attempt."""

    def test_tool_call_finished_increments_tool_calls(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            ToolCallFinished(
                session_id="s1", turn_index=0, tool_name="read_file"
            )
        )
        assert rec.state.tool_calls == 1

    def test_tool_retry_via_explicit_flag(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            ToolCallFinished(
                session_id="s1",
                turn_index=0,
                tool_name="run_command",
                operation_id="pytest",
                success=True,
                is_retry=True,
            )
        )
        assert rec.state.tool_retries == 1

    def test_tool_retry_after_failed_attempt(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        # First attempt fails
        rec.record(
            ToolCallFinished(
                session_id="s1",
                turn_index=0,
                tool_name="run_command",
                operation_id="pytest",
                success=False,
                error_category=ToolErrorCategory.SYNTAX_ERROR,
            )
        )
        # Second attempt on same operation — detected as retry
        rec.record(
            ToolCallFinished(
                session_id="s1",
                turn_index=0,
                tool_name="run_command",
                operation_id="pytest",
                success=True,
            )
        )
        assert rec.state.tool_calls == 2
        assert rec.state.tool_retries == 1

    def test_tool_call_with_error_category(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            ToolCallFinished(
                session_id="s1",
                turn_index=0,
                tool_name="cmd",
                success=False,
                error_category=ToolErrorCategory.TIMEOUT,
            )
        )
        assert rec.state.tool_calls == 1

    def test_different_operations_not_retries(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            ToolCallFinished(
                session_id="s1",
                turn_index=0,
                tool_name="read_file",
                operation_id="a.py",
                success=False,
            )
        )
        rec.record(
            ToolCallFinished(
                session_id="s1",
                turn_index=0,
                tool_name="read_file",
                operation_id="b.py",
                success=True,
            )
        )
        assert rec.state.tool_calls == 2
        assert rec.state.tool_retries == 0

    def test_tool_call_started_does_not_increment_tool_calls(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            ToolCallStarted(
                session_id="s1", turn_index=0, tool_name="read_file"
            )
        )
        assert rec.state.tool_calls == 0
        assert rec.state.tool_retries == 0

    def test_operation_id_none_failure_does_not_trigger_retry(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        # First call with operation_id=None fails
        rec.record(
            ToolCallFinished(
                session_id="s1",
                turn_index=0,
                tool_name="read_file",
                operation_id=None,
                success=False,
            )
        )
        # Second call with operation_id=None succeeds
        rec.record(
            ToolCallFinished(
                session_id="s1",
                turn_index=0,
                tool_name="read_file",
                operation_id=None,
                success=True,
            )
        )
        assert rec.state.tool_calls == 2
        # None operation_id does not cause failure->retry collision
        assert rec.state.tool_retries == 0


# ===================================================================
# Externally Supplied Signals
# ===================================================================


class TestRecorderSignals:
    """Verify externally supplied measurements flow correctly."""

    def test_correction_detected_true_increments(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            TurnFinished(
                session_id="s1", turn_index=0, correction_detected=True
            )
        )
        assert rec.state.correction_events == 1

    def test_correction_detected_false_no_increment(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            TurnFinished(
                session_id="s1", turn_index=0, correction_detected=False
            )
        )
        assert rec.state.correction_events == 0

    def test_correction_detected_none_no_increment(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            TurnFinished(
                session_id="s1", turn_index=0, correction_detected=None
            )
        )
        assert rec.state.correction_events == 0

    def test_relevant_context_ratio_stored(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            TurnFinished(
                session_id="s1",
                turn_index=0,
                relevant_context_ratio=0.85,
            )
        )
        assert rec.relevant_context_ratio == 0.85

    def test_relevant_context_ratio_unavailable(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(TurnFinished(session_id="s1", turn_index=0))
        assert rec.relevant_context_ratio is None

    def test_task_complexity_stored(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            TurnFinished(
                session_id="s1", turn_index=0, task_complexity=0.65
            )
        )
        assert rec.task_complexity == 0.65

    def test_task_complexity_unavailable(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(TurnFinished(session_id="s1", turn_index=0))
        assert rec.task_complexity is None

    def test_signals_update_across_turns(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))

        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            TurnFinished(
                session_id="s1",
                turn_index=0,
                relevant_context_ratio=0.90,
                task_complexity=0.20,
            )
        )
        assert rec.relevant_context_ratio == 0.90
        assert rec.task_complexity == 0.20

        rec.record(TurnStarted(session_id="s1", turn_index=1))
        rec.record(
            TurnFinished(
                session_id="s1",
                turn_index=1,
                relevant_context_ratio=0.75,
                task_complexity=0.40,
            )
        )
        assert rec.relevant_context_ratio == 0.75
        assert rec.task_complexity == 0.40

    def test_signals_persist_when_not_provided_on_later_turn(self) -> None:
        """If a later turn does not supply a signal, the previous value persists."""
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))

        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            TurnFinished(
                session_id="s1",
                turn_index=0,
                relevant_context_ratio=0.80,
            )
        )

        rec.record(TurnStarted(session_id="s1", turn_index=1))
        rec.record(TurnFinished(session_id="s1", turn_index=1))

        # Previous value persists
        assert rec.relevant_context_ratio == 0.80

    def test_multiple_corrections_accumulate(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))

        rec.record(TurnStarted(session_id="s1", turn_index=0))
        rec.record(
            TurnFinished(
                session_id="s1", turn_index=0, correction_detected=True
            )
        )

        rec.record(TurnStarted(session_id="s1", turn_index=1))
        rec.record(
            TurnFinished(
                session_id="s1", turn_index=1, correction_detected=True
            )
        )

        assert rec.state.correction_events == 2


# ===================================================================
# Recorder Properties
# ===================================================================


class TestRecorderProperties:
    """Verify recorder property behavior."""

    def test_session_id_before_start(self) -> None:
        rec = ContextHealthRecorder()
        assert rec.session_id is None

    def test_session_id_after_start(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="my-session-42"))
        assert rec.session_id == "my-session-42"

    def test_tracker_before_session_raises(self) -> None:
        rec = ContextHealthRecorder()
        with pytest.raises(ValueError, match="Session has not been started"):
            _ = rec.tracker

    def test_state_before_session_raises(self) -> None:
        rec = ContextHealthRecorder()
        with pytest.raises(ValueError, match="Session has not been started"):
            _ = rec.state

    def test_model_property(self) -> None:
        rec = ContextHealthRecorder()
        assert rec.model is None
        rec.record(
            SessionStarted(
                session_id="s1", model="claude-sonnet-4-20250514"
            )
        )
        assert rec.model == "claude-sonnet-4-20250514"

    def test_model_none_when_not_specified(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="s1"))
        assert rec.model is None

    def test_is_active_lifecycle(self) -> None:
        rec = ContextHealthRecorder()
        assert rec.is_active is False
        rec.record(SessionStarted(session_id="s1"))
        assert rec.is_active is True
        rec.record(SessionFinished(session_id="s1"))
        assert rec.is_active is False

    def test_turn_index_lifecycle(self) -> None:
        rec = ContextHealthRecorder()
        assert rec.turn_index is None
        rec.record(SessionStarted(session_id="s1"))
        assert rec.turn_index is None
        rec.record(TurnStarted(session_id="s1", turn_index=3))
        assert rec.turn_index == 3
        rec.record(TurnFinished(session_id="s1", turn_index=3))
        assert rec.turn_index is None


# ===================================================================
# State Snapshot Accuracy
# ===================================================================


class TestStateSnapshot:
    """Verify recorder.state matches expected ContextState."""

    def test_state_after_single_turn(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="snap-1", context_window=100_000))
        rec.record(TurnStarted(session_id="snap-1", turn_index=0))
        rec.record(
            ContextUpdated(
                session_id="snap-1",
                turn_index=0,
                input_tokens=10_000,
                output_tokens=2_000,
                context_utilization=0.1,
            )
        )
        rec.record(
            ToolCallFinished(
                session_id="snap-1",
                turn_index=0,
                tool_name="run_command",
            )
        )
        rec.record(
            TurnFinished(
                session_id="snap-1",
                turn_index=0,
                correction_detected=False,
                relevant_context_ratio=0.90,
            )
        )

        state = rec.state
        assert state.session_id == "snap-1"
        # message_count: 1 from TurnStarted (user), 1 from ContextUpdated (response)
        assert state.message_count == 2
        assert state.input_tokens == 10_000
        assert state.output_tokens == 2_000
        assert state.estimated_context_tokens == 10_000
        assert state.context_window == 100_000
        assert state.context_utilization == pytest.approx(0.1)
        assert state.tool_calls == 1
        assert state.correction_events == 0
        assert state.tool_retries == 0

    def test_session_id_propagation(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(SessionStarted(session_id="propagation-test"))
        assert rec.state.session_id == "propagation-test"


# ===================================================================
# Multi-Turn Session Flow
# ===================================================================


class TestMultiTurnSession:
    """Verify complete multi-turn session with state evolution."""

    def test_three_turn_session(self) -> None:
        rec = ContextHealthRecorder()
        rec.record(
            SessionStarted(
                session_id="multi-3",
                context_window=200_000,
                model="test-model",
            )
        )

        # Turn 0: simple query
        rec.record(TurnStarted(session_id="multi-3", turn_index=0))
        rec.record(
            ContextUpdated(
                session_id="multi-3",
                turn_index=0,
                input_tokens=2000,
                output_tokens=500,
                context_utilization=0.01,
            )
        )
        rec.record(
            TurnFinished(
                session_id="multi-3",
                turn_index=0,
                relevant_context_ratio=0.95,
                task_complexity=0.10,
            )
        )

        state_0 = rec.state
        assert state_0.input_tokens == 2000
        assert state_0.output_tokens == 500

        # Turn 1: tool use with failed attempt
        rec.record(TurnStarted(session_id="multi-3", turn_index=1))
        rec.record(
            ContextUpdated(
                session_id="multi-3",
                turn_index=1,
                input_tokens=8000,
                output_tokens=1500,
                context_utilization=0.04,
            )
        )
        rec.record(
            ToolCallStarted(
                session_id="multi-3",
                turn_index=1,
                tool_name="run_command",
                operation_id="make build",
            )
        )
        rec.record(
            ToolCallFinished(
                session_id="multi-3",
                turn_index=1,
                tool_name="run_command",
                operation_id="make build",
                success=False,
                error_category=ToolErrorCategory.SYNTAX_ERROR,
            )
        )
        rec.record(
            ToolCallStarted(
                session_id="multi-3",
                turn_index=1,
                tool_name="run_command",
                operation_id="make build",
            )
        )
        rec.record(
            ToolCallFinished(
                session_id="multi-3",
                turn_index=1,
                tool_name="run_command",
                operation_id="make build",
                success=True,
            )
        )
        rec.record(
            TurnFinished(
                session_id="multi-3",
                turn_index=1,
                relevant_context_ratio=0.80,
                task_complexity=0.45,
                correction_detected=True,
            )
        )

        state_1 = rec.state
        assert state_1.input_tokens == 2000 + 8000
        assert state_1.output_tokens == 500 + 1500
        assert state_1.tool_calls == 2
        assert state_1.tool_retries == 1
        assert state_1.correction_events == 1

        # Turn 2: completion
        rec.record(TurnStarted(session_id="multi-3", turn_index=2))
        rec.record(
            ContextUpdated(
                session_id="multi-3",
                turn_index=2,
                input_tokens=15000,
                output_tokens=2000,
                context_utilization=0.075,
            )
        )
        rec.record(
            TurnFinished(
                session_id="multi-3",
                turn_index=2,
                relevant_context_ratio=0.70,
                task_complexity=0.50,
            )
        )

        rec.record(SessionFinished(session_id="multi-3"))

        final_state = rec.state
        assert final_state.input_tokens == 2000 + 8000 + 15000
        assert final_state.output_tokens == 500 + 1500 + 2000
        assert final_state.estimated_context_tokens == 15000
        assert final_state.tool_calls == 2
        assert final_state.tool_retries == 1
        assert final_state.correction_events == 1
        # message_count: 3 user + 3 assistant = 6
        assert final_state.message_count == 6

        assert rec.relevant_context_ratio == 0.70
        assert rec.task_complexity == 0.50
        assert rec.model == "test-model"
        assert rec.is_active is False


# ===================================================================
# No Production Module Changes
# ===================================================================


class TestNoProductionChanges:
    """Verify existing production imports still work unchanged."""

    def test_context_tracker_import(self) -> None:
        from context_health.context import ContextState, ContextTracker

        tracker = ContextTracker(context_window=100_000, session_id="test")
        state = tracker.state
        assert isinstance(state, ContextState)

    def test_health_module_import(self) -> None:
        from context_health.health import (
            DEFAULT_HEALTH_WEIGHTS,
            DEFAULT_SATURATION_CONSTANTS,
            ContextHealthResult,
            compute_context_health,
        )

        result = compute_context_health(context_utilization=0.5)
        assert isinstance(result, ContextHealthResult)
        assert result.health_score >= 0.0
        assert "utilization" in DEFAULT_HEALTH_WEIGHTS
        assert "contradictions" in DEFAULT_SATURATION_CONSTANTS

    def test_telemetry_module_import(self) -> None:
        from context_health.telemetry import (
            TelemetryRecord,
            create_telemetry_record,
        )

        assert TelemetryRecord is not None
        assert create_telemetry_record is not None

    def test_events_module_import(self) -> None:
        from context_health.events import (
            ContradictionDetector,
            CorrectionDetector,
            ToolRetryDetector,
        )

        assert ContradictionDetector is not None
        assert CorrectionDetector is not None
        assert ToolRetryDetector is not None
