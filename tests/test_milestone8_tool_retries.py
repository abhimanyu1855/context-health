"""Milestone 8 tests — tool retry tracking.

ALL tests are deterministic. ZERO network/API/LLM calls are made.
Tests verify positive/negative retry detection, false-positive protection,
stateful and stateless tracking, ContextTracker integration, immutability,
and sequence protocol conformance.
"""

from __future__ import annotations

import pytest

from context_health.context import ContextState, ContextTracker
from context_health.tool_retries import (
    ToolAttempt,
    ToolRetryDetector,
    ToolRetryEvent,
    ToolRetryResult,
    detect_tool_retries,
)


# ---------------------------------------------------------------------------
# 1. Positive Cases — Explicit and Failure-Driven Retries
# ---------------------------------------------------------------------------


class TestPositiveToolRetries:
    """Verify that genuine tool retries are accurately detected."""

    def test_explicit_retry_flag(self) -> None:
        detector = ToolRetryDetector()
        event = detector.record_attempt(
            tool_name="read_file",
            operation_id="src/main.py",
            success=True,
            is_retry=True,
        )
        assert event is not None
        assert event.tool_name == "read_file"
        assert event.operation_id == "src/main.py"
        assert event.reason == "explicit_retry"
        assert detector.retry_count == 1

    def test_failed_attempt_followed_by_retry(self) -> None:
        detector = ToolRetryDetector()

        # Attempt 1: fails
        ev1 = detector.record_attempt(
            tool_name="run_command",
            operation_id="pytest",
            success=False,
            error_message="1 test failed",
        )
        assert ev1 is None
        assert detector.retry_count == 0

        # Attempt 2: retried
        ev2 = detector.record_attempt(
            tool_name="run_command",
            operation_id="pytest",
            success=True,
        )
        assert ev2 is not None
        assert ev2.tool_name == "run_command"
        assert ev2.operation_id == "pytest"
        assert ev2.attempt_number == 2
        assert ev2.reason == "failed_previous_attempt"
        assert detector.retry_count == 1

    def test_multiple_consecutive_retries(self) -> None:
        detector = ToolRetryDetector()

        # Attempt 1: fail
        detector.record_attempt("bash", operation_id="cargo build", success=False)
        assert detector.retry_count == 0

        # Attempt 2: retry, fail again
        ev2 = detector.record_attempt("bash", operation_id="cargo build", success=False)
        assert ev2 is not None
        assert ev2.attempt_number == 2
        assert detector.retry_count == 1

        # Attempt 3: retry, succeed
        ev3 = detector.record_attempt("bash", operation_id="cargo build", success=True)
        assert ev3 is not None
        assert ev3.attempt_number == 3
        assert detector.retry_count == 2

    def test_multiple_different_tools_retried(self) -> None:
        attempts = [
            ToolAttempt(tool_name="tool_a", operation_id="op1", success=False),
            ToolAttempt(tool_name="tool_b", operation_id="op2", success=False),
            ToolAttempt(tool_name="tool_a", operation_id="op1", success=True),  # retry of tool_a
            ToolAttempt(tool_name="tool_b", operation_id="op2", success=True),  # retry of tool_b
        ]
        result = detect_tool_retries(attempts)
        assert result.retry_count == 2
        assert len(result.events) == 2
        assert result.events[0].tool_name == "tool_a"
        assert result.events[1].tool_name == "tool_b"


# ---------------------------------------------------------------------------
# 2. Negative Cases — False-Positive Protection
# ---------------------------------------------------------------------------


class TestNegativeToolRetries:
    """Verify that ordinary tool calls and distinct operations are NOT flagged as retries."""

    def test_same_tool_different_operations(self) -> None:
        attempts = [
            ToolAttempt(tool_name="read_file", operation_id="a.py", success=True),
            ToolAttempt(tool_name="read_file", operation_id="b.py", success=True),
            ToolAttempt(tool_name="read_file", operation_id="c.py", success=True),
        ]
        result = detect_tool_retries(attempts)
        assert result.retry_count == 0
        assert len(result.events) == 0

    def test_same_tool_different_operations_after_failure(self) -> None:
        """A failure in operation A does not make an attempt on operation B a retry."""
        attempts = [
            ToolAttempt(tool_name="read_file", operation_id="a.py", success=False),
            ToolAttempt(tool_name="read_file", operation_id="b.py", success=True),
        ]
        result = detect_tool_retries(attempts)
        assert result.retry_count == 0

    def test_consecutive_successful_independent_calls(self) -> None:
        """Repeated successful independent calls are NOT automatically retries."""
        attempts = [
            ToolAttempt(tool_name="search", operation_id="redis", success=True),
            ToolAttempt(tool_name="search", operation_id="redis", success=True),
        ]
        result = detect_tool_retries(attempts)
        assert result.retry_count == 0

    def test_different_tools_same_operation_id(self) -> None:
        attempts = [
            ToolAttempt(tool_name="tool_one", operation_id="test", success=False),
            ToolAttempt(tool_name="tool_two", operation_id="test", success=True),
        ]
        result = detect_tool_retries(attempts)
        assert result.retry_count == 0

    def test_no_operation_id_without_explicit_retry(self) -> None:
        """Anonymous tool calls without operation_id and without is_retry=True are not retries."""
        attempts = [
            ToolAttempt(tool_name="generic_tool", success=False),
            ToolAttempt(tool_name="generic_tool", success=True),
        ]
        result = detect_tool_retries(attempts)
        assert result.retry_count == 0

    def test_empty_sequence(self) -> None:
        result = detect_tool_retries([])
        assert result.retry_count == 0
        assert len(result.events) == 0


# ---------------------------------------------------------------------------
# 3. Stateful Incremental Detection & Reset
# ---------------------------------------------------------------------------


class TestStatefulToolRetryDetector:
    """Verify stateful ToolRetryDetector operation."""

    def test_incremental_workflow(self) -> None:
        detector = ToolRetryDetector()

        # Step 1: success
        assert detector.record_attempt("run_test", operation_id="test_a", success=True) is None
        assert detector.retry_count == 0

        # Step 2: failure
        assert detector.record_attempt("run_test", operation_id="test_b", success=False) is None
        assert detector.retry_count == 0

        # Step 3: retry test_b
        ev = detector.record_attempt("run_test", operation_id="test_b", success=True)
        assert ev is not None
        assert ev.attempt_number == 2
        assert detector.retry_count == 1

        # Reset
        detector.reset()
        assert detector.retry_count == 0
        assert detector.events == []


# ---------------------------------------------------------------------------
# 4. ContextTracker & ContextState Integration
# ---------------------------------------------------------------------------


class TestContextTrackerToolRetries:
    """Verify tool retry integration in ContextTracker and ContextState."""

    def test_context_state_defaults_and_custom(self) -> None:
        state = ContextState()
        assert state.tool_retries == 0
        assert state.tool_calls == 0

        custom = ContextState(tool_retries=3, tool_calls=10)
        assert custom.tool_retries == 3
        assert custom.tool_calls == 10

    def test_tracker_record_tool_call_does_not_increment_retries(self) -> None:
        tracker = ContextTracker()
        tracker.record_tool_call()
        tracker.record_tool_call()
        assert tracker.state.tool_calls == 2
        assert tracker.state.tool_retries == 0

    def test_tracker_record_tool_retry_explicit(self) -> None:
        tracker = ContextTracker()
        tracker.record_tool_retry()
        tracker.record_tool_retry()
        assert tracker.state.tool_retries == 2

    def test_tracker_record_tool_attempt_auto_detects(self) -> None:
        tracker = ContextTracker()

        # Call 1: tool attempt fails
        ev1 = tracker.record_tool_attempt("run_cmd", operation_id="npm test", success=False)
        assert ev1 is None
        assert tracker.state.tool_calls == 1
        assert tracker.state.tool_retries == 0

        # Call 2: tool attempt retry succeeds
        ev2 = tracker.record_tool_attempt("run_cmd", operation_id="npm test", success=True)
        assert ev2 is not None
        assert tracker.state.tool_calls == 2
        assert tracker.state.tool_retries == 1

        # Call 3: ordinary independent tool call
        ev3 = tracker.record_tool_attempt("read_file", operation_id="package.json", success=True)
        assert ev3 is None
        assert tracker.state.tool_calls == 3
        assert tracker.state.tool_retries == 1

    def test_tracker_record_tool_attempt_explicit_retry(self) -> None:
        tracker = ContextTracker()
        ev = tracker.record_tool_attempt("custom_tool", is_retry=True)
        assert ev is not None
        assert tracker.state.tool_calls == 1
        assert tracker.state.tool_retries == 1


# ---------------------------------------------------------------------------
# 5. Data Model & Sequence Protocol
# ---------------------------------------------------------------------------


class TestDataModelAndSequenceProtocol:
    """Verify immutability and sequence protocol for tool retry objects."""

    def test_tool_attempt_immutability(self) -> None:
        attempt = ToolAttempt(tool_name="bash", operation_id="ls")
        with pytest.raises(AttributeError):
            attempt.tool_name = "zsh"  # type: ignore[misc]

    def test_tool_retry_event_immutability(self) -> None:
        event = ToolRetryEvent(tool_name="bash", operation_id="ls", attempt_number=2)
        with pytest.raises(AttributeError):
            event.attempt_number = 3  # type: ignore[misc]

    def test_tool_retry_result_sequence_protocol(self) -> None:
        attempts = [
            ToolAttempt("bash", operation_id="test", success=False),
            ToolAttempt("bash", operation_id="test", success=True),
            ToolAttempt("read_file", operation_id="x.py", is_retry=True),
        ]
        result = detect_tool_retries(attempts)
        assert len(result) == 2
        assert result[0].tool_name == "bash"
        assert result[1].tool_name == "read_file"
        for ev in result:
            assert isinstance(ev, ToolRetryEvent)


# ---------------------------------------------------------------------------
# 6. Offline Independence
# ---------------------------------------------------------------------------


class TestOfflineIndependence:
    """Verify zero external/LLM imports in tool_retries module."""

    def test_no_forbidden_imports(self) -> None:
        import context_health.tool_retries as retry_mod

        with open(retry_mod.__file__) as f:
            code = f.read()

        for forbidden in ["import anthropic", "import httpx", "import requests", "import urllib"]:
            assert forbidden not in code, f"Forbidden import found: {forbidden}"
