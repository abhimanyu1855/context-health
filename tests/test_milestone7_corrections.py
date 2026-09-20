"""Milestone 7 tests — correction event detection.

ALL tests are deterministic. ZERO network/API/LLM calls are made.
Tests verify positive/negative examples, false-positive protection,
stateful and stateless tracking, ContextTracker integration, and bounds.
"""

from __future__ import annotations

import pytest

from context_health.context import ContextState, ContextTracker
from context_health.corrections import (
    CorrectionDetector,
    CorrectionEvent,
    CorrectionResult,
    detect_corrections,
    detect_single_message_correction,
    is_correction,
)


# ---------------------------------------------------------------------------
# 1. Positive Correction Examples
# ---------------------------------------------------------------------------


class TestPositiveCorrections:
    """Verify that explicit correction and redirect patterns are detected."""

    @pytest.mark.parametrize(
        ("message", "expected_type"),
        [
            ("No, use PostgreSQL instead.", "rejection_redirect"),
            ("That's wrong, use Redis.", "error_assertion"),
            ("Don't do that; use X instead.", "rejection_redirect"),
            ("I meant X, not Y.", "intent_clarification"),
            ("Actually, change that to X.", "explicit_override"),
            ("You misunderstood the requirement.", "misunderstanding"),
            ("That's not what I asked for.", "misunderstanding"),
            ("No, that's not right.", "error_assertion"),
            ("Stop, that's wrong.", "error_assertion"),
            ("Wrong: use Redis instead.", "error_assertion"),
            ("I never asked for this feature.", "misunderstanding"),
            ("I didn't ask for extra dependencies.", "misunderstanding"),
            ("Nope, switch to SQLite.", "rejection_redirect"),
            ("Wait, that is incorrect.", "error_assertion"),
            ("Instead of that, do Y.", "rejection_redirect"),
            ("Undo that and implement Z.", "rejection_redirect"),
        ],
    )
    def test_positive_correction_recognition(self, message: str, expected_type: str) -> None:
        assert is_correction(message) is True
        event = detect_single_message_correction(message)
        assert event is not None
        assert isinstance(event, CorrectionEvent)
        assert event.message == message
        assert event.correction_type == expected_type
        assert len(event.matched_phrase) > 0


# ---------------------------------------------------------------------------
# 2. Negative / Non-Correction Examples
# ---------------------------------------------------------------------------


class TestNegativeCorrections:
    """Verify that ordinary instructions and inquiries are NOT flagged as corrections."""

    @pytest.mark.parametrize(
        "message",
        [
            "Use PostgreSQL for the database.",
            "Add Redis caching.",
            "Can you explain this?",
            "What does this function do?",
            "Normal follow-up requirements that do not explicitly correct previous agent behavior.",
            "No problem at all.",
            "There are no errors in the logs.",
            "Actually, Python 3.11 is great.",
            "Please modify client.py.",
            "Change the color to blue.",
            "Create a new file named utils.py.",
            "Show me the test results.",
            "",
            "   \n\t  ",
        ],
    )
    def test_negative_non_correction_recognition(self, message: str) -> None:
        assert is_correction(message) is False
        assert detect_single_message_correction(message) is None


# ---------------------------------------------------------------------------
# 3. Punctuation and Case Variations
# ---------------------------------------------------------------------------


class TestPunctuationAndCaseVariations:
    """Verify robust detection across case and punctuation variations."""

    def test_case_variations(self) -> None:
        assert is_correction("NO, USE POSTGRESQL INSTEAD.") is True
        assert is_correction("THAT'S WRONG, USE REDIS.") is True
        assert is_correction("i meant x, not y") is True

    def test_punctuation_variations(self) -> None:
        assert is_correction("No! Use PostgreSQL instead.") is True
        assert is_correction("That's wrong... use Redis!") is True
        assert is_correction("Wrong; use SQLite.") is True


# ---------------------------------------------------------------------------
# 4. Multi-Turn / Sequence Detection & Count Accumulation
# ---------------------------------------------------------------------------


class TestSequenceDetection:
    """Verify detect_corrections across multi-turn message sequences."""

    def test_clean_session_zero_corrections(self) -> None:
        messages = [
            "Please create a web server.",
            "Add user authentication.",
            "Write comprehensive tests.",
        ]
        result = detect_corrections(messages)
        assert result.correction_count == 0
        assert len(result.events) == 0
        assert len(result) == 0

    def test_session_with_multiple_corrections(self) -> None:
        messages = [
            "Please set up the database using MySQL.",
            "No, use PostgreSQL instead.",
            "Add user authentication.",
            "That's wrong, use Redis for session storage.",
            "Run the test suite.",
        ]
        result = detect_corrections(messages)
        assert result.correction_count == 2
        assert len(result.events) == 2
        assert result[0].turn_index == 1
        assert result[0].correction_type == "rejection_redirect"
        assert result[1].turn_index == 3
        assert result[1].correction_type == "error_assertion"

    def test_empty_sequence(self) -> None:
        result = detect_corrections([])
        assert result.correction_count == 0
        assert len(result.events) == 0


# ---------------------------------------------------------------------------
# 5. Stateful Incremental Detection & Reset
# ---------------------------------------------------------------------------


class TestStatefulCorrectionDetector:
    """Verify stateful CorrectionDetector incremental methods."""

    def test_incremental_workflow(self) -> None:
        detector = CorrectionDetector()
        assert detector.add_message("Create an API endpoint.") is None
        assert detector.correction_count == 0

        ev1 = detector.add_message("No, use GraphQL instead.", turn_index=1)
        assert ev1 is not None
        assert ev1.correction_type == "rejection_redirect"
        assert detector.correction_count == 1

        assert detector.add_message("Now add unit tests.") is None
        assert detector.correction_count == 1

        ev2 = detector.add_message("You misunderstood the requirement.", turn_index=3)
        assert ev2 is not None
        assert detector.correction_count == 2

        detector.reset()
        assert detector.correction_count == 0
        assert detector.events == []


# ---------------------------------------------------------------------------
# 6. ContextTracker & ContextState Integration
# ---------------------------------------------------------------------------


class TestContextTrackerIntegration:
    """Verify integration of correction tracking into ContextTracker and ContextState."""

    def test_tracker_auto_records_correction_from_user_message(self) -> None:
        tracker = ContextTracker()
        tracker.record_user_message("Please set up Redis.")
        assert tracker.state.correction_events == 0
        assert tracker.state.message_count == 1

        tracker.record_assistant_response(input_tokens=100, output_tokens=50)

        tracker.record_user_message("No, use PostgreSQL instead.")
        assert tracker.state.correction_events == 1
        assert tracker.state.message_count == 3

    def test_tracker_explicit_record_correction_event(self) -> None:
        tracker = ContextTracker()
        tracker.record_correction_event()
        tracker.record_correction_event()
        assert tracker.state.correction_events == 2

    def test_context_state_default_and_custom(self) -> None:
        state = ContextState()
        assert state.correction_events == 0

        custom_state = ContextState(correction_events=5)
        assert custom_state.correction_events == 5


# ---------------------------------------------------------------------------
# 7. Immutability and Sequence Protocol
# ---------------------------------------------------------------------------


class TestResultInterface:
    """Verify CorrectionResult and CorrectionEvent immutability and sequence protocol."""

    def test_immutability(self) -> None:
        event = CorrectionEvent(
            message="That's wrong.",
            correction_type="error_assertion",
            matched_phrase="That's wrong",
            turn_index=0,
        )
        with pytest.raises(AttributeError):
            event.message = "mutated"  # type: ignore[misc]

    def test_sequence_protocol(self) -> None:
        result = detect_corrections(["That's wrong.", "No, use X."])
        assert len(result) == 2
        assert result[0].correction_type == "error_assertion"
        assert result[1].correction_type == "rejection_redirect"
        for ev in result:
            assert isinstance(ev, CorrectionEvent)


# ---------------------------------------------------------------------------
# 8. Offline Independence (No Network / LLM Calls)
# ---------------------------------------------------------------------------


class TestOfflineIndependence:
    """Verify zero external/LLM imports in corrections module."""

    def test_no_forbidden_imports(self) -> None:
        import context_health.corrections as corr_mod

        with open(corr_mod.__file__) as f:
            code = f.read()

        for forbidden in ["import anthropic", "import httpx", "import requests", "import urllib"]:
            assert forbidden not in code, f"Forbidden import found: {forbidden}"
