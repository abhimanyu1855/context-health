"""Milestone 3 tests — context tracking.

ALL tests are deterministic.  ZERO network/API calls are made.
No provider-specific SDK objects are used inside the tracker.
"""

from __future__ import annotations

import uuid

import pytest

from context_health.config import (
    MODEL_CONTEXT_WINDOWS,
    DEFAULT_CONTEXT_WINDOW,
    get_context_window,
)
from context_health.context import ContextState, ContextTracker


# ---------------------------------------------------------------------------
# 1. ContextState default construction
# ---------------------------------------------------------------------------


class TestContextStateDefaults:
    """Verify ContextState can be constructed with defaults."""

    def test_default_construction(self) -> None:
        state = ContextState()
        assert state.session_id == ""
        assert state.message_count == 0
        assert state.input_tokens == 0
        assert state.output_tokens == 0
        assert state.estimated_context_tokens == 0
        assert state.context_window == 200_000
        assert state.context_utilization == 0.0
        assert state.files_referenced == []
        assert state.tool_calls == 0

    def test_custom_construction(self) -> None:
        state = ContextState(
            session_id="test-123",
            message_count=5,
            input_tokens=100,
            output_tokens=50,
            estimated_context_tokens=80,
            context_window=100_000,
            context_utilization=0.0008,
            files_referenced=["main.py"],
            tool_calls=2,
        )
        assert state.session_id == "test-123"
        assert state.message_count == 5
        assert state.estimated_context_tokens == 80
        assert state.files_referenced == ["main.py"]


# ---------------------------------------------------------------------------
# 2–3. Session ID generation
# ---------------------------------------------------------------------------


class TestSessionID:
    """Verify session IDs are generated or accepted."""

    def test_auto_generated_session_id(self) -> None:
        tracker = ContextTracker()
        state = tracker.state
        # Auto-generated ID should be a 32-char hex string (uuid4).
        assert len(state.session_id) == 32
        # Confirm it's valid hex.
        int(state.session_id, 16)

    def test_explicit_session_id(self) -> None:
        tracker = ContextTracker(session_id="my-session-001")
        assert tracker.state.session_id == "my-session-001"

    def test_unique_session_ids(self) -> None:
        ids = {ContextTracker().state.session_id for _ in range(20)}
        assert len(ids) == 20


# ---------------------------------------------------------------------------
# 4–6. Recording messages and message count
# ---------------------------------------------------------------------------


class TestMessageRecording:
    """Verify user and assistant messages are counted."""

    def test_record_user_message(self) -> None:
        tracker = ContextTracker()
        tracker.record_user_message()
        assert tracker.state.message_count == 1

    def test_record_assistant_message(self) -> None:
        tracker = ContextTracker()
        tracker.record_assistant_response(input_tokens=10, output_tokens=5)
        assert tracker.state.message_count == 1

    def test_message_count_accumulates(self) -> None:
        tracker = ContextTracker()
        tracker.record_user_message()
        tracker.record_assistant_response(input_tokens=10, output_tokens=5)
        tracker.record_user_message()
        tracker.record_assistant_response(input_tokens=20, output_tokens=10)
        assert tracker.state.message_count == 4


# ---------------------------------------------------------------------------
# 7–8. Token tracking (cumulative telemetry)
# ---------------------------------------------------------------------------


class TestTokenTracking:
    """Verify input and output tokens accumulate correctly."""

    def test_input_tokens_cumulative(self) -> None:
        tracker = ContextTracker()
        tracker.record_assistant_response(input_tokens=100, output_tokens=0)
        tracker.record_assistant_response(input_tokens=200, output_tokens=0)
        assert tracker.state.input_tokens == 300

    def test_output_tokens_cumulative(self) -> None:
        tracker = ContextTracker()
        tracker.record_assistant_response(input_tokens=0, output_tokens=50)
        tracker.record_assistant_response(input_tokens=0, output_tokens=75)
        assert tracker.state.output_tokens == 125

    def test_user_message_does_not_add_tokens(self) -> None:
        tracker = ContextTracker()
        tracker.record_user_message()
        assert tracker.state.input_tokens == 0
        assert tracker.state.output_tokens == 0


# ---------------------------------------------------------------------------
# 9. Estimated context tokens — latest input token count
# ---------------------------------------------------------------------------


class TestEstimatedContextTokens:
    """Verify estimated_context_tokens uses latest input_tokens, not cumulative."""

    def test_uses_latest_input_tokens(self) -> None:
        tracker = ContextTracker()
        tracker.record_assistant_response(input_tokens=500, output_tokens=100)
        assert tracker.state.estimated_context_tokens == 500

    def test_replaces_on_next_turn(self) -> None:
        """Second turn should use the second input_tokens, not sum."""
        tracker = ContextTracker()
        tracker.record_assistant_response(input_tokens=500, output_tokens=100)
        tracker.record_assistant_response(input_tokens=1200, output_tokens=200)
        assert tracker.state.estimated_context_tokens == 1200
        # Cumulative telemetry still correct:
        assert tracker.state.input_tokens == 1700

    def test_does_not_double_count(self) -> None:
        """Multiple turns must NOT sum all input_tokens for estimated occupancy."""
        tracker = ContextTracker(context_window=200_000)
        tracker.record_assistant_response(input_tokens=500, output_tokens=100)
        tracker.record_assistant_response(input_tokens=1200, output_tokens=200)
        tracker.record_assistant_response(input_tokens=2000, output_tokens=300)
        # Estimated context occupancy = latest input only = 2000
        assert tracker.state.estimated_context_tokens == 2000
        # NOT 500 + 1200 + 2000 + 100 + 200 + 300

    def test_zero_before_any_response(self) -> None:
        tracker = ContextTracker()
        assert tracker.state.estimated_context_tokens == 0


# ---------------------------------------------------------------------------
# 10. Context utilization — uses estimated context tokens
# ---------------------------------------------------------------------------


class TestContextUtilization:
    """Verify utilization = estimated_context_tokens / context_window."""

    def test_utilization_formula(self) -> None:
        tracker = ContextTracker(context_window=100_000)
        tracker.record_assistant_response(input_tokens=10_000, output_tokens=5_000)
        state = tracker.state
        # estimated_context_tokens = 10_000 (latest input)
        # 10_000 / 100_000 = 0.10
        assert state.context_utilization == pytest.approx(0.10)

    def test_utilization_zero_initially(self) -> None:
        tracker = ContextTracker()
        assert tracker.state.context_utilization == 0.0

    def test_utilization_uses_latest_input(self) -> None:
        tracker = ContextTracker(context_window=200_000)
        tracker.record_assistant_response(input_tokens=50_000, output_tokens=10_000)
        tracker.record_assistant_response(input_tokens=100_000, output_tokens=20_000)
        # estimated = 100_000 (latest), utilization = 100_000 / 200_000 = 0.5
        assert tracker.state.context_utilization == pytest.approx(0.5)

    def test_utilization_deterministic(self) -> None:
        tracker = ContextTracker(context_window=200_000)
        tracker.record_assistant_response(input_tokens=50_000, output_tokens=10_000)
        # 50_000 / 200_000 = 0.25
        assert tracker.state.context_utilization == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# 11. Utilization clamped at 1.0
# ---------------------------------------------------------------------------


class TestUtilizationClamping:
    """Verify utilization is clamped to [0.0, 1.0]."""

    def test_utilization_clamped_at_one(self) -> None:
        tracker = ContextTracker(context_window=100)
        tracker.record_assistant_response(input_tokens=200, output_tokens=50)
        # estimated = 200, 200/100 = 2.0 → clamped to 1.0
        assert tracker.state.context_utilization == 1.0

    def test_utilization_exactly_one(self) -> None:
        tracker = ContextTracker(context_window=1000)
        tracker.record_assistant_response(input_tokens=1000, output_tokens=100)
        # estimated = 1000, 1000/1000 = 1.0
        assert tracker.state.context_utilization == 1.0

    def test_utilization_never_negative(self) -> None:
        tracker = ContextTracker(context_window=100)
        assert tracker.state.context_utilization >= 0.0

    def test_near_limit_utilization(self) -> None:
        tracker = ContextTracker(context_window=200_000)
        tracker.record_assistant_response(input_tokens=199_000, output_tokens=500)
        # estimated = 199_000, 199_000/200_000 = 0.995
        assert tracker.state.context_utilization == pytest.approx(0.995)

    def test_over_limit_utilization(self) -> None:
        tracker = ContextTracker(context_window=100_000)
        tracker.record_assistant_response(input_tokens=150_000, output_tokens=10_000)
        # estimated = 150_000, 150_000/100_000 = 1.5 → clamped to 1.0
        assert tracker.state.context_utilization == 1.0

    def test_zero_context_window(self) -> None:
        """Zero context window should not cause division by zero."""
        tracker = ContextTracker(context_window=0)
        tracker.record_assistant_response(input_tokens=100, output_tokens=50)
        assert tracker.state.context_utilization == 0.0


# ---------------------------------------------------------------------------
# 12. Files referenced tracking
# ---------------------------------------------------------------------------


class TestFilesReferenced:
    """Verify file reference tracking."""

    def test_record_file(self) -> None:
        tracker = ContextTracker()
        tracker.record_file_reference("src/main.py")
        assert tracker.state.files_referenced == ["src/main.py"]

    def test_multiple_files(self) -> None:
        tracker = ContextTracker()
        tracker.record_file_reference("src/main.py")
        tracker.record_file_reference("tests/test_main.py")
        assert tracker.state.files_referenced == ["src/main.py", "tests/test_main.py"]

    def test_duplicate_ignored(self) -> None:
        tracker = ContextTracker()
        tracker.record_file_reference("src/main.py")
        tracker.record_file_reference("src/main.py")
        assert tracker.state.files_referenced == ["src/main.py"]

    def test_empty_initially(self) -> None:
        tracker = ContextTracker()
        assert tracker.state.files_referenced == []


# ---------------------------------------------------------------------------
# 13. Tool call tracking
# ---------------------------------------------------------------------------


class TestToolCallTracking:
    """Verify tool call counting."""

    def test_record_tool_call(self) -> None:
        tracker = ContextTracker()
        tracker.record_tool_call()
        assert tracker.state.tool_calls == 1

    def test_multiple_tool_calls(self) -> None:
        tracker = ContextTracker()
        for _ in range(5):
            tracker.record_tool_call()
        assert tracker.state.tool_calls == 5

    def test_zero_initially(self) -> None:
        tracker = ContextTracker()
        assert tracker.state.tool_calls == 0


# ---------------------------------------------------------------------------
# 14. Current state retrieval
# ---------------------------------------------------------------------------


class TestStateRetrieval:
    """Verify .state returns a coherent snapshot."""

    def test_state_is_context_state(self) -> None:
        tracker = ContextTracker()
        assert isinstance(tracker.state, ContextState)

    def test_state_returns_copy_of_files(self) -> None:
        """Mutating the returned list should not affect the tracker."""
        tracker = ContextTracker()
        tracker.record_file_reference("a.py")
        state = tracker.state
        state.files_referenced.append("INJECTED")
        assert "INJECTED" not in tracker.state.files_referenced


# ---------------------------------------------------------------------------
# 15. Multiple turns accumulating correctly
# ---------------------------------------------------------------------------


class TestMultipleTurns:
    """Simulate a realistic multi-turn session."""

    def test_three_turn_session(self) -> None:
        tracker = ContextTracker(context_window=200_000, session_id="multi-turn")

        # Turn 1: model reads 500 tokens, writes 200
        tracker.record_user_message()
        tracker.record_assistant_response(input_tokens=500, output_tokens=200)
        tracker.record_file_reference("main.py")

        # Turn 2: model reads 1200 tokens (includes prior history), writes 400
        tracker.record_user_message()
        tracker.record_assistant_response(input_tokens=1200, output_tokens=400)
        tracker.record_tool_call()
        tracker.record_file_reference("utils.py")

        # Turn 3: model reads 2000 tokens (includes prior history), writes 600
        tracker.record_user_message()
        tracker.record_assistant_response(input_tokens=2000, output_tokens=600)
        tracker.record_tool_call()
        tracker.record_tool_call()
        tracker.record_file_reference("main.py")  # duplicate

        state = tracker.state

        assert state.session_id == "multi-turn"
        assert state.message_count == 6  # 3 user + 3 assistant
        # Cumulative telemetry:
        assert state.input_tokens == 3700  # 500 + 1200 + 2000
        assert state.output_tokens == 1200  # 200 + 400 + 600
        # Estimated context occupancy = latest input = 2000
        assert state.estimated_context_tokens == 2000
        assert state.context_utilization == pytest.approx(2000 / 200_000)
        assert state.files_referenced == ["main.py", "utils.py"]
        assert state.tool_calls == 3

    def test_multiple_turns_not_double_counting(self) -> None:
        """Growing input_tokens per turn should reflect context growth, not sum."""
        tracker = ContextTracker(context_window=100_000)

        # Realistic pattern: each turn's input_tokens grows as history grows
        tracker.record_assistant_response(input_tokens=1000, output_tokens=200)
        tracker.record_assistant_response(input_tokens=2500, output_tokens=300)
        tracker.record_assistant_response(input_tokens=4200, output_tokens=400)

        state = tracker.state
        # Estimated occupancy should be 4200 (latest), not 1000+2500+4200
        assert state.estimated_context_tokens == 4200
        assert state.context_utilization == pytest.approx(4200 / 100_000)


# ---------------------------------------------------------------------------
# 16. No real API calls
# ---------------------------------------------------------------------------


class TestNoAPICalls:
    """Confirm the context tracker makes zero API/network calls."""

    def test_tracker_is_offline(self) -> None:
        """ContextTracker never imports or uses anthropic SDK."""
        import context_health.context as ctx_module
        source = open(ctx_module.__file__).read()
        assert "import anthropic" not in source
        assert "import requests" not in source
        assert "import httpx" not in source
        assert "import urllib" not in source


# ---------------------------------------------------------------------------
# 17. No provider-specific SDK objects inside the tracker
# ---------------------------------------------------------------------------


class TestProviderAgnostic:
    """Verify the tracker does not depend on provider SDK types."""

    def test_context_state_has_no_sdk_types(self) -> None:
        state = ContextState()
        for attr_name in vars(state):
            val = getattr(state, attr_name)
            type_name = type(val).__module__
            assert not type_name.startswith("anthropic"), (
                f"ContextState.{attr_name} has provider-specific type: {type(val)}"
            )

    def test_tracker_accepts_plain_ints(self) -> None:
        """record_assistant_response takes plain ints, not SDK objects."""
        tracker = ContextTracker()
        tracker.record_assistant_response(input_tokens=42, output_tokens=17)
        state = tracker.state
        assert state.input_tokens == 42
        assert state.output_tokens == 17


# ---------------------------------------------------------------------------
# 18. Model context-window configuration
# ---------------------------------------------------------------------------


class TestModelContextWindows:
    """Verify model -> context_window mapping."""

    def test_known_model_lookup(self) -> None:
        window = get_context_window("claude-sonnet-4-20250514")
        assert window == 200_000

    def test_unknown_model_returns_default(self) -> None:
        window = get_context_window("unknown-model-xyz")
        assert window == DEFAULT_CONTEXT_WINDOW

    def test_two_different_models_can_differ(self) -> None:
        """Different models CAN have different context windows."""
        # Add a hypothetical smaller model to the mapping for this test.
        # We test the mechanism, not specific model values.
        original = MODEL_CONTEXT_WINDOWS.copy()
        try:
            MODEL_CONTEXT_WINDOWS["test-small-model"] = 50_000
            MODEL_CONTEXT_WINDOWS["test-large-model"] = 500_000

            assert get_context_window("test-small-model") == 50_000
            assert get_context_window("test-large-model") == 500_000
            assert get_context_window("test-small-model") != get_context_window("test-large-model")
        finally:
            MODEL_CONTEXT_WINDOWS.clear()
            MODEL_CONTEXT_WINDOWS.update(original)

    def test_tracker_with_model_specific_window(self) -> None:
        """Tracker should work with model-specific context windows."""
        small_window = 50_000
        tracker = ContextTracker(context_window=small_window)
        tracker.record_assistant_response(input_tokens=25_000, output_tokens=5_000)
        # estimated = 25_000, utilization = 25_000 / 50_000 = 0.5
        assert tracker.state.context_utilization == pytest.approx(0.5)

        large_window = 500_000
        tracker2 = ContextTracker(context_window=large_window)
        tracker2.record_assistant_response(input_tokens=25_000, output_tokens=5_000)
        # estimated = 25_000, utilization = 25_000 / 500_000 = 0.05
        assert tracker2.state.context_utilization == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# 19. Missing usage
# ---------------------------------------------------------------------------


class TestMissingUsage:
    """Verify sensible behavior when no token usage has been recorded."""

    def test_no_responses_recorded(self) -> None:
        tracker = ContextTracker()
        tracker.record_user_message()
        tracker.record_user_message()
        state = tracker.state
        assert state.estimated_context_tokens == 0
        assert state.input_tokens == 0
        assert state.output_tokens == 0
        assert state.context_utilization == 0.0

    def test_zero_token_response(self) -> None:
        tracker = ContextTracker()
        tracker.record_assistant_response(input_tokens=0, output_tokens=0)
        state = tracker.state
        assert state.estimated_context_tokens == 0
        assert state.context_utilization == 0.0
