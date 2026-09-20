"""Milestone 4 tests — task complexity estimation.

ALL tests are deterministic. ZERO network/API/LLM calls are made.
Tests verify observable behavior, component score explanations, and bounds.
"""

from __future__ import annotations

import pytest

from context_health.complexity import (
    TaskComplexity,
    TaskComplexityEstimator,
    count_explicit_requirements,
    estimate_task_complexity,
)
from context_health.context import ContextState


# ---------------------------------------------------------------------------
# 1. Empty task
# ---------------------------------------------------------------------------


class TestEmptyTask:
    """Verify empty task yields zero complexity across all components."""

    def test_empty_task_score_is_zero(self) -> None:
        result = estimate_task_complexity()
        assert result.task_complexity == 0.0
        assert result.message_length_score == 0.0
        assert result.turn_count_score == 0.0
        assert result.file_reference_score == 0.0
        assert result.tool_call_score == 0.0
        assert result.requirement_count_score == 0.0

    def test_whitespace_only_task(self) -> None:
        result = estimate_task_complexity(user_message="   \n\t  ")
        assert result.task_complexity == 0.0
        assert result.message_length_score == 0.0
        assert result.requirement_count_score == 0.0


# ---------------------------------------------------------------------------
# 2. Very short task
# ---------------------------------------------------------------------------


class TestShortTask:
    """Verify very short tasks produce minimal complexity."""

    def test_short_task_scores_low(self) -> None:
        result = estimate_task_complexity(user_message="Fix typo in README.")
        assert 0.0 < result.task_complexity < 0.05
        assert 0.0 < result.message_length_score < 0.05
        assert result.turn_count_score == 0.0
        assert result.file_reference_score == 0.0
        assert result.tool_call_score == 0.0
        assert result.requirement_count_score == 0.0


# ---------------------------------------------------------------------------
# 3. Longer task
# ---------------------------------------------------------------------------


class TestLongerTask:
    """Verify longer user messages increase message_length_score."""

    def test_longer_task_increases_message_length_score(self) -> None:
        short_result = estimate_task_complexity(user_message="Short request")
        long_message = "A" * 750  # half of 1500 cap
        long_result = estimate_task_complexity(user_message=long_message)

        assert long_result.message_length_score > short_result.message_length_score
        assert long_result.message_length_score == pytest.approx(0.5, abs=0.01)
        assert long_result.task_complexity > short_result.task_complexity

    def test_message_length_saturation(self) -> None:
        very_long_message = "B" * 3000
        result = estimate_task_complexity(user_message=very_long_message)
        assert result.message_length_score == 1.0


# ---------------------------------------------------------------------------
# 4. Multiple turns
# ---------------------------------------------------------------------------


class TestMultipleTurns:
    """Verify increasing conversation turns increases turn_count_score."""

    def test_turn_count_increases_score(self) -> None:
        turn_0 = estimate_task_complexity(turn_count=0)
        turn_5 = estimate_task_complexity(turn_count=5)
        turn_10 = estimate_task_complexity(turn_count=10)

        assert turn_0.turn_count_score == 0.0
        assert turn_5.turn_count_score == pytest.approx(0.5, abs=0.01)
        assert turn_10.turn_count_score == 1.0
        assert turn_10.task_complexity > turn_5.task_complexity > turn_0.task_complexity

    def test_turn_count_saturation(self) -> None:
        result = estimate_task_complexity(turn_count=50)
        assert result.turn_count_score == 1.0


# ---------------------------------------------------------------------------
# 5. Multiple file references
# ---------------------------------------------------------------------------


class TestMultipleFiles:
    """Verify file reference counts increase file_reference_score."""

    def test_file_count_increases_score(self) -> None:
        res_0 = estimate_task_complexity(file_count=0)
        res_2 = estimate_task_complexity(file_count=2)
        res_5 = estimate_task_complexity(file_count=5)

        assert res_0.file_reference_score == 0.0
        assert res_2.file_reference_score == pytest.approx(0.4, abs=0.01)
        assert res_5.file_reference_score == 1.0
        assert res_5.task_complexity > res_2.task_complexity > res_0.task_complexity

    def test_files_referenced_list_passed(self) -> None:
        files = ["a.py", "b.py", "c.py"]
        res = estimate_task_complexity(files_referenced=files)
        assert res.file_reference_score == pytest.approx(0.6, abs=0.01)

    def test_file_count_saturation(self) -> None:
        res = estimate_task_complexity(file_count=20)
        assert res.file_reference_score == 1.0


# ---------------------------------------------------------------------------
# 6. Multiple tool calls
# ---------------------------------------------------------------------------


class TestMultipleToolCalls:
    """Verify tool call count increases tool_call_score."""

    def test_tool_call_increases_score(self) -> None:
        t_0 = estimate_task_complexity(tool_calls=0)
        t_5 = estimate_task_complexity(tool_calls=5)
        t_10 = estimate_task_complexity(tool_calls=10)

        assert t_0.tool_call_score == 0.0
        assert t_5.tool_call_score == pytest.approx(0.5, abs=0.01)
        assert t_10.tool_call_score == 1.0
        assert t_10.task_complexity > t_5.task_complexity > t_0.task_complexity

    def test_tool_calls_saturation(self) -> None:
        res = estimate_task_complexity(tool_calls=100)
        assert res.tool_call_score == 1.0


# ---------------------------------------------------------------------------
# 7. Multiple explicit requirements
# ---------------------------------------------------------------------------


class TestExplicitRequirements:
    """Verify requirement detection and score scaling."""

    def test_numbered_list_requirements(self) -> None:
        msg = (
            "Please build the feature with these requirements:\n"
            "1. Must support python 3.11\n"
            "2. Ensure backwards compatibility\n"
            "3. Need to add unit tests\n"
            "4. Require zero external API calls\n"
            "5. Shall clamp all outputs\n"
        )
        assert count_explicit_requirements(msg) == 5
        res = estimate_task_complexity(user_message=msg)
        assert res.requirement_count_score == 1.0

    def test_bullet_list_requirements(self) -> None:
        msg = (
            "Task specification:\n"
            "- First requirement\n"
            "- Second requirement\n"
            "- Third requirement\n"
        )
        assert count_explicit_requirements(msg) == 3
        res = estimate_task_complexity(user_message=msg)
        assert res.requirement_count_score == pytest.approx(0.6, abs=0.01)

    def test_prose_keywords_requirements(self) -> None:
        msg = "You must do X, ensure Y, and we need to verify Z."
        assert count_explicit_requirements(msg) == 3
        res = estimate_task_complexity(user_message=msg)
        assert res.requirement_count_score == pytest.approx(0.6, abs=0.01)

    def test_explicit_requirement_count_override(self) -> None:
        res = estimate_task_complexity(requirement_count=4)
        assert res.requirement_count_score == pytest.approx(0.8, abs=0.01)


# ---------------------------------------------------------------------------
# 8. Relative ordering: Simple vs. Substantially More Complex
# ---------------------------------------------------------------------------


class TestRelativeComplexityOrdering:
    """Verify simple tasks produce lower scores than complex multi-turn tasks."""

    def test_simple_vs_complex(self) -> None:
        simple = estimate_task_complexity(
            user_message="Change color to blue",
            turn_count=1,
            file_count=1,
            tool_calls=0,
        )

        complex_task = estimate_task_complexity(
            user_message=(
                "Refactor authentication module.\n"
                "1. Must support OAuth2 and SAML\n"
                "2. Ensure zero downtime\n"
                "3. Need to migrate legacy tokens\n"
                "4. Require unit and integration tests\n"
                "5. Shall audit all auth routes\n"
            ),
            turn_count=8,
            file_count=5,
            tool_calls=7,
        )

        assert simple.task_complexity < complex_task.task_complexity
        assert complex_task.task_complexity >= 0.70


# ---------------------------------------------------------------------------
# 9–11. Mathematical bounds [0.0, 1.0]
# ---------------------------------------------------------------------------


class TestScoreBounds:
    """Verify task_complexity and all component scores are strictly in [0.0, 1.0]."""

    @pytest.mark.parametrize(
        ("kwargs"),
        [
            {},
            {"user_message": "hi"},
            {"user_message": "X" * 10000, "turn_count": 100, "file_count": 50, "tool_calls": 200},
            {"turn_count": -5, "file_count": -2, "tool_calls": -10, "requirement_count": -3},
        ],
    )
    def test_bounds_respected(self, kwargs: dict) -> None:
        res = estimate_task_complexity(**kwargs)
        for score_name in [
            "task_complexity",
            "message_length_score",
            "turn_count_score",
            "file_reference_score",
            "tool_call_score",
            "requirement_count_score",
        ]:
            val = getattr(res, score_name)
            assert 0.0 <= val <= 1.0, f"{score_name} was {val}, out of [0.0, 1.0]"

    def test_maximum_inputs_cap_at_exactly_one(self) -> None:
        res = estimate_task_complexity(
            user_message="Z" * 10000,
            turn_count=100,
            file_count=50,
            tool_calls=50,
            requirement_count=50,
        )
        assert res.task_complexity == 1.0
        assert res.message_length_score == 1.0
        assert res.turn_count_score == 1.0
        assert res.file_reference_score == 1.0
        assert res.tool_call_score == 1.0
        assert res.requirement_count_score == 1.0


# ---------------------------------------------------------------------------
# 12. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Verify that identical inputs yield identical outputs."""

    def test_same_inputs_same_score(self) -> None:
        kwargs = {
            "user_message": "Run database migrations and ensure schema validity.",
            "turn_count": 3,
            "file_count": 2,
            "tool_calls": 4,
        }
        res1 = estimate_task_complexity(**kwargs)
        res2 = estimate_task_complexity(**kwargs)

        assert res1 == res2
        assert res1.task_complexity == res2.task_complexity


# ---------------------------------------------------------------------------
# 13. Offline & Provider Agnostic (Zero Network/LLM calls)
# ---------------------------------------------------------------------------


class TestOfflineIndependence:
    """Verify no network, LLM, or provider imports are used."""

    def test_no_network_or_llm_imports(self) -> None:
        import context_health.complexity as comp_module

        with open(comp_module.__file__) as f:
            content = f.read()

        for forbidden in ["import anthropic", "import httpx", "import requests", "import urllib"]:
            assert forbidden not in content, f"Forbidden import found: {forbidden}"


# ---------------------------------------------------------------------------
# 14. Component Score Explanations
# ---------------------------------------------------------------------------


class TestComponentExplanation:
    """Verify that component scores clearly explain where complexity came from."""

    def test_isolated_component_contribution(self) -> None:
        estimator = TaskComplexityEstimator()

        # Only file references present
        res = estimator.estimate(file_count=5)
        assert res.file_reference_score == 1.0
        assert res.message_length_score == 0.0
        assert res.turn_count_score == 0.0
        assert res.tool_call_score == 0.0
        assert res.requirement_count_score == 0.0
        assert res.task_complexity == pytest.approx(0.20, abs=0.001)

    def test_custom_weights(self) -> None:
        custom_estimator = TaskComplexityEstimator(
            weights={
                "message_length": 0.50,
                "turn_count": 0.50,
                "file_reference": 0.0,
                "tool_call": 0.0,
                "requirement_count": 0.0,
            }
        )
        res = custom_estimator.estimate(
            user_message="A" * 1500,
            turn_count=10,
            file_count=5,
        )
        # 0.5 * 1.0 + 0.5 * 1.0 = 1.0
        assert res.task_complexity == 1.0


# ---------------------------------------------------------------------------
# 15. ContextState Integration
# ---------------------------------------------------------------------------


class TestContextStateIntegration:
    """Verify estimator can extract metrics directly from ContextState."""

    def test_extract_from_context_state(self) -> None:
        state = ContextState(
            session_id="session-xyz",
            message_count=6,  # 3 turns
            files_referenced=["models.py", "views.py"],
            tool_calls=4,
        )
        res = estimate_task_complexity(context_state=state)

        assert res.turn_count_score == pytest.approx(0.3, abs=0.01)  # 3 / 10
        assert res.file_reference_score == pytest.approx(0.4, abs=0.01)  # 2 / 5
        assert res.tool_call_score == pytest.approx(0.4, abs=0.01)  # 4 / 10
        assert res.message_length_score == 0.0
        assert res.requirement_count_score == 0.0
        # 0.20 * 0.3 + 0.20 * 0.4 + 0.20 * 0.4 = 0.06 + 0.08 + 0.08 = 0.22
        assert res.task_complexity == pytest.approx(0.22, abs=0.001)
