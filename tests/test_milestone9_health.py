"""Milestone 9 tests — Context Health score computation.

ALL tests are deterministic. ZERO network/API/LLM calls are made.
Tests verify formula weights, monotonic properties, event saturation,
status bands, input sanitization/clamping, explainability, ContextState integration,
and offline independence.
"""

from __future__ import annotations

import math
import pytest

from context_health.context import ContextState, ContextTracker
from context_health.health import (
    DEFAULT_HEALTH_WEIGHTS,
    DEFAULT_SATURATION_CONSTANTS,
    ContextHealthResult,
    HealthPenalties,
    HealthSignalInputs,
    HealthWeightedContributions,
    calculate_context_health,
    compute_context_health,
    get_health_status,
)


# ---------------------------------------------------------------------------
# 1. Baseline & Extremes
# ---------------------------------------------------------------------------


class TestHealthBaselines:
    """Verify baseline ideal and worst-case health calculations."""

    def test_ideal_inputs_returns_100_healthy(self) -> None:
        result = compute_context_health(
            context_utilization=0.0,
            relevant_context_ratio=1.0,
            task_complexity=0.0,
            contradiction_count=0,
            correction_count=0,
            retry_count=0,
        )
        assert result.health_score == 100.0
        assert result.status == "healthy"
        assert result.penalties.total_weighted_penalty == 0.0

    def test_worst_case_inputs_returns_0_critical(self) -> None:
        result = compute_context_health(
            context_utilization=1.0,
            relevant_context_ratio=0.0,
            task_complexity=1.0,
            contradiction_count=1000,
            correction_count=1000,
            retry_count=1000,
        )
        # Event penalties approach 1.0 (e.g. 1000/1003 ~ 0.997)
        # Total penalty approaches 1.0, so health approaches 0
        assert result.health_score <= 1.0
        assert result.status == "critical"

    def test_default_values_ideal(self) -> None:
        result = compute_context_health(context_utilization=0.0)
        assert result.health_score == 100.0
        assert result.status == "healthy"


# ---------------------------------------------------------------------------
# 2. Individual Signal Weights & Penalty Calculations
# ---------------------------------------------------------------------------


class TestIndividualSignalPenalties:
    """Verify each signal's exact weight contribution when isolated."""

    def test_utilization_penalty_isolated(self) -> None:
        # Utilization = 1.0 (weight 0.25) -> Penalty = 0.25 -> Score = 75.0
        result = compute_context_health(
            context_utilization=1.0,
            relevant_context_ratio=1.0,
            task_complexity=0.0,
        )
        assert result.health_score == 75.0
        assert result.penalties.utilization_penalty == 1.0
        assert result.contributions.utilization == 0.25
        assert result.status == "watch"

    def test_relevance_penalty_isolated(self) -> None:
        # Relevance = 0.0 (weight 0.30) -> Penalty = 0.30 -> Score = 70.0
        result = compute_context_health(
            context_utilization=0.0,
            relevant_context_ratio=0.0,
            task_complexity=0.0,
        )
        assert result.health_score == 70.0
        assert result.penalties.relevance_penalty == 1.0
        assert result.contributions.relevance == 0.30
        assert result.status == "watch"

    def test_complexity_penalty_isolated(self) -> None:
        # Complexity = 1.0 (weight 0.15) -> Penalty = 0.15 -> Score = 85.0
        result = compute_context_health(
            context_utilization=0.0,
            relevant_context_ratio=1.0,
            task_complexity=1.0,
        )
        assert result.health_score == 85.0
        assert result.penalties.complexity_penalty == 1.0
        assert result.contributions.complexity == 0.15
        assert result.status == "healthy"

    def test_event_penalties_at_k(self) -> None:
        # At count = 3 (k=3), normalized penalty = 3 / (3 + 3) = 0.50
        # Weight = 0.10 -> Contribution = 0.05 -> Score = 95.0
        res_contra = compute_context_health(
            context_utilization=0.0,
            contradiction_count=3,
        )
        assert res_contra.penalties.contradiction_penalty == pytest.approx(0.50)
        assert res_contra.contributions.contradictions == pytest.approx(0.05)
        assert res_contra.health_score == 95.0

        res_corr = compute_context_health(
            context_utilization=0.0,
            correction_count=3,
        )
        assert res_corr.penalties.correction_penalty == pytest.approx(0.50)
        assert res_corr.contributions.corrections == pytest.approx(0.05)
        assert res_corr.health_score == 95.0

        res_retry = compute_context_health(
            context_utilization=0.0,
            retry_count=3,
        )
        assert res_retry.penalties.retry_penalty == pytest.approx(0.50)
        assert res_retry.contributions.retries == pytest.approx(0.05)
        assert res_retry.health_score == 95.0


# ---------------------------------------------------------------------------
# 3. Core Hypothesis Validation Test
# ---------------------------------------------------------------------------


class TestCoreHypothesisBehavior:
    """Verify that for identical utilization, high relevance yields higher health than low relevance."""

    def test_relevance_distinction_at_same_utilization(self) -> None:
        # Both sessions have 80% context utilization
        session_a_high_rel = compute_context_health(
            context_utilization=0.80,
            relevant_context_ratio=0.90,
            task_complexity=0.30,
        )
        session_b_low_rel = compute_context_health(
            context_utilization=0.80,
            relevant_context_ratio=0.20,
            task_complexity=0.30,
        )
        assert session_a_high_rel.health_score > session_b_low_rel.health_score
        # Difference should reflect 0.30 * (0.90 - 0.20) = 0.21 -> 21 points
        score_diff = session_a_high_rel.health_score - session_b_low_rel.health_score
        assert score_diff == pytest.approx(21.0, abs=0.1)


# ---------------------------------------------------------------------------
# 4. Monotonicity Invariants
# ---------------------------------------------------------------------------


class TestMonotonicity:
    """Verify that health behaves monotonically with respect to all individual signals."""

    def test_utilization_monotonicity(self) -> None:
        scores = [
            compute_context_health(context_utilization=u, relevant_context_ratio=0.8).health_score
            for u in [0.0, 0.25, 0.50, 0.75, 1.0]
        ]
        assert scores == sorted(scores, reverse=True)

    def test_relevance_monotonicity(self) -> None:
        scores = [
            compute_context_health(context_utilization=0.5, relevant_context_ratio=r).health_score
            for r in [0.0, 0.25, 0.50, 0.75, 1.0]
        ]
        assert scores == sorted(scores)

    def test_complexity_monotonicity(self) -> None:
        scores = [
            compute_context_health(context_utilization=0.5, task_complexity=c).health_score
            for c in [0.0, 0.25, 0.50, 0.75, 1.0]
        ]
        assert scores == sorted(scores, reverse=True)

    def test_contradiction_monotonicity(self) -> None:
        scores = [
            compute_context_health(context_utilization=0.5, contradiction_count=c).health_score
            for c in [0, 1, 3, 5, 10, 50]
        ]
        assert scores == sorted(scores, reverse=True)

    def test_correction_monotonicity(self) -> None:
        scores = [
            compute_context_health(context_utilization=0.5, correction_count=c).health_score
            for c in [0, 1, 3, 5, 10, 50]
        ]
        assert scores == sorted(scores, reverse=True)

    def test_retry_monotonicity(self) -> None:
        scores = [
            compute_context_health(context_utilization=0.5, retry_count=r).health_score
            for r in [0, 1, 3, 5, 10, 50]
        ]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# 5. Status Bands & Boundaries
# ---------------------------------------------------------------------------


class TestStatusBands:
    """Verify exact status threshold assignments."""

    @pytest.mark.parametrize(
        ("score", "expected_status"),
        [
            (100.0, "healthy"),
            (85.5, "healthy"),
            (80.0, "healthy"),
            (79.99, "watch"),
            (70.0, "watch"),
            (60.0, "watch"),
            (59.99, "warning"),
            (50.0, "warning"),
            (40.0, "warning"),
            (39.99, "critical"),
            (15.0, "critical"),
            (0.0, "critical"),
        ],
    )
    def test_status_bands(self, score: float, expected_status: str) -> None:
        assert get_health_status(score) == expected_status


# ---------------------------------------------------------------------------
# 6. Input Clamping, Sanitization & Edge Cases
# ---------------------------------------------------------------------------


class TestInputSanitization:
    """Verify robust sanitization of out-of-bounds, negative, or NaN inputs."""

    def test_ratios_clamped_above_1(self) -> None:
        result = compute_context_health(
            context_utilization=2.5,
            relevant_context_ratio=1.8,
            task_complexity=3.0,
        )
        assert result.inputs.context_utilization == 1.0
        assert result.inputs.relevant_context_ratio == 1.0
        assert result.inputs.task_complexity == 1.0

    def test_ratios_clamped_below_0(self) -> None:
        result = compute_context_health(
            context_utilization=-0.5,
            relevant_context_ratio=-1.0,
            task_complexity=-0.2,
        )
        assert result.inputs.context_utilization == 0.0
        assert result.inputs.relevant_context_ratio == 0.0
        assert result.inputs.task_complexity == 0.0

    def test_negative_event_counts_sanitized_to_zero(self) -> None:
        result = compute_context_health(
            context_utilization=0.0,
            contradiction_count=-5,
            correction_count=-3,
            retry_count=-1,
        )
        assert result.inputs.contradiction_count == 0
        assert result.inputs.correction_count == 0
        assert result.inputs.retry_count == 0
        assert result.health_score == 100.0

    def test_nan_inputs_handled_safely(self) -> None:
        result = compute_context_health(
            context_utilization=float("nan"),
            relevant_context_ratio=float("nan"),
            task_complexity=float("nan"),
        )
        assert not math.isnan(result.health_score)
        assert 0.0 <= result.health_score <= 100.0


# ---------------------------------------------------------------------------
# 7. ContextState Integration
# ---------------------------------------------------------------------------


class TestContextStateIntegration:
    """Verify calculate_context_health works with ContextState snapshot."""

    def test_calculate_from_state(self) -> None:
        tracker = ContextTracker(context_window=100_000)
        tracker.record_assistant_response(input_tokens=50_000, output_tokens=1000)
        tracker.record_correction_event()
        tracker.record_tool_retry()

        state = tracker.state
        assert state.context_utilization == pytest.approx(0.50)
        assert state.correction_events == 1
        assert state.tool_retries == 1

        result = calculate_context_health(
            state,
            relevant_context_ratio=0.80,
            task_complexity=0.20,
            contradiction_count=0,
        )

        assert isinstance(result, ContextHealthResult)
        assert result.inputs.context_utilization == pytest.approx(0.50)
        assert result.inputs.correction_count == 1
        assert result.inputs.retry_count == 1
        assert result.inputs.relevant_context_ratio == 0.80
        assert result.inputs.task_complexity == 0.20
        assert 0.0 <= result.health_score <= 100.0


# ---------------------------------------------------------------------------
# 8. Data Model, Immutability & Explainability
# ---------------------------------------------------------------------------


class TestDataModelAndExplainability:
    """Verify data model immutability and explain string."""

    def test_result_immutability(self) -> None:
        result = compute_context_health(context_utilization=0.2)
        with pytest.raises(AttributeError):
            result.health_score = 99.0  # type: ignore[misc]

    def test_explain_string_content(self) -> None:
        result = compute_context_health(
            context_utilization=0.5,
            relevant_context_ratio=0.7,
            task_complexity=0.4,
            contradiction_count=1,
            correction_count=2,
            retry_count=1,
        )
        text = result.explain()
        assert "Context Health:" in text
        assert "Utilization:" in text
        assert "Relevance:" in text
        assert "Complexity:" in text
        assert "Contradictions" in text
        assert "Corrections" in text
        assert "Retries" in text


# ---------------------------------------------------------------------------
# 9. Offline Independence
# ---------------------------------------------------------------------------


class TestOfflineIndependence:
    """Verify zero external/LLM imports in health module."""

    def test_no_forbidden_imports(self) -> None:
        import context_health.health as health_mod

        with open(health_mod.__file__) as f:
            code = f.read()

        for forbidden in ["import anthropic", "import httpx", "import requests", "import urllib"]:
            assert forbidden not in code, f"Forbidden import found: {forbidden}"
