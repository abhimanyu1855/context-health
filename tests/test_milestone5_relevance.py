"""Milestone 5 tests — relevant context ratio.

ALL tests are deterministic. ZERO network/API/LLM calls are made.
Tests verify mathematical bounds, edge cases, context item aggregation, and public behavior.
"""

from __future__ import annotations

import pytest

from context_health.relevance import (
    ContextItem,
    RelevantContextEstimator,
    RelevantContextState,
    calculate_relevance_from_items,
    calculate_relevant_context_ratio,
)


# ---------------------------------------------------------------------------
# 1–2. Empty context & Zero total tokens
# ---------------------------------------------------------------------------


class TestZeroAndEmptyContext:
    """Verify behavior when no tokens or an empty context are provided."""

    def test_default_empty_calculation(self) -> None:
        state = calculate_relevant_context_ratio()
        assert state.total_context_tokens == 0
        assert state.relevant_context_tokens == 0
        assert state.irrelevant_context_tokens == 0
        assert state.relevant_context_ratio == 0.0

    def test_zero_total_tokens(self) -> None:
        state = calculate_relevant_context_ratio(total_context_tokens=0, relevant_context_tokens=0)
        assert state.total_context_tokens == 0
        assert state.relevant_context_tokens == 0
        assert state.irrelevant_context_tokens == 0
        assert state.relevant_context_ratio == 0.0

    def test_empty_items_list(self) -> None:
        state = calculate_relevance_from_items([])
        assert state.total_context_tokens == 0
        assert state.relevant_context_tokens == 0
        assert state.irrelevant_context_tokens == 0
        assert state.relevant_context_ratio == 0.0


# ---------------------------------------------------------------------------
# 3–6. Ratios: 100%, 0%, 50%, and Arbitrary Partial
# ---------------------------------------------------------------------------


class TestRatios:
    """Verify calculation of various relevance fractions."""

    def test_100_percent_relevant(self) -> None:
        state = calculate_relevant_context_ratio(
            total_context_tokens=1000,
            relevant_context_tokens=1000,
        )
        assert state.total_context_tokens == 1000
        assert state.relevant_context_tokens == 1000
        assert state.irrelevant_context_tokens == 0
        assert state.relevant_context_ratio == 1.0

    def test_0_percent_relevant(self) -> None:
        state = calculate_relevant_context_ratio(
            total_context_tokens=1000,
            relevant_context_tokens=0,
        )
        assert state.total_context_tokens == 1000
        assert state.relevant_context_tokens == 0
        assert state.irrelevant_context_tokens == 1000
        assert state.relevant_context_ratio == 0.0

    def test_50_percent_relevant(self) -> None:
        state = calculate_relevant_context_ratio(
            total_context_tokens=2000,
            relevant_context_tokens=1000,
        )
        assert state.total_context_tokens == 2000
        assert state.relevant_context_tokens == 1000
        assert state.irrelevant_context_tokens == 1000
        assert state.relevant_context_ratio == pytest.approx(0.5)

    def test_arbitrary_partial_ratio(self) -> None:
        state = calculate_relevant_context_ratio(
            total_context_tokens=10_000,
            relevant_context_tokens=3_333,
        )
        assert state.total_context_tokens == 10_000
        assert state.relevant_context_tokens == 3_333
        assert state.irrelevant_context_tokens == 6_667
        assert state.relevant_context_ratio == pytest.approx(0.3333, abs=0.0001)


# ---------------------------------------------------------------------------
# 7. Relevant + Irrelevant equals Total
# ---------------------------------------------------------------------------


class TestPartitionIdentity:
    """Verify that relevant_tokens + irrelevant_tokens == total_tokens."""

    @pytest.mark.parametrize(
        ("total", "relevant"),
        [
            (100, 100),
            (100, 0),
            (100, 42),
            (50_000, 12_345),
            (200_000, 199_999),
        ],
    )
    def test_relevant_plus_irrelevant_equals_total(self, total: int, relevant: int) -> None:
        state = calculate_relevant_context_ratio(
            total_context_tokens=total,
            relevant_context_tokens=relevant,
        )
        assert state.relevant_context_tokens + state.irrelevant_context_tokens == state.total_context_tokens


# ---------------------------------------------------------------------------
# 8–11. Edge Cases: Excess, Negatives, Clamping, and Zero Bounds
# ---------------------------------------------------------------------------


class TestEdgeCasesAndSanitization:
    """Verify bounds, clamping, and handling of invalid/out-of-range inputs."""

    def test_relevant_greater_than_total(self) -> None:
        """When relevant > total, relevant is clamped to total."""
        state = calculate_relevant_context_ratio(
            total_context_tokens=1000,
            relevant_context_tokens=2500,
        )
        assert state.total_context_tokens == 1000
        assert state.relevant_context_tokens == 1000
        assert state.irrelevant_context_tokens == 0
        assert state.relevant_context_ratio == 1.0

    def test_negative_total_tokens(self) -> None:
        """Negative total is sanitized to 0."""
        state = calculate_relevant_context_ratio(
            total_context_tokens=-500,
            relevant_context_tokens=200,
        )
        assert state.total_context_tokens == 0
        assert state.relevant_context_tokens == 0
        assert state.irrelevant_context_tokens == 0
        assert state.relevant_context_ratio == 0.0

    def test_negative_relevant_tokens(self) -> None:
        """Negative relevant count is sanitized to 0."""
        state = calculate_relevant_context_ratio(
            total_context_tokens=1000,
            relevant_context_tokens=-200,
        )
        assert state.total_context_tokens == 1000
        assert state.relevant_context_tokens == 0
        assert state.irrelevant_context_tokens == 1000
        assert state.relevant_context_ratio == 0.0

    def test_floating_point_inputs(self) -> None:
        state = calculate_relevant_context_ratio(
            total_context_tokens=100.4,
            relevant_context_tokens=50.2,
        )
        assert state.total_context_tokens == 100
        assert state.relevant_context_tokens == 50
        assert state.irrelevant_context_tokens == 50
        assert state.relevant_context_ratio == 0.5

    def test_ratio_strictly_bounded(self) -> None:
        test_cases = [
            (-100, -100),
            (100, 200),
            (0, 100),
            (1_000_000, 500_000),
        ]
        for tot, rel in test_cases:
            state = calculate_relevant_context_ratio(
                total_context_tokens=tot,
                relevant_context_tokens=rel,
            )
            assert 0.0 <= state.relevant_context_ratio <= 1.0
            assert state.irrelevant_context_tokens >= 0


# ---------------------------------------------------------------------------
# 12–13. ContextItem Aggregation
# ---------------------------------------------------------------------------


class TestContextItemAggregation:
    """Verify calculating relevance by aggregating individual ContextItems."""

    def test_all_items_relevant(self) -> None:
        items = [
            ContextItem(identifier="file_1.py", token_count=500, relevant=True),
            ContextItem(identifier="file_2.py", token_count=300, relevant=True),
        ]
        state = calculate_relevance_from_items(items)
        assert state.total_context_tokens == 800
        assert state.relevant_context_tokens == 800
        assert state.irrelevant_context_tokens == 0
        assert state.relevant_context_ratio == 1.0

    def test_mixed_relevant_and_irrelevant_items(self) -> None:
        items = [
            ContextItem(identifier="relevant_doc.md", token_count=400, relevant=True),
            ContextItem(identifier="unrelated_log.txt", token_count=600, relevant=False),
            ContextItem(identifier="test_suite.py", token_count=200, relevant=True),
        ]
        state = calculate_relevance_from_items(items)
        # Total: 400 + 600 + 200 = 1200
        # Relevant: 400 + 200 = 600
        # Irrelevant: 600
        # Ratio: 600 / 1200 = 0.5
        assert state.total_context_tokens == 1200
        assert state.relevant_context_tokens == 600
        assert state.irrelevant_context_tokens == 600
        assert state.relevant_context_ratio == 0.5

    def test_items_with_negative_tokens_sanitized(self) -> None:
        items = [
            ContextItem(identifier="corrupted_item", token_count=-100, relevant=True),
            ContextItem(identifier="valid_item", token_count=500, relevant=True),
        ]
        state = calculate_relevance_from_items(items)
        assert state.total_context_tokens == 500
        assert state.relevant_context_tokens == 500
        assert state.relevant_context_ratio == 1.0


# ---------------------------------------------------------------------------
# 14. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Verify calculations are 100% deterministic."""

    def test_identical_inputs_identical_output(self) -> None:
        res1 = calculate_relevant_context_ratio(total_context_tokens=1500, relevant_context_tokens=750)
        res2 = calculate_relevant_context_ratio(total_context_tokens=1500, relevant_context_tokens=750)
        assert res1 == res2
        assert res1.relevant_context_ratio == res2.relevant_context_ratio


# ---------------------------------------------------------------------------
# 15. Offline / No API / No Network Calls
# ---------------------------------------------------------------------------


class TestOfflineIndependence:
    """Verify that relevance calculation has no network or LLM dependencies."""

    def test_no_forbidden_imports(self) -> None:
        import context_health.relevance as rel_mod

        with open(rel_mod.__file__) as f:
            code = f.read()

        for forbidden in ["import anthropic", "import httpx", "import requests", "import urllib"]:
            assert forbidden not in code, f"Forbidden import found: {forbidden}"
