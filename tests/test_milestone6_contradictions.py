"""Milestone 6 tests — contradiction detection.

ALL tests are deterministic. ZERO network/API/LLM calls are made.
Tests verify rule-based detection, false-positive protection, normalization, and public behavior.
"""

from __future__ import annotations

import pytest

from context_health.contradictions import (
    Contradiction,
    ContradictionDetector,
    ContradictionResult,
    detect_contradictions,
)


# ---------------------------------------------------------------------------
# 1–2. Empty input & Single statement
# ---------------------------------------------------------------------------


class TestEmptyAndSingleInput:
    """Verify behavior on empty sequences and single statements."""

    def test_empty_sequence(self) -> None:
        result = detect_contradictions([])
        assert result.contradiction_count == 0
        assert len(result.contradictions) == 0
        assert len(result) == 0

    def test_single_statement(self) -> None:
        result = detect_contradictions(["Use PostgreSQL for the primary database."])
        assert result.contradiction_count == 0
        assert len(result.contradictions) == 0


# ---------------------------------------------------------------------------
# 3–10. Core Action Contradiction Pairs
# ---------------------------------------------------------------------------


class TestCoreContradictionPairs:
    """Verify direct contradiction detection for all 7 supported action pairs."""

    def test_direct_use_do_not_use(self) -> None:
        statements = [
            "Use PostgreSQL.",
            "Do not use PostgreSQL.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 1
        c = result.contradictions[0]
        assert c.first_statement == "Use PostgreSQL."
        assert c.second_statement == "Do not use PostgreSQL."
        assert c.contradiction_type == "use_reversal"

    def test_dont_use_vs_use(self) -> None:
        statements = [
            "Don't use Redis.",
            "Use Redis.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 1
        c = result.contradictions[0]
        assert c.first_statement == "Don't use Redis."
        assert c.second_statement == "Use Redis."
        assert c.contradiction_type == "use_reversal"

    def test_enable_disable(self) -> None:
        statements = [
            "Enable caching.",
            "Disable caching.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 1
        c = result.contradictions[0]
        assert c.first_statement == "Enable caching."
        assert c.second_statement == "Disable caching."
        assert c.contradiction_type == "enable_reversal"

    def test_allow_disallow(self) -> None:
        statements = [
            "Allow anonymous access.",
            "Disallow anonymous access.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 1
        c = result.contradictions[0]
        assert c.first_statement == "Allow anonymous access."
        assert c.second_statement == "Disallow anonymous access."
        assert c.contradiction_type == "allow_reversal"

    def test_include_exclude(self) -> None:
        statements = [
            "Include legacy endpoints.",
            "Exclude legacy endpoints.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 1
        c = result.contradictions[0]
        assert c.first_statement == "Include legacy endpoints."
        assert c.second_statement == "Exclude legacy endpoints."
        assert c.contradiction_type == "include_reversal"

    def test_add_do_not_add(self) -> None:
        statements = [
            "Add schema validation.",
            "Do not add schema validation.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 1
        c = result.contradictions[0]
        assert c.first_statement == "Add schema validation."
        assert c.second_statement == "Do not add schema validation."
        assert c.contradiction_type == "add_reversal"

    def test_modify_do_not_modify(self) -> None:
        statements = [
            "Don't modify client.py.",
            "Modify client.py.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 1
        c = result.contradictions[0]
        assert c.first_statement == "Don't modify client.py."
        assert c.second_statement == "Modify client.py."
        assert c.contradiction_type == "modify_reversal"

    def test_remove_do_not_remove(self) -> None:
        statements = [
            "Remove deprecated helpers.",
            "Do not remove deprecated helpers.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 1
        c = result.contradictions[0]
        assert c.first_statement == "Remove deprecated helpers."
        assert c.second_statement == "Do not remove deprecated helpers."
        assert c.contradiction_type == "remove_reversal"


# ---------------------------------------------------------------------------
# 11–13. Normalization (Case, Whitespace, Punctuation)
# ---------------------------------------------------------------------------


class TestNormalization:
    """Verify target and command normalization."""

    def test_case_normalization(self) -> None:
        statements = [
            "USE REDIS FOR SESSIONS",
            "do not use redis for sessions",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 1

    def test_whitespace_normalization(self) -> None:
        statements = [
            "  Use    PostgreSQL   ",
            "Do not use PostgreSQL",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 1

    def test_punctuation_normalization(self) -> None:
        statements = [
            "Use SQLite!",
            "Do not use SQLite.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 1


# ---------------------------------------------------------------------------
# 14–15. Multiple & Non-Adjacent Contradictions
# ---------------------------------------------------------------------------


class TestComplexSequences:
    """Verify detection across sequences with multiple and non-adjacent conflicts."""

    def test_multiple_contradictions(self) -> None:
        statements = [
            "Use Redis for caching.",
            "Enable query logging.",
            "Do not use Redis for caching.",
            "Disable query logging.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 2
        types = [c.contradiction_type for c in result.contradictions]
        assert "use_reversal" in types
        assert "enable_reversal" in types

    def test_non_adjacent_contradiction(self) -> None:
        statements = [
            "Use PostgreSQL.",
            "Modify context.py.",
            "Run unit tests.",
            "Do not use PostgreSQL.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 1
        c = result.contradictions[0]
        assert c.first_statement == "Use PostgreSQL."
        assert c.second_statement == "Do not use PostgreSQL."


# ---------------------------------------------------------------------------
# 16–18. False-Positive Protections
# ---------------------------------------------------------------------------


class TestFalsePositiveProtection:
    """Verify that unrelated or concordant statements do NOT trigger contradictions."""

    def test_duplicate_identical_statements(self) -> None:
        statements = [
            "Use Redis.",
            "Use Redis.",
            "Use Redis.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 0

    def test_duplicate_negative_statements(self) -> None:
        statements = [
            "Do not use Redis.",
            "Do not use Redis.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 0

    def test_different_objects_not_contradictory(self) -> None:
        statements = [
            "Use Redis for caching.",
            "Use PostgreSQL for persistence.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 0

    def test_different_files_modified_not_contradictory(self) -> None:
        statements = [
            "Modify context.py.",
            "Modify complexity.py.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 0

    def test_different_features_enabled_not_contradictory(self) -> None:
        statements = [
            "Enable caching.",
            "Enable logging.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 0

    def test_same_polarity_extended_phrasing_not_contradictory(self) -> None:
        statements = [
            "Use Redis.",
            "Use Redis for caching.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 0

    def test_different_actions_same_target_not_contradictory(self) -> None:
        statements = [
            "Enable caching.",
            "Do not use caching.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 0


# ---------------------------------------------------------------------------
# 19. Empty strings and Malformed Inputs
# ---------------------------------------------------------------------------


class TestMalformedInputs:
    """Verify safe handling of empty and blank inputs."""

    def test_empty_strings_handled_safely(self) -> None:
        statements = ["", "   ", "\n\t", "Use Redis.", ""]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 0


# ---------------------------------------------------------------------------
# 20. Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Verify that identical inputs yield identical outputs."""

    def test_deterministic_output(self) -> None:
        statements = [
            "Allow root access.",
            "Disallow root access.",
        ]
        res1 = detect_contradictions(statements)
        res2 = detect_contradictions(statements)
        assert res1 == res2
        assert res1.contradiction_count == res2.contradiction_count


# ---------------------------------------------------------------------------
# 21. Contradiction Count & Result Interface
# ---------------------------------------------------------------------------


class TestResultInterface:
    """Verify ContradictionResult public properties and sequence protocol."""

    def test_result_properties_and_len(self) -> None:
        statements = [
            "Add telemetry.",
            "Do not add telemetry.",
        ]
        result = detect_contradictions(statements)
        assert result.contradiction_count == 1
        assert len(result) == 1
        assert result[0].contradiction_type == "add_reversal"
        for item in result:
            assert isinstance(item, Contradiction)


# ---------------------------------------------------------------------------
# 22. Incremental Detection Support
# ---------------------------------------------------------------------------


class TestIncrementalDetector:
    """Verify stateful ContradictionDetector.add_statement()."""

    def test_incremental_workflow(self) -> None:
        detector = ContradictionDetector()
        assert detector.add_statement("Use Redis.") == []
        assert detector.add_statement("Modify main.py.") == []

        new_c = detector.add_statement("Do not use Redis.")
        assert len(new_c) == 1
        assert new_c[0].contradiction_type == "use_reversal"
        assert detector.contradiction_count == 1

        detector.reset()
        assert detector.contradiction_count == 0


# ---------------------------------------------------------------------------
# 23. Offline & Provider Agnostic (Zero Network / LLM Calls)
# ---------------------------------------------------------------------------


class TestOfflineIndependence:
    """Verify no network or external LLM imports exist."""

    def test_no_forbidden_imports(self) -> None:
        import context_health.contradictions as cd_mod

        with open(cd_mod.__file__) as f:
            code = f.read()

        for forbidden in ["import anthropic", "import httpx", "import requests", "import urllib"]:
            assert forbidden not in code, f"Forbidden import found: {forbidden}"
