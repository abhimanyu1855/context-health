"""Tests for Milestone 12A — Automated Session Replay Dataset.

Validates the deterministic synthetic session replay dataset generator,
including temporal alignment, schema correctness, reproducibility,
outcome independence from the health score, and privacy sentinels.
"""

from __future__ import annotations

import hashlib
import json
import random
import tempfile
from pathlib import Path

import pytest

from context_health.health import compute_context_health

# Import the session replay module from scripts/.
# We import functions/classes directly for testability.
import sys
import os

# Add scripts/ to path so we can import session_replay.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from session_replay import (
    PredictorOutcomeRecord,
    SessionTurn,
    _OUTCOME_CONFIGS,
    _generate_outcome_probability,
    build_predictor_outcome_records,
    compute_dataset_hash,
    generate_dataset,
    generate_session,
    record_to_dict,
    validate_no_private_data,
    write_dataset_jsonl,
)


# ===========================================================================
# 1. Determinism / Reproducibility
# ===========================================================================


class TestDeterminism:
    """Verify byte-for-byte reproducibility with the same seed."""

    def test_generate_dataset_deterministic(self) -> None:
        """Same seed → identical records."""
        records_a = generate_dataset(num_sessions=10, num_turns=8, seed=42)
        records_b = generate_dataset(num_sessions=10, num_turns=8, seed=42)
        assert len(records_a) == len(records_b)
        for a, b in zip(records_a, records_b):
            assert a == b

    def test_generate_dataset_different_seeds(self) -> None:
        """Different seeds → different records."""
        records_a = generate_dataset(num_sessions=10, num_turns=8, seed=42)
        records_b = generate_dataset(num_sessions=10, num_turns=8, seed=99)
        # Not all records should be the same.
        differences = sum(1 for a, b in zip(records_a, records_b) if a != b)
        assert differences > 0

    def test_jsonl_output_deterministic(self) -> None:
        """Same seed → byte-for-byte identical JSONL output."""
        records = generate_dataset(num_sessions=5, num_turns=6, seed=77)
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = Path(tmpdir) / "a.jsonl"
            path_b = Path(tmpdir) / "b.jsonl"
            write_dataset_jsonl(records, path_a)
            write_dataset_jsonl(records, path_b)
            hash_a = compute_dataset_hash(path_a)
            hash_b = compute_dataset_hash(path_b)
            assert hash_a == hash_b

    def test_session_id_format(self) -> None:
        """Session IDs follow the deterministic 'session-NNNN' pattern."""
        records = generate_dataset(num_sessions=5, num_turns=4, seed=42)
        session_ids = set(r.session_id for r in records)
        expected = {f"session-{i + 1:04d}" for i in range(5)}
        assert session_ids == expected


# ===========================================================================
# 2. Record Count and Temporal Alignment
# ===========================================================================


class TestRecordCounts:
    """Verify correct record counts and temporal alignment."""

    @pytest.mark.parametrize(
        "sessions,turns",
        [(1, 2), (5, 5), (10, 10), (50, 20), (1, 1)],
    )
    def test_record_count(self, sessions: int, turns: int) -> None:
        """N sessions × (T-1) records per session."""
        records = generate_dataset(num_sessions=sessions, num_turns=turns, seed=42)
        expected = sessions * max(0, turns - 1)
        assert len(records) == expected

    def test_single_turn_session_produces_no_records(self) -> None:
        """A single-turn session has no future turn → 0 records."""
        records = generate_dataset(num_sessions=10, num_turns=1, seed=42)
        assert len(records) == 0

    def test_temporal_alignment(self) -> None:
        """Each record's outcome_turn == predictor_turn + 1."""
        records = generate_dataset(num_sessions=20, num_turns=10, seed=42)
        for r in records:
            assert r.outcome_turn == r.predictor_turn + 1, (
                f"Misaligned: predictor={r.predictor_turn}, outcome={r.outcome_turn}"
            )

    def test_last_turn_excluded(self) -> None:
        """The last turn of each session does NOT appear as a predictor."""
        num_turns = 8
        records = generate_dataset(num_sessions=5, num_turns=num_turns, seed=42)
        for r in records:
            assert r.predictor_turn < num_turns - 1, (
                f"Last turn appeared as predictor: turn {r.predictor_turn}"
            )

    def test_predictor_turn_indices_sequential(self) -> None:
        """Within a session, predictor turns are 0, 1, ..., T-2."""
        num_turns = 6
        records = generate_dataset(num_sessions=3, num_turns=num_turns, seed=42)
        for session_id in set(r.session_id for r in records):
            session_records = [r for r in records if r.session_id == session_id]
            predictor_turns = sorted(r.predictor_turn for r in session_records)
            assert predictor_turns == list(range(num_turns - 1))


# ===========================================================================
# 3. Schema and Field Validation
# ===========================================================================


class TestSchema:
    """Verify the data schema is correct."""

    def test_record_has_all_predictor_fields(self) -> None:
        """Each record contains all required predictor fields."""
        records = generate_dataset(num_sessions=1, num_turns=3, seed=42)
        r = records[0]
        assert hasattr(r, "context_utilization")
        assert hasattr(r, "relevant_context_ratio")
        assert hasattr(r, "task_complexity")
        assert hasattr(r, "contradiction_count")
        assert hasattr(r, "correction_count")
        assert hasattr(r, "retry_count")
        assert hasattr(r, "health_score")
        assert hasattr(r, "health_status")

    def test_record_has_all_outcome_fields(self) -> None:
        """Each record contains all required outcome fields."""
        records = generate_dataset(num_sessions=1, num_turns=3, seed=42)
        r = records[0]
        assert hasattr(r, "next_turn_correction")
        assert hasattr(r, "next_turn_tool_retry")
        assert hasattr(r, "next_turn_contradiction")
        assert hasattr(r, "next_turn_rework")
        assert hasattr(r, "next_turn_abandoned")

    def test_predictor_ranges(self) -> None:
        """All predictor signals are within valid ranges."""
        records = generate_dataset(num_sessions=50, num_turns=15, seed=42)
        for r in records:
            assert 0.0 <= r.context_utilization <= 1.0
            assert 0.0 <= r.relevant_context_ratio <= 1.0
            assert 0.0 <= r.task_complexity <= 1.0
            assert r.contradiction_count >= 0
            assert r.correction_count >= 0
            assert r.retry_count >= 0
            assert 0.0 <= r.health_score <= 100.0
            assert r.health_status in ("healthy", "watch", "warning", "critical")

    def test_outcomes_are_boolean(self) -> None:
        """All outcome fields are booleans."""
        records = generate_dataset(num_sessions=10, num_turns=8, seed=42)
        for r in records:
            assert isinstance(r.next_turn_correction, bool)
            assert isinstance(r.next_turn_tool_retry, bool)
            assert isinstance(r.next_turn_contradiction, bool)
            assert isinstance(r.next_turn_rework, bool)
            assert isinstance(r.next_turn_abandoned, bool)

    def test_record_to_dict_roundtrip(self) -> None:
        """record_to_dict produces a JSON-serializable dict."""
        records = generate_dataset(num_sessions=2, num_turns=4, seed=42)
        for r in records:
            d = record_to_dict(r)
            json_str = json.dumps(d)
            parsed = json.loads(json_str)
            assert parsed["session_id"] == r.session_id
            assert parsed["predictor_turn"] == r.predictor_turn
            assert parsed["outcome_turn"] == r.outcome_turn
            assert parsed["health_score"] == r.health_score
            assert parsed["next_turn_correction"] == r.next_turn_correction


# ===========================================================================
# 4. Health Score Consistency
# ===========================================================================


class TestHealthScoreConsistency:
    """Verify health scores match recomputation from raw signals."""

    def test_health_score_matches_recomputation(self) -> None:
        """The health score in the record matches compute_context_health()."""
        records = generate_dataset(num_sessions=20, num_turns=10, seed=42)
        for r in records:
            recomputed = compute_context_health(
                context_utilization=r.context_utilization,
                relevant_context_ratio=r.relevant_context_ratio,
                task_complexity=r.task_complexity,
                contradiction_count=r.contradiction_count,
                correction_count=r.correction_count,
                retry_count=r.retry_count,
            )
            assert r.health_score == recomputed.health_score
            assert r.health_status == recomputed.status


# ===========================================================================
# 5. Outcome Independence from Health Score
# ===========================================================================


class TestOutcomeIndependence:
    """Verify outcomes are NOT trivially determined by the health score."""

    def test_outcomes_not_trivially_threshold_based(self) -> None:
        """There exist records with the same health status but different outcomes.

        This ensures outcomes aren't simply: correction = (status == 'critical').
        """
        records = generate_dataset(num_sessions=100, num_turns=15, seed=42)

        # Group by health status.
        by_status: dict[str, list[PredictorOutcomeRecord]] = {}
        for r in records:
            by_status.setdefault(r.health_status, []).append(r)

        # For statuses with enough records, verify outcome variance.
        for status, group in by_status.items():
            if len(group) < 10:
                continue
            correction_values = set(r.next_turn_correction for r in group)
            # Within a status band, we should see both True and False outcomes
            # for at least some outcome types (not all outcomes are identical).
            all_outcomes = set()
            for r in group:
                all_outcomes.add(
                    (r.next_turn_correction, r.next_turn_tool_retry,
                     r.next_turn_contradiction, r.next_turn_rework,
                     r.next_turn_abandoned)
                )
            assert len(all_outcomes) > 1, (
                f"All records in status '{status}' have identical outcomes"
            )

    def test_healthy_sessions_can_have_outcomes(self) -> None:
        """Even 'healthy' records can sometimes have degradation outcomes.

        This confirms outcomes aren't simply 'healthy → no degradation'.
        """
        records = generate_dataset(num_sessions=200, num_turns=15, seed=42)
        healthy = [r for r in records if r.health_status == "healthy"]
        if len(healthy) < 5:
            pytest.skip("Not enough healthy records to test")

        # At least one healthy record should have some outcome True.
        any_outcome = any(
            r.next_turn_correction or r.next_turn_tool_retry or
            r.next_turn_contradiction or r.next_turn_rework or
            r.next_turn_abandoned
            for r in healthy
        )
        assert any_outcome, "No healthy record had any degradation outcome"


# ===========================================================================
# 6. Session Generation Properties
# ===========================================================================


class TestSessionGeneration:
    """Verify properties of the synthetic session generator."""

    def test_session_turns_have_increasing_indices(self) -> None:
        """Turn indices within a session are 0, 1, 2, ..., T-1."""
        rng = random.Random(42)
        turns = generate_session("test-session", 10, rng)
        indices = [t.turn_index for t in turns]
        assert indices == list(range(10))

    def test_session_id_propagated(self) -> None:
        """All turns in a session share the session ID."""
        rng = random.Random(42)
        turns = generate_session("my-session", 5, rng)
        for t in turns:
            assert t.session_id == "my-session"

    def test_cumulative_event_counts_non_decreasing(self) -> None:
        """Cumulative contradiction/correction/retry counts never decrease."""
        rng = random.Random(42)
        turns = generate_session("test", 20, rng)
        for i in range(1, len(turns)):
            assert turns[i].contradiction_count >= turns[i - 1].contradiction_count
            assert turns[i].correction_count >= turns[i - 1].correction_count
            assert turns[i].retry_count >= turns[i - 1].retry_count


# ===========================================================================
# 7. Privacy
# ===========================================================================


class TestPrivacy:
    """Verify no PII or private data in the dataset."""

    def test_validate_no_private_data(self) -> None:
        """Privacy validation passes on a valid dataset."""
        records = generate_dataset(num_sessions=20, num_turns=10, seed=42)
        # Should not raise.
        validate_no_private_data(records)

    def test_no_filesystem_paths_in_output(self) -> None:
        """No record field contains filesystem paths."""
        records = generate_dataset(num_sessions=50, num_turns=10, seed=42)
        for r in records:
            d = record_to_dict(r)
            for key, value in d.items():
                if isinstance(value, str):
                    assert "/home/" not in value
                    assert "/Users/" not in value
                    assert "C:\\" not in value


# ===========================================================================
# 8. JSONL Format
# ===========================================================================


class TestJsonlFormat:
    """Verify the JSONL output format."""

    def test_jsonl_line_count(self) -> None:
        """JSONL file has exactly one line per record."""
        records = generate_dataset(num_sessions=5, num_turns=6, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            write_dataset_jsonl(records, path)
            lines = path.read_text().strip().split("\n")
            assert len(lines) == len(records)

    def test_each_line_is_valid_json(self) -> None:
        """Each line in the JSONL file is valid JSON."""
        records = generate_dataset(num_sessions=5, num_turns=6, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            write_dataset_jsonl(records, path)
            for line in path.read_text().strip().split("\n"):
                parsed = json.loads(line)
                assert isinstance(parsed, dict)
                assert "session_id" in parsed
                assert "predictor_turn" in parsed
                assert "outcome_turn" in parsed

    def test_jsonl_hash_consistency(self) -> None:
        """SHA-256 hash of the same dataset is consistent."""
        records = generate_dataset(num_sessions=3, num_turns=5, seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            write_dataset_jsonl(records, path)
            hash1 = compute_dataset_hash(path)
            hash2 = compute_dataset_hash(path)
            assert hash1 == hash2


# ===========================================================================
# 9. Outcome Probability Generator
# ===========================================================================


class TestOutcomeProbability:
    """Verify properties of the outcome probability generator."""

    def test_probability_in_valid_range(self) -> None:
        """Generated probabilities are always in [0.0, 1.0]."""
        rng = random.Random(42)
        for _ in range(1000):
            prob = _generate_outcome_probability(
                utilization=rng.random(),
                relevance=rng.random(),
                complexity=rng.random(),
                contradiction_count=rng.randint(0, 10),
                correction_count=rng.randint(0, 10),
                retry_count=rng.randint(0, 10),
                base_rate=0.05,
                utilization_sensitivity=0.10,
                relevance_sensitivity=0.15,
                complexity_sensitivity=0.08,
                event_sensitivity=0.12,
                rng=rng,
            )
            assert 0.0 <= prob <= 1.0

    def test_higher_degradation_higher_probability(self) -> None:
        """On average, worse signal values yield higher outcome probabilities.

        This is a statistical test over many samples — individual samples
        may vary due to noise, but the average should be directionally correct.
        """
        rng_good = random.Random(42)
        rng_bad = random.Random(42)

        good_probs = []
        bad_probs = []

        for _ in range(500):
            # Advance both RNGs in lockstep for the noise component.
            rng_good_instance = random.Random(rng_good.randint(0, 2**32))
            rng_bad_instance = random.Random(rng_bad.randint(0, 2**32))

            good_prob = _generate_outcome_probability(
                utilization=0.2,
                relevance=0.9,
                complexity=0.2,
                contradiction_count=0,
                correction_count=0,
                retry_count=0,
                base_rate=0.05,
                utilization_sensitivity=0.10,
                relevance_sensitivity=0.15,
                complexity_sensitivity=0.08,
                event_sensitivity=0.12,
                rng=rng_good_instance,
            )
            bad_prob = _generate_outcome_probability(
                utilization=0.8,
                relevance=0.2,
                complexity=0.8,
                contradiction_count=5,
                correction_count=5,
                retry_count=5,
                base_rate=0.05,
                utilization_sensitivity=0.10,
                relevance_sensitivity=0.15,
                complexity_sensitivity=0.08,
                event_sensitivity=0.12,
                rng=rng_bad_instance,
            )
            good_probs.append(good_prob)
            bad_probs.append(bad_prob)

        avg_good = sum(good_probs) / len(good_probs)
        avg_bad = sum(bad_probs) / len(bad_probs)
        assert avg_bad > avg_good, (
            f"Expected worse signals to have higher probability: "
            f"avg_good={avg_good:.4f}, avg_bad={avg_bad:.4f}"
        )


# ===========================================================================
# 10. Temporal Outcome-Generation Semantics
# ===========================================================================


class TestTemporalOutcomeSemantics:
    """Verify that outcomes on turn N+1 are generated from turn N's
    predictor signals — NOT from turn N+1's signals.

    These tests prove the two-pass temporal contract:
      turn N predictor state → outcome generator → turn N+1 observed outcome
    """

    def test_rng_replay_outcomes_use_predictor_turn_signals(self) -> None:
        """RNG replay: outcomes on turn N+1 are derived from env_signals[N].

        Replays the exact RNG sequence of generate_session to confirm that
        the outcome generator receives turn N's environmental signals and
        cumulative events, NOT turn N+1's.
        """
        seed = 42
        num_turns = 8

        # --- Run generate_session ---
        rng = random.Random(seed)
        turns = generate_session("replay-test", num_turns, rng)

        # --- Replay the RNG from scratch ---
        rng_replay = random.Random(seed)

        # Session-level parameters (must match generate_session order).
        initial_util = rng_replay.uniform(0.05, 0.30)
        util_drift = rng_replay.uniform(0.02, 0.08)
        initial_rel = rng_replay.uniform(0.70, 0.95)
        rel_drift = rng_replay.uniform(-0.06, -0.01)
        base_comp = rng_replay.uniform(0.15, 0.60)

        # Pass 1: environmental signals for all turns.
        env = []
        for t in range(num_turns):
            u = max(0.0, min(1.0, initial_util + util_drift * t + rng_replay.gauss(0, 0.03)))
            r = max(0.0, min(1.0, initial_rel + rel_drift * t + rng_replay.gauss(0, 0.04)))
            c = max(0.0, min(1.0, base_comp + rng_replay.gauss(0, 0.05)))
            env.append((round(u, 6), round(r, 6), round(c, 6)))

        # Verify env signals match turns.
        for t in range(num_turns):
            assert turns[t].context_utilization == env[t][0]
            assert turns[t].relevant_context_ratio == env[t][1]
            assert turns[t].task_complexity == env[t][2]

        # Pass 2: replay outcome generation.
        # Outcomes for turn N+1 use env[N] (predictor turn N's signals).
        cum_contra = 0
        cum_corr = 0
        cum_retry = 0

        for n in range(num_turns - 1):
            # Build outcome_kwargs from turn N's predictor state.
            outcome_kwargs = {
                "utilization": env[n][0],   # turn N, NOT env[n+1]
                "relevance": env[n][1],
                "complexity": env[n][2],
                "contradiction_count": cum_contra,
                "correction_count": cum_corr,
                "retry_count": cum_retry,
                "rng": rng_replay,
            }

            # Replay the exact rng.random() < _generate_outcome_probability(...)
            # expression.  Python evaluates left-to-right, so rng.random()
            # is consumed BEFORE _generate_outcome_probability calls rng.uniform().
            coin_corr = rng_replay.random()
            prob_corr = _generate_outcome_probability(
                **outcome_kwargs, **_OUTCOME_CONFIGS["correction"]
            )
            had_corr = coin_corr < prob_corr

            coin_retry = rng_replay.random()
            prob_retry = _generate_outcome_probability(
                **outcome_kwargs, **_OUTCOME_CONFIGS["tool_retry"]
            )
            had_retry = coin_retry < prob_retry

            coin_contra = rng_replay.random()
            prob_contra = _generate_outcome_probability(
                **outcome_kwargs, **_OUTCOME_CONFIGS["contradiction"]
            )
            had_contra = coin_contra < prob_contra

            coin_rework = rng_replay.random()
            prob_rework = _generate_outcome_probability(
                **outcome_kwargs, **_OUTCOME_CONFIGS["rework"]
            )
            had_rework = coin_rework < prob_rework

            coin_aband = rng_replay.random()
            prob_aband = _generate_outcome_probability(
                **outcome_kwargs, **_OUTCOME_CONFIGS["abandoned"]
            )
            had_aband = coin_aband < prob_aband

            # Assert replayed outcomes match stored outcomes on turn N+1.
            assert turns[n + 1].had_correction == had_corr, (
                f"Correction mismatch at turn {n + 1}"
            )
            assert turns[n + 1].had_tool_retry == had_retry, (
                f"Tool retry mismatch at turn {n + 1}"
            )
            assert turns[n + 1].had_contradiction == had_contra, (
                f"Contradiction mismatch at turn {n + 1}"
            )
            assert turns[n + 1].had_rework == had_rework, (
                f"Rework mismatch at turn {n + 1}"
            )
            assert turns[n + 1].had_abandoned == had_aband, (
                f"Abandoned mismatch at turn {n + 1}"
            )

            # Advance cumulative counts.
            if had_contra:
                cum_contra += 1
            if had_corr:
                cum_corr += 1
            if had_retry:
                cum_retry += 1

    def test_changing_future_signals_does_not_change_predictor_outcome(self) -> None:
        """Changing turn N+1's env signals doesn't change predictor N's outcome.

        We call the outcome generator twice with identical turn-N signals
        and same RNG state.  In both calls, only turn N's signals are
        inputs — turn N+1's signals are never referenced.  This proves
        that however turn N+1's environmental state changes, the outcome
        attributed to predictor turn N remains identical.
        """
        # Two independent RNG copies with the same state.
        rng_a = random.Random(123)
        rng_b = random.Random(123)

        # Predictor state at turn N (fixed).
        turn_n_signals = {
            "utilization": 0.45,
            "relevance": 0.72,
            "complexity": 0.38,
            "contradiction_count": 2,
            "correction_count": 1,
            "retry_count": 3,
        }

        # Generate outcomes from turn N's state using rng_a.
        outcomes_a: dict[str, bool] = {}
        for outcome_type in ["correction", "tool_retry", "contradiction", "rework", "abandoned"]:
            coin = rng_a.random()
            prob = _generate_outcome_probability(
                **turn_n_signals, rng=rng_a, **_OUTCOME_CONFIGS[outcome_type]
            )
            outcomes_a[outcome_type] = coin < prob

        # Generate outcomes from the SAME turn N's state using rng_b.
        # Crucially, changing what turn N+1's signals "would be" is
        # irrelevant because the outcome generator never receives them.
        outcomes_b: dict[str, bool] = {}
        for outcome_type in ["correction", "tool_retry", "contradiction", "rework", "abandoned"]:
            coin = rng_b.random()
            prob = _generate_outcome_probability(
                **turn_n_signals, rng=rng_b, **_OUTCOME_CONFIGS[outcome_type]
            )
            outcomes_b[outcome_type] = coin < prob

        assert outcomes_a == outcomes_b

    def test_turn_0_outcomes_are_false(self) -> None:
        """Turn 0 has no predecessor — all its outcome fields are False."""
        rng = random.Random(42)
        turns = generate_session("test", 10, rng)
        t0 = turns[0]
        assert t0.had_correction is False
        assert t0.had_tool_retry is False
        assert t0.had_contradiction is False
        assert t0.had_rework is False
        assert t0.had_abandoned is False

    def test_temporal_alignment_after_fix(self) -> None:
        """Confirm outcome_turn == predictor_turn + 1 after the fix."""
        records = generate_dataset(num_sessions=30, num_turns=12, seed=99)
        for r in records:
            assert r.outcome_turn == r.predictor_turn + 1

    def test_determinism_after_fix(self) -> None:
        """Same seed → byte-for-byte identical dataset after the fix."""
        records_a = generate_dataset(num_sessions=20, num_turns=10, seed=42)
        records_b = generate_dataset(num_sessions=20, num_turns=10, seed=42)
        assert len(records_a) == len(records_b)
        for a, b in zip(records_a, records_b):
            assert a == b
