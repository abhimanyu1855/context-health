#!/usr/bin/env python3
"""Automated Session Replay Dataset Generator (Milestone 12A).

Creates a deterministic synthetic dataset of coding-agent sessions for
investigating whether context state at turn N predicts observable
degradation at turn N+1.

Data Model
----------
Each row in the output dataset represents ONE turn with:

- **Predictor fields** (turn N): observable context signals and the
  computed Context Health score at turn N.
- **Outcome fields** (turn N+1): binary/count degradation indicators
  observed at turn N+1.

The last turn of each session is excluded because there is no future
turn to provide outcome data.

Key Design Constraints
----------------------
1. **Temporal split**: Predictors come from turn N; outcomes come from
   turn N+1.  No future information leaks into predictors.
2. **Outcome generator independence**: Outcome probabilities are derived
   from the underlying *predictor signals* (utilization, relevance,
   complexity, event counts) with independent noise — NOT from the
   computed Context Health score.  The health score is a predictor,
   not ground truth.
3. **Determinism**: Given the same seed, sessions/turns/seed
   configuration, the output is byte-for-byte reproducible.
4. **Standard library only**: No pandas, numpy, scipy, or sklearn.

Important Disclaimers
---------------------
This is EXPERIMENTAL INFRASTRUCTURE for offline analysis.

It does NOT:
- Validate the Context Health hypothesis on real sessions.
- Prove that context signals predict degradation.
- Modify the Context Health scoring engine.

The synthetic outcome generator encodes *plausible* relationships
between predictor signals and degradation outcomes for pipeline
testing purposes only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from context_health.health import compute_context_health


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionTurn:
    """Raw turn-level state for a synthetic session."""

    session_id: str
    turn_index: int

    # Context signals
    context_utilization: float
    relevant_context_ratio: float
    task_complexity: float
    contradiction_count: int
    correction_count: int
    retry_count: int

    # Outcome signals (observed at THIS turn, used as N+1 outcome for previous turn)
    had_correction: bool
    had_tool_retry: bool
    had_contradiction: bool
    had_rework: bool
    had_abandoned: bool


@dataclass(frozen=True)
class PredictorOutcomeRecord:
    """Temporal predictor→outcome record for the dataset.

    Predictors are from turn N.
    Outcomes are from turn N+1.
    """

    # Identity
    session_id: str
    predictor_turn: int
    outcome_turn: int

    # --- Predictor fields (turn N) ---
    context_utilization: float
    relevant_context_ratio: float
    task_complexity: float
    contradiction_count: int
    correction_count: int
    retry_count: int
    health_score: float
    health_status: str

    # --- Outcome fields (turn N+1) ---
    next_turn_correction: bool
    next_turn_tool_retry: bool
    next_turn_contradiction: bool
    next_turn_rework: bool
    next_turn_abandoned: bool


# ---------------------------------------------------------------------------
# Session generation
# ---------------------------------------------------------------------------


def _generate_outcome_probability(
    *,
    utilization: float,
    relevance: float,
    complexity: float,
    contradiction_count: int,
    correction_count: int,
    retry_count: int,
    base_rate: float,
    utilization_sensitivity: float,
    relevance_sensitivity: float,
    complexity_sensitivity: float,
    event_sensitivity: float,
    rng: random.Random,
) -> float:
    """Compute outcome probability from underlying predictor signals.

    The probability is a function of the raw signals, NOT the health score.
    Each signal contributes independently with configurable sensitivity,
    plus uniform noise.

    This is a synthetic generator for pipeline testing — it encodes
    plausible (not validated) relationships.
    """
    # Each factor pushes probability up from the base rate.
    # Higher utilization → higher degradation probability.
    util_factor = utilization * utilization_sensitivity
    # Lower relevance → higher degradation probability.
    irrel_factor = (1.0 - relevance) * relevance_sensitivity
    # Higher complexity → higher degradation probability.
    comp_factor = complexity * complexity_sensitivity
    # More events → higher degradation (saturating).
    total_events = contradiction_count + correction_count + retry_count
    event_factor = (total_events / (total_events + 5.0)) * event_sensitivity

    raw_prob = base_rate + util_factor + irrel_factor + comp_factor + event_factor

    # Add uniform noise in [-0.08, +0.08] to prevent trivial decodability.
    noise = rng.uniform(-0.08, 0.08)
    noisy_prob = raw_prob + noise

    return max(0.0, min(1.0, noisy_prob))


# Outcome configuration: base_rate, sensitivities per signal.
# These are chosen so that outcomes are NOT trivially predictable from
# any single signal, and the overall rates are realistic-ish.
_OUTCOME_CONFIGS: dict[str, dict[str, float]] = {
    "correction": {
        "base_rate": 0.05,
        "utilization_sensitivity": 0.10,
        "relevance_sensitivity": 0.15,
        "complexity_sensitivity": 0.08,
        "event_sensitivity": 0.12,
    },
    "tool_retry": {
        "base_rate": 0.06,
        "utilization_sensitivity": 0.08,
        "relevance_sensitivity": 0.10,
        "complexity_sensitivity": 0.12,
        "event_sensitivity": 0.15,
    },
    "contradiction": {
        "base_rate": 0.03,
        "utilization_sensitivity": 0.12,
        "relevance_sensitivity": 0.12,
        "complexity_sensitivity": 0.06,
        "event_sensitivity": 0.10,
    },
    "rework": {
        "base_rate": 0.04,
        "utilization_sensitivity": 0.12,
        "relevance_sensitivity": 0.18,
        "complexity_sensitivity": 0.10,
        "event_sensitivity": 0.10,
    },
    "abandoned": {
        "base_rate": 0.01,
        "utilization_sensitivity": 0.06,
        "relevance_sensitivity": 0.08,
        "complexity_sensitivity": 0.04,
        "event_sensitivity": 0.05,
    },
}


def generate_session(
    session_id: str,
    num_turns: int,
    rng: random.Random,
) -> list[SessionTurn]:
    """Generate a single synthetic session with deterministic signals.

    The session simulates a coding-agent conversation where context
    utilization generally increases and relevance generally decreases
    over turns, with noise.

    Two-pass architecture
    ---------------------
    **Pass 1** generates environmental predictor signals (utilization,
    relevance, complexity) for every turn up front.

    **Pass 2** generates outcomes.  The outcome observed at turn N+1 is
    generated from the predictor state available at turn N — i.e. turn
    N's environmental signals and cumulative event counts.  Turn 0 has
    no predecessor, so its outcome fields are ``False``.

    This ensures the temporal contract:  turn N predictor signals →
    future behavioral risk → turn N+1 observed outcome.  No future
    information (turn N+1's signals) leaks into the outcome that is
    attributed to predictor turn N.
    """
    # Session-level characteristics (sampled once per session).
    initial_utilization = rng.uniform(0.05, 0.30)
    utilization_drift = rng.uniform(0.02, 0.08)  # per-turn increase
    initial_relevance = rng.uniform(0.70, 0.95)
    relevance_drift = rng.uniform(-0.06, -0.01)  # per-turn decrease
    base_complexity = rng.uniform(0.15, 0.60)

    # ------------------------------------------------------------------
    # Pass 1: Generate environmental signals for ALL turns.
    # ------------------------------------------------------------------
    env_signals: list[tuple[float, float, float]] = []
    for t in range(num_turns):
        util_raw = initial_utilization + utilization_drift * t + rng.gauss(0, 0.03)
        utilization = max(0.0, min(1.0, util_raw))

        rel_raw = initial_relevance + relevance_drift * t + rng.gauss(0, 0.04)
        relevance = max(0.0, min(1.0, rel_raw))

        comp_raw = base_complexity + rng.gauss(0, 0.05)
        complexity = max(0.0, min(1.0, comp_raw))

        env_signals.append((
            round(utilization, 6),
            round(relevance, 6),
            round(complexity, 6),
        ))

    # ------------------------------------------------------------------
    # Pass 2: Generate outcomes.
    #
    # Turn 0 has no predecessor → outcomes are False.
    # For each predictor turn N (0 … T-2), generate the outcome that
    # will be observed at turn N+1, using turn N's predictor signals.
    # ------------------------------------------------------------------
    turns: list[SessionTurn] = []

    cumulative_contradictions = 0
    cumulative_corrections = 0
    cumulative_retries = 0

    # --- Turn 0: no predecessor, outcomes are False ---
    turns.append(
        SessionTurn(
            session_id=session_id,
            turn_index=0,
            context_utilization=env_signals[0][0],
            relevant_context_ratio=env_signals[0][1],
            task_complexity=env_signals[0][2],
            contradiction_count=cumulative_contradictions,
            correction_count=cumulative_corrections,
            retry_count=cumulative_retries,
            had_correction=False,
            had_tool_retry=False,
            had_contradiction=False,
            had_rework=False,
            had_abandoned=False,
        )
    )

    # --- Turns 1 … T-1: outcomes generated from PREVIOUS turn's state ---
    for n in range(num_turns - 1):
        # Predictor state at turn N: environmental signals + cumulative events.
        pred_util, pred_rel, pred_comp = env_signals[n]

        outcome_kwargs = {
            "utilization": pred_util,
            "relevance": pred_rel,
            "complexity": pred_comp,
            "contradiction_count": cumulative_contradictions,
            "correction_count": cumulative_corrections,
            "retry_count": cumulative_retries,
            "rng": rng,
        }

        had_correction = rng.random() < _generate_outcome_probability(
            **outcome_kwargs, **_OUTCOME_CONFIGS["correction"]
        )
        had_tool_retry = rng.random() < _generate_outcome_probability(
            **outcome_kwargs, **_OUTCOME_CONFIGS["tool_retry"]
        )
        had_contradiction = rng.random() < _generate_outcome_probability(
            **outcome_kwargs, **_OUTCOME_CONFIGS["contradiction"]
        )
        had_rework = rng.random() < _generate_outcome_probability(
            **outcome_kwargs, **_OUTCOME_CONFIGS["rework"]
        )
        had_abandoned = rng.random() < _generate_outcome_probability(
            **outcome_kwargs, **_OUTCOME_CONFIGS["abandoned"]
        )

        # Accumulate event counts for subsequent predictor states.
        if had_contradiction:
            cumulative_contradictions += 1
        if had_correction:
            cumulative_corrections += 1
        if had_tool_retry:
            cumulative_retries += 1

        turns.append(
            SessionTurn(
                session_id=session_id,
                turn_index=n + 1,
                context_utilization=env_signals[n + 1][0],
                relevant_context_ratio=env_signals[n + 1][1],
                task_complexity=env_signals[n + 1][2],
                contradiction_count=cumulative_contradictions,
                correction_count=cumulative_corrections,
                retry_count=cumulative_retries,
                had_correction=had_correction,
                had_tool_retry=had_tool_retry,
                had_contradiction=had_contradiction,
                had_rework=had_rework,
                had_abandoned=had_abandoned,
            )
        )

    return turns


def build_predictor_outcome_records(
    turns: list[SessionTurn],
) -> list[PredictorOutcomeRecord]:
    """Build temporal predictor→outcome records from session turns.

    For each turn N (where N < last turn), create a record with:
    - Predictors from turn N (including the computed health score).
    - Outcomes from turn N+1.

    The last turn is excluded because no future outcome exists.
    """
    records: list[PredictorOutcomeRecord] = []

    for i in range(len(turns) - 1):
        turn_n = turns[i]
        turn_n1 = turns[i + 1]

        # Compute health score for turn N using the existing API.
        health_result = compute_context_health(
            context_utilization=turn_n.context_utilization,
            relevant_context_ratio=turn_n.relevant_context_ratio,
            task_complexity=turn_n.task_complexity,
            contradiction_count=turn_n.contradiction_count,
            correction_count=turn_n.correction_count,
            retry_count=turn_n.retry_count,
        )

        records.append(
            PredictorOutcomeRecord(
                session_id=turn_n.session_id,
                predictor_turn=turn_n.turn_index,
                outcome_turn=turn_n1.turn_index,
                # Predictors
                context_utilization=turn_n.context_utilization,
                relevant_context_ratio=turn_n.relevant_context_ratio,
                task_complexity=turn_n.task_complexity,
                contradiction_count=turn_n.contradiction_count,
                correction_count=turn_n.correction_count,
                retry_count=turn_n.retry_count,
                health_score=health_result.health_score,
                health_status=health_result.status,
                # Outcomes (from turn N+1)
                next_turn_correction=turn_n1.had_correction,
                next_turn_tool_retry=turn_n1.had_tool_retry,
                next_turn_contradiction=turn_n1.had_contradiction,
                next_turn_rework=turn_n1.had_rework,
                next_turn_abandoned=turn_n1.had_abandoned,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------


def generate_dataset(
    *,
    num_sessions: int,
    num_turns: int,
    seed: int,
) -> list[PredictorOutcomeRecord]:
    """Generate the complete predictor→outcome dataset.

    Parameters
    ----------
    num_sessions:
        Number of synthetic sessions to generate.
    num_turns:
        Number of turns per session.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    list[PredictorOutcomeRecord]
        The complete dataset of predictor→outcome records.
    """
    rng = random.Random(seed)
    all_records: list[PredictorOutcomeRecord] = []

    for i in range(num_sessions):
        session_id = f"session-{i + 1:04d}"
        turns = generate_session(session_id, num_turns, rng)
        records = build_predictor_outcome_records(turns)
        all_records.extend(records)

    return all_records


def record_to_dict(record: PredictorOutcomeRecord) -> dict[str, Any]:
    """Convert a PredictorOutcomeRecord to a JSON-serializable dict."""
    return asdict(record)


def write_dataset_jsonl(
    records: list[PredictorOutcomeRecord],
    output_path: Path,
) -> None:
    """Write the dataset to a JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            line = json.dumps(record_to_dict(record), separators=(",", ":"))
            f.write(line + "\n")


def compute_dataset_hash(output_path: Path) -> str:
    """Compute SHA-256 hash of the dataset file for determinism verification."""
    h = hashlib.sha256()
    with open(output_path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Dataset statistics
# ---------------------------------------------------------------------------


def print_dataset_statistics(records: list[PredictorOutcomeRecord]) -> None:
    """Print summary statistics for the generated dataset."""
    n = len(records)
    if n == 0:
        print("  No records generated.")
        return

    # Session counts
    sessions = set(r.session_id for r in records)

    # Outcome rates
    correction_count = sum(1 for r in records if r.next_turn_correction)
    retry_count = sum(1 for r in records if r.next_turn_tool_retry)
    contradiction_count = sum(1 for r in records if r.next_turn_contradiction)
    rework_count = sum(1 for r in records if r.next_turn_rework)
    abandoned_count = sum(1 for r in records if r.next_turn_abandoned)

    # Predictor summary
    health_scores = [r.health_score for r in records]
    utilizations = [r.context_utilization for r in records]
    relevances = [r.relevant_context_ratio for r in records]

    print(f"  Total records: {n}")
    print(f"  Sessions:      {len(sessions)}")
    print(f"  Records/session: {n // len(sessions)}")
    print()
    print("  Outcome rates:")
    print(f"    next_turn_correction:    {correction_count:4d} ({correction_count / n * 100:5.1f}%)")
    print(f"    next_turn_tool_retry:    {retry_count:4d} ({retry_count / n * 100:5.1f}%)")
    print(f"    next_turn_contradiction: {contradiction_count:4d} ({contradiction_count / n * 100:5.1f}%)")
    print(f"    next_turn_rework:        {rework_count:4d} ({rework_count / n * 100:5.1f}%)")
    print(f"    next_turn_abandoned:     {abandoned_count:4d} ({abandoned_count / n * 100:5.1f}%)")
    print()
    print("  Predictor distributions:")
    print(f"    health_score:         min={min(health_scores):6.2f}  max={max(health_scores):6.2f}  mean={sum(health_scores) / n:6.2f}")
    print(f"    context_utilization:  min={min(utilizations):6.4f}  max={max(utilizations):6.4f}  mean={sum(utilizations) / n:6.4f}")
    print(f"    relevant_context_ratio: min={min(relevances):6.4f}  max={max(relevances):6.4f}  mean={sum(relevances) / n:6.4f}")


# ---------------------------------------------------------------------------
# Privacy sentinels
# ---------------------------------------------------------------------------


def validate_no_private_data(records: list[PredictorOutcomeRecord]) -> None:
    """Verify the dataset contains no PII or private data.

    Checks that session IDs are synthetic, no filesystem paths,
    no API keys, no email addresses.
    """
    for record in records:
        d = record_to_dict(record)
        for key, value in d.items():
            if isinstance(value, str):
                # Session IDs should be synthetic.
                if key == "session_id":
                    assert value.startswith("session-"), (
                        f"Session ID does not follow synthetic pattern: {value}"
                    )
                # No filesystem paths.
                assert "/" not in value or value.startswith("session"), (
                    f"Potential path in field '{key}': {value}"
                )
                # No API key patterns.
                assert not value.startswith("sk-"), (
                    f"Potential API key in field '{key}'"
                )
                # No email patterns.
                assert "@" not in value, (
                    f"Potential email in field '{key}': {value}"
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic session replay dataset for Context Health analysis.",
        epilog=(
            "This is EXPERIMENTAL INFRASTRUCTURE for offline analysis.\n"
            "It does NOT validate the Context Health hypothesis on real sessions."
        ),
    )
    parser.add_argument(
        "--sessions",
        type=int,
        default=100,
        help="Number of synthetic sessions to generate (default: 100).",
    )
    parser.add_argument(
        "--turns",
        type=int,
        default=10,
        help="Number of turns per session (default: 10).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/session_replay.jsonl",
        help="Output JSONL file path (default: data/session_replay.jsonl).",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CONTEXT HEALTH V0.1 — SESSION REPLAY DATASET GENERATOR")
    print("=" * 80)
    print()
    print(f"  Sessions:  {args.sessions}")
    print(f"  Turns:     {args.turns}")
    print(f"  Seed:      {args.seed}")
    print(f"  Output:    {args.output}")
    print()

    # Generate dataset.
    print("[GENERATING SESSIONS]")
    records = generate_dataset(
        num_sessions=args.sessions,
        num_turns=args.turns,
        seed=args.seed,
    )
    print(f"  Generated {len(records)} predictor→outcome records.")
    print()

    # Validate no private data.
    print("[PRIVACY VALIDATION]")
    validate_no_private_data(records)
    print("  ✓ No PII or private data detected.")
    print()

    # Print statistics.
    print("[DATASET STATISTICS]")
    print_dataset_statistics(records)
    print()

    # Write to JSONL.
    output_path = Path(args.output)
    print(f"[WRITING DATASET → {output_path}]")
    write_dataset_jsonl(records, output_path)
    print(f"  ✓ Wrote {len(records)} records to {output_path}")
    print()

    # Compute and print determinism hash.
    file_hash = compute_dataset_hash(output_path)
    print(f"[DETERMINISM HASH]")
    print(f"  SHA-256: {file_hash}")
    print()

    # Data validation summary.
    print("[DATA VALIDATION]")
    expected_records = args.sessions * (args.turns - 1)
    assert len(records) == expected_records, (
        f"Expected {expected_records} records, got {len(records)}"
    )
    print(f"  ✓ Record count: {len(records)} == {args.sessions} × ({args.turns} - 1)")

    # Verify temporal alignment.
    for r in records:
        assert r.outcome_turn == r.predictor_turn + 1, (
            f"Temporal misalignment: predictor={r.predictor_turn}, outcome={r.outcome_turn}"
        )
    print("  ✓ All records have outcome_turn == predictor_turn + 1")

    # Verify health score range.
    for r in records:
        assert 0.0 <= r.health_score <= 100.0, (
            f"Health score out of range: {r.health_score}"
        )
    print("  ✓ All health scores in [0.0, 100.0]")

    # Verify predictor ranges.
    for r in records:
        assert 0.0 <= r.context_utilization <= 1.0
        assert 0.0 <= r.relevant_context_ratio <= 1.0
        assert 0.0 <= r.task_complexity <= 1.0
    print("  ✓ All predictor signals in valid ranges")

    print()
    print("ALL VALIDATIONS PASSED")
    print("=" * 80)
    print(
        "This dataset is for EXPERIMENTAL analysis only.\n"
        "It does NOT validate the Context Health hypothesis on real sessions\n"
        "and does NOT establish predictive validity."
    )
    print("=" * 80)


if __name__ == "__main__":
    main()
