"""Milestone 11 tests — End-to-End Integration + V0.1 Validation.

ALL tests are deterministic. ZERO network/API/LLM calls are made.
Validates the end-to-end integration of all V0.1 components:
- Session context tracking (tokens, turns, files, tool calls)
- Task complexity estimation
- Relevant context ratio calculation
- Contradiction and correction event detection
- Tool retry observation
- Context Health 0–100 scoring
- Telemetry round-trip JSONL persistence
- Explain and trend analysis
- CLI commands integration
- Privacy guarantees and determinism invariants
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from context_health.cli import app
from context_health.complexity import estimate_task_complexity
from context_health.context import ContextState, ContextTracker
from context_health.contradictions import detect_contradictions
from context_health.corrections import detect_corrections
from context_health.health import (
    ContextHealthResult,
    calculate_context_health,
    compute_context_health,
    get_health_status,
)
from context_health.relevance import ContextItem, calculate_relevance_from_items
from context_health.telemetry import (
    JsonlTelemetryWriter,
    TelemetryRecord,
    create_telemetry_record,
    get_recommendation,
)
from context_health.tool_retries import ToolAttempt, detect_tool_retries

runner = CliRunner()


# ---------------------------------------------------------------------------
# 1. End-to-End Multi-Turn Session Validation Pipeline
# ---------------------------------------------------------------------------


class TestEndToEndSessionPipeline:
    """Verify that a full multi-turn session seamlessly connects all signal modules to scoring."""

    def test_multi_turn_pipeline_evolution(self) -> None:
        tracker = ContextTracker(context_window=200_000, session_id="session-e2e-001")

        # --- Turn 0: Healthy initial turn ---
        tracker.record_user_message("Please create a simple CLI tool in Python.")
        tracker.record_assistant_response(input_tokens=2_000, output_tokens=300)
        tracker.record_file_reference("src/main.py")

        state_0 = tracker.state
        assert state_0.message_count == 2
        assert state_0.estimated_context_tokens == 2_000
        assert state_0.context_utilization == pytest.approx(2_000 / 200_000)

        # Compute relevant context ratio from explicit items
        items_0 = [
            ContextItem(identifier="src/main.py", token_count=1_800, relevant=True),
            ContextItem(identifier="scratch.txt", token_count=200, relevant=False),
        ]
        rel_0 = calculate_relevance_from_items(items_0)
        assert rel_0.relevant_context_ratio == pytest.approx(0.90)

        # Estimate task complexity from state
        comp_0 = estimate_task_complexity(
            user_message="Please create a simple CLI tool in Python.",
            context_state=state_0,
        )

        health_0 = calculate_context_health(
            state_0,
            relevant_context_ratio=rel_0.relevant_context_ratio,
            task_complexity=comp_0.task_complexity,
            contradiction_count=0,
        )

        assert 0.0 <= health_0.health_score <= 100.0
        assert health_0.status in {"healthy", "watch", "warning", "critical"}
        assert health_0.health_score >= 80.0  # Turn 0 should be healthy

        # --- Turn 1: Evolution with instructions, files, and a correction ---
        tracker.record_user_message("No, use PostgreSQL instead of SQLite.")
        tracker.record_assistant_response(input_tokens=15_000, output_tokens=800)
        tracker.record_file_reference("src/database.py")

        state_1 = tracker.state
        assert state_1.correction_events == 1
        assert state_1.estimated_context_tokens == 15_000

        # Contradiction detection on user instructions
        statements = [
            "Use SQLite for storage.",
            "Do not use SQLite for storage.",
        ]
        contra_res = detect_contradictions(statements)
        assert contra_res.contradiction_count == 1

        items_1 = [
            ContextItem(identifier="src/main.py", token_count=5_000, relevant=True),
            ContextItem(identifier="src/database.py", token_count=5_000, relevant=True),
            ContextItem(identifier="old_sqlite.py", token_count=5_000, relevant=False),
        ]
        rel_1 = calculate_relevance_from_items(items_1)
        assert rel_1.relevant_context_ratio == pytest.approx(10_000 / 15_000, abs=1e-4)

        comp_1 = estimate_task_complexity(
            user_message="No, use PostgreSQL instead of SQLite.",
            context_state=state_1,
        )

        health_1 = calculate_context_health(
            state_1,
            relevant_context_ratio=rel_1.relevant_context_ratio,
            task_complexity=comp_1.task_complexity,
            contradiction_count=contra_res.contradiction_count,
        )

        # Health in Turn 1 should reflect the added penalties (utilization, correction, contradiction)
        assert health_1.health_score < health_0.health_score
        assert health_1.inputs.correction_count == 1
        assert health_1.inputs.contradiction_count == 1

        # --- Turn 2: Tool execution failure and retry ---
        # Attempt 1 fails
        tracker.record_tool_attempt(
            "run_command",
            operation_id="pytest",
            success=False,
            error_message="1 failed",
        )
        assert tracker.state.tool_calls == 1
        assert tracker.state.tool_retries == 0

        # Attempt 2 retry succeeds
        tracker.record_tool_attempt(
            "run_command",
            operation_id="pytest",
            success=True,
        )
        assert tracker.state.tool_calls == 2
        assert tracker.state.tool_retries == 1

        tracker.record_user_message("Run the test suite again.")
        tracker.record_assistant_response(input_tokens=40_000, output_tokens=1_000)

        state_2 = tracker.state
        comp_2 = estimate_task_complexity(
            user_message="Run the test suite again.",
            context_state=state_2,
        )

        health_2 = calculate_context_health(
            state_2,
            relevant_context_ratio=0.60,
            task_complexity=comp_2.task_complexity,
            contradiction_count=contra_res.contradiction_count,
        )

        assert health_2.inputs.retry_count == 1
        assert health_2.inputs.correction_count == 1
        assert health_2.inputs.contradiction_count == 1
        assert health_2.health_score <= health_1.health_score
        assert 0.0 <= health_2.health_score <= 100.0


# ---------------------------------------------------------------------------
# 2. Telemetry Round-Trip & Session Persistence
# ---------------------------------------------------------------------------


class TestTelemetryRoundTrip:
    """Verify that measurements survive the JSONL round-trip intact."""

    def test_multi_turn_telemetry_persistence(self, tmp_path: Path) -> None:
        telemetry_file = tmp_path / "telemetry.jsonl"
        writer = JsonlTelemetryWriter(telemetry_file)

        session_id = "test-e2e-session-uuid"
        scores = []

        for turn in range(3):
            util = 0.15 * (turn + 1)
            rel = 1.0 - (0.15 * turn)
            comp = 0.10 * (turn + 1)
            corr = turn
            retry = 1 if turn == 2 else 0

            health_res = compute_context_health(
                context_utilization=util,
                relevant_context_ratio=rel,
                task_complexity=comp,
                correction_count=corr,
                retry_count=retry,
            )
            scores.append(health_res.health_score)

            rec = create_telemetry_record(
                session_id=session_id,
                turn_index=turn,
                health_result=health_res,
                model="claude-sonnet-4-20250514",
                estimated_context_tokens=int(util * 200_000),
                context_window=200_000,
                tool_call_count=turn * 2,
            )
            writer.write(rec)

        assert telemetry_file.exists()

        records = writer.read_session(session_id)
        assert len(records) == 3

        for turn, rec in enumerate(records):
            assert rec.session_id == session_id
            assert rec.turn_index == turn
            assert rec.health_score == scores[turn]
            assert rec.model == "claude-sonnet-4-20250514"
            assert rec.context_window == 200_000
            assert "utilization" in rec.penalties
            assert "relevance" in rec.penalties
            assert "utilization" in rec.contributions
            assert rec.recommendation == get_recommendation(rec.health_status)

        # Verify JSONL lines are distinct and valid JSON
        with open(telemetry_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)
            assert parsed["session_id"] == session_id


# ---------------------------------------------------------------------------
# 3. Explain / Breakdown Consistency
# ---------------------------------------------------------------------------


class TestExplainBreakdownConsistency:
    """Verify that explain reads persisted records without altering scores."""

    def test_explain_matches_persisted_result(self, tmp_path: Path) -> None:
        telemetry_file = tmp_path / "telemetry.jsonl"
        writer = JsonlTelemetryWriter(telemetry_file)

        health_res = compute_context_health(
            context_utilization=0.45,
            relevant_context_ratio=0.85,
            task_complexity=0.30,
            correction_count=1,
        )
        record = create_telemetry_record(
            session_id="session-explain",
            turn_index=0,
            health_result=health_res,
        )
        writer.write(record)

        # Invoke CLI explain on the file
        exp_res = runner.invoke(
            app,
            ["explain", "--session-id", "session-explain", "--telemetry-path", str(telemetry_file)],
        )
        assert exp_res.exit_code == 0
        assert f"{health_res.health_score:.1f}" in exp_res.output
        assert health_res.status in exp_res.output

        # Verify file was not mutated
        records_after = writer.read_all()
        assert len(records_after) == 1
        assert records_after[0].health_score == health_res.health_score


# ---------------------------------------------------------------------------
# 4. CLI Multi-Turn Integration
# ---------------------------------------------------------------------------


class TestCLIIntegration:
    """Verify end-to-end CLI commands."""

    def test_cli_multi_turn_measure_and_history(self, tmp_path: Path) -> None:
        telemetry_file = tmp_path / "telemetry.jsonl"
        sid = "cli-session-123"

        # Turn 0
        res0 = runner.invoke(
            app,
            [
                "measure",
                "--session-id",
                sid,
                "--turn-index",
                "0",
                "--utilization",
                "0.20",
                "--relevance",
                "1.0",
                "--complexity",
                "0.10",
                "--telemetry-path",
                str(telemetry_file),
            ],
        )
        assert res0.exit_code == 0
        assert "Context Health:" in res0.output
        assert "healthy" in res0.output

        # Turn 1
        res1 = runner.invoke(
            app,
            [
                "measure",
                "--session-id",
                sid,
                "--turn-index",
                "1",
                "--utilization",
                "0.60",
                "--relevance",
                "0.60",
                "--complexity",
                "0.40",
                "--corrections",
                "2",
                "--telemetry-path",
                str(telemetry_file),
            ],
        )
        assert res1.exit_code == 0
        assert "Session Trend:" in res1.output
        assert "Score Change:" in res1.output

        # History command
        hist_res = runner.invoke(
            app,
            ["history", "--session-id", sid, "--telemetry-path", str(telemetry_file)],
        )
        assert hist_res.exit_code == 0
        assert "Displaying 2 recent telemetry record(s):" in hist_res.output

    def test_cli_version_deterministic(self) -> None:
        res = runner.invoke(app, ["version"])
        assert res.exit_code == 0
        assert "context-health 0.1.0" in res.output


# ---------------------------------------------------------------------------
# 5. Determinism & Reproducibility
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Verify that identical inputs produce strictly identical results."""

    def test_identical_inputs_identical_health(self) -> None:
        state = ContextState(
            context_utilization=0.42,
            correction_events=2,
            tool_retries=1,
        )

        res_a = calculate_context_health(
            state,
            relevant_context_ratio=0.75,
            task_complexity=0.35,
            contradiction_count=1,
        )
        res_b = calculate_context_health(
            state,
            relevant_context_ratio=0.75,
            task_complexity=0.35,
            contradiction_count=1,
        )

        assert res_a.health_score == res_b.health_score
        assert res_a.status == res_b.status
        assert res_a.penalties == res_b.penalties
        assert res_a.contributions == res_b.contributions

    def test_telemetry_serialization_deterministic(self) -> None:
        health_res = compute_context_health(
            context_utilization=0.30,
            relevant_context_ratio=0.80,
            task_complexity=0.20,
        )
        rec_a = create_telemetry_record(
            session_id="det-session",
            turn_index=1,
            health_result=health_res,
            timestamp="2026-09-20T12:00:00+00:00",
        )
        rec_b = create_telemetry_record(
            session_id="det-session",
            turn_index=1,
            health_result=health_res,
            timestamp="2026-09-20T12:00:00+00:00",
        )
        assert rec_a.to_json() == rec_b.to_json()


# ---------------------------------------------------------------------------
# 6. Privacy & Safety Sentinels
# ---------------------------------------------------------------------------


class TestPrivacySentinels:
    """Verify that sensitive sentinels and raw transcripts never leak into telemetry."""

    def test_sentinels_never_in_telemetry(self, tmp_path: Path) -> None:
        telemetry_file = tmp_path / "telemetry.jsonl"
        writer = JsonlTelemetryWriter(telemetry_file)

        # Create session with sensitive in-memory representations
        tracker = ContextTracker()
        tracker.record_user_message("SUPER_SECRET_TEST_API_KEY with PRIVATE_PROMPT_SENTINEL")
        tracker.record_assistant_response(input_tokens=500, output_tokens=100)

        health_res = calculate_context_health(
            tracker.state,
            relevant_context_ratio=0.90,
            task_complexity=0.20,
        )

        record = create_telemetry_record(
            session_id="privacy-session",
            turn_index=0,
            health_result=health_res,
        )
        writer.write(record)

        raw_jsonl = telemetry_file.read_text(encoding="utf-8")

        sentinels = [
            "SUPER_SECRET_TEST_API_KEY",
            "PRIVATE_PROMPT_SENTINEL",
            "PRIVATE_RESPONSE_SENTINEL",
            "AUTHORIZATION_HEADER_SENTINEL",
            "Bearer ",
            "sk-ant-",
        ]

        for sentinel in sentinels:
            assert sentinel not in raw_jsonl, f"Sensitive sentinel leaked into telemetry: {sentinel}"


# ---------------------------------------------------------------------------
# 7. Hypothesis Signals Availability
# ---------------------------------------------------------------------------


class TestHypothesisSignalsAvailability:
    """Verify that all required telemetry fields for testing the hypothesis are present."""

    def test_hypothesis_required_fields_present(self) -> None:
        health_res = compute_context_health(
            context_utilization=0.60,
            relevant_context_ratio=0.70,
            task_complexity=0.40,
            contradiction_count=1,
            correction_count=2,
            retry_count=1,
        )
        record = create_telemetry_record(
            session_id="hypo-session",
            turn_index=3,
            health_result=health_res,
            model="claude-sonnet-4-20250514",
        )

        d = record.to_dict()

        required_signals = [
            "context_utilization",
            "relevant_context_ratio",
            "health_score",
            "task_complexity",
            "contradiction_count",
            "correction_event_count",
            "tool_retry_count",
            "model",
            "session_id",
            "turn_index",
        ]

        for signal in required_signals:
            assert signal in d, f"Missing hypothesis signal in telemetry: {signal}"
            assert d[signal] is not None or signal == "model"


# ---------------------------------------------------------------------------
# 8. Single Source of Truth & Offline Independence
# ---------------------------------------------------------------------------


class TestSingleSourceOfTruthAndOffline:
    """Verify single source of truth for scoring and zero network imports."""

    def test_scoring_centralized_in_health_py(self) -> None:
        import context_health.cli as cli_mod
        import context_health.telemetry as tel_mod

        cli_src = open(cli_mod.__file__).read()
        tel_src = open(tel_mod.__file__).read()

        # Confirm CLI and Telemetry do not define their own scoring formulas
        assert "0.25 *" not in cli_src
        assert "0.30 *" not in cli_src
        assert "0.25 *" not in tel_src
        assert "0.30 *" not in tel_src

    def test_zero_network_imports(self) -> None:
        import context_health.complexity as c_mod
        import context_health.context as ctx_mod
        import context_health.contradictions as cd_mod
        import context_health.corrections as cr_mod
        import context_health.health as h_mod
        import context_health.relevance as r_mod
        import context_health.telemetry as t_mod
        import context_health.tool_retries as tr_mod

        modules = [ctx_mod, c_mod, r_mod, cd_mod, cr_mod, tr_mod, h_mod, t_mod]
        for mod in modules:
            src = open(mod.__file__).read()
            for forbidden in ["import anthropic", "import httpx", "import requests", "import urllib"]:
                assert forbidden not in src, f"Forbidden import {forbidden} in {mod.__name__}"
