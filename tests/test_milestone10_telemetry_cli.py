"""Milestone 10 tests — Telemetry and CLI experience.

ALL tests are deterministic. ZERO network/API/LLM calls are made.
Tests verify JSONL serialization, append-only persistence, schema compliance,
privacy guarantees, CLI commands, formatting, change explanation, and offline execution.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from context_health.cli import app, format_health_display
from context_health.health import compute_context_health
from context_health.telemetry import (
    JsonlTelemetryWriter,
    TelemetryRecord,
    create_telemetry_record,
    get_recommendation,
)

runner = CliRunner()


# ---------------------------------------------------------------------------
# 1. Telemetry Serialization & Storage
# ---------------------------------------------------------------------------


class TestTelemetrySerialization:
    """Verify JSONL serialization, append behavior, and directory creation."""

    def test_single_record_write_and_read(self, tmp_path: Path) -> None:
        telemetry_file = tmp_path / "subdir" / "telemetry.jsonl"
        writer = JsonlTelemetryWriter(telemetry_file)

        health_res = compute_context_health(
            context_utilization=0.40,
            relevant_context_ratio=0.85,
            task_complexity=0.30,
        )
        record = create_telemetry_record(
            session_id="session-1",
            turn_index=0,
            health_result=health_res,
        )

        writer.write(record)

        assert telemetry_file.exists()
        records = writer.read_all()
        assert len(records) == 1
        assert records[0].session_id == "session-1"
        assert records[0].health_score == health_res.health_score
        assert records[0].context_utilization == pytest.approx(0.40)
        assert records[0].relevant_context_ratio == pytest.approx(0.85)

    def test_jsonl_exact_one_object_per_line(self, tmp_path: Path) -> None:
        telemetry_file = tmp_path / "telemetry.jsonl"
        writer = JsonlTelemetryWriter(telemetry_file)

        for turn in range(3):
            res = compute_context_health(context_utilization=0.1 * turn)
            rec = create_telemetry_record(
                session_id="session-abc",
                turn_index=turn,
                health_result=res,
            )
            writer.write(rec)

        with open(telemetry_file, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        assert len(lines) == 3
        for line in lines:
            parsed = json.loads(line)
            assert isinstance(parsed, dict)
            assert "health_score" in parsed
            assert "session_id" in parsed

    def test_append_preserves_previous_records(self, tmp_path: Path) -> None:
        telemetry_file = tmp_path / "telemetry.jsonl"
        writer = JsonlTelemetryWriter(telemetry_file)

        res1 = compute_context_health(context_utilization=0.2)
        writer.write(create_telemetry_record(session_id="s1", turn_index=0, health_result=res1))

        res2 = compute_context_health(context_utilization=0.4)
        writer.write(create_telemetry_record(session_id="s1", turn_index=1, health_result=res2))

        all_records = writer.read_all()
        assert len(all_records) == 2
        assert all_records[0].turn_index == 0
        assert all_records[1].turn_index == 1

    def test_empty_or_nonexistent_file(self, tmp_path: Path) -> None:
        writer = JsonlTelemetryWriter(tmp_path / "nonexistent.jsonl")
        assert writer.read_all() == []
        assert writer.read_latest() is None


# ---------------------------------------------------------------------------
# 2. Telemetry Schema & Data Model
# ---------------------------------------------------------------------------


class TestTelemetrySchema:
    """Verify required fields, types, and serialization methods."""

    def test_schema_fields_complete(self) -> None:
        health_res = compute_context_health(
            context_utilization=0.5,
            relevant_context_ratio=0.7,
            task_complexity=0.3,
            contradiction_count=1,
            correction_count=2,
            retry_count=1,
        )
        record = create_telemetry_record(
            session_id="test-session",
            turn_index=2,
            health_result=health_res,
            model="claude-sonnet-4-20250514",
            estimated_context_tokens=100_000,
            context_window=200_000,
            tool_call_count=5,
        )

        d = record.to_dict()
        assert d["session_id"] == "test-session"
        assert d["turn_index"] == 2
        assert d["health_score"] == health_res.health_score
        assert d["model"] == "claude-sonnet-4-20250514"
        assert d["estimated_context_tokens"] == 100_000
        assert d["context_window"] == 200_000
        assert d["tool_call_count"] == 5
        assert d["recommendation"] == get_recommendation(health_res.status)
        assert "utilization" in d["penalties"]
        assert "relevance" in d["penalties"]
        assert "utilization" in d["contributions"]

        json_str = record.to_json()
        restored = TelemetryRecord.from_json(json_str)
        assert restored == record


# ---------------------------------------------------------------------------
# 3. Privacy & Safety Guarantees
# ---------------------------------------------------------------------------


class TestTelemetryPrivacy:
    """Verify no API keys, secrets, or raw transcripts are persisted."""

    def test_no_sensitive_fields(self) -> None:
        health_res = compute_context_health(context_utilization=0.3)
        record = create_telemetry_record(
            session_id="s-privacy",
            turn_index=0,
            health_result=health_res,
        )
        json_str = record.to_json()

        for forbidden in ["api_key", "sk-", "Bearer", "secret", "prompt", "raw_response", "transcript"]:
            assert forbidden not in json_str.lower(), f"Potential secret field in telemetry: {forbidden}"


# ---------------------------------------------------------------------------
# 4. Session Filtering & Retrieval
# ---------------------------------------------------------------------------


class TestSessionRetrieval:
    """Verify session isolation and retrieval by session_id."""

    def test_filter_by_session(self, tmp_path: Path) -> None:
        telemetry_file = tmp_path / "telemetry.jsonl"
        writer = JsonlTelemetryWriter(telemetry_file)

        res = compute_context_health(context_utilization=0.1)
        writer.write(create_telemetry_record(session_id="session-A", turn_index=0, health_result=res))
        writer.write(create_telemetry_record(session_id="session-B", turn_index=0, health_result=res))
        writer.write(create_telemetry_record(session_id="session-A", turn_index=1, health_result=res))

        records_a = writer.read_session("session-A")
        assert len(records_a) == 2
        assert all(r.session_id == "session-A" for r in records_a)

        latest_a = writer.read_latest(session_id="session-A")
        assert latest_a is not None
        assert latest_a.turn_index == 1


# ---------------------------------------------------------------------------
# 5. CLI Commands
# ---------------------------------------------------------------------------


class TestCLIExperience:
    """Verify CLI commands using Typer's CliRunner."""

    def test_cli_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "Measure the health of an AI coding-agent context." in result.output
        assert "measure" in result.output
        assert "history" in result.output
        assert "explain" in result.output

    def test_cli_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "context-health" in result.output

    def test_cli_measure_command(self, tmp_path: Path) -> None:
        telemetry_file = tmp_path / "telemetry.jsonl"
        result = runner.invoke(
            app,
            [
                "measure",
                "--utilization",
                "0.40",
                "--relevance",
                "0.90",
                "--complexity",
                "0.20",
                "--corrections",
                "1",
                "--telemetry-path",
                str(telemetry_file),
            ],
        )
        assert result.exit_code == 0
        assert "Context Health:" in result.output
        assert "utilization: 40.0%" in result.output
        assert "relevant:    90.0%" in result.output
        assert "corrections:    1" in result.output
        assert telemetry_file.exists()

    def test_cli_measure_no_save(self, tmp_path: Path) -> None:
        telemetry_file = tmp_path / "telemetry.jsonl"
        result = runner.invoke(
            app,
            [
                "measure",
                "--utilization",
                "0.50",
                "--no-save",
                "--telemetry-path",
                str(telemetry_file),
            ],
        )
        assert result.exit_code == 0
        assert "Context Health:" in result.output
        assert not telemetry_file.exists()

    def test_cli_measure_trend_explanation(self, tmp_path: Path) -> None:
        telemetry_file = tmp_path / "telemetry.jsonl"
        sid = "trend-session"

        # Turn 0: healthy
        runner.invoke(
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
                "--telemetry-path",
                str(telemetry_file),
            ],
        )

        # Turn 1: degraded
        res2 = runner.invoke(
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
                "0.50",
                "--corrections",
                "2",
                "--telemetry-path",
                str(telemetry_file),
            ],
        )
        assert res2.exit_code == 0
        assert "Session Trend:" in res2.output
        assert "Previous Turn (0):" in res2.output
        assert "Score Change:" in res2.output
        assert "Main Signal Changes:" in res2.output
        assert "Explanation:" in res2.output

    def test_cli_history_command(self, tmp_path: Path) -> None:
        telemetry_file = tmp_path / "telemetry.jsonl"
        runner.invoke(
            app,
            [
                "measure",
                "--utilization",
                "0.30",
                "--telemetry-path",
                str(telemetry_file),
            ],
        )

        hist_res = runner.invoke(app, ["history", "--telemetry-path", str(telemetry_file)])
        assert hist_res.exit_code == 0
        assert "Displaying 1 recent telemetry record(s):" in hist_res.output

    def test_cli_explain_command(self, tmp_path: Path) -> None:
        telemetry_file = tmp_path / "telemetry.jsonl"
        runner.invoke(
            app,
            [
                "measure",
                "--utilization",
                "0.35",
                "--relevance",
                "0.80",
                "--telemetry-path",
                str(telemetry_file),
            ],
        )

        exp_res = runner.invoke(app, ["explain", "--telemetry-path", str(telemetry_file)])
        assert exp_res.exit_code == 0
        assert "Context Health Measurement Breakdown" in exp_res.output
        assert "Normalized Signal Penalties" in exp_res.output
        assert "Weighted Signal Contributions" in exp_res.output


# ---------------------------------------------------------------------------
# 6. Offline Independence
# ---------------------------------------------------------------------------


class TestOfflineIndependence:
    """Verify zero external/LLM imports in telemetry and cli modules."""

    def test_no_forbidden_imports(self) -> None:
        import context_health.telemetry as tel_mod

        with open(tel_mod.__file__) as f:
            code = f.read()

        for forbidden in ["import anthropic", "import httpx", "import requests", "import urllib"]:
            assert forbidden not in code, f"Forbidden import found: {forbidden}"
