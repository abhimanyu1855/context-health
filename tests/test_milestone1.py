"""Milestone 1 tests — package import, CLI startup, and config loading."""

import pytest
from typer.testing import CliRunner

from context_health import __version__
from context_health.cli import app
from context_health.config import ContextHealthConfig
from context_health.models import Message, TurnMetrics, SessionSummary


# ---------------------------------------------------------------------------
# 1. Package imports correctly
# ---------------------------------------------------------------------------


class TestPackageImport:
    """Verify the package and all sub-modules are importable."""

    def test_version_is_string(self) -> None:
        assert isinstance(__version__, str)

    def test_version_value(self) -> None:
        assert __version__ == "0.1.0"

    @pytest.mark.parametrize(
        "module_name",
        [
            "context_health",
            "context_health.cli",
            "context_health.client",
            "context_health.context",
            "context_health.complexity",
            "context_health.health",
            "context_health.telemetry",
            "context_health.events",
            "context_health.models",
            "context_health.config",
        ],
    )
    def test_submodule_imports(self, module_name: str) -> None:
        """Every declared module must be importable."""
        __import__(module_name)


# ---------------------------------------------------------------------------
# 2. CLI starts successfully
# ---------------------------------------------------------------------------


runner = CliRunner()


class TestCLI:
    """Verify the Typer CLI launches and responds."""

    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "context-health" in result.output.lower() or "context" in result.output.lower()

    def test_version_command(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_chat_command_runs(self) -> None:
        result = runner.invoke(app, ["chat"])
        assert result.exit_code == 0
        assert "placeholder" in result.output.lower()


# ---------------------------------------------------------------------------
# 3. Basic configuration loading works
# ---------------------------------------------------------------------------


class TestConfig:
    """Verify ContextHealthConfig loads with defaults and overrides."""

    def test_default_config(self) -> None:
        cfg = ContextHealthConfig()
        assert cfg.model == "claude-sonnet-4-20250514"
        assert cfg.max_tokens == 1024
        assert cfg.max_context_window == 200_000
        assert cfg.telemetry_dir == ".context_health"

    def test_custom_config(self) -> None:
        cfg = ContextHealthConfig(model="claude-haiku-4-20250414", max_tokens=512)
        assert cfg.model == "claude-haiku-4-20250414"
        assert cfg.max_tokens == 512

    def test_invalid_max_tokens_rejected(self) -> None:
        with pytest.raises(Exception):
            ContextHealthConfig(max_tokens=0)


# ---------------------------------------------------------------------------
# 4. Models — basic validation
# ---------------------------------------------------------------------------


class TestModels:
    """Verify Pydantic models instantiate and validate."""

    def test_message(self) -> None:
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_turn_metrics_defaults(self) -> None:
        tm = TurnMetrics(turn_number=0)
        assert tm.context_health_score == 100.0

    def test_session_summary(self) -> None:
        ss = SessionSummary(session_id="test-001")
        assert ss.total_turns == 0
        assert ss.final_context_health_score == 100.0

    def test_health_score_bounds(self) -> None:
        with pytest.raises(Exception):
            TurnMetrics(turn_number=0, context_health_score=101.0)
        with pytest.raises(Exception):
            TurnMetrics(turn_number=0, context_health_score=-1.0)
