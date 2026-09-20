"""CLI entry-point for Context Health.

Provides commands to measure context health, explain signal contributions,
view measurement history, and persist structured JSONL telemetry.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import typer

from context_health import __version__
from context_health.health import compute_context_health
from context_health.telemetry import (
    DEFAULT_TELEMETRY_FILE,
    JsonlTelemetryWriter,
    TelemetryRecord,
    create_telemetry_record,
    get_recommendation,
)

app = typer.Typer(
    name="context-health",
    help="Measure the health of an AI coding-agent context.",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Formatting Helpers
# ---------------------------------------------------------------------------


def format_health_display(
    record: TelemetryRecord,
    previous: TelemetryRecord | None = None,
) -> str:
    """Format a TelemetryRecord into a clean, human-readable terminal display."""
    lines = [
        f"Context Health: {record.health_score:.1f}  [{record.health_status}]  (Experimental heuristic)",
        "",
        "Context:",
        f"  utilization: {record.context_utilization * 100:.1f}%",
        f"  relevant:    {record.relevant_context_ratio * 100:.1f}%",
        f"  complexity:  {record.task_complexity * 100:.1f}%",
        "",
        "Signals:",
        f"  contradictions: {record.contradiction_count}",
        f"  corrections:    {record.correction_event_count}",
        f"  tool retries:   {record.tool_retry_count}",
        "",
        "Recommendation:",
        f"  {record.recommendation} (informational UX guidance)",
    ]

    if previous is not None and previous.session_id == record.session_id:
        score_diff = record.health_score - previous.health_score
        sign = "+" if score_diff > 0 else ""
        lines.extend(
            [
                "",
                "Session Trend:",
                f"  Previous Turn ({previous.turn_index}): {previous.health_score:.1f} [{previous.health_status}]",
                f"  Score Change: {sign}{score_diff:.1f}",
            ]
        )

        change_notes: list[str] = []
        if record.relevant_context_ratio != previous.relevant_context_ratio:
            change_notes.append(
                f"relevant context: {previous.relevant_context_ratio * 100:.1f}% -> {record.relevant_context_ratio * 100:.1f}%"
            )
        if record.context_utilization != previous.context_utilization:
            change_notes.append(
                f"utilization: {previous.context_utilization * 100:.1f}% -> {record.context_utilization * 100:.1f}%"
            )
        if record.task_complexity != previous.task_complexity:
            change_notes.append(
                f"complexity: {previous.task_complexity * 100:.1f}% -> {record.task_complexity * 100:.1f}%"
            )
        if record.contradiction_count != previous.contradiction_count:
            change_notes.append(
                f"contradictions: {previous.contradiction_count} -> {record.contradiction_count}"
            )
        if record.correction_event_count != previous.correction_event_count:
            change_notes.append(
                f"corrections: {previous.correction_event_count} -> {record.correction_event_count}"
            )
        if record.tool_retry_count != previous.tool_retry_count:
            change_notes.append(
                f"tool retries: {previous.tool_retry_count} -> {record.tool_retry_count}"
            )

        if change_notes:
            lines.append("  Main Signal Changes:")
            for note in change_notes:
                lines.append(f"    - {note}")

        if score_diff < 0:
            lines.append(f"  Explanation: Health decreased by {abs(score_diff):.1f} points because observed signal penalties increased.")
        elif score_diff > 0:
            lines.append(f"  Explanation: Health increased by {score_diff:.1f} points because context conditions improved.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------


@app.command()
def measure(
    utilization: float = typer.Option(
        ...,
        "--utilization",
        "-u",
        help="Context window utilization in [0.0, 1.0].",
    ),
    relevance: float = typer.Option(
        1.0,
        "--relevance",
        "-r",
        help="Estimated relevant-context ratio in [0.0, 1.0].",
    ),
    complexity: float = typer.Option(
        0.0,
        "--complexity",
        "-c",
        help="Task complexity score in [0.0, 1.0].",
    ),
    contradictions: int = typer.Option(
        0,
        "--contradictions",
        help="Contradiction count.",
    ),
    corrections: int = typer.Option(
        0,
        "--corrections",
        help="Correction event count.",
    ),
    retries: int = typer.Option(
        0,
        "--retries",
        help="Tool retry count.",
    ),
    session_id: Optional[str] = typer.Option(
        None,
        "--session-id",
        "-s",
        help="Session identifier. Generated if omitted.",
    ),
    turn_index: int = typer.Option(
        0,
        "--turn-index",
        "-t",
        help="Turn index in session.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model identifier (optional).",
    ),
    telemetry_path: Optional[Path] = typer.Option(
        None,
        "--telemetry-path",
        help="Custom path to telemetry JSONL file.",
    ),
    save: bool = typer.Option(
        True,
        "--save/--no-save",
        help="Persist measurement to local JSONL telemetry.",
    ),
) -> None:
    """Compute and display Context Health, recording telemetry."""
    sid = session_id or uuid.uuid4().hex
    writer = JsonlTelemetryWriter(telemetry_path)

    # Find previous record in session if available
    previous = writer.read_latest(session_id=sid)

    health_result = compute_context_health(
        context_utilization=utilization,
        relevant_context_ratio=relevance,
        task_complexity=complexity,
        contradiction_count=contradictions,
        correction_count=corrections,
        retry_count=retries,
    )

    record = create_telemetry_record(
        session_id=sid,
        turn_index=turn_index,
        health_result=health_result,
        model=model,
    )

    if save:
        writer.write(record)

    output = format_health_display(record, previous=previous)
    typer.echo(output)


@app.command()
def history(
    session_id: Optional[str] = typer.Option(
        None,
        "--session-id",
        "-s",
        help="Filter telemetry by session ID.",
    ),
    limit: int = typer.Option(
        10,
        "--limit",
        "-n",
        help="Maximum records to display.",
    ),
    telemetry_path: Optional[Path] = typer.Option(
        None,
        "--telemetry-path",
        help="Custom path to telemetry JSONL file.",
    ),
) -> None:
    """Display recent Context Health telemetry records."""
    writer = JsonlTelemetryWriter(telemetry_path)
    records = writer.read_session(session_id) if session_id else writer.read_all()

    if not records:
        typer.echo("No telemetry records found.")
        return

    recent = records[-limit:]
    typer.echo(f"Displaying {len(recent)} recent telemetry record(s):")
    typer.echo("-" * 65)
    for r in recent:
        typer.echo(
            f"[{r.timestamp}] Session: {r.session_id[:8]}.. Turn: {r.turn_index} | Health: {r.health_score:5.1f} [{r.health_status}] (Util: {r.context_utilization*100:4.1f}%, Rel: {r.relevant_context_ratio*100:4.1f}%, Rec: {r.recommendation})"
        )


@app.command()
def explain(
    session_id: Optional[str] = typer.Option(
        None,
        "--session-id",
        "-s",
        help="Session ID to explain latest measurement for.",
    ),
    telemetry_path: Optional[Path] = typer.Option(
        None,
        "--telemetry-path",
        help="Custom path to telemetry JSONL file.",
    ),
) -> None:
    """Show detailed penalty and contribution breakdown for the latest measurement."""
    writer = JsonlTelemetryWriter(telemetry_path)
    record = writer.read_latest(session_id=session_id)

    if not record:
        typer.echo("No telemetry records found to explain.")
        return

    typer.echo(f"Context Health Measurement Breakdown (Session: {record.session_id}, Turn: {record.turn_index})")
    typer.echo(f"Score: {record.health_score:.1f}/100 [{record.health_status}] (Recommendation: {record.recommendation})")
    typer.echo("-" * 65)
    typer.echo("Normalized Signal Penalties in [0.0, 1.0]:")
    for k, v in record.penalties.items():
        typer.echo(f"  - {k:<15}: {v:.3f}")
    typer.echo("Weighted Signal Contributions:")
    for k, v in record.contributions.items():
        typer.echo(f"  - {k:<15}: {v:.3f}")


@app.command()
def version() -> None:
    """Print the current version."""
    typer.echo(f"context-health {__version__}")


@app.command()
def chat(
    model: str = typer.Option("claude-sonnet-4-20250514", help="Anthropic model to use."),
    max_tokens: int = typer.Option(1024, help="Max tokens per response."),
) -> None:
    """Start an interactive chat session with context-health tracking."""
    typer.echo(f"[context-health] chat session placeholder (model={model}, max_tokens={max_tokens})")


if __name__ == "__main__":
    app()
