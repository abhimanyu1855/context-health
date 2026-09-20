"""Structured JSONL telemetry for Context Health.

Persists local, append-only measurement records to enable session reconstruction,
trend inspection, and offline analysis of the Context Health hypothesis.

Privacy & Security
------------------
Telemetry records contain ONLY structured numeric and categorical measurements.
They do NOT persist:
- API keys or secrets
- Authorization headers
- Full prompt texts or raw conversation transcripts
- Raw model generation outputs
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from context_health.health import ContextHealthResult, get_health_status


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TELEMETRY_DIR = Path.home() / ".context-health"
DEFAULT_TELEMETRY_FILE = DEFAULT_TELEMETRY_DIR / "telemetry.jsonl"


# ---------------------------------------------------------------------------
# Recommendation Mapping (Informational UX Guidance)
# ---------------------------------------------------------------------------

STATUS_RECOMMENDATIONS: dict[str, str] = {
    "healthy": "CONTINUE",
    "watch": "MONITOR",
    "warning": "EVALUATE",
    "critical": "CONSIDER_FRESH_CONTEXT",
}


def get_recommendation(status: str) -> str:
    """Return an informational UX recommendation based on status."""
    return STATUS_RECOMMENDATIONS.get(status.lower(), "MONITOR")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TelemetryRecord:
    """Represents a single point-in-time Context Health measurement record.

    Attributes
    ----------
    timestamp:
        ISO 8601 UTC timestamp string.
    session_id:
        Opaque session identifier.
    turn_index:
        Zero-indexed turn number in the session.
    health_score:
        Calculated Context Health score in [0.0, 100.0].
    health_status:
        UX band: 'healthy', 'watch', 'warning', or 'critical'.
    context_utilization:
        Context window utilization in [0.0, 1.0].
    relevant_context_ratio:
        Estimated relevant context ratio in [0.0, 1.0].
    task_complexity:
        Task complexity score in [0.0, 1.0].
    contradiction_count:
        Number of detected instruction contradictions.
    correction_event_count:
        Number of detected user corrections.
    tool_call_count:
        Cumulative tool calls.
    tool_retry_count:
        Cumulative tool retries.
    model:
        Optional model identifier.
    estimated_context_tokens:
        Latest provider-reported token occupancy proxy.
    context_window:
        Configured model context window size.
    recommendation:
        Informational UX guidance based on status.
    penalties:
        Normalized individual penalties for each signal.
    contributions:
        Weighted points contributed by each signal to total penalty.
    """

    timestamp: str
    session_id: str
    turn_index: int
    health_score: float
    health_status: str
    context_utilization: float
    relevant_context_ratio: float
    task_complexity: float
    contradiction_count: int
    correction_event_count: int
    tool_call_count: int
    tool_retry_count: int
    model: str | None = None
    estimated_context_tokens: int | None = None
    context_window: int | None = None
    recommendation: str = "CONTINUE"
    penalties: dict[str, float] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert record to a JSON-serializable dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize record to a single-line JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TelemetryRecord:
        """Construct a TelemetryRecord from a dictionary."""
        return cls(
            timestamp=data["timestamp"],
            session_id=data["session_id"],
            turn_index=int(data["turn_index"]),
            health_score=float(data["health_score"]),
            health_status=str(data["health_status"]),
            context_utilization=float(data["context_utilization"]),
            relevant_context_ratio=float(data["relevant_context_ratio"]),
            task_complexity=float(data["task_complexity"]),
            contradiction_count=int(data.get("contradiction_count", 0)),
            correction_event_count=int(data.get("correction_event_count", 0)),
            tool_call_count=int(data.get("tool_call_count", 0)),
            tool_retry_count=int(data.get("tool_retry_count", 0)),
            model=data.get("model"),
            estimated_context_tokens=data.get("estimated_context_tokens"),
            context_window=data.get("context_window"),
            recommendation=data.get("recommendation", get_recommendation(data.get("health_status", "healthy"))),
            penalties=dict(data.get("penalties", {})),
            contributions=dict(data.get("contributions", {})),
        )

    @classmethod
    def from_json(cls, line: str) -> TelemetryRecord:
        """Deserialize a TelemetryRecord from a JSON string."""
        return cls.from_dict(json.loads(line.strip()))


# ---------------------------------------------------------------------------
# Record factory
# ---------------------------------------------------------------------------


def create_telemetry_record(
    *,
    session_id: str,
    turn_index: int,
    health_result: ContextHealthResult,
    model: str | None = None,
    estimated_context_tokens: int | None = None,
    context_window: int | None = None,
    tool_call_count: int = 0,
    timestamp: str | None = None,
) -> TelemetryRecord:
    """Construct a TelemetryRecord from a ContextHealthResult."""
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    status = health_result.status
    rec = get_recommendation(status)

    penalties = {
        "utilization": health_result.penalties.utilization_penalty,
        "relevance": health_result.penalties.relevance_penalty,
        "complexity": health_result.penalties.complexity_penalty,
        "contradictions": health_result.penalties.contradiction_penalty,
        "corrections": health_result.penalties.correction_penalty,
        "retries": health_result.penalties.retry_penalty,
        "total_weighted": health_result.penalties.total_weighted_penalty,
    }

    contributions = {
        "utilization": health_result.contributions.utilization,
        "relevance": health_result.contributions.relevance,
        "complexity": health_result.contributions.complexity,
        "contradictions": health_result.contributions.contradictions,
        "corrections": health_result.contributions.corrections,
        "retries": health_result.contributions.retries,
    }

    return TelemetryRecord(
        timestamp=ts,
        session_id=session_id,
        turn_index=turn_index,
        health_score=health_result.health_score,
        health_status=status,
        context_utilization=health_result.inputs.context_utilization,
        relevant_context_ratio=health_result.inputs.relevant_context_ratio,
        task_complexity=health_result.inputs.task_complexity,
        contradiction_count=health_result.inputs.contradiction_count,
        correction_event_count=health_result.inputs.correction_count,
        tool_call_count=tool_call_count,
        tool_retry_count=health_result.inputs.retry_count,
        model=model,
        estimated_context_tokens=estimated_context_tokens,
        context_window=context_window,
        recommendation=rec,
        penalties=penalties,
        contributions=contributions,
    )


# ---------------------------------------------------------------------------
# Telemetry Writer / Reader
# ---------------------------------------------------------------------------


class JsonlTelemetryWriter:
    """Appends and inspects JSONL telemetry records locally."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else DEFAULT_TELEMETRY_FILE

    def write(self, record: TelemetryRecord) -> None:
        """Append a single TelemetryRecord to the JSONL file."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = record.to_json() + "\n"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line)

    def read_all(self) -> list[TelemetryRecord]:
        """Read all telemetry records from the file in chronological order."""
        if not self.path.exists():
            return []

        records: list[TelemetryRecord] = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    records.append(TelemetryRecord.from_json(line_str))
        return records

    def read_session(self, session_id: str) -> list[TelemetryRecord]:
        """Read all telemetry records for a specific session ID."""
        return [r for r in self.read_all() if r.session_id == session_id]

    def read_latest(self, session_id: str | None = None) -> TelemetryRecord | None:
        """Return the most recent telemetry record, optionally filtered by session."""
        records = self.read_session(session_id) if session_id else self.read_all()
        return records[-1] if records else None
