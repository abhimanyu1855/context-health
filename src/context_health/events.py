"""Event detection — contradictions, corrections, tool retries."""

from __future__ import annotations

from context_health.contradictions import (
    Contradiction,
    ContradictionDetector,
    ContradictionResult,
    detect_contradictions,
)
from context_health.corrections import (
    CorrectionDetector,
    CorrectionEvent,
    CorrectionResult,
    detect_corrections,
    is_correction,
)
from context_health.tool_retries import (
    ToolAttempt,
    ToolRetryDetector,
    ToolRetryEvent,
    ToolRetryResult,
    detect_tool_retries,
)

__all__ = [
    "Contradiction",
    "ContradictionDetector",
    "ContradictionResult",
    "CorrectionDetector",
    "CorrectionEvent",
    "CorrectionResult",
    "ToolAttempt",
    "ToolRetryDetector",
    "ToolRetryEvent",
    "ToolRetryResult",
    "detect_contradictions",
    "detect_corrections",
    "detect_tool_retries",
    "is_correction",
]
