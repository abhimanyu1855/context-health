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

__all__ = [
    "Contradiction",
    "ContradictionDetector",
    "ContradictionResult",
    "CorrectionDetector",
    "CorrectionEvent",
    "CorrectionResult",
    "detect_contradictions",
    "detect_corrections",
    "is_correction",
]
