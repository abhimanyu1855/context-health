"""Context Health scoring engine.

Computes a deterministic, explainable 0–100 Context Health score from observable
session signals.

Important Disclaimers (V0.1)
-----------------------------
Context Health is an **experimental heuristic observability metric**.

It is **NOT**:
- a hallucination probability or detector
- the probability that the agent is correct
- the probability that the agent will fail
- a machine-learning prediction
- a validated reliability score

It has **NOT** been empirically validated against task success or benchmark
evaluations. The weights, saturation constants, and status thresholds represent
an initial prior based on the hypothesis:

> "For the same context utilization level, sessions with high irrelevant-context
> density (low relevant-context ratio) may degrade faster than sessions with low
> irrelevant-context density."

The score's purpose is to provide a unified, transparent signal for testing this
hypothesis and observing agent context conditions over time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from context_health.context import ContextState


# ---------------------------------------------------------------------------
# Constants & Defaults
# ---------------------------------------------------------------------------

DEFAULT_HEALTH_WEIGHTS: dict[str, float] = {
    "utilization": 0.25,
    "relevance": 0.30,
    "complexity": 0.15,
    "contradictions": 0.10,
    "corrections": 0.10,
    "retries": 0.10,
}

DEFAULT_SATURATION_CONSTANTS: dict[str, float] = {
    "contradictions": 3.0,
    "corrections": 3.0,
    "retries": 3.0,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HealthSignalInputs:
    """Normalized input signals used for Context Health calculation."""

    context_utilization: float
    relevant_context_ratio: float
    task_complexity: float
    contradiction_count: int = 0
    correction_count: int = 0
    retry_count: int = 0


@dataclass(frozen=True)
class HealthPenalties:
    """Normalized penalties in [0.0, 1.0] for each signal category."""

    utilization_penalty: float
    relevance_penalty: float
    complexity_penalty: float
    contradiction_penalty: float
    correction_penalty: float
    retry_penalty: float
    total_weighted_penalty: float


@dataclass(frozen=True)
class HealthWeightedContributions:
    """Effective contribution of each signal penalty to the total penalty."""

    utilization: float
    relevance: float
    complexity: float
    contradictions: float
    corrections: float
    retries: float


@dataclass(frozen=True)
class ContextHealthResult:
    """Complete, explainable result of a Context Health calculation.

    Attributes
    ----------
    health_score:
        Final score clamped to [0.0, 100.0].
    status:
        UX status category: 'healthy' (80-100), 'watch' (60-79.99),
        'warning' (40-59.99), or 'critical' (0-39.99).
    inputs:
        Sanitized input signals.
    penalties:
        Normalized individual penalties in [0.0, 1.0].
    contributions:
        Weighted penalty points contributed by each signal.
    weights:
        The weights applied to each category (sum to 1.0).
    saturation_constants:
        The saturation constants applied to event-based signals.
    """

    health_score: float
    status: str
    inputs: HealthSignalInputs
    penalties: HealthPenalties
    contributions: HealthWeightedContributions
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_HEALTH_WEIGHTS))
    saturation_constants: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SATURATION_CONSTANTS)
    )

    def explain(self) -> str:
        """Return a human-readable multi-line explanation of the score."""
        lines = [
            f"Context Health: {self.health_score:.1f}/100 ({self.status})",
            f"  Total Weighted Penalty: {self.penalties.total_weighted_penalty:.3f}",
            "  Signal Penalties & Contributions:",
            f"    - Utilization: penalty={self.penalties.utilization_penalty:.3f}, weight={self.weights['utilization']:.2f}, contribution={self.contributions.utilization:.3f}",
            f"    - Relevance:   penalty={self.penalties.relevance_penalty:.3f}, weight={self.weights['relevance']:.2f}, contribution={self.contributions.relevance:.3f}",
            f"    - Complexity:  penalty={self.penalties.complexity_penalty:.3f}, weight={self.weights['complexity']:.2f}, contribution={self.contributions.complexity:.3f}",
            f"    - Contradictions (count={self.inputs.contradiction_count}): penalty={self.penalties.contradiction_penalty:.3f}, weight={self.weights['contradictions']:.2f}, contribution={self.contributions.contradictions:.3f}",
            f"    - Corrections (count={self.inputs.correction_count}):    penalty={self.penalties.correction_penalty:.3f}, weight={self.weights['corrections']:.2f}, contribution={self.contributions.corrections:.3f}",
            f"    - Retries (count={self.inputs.retry_count}):        penalty={self.penalties.retry_penalty:.3f}, weight={self.weights['retries']:.2f}, contribution={self.contributions.retries:.3f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------


def get_health_status(score: float) -> str:
    """Determine UX status band for a given health score.

    Thresholds (V0.1 UX bands, not empirically validated):
    - 80.0 <= score <= 100.0: 'healthy'
    - 60.0 <= score < 80.0:  'watch'
    - 40.0 <= score < 60.0:  'warning'
    -  0.0 <= score < 40.0:  'critical'
    """
    if score >= 80.0:
        return "healthy"
    elif score >= 60.0:
        return "watch"
    elif score >= 40.0:
        return "warning"
    else:
        return "critical"


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    """Clamp a floating point value to [low, high], handling NaN/inf."""
    if math.isnan(value):
        return low
    return max(low, min(high, value))


def compute_context_health(
    *,
    context_utilization: float,
    relevant_context_ratio: float = 1.0,
    task_complexity: float = 0.0,
    contradiction_count: int = 0,
    correction_count: int = 0,
    retry_count: int = 0,
    weights: dict[str, float] | None = None,
    saturation_constants: dict[str, float] | None = None,
) -> ContextHealthResult:
    """Compute the 0–100 Context Health score from individual signals.

    Parameters
    ----------
    context_utilization:
        Current context window utilization in [0.0, 1.0].
    relevant_context_ratio:
        Estimated ratio of relevant context in [0.0, 1.0]. Defaults to 1.0.
    task_complexity:
        Task complexity score in [0.0, 1.0]. Defaults to 0.0.
    contradiction_count:
        Number of detected instruction contradictions. Defaults to 0.
    correction_count:
        Number of detected user correction events. Defaults to 0.
    retry_count:
        Number of detected tool retries. Defaults to 0.
    weights:
        Optional custom signal weights dictionary summing to 1.0.
    saturation_constants:
        Optional custom saturation constants for event signals.

    Returns
    -------
    ContextHealthResult
        The complete score result with all intermediate contributions and penalties.
    """
    # 1. Resolve configuration
    effective_weights = dict(DEFAULT_HEALTH_WEIGHTS)
    if weights:
        effective_weights.update(weights)

    effective_k = dict(DEFAULT_SATURATION_CONSTANTS)
    if saturation_constants:
        effective_k.update(saturation_constants)

    # 2. Sanitize and clamp inputs
    sanitized_utilization = _clamp(context_utilization, 0.0, 1.0)
    sanitized_relevance = _clamp(relevant_context_ratio, 0.0, 1.0)
    sanitized_complexity = _clamp(task_complexity, 0.0, 1.0)

    sanitized_contradictions = max(0, contradiction_count)
    sanitized_corrections = max(0, correction_count)
    sanitized_retries = max(0, retry_count)

    inputs = HealthSignalInputs(
        context_utilization=sanitized_utilization,
        relevant_context_ratio=sanitized_relevance,
        task_complexity=sanitized_complexity,
        contradiction_count=sanitized_contradictions,
        correction_count=sanitized_corrections,
        retry_count=sanitized_retries,
    )

    # 3. Compute normalized penalties in [0.0, 1.0]
    utilization_penalty = sanitized_utilization
    relevance_penalty = 1.0 - sanitized_relevance
    complexity_penalty = sanitized_complexity

    k_contra = max(0.1, effective_k.get("contradictions", 3.0))
    k_corr = max(0.1, effective_k.get("corrections", 3.0))
    k_retry = max(0.1, effective_k.get("retries", 3.0))

    contradiction_penalty = sanitized_contradictions / (sanitized_contradictions + k_contra)
    correction_penalty = sanitized_corrections / (sanitized_corrections + k_corr)
    retry_penalty = sanitized_retries / (sanitized_retries + k_retry)

    # 4. Compute weighted contributions
    w_util = effective_weights.get("utilization", 0.25)
    w_rel = effective_weights.get("relevance", 0.30)
    w_comp = effective_weights.get("complexity", 0.15)
    w_contra = effective_weights.get("contradictions", 0.10)
    w_corr = effective_weights.get("corrections", 0.10)
    w_retry = effective_weights.get("retries", 0.10)

    contrib_util = w_util * utilization_penalty
    contrib_rel = w_rel * relevance_penalty
    contrib_comp = w_comp * complexity_penalty
    contrib_contra = w_contra * contradiction_penalty
    contrib_corr = w_corr * correction_penalty
    contrib_retry = w_retry * retry_penalty

    total_penalty = (
        contrib_util
        + contrib_rel
        + contrib_comp
        + contrib_contra
        + contrib_corr
        + contrib_retry
    )
    total_penalty = _clamp(total_penalty, 0.0, 1.0)

    penalties = HealthPenalties(
        utilization_penalty=utilization_penalty,
        relevance_penalty=relevance_penalty,
        complexity_penalty=complexity_penalty,
        contradiction_penalty=contradiction_penalty,
        correction_penalty=correction_penalty,
        retry_penalty=retry_penalty,
        total_weighted_penalty=total_penalty,
    )

    contributions = HealthWeightedContributions(
        utilization=contrib_util,
        relevance=contrib_rel,
        complexity=contrib_comp,
        contradictions=contrib_contra,
        corrections=contrib_corr,
        retries=contrib_retry,
    )

    # 5. Compute final score and status
    raw_score = 100.0 * (1.0 - total_penalty)
    final_score = round(max(0.0, min(100.0, raw_score)), 2)
    status = get_health_status(final_score)

    return ContextHealthResult(
        health_score=final_score,
        status=status,
        inputs=inputs,
        penalties=penalties,
        contributions=contributions,
        weights=effective_weights,
        saturation_constants=effective_k,
    )


def calculate_context_health(
    state: ContextState,
    *,
    relevant_context_ratio: float = 1.0,
    task_complexity: float = 0.0,
    contradiction_count: int = 0,
    weights: dict[str, float] | None = None,
    saturation_constants: dict[str, float] | None = None,
) -> ContextHealthResult:
    """Calculate Context Health directly from a ContextState snapshot.

    Extracts context_utilization, correction_events, and tool_retries from
    the state, combining them with the provided task signals.
    """
    return compute_context_health(
        context_utilization=state.context_utilization,
        relevant_context_ratio=relevant_context_ratio,
        task_complexity=task_complexity,
        contradiction_count=contradiction_count,
        correction_count=state.correction_events,
        retry_count=state.tool_retries,
        weights=weights,
        saturation_constants=saturation_constants,
    )
