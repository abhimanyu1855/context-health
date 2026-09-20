"""Relevant context ratio metric for Context Health.

Hypothesis
----------
"For the same context utilization level, sessions with higher
irrelevant-context density (lower relevant-context ratio) may degrade
faster than sessions with lower irrelevant-context density."

Important Disclaimers (V0.1)
-----------------------------
Relevant Context Ratio is an **experimental metric** representing the
fraction of currently measured context tokens considered relevant to the
task.

In V0.1:
- Relevance must be **explicitly supplied or annotated**.
- The system does **NOT** automatically infer or evaluate semantic relevance.
- This metric does **NOT** claim to predict hallucinations, reliability,
  or degradation.
- No machine-learning, embeddings, or LLM evaluation loops are used.

This milestone provides the foundational measurement primitive for future
validation and experimentation.

Formula & Bounds
----------------
    relevant_context_ratio = relevant_context_tokens / total_context_tokens
    irrelevant_context_tokens = total_context_tokens - relevant_context_tokens

- Bounded in [0.0, 1.0].
- When total_context_tokens == 0, relevant_context_ratio is defined as 0.0.
- If relevant_context_tokens > total_context_tokens, relevant tokens is
  clamped to total_context_tokens, yielding ratio = 1.0 and irrelevant = 0.
- Negative token counts are sanitized to 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContextItem:
    """An individual piece of context annotated with relevance.

    Parameters
    ----------
    identifier:
        A unique label or name for this context item (e.g. file path,
        message ID, tool output name).
    token_count:
        Number of tokens consumed by this item.
    relevant:
        Explicit boolean indicating whether this item is considered relevant.
    """

    identifier: str
    token_count: int
    relevant: bool = True


@dataclass(frozen=True)
class RelevantContextState:
    """Snapshot of relevant context measurements.

    Attributes
    ----------
    total_context_tokens:
        Total tokens considered in the relevance evaluation.
    relevant_context_tokens:
        Subset of total tokens explicitly annotated/marked as relevant.
    irrelevant_context_tokens:
        Subset of total tokens not considered relevant (total - relevant).
    relevant_context_ratio:
        Fraction of total tokens that are relevant (in [0.0, 1.0]).
    """

    total_context_tokens: int
    relevant_context_tokens: int
    irrelevant_context_tokens: int
    relevant_context_ratio: float


# ---------------------------------------------------------------------------
# Calculator / Estimator
# ---------------------------------------------------------------------------


class RelevantContextEstimator:
    """Calculates relevant-context metrics from explicit inputs or items.

    Strictly deterministic and local; requires zero LLM or external calls.
    """

    @staticmethod
    def calculate(
        *,
        total_context_tokens: int | float = 0,
        relevant_context_tokens: int | float = 0,
    ) -> RelevantContextState:
        """Calculate relevant context state from explicit token counts.

        Parameters
        ----------
        total_context_tokens:
            The total tokens evaluated (sanitized to non-negative int).
        relevant_context_tokens:
            The relevant tokens evaluated (sanitized to non-negative int).

        Returns
        -------
        RelevantContextState
            Validated and bounded relevance metrics.
        """
        # Sanitize negative inputs to 0, convert to integer counts
        safe_total = max(0, int(round(total_context_tokens)))
        safe_relevant = max(0, int(round(relevant_context_tokens)))

        # Handle zero total
        if safe_total == 0:
            return RelevantContextState(
                total_context_tokens=0,
                relevant_context_tokens=0,
                irrelevant_context_tokens=0,
                relevant_context_ratio=0.0,
            )

        # Clamp relevant tokens to total tokens if caller provided an excess
        bounded_relevant = min(safe_relevant, safe_total)
        irrelevant = safe_total - bounded_relevant
        ratio = bounded_relevant / safe_total
        clamped_ratio = max(0.0, min(1.0, ratio))

        return RelevantContextState(
            total_context_tokens=safe_total,
            relevant_context_tokens=bounded_relevant,
            irrelevant_context_tokens=max(0, irrelevant),
            relevant_context_ratio=round(clamped_ratio, 4),
        )

    @classmethod
    def calculate_from_items(
        cls,
        items: Sequence[ContextItem],
    ) -> RelevantContextState:
        """Calculate relevant context state by aggregating a list of ContextItems.

        Parameters
        ----------
        items:
            Sequence of ContextItem instances, each with token_count and
            explicit boolean relevance.

        Returns
        -------
        RelevantContextState
            Aggregated and bounded relevance metrics.
        """
        if not items:
            return cls.calculate(total_context_tokens=0, relevant_context_tokens=0)

        total_tokens = 0
        relevant_tokens = 0

        for item in items:
            item_tokens = max(0, int(round(item.token_count)))
            total_tokens += item_tokens
            if item.relevant:
                relevant_tokens += item_tokens

        return cls.calculate(
            total_context_tokens=total_tokens,
            relevant_context_tokens=relevant_tokens,
        )


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def calculate_relevant_context_ratio(
    *,
    total_context_tokens: int | float = 0,
    relevant_context_tokens: int | float = 0,
) -> RelevantContextState:
    """Convenience helper to calculate relevant context state from token counts."""
    return RelevantContextEstimator.calculate(
        total_context_tokens=total_context_tokens,
        relevant_context_tokens=relevant_context_tokens,
    )


def calculate_relevance_from_items(
    items: Sequence[ContextItem],
) -> RelevantContextState:
    """Convenience helper to calculate relevant context state from ContextItems."""
    return RelevantContextEstimator.calculate_from_items(items)
