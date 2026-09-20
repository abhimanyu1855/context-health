"""Configuration for Context Health."""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Model context-window mapping
# ---------------------------------------------------------------------------

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-sonnet-4-20250514": 200_000,
    "claude-haiku-4-20250414": 200_000,
    "claude-opus-4-20250514": 200_000,
    "claude-sonnet-4-5-20250929": 200_000,
    "claude-opus-4-5-20251101": 200_000,
}
"""Known context-window sizes (in tokens) for supported models.

This is a simple lookup table — NOT model routing or automatic model
selection.  If a model is not listed here, callers should fall back to
a configured default.
"""

DEFAULT_CONTEXT_WINDOW: int = 200_000
"""Fallback context-window size when the model is not in the mapping."""


def get_context_window(model: str) -> int:
    """Return the context-window size for a model.

    Falls back to :data:`DEFAULT_CONTEXT_WINDOW` if the model is unknown.
    """
    return MODEL_CONTEXT_WINDOWS.get(model, DEFAULT_CONTEXT_WINDOW)


# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------


class ContextHealthConfig(BaseModel):
    """Runtime configuration for a Context Health session."""

    model: str = Field("claude-sonnet-4-20250514", description="Anthropic model identifier.")
    max_tokens: int = Field(1024, ge=1, description="Max tokens per response.")
    max_context_window: int = Field(200_000, ge=1, description="Model context window size in tokens.")
    telemetry_dir: str = Field(".context_health", description="Directory for JSONL telemetry output.")
