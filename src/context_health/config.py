"""Configuration for Context Health."""

from pydantic import BaseModel, Field


class ContextHealthConfig(BaseModel):
    """Runtime configuration for a Context Health session."""

    model: str = Field("claude-sonnet-4-20250514", description="Anthropic model identifier.")
    max_tokens: int = Field(1024, ge=1, description="Max tokens per response.")
    max_context_window: int = Field(200_000, ge=1, description="Model context window size in tokens.")
    telemetry_dir: str = Field(".context_health", description="Directory for JSONL telemetry output.")
