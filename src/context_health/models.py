"""Pydantic models for Context Health data structures."""

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation turn."""

    role: str = Field(..., description="Message role: 'user' or 'assistant'.")
    content: str = Field(..., description="Message text content.")


class TurnMetrics(BaseModel):
    """Metrics captured for a single conversation turn."""

    turn_number: int = Field(..., ge=0)
    input_tokens: int = Field(0, ge=0)
    output_tokens: int = Field(0, ge=0)
    context_health_score: float = Field(100.0, ge=0.0, le=100.0)


class SessionSummary(BaseModel):
    """Summary of a complete chat session."""

    session_id: str
    total_turns: int = Field(0, ge=0)
    total_input_tokens: int = Field(0, ge=0)
    total_output_tokens: int = Field(0, ge=0)
    final_context_health_score: float = Field(100.0, ge=0.0, le=100.0)
