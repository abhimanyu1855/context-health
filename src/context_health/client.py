"""Anthropic API client wrapper for Context Health.

Provides a provider-agnostic interface so the rest of the application never
depends on raw Anthropic SDK response types.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import anthropic
from anthropic.types import TextBlock


# ---------------------------------------------------------------------------
# Internal response representation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompletionResponse:
    """Provider-agnostic response from a completion request.

    Contains only the data that Context Health needs.  No raw SDK objects
    are exposed.
    """

    text: str
    input_tokens: int
    output_tokens: int
    model: str
    stop_reason: str | None = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class APIKeyMissingError(Exception):
    """Raised when the ANTHROPIC_API_KEY environment variable is not set."""

    def __init__(self) -> None:
        super().__init__(
            "ANTHROPIC_API_KEY environment variable is not set. "
            "Set it with: export ANTHROPIC_API_KEY='sk-...'"
        )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class AnthropicClient:
    """Thin wrapper around the Anthropic Python SDK.

    Responsibilities:
    - Read ANTHROPIC_API_KEY from the environment.
    - Send multi-turn messages with an optional system prompt.
    - Return a :class:`CompletionResponse` (no raw SDK objects leak out).
    """

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
        system_prompt: str | None = None,
        api_key: str | None = None,
    ) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise APIKeyMissingError()

        self._model = model
        self._max_tokens = max_tokens
        self._system_prompt = system_prompt

        # The key is passed directly to the SDK and never stored on self.
        self._client = anthropic.Anthropic(api_key=resolved_key)

    # -- public properties ---------------------------------------------------

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def system_prompt(self) -> str | None:
        return self._system_prompt

    # -- public API ----------------------------------------------------------

    def send(
        self,
        messages: list[dict[str, str]],
    ) -> CompletionResponse:
        """Send a list of messages and return a :class:`CompletionResponse`.

        Parameters
        ----------
        messages:
            A list of message dicts, each with ``role`` (``"user"`` or
            ``"assistant"``) and ``content`` keys.

        Returns
        -------
        CompletionResponse
            Provider-agnostic response containing text, token counts, model,
            and stop reason.

        Raises
        ------
        anthropic.APIError
            On any Anthropic-level error (auth, rate-limit, server, etc.).
        """
        kwargs: dict = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "messages": messages,
        }
        if self._system_prompt is not None:
            kwargs["system"] = self._system_prompt

        response = self._client.messages.create(**kwargs)

        # Extract text from content blocks.
        text_parts: list[str] = []
        for block in response.content:
            if isinstance(block, TextBlock):
                text_parts.append(block.text)
        text = "".join(text_parts)

        return CompletionResponse(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
            stop_reason=response.stop_reason,
        )
