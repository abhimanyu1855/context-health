"""Milestone 2 tests — Anthropic client wrapper.

ALL tests use mocks.  ZERO real Anthropic API calls are made.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from context_health.client import AnthropicClient, APIKeyMissingError, CompletionResponse


# ---------------------------------------------------------------------------
# Helpers — fake SDK response objects
# ---------------------------------------------------------------------------


def _make_fake_response(
    *,
    text: str = "Hello from Claude.",
    input_tokens: int = 10,
    output_tokens: int = 5,
    model: str = "claude-sonnet-4-20250514",
    stop_reason: str | None = "end_turn",
) -> MagicMock:
    """Build a MagicMock that mimics ``anthropic.types.Message``."""
    text_block = MagicMock()
    text_block.text = text
    # isinstance() checks need the real type
    from anthropic.types import TextBlock

    text_block.__class__ = TextBlock

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens

    response = MagicMock()
    response.content = [text_block]
    response.usage = usage
    response.model = model
    response.stop_reason = stop_reason
    return response


def _make_client(
    *,
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 1024,
    system_prompt: str | None = None,
) -> AnthropicClient:
    """Create an AnthropicClient with a fake API key (no real calls)."""
    return AnthropicClient(
        model=model,
        max_tokens=max_tokens,
        system_prompt=system_prompt,
        api_key="sk-ant-test-fake-key-for-unit-tests",
    )


# ---------------------------------------------------------------------------
# 1. API key configuration
# ---------------------------------------------------------------------------


class TestAPIKeyConfig:
    """Verify API key is read from param or environment."""

    def test_explicit_api_key(self) -> None:
        """Constructing with an explicit key should succeed."""
        client = _make_client()
        assert client.model == "claude-sonnet-4-20250514"

    def test_env_api_key(self) -> None:
        """Constructing without explicit key should read ANTHROPIC_API_KEY."""
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env-key"}):
            client = AnthropicClient()
            assert client.model == "claude-sonnet-4-20250514"


# ---------------------------------------------------------------------------
# 2. Missing API key handling
# ---------------------------------------------------------------------------


class TestMissingAPIKey:
    """Verify clear error when API key is absent."""

    def test_missing_key_raises(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            # Remove ANTHROPIC_API_KEY if it exists
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(APIKeyMissingError) as exc_info:
                    AnthropicClient()
                assert "ANTHROPIC_API_KEY" in str(exc_info.value)

    def test_error_message_is_actionable(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            env = os.environ.copy()
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(APIKeyMissingError) as exc_info:
                    AnthropicClient()
                msg = str(exc_info.value)
                assert "export" in msg.lower() or "set" in msg.lower()


# ---------------------------------------------------------------------------
# 3. Model selection
# ---------------------------------------------------------------------------


class TestModelSelection:
    """Verify model can be configured."""

    def test_default_model(self) -> None:
        client = _make_client()
        assert client.model == "claude-sonnet-4-20250514"

    def test_custom_model(self) -> None:
        client = _make_client(model="claude-haiku-4-20250414")
        assert client.model == "claude-haiku-4-20250414"


# ---------------------------------------------------------------------------
# 4. System prompt handling
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    """Verify system prompt is forwarded to the SDK."""

    def test_no_system_prompt(self) -> None:
        client = _make_client()
        assert client.system_prompt is None

    def test_system_prompt_stored(self) -> None:
        client = _make_client(system_prompt="You are a helpful assistant.")
        assert client.system_prompt == "You are a helpful assistant."

    @patch("context_health.client.anthropic.Anthropic")
    def test_system_prompt_passed_to_sdk(self, mock_anthropic_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance
        mock_instance.messages.create.return_value = _make_fake_response()

        client = AnthropicClient(
            api_key="sk-ant-test",
            system_prompt="Be concise.",
        )
        client.send([{"role": "user", "content": "Hi"}])

        call_kwargs = mock_instance.messages.create.call_args
        assert call_kwargs.kwargs.get("system") == "Be concise." or \
               (call_kwargs[1].get("system") == "Be concise.")

    @patch("context_health.client.anthropic.Anthropic")
    def test_no_system_key_when_none(self, mock_anthropic_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance
        mock_instance.messages.create.return_value = _make_fake_response()

        client = AnthropicClient(api_key="sk-ant-test")
        client.send([{"role": "user", "content": "Hi"}])

        call_kwargs = mock_instance.messages.create.call_args
        # "system" should not be in kwargs when system_prompt is None
        passed_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert "system" not in passed_kwargs


# ---------------------------------------------------------------------------
# 5. Multi-turn messages
# ---------------------------------------------------------------------------


class TestMultiTurnMessages:
    """Verify multi-turn conversation messages are forwarded."""

    @patch("context_health.client.anthropic.Anthropic")
    def test_messages_forwarded(self, mock_anthropic_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance
        mock_instance.messages.create.return_value = _make_fake_response()

        client = AnthropicClient(api_key="sk-ant-test")
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        client.send(messages)

        call_kwargs = mock_instance.messages.create.call_args
        passed_kwargs = call_kwargs.kwargs if call_kwargs.kwargs else call_kwargs[1]
        assert passed_kwargs["messages"] == messages


# ---------------------------------------------------------------------------
# 6. Assistant response extraction
# ---------------------------------------------------------------------------


class TestResponseExtraction:
    """Verify response text is extracted correctly."""

    @patch("context_health.client.anthropic.Anthropic")
    def test_text_extraction(self, mock_anthropic_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance
        mock_instance.messages.create.return_value = _make_fake_response(
            text="The answer is 42."
        )

        client = AnthropicClient(api_key="sk-ant-test")
        result = client.send([{"role": "user", "content": "What is the answer?"}])

        assert isinstance(result, CompletionResponse)
        assert result.text == "The answer is 42."

    @patch("context_health.client.anthropic.Anthropic")
    def test_multiple_text_blocks(self, mock_anthropic_cls: MagicMock) -> None:
        """When the response has multiple TextBlocks, they are concatenated."""
        from anthropic.types import TextBlock

        block1 = MagicMock()
        block1.__class__ = TextBlock
        block1.text = "Part 1. "

        block2 = MagicMock()
        block2.__class__ = TextBlock
        block2.text = "Part 2."

        usage = MagicMock()
        usage.input_tokens = 8
        usage.output_tokens = 4

        response = MagicMock()
        response.content = [block1, block2]
        response.usage = usage
        response.model = "claude-sonnet-4-20250514"
        response.stop_reason = "end_turn"

        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance
        mock_instance.messages.create.return_value = response

        client = AnthropicClient(api_key="sk-ant-test")
        result = client.send([{"role": "user", "content": "Tell me."}])
        assert result.text == "Part 1. Part 2."


# ---------------------------------------------------------------------------
# 7. Usage / token extraction
# ---------------------------------------------------------------------------


class TestUsageExtraction:
    """Verify token counts and model info are extracted."""

    @patch("context_health.client.anthropic.Anthropic")
    def test_token_counts(self, mock_anthropic_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance
        mock_instance.messages.create.return_value = _make_fake_response(
            input_tokens=150,
            output_tokens=75,
        )

        client = AnthropicClient(api_key="sk-ant-test")
        result = client.send([{"role": "user", "content": "Count."}])

        assert result.input_tokens == 150
        assert result.output_tokens == 75

    @patch("context_health.client.anthropic.Anthropic")
    def test_model_returned(self, mock_anthropic_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance
        mock_instance.messages.create.return_value = _make_fake_response(
            model="claude-haiku-4-20250414",
        )

        client = AnthropicClient(api_key="sk-ant-test")
        result = client.send([{"role": "user", "content": "Hi"}])
        assert result.model == "claude-haiku-4-20250414"

    @patch("context_health.client.anthropic.Anthropic")
    def test_stop_reason(self, mock_anthropic_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance
        mock_instance.messages.create.return_value = _make_fake_response(
            stop_reason="max_tokens",
        )

        client = AnthropicClient(api_key="sk-ant-test")
        result = client.send([{"role": "user", "content": "Hi"}])
        assert result.stop_reason == "max_tokens"


# ---------------------------------------------------------------------------
# 8. API error handling
# ---------------------------------------------------------------------------


class TestAPIErrorHandling:
    """Verify Anthropic API errors propagate cleanly."""

    @patch("context_health.client.anthropic.Anthropic")
    def test_authentication_error(self, mock_anthropic_cls: MagicMock) -> None:
        import anthropic as anth

        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance

        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.headers = {}
        mock_response.text = "Unauthorized"

        mock_instance.messages.create.side_effect = anth.AuthenticationError(
            message="Invalid API key",
            response=mock_response,
            body=None,
        )

        client = AnthropicClient(api_key="sk-ant-bad-key")
        with pytest.raises(anth.AuthenticationError):
            client.send([{"role": "user", "content": "Hi"}])

    @patch("context_health.client.anthropic.Anthropic")
    def test_rate_limit_error(self, mock_anthropic_cls: MagicMock) -> None:
        import anthropic as anth

        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance

        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_response.headers = {}
        mock_response.text = "Rate limited"

        mock_instance.messages.create.side_effect = anth.RateLimitError(
            message="Rate limited",
            response=mock_response,
            body=None,
        )

        client = AnthropicClient(api_key="sk-ant-test")
        with pytest.raises(anth.RateLimitError):
            client.send([{"role": "user", "content": "Hi"}])


# ---------------------------------------------------------------------------
# 9. API key does not appear in logs or returned objects
# ---------------------------------------------------------------------------


class TestAPIKeyNotExposed:
    """Verify the API key is never stored on the client or in responses."""

    def test_key_not_in_client_dict(self) -> None:
        client = _make_client()
        client_attrs = vars(client)
        for attr_name, attr_value in client_attrs.items():
            if isinstance(attr_value, str):
                assert "sk-ant-test-fake-key" not in attr_value, (
                    f"API key found in client attribute: {attr_name}"
                )

    @patch("context_health.client.anthropic.Anthropic")
    def test_key_not_in_response(self, mock_anthropic_cls: MagicMock) -> None:
        mock_instance = MagicMock()
        mock_anthropic_cls.return_value = mock_instance
        mock_instance.messages.create.return_value = _make_fake_response()

        client = AnthropicClient(api_key="sk-ant-secret-key-12345")
        result = client.send([{"role": "user", "content": "Hi"}])

        # Check no field of CompletionResponse contains the key
        assert "sk-ant-secret-key-12345" not in result.text
        assert "sk-ant-secret-key-12345" not in result.model
        assert "sk-ant-secret-key-12345" not in str(result.stop_reason)

    def test_key_not_in_repr(self) -> None:
        client = _make_client()
        client_repr = repr(client)
        assert "sk-ant-test-fake-key" not in client_repr
