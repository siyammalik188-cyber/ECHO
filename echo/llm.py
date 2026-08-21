"""The LLM boundary: send messages out, get a reply back.

Everything that knows about a specific provider lives in this file. The rest of
ECHO talks to the protocols below, so swapping providers (or dropping in a fake
for tests) means writing one small class.

Two protocols, because ECHO asks the model for two different shapes of answer:
`LLMClient` for a conversational reply, `StructuredLLMClient` for JSON matching
a schema (used by memory extraction).
"""

from __future__ import annotations

import json
from typing import Any, Protocol, runtime_checkable

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16000

# Opus 5's safety classifiers can decline a request. With this opted in, the API
# re-runs the declined request on Anthropic's recommended fallback model inside
# the same call instead of handing back an empty response. Delete the two
# `betas` / `fallbacks` entries in `_request()` to turn it off.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """The model did not return a usable reply."""


@runtime_checkable
class LLMClient(Protocol):
    """What a conversation needs from a language model."""

    def complete(
        self, messages: list[dict[str, str]], system: str | None = None
    ) -> str:
        """Return the assistant's reply to `messages`."""
        ...


@runtime_checkable
class StructuredLLMClient(Protocol):
    """What memory extraction needs: an answer shaped like a given schema."""

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        system: str | None = None,
    ) -> dict[str, Any]:
        """Return a dict conforming to `schema`."""
        ...


class AnthropicLLM:
    """Implements both protocols against the Anthropic Messages API.

    Reads credentials the way the SDK does: `ANTHROPIC_API_KEY`, then
    `ANTHROPIC_AUTH_TOKEN`, then an `ant auth login` profile.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        if client is not None:
            self._client = client
        else:
            import anthropic  # imported here so the rest of ECHO works without the SDK

            self._client = (
                anthropic.Anthropic(api_key=api_key)
                if api_key
                else anthropic.Anthropic()
            )

    # --------------------------------------------------------------- helpers

    def _request(
        self, messages: list[dict[str, str]], system: str | None
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
        if system:
            request["system"] = system
        return request

    def _text(self, response: Any) -> str:
        """Pull the reply text out of a response, or raise a useful error."""
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            category = getattr(details, "category", None)
            raise LLMError(f"model declined the request (category={category})")

        text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        if not text:
            raise LLMError(
                f"no text in response (stop_reason={getattr(response, 'stop_reason', None)})"
            )
        return text

    # --------------------------------------------------------------- the API

    def complete(
        self, messages: list[dict[str, str]], system: str | None = None
    ) -> str:
        request = self._request(messages, system)
        return self._text(self._client.beta.messages.create(**request))

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        system: str | None = None,
    ) -> dict[str, Any]:
        request = self._request(messages, system)
        request["output_config"] = {"format": {"type": "json_schema", "schema": schema}}

        text = self._text(self._client.beta.messages.create(**request))
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"structured response was not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise LLMError(
                f"structured response was {type(payload).__name__}, expected an object"
            )
        return payload
