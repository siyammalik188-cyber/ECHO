"""The LLM boundary: send messages out, get one text reply back.

Everything that knows about a specific provider lives in this file. The rest of
ECHO talks to the `LLMClient` protocol, so swapping providers (or dropping in a
fake for tests) means writing one class with one method.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_MAX_TOKENS = 16000

# Opus 5's safety classifiers can decline a request. With this opted in, the API
# re-runs the declined request on Anthropic's recommended fallback model inside
# the same call instead of handing back an empty response. Delete the three
# `betas=` / `fallbacks=` lines below (and switch back to `client.messages`) to
# turn it off.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMError(RuntimeError):
    """The model did not return a usable reply."""


@runtime_checkable
class LLMClient(Protocol):
    """What ECHO needs from a language model. That's the whole interface."""

    def complete(
        self, messages: list[dict[str, str]], system: str | None = None
    ) -> str:
        """Return the assistant's reply to `messages`."""
        ...


class AnthropicLLM:
    """`LLMClient` backed by the Anthropic Messages API.

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

    def complete(
        self, messages: list[dict[str, str]], system: str | None = None
    ) -> str:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": messages,
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
        if system:
            request["system"] = system

        response = self._client.beta.messages.create(**request)

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
