"""Test doubles.

`FakeLLM` is a stand-in for a real model so the test suite runs offline with no
API key. It lives in `tests/` on purpose — nothing in the `echo` package fakes
a model response.
"""

from __future__ import annotations


class FakeLLM:
    """Records what it was asked and returns canned replies."""

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or ["ok"])
        self.calls: list[tuple[list[dict[str, str]], str | None]] = []

    def complete(
        self, messages: list[dict[str, str]], system: str | None = None
    ) -> str:
        self.calls.append(([dict(m) for m in messages], system))
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class ExplodingLLM:
    """Always fails, for testing error paths."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc or RuntimeError("boom")

    def complete(
        self, messages: list[dict[str, str]], system: str | None = None
    ) -> str:
        raise self.exc
