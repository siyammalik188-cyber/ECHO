"""Test doubles.

These stand in for a real model so the suite runs offline with no API key. They
live in `tests/` on purpose — nothing in the `echo` package fakes a model.
"""

from __future__ import annotations

from typing import Any

from echo.extraction import MemoryCandidate
from echo.memory import MemoryType


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


class FakeStructuredLLM:
    """Returns a canned extraction payload; records the schema it was handed."""

    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload if payload is not None else {"memories": []}
        self.calls: list[tuple[list[dict[str, str]], dict[str, Any], str | None]] = []

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        system: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(([dict(m) for m in messages], schema, system))
        return self.payload


class FakeExtractor:
    """Proposes a fixed list of candidates. Records every conversation it saw."""

    def __init__(self, candidates: list[MemoryCandidate] | None = None) -> None:
        self.candidates = list(candidates or [])
        self.seen: list[str] = []

    def extract(self, conversation) -> list[MemoryCandidate]:
        self.seen.append(conversation.id)
        return list(self.candidates)


def candidate(
    content: str,
    memory_type: MemoryType = MemoryType.FACT,
    confidence: float = 0.9,
    importance: float = 0.8,
) -> MemoryCandidate:
    """Shorthand for a well-formed candidate that clears the default thresholds."""
    return MemoryCandidate(
        content=content,
        memory_type=memory_type,
        confidence=confidence,
        importance=importance,
    )
