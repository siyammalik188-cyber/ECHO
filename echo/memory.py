"""The persistent memory record.

This is the second of ECHO's three kinds of state. The three are deliberately
distinct and must not be confused:

1. **Conversation history** (`conversation.py`) — the verbatim transcript of one
   conversation. Durable, but it is a record of what was *said*, not of what is
   *known*. It is never treated as memory.
2. **Persistent memory** (this file) — discrete, extracted facts that outlive the
   conversation they came from and are visible to every later conversation.
   Nothing becomes a memory except through the extraction step in
   `extraction.py`.
3. **Temporary context** (`context.py`) — per-turn scratch state. Never written
   to disk, never survives the process.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class MemoryType(str, Enum):
    """What kind of thing this memory is.

    Kept small on purpose. A type that cannot be acted on differently from
    another type is not worth having.
    """

    IDENTITY = "identity"  # who the user is: name, role, location
    PREFERENCE = "preference"  # how they want things done
    FACT = "fact"  # a durable fact about their world
    GOAL = "goal"  # something they are trying to achieve
    DECISION = "decision"  # a choice that was made and should stick


@dataclass
class Memory:
    """One persistent memory.

    `confidence` is how sure we are the statement is true; `importance` is how
    much it should matter later. They are separate on purpose — a throwaway
    remark can be certainly true and not worth keeping.
    """

    content: str
    memory_type: MemoryType
    source_conversation: str
    confidence: float
    importance: float
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: str = field(default_factory=_now)
    last_accessed: str | None = None
    access_count: int = 0

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("memory content must not be empty")
        self.memory_type = MemoryType(self.memory_type)
        for name in ("confidence", "importance"):
            value = getattr(self, name)
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be within 0.0–1.0, got {value!r}")
            setattr(self, name, float(value))

    def touch(self, when: str | None = None) -> None:
        """Record that this memory was retrieved."""
        self.last_accessed = when or _now()
        self.access_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "source_conversation": self.source_conversation,
            "confidence": self.confidence,
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Memory":
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            source_conversation=data["source_conversation"],
            confidence=data["confidence"],
            importance=data["importance"],
            created_at=data["created_at"],
            last_accessed=data.get("last_accessed"),
            access_count=data.get("access_count", 0),
        )
