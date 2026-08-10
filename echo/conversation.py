"""Conversation state: the messages ECHO is currently holding onto."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

SCHEMA_VERSION = 1
VALID_ROLES = ("user", "assistant")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Message:
    role: str
    content: str
    timestamp: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.role not in VALID_ROLES:
            raise ValueError(f"role must be one of {VALID_ROLES}, got {self.role!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", _now()),
        )


@dataclass
class Conversation:
    """An ordered list of messages plus an optional system prompt.

    This is plain in-memory state. It does not talk to an LLM and it does not
    touch the disk; `Agent` and `storage` do those two things respectively.
    """

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    system: str | None = None
    messages: list[Message] = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    def add(self, role: str, content: str) -> Message:
        message = Message(role=role, content=content)
        self.messages.append(message)
        return message

    def add_user(self, content: str) -> Message:
        return self.add("user", content)

    def add_assistant(self, content: str) -> Message:
        return self.add("assistant", content)

    def to_api_messages(self) -> list[dict[str, str]]:
        """The `messages` payload the Claude API expects.

        The system prompt is deliberately excluded — it is a separate
        top-level request parameter, not a message.
        """
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "id": self.id,
            "system": self.system,
            "created_at": self.created_at,
            "messages": [m.to_dict() for m in self.messages],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Conversation":
        version = data.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {version!r} (this build reads {SCHEMA_VERSION})"
            )
        return cls(
            id=data["id"],
            system=data.get("system"),
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            created_at=data.get("created_at", _now()),
        )

    def __len__(self) -> int:
        return len(self.messages)
