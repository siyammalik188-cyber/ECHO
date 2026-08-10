"""The deliberate step between "was said" and "is remembered".

Nothing in ECHO turns conversation text into a memory except `consolidate()`,
and it is never called on your behalf — `Agent.send()` does not touch it. Saving
a whole transcript and calling it memory is exactly what this module exists to
avoid.

Two gates stand between a conversation and the store:

1. **Extraction.** A separate model call, with its own prompt, reads the
   transcript and proposes discrete durable statements. Most conversation text
   produces nothing.
2. **Thresholds.** Proposals below `MIN_CONFIDENCE` or `MIN_IMPORTANCE` are
   discarded here, in ordinary code, regardless of what the extractor thought.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .conversation import Conversation
from .llm import StructuredLLMClient
from .memory import Memory, MemoryType
from .memory_store import MemoryStore

MIN_CONFIDENCE = 0.6
MIN_IMPORTANCE = 0.3


@dataclass
class MemoryCandidate:
    """A proposed memory. Not yet stored, and may never be."""

    content: str
    memory_type: MemoryType
    confidence: float
    importance: float

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("candidate content must not be empty")
        self.memory_type = MemoryType(self.memory_type)
        for name in ("confidence", "importance"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within 0.0–1.0, got {value!r}")
            setattr(self, name, value)


@runtime_checkable
class MemoryExtractor(Protocol):
    """Reads a conversation, proposes candidates. May legitimately return none."""

    def extract(self, conversation: Conversation) -> list[MemoryCandidate]: ...


EXTRACTION_SYSTEM = """\
You review a conversation transcript and pull out only the information worth \
remembering after the conversation ends.

Record a memory only for durable facts about the user or their world: who they \
are, stable preferences about how they want things done, concrete facts they \
stated about their situation, goals they are pursuing, and decisions they made \
that should stick.

Do not record:
- greetings, small talk, thanks, or conversational filler
- questions the user asked, or anything you said
- anything true only right now (what they are doing this minute, the weather)
- anything you inferred but were not told
- a summary of the conversation — you are extracting statements, not summarizing

Most conversations contain nothing worth recording. Returning an empty list is \
the correct and common answer. Do not invent memories to fill the list.

Each memory must be one self-contained sentence that still makes sense with no \
conversation around it. Write it in the third person about the user.

confidence: how certain you are the statement is true, 0.0-1.0.
importance: how much it should matter in future conversations, 0.0-1.0.\
"""

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "memories": {
            "type": "array",
            "description": "Durable memories found. Empty when there are none.",
            "items": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "One self-contained sentence about the user.",
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": [t.value for t in MemoryType],
                    },
                    "confidence": {"type": "number"},
                    "importance": {"type": "number"},
                },
                "required": ["content", "memory_type", "confidence", "importance"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["memories"],
    "additionalProperties": False,
}


def render_transcript(conversation: Conversation) -> str:
    return "\n".join(f"{m.role}: {m.content}" for m in conversation.messages)


class LLMMemoryExtractor:
    """`MemoryExtractor` that asks a model, in its own call with its own prompt.

    Separate from the conversation's own model call on purpose: extraction is a
    review of what was said, not a continuation of it.
    """

    def __init__(self, llm: StructuredLLMClient) -> None:
        self.llm = llm

    def extract(self, conversation: Conversation) -> list[MemoryCandidate]:
        if not conversation.messages:
            return []

        payload = self.llm.complete_structured(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Extract durable memories from this transcript.\n\n"
                        f"<transcript>\n{render_transcript(conversation)}\n</transcript>"
                    ),
                }
            ],
            system=EXTRACTION_SYSTEM,
            schema=EXTRACTION_SCHEMA,
        )
        return self._to_candidates(payload)

    @staticmethod
    def _to_candidates(payload: dict[str, Any]) -> list[MemoryCandidate]:
        """Convert a raw extraction payload, dropping anything malformed.

        A candidate we cannot parse is discarded rather than stored badly — one
        bad row should not fail the whole run or poison the store.
        """
        candidates: list[MemoryCandidate] = []
        for row in payload.get("memories", []):
            try:
                candidates.append(
                    MemoryCandidate(
                        content=row["content"],
                        memory_type=MemoryType(row["memory_type"]),
                        confidence=row["confidence"],
                        importance=row["importance"],
                    )
                )
            except (KeyError, ValueError, TypeError):
                continue
        return candidates


def consolidate(
    conversation: Conversation,
    extractor: MemoryExtractor,
    store: MemoryStore,
    min_confidence: float = MIN_CONFIDENCE,
    min_importance: float = MIN_IMPORTANCE,
) -> list[Memory]:
    """Run extraction over `conversation` and store what clears the thresholds.

    Returns only the memories actually added — candidates that were filtered
    out, and ones already held verbatim, are not included.
    """
    added: list[Memory] = []
    for candidate in extractor.extract(conversation):
        if candidate.confidence < min_confidence:
            continue
        if candidate.importance < min_importance:
            continue
        memory = Memory(
            content=candidate.content,
            memory_type=candidate.memory_type,
            source_conversation=conversation.id,
            confidence=candidate.confidence,
            importance=candidate.importance,
        )
        if store.add(memory):
            added.append(memory)
    return added


def parse_extraction_json(text: str) -> dict[str, Any]:
    """Helper for callers holding raw JSON text rather than a parsed payload."""
    return json.loads(text)
