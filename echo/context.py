"""Temporary context: the third kind of state, and the only one that is not durable.

A `WorkingContext` holds what matters for the turn being taken right now —
memories just recalled, notes assembled to steer the next reply. It has no
`to_dict`, no `save`, and nothing in `storage.py` or `memory_store.py` will
write it. When the process ends it is gone, by design.

The absence of serialization here is the feature. If temporary context could be
saved it would eventually be saved, and the line between "what ECHO knows" and
"what ECHO happened to be holding" would blur.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .memory import Memory


@dataclass
class WorkingContext:
    notes: list[str] = field(default_factory=list)
    recalled: list[Memory] = field(default_factory=list)

    def add_note(self, note: str) -> None:
        if note.strip():
            self.notes.append(note.strip())

    def add_memories(self, memories: list[Memory]) -> None:
        """Attach recalled memories, skipping ones already present."""
        held = {m.id for m in self.recalled}
        self.recalled.extend(m for m in memories if m.id not in held)

    def clear(self) -> None:
        self.notes.clear()
        self.recalled.clear()

    def is_empty(self) -> bool:
        return not self.notes and not self.recalled

    def render(self) -> str | None:
        """Format the context for appending to a system prompt, or None if empty.

        This text is passed as a request parameter for a single call. It is
        never appended to the conversation, so it does not become history.
        """
        if self.is_empty():
            return None

        sections: list[str] = []
        if self.recalled:
            lines = "\n".join(f"- {m.content}" for m in self.recalled)
            sections.append(f"What you remember about this user:\n{lines}")
        if self.notes:
            lines = "\n".join(f"- {note}" for note in self.notes)
            sections.append(f"Notes for this turn:\n{lines}")
        return "\n\n".join(sections)

    def __len__(self) -> int:
        return len(self.notes) + len(self.recalled)
