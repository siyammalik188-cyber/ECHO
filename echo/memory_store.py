"""Where persistent memories live: one JSON file, shared across conversations.

Deliberately not a vector database. Retrieval is keyword overlap — enough to
prove the shape of the system, and easy to replace without touching callers.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Iterator

from .memory import Memory

SCHEMA_VERSION = 1
DEFAULT_FILENAME = "memories.json"

_WORD = re.compile(r"[a-z0-9']+")

# Words too common to say anything about relevance.
_STOPWORDS = frozenset(
    """a an and are as at be by do does for from had has have he her his i in is it
    its me my of on or our she that the their them they this to was we were what
    when where which who will with you your""".split()
)


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS}


class MemoryStore:
    """An in-memory collection of `Memory` records with a JSON file behind it.

    Mutations stay in memory until `save()` is called, so a caller can decide
    when a batch of changes becomes durable.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._memories: dict[str, Memory] = {}

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path: Path | str) -> "MemoryStore":
        """Open the store at `path`. A missing file is an empty store, not an error."""
        store = cls(path)
        if not store.path.is_file():
            return store

        with store.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported memory schema_version {version!r} "
                f"(this build reads {SCHEMA_VERSION})"
            )
        for record in payload.get("memories", []):
            memory = Memory.from_dict(record)
            store._memories[memory.id] = memory
        return store

    @classmethod
    def in_directory(cls, directory: Path | str) -> "MemoryStore":
        return cls.load(Path(directory) / DEFAULT_FILENAME)

    # ----------------------------------------------------------------- write

    def add(self, memory: Memory) -> bool:
        """Store `memory`. Returns False if identical content is already held.

        The duplicate check is exact-match only. It exists so re-running
        extraction over the same conversation does not pile up copies — it is
        not belief updating, and it does not merge, reconcile, or supersede
        anything.
        """
        normalized = memory.content.strip().lower()
        for existing in self._memories.values():
            if existing.content.strip().lower() == normalized:
                return False
        self._memories[memory.id] = memory
        return True

    def save(self) -> Path:
        """Write the store to disk atomically (temp file, then rename)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "memories": [m.to_dict() for m in self._memories.values()],
        }
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self.path)
        return self.path

    # ------------------------------------------------------------------ read

    def get(self, memory_id: str, touch: bool = True) -> Memory | None:
        memory = self._memories.get(memory_id)
        if memory is not None and touch:
            memory.touch()
        return memory

    def all(self) -> list[Memory]:
        """Every memory, newest first. Does not count as an access."""
        return sorted(self._memories.values(), key=lambda m: m.created_at, reverse=True)

    def search(self, query: str, limit: int = 5, touch: bool = True) -> list[Memory]:
        """Return memories whose content overlaps `query`, best match first.

        Ranked by word overlap, then importance, then confidence. Retrieval is
        an access: matched memories get `last_accessed` and `access_count`
        updated unless `touch=False`.
        """
        wanted = _tokens(query)
        if not wanted:
            return []

        scored: list[tuple[float, Memory]] = []
        for memory in self._memories.values():
            overlap = wanted & _tokens(memory.content)
            if overlap:
                scored.append((len(overlap) / len(wanted), memory))

        scored.sort(
            key=lambda pair: (pair[0], pair[1].importance, pair[1].confidence),
            reverse=True,
        )
        results = [memory for _, memory in scored[:limit]]
        if touch:
            for memory in results:
                memory.touch()
        return results

    def __len__(self) -> int:
        return len(self._memories)

    def __iter__(self) -> Iterator[Memory]:
        return iter(self.all())
