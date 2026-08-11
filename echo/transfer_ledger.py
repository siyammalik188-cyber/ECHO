"""Persistent, append-only history of patterns and the transfers they drove.

Two things are stored: every version of every `StructuralPattern`, and every
`TransferRecord`. Neither is ever edited. A pattern that is weakened keeps the
version that was confident, and a transfer that turned out badly keeps the
record saying it was attempted — "never rewrite historical transfer decisions"
is enforced by there being no method that could.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .pattern import StructuralPattern
from .transfer import TransferRecord

SCHEMA_VERSION = 1
DEFAULT_FILENAME = "transfer.json"


class TransferLedger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._patterns: list[StructuralPattern] = []
        self._records: list[TransferRecord] = []

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path: Path | str) -> "TransferLedger":
        ledger = cls(path)
        if not ledger.path.is_file():
            return ledger
        with ledger.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported transfer schema_version {version!r} "
                f"(this build reads {SCHEMA_VERSION})"
            )
        ledger._patterns = [
            StructuralPattern.from_dict(p) for p in payload.get("patterns", [])
        ]
        ledger._records = [
            TransferRecord.from_dict(r) for r in payload.get("transfers", [])
        ]
        return ledger

    @classmethod
    def in_directory(cls, directory: Path | str) -> "TransferLedger":
        return cls.load(Path(directory) / DEFAULT_FILENAME)

    # ----------------------------------------------------------------- write

    def add_pattern(self, pattern: StructuralPattern) -> StructuralPattern:
        """Append a pattern version. Earlier versions are never replaced."""
        self._patterns.append(pattern)
        return pattern

    def add_transfer(self, record: TransferRecord) -> TransferRecord:
        self._records.append(record)
        return record

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "patterns": [p.to_dict() for p in self._patterns],
            "transfers": [r.to_dict() for r in self._records],
        }
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self.path)
        return self.path

    # ------------------------------------------------------------------ read

    def patterns(self) -> tuple[StructuralPattern, ...]:
        return tuple(self._patterns)

    def transfers(self) -> tuple[TransferRecord, ...]:
        return tuple(self._records)

    def versions(self, pattern_id: str) -> tuple[StructuralPattern, ...]:
        """Every recorded version of one pattern, oldest first."""
        return tuple(p for p in self._patterns if p.pattern_id == pattern_id)

    def current(self, pattern_id: str) -> StructuralPattern | None:
        versions = self.versions(pattern_id)
        return versions[-1] if versions else None

    def for_target(self, environment_id: str) -> tuple[TransferRecord, ...]:
        return tuple(
            r for r in self._records if r.target_environment_id == environment_id
        )

    def counts(self) -> dict[str, int]:
        """How the transfers turned out, by result."""
        out: dict[str, int] = {}
        for record in self._records:
            out[record.result.value] = out.get(record.result.value, 0) + 1
        return out

    def __len__(self) -> int:
        # As elsewhere in ECHO: never use a ledger in a boolean context. An
        # empty one is falsy, which has caused a real bug here before.
        return len(self._records)
