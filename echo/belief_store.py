"""Persistence for beliefs and the evidence behind them.

One JSON file holding both, because a belief without its evidence is not
auditable: the revision history names evidence ids, and those ids have to
resolve to something after a restart.

Mirrors `memory_store.py` — atomic write, schema version, missing file is an
empty store rather than an error.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterator

from .belief import Belief, Evidence

SCHEMA_VERSION = 1
DEFAULT_FILENAME = "beliefs.json"


class BeliefStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._beliefs: dict[str, Belief] = {}
        self._evidence: dict[str, Evidence] = {}

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path: Path | str) -> "BeliefStore":
        store = cls(path)
        if not store.path.is_file():
            return store

        with store.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported belief schema_version {version!r} "
                f"(this build reads {SCHEMA_VERSION})"
            )
        for record in payload.get("evidence", []):
            evidence = Evidence.from_dict(record)
            store._evidence[evidence.id] = evidence
        for record in payload.get("beliefs", []):
            belief = Belief.from_dict(record)
            store._beliefs[belief.id] = belief
        return store

    @classmethod
    def in_directory(cls, directory: Path | str) -> "BeliefStore":
        return cls.load(Path(directory) / DEFAULT_FILENAME)

    # ----------------------------------------------------------------- write

    def add_belief(self, belief: Belief) -> Belief:
        if belief.id in self._beliefs:
            raise ValueError(f"belief {belief.id} is already in this store")
        self._beliefs[belief.id] = belief
        return belief

    def add_evidence(self, evidence: Evidence) -> Evidence:
        if evidence.id in self._evidence:
            raise ValueError(f"evidence {evidence.id} is already in this store")
        self._evidence[evidence.id] = evidence
        return evidence

    def consider(self, belief_id: str, evidence: Evidence, **kwargs):
        """Register `evidence` (if new) and weigh it against a belief.

        The convenience path: keeps the evidence registry and the belief's
        history in step, so a revision can never reference evidence the store
        cannot produce later.
        """
        belief = self.get_belief(belief_id)
        if belief is None:
            raise KeyError(f"no belief {belief_id!r} in this store")
        if evidence.id not in self._evidence:
            self.add_evidence(evidence)
        return belief.consider(evidence, **kwargs)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "beliefs": [b.to_dict() for b in self._beliefs.values()],
            "evidence": [e.to_dict() for e in self._evidence.values()],
        }
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self.path)
        return self.path

    # ------------------------------------------------------------------ read

    def get_belief(self, belief_id: str) -> Belief | None:
        return self._beliefs.get(belief_id)

    def get_evidence(self, evidence_id: str) -> Evidence | None:
        return self._evidence.get(evidence_id)

    def beliefs(self) -> list[Belief]:
        return sorted(self._beliefs.values(), key=lambda b: b.created_at)

    def evidence(self) -> list[Evidence]:
        return sorted(self._evidence.values(), key=lambda e: e.recorded_at)

    def evidence_for(self, belief_id: str) -> list[Evidence]:
        """Every evidence item this belief has considered, in the order it saw them."""
        belief = self.get_belief(belief_id)
        if belief is None:
            return []
        found = []
        for revision in belief.revision_history:
            evidence = self._evidence.get(revision.evidence_id)
            if evidence is not None:
                found.append(evidence)
        return found

    def __len__(self) -> int:
        return len(self._beliefs)

    def __iter__(self) -> Iterator[Belief]:
        return iter(self.beliefs())
