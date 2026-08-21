"""The learning ledger: what was learned, what caused it, and whether it helped.

Append-only, like the prediction ledger. It answers, permanently:

- What did I learn?              → `revisions()` and `experiences()`
- What experience caused it?     → `triggering_experiences(revision_id)`
- What strategy changed?         → `from_strategy_id` / `to_strategy_id`
- Why did I change it?           → `reason` and the attached change signal
- Did it actually improve?       → `outcome`, `measured_before`, `measured_after`
- What would reverse it?         → `reversal_condition`

Strategies are never deleted. A retired strategy keeps its full performance
history, so a later reader can see what was tried and how it went.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .learning import LearningExperience, StrategyRevision
from .strategy import Strategy

SCHEMA_VERSION = 1
DEFAULT_FILENAME = "learning.json"


class LearningLedger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._experiences: list[LearningExperience] = []
        self._revisions: list[StrategyRevision] = []
        self._strategies: dict[str, Strategy] = {}
        self._strategy_order: list[str] = []

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path: Path | str) -> "LearningLedger":
        ledger = cls(path)
        if not ledger.path.is_file():
            return ledger

        with ledger.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported learning schema_version {version!r} "
                f"(this build reads {SCHEMA_VERSION})"
            )
        for record in payload.get("strategies", []):
            strategy = Strategy.from_dict(record)
            ledger._strategies[strategy.id] = strategy
            ledger._strategy_order.append(strategy.id)
        ledger._experiences = [
            LearningExperience.from_dict(e) for e in payload.get("experiences", [])
        ]
        ledger._revisions = [
            StrategyRevision.from_dict(r) for r in payload.get("revisions", [])
        ]
        return ledger

    @classmethod
    def in_directory(cls, directory: Path | str) -> "LearningLedger":
        return cls.load(Path(directory) / DEFAULT_FILENAME)

    # ----------------------------------------------------------------- write

    def register_strategy(self, strategy: Strategy) -> Strategy:
        if strategy.id not in self._strategies:
            self._strategies[strategy.id] = strategy
            self._strategy_order.append(strategy.id)
        return strategy

    def add_experience(self, experience: LearningExperience) -> LearningExperience:
        self._experiences.append(experience)
        return experience

    def add_revision(self, revision: StrategyRevision) -> StrategyRevision:
        self._revisions.append(revision)
        return revision

    def conclude_revision(
        self, revision_id: str, outcome: str, before: dict[str, Any], after: dict[str, Any]
    ) -> StrategyRevision:
        """Record the verdict on a revision, in place in the sequence.

        The original revision object is not mutated — a new one carrying the
        verdict replaces it at the same position, so the ordering of history is
        preserved without any field being overwritten in place.
        """
        for index, revision in enumerate(self._revisions):
            if revision.id == revision_id:
                concluded = revision.concluded(outcome, before, after)
                self._revisions[index] = concluded
                return concluded
        raise KeyError(f"no revision {revision_id!r} in this ledger")

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "strategies": [self._strategies[sid].to_dict() for sid in self._strategy_order],
            "experiences": [e.to_dict() for e in self._experiences],
            "revisions": [r.to_dict() for r in self._revisions],
        }
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self.path)
        return self.path

    # ------------------------------------------------------------------ read

    def experiences(self) -> tuple[LearningExperience, ...]:
        return tuple(self._experiences)

    def revisions(self) -> tuple[StrategyRevision, ...]:
        return tuple(self._revisions)

    def strategies(self) -> tuple[Strategy, ...]:
        return tuple(self._strategies[sid] for sid in self._strategy_order)

    def get_strategy(self, strategy_id: str) -> Strategy | None:
        return self._strategies.get(strategy_id)

    def active_strategy(self) -> Strategy | None:
        for strategy in reversed(self.strategies()):
            if strategy.status.value == "active":
                return strategy
        return None

    def triggering_experiences(self, revision_id: str) -> tuple[LearningExperience, ...]:
        for revision in self._revisions:
            if revision.id == revision_id:
                wanted = set(revision.triggering_experience_ids)
                return tuple(e for e in self._experiences if e.id in wanted)
        return ()

    def explanations(self) -> dict[str, int]:
        """How often each failure explanation was reached."""
        counts: dict[str, int] = {}
        for experience in self._experiences:
            key = experience.failure_analysis.explanation.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    def what_did_i_learn(self) -> list[dict[str, Any]]:
        """The plain-language answer, one entry per strategy revision."""
        out = []
        for revision in self._revisions:
            source = self._strategies.get(revision.from_strategy_id)
            target = self._strategies.get(revision.to_strategy_id)
            out.append(
                {
                    "at_trial": revision.at_trial,
                    "changed_from": (
                        f"{source.description} (v{source.version})" if source else revision.from_strategy_id
                    ),
                    "changed_to": (
                        f"{target.description} (v{target.version})" if target else revision.to_strategy_id
                    ),
                    "why": revision.reason,
                    "confidence": revision.confidence,
                    "triggered_by": list(revision.triggering_experience_ids),
                    "did_it_help": revision.outcome,
                    "before": revision.measured_before,
                    "after": revision.measured_after,
                    "would_reverse_if": revision.reversal_condition,
                }
            )
        return out

    def __len__(self) -> int:
        return len(self._experiences)
