"""The discovery ledger: a permanent, restart-surviving record of one search.

For every promoted hypothesis the challenge asks eight questions. This record
answers all of them from stored data, without re-running anything:

- What did I discover?              → `discovered_expression`
- Which observations led to it?     → `evidence_window`, `discovery_reason`
- What alternatives did I test?     → `alternatives`
- Why did it beat them?             → `winner_rank` and the penalised scores
- How complex is it?               → `complexity`
- Did it survive unseen data?       → `survived_holdout`, `test_score`
- What was performance before?      → `baseline_before` (the base rate)
- What happened after?              → `test_score` vs the baselines on the holdout

A `DiscoveryRecord` is frozen and append-only within a ledger. Rejected and
retired hypotheses are kept, so a later reader sees not only what won but what
was tried and thrown away.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hypothesis import Hypothesis

SCHEMA_VERSION = 1
DEFAULT_FILENAME = "discovery.json"


@dataclass(frozen=True)
class DiscoveryRecord:
    observations_id: str
    outcome: str  # DISCOVERY SUCCESS | DISCOVERY PARTIAL | DISCOVERY FAILURE
    config: dict[str, Any]
    source_name: str
    proposed: int
    screened: int
    fitted: int
    rejected_count: int
    validated_count: int
    promoted_count: int
    discovered_expression: str | None
    discovered_complexity: int | None
    discovery_reason: str | None
    evidence_window: tuple[int, int] | None
    confirmation: dict[str, Any] | None
    survived_holdout: bool | None
    training_score: dict[str, Any] | None
    validation_score: dict[str, Any] | None
    test_score: dict[str, Any] | None
    baseline_before: dict[str, Any] | None
    holdout_baselines: list[dict[str, Any]]
    alternatives: list[dict[str, Any]]
    overfitting_control: dict[str, Any] | None
    hypotheses: tuple[Hypothesis, ...] = field(default_factory=tuple)
    seconds: float = 0.0

    def provenance(self) -> dict[str, Any]:
        """The eight-question answer, assembled from stored fields alone."""
        return {
            "what_did_i_discover": self.discovered_expression or "nothing reliable",
            "which_observations_led_to_it": {
                "evidence_window": list(self.evidence_window) if self.evidence_window else None,
                "reason": self.discovery_reason,
            },
            "what_alternatives_did_i_test": self.alternatives,
            "why_did_it_beat_them": (
                "lowest complexity-penalised Brier on VAL_A, then confirmed on VAL_B; "
                "see confirmation"
                if self.discovered_expression
                else "no candidate cleared confirmation"
            ),
            "how_complex_is_it": self.discovered_complexity,
            "did_it_survive_unseen_data": self.survived_holdout,
            "performance_before_discovery": self.baseline_before,
            "performance_after_discovery": self.test_score,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations_id": self.observations_id,
            "outcome": self.outcome,
            "config": self.config,
            "source_name": self.source_name,
            "proposed": self.proposed,
            "screened": self.screened,
            "fitted": self.fitted,
            "rejected_count": self.rejected_count,
            "validated_count": self.validated_count,
            "promoted_count": self.promoted_count,
            "discovered_expression": self.discovered_expression,
            "discovered_complexity": self.discovered_complexity,
            "discovery_reason": self.discovery_reason,
            "evidence_window": list(self.evidence_window) if self.evidence_window else None,
            "confirmation": self.confirmation,
            "survived_holdout": self.survived_holdout,
            "training_score": self.training_score,
            "validation_score": self.validation_score,
            "test_score": self.test_score,
            "baseline_before": self.baseline_before,
            "holdout_baselines": self.holdout_baselines,
            "alternatives": self.alternatives,
            "overfitting_control": self.overfitting_control,
            "hypotheses": [h.to_dict() for h in self.hypotheses],
            "seconds": self.seconds,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiscoveryRecord":
        window = payload.get("evidence_window")
        return cls(
            observations_id=str(payload["observations_id"]),
            outcome=str(payload["outcome"]),
            config=dict(payload.get("config", {})),
            source_name=str(payload.get("source_name", "")),
            proposed=int(payload["proposed"]),
            screened=int(payload["screened"]),
            fitted=int(payload["fitted"]),
            rejected_count=int(payload["rejected_count"]),
            validated_count=int(payload["validated_count"]),
            promoted_count=int(payload["promoted_count"]),
            discovered_expression=payload.get("discovered_expression"),
            discovered_complexity=payload.get("discovered_complexity"),
            discovery_reason=payload.get("discovery_reason"),
            evidence_window=tuple(window) if window else None,  # type: ignore[arg-type]
            confirmation=payload.get("confirmation"),
            survived_holdout=payload.get("survived_holdout"),
            training_score=payload.get("training_score"),
            validation_score=payload.get("validation_score"),
            test_score=payload.get("test_score"),
            baseline_before=payload.get("baseline_before"),
            holdout_baselines=list(payload.get("holdout_baselines", [])),
            alternatives=list(payload.get("alternatives", [])),
            overfitting_control=payload.get("overfitting_control"),
            hypotheses=tuple(
                Hypothesis.from_dict(h) for h in payload.get("hypotheses", [])
            ),
            seconds=float(payload.get("seconds", 0.0)),
        )


class DiscoveryLedger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._records: list[DiscoveryRecord] = []

    @classmethod
    def load(cls, path: Path | str) -> "DiscoveryLedger":
        ledger = cls(path)
        if not ledger.path.is_file():
            return ledger
        with ledger.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported discovery schema_version {version!r} "
                f"(this build reads {SCHEMA_VERSION})"
            )
        ledger._records = [
            DiscoveryRecord.from_dict(record) for record in payload.get("records", [])
        ]
        return ledger

    @classmethod
    def in_directory(cls, directory: Path | str) -> "DiscoveryLedger":
        return cls.load(Path(directory) / DEFAULT_FILENAME)

    def add(self, record: DiscoveryRecord) -> DiscoveryRecord:
        self._records.append(record)
        return record

    def records(self) -> tuple[DiscoveryRecord, ...]:
        return tuple(self._records)

    def by_observations(self, observations_id: str) -> DiscoveryRecord | None:
        for record in reversed(self._records):
            if record.observations_id == observations_id:
                return record
        return None

    def __len__(self) -> int:
        # Deliberately not used in a boolean context anywhere. Same footgun as
        # Conversation.__len__ and LearningLedger.__len__: an empty ledger is
        # falsy, so callers must ask `len(ledger) == 0`, never `if ledger:`.
        return len(self._records)

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "records": [record.to_dict() for record in self._records],
        }
        tmp = self.path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(self.path)
        return self.path
