"""Strategies: the thing that actually changes when ECHO learns.

A **strategy** is a named, versioned way of turning history into a probability.
Learning, in this system, means proposing a different strategy on the evidence
of accumulated errors, testing it, and promoting or discarding it on measured
performance. It does not mean nudging a number.

Statuses form a one-way-ish lifecycle:

    CANDIDATE → TESTING → ACTIVE
                   ↓         ↓
                RETIRED   WEAKENED → RETIRED

Nothing is ever deleted. A retired strategy stays in the registry with its
performance history intact, so "what did I used to do, and how did it go" is
always answerable.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from .predictors import LAPLACE_ALL, predict


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


class StrategyStatus(str, Enum):
    CANDIDATE = "candidate"  # proposed, never used
    TESTING = "testing"  # issuing predictions on trial
    ACTIVE = "active"  # the strategy in use
    WEAKENED = "weakened"  # superseded, kept for reference
    RETIRED = "retired"  # tried and failed, or replaced


@dataclass(frozen=True)
class PerformanceRecord:
    """How a strategy did over one stretch of predictions."""

    label: str  # e.g. "trials 30-39", "test window"
    predictions: int
    mean_brier: float
    miss_rate: float
    at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "predictions": self.predictions,
            "mean_brier": round(self.mean_brier, 6),
            "miss_rate": round(self.miss_rate, 6),
            "at": self.at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PerformanceRecord":
        return cls(
            label=data["label"],
            predictions=data["predictions"],
            mean_brier=data["mean_brier"],
            miss_rate=data["miss_rate"],
            at=data["at"],
        )


class Strategy:
    """A versioned predictor configuration with a performance record.

    Mutable only through `set_status()` and `record_performance()`, both of
    which append rather than overwrite. `version` increments whenever the
    predictor configuration itself is superseded by a descendant.
    """

    def __init__(
        self,
        description: str,
        kind: str = LAPLACE_ALL,
        params: dict[str, Any] | None = None,
        assumptions: tuple[str, ...] = (),
        applicable_conditions: tuple[str, ...] = (),
        confidence: float = 0.5,
        status: StrategyStatus = StrategyStatus.CANDIDATE,
        version: int = 1,
        id: str | None = None,
        created_at: str | None = None,
        parent_id: str | None = None,
    ) -> None:
        self.id = id or _new_id("STRAT")
        self.description = description
        self.kind = kind
        self.params = dict(params or {})
        self.assumptions = tuple(assumptions)
        self.applicable_conditions = tuple(applicable_conditions)
        self.version = version
        self.parent_id = parent_id
        self._confidence = float(confidence)
        self._status = StrategyStatus(status)
        self.created_at = created_at or _now()
        self._performance: list[PerformanceRecord] = []
        self._status_log: list[tuple[str, str, str]] = []  # (at, status, reason)

    # ---------------------------------------------------------------- state

    @property
    def status(self) -> StrategyStatus:
        return self._status

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def performance_history(self) -> tuple[PerformanceRecord, ...]:
        return tuple(self._performance)

    @property
    def status_history(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(self._status_log)

    @property
    def is_usable(self) -> bool:
        return self._status in (StrategyStatus.ACTIVE, StrategyStatus.TESTING)

    def set_status(
        self, status: StrategyStatus, reason: str, confidence: float | None = None
    ) -> None:
        """Move to a new status, keeping the reason and the previous one."""
        status = StrategyStatus(status)
        if self._status is StrategyStatus.RETIRED and status is not StrategyStatus.RETIRED:
            raise ValueError(
                f"strategy {self.id} is retired; a retired strategy is not revived, "
                "a new one is proposed"
            )
        self._status_log.append((_now(), status.value, reason))
        self._status = status
        if confidence is not None:
            self._confidence = float(confidence)

    def record_performance(self, record: PerformanceRecord) -> None:
        self._performance.append(record)

    # ------------------------------------------------------------ behaviour

    def predict(self, view) -> tuple[float, str]:
        return predict(self.kind, self.params, view)

    def descendant(
        self,
        description: str,
        kind: str | None = None,
        params: dict[str, Any] | None = None,
        assumptions: tuple[str, ...] = (),
        applicable_conditions: tuple[str, ...] = (),
        confidence: float = 0.4,
    ) -> "Strategy":
        """Propose a successor. The parent is untouched and keeps its history."""
        return Strategy(
            description=description,
            kind=kind or self.kind,
            params=params if params is not None else dict(self.params),
            assumptions=assumptions,
            applicable_conditions=applicable_conditions,
            confidence=confidence,
            status=StrategyStatus.CANDIDATE,
            version=self.version + 1,
            parent_id=self.id,
        )

    # ---------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy_id": self.id,
            "description": self.description,
            "kind": self.kind,
            "params": dict(self.params),
            "assumptions": list(self.assumptions),
            "applicable_conditions": list(self.applicable_conditions),
            "version": self.version,
            "parent_id": self.parent_id,
            "performance_history": [p.to_dict() for p in self._performance],
            "confidence": self._confidence,
            "status": self._status.value,
            "status_history": [list(entry) for entry in self._status_log],
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Strategy":
        strategy = cls(
            id=data["strategy_id"],
            description=data["description"],
            kind=data["kind"],
            params=data.get("params", {}),
            assumptions=tuple(data.get("assumptions", [])),
            applicable_conditions=tuple(data.get("applicable_conditions", [])),
            confidence=data["confidence"],
            status=StrategyStatus(data["status"]),
            version=data["version"],
            created_at=data["created_at"],
            parent_id=data.get("parent_id"),
        )
        strategy._performance = [
            PerformanceRecord.from_dict(p) for p in data.get("performance_history", [])
        ]
        strategy._status_log = [tuple(e) for e in data.get("status_history", [])]
        return strategy

    def __repr__(self) -> str:
        return (
            f"Strategy({self.id}, v{self.version}, {self.kind}, "
            f"{self.params}, {self._status.value})"
        )


def initial_strategy() -> Strategy:
    """The Challenge-3 predictor, stated honestly including what it assumes."""
    strategy = Strategy(
        description="Laplace-smoothed frequency over the entire observation history.",
        kind=LAPLACE_ALL,
        params={"alpha": 1.0},
        assumptions=(
            "The process is stationary — the underlying probability does not change.",
            "All observations are equally informative regardless of age.",
            "Observations are reliable reports of what actually happened.",
        ),
        applicable_conditions=(
            "Stationary processes.",
            "Any process where old evidence remains as relevant as new evidence.",
        ),
        confidence=0.5,
        status=StrategyStatus.ACTIVE,
        version=1,
    )
    strategy.set_status(
        StrategyStatus.ACTIVE, "initial strategy at the start of the run", confidence=0.5
    )
    return strategy
