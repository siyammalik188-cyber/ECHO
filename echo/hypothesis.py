"""The hypothesis record: what was proposed, what it scored, and what became of it.

A `Hypothesis` is frozen. Nothing about one is ever edited in place — not its
expression, not its scores, not its status. A status change or a newly measured
score produces a **new** `Hypothesis` carrying the same `hypothesis_id` and the
next `revision` number, and the ledger keeps every revision. So the question
"what did this look like before it was rejected?" always has an answer, and
"never silently modify a historical hypothesis" is a property of the type rather
than a rule someone has to remember.

`parent_hypothesis_id` is set when a candidate was derived from another one
rather than enumerated from scratch, so the shape of the search is recoverable
afterwards.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Mapping

from . import expressions as ex
from .expressions import Expr


class HypothesisStatus(str, Enum):
    CANDIDATE = "candidate"  # proposed and screened, not yet fitted
    TESTING = "testing"  # fitted on TRAIN, ranked on VAL_A
    VALIDATED = "validated"  # survived the confirmation block VAL_B
    ACTIVE = "active"  # promoted; the decision was made before TEST was read
    REJECTED = "rejected"  # failed a gate
    RETIRED = "retired"  # was promoted, then did not survive unseen data


#: Transitions the ledger will accept. A rejected hypothesis stays rejected: it
#: remains in history as a record of what was tried, and cannot be quietly
#: resurrected once the data has spoken.
_ALLOWED: dict[HypothesisStatus, tuple[HypothesisStatus, ...]] = {
    HypothesisStatus.CANDIDATE: (HypothesisStatus.TESTING, HypothesisStatus.REJECTED),
    HypothesisStatus.TESTING: (HypothesisStatus.VALIDATED, HypothesisStatus.REJECTED),
    HypothesisStatus.VALIDATED: (HypothesisStatus.ACTIVE, HypothesisStatus.REJECTED),
    HypothesisStatus.ACTIVE: (HypothesisStatus.RETIRED,),
    HypothesisStatus.REJECTED: (),
    HypothesisStatus.RETIRED: (),
}


def hypothesis_id(expr: Expr) -> str:
    """Stable and content-derived: the same expression always gets the same id.

    Deliberately not a counter and not a UUID. Two runs of the same experiment
    must produce the same identifiers, or "reproducible" would only mean the
    numbers matched.
    """
    digest = hashlib.sha256(expr.text().encode("utf-8")).hexdigest()[:12]
    return f"HYP-{digest}"


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    expression: Expr
    variables_used: tuple[str, ...]
    complexity: int
    created_at: int  # logical tick, not wall clock — reproducibility again
    evidence_window: tuple[int, int]  # the rows this hypothesis was fitted on
    discovery_reason: str
    status: HypothesisStatus = HypothesisStatus.CANDIDATE
    training_score: dict[str, Any] | None = None
    validation_score: dict[str, Any] | None = None
    test_score: dict[str, Any] | None = None
    parent_hypothesis_id: str | None = None
    revision: int = 0
    status_reason: str = "proposed"
    screening_statistic: float | None = None
    penalised_score: float | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    # ------------------------------------------------------------ constructor

    @classmethod
    def propose(
        cls,
        expr: Expr,
        *,
        created_at: int,
        evidence_window: tuple[int, int],
        discovery_reason: str,
        parent_hypothesis_id: str | None = None,
        screening_statistic: float | None = None,
    ) -> "Hypothesis":
        ex.validate(expr)
        return cls(
            hypothesis_id=hypothesis_id(expr),
            expression=expr,
            variables_used=expr.variables(),
            complexity=expr.complexity(),
            created_at=created_at,
            evidence_window=evidence_window,
            discovery_reason=discovery_reason,
            parent_hypothesis_id=parent_hypothesis_id,
            screening_statistic=screening_statistic,
        )

    # -------------------------------------------------------------- revisions

    def _next(self, **changes: Any) -> "Hypothesis":
        return replace(self, revision=self.revision + 1, **changes)

    def with_scores(
        self,
        *,
        training: dict[str, Any] | None = None,
        validation: dict[str, Any] | None = None,
        test: dict[str, Any] | None = None,
        penalised: float | None = None,
        reason: str | None = None,
    ) -> "Hypothesis":
        """A new revision carrying measured scores. The old one is untouched."""
        return self._next(
            training_score=self.training_score if training is None else training,
            validation_score=self.validation_score if validation is None else validation,
            test_score=self.test_score if test is None else test,
            penalised_score=self.penalised_score if penalised is None else penalised,
            status_reason=self.status_reason if reason is None else reason,
        )

    def with_status(self, status: HypothesisStatus, reason: str) -> "Hypothesis":
        if status not in _ALLOWED[self.status]:
            raise ValueError(
                f"{self.hypothesis_id} cannot move from {self.status.value} to "
                f"{status.value}"
            )
        return self._next(status=status, status_reason=reason)

    def with_note(self, note: str) -> "Hypothesis":
        return self._next(notes=self.notes + (note,))

    # ---------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "expression": self.expression.to_dict(),
            "expression_text": self.expression.text(),
            "variables_used": list(self.variables_used),
            "complexity": self.complexity,
            "created_at": self.created_at,
            "evidence_window": list(self.evidence_window),
            "training_score": self.training_score,
            "validation_score": self.validation_score,
            "test_score": self.test_score,
            "status": self.status.value,
            "status_reason": self.status_reason,
            "parent_hypothesis_id": self.parent_hypothesis_id,
            "discovery_reason": self.discovery_reason,
            "revision": self.revision,
            "screening_statistic": self.screening_statistic,
            "penalised_score": self.penalised_score,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Hypothesis":
        expr = ex.from_dict(payload["expression"])
        return cls(
            hypothesis_id=str(payload["hypothesis_id"]),
            expression=expr,
            variables_used=tuple(payload["variables_used"]),
            complexity=int(payload["complexity"]),
            created_at=int(payload["created_at"]),
            evidence_window=tuple(payload["evidence_window"]),  # type: ignore[arg-type]
            discovery_reason=str(payload["discovery_reason"]),
            status=HypothesisStatus(payload["status"]),
            training_score=payload.get("training_score"),
            validation_score=payload.get("validation_score"),
            test_score=payload.get("test_score"),
            parent_hypothesis_id=payload.get("parent_hypothesis_id"),
            revision=int(payload.get("revision", 0)),
            status_reason=str(payload.get("status_reason", "")),
            screening_statistic=payload.get("screening_statistic"),
            penalised_score=payload.get("penalised_score"),
            notes=tuple(payload.get("notes", ())),
        )
