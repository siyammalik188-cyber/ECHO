"""Uncertain beliefs, the evidence that bears on them, and revision under evidence.

This is a **separate** structure from memory. Memories (`memory.py`) remain what
they were: immutable statements, never revised, never reconciled. A belief is a
different kind of thing — a proposition ECHO is not sure about, which can move
when evidence arrives. Keeping them apart is deliberate; conflating "what I was
told" with "what I currently think is true" is how a memory store quietly turns
into a rumour mill.

**BELIEF REVISION IS NOT LEARNING.** Nothing here generalises, transfers to new
propositions, or changes how ECHO behaves next time. A belief moves when
evidence about *that specific proposition* arrives, by a fixed rule that is the
same on the first revision and the thousandth. That is all.

## How a belief moves

Confidence is updated in log-odds, the standard way to accumulate independent
evidence:

    logit(c) += ±scale × reliability × relevance

`reliability × relevance` is the evidence's **weight**. A tabloid rumour about
the right topic (low reliability, high relevance) and a forensic report about
the wrong topic (high reliability, low relevance) both move the belief very
little; only evidence that is both trustworthy and pertinent moves it much.

Two consequences that matter, and that the tests pin down:

- Weak contradictory evidence does **not** meaningfully dent a strong belief.
  Nothing here mechanically lowers confidence just because something disagreed.
- Evidence below `MIN_RELEVANCE` is recorded and explicitly **not applied**. The
  belief is unchanged and the decision is written into the revision history with
  its reason.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# Log-odds shift per unit of evidence weight. Fixed for the life of the system:
# it is a constant of the update rule, not something ECHO tunes. If this ever
# becomes adaptive, the system stops being "belief revision" and starts being
# something that needs a far more careful name.
EVIDENCE_SCALE = 1.0

# Evidence less pertinent than this is recorded but does not move the belief.
MIN_RELEVANCE = 0.2

# Confidence is clamped away from 0 and 1 so log-odds stays finite: a belief can
# become very unlikely, never impossible, and no amount of evidence makes it
# unfalsifiable.
MIN_CONFIDENCE = 0.01
MAX_CONFIDENCE = 0.99


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _check_unit(name: str, value: float) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be within 0.0–1.0, got {value!r}")
    return value


def logit(p: float) -> float:
    p = min(max(p, MIN_CONFIDENCE), MAX_CONFIDENCE)
    return math.log(p / (1.0 - p))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class EvidenceStance(str, Enum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"


@dataclass(frozen=True)
class Evidence:
    """One observation bearing on a proposition.

    Frozen: evidence is a record of what was observed. Revising the record
    rather than the belief would make the audit trail meaningless.

    `stance` is supplied, not inferred. ECHO is told whether an observation
    points for or against the proposition; what it decides is how much that
    should matter. See the limitations section of the ECHO 2 report — inferring
    stance from raw natural language is a different and much harder problem.
    """

    description: str
    source: str
    reliability: float  # how much the source can be trusted, 0–1
    relevance: float  # how much it bears on this proposition, 0–1
    stance: EvidenceStance
    id: str = field(default_factory=lambda: _new_id("EV"))
    recorded_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.description.strip():
            raise ValueError("evidence description must not be empty")
        if not self.source.strip():
            raise ValueError("evidence source must not be empty")
        object.__setattr__(self, "reliability", _check_unit("reliability", self.reliability))
        object.__setattr__(self, "relevance", _check_unit("relevance", self.relevance))
        object.__setattr__(self, "stance", EvidenceStance(self.stance))

    @property
    def weight(self) -> float:
        """How much this evidence can move a belief: reliability × relevance."""
        return self.reliability * self.relevance

    @property
    def supports(self) -> bool:
        return self.stance is EvidenceStance.SUPPORTS

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "source": self.source,
            "reliability": self.reliability,
            "relevance": self.relevance,
            "stance": self.stance.value,
            "recorded_at": self.recorded_at,
            "weight": round(self.weight, 6),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evidence":
        return cls(
            id=data["id"],
            description=data["description"],
            source=data["source"],
            reliability=data["reliability"],
            relevance=data["relevance"],
            stance=EvidenceStance(data["stance"]),
            recorded_at=data["recorded_at"],
        )


@dataclass(frozen=True)
class Revision:
    """One considered piece of evidence, and what it did to the belief.

    Written whether or not the belief moved: a decision *not* to revise is as
    much a part of the record as a decision to revise, and hiding it would make
    "the belief did not change" indistinguishable from "the evidence was never
    seen". Frozen, so history cannot be edited after the fact.
    """

    evidence_id: str
    previous_confidence: float
    new_confidence: float
    applied: bool
    reason: str
    evidence_snapshot: dict[str, Any]
    id: str = field(default_factory=lambda: _new_id("REV"))
    at: str = field(default_factory=_now)

    @property
    def delta(self) -> float:
        return self.new_confidence - self.previous_confidence

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "at": self.at,
            "evidence_id": self.evidence_id,
            "previous_confidence": self.previous_confidence,
            "new_confidence": self.new_confidence,
            "delta": round(self.delta, 6),
            "applied": self.applied,
            "reason": self.reason,
            "evidence_snapshot": self.evidence_snapshot,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Revision":
        return cls(
            id=data["id"],
            at=data["at"],
            evidence_id=data["evidence_id"],
            previous_confidence=data["previous_confidence"],
            new_confidence=data["new_confidence"],
            applied=data["applied"],
            reason=data["reason"],
            evidence_snapshot=data.get("evidence_snapshot", {}),
        )


class Belief:
    """A proposition held with a confidence that evidence can move.

    Confidence and history are exposed read-only. The only way to change a
    belief is `consider()`, which always writes a revision entry first. There is
    no setter, no `clear_history()`, and no path that changes confidence without
    recording why.
    """

    def __init__(
        self,
        proposition: str,
        confidence: float = 0.5,
        id: str | None = None,
        created_at: str | None = None,
        updated_at: str | None = None,
    ) -> None:
        if not proposition.strip():
            raise ValueError("proposition must not be empty")
        self.id = id or _new_id("BEL")
        self.proposition = proposition
        self._confidence = _check_unit("confidence", confidence)
        self.created_at = created_at or _now()
        self.updated_at = updated_at or self.created_at
        self._supporting: list[str] = []
        self._contradicting: list[str] = []
        self._history: list[Revision] = []

    # ------------------------------------------------------------- read-only

    @property
    def confidence(self) -> float:
        """Current confidence. Read-only — use `consider()`."""
        return self._confidence

    @property
    def supporting_evidence(self) -> tuple[str, ...]:
        """Ids of every evidence item considered that argued *for* the proposition.

        Includes items that were recorded but not applied; `revision_history`
        says which actually moved the belief.
        """
        return tuple(self._supporting)

    @property
    def contradicting_evidence(self) -> tuple[str, ...]:
        """Ids of every evidence item considered that argued *against* it."""
        return tuple(self._contradicting)

    @property
    def revision_history(self) -> tuple[Revision, ...]:
        """The full record, oldest first. A copy — mutating it changes nothing."""
        return tuple(self._history)

    @property
    def revision_count(self) -> int:
        """Revisions that actually moved the belief (not bare considerations)."""
        return sum(1 for r in self._history if r.applied)

    @property
    def considered_count(self) -> int:
        return len(self._history)

    @property
    def initial_confidence(self) -> float:
        """What this belief was held at before any evidence was considered."""
        return self._history[0].previous_confidence if self._history else self._confidence

    def has_considered(self, evidence_id: str) -> bool:
        return any(r.evidence_id == evidence_id for r in self._history)

    # -------------------------------------------------------------- revision

    def consider(
        self,
        evidence: Evidence,
        scale: float = EVIDENCE_SCALE,
        min_relevance: float = MIN_RELEVANCE,
    ) -> Revision:
        """Weigh `evidence` against this belief and revise if it warrants it.

        Always returns a `Revision`, appended to the history — including when
        the decision is to leave the belief alone. Raises if the same evidence
        is considered twice, which would double-count it.
        """
        if self.has_considered(evidence.id):
            raise ValueError(
                f"evidence {evidence.id} has already been considered for belief {self.id}"
            )

        previous = self._confidence

        if evidence.relevance < min_relevance:
            revision = Revision(
                evidence_id=evidence.id,
                previous_confidence=previous,
                new_confidence=previous,
                applied=False,
                reason=(
                    f"Not applied. {evidence.id} ({evidence.source}) has relevance "
                    f"{evidence.relevance:.2f}, below the {min_relevance:.2f} threshold, "
                    f"so it does not bear on this proposition. Confidence unchanged "
                    f"at {previous:.3f}."
                ),
                evidence_snapshot=evidence.to_dict(),
            )
        else:
            direction = 1.0 if evidence.supports else -1.0
            shift = direction * scale * evidence.weight
            updated = sigmoid(logit(previous) + shift)
            updated = min(max(updated, MIN_CONFIDENCE), MAX_CONFIDENCE)
            self._confidence = updated
            revision = Revision(
                evidence_id=evidence.id,
                previous_confidence=previous,
                new_confidence=updated,
                applied=True,
                reason=self._explain(evidence, previous, updated, shift),
                evidence_snapshot=evidence.to_dict(),
            )

        (self._supporting if evidence.supports else self._contradicting).append(evidence.id)
        self._history.append(revision)
        self.updated_at = revision.at
        return revision

    @staticmethod
    def _explain(
        evidence: Evidence, previous: float, updated: float, shift: float
    ) -> str:
        """Say, in plain terms, why this evidence moved the belief as far as it did."""
        delta = updated - previous
        weight = evidence.weight

        if weight >= 0.6:
            strength = "heavily, since it is both reliable and highly pertinent"
        elif weight >= 0.3:
            strength = "moderately"
        elif evidence.reliability < 0.35:
            strength = "barely, because the source is weak"
        elif evidence.relevance < 0.35:
            strength = "barely, because it only glances at the proposition"
        else:
            strength = "barely"

        return (
            f"Applied {evidence.id} ({evidence.source}), which "
            f"{evidence.stance.value} the proposition. Weighted {strength}: "
            f"reliability {evidence.reliability:.2f} × relevance "
            f"{evidence.relevance:.2f} = {weight:.2f}, giving a log-odds shift of "
            f"{shift:+.3f}. Confidence {previous:.3f} → {updated:.3f} "
            f"({delta:+.3f})."
        )

    # ----------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "belief_id": self.id,
            "proposition": self.proposition,
            "confidence": self._confidence,
            "supporting_evidence": list(self._supporting),
            "contradicting_evidence": list(self._contradicting),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "revision_count": self.revision_count,
            "revision_history": [r.to_dict() for r in self._history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Belief":
        belief = cls(
            proposition=data["proposition"],
            confidence=data["confidence"],
            id=data["belief_id"],
            created_at=data["created_at"],
            updated_at=data["updated_at"],
        )
        belief._supporting = list(data.get("supporting_evidence", []))
        belief._contradicting = list(data.get("contradicting_evidence", []))
        belief._history = [Revision.from_dict(r) for r in data.get("revision_history", [])]
        return belief

    def __repr__(self) -> str:
        return (
            f"Belief({self.id}, {self.proposition!r}, confidence={self._confidence:.3f}, "
            f"revisions={self.revision_count})"
        )
