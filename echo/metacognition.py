"""Tracking how reliable ECHO's own answers have been, per kind of question.

Two numbers that people habitually conflate:

- **belief** — how likely the proposition is, given the evidence. 0.82.
- **meta-confidence** — how often claims *like this one*, from this part of the
  system, have turned out right. 0.54.

They are different quantities with different evidence behind them. A belief is
about the world; a meta-confidence is about ECHO's track record on a class of
question. The second is measured, never asserted: it comes from a tally of
past claims and their outcomes, and with no track record it is not high or low
but `UNTESTED`, which is a third thing.

**No introspection is involved and none is simulated.** There is no method here
that returns "I feel uncertain". Everything is a count, a frequency, or a
proper scoring rule over a `DomainRecord`, and every number in a report can be
traced to the outcomes that produced it. A system that reported feelings would
be less informative than one that reports `n = 12, Brier = 0.31`, not more.

**Error prediction.** Before an outcome is known, ECHO can be asked how likely
it is to be wrong. That is a prediction like any other, and it is scored like
any other — the point is whether ECHO's estimate of its own failure rate tracks
its actual failure rate, which is a measurable thing and frequently false.

**Abstention.** Saying "I don't know" is an available action, taken when
meta-confidence for the domain falls below a threshold fixed in advance. It is
not free: the report measures accuracy when answering, accuracy on the
questions it declined, how often it was confidently wrong, and how often it
abstained on questions it would have got right. A system that abstains on
everything scores perfectly on the first metric and terribly on the last.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# ------------------------------------------------------------------- settings
#
# Pre-registered. Fixed before the first metacognition run.

#: Below this many scored claims, a domain is UNTESTED — not weak, not strong.
#: Untested is a distinct state and collapsing it into "low confidence" would
#: throw away the difference between "known to be bad" and "no idea".
MIN_FOR_ASSESSMENT = 8

#: Brier thresholds separating the competence bands. 0.25 is the score of
#: always saying 0.5, so anything at or above it is no better than shrugging.
STRONG_BRIER = 0.15
WEAK_BRIER = 0.25

#: Abstain when meta-confidence for the domain falls below this.
ABSTENTION_THRESHOLD = 0.55

#: A claim counts as high-confidence at or above this.
HIGH_CONFIDENCE = 0.80


class Competence(str, Enum):
    STRONG = "STRONG"
    UNCERTAIN = "UNCERTAIN"
    WEAK = "WEAK"
    UNTESTED = "UNTESTED"


#: The parts of the system whose reliability is tracked separately. Reliability
#: is not one number: being good at prediction says nothing about being good at
#: causal inference, and averaging them would hide both.
DOMAINS = (
    "prediction",
    "discovery",
    "transfer",
    "experimentation",
    "causal_inference",
)


@dataclass(frozen=True)
class Claim:
    """One thing ECHO said, with what it thought at the time. Immutable."""

    claim_id: str
    domain: str
    proposition: str
    belief: float
    #: What ECHO expected its own chance of being wrong to be, stated *before*
    #: the outcome was known.
    predicted_error: float
    created_at: int
    #: Whether the proposition turned out to be TRUE. None means still open.
    #:
    #: Deliberately the truth of the proposition and not "was ECHO right".
    #: Those are different quantities and scoring a belief against the wrong
    #: one is a real bug that this field exists to prevent — see the report.
    outcome: bool | None = None
    abstained: bool = False
    meta_confidence_at_claim: float | None = None
    note: str = ""

    def resolved(self, outcome: bool) -> "Claim":
        """A new claim carrying the outcome. The original is untouched."""
        if self.outcome is not None:
            raise ValueError(f"{self.claim_id} already has an outcome")
        return replace(self, outcome=outcome)

    @property
    def is_open(self) -> bool:
        return self.outcome is None

    @property
    def correct(self) -> bool | None:
        """Did ECHO land on the right side of 0.5?"""
        if self.outcome is None:
            return None
        return (self.belief > 0.5) == self.outcome

    @property
    def brier(self) -> float | None:
        """Squared error of the belief against what actually happened.

        Scored against the proposition's truth, not against whether ECHO was
        right. A belief of 0.12 on a proposition that turned out false is an
        excellent call; scoring it against `correct = True` would record it as
        one of the worst possible, which is exactly backwards.
        """
        if self.outcome is None:
            return None
        return (self.belief - (1.0 if self.outcome else 0.0)) ** 2

    @property
    def was_confidently_wrong(self) -> bool:
        return (
            self.correct is False
            and not self.abstained
            and max(self.belief, 1.0 - self.belief) >= HIGH_CONFIDENCE
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "domain": self.domain,
            "proposition": self.proposition,
            "belief": self.belief,
            "predicted_error": self.predicted_error,
            "created_at": self.created_at,
            "outcome": self.outcome,
            "abstained": self.abstained,
            "meta_confidence_at_claim": self.meta_confidence_at_claim,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Claim":
        return cls(
            claim_id=str(payload["claim_id"]),
            domain=str(payload["domain"]),
            proposition=str(payload["proposition"]),
            belief=float(payload["belief"]),
            predicted_error=float(payload["predicted_error"]),
            created_at=int(payload["created_at"]),
            outcome=payload.get("outcome"),
            abstained=bool(payload.get("abstained", False)),
            meta_confidence_at_claim=payload.get("meta_confidence_at_claim"),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True)
class DomainRecord:
    """The tally for one domain. Every field is a count or a mean of counts."""

    domain: str
    answered: int
    correct: int
    abstained: int
    abstained_would_have_been_correct: int
    brier_sum: float
    predicted_error_sum: float
    confidently_wrong: int

    @property
    def accuracy(self) -> float | None:
        return self.correct / self.answered if self.answered else None

    @property
    def brier(self) -> float | None:
        return self.brier_sum / self.answered if self.answered else None

    @property
    def observed_error_rate(self) -> float | None:
        return 1.0 - self.accuracy if self.accuracy is not None else None

    @property
    def predicted_error_rate(self) -> float | None:
        return self.predicted_error_sum / self.answered if self.answered else None

    @property
    def error_prediction_gap(self) -> float | None:
        """Predicted minus observed. Positive means ECHO over-warned."""
        if self.predicted_error_rate is None or self.observed_error_rate is None:
            return None
        return self.predicted_error_rate - self.observed_error_rate

    @property
    def unnecessary_abstentions(self) -> int:
        return self.abstained_would_have_been_correct

    @property
    def meta_confidence(self) -> float:
        """How much to trust this domain, from its record alone.

        Laplace-smoothed accuracy, so one lucky answer is not a track record
        and an empty record sits at 0.5 rather than at either extreme.
        """
        return (self.correct + 1) / (self.answered + 2)

    @property
    def competence(self) -> Competence:
        if self.answered < MIN_FOR_ASSESSMENT:
            return Competence.UNTESTED
        brier = self.brier
        assert brier is not None
        if brier <= STRONG_BRIER:
            return Competence.STRONG
        if brier >= WEAK_BRIER:
            return Competence.WEAK
        return Competence.UNCERTAIN

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "answered": self.answered,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "abstained": self.abstained,
            "unnecessary_abstentions": self.unnecessary_abstentions,
            "brier": self.brier,
            "meta_confidence": self.meta_confidence,
            "competence": self.competence.value,
            "predicted_error_rate": self.predicted_error_rate,
            "observed_error_rate": self.observed_error_rate,
            "error_prediction_gap": self.error_prediction_gap,
            "confidently_wrong": self.confidently_wrong,
        }


EMPTY = {
    "answered": 0,
    "correct": 0,
    "abstained": 0,
    "abstained_would_have_been_correct": 0,
    "brier_sum": 0.0,
    "predicted_error_sum": 0.0,
    "confidently_wrong": 0,
}


class Metacognition:
    """The self-assessment layer. Counts in, bands out; nothing in between.

    Deliberately not a model of ECHO's "state of mind". It is a scoreboard,
    and the only thing it can tell you is what happened last time questions of
    this kind were asked.
    """

    SCHEMA_VERSION = 1
    DEFAULT_FILENAME = "metacognition.json"

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._claims: list[Claim] = []

    # ---------------------------------------------------------------- write

    def record(self, claim: Claim) -> Claim:
        stamped = (
            claim
            if claim.meta_confidence_at_claim is not None
            else replace(
                claim, meta_confidence_at_claim=self.meta_confidence(claim.domain)
            )
        )
        self._claims.append(stamped)
        return stamped

    def resolve(self, claim_id: str, outcome: bool) -> Claim:
        """Attach an outcome. Replaces the entry rather than mutating it."""
        for index, claim in enumerate(self._claims):
            if claim.claim_id == claim_id:
                resolved = claim.resolved(outcome)
                self._claims[index] = resolved
                return resolved
        raise KeyError(f"no claim {claim_id!r}")

    # ----------------------------------------------------------------- read

    def claims(self, domain: str | None = None) -> tuple[Claim, ...]:
        if domain is None:
            return tuple(self._claims)
        return tuple(c for c in self._claims if c.domain == domain)

    def tally(self, domain: str) -> DomainRecord:
        counts = dict(EMPTY)
        for claim in self._claims:
            if claim.domain != domain or claim.correct is None:
                continue
            if claim.abstained:
                counts["abstained"] += 1
                if claim.correct:
                    counts["abstained_would_have_been_correct"] += 1
                continue
            counts["answered"] += 1
            counts["correct"] += int(claim.correct)
            brier = claim.brier
            counts["brier_sum"] += brier if brier is not None else 0.0
            counts["predicted_error_sum"] += claim.predicted_error
            counts["confidently_wrong"] += int(claim.was_confidently_wrong)
        return DomainRecord(domain=domain, **counts)  # type: ignore[arg-type]

    def meta_confidence(self, domain: str) -> float:
        return self.tally(domain).meta_confidence

    def competence(self, domain: str) -> Competence:
        return self.tally(domain).competence

    def report(self) -> dict[str, DomainRecord]:
        domains = sorted({c.domain for c in self._claims} | set(DOMAINS))
        return {domain: self.tally(domain) for domain in domains}

    # ------------------------------------------------------------ decisions

    def should_abstain(
        self, domain: str, *, threshold: float = ABSTENTION_THRESHOLD
    ) -> bool:
        """Decline when the track record for this kind of question is poor.

        An untested domain is *not* an automatic abstention: refusing to answer
        anything ECHO has not already been scored on would make the track
        record unfillable. It answers, and the outcome becomes evidence.
        """
        record = self.tally(domain)
        if record.competence is Competence.UNTESTED:
            return False
        return record.meta_confidence < threshold

    def expected_error(self, domain: str) -> float:
        """ECHO's estimate of how likely it is to be wrong here, from its record."""
        record = self.tally(domain)
        if record.competence is Competence.UNTESTED:
            return 0.5  # no basis for a better guess, and saying so is the answer
        return 1.0 - record.meta_confidence

    def rank_by_expected_success(self, domains: Sequence[str]) -> list[tuple[str, float]]:
        """Which of these is ECHO more likely to get right? Ordered, with numbers.

        This is the whole self-diagnostic: no labels about which problem is
        "easy" are involved, only each domain's own measured record.
        """
        scored = [(domain, 1.0 - self.expected_error(domain)) for domain in domains]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    # --------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "claims": [c.to_dict() for c in self._claims],
        }

    def save(self, path: Path | str | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("no path to save to")
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "Metacognition":
        layer = cls(path)
        if not layer.path or not layer.path.is_file():
            return layer
        with layer.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(
                f"unsupported metacognition schema_version {payload.get('schema_version')!r}"
            )
        layer._claims = [Claim.from_dict(c) for c in payload.get("claims", [])]
        return layer

    def __len__(self) -> int:
        # Never use in a boolean context; an empty layer is falsy.
        return len(self._claims)


# ------------------------------------------------------- calibration scoring


def calibration_of_error_prediction(
    claims: Iterable[Claim], buckets: int = 5
) -> list[dict[str, Any]]:
    """Is ECHO right about how often it is wrong?

    Buckets claims by the error rate ECHO predicted, and compares each bucket's
    predicted rate against the rate actually observed in it. A well-calibrated
    self-assessment has the two columns matching; the usual failure is a system
    that predicts a 10% error rate and delivers 40%.
    """
    scored = [c for c in claims if c.correct is not None and not c.abstained]
    rows: list[dict[str, Any]] = []
    for index in range(buckets):
        low = index / buckets
        high = (index + 1) / buckets
        members = [
            c
            for c in scored
            if (low <= c.predicted_error < high)
            or (index == buckets - 1 and c.predicted_error == 1.0)
        ]
        if not members:
            continue
        predicted = sum(c.predicted_error for c in members) / len(members)
        observed = sum(1 for c in members if not c.correct) / len(members)
        rows.append(
            {
                "range": f"{low:.1f}–{high:.1f}",
                "count": len(members),
                "mean_predicted_error": predicted,
                "observed_error_rate": observed,
                "gap": predicted - observed,
            }
        )
    return rows


def expected_calibration_error(rows: Sequence[Mapping[str, Any]]) -> float | None:
    """Weighted mean absolute gap between predicted and observed error rates."""
    total = sum(int(row["count"]) for row in rows)
    if total == 0:
        return None
    return sum(int(row["count"]) * abs(float(row["gap"])) for row in rows) / total
