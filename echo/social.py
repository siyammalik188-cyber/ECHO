"""Learning from other agents without believing them.

Other systems say things. Some are usually right, some are usually wrong, and
one of them here is wrong on purpose. ECHO is told none of that. It has to work
out who is worth listening to from what happens next, and it has to do so while
still being able to side with a lone dissenter against everyone else.

**Claims are parsed, not swallowed.** `parse_claim` reads a sentence like

    "I think hypothesis 2 is more likely because observation X occurred"

and extracts four things — what is being claimed, what is offered as evidence,
how strongly it is held, and who said it. Extraction is deliberately shallow
pattern-matching over a closed vocabulary: there is no model in the loop, and a
claim that cannot be parsed is refused rather than guessed at. Nothing about
being parsed makes a claim true.

**Reliability is earned and revisable.** Each source carries a Laplace-smoothed
record of how often its claims turned out right. It starts at 0.5 — neither
trusted nor distrusted — and moves only on resolved outcomes. Because it is a
running tally rather than a fixed parameter, a source that becomes unreliable
is tracked down again, which is the case ECHO 11 leans on.

**Evidence outranks headcount.** The aggregation is a log-odds sum in which
each claim's weight comes from its source's reliability and its own stated
strength, and a source below 0.5 contributes *negative* weight — believing the
opposite of an anti-reliable source is the correct response, not ignoring it.
Three agreeing mediocre sources can be outweighed by one good one, which is
what stops this from being a vote.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# ------------------------------------------------------------------- settings
#
# Pre-registered, fixed before the first social run.

#: A source with no record sits here: worth neither trusting nor discounting.
PRIOR_RELIABILITY = 0.5

#: Strength of the Laplace prior on a source's record. Higher means slower to
#: trust and slower to condemn.
RELIABILITY_PRIOR_WEIGHT = 2.0

#: Reliabilities are clamped away from 0 and 1 so no single source can ever
#: become infinitely persuasive, however long its streak.
MIN_RELIABILITY = 0.02
MAX_RELIABILITY = 0.98

#: Confidence words a claim may use, and what each is taken to mean.
CONFIDENCE_WORDS: dict[str, float] = {
    "certain": 0.95,
    "certainly": 0.95,
    "sure": 0.90,
    "confident": 0.85,
    "strongly": 0.85,
    "likely": 0.75,
    "probably": 0.75,
    "think": 0.65,
    "believe": 0.65,
    "suspect": 0.55,
    "guess": 0.55,
    "unsure": 0.52,
    "maybe": 0.52,
    "perhaps": 0.52,
}

DEFAULT_CONFIDENCE = 0.65


class SocialError(ValueError):
    pass


# ------------------------------------------------------------- parsed claims


@dataclass(frozen=True)
class Claim:
    """What one agent said, pulled apart. Being parsed is not being believed."""

    source: str
    proposition: str
    #: True if the source asserts the proposition, False if it denies it.
    asserts: bool
    confidence: float
    evidence: tuple[str, ...]
    raw: str
    created_at: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "proposition": self.proposition,
            "asserts": self.asserts,
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "raw": self.raw,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Claim":
        return cls(
            source=str(payload["source"]),
            proposition=str(payload["proposition"]),
            asserts=bool(payload["asserts"]),
            confidence=float(payload["confidence"]),
            evidence=tuple(payload.get("evidence", ())),
            raw=str(payload.get("raw", "")),
            created_at=int(payload.get("created_at", 0)),
        )


#: Recognised ways of naming a proposition. Closed by design: an utterance that
#: does not match is refused, not guessed at.
_PROPOSITION = re.compile(
    # The separator may be a space or a hyphen: sources say both "hypothesis 2"
    # and "hypothesis-2", and refusing one of those would reject half the
    # vocabulary for a reason that has nothing to do with meaning.
    r"\b(hypothesis|model|option|structure|pattern)[\s\-_]+([A-Za-z0-9_]+)", re.I
)
_EVIDENCE = re.compile(r"\bbecause\s+(.+?)(?:[.;]|$)", re.I)
_NEGATION = re.compile(
    r"\b(not|isn'?t|is not|unlikely|wrong|false|rule out|ruled out|against)\b", re.I
)


def parse_claim(source: str, utterance: str, *, created_at: int = 0) -> Claim:
    """Pull `claim / evidence / confidence / source` out of a sentence.

    Shallow and closed on purpose. No model is consulted, so nothing here can
    be talked into an interpretation; an utterance outside the vocabulary
    raises rather than producing a confident misreading.
    """
    if not utterance or not utterance.strip():
        raise SocialError("an empty utterance carries no claim")

    match = _PROPOSITION.search(utterance)
    if match is None:
        raise SocialError(f"no recognisable proposition in {utterance!r}")
    proposition = f"{match.group(1).lower()}-{match.group(2).lower()}"

    evidence_match = _EVIDENCE.search(utterance)
    evidence = (
        tuple(part.strip() for part in evidence_match.group(1).split(" and "))
        if evidence_match
        else ()
    )

    # The confidence word nearest the start wins; ties go to the stronger.
    lowered = utterance.lower()
    best: tuple[int, float] | None = None
    for word, value in CONFIDENCE_WORDS.items():
        position = lowered.find(word)
        if position >= 0 and (best is None or position < best[0]):
            best = (position, value)
    confidence = best[1] if best else DEFAULT_CONFIDENCE

    # Only negation *before* the "because" counts; the evidence clause is not
    # the claim, and "because X did not occur" must not flip the assertion.
    head = lowered.split("because", 1)[0]
    asserts = _NEGATION.search(head) is None

    return Claim(
        source=source,
        proposition=proposition,
        asserts=asserts,
        confidence=confidence,
        evidence=evidence,
        raw=utterance.strip(),
        created_at=created_at,
    )


# ------------------------------------------------------------- reliability


@dataclass(frozen=True)
class SourceRecord:
    """A running, revisable account of how often one source has been right."""

    source: str
    resolved: int = 0
    correct: int = 0
    #: Every resolved claim, oldest first, as (tick, was_right).
    history: tuple[tuple[int, bool], ...] = ()

    @property
    def reliability(self) -> float:
        """Laplace-smoothed accuracy, clamped away from certainty."""
        value = (self.correct + PRIOR_RELIABILITY * RELIABILITY_PRIOR_WEIGHT) / (
            self.resolved + RELIABILITY_PRIOR_WEIGHT
        )
        return min(MAX_RELIABILITY, max(MIN_RELIABILITY, value))

    def recent_reliability(self, window: int = 20) -> float | None:
        """Accuracy over the last `window` resolutions, for spotting a turn."""
        if len(self.history) < window:
            return None
        recent = self.history[-window:]
        return sum(1 for _, right in recent if right) / len(recent)

    def with_outcome(self, tick: int, was_right: bool) -> "SourceRecord":
        return replace(
            self,
            resolved=self.resolved + 1,
            correct=self.correct + int(was_right),
            history=self.history + ((tick, was_right),),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "resolved": self.resolved,
            "correct": self.correct,
            "reliability": self.reliability,
            "history": [list(h) for h in self.history],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SourceRecord":
        return cls(
            source=str(payload["source"]),
            resolved=int(payload["resolved"]),
            correct=int(payload["correct"]),
            history=tuple((int(t), bool(r)) for t, r in payload.get("history", [])),
        )


# ------------------------------------------------------------- aggregation


def _logit(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass(frozen=True)
class Contribution:
    """One claim's push on one proposition, and where the push came from."""

    source: str
    asserts: bool
    confidence: float
    reliability: float
    weight: float  # signed log-odds contribution

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "asserts": self.asserts,
            "confidence": self.confidence,
            "reliability": self.reliability,
            "weight": self.weight,
        }


@dataclass(frozen=True)
class Aggregate:
    """What testimony alone implies about a proposition, and why."""

    proposition: str
    probability: float
    prior: float
    contributions: tuple[Contribution, ...]
    #: How many sources asserted vs denied, kept so the report can show that
    #: the answer is not the headcount.
    asserting: int
    denying: int

    @property
    def majority_asserts(self) -> bool:
        return self.asserting > self.denying

    @property
    def followed_majority(self) -> bool:
        return (self.probability > 0.5) == self.majority_asserts

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposition": self.proposition,
            "probability": self.probability,
            "prior": self.prior,
            "asserting": self.asserting,
            "denying": self.denying,
            "followed_majority": self.followed_majority,
            "contributions": [c.to_dict() for c in self.contributions],
        }


def aggregate(
    claims: Sequence[Claim],
    reliabilities: Mapping[str, float],
    *,
    prior: float = 0.5,
    proposition: str | None = None,
) -> Aggregate:
    """Combine testimony in log-odds, weighted by who said it and how firmly.

    A source at reliability 0.5 contributes exactly nothing: hearing from
    someone whose record is a coin flip should not move anything. A source
    *below* 0.5 contributes negative weight, so a reliably wrong agent is
    informative in reverse — which is the correct treatment of the deceptive
    agent and is not the same as ignoring it.
    """
    if proposition is None:
        if not claims:
            raise SocialError("no claims and no proposition to aggregate")
        proposition = claims[0].proposition

    relevant = [c for c in claims if c.proposition == proposition]
    total = _logit(prior)
    contributions: list[Contribution] = []
    asserting = denying = 0

    for claim in relevant:
        reliability = reliabilities.get(claim.source, PRIOR_RELIABILITY)
        # How much this source's testimony is worth, in log-odds. Zero at 0.5,
        # negative below it.
        source_weight = _logit(reliability)
        strength = 2.0 * (claim.confidence - 0.5)  # 0 at a shrug, 1 at certainty
        direction = 1.0 if claim.asserts else -1.0
        weight = direction * source_weight * strength
        total += weight
        contributions.append(
            Contribution(
                source=claim.source,
                asserts=claim.asserts,
                confidence=claim.confidence,
                reliability=reliability,
                weight=weight,
            )
        )
        if claim.asserts:
            asserting += 1
        else:
            denying += 1

    return Aggregate(
        proposition=proposition,
        probability=_sigmoid(total),
        prior=prior,
        contributions=tuple(contributions),
        asserting=asserting,
        denying=denying,
    )


# ----------------------------------------------------------------- the ledger


class SocialLedger:
    """Claims heard, outcomes seen, and the reliability that follows.

    Append-only. A claim is never edited and a reliability is never set — it is
    always recomputed from the record, so there is no way to assert that a
    source is trustworthy without the outcomes to back it.
    """

    SCHEMA_VERSION = 1
    DEFAULT_FILENAME = "social.json"

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._claims: list[Claim] = []
        self._records: dict[str, SourceRecord] = {}
        self._resolutions: list[dict[str, Any]] = []

    # ----------------------------------------------------------------- write

    def hear(self, claim: Claim) -> Claim:
        """Record a claim. Explicitly does not believe it."""
        self._claims.append(claim)
        self._records.setdefault(claim.source, SourceRecord(claim.source))
        return claim

    def resolve(self, proposition: str, was_true: bool, *, tick: int = 0) -> list[str]:
        """Score every source that spoke on a proposition. Returns who was right."""
        right: list[str] = []
        for claim in self._claims:
            if claim.proposition != proposition:
                continue
            was_right = claim.asserts == was_true
            record = self._records.setdefault(claim.source, SourceRecord(claim.source))
            self._records[claim.source] = record.with_outcome(tick, was_right)
            if was_right:
                right.append(claim.source)
        self._resolutions.append(
            {"proposition": proposition, "was_true": was_true, "tick": tick}
        )
        return right

    # ------------------------------------------------------------------ read

    def claims(self, proposition: str | None = None) -> tuple[Claim, ...]:
        if proposition is None:
            return tuple(self._claims)
        return tuple(c for c in self._claims if c.proposition == proposition)

    def sources(self) -> tuple[str, ...]:
        return tuple(sorted(self._records))

    def record(self, source: str) -> SourceRecord:
        return self._records.get(source, SourceRecord(source))

    def reliability(self, source: str) -> float:
        return self.record(source).reliability

    def reliabilities(self) -> dict[str, float]:
        return {name: r.reliability for name, r in self._records.items()}

    def ranked(self) -> list[tuple[str, float]]:
        pairs = [(name, r.reliability) for name, r in self._records.items()]
        pairs.sort(key=lambda item: (-item[1], item[0]))
        return pairs

    def believe(
        self, proposition: str, *, prior: float = 0.5
    ) -> Aggregate:
        """What the testimony so far implies, weighted by earned reliability."""
        return aggregate(
            self.claims(proposition),
            self.reliabilities(),
            prior=prior,
            proposition=proposition,
        )

    def contradictions(self) -> list[str]:
        """Propositions on which sources disagree. Both sides are kept."""
        out: list[str] = []
        for proposition in sorted({c.proposition for c in self._claims}):
            claims = self.claims(proposition)
            if any(c.asserts for c in claims) and any(not c.asserts for c in claims):
                out.append(proposition)
        return out

    # ----------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "claims": [c.to_dict() for c in self._claims],
            "records": [r.to_dict() for r in self._records.values()],
            "resolutions": list(self._resolutions),
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
    def load(cls, path: Path | str) -> "SocialLedger":
        ledger = cls(path)
        if not ledger.path or not ledger.path.is_file():
            return ledger
        with ledger.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(
                f"unsupported social schema_version {payload.get('schema_version')!r}"
            )
        ledger._claims = [Claim.from_dict(c) for c in payload.get("claims", [])]
        ledger._records = {
            r["source"]: SourceRecord.from_dict(r) for r in payload.get("records", [])
        }
        ledger._resolutions = list(payload.get("resolutions", []))
        return ledger

    def __len__(self) -> int:
        # Never use in a boolean context; an empty ledger is falsy.
        return len(self._claims)
