"""The loop: one cycle that uses every module, and remembers why it did.

Ten capabilities were built one at a time. This runs them as a single cycle:

```
OBSERVE → REMEMBER → BELIEVE → PREDICT → ACT/EXPERIMENT → OBSERVE RESULT
   → ANALYSE ERROR → LEARN → DISCOVER → ABSTRACT → TRANSFER
   → REASSESS CAUSAL MODELS → UPDATE META-CONFIDENCE → CONSULT AGENTS → REPEAT
```

**This is a coordinator, not an intelligence.** It owns no reasoning of its own:
every step delegates to the module that was built and tested for it, and the
only thing this file adds is order, plumbing, and a record of what happened.
That constraint is enforced by a test which asserts no arithmetic beyond
bookkeeping lives here.

**Provenance is the product.** Every conclusion accumulates a `Trace` naming the
evidence, the experiments, the predictions, the prior belief, the sources that
contributed, the patterns transferred, and how confidence moved. `why(claim)`
returns that trace. A conclusion without one is a bug, and a test looks for
exactly that.

**Novelty is a first-class answer.** When nothing in the pattern library matches
what is happening, the correct output is `NO KNOWN PATTERN` followed by
experimentation and discovery — not the closest old explanation stretched to
fit. Forcing a stale pattern onto a new world is the failure mode this exists
to prevent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .metacognition import Claim as MetaClaim
from .metacognition import Metacognition
from .pattern import PatternStatus, StructuralPattern
from .social import SocialLedger
from .social import parse_claim as parse_social_claim

# ------------------------------------------------------------------- settings

#: Below this match quality, the pattern library is judged not to cover the
#: situation and the answer is NO KNOWN PATTERN.
NOVELTY_THRESHOLD = 0.55

#: A source whose recent reliability falls this far below its lifetime figure
#: has turned, and is flagged rather than quietly averaged.
RELIABILITY_DROP = 0.25

#: Window for judging whether a source has turned. Short windows make a
#: good source look turned on an ordinary run of bad luck.
RECENT_WINDOW = 24

#: ...and the recent record must be genuinely poor, not merely below the
#: source's own average. Without this, every source trips eventually.
TURNED_CEILING = 0.5


class Stage(str, Enum):
    OBSERVE = "observe"
    REMEMBER = "remember"
    BELIEVE = "believe"
    PREDICT = "predict"
    ACT = "act"
    OBSERVE_RESULT = "observe_result"
    ANALYSE_ERROR = "analyse_error"
    LEARN = "learn"
    DISCOVER = "discover"
    ABSTRACT = "abstract"
    TRANSFER = "transfer"
    REASSESS_CAUSAL = "reassess_causal"
    UPDATE_META = "update_meta"
    CONSULT = "consult"


#: The order the loop runs in. Declared once so the test that checks the loop
#: covers every stage has something to check against.
LOOP_ORDER: tuple[Stage, ...] = (
    Stage.OBSERVE,
    Stage.REMEMBER,
    Stage.BELIEVE,
    Stage.PREDICT,
    Stage.ACT,
    Stage.OBSERVE_RESULT,
    Stage.ANALYSE_ERROR,
    Stage.LEARN,
    Stage.DISCOVER,
    Stage.ABSTRACT,
    Stage.TRANSFER,
    Stage.REASSESS_CAUSAL,
    Stage.UPDATE_META,
    Stage.CONSULT,
)


class FailureKind(str, Enum):
    REGIME_CHANGE = "REGIME_CHANGE"
    SOURCE_TURNED = "SOURCE_TURNED"
    PATTERN_STOPPED_WORKING = "PATTERN_STOPPED_WORKING"
    TRANSFER_HARMFUL = "TRANSFER_HARMFUL"
    CAUSAL_MODEL_INCONSISTENT = "CAUSAL_MODEL_INCONSISTENT"


@dataclass(frozen=True)
class TraceStep:
    """One thing that happened, and which module did it."""

    stage: Stage
    module: str
    summary: str
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage.value,
            "module": self.module,
            "summary": self.summary,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Trace:
    """Why ECHO concluded something. The answer to `why(...)`."""

    conclusion: str
    tick: int
    steps: tuple[TraceStep, ...]
    evidence: tuple[str, ...] = ()
    experiments: tuple[str, ...] = ()
    predictions: tuple[str, ...] = ()
    prior_belief: float | None = None
    current_belief: float | None = None
    source_contributions: tuple[tuple[str, float], ...] = ()
    transferred_patterns: tuple[str, ...] = ()
    confidence_before: float | None = None
    confidence_after: float | None = None

    @property
    def confidence_change(self) -> float | None:
        if self.confidence_before is None or self.confidence_after is None:
            return None
        return self.confidence_after - self.confidence_before

    def explain(self) -> str:
        """A plain-language account. Descriptive only; claims no understanding."""
        lines = [f"CONCLUSION: {self.conclusion}", f"AT TICK:    {self.tick}"]
        if self.prior_belief is not None and self.current_belief is not None:
            lines.append(
                f"BELIEF:     {self.prior_belief:.4f} -> {self.current_belief:.4f}"
            )
        if self.confidence_change is not None:
            lines.append(
                f"CONFIDENCE: {self.confidence_before:.4f} -> "
                f"{self.confidence_after:.4f} ({self.confidence_change:+.4f})"
            )
        for label, values in (
            ("EVIDENCE", self.evidence),
            ("EXPERIMENTS", self.experiments),
            ("PREDICTIONS", self.predictions),
            ("PATTERNS", self.transferred_patterns),
        ):
            for value in values:
                lines.append(f"{label + ':':12s}{value}")
        for source, weight in self.source_contributions:
            lines.append(f"SOURCE:     {source} contributed {weight:+.4f}")
        for step in self.steps:
            lines.append(f"  [{step.stage.value}] {step.module}: {step.summary}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "conclusion": self.conclusion,
            "tick": self.tick,
            "steps": [s.to_dict() for s in self.steps],
            "evidence": list(self.evidence),
            "experiments": list(self.experiments),
            "predictions": list(self.predictions),
            "prior_belief": self.prior_belief,
            "current_belief": self.current_belief,
            "source_contributions": [list(s) for s in self.source_contributions],
            "transferred_patterns": list(self.transferred_patterns),
            "confidence_before": self.confidence_before,
            "confidence_after": self.confidence_after,
            "confidence_change": self.confidence_change,
        }


@dataclass(frozen=True)
class CycleResult:
    """One turn of the loop."""

    tick: int
    stages_run: tuple[Stage, ...]
    trace: Trace
    prediction: float | None
    outcome: bool | None
    known_pattern: str | None
    novelty: bool
    failures_detected: tuple[FailureKind, ...] = ()
    abstained: bool = False

    @property
    def correct(self) -> bool | None:
        if self.prediction is None or self.outcome is None:
            return None
        return (self.prediction > 0.5) == self.outcome

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "stages_run": [s.value for s in self.stages_run],
            "trace": self.trace.to_dict(),
            "prediction": self.prediction,
            "outcome": self.outcome,
            "correct": self.correct,
            "known_pattern": self.known_pattern,
            "novelty": self.novelty,
            "failures_detected": [f.value for f in self.failures_detected],
            "abstained": self.abstained,
        }


# ------------------------------------------------------------ knowledge state


@dataclass
class KnowledgeConflict:
    """A belief that new evidence contradicts. The old one is never deleted."""

    proposition: str
    old_belief: float
    new_evidence: str
    revision_reason: str
    current_belief: float
    tick: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposition": self.proposition,
            "old_belief": self.old_belief,
            "new_evidence": self.new_evidence,
            "revision_reason": self.revision_reason,
            "current_belief": self.current_belief,
            "tick": self.tick,
        }


class PersistentState:
    """Everything ECHO keeps across a restart — and nothing it should not.

    Working context is deliberately absent. There is no field for it, no
    `to_dict` entry, and a test asserts the reloaded state has no scratch
    attribute: temporary state that could be persisted eventually would be.
    """

    SCHEMA_VERSION = 1

    def __init__(self) -> None:
        self.memories: list[dict[str, Any]] = []
        self.beliefs: dict[str, float] = {}
        self.belief_history: list[KnowledgeConflict] = []
        self.strategies: list[dict[str, Any]] = []
        self.patterns: list[StructuralPattern] = []
        self.metacognition = Metacognition()
        self.social = SocialLedger()
        self.causal_posterior: dict[str, float] = {}
        self.tick: int = 0

    # --------------------------------------------------------------- beliefs

    def revise(
        self, proposition: str, new_belief: float, evidence: str, reason: str
    ) -> KnowledgeConflict | None:
        """Update a belief, keeping the old one and why it changed.

        Never an overwrite. The previous value, the evidence that moved it and
        the reason are all appended to the history, so a later reader can see
        what ECHO used to think and what changed its mind.
        """
        old = self.beliefs.get(proposition)
        self.beliefs[proposition] = new_belief
        if old is None:
            return None
        conflict = KnowledgeConflict(
            proposition=proposition,
            old_belief=old,
            new_evidence=evidence,
            revision_reason=reason,
            current_belief=new_belief,
            tick=self.tick,
        )
        self.belief_history.append(conflict)
        return conflict

    def revisions_of(self, proposition: str) -> list[KnowledgeConflict]:
        return [c for c in self.belief_history if c.proposition == proposition]

    # ----------------------------------------------------------- persistence

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "memories": self.memories,
            "beliefs": self.beliefs,
            "belief_history": [c.to_dict() for c in self.belief_history],
            "strategies": self.strategies,
            "patterns": [p.to_dict() for p in self.patterns],
            "metacognition": self.metacognition.to_dict(),
            "social": self.social.to_dict(),
            "causal_posterior": self.causal_posterior,
            "tick": self.tick,
        }

    def save(self, path: Path | str) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(target)
        return target

    @classmethod
    def load(cls, path: Path | str) -> "PersistentState":
        state = cls()
        source = Path(path)
        if not source.is_file():
            return state
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError(
                f"unsupported integration schema_version {payload.get('schema_version')!r}"
            )
        state.memories = list(payload.get("memories", []))
        state.beliefs = dict(payload.get("beliefs", {}))
        state.belief_history = [
            KnowledgeConflict(**c) for c in payload.get("belief_history", [])
        ]
        state.strategies = list(payload.get("strategies", []))
        state.patterns = [
            StructuralPattern.from_dict(p) for p in payload.get("patterns", [])
        ]
        meta = Metacognition()
        meta._claims = [  # noqa: SLF001 - reconstructing our own type
            MetaClaim.from_dict(c) for c in payload.get("metacognition", {}).get("claims", [])
        ]
        state.metacognition = meta
        social = SocialLedger()
        social_payload = payload.get("social", {})
        if social_payload:
            from .social import Claim as SocialClaim
            from .social import SourceRecord

            social._claims = [  # noqa: SLF001
                SocialClaim.from_dict(c) for c in social_payload.get("claims", [])
            ]
            social._records = {  # noqa: SLF001
                r["source"]: SourceRecord.from_dict(r)
                for r in social_payload.get("records", [])
            }
            social._resolutions = list(social_payload.get("resolutions", []))  # noqa: SLF001
        state.social = social
        state.causal_posterior = dict(payload.get("causal_posterior", {}))
        state.tick = int(payload.get("tick", 0))
        return state


# ------------------------------------------------------------- pattern match


def match_quality(pattern: StructuralPattern, signature: Mapping[str, Any]) -> float:
    """How well a known pattern fits the situation in front of ECHO.

    Deliberately crude and deliberately explicit: a fraction of the situation's
    declared features that the pattern's own shape accounts for. The point is
    not that this is a good matcher — it is that the number is inspectable and
    that a low one produces `NO KNOWN PATTERN` rather than a stretched
    explanation.
    """
    wanted = set(signature.get("features", ()))
    if not wanted:
        return 0.0
    covered = {node.op for node in pattern.template.walk()}
    overlap = len(wanted & covered)
    return overlap / len(wanted)


def best_match(
    patterns: Sequence[StructuralPattern], signature: Mapping[str, Any]
) -> tuple[StructuralPattern | None, float]:
    best: StructuralPattern | None = None
    score = 0.0
    for pattern in patterns:
        if pattern.status in (PatternStatus.REJECTED, PatternStatus.RETIRED):
            continue
        quality = match_quality(pattern, signature)
        if quality > score:
            best, score = pattern, quality
    return best, score


def is_novel(
    patterns: Sequence[StructuralPattern],
    signature: Mapping[str, Any],
    *,
    threshold: float = NOVELTY_THRESHOLD,
) -> tuple[bool, StructuralPattern | None, float]:
    """`NO KNOWN PATTERN` is an answer, not a failure to produce one."""
    pattern, quality = best_match(patterns, signature)
    return (quality < threshold, pattern, quality)


def source_turned(
    ledger: SocialLedger, source: str, *, window: int = RECENT_WINDOW
) -> bool:
    """Has a source that used to be reliable stopped being so?

    Lifetime accuracy hides this by construction — that is the whole point of
    comparing a recent window against it.
    """
    record = ledger.record(source)
    recent = record.recent_reliability(window)
    if recent is None:
        return False
    return recent < TURNED_CEILING and record.reliability - recent >= RELIABILITY_DROP
