"""Explicit predictions, the information available when they were made, and
their evaluation against what actually happened.

**PREDICTION IS NOT LEARNING.** Nothing here changes ECHO. A prediction is
issued, an outcome arrives, an error is recorded, and an `Experience` is filed.
Nothing reads those experiences back to alter a belief, a probability, or a
strategy. This is the experience loop that learning would later need — the loop
itself, empty.

## Why the shapes are what they are

`Prediction` is frozen and holds **only** what was known when it was issued. It
has no outcome fields at all, so there is no attribute to overwrite when reality
arrives. Evaluation produces a **new** `PredictionRecord` that wraps the *same*
frozen `Prediction` instance alongside a new `Evaluation`. "The prediction
remains unchanged; only evaluation fields are added" is therefore enforced by
the type, not by discipline.

## Temporal integrity

A prediction may only cite information that existed strictly *before* its
timestamp. This is enforced twice, on purpose:

1. `Timeline.view_as_of(t)` hands the predictor a view that physically cannot
   produce an observation at or after `t`. Leakage is prevented by construction.
2. `issue_prediction()` independently re-checks every cited id against the
   timeline and raises `TemporalLeakageError` if any is from the future.

The second check exists because the first can be bypassed by a caller who goes
around the view. A prediction that used future information is not a good
prediction with a caveat — it is not a prediction at all, and the system refuses
to record it.

Time is a **logical tick** (an integer), not a wall clock. Ordering is what
matters for leakage, and integers make the whole experiment reproducible.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable

# Probabilities are recorded exactly as stated, including 0.0 and 1.0. They are
# only clamped where the mathematics demands it (log loss), and that clamping is
# reported rather than hidden.
LOG_LOSS_EPSILON = 1e-15


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


class TemporalLeakageError(RuntimeError):
    """A prediction tried to use information that did not exist yet."""


class EvaluationStatus(str, Enum):
    PENDING = "pending"
    EVALUATED = "evaluated"


# --------------------------------------------------------------------- time


@dataclass(frozen=True)
class Observation:
    """Something ECHO saw, stamped with when it became available."""

    content: str
    at: int  # logical tick
    id: str = field(default_factory=lambda: _new_id("OBS"))
    payload: dict[str, Any] = field(default_factory=dict)
    recorded_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if self.at < 0:
            raise ValueError(f"observation tick must be >= 0, got {self.at}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content,
            "at": self.at,
            "payload": dict(self.payload),
            "recorded_at": self.recorded_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Observation":
        return cls(
            id=data["id"],
            content=data["content"],
            at=data["at"],
            payload=data.get("payload", {}),
            recorded_at=data["recorded_at"],
        )


class TimelineView:
    """A window onto the timeline that stops at `horizon`.

    There is no method on this object that returns an observation at or after
    the horizon. A predictor handed one of these cannot see the future even if
    it tries — the data is not reachable from here.
    """

    def __init__(self, horizon: int, observations: Iterable[Observation]) -> None:
        self.horizon = horizon
        self._observations = tuple(
            sorted((o for o in observations if o.at < horizon), key=lambda o: (o.at, o.id))
        )

    @property
    def observations(self) -> tuple[Observation, ...]:
        return self._observations

    def ids(self) -> tuple[str, ...]:
        return tuple(o.id for o in self._observations)

    def get(self, observation_id: str) -> Observation | None:
        for observation in self._observations:
            if observation.id == observation_id:
                return observation
        return None

    def outcomes(self, key: str = "outcome") -> tuple[Any, ...]:
        return tuple(o.payload[key] for o in self._observations if key in o.payload)

    def __len__(self) -> int:
        return len(self._observations)

    def __iter__(self):
        return iter(self._observations)

    def __repr__(self) -> str:
        return f"TimelineView(horizon={self.horizon}, observations={len(self)})"


class Timeline:
    """Every observation, and the authority on when each became available."""

    def __init__(self) -> None:
        self._observations: dict[str, Observation] = {}

    def record(self, observation: Observation) -> Observation:
        if observation.id in self._observations:
            raise ValueError(f"observation {observation.id} is already on the timeline")
        self._observations[observation.id] = observation
        return observation

    def add(self, content: str, at: int, **payload: Any) -> Observation:
        return self.record(Observation(content=content, at=at, payload=payload))

    def get(self, observation_id: str) -> Observation | None:
        return self._observations.get(observation_id)

    def all(self) -> tuple[Observation, ...]:
        return tuple(sorted(self._observations.values(), key=lambda o: (o.at, o.id)))

    def view_as_of(self, horizon: int) -> TimelineView:
        """Everything strictly before `horizon`. The only sanctioned way to predict."""
        return TimelineView(horizon, self._observations.values())

    def check_available(self, observation_ids: Iterable[str], before: int) -> None:
        """Raise unless every id exists and predates `before`.

        The independent second check. `unknown` is treated as leakage too: an id
        that cannot be resolved cannot be shown to predate the prediction.
        """
        future, unknown = [], []
        for observation_id in observation_ids:
            observation = self._observations.get(observation_id)
            if observation is None:
                unknown.append(observation_id)
            elif observation.at >= before:
                future.append((observation_id, observation.at))

        if unknown or future:
            parts = []
            if future:
                parts.append(
                    "information from at or after the prediction time: "
                    + ", ".join(f"{oid} (t={t}, prediction at t={before})" for oid, t in future)
                )
            if unknown:
                parts.append("unknown observation ids: " + ", ".join(unknown))
            raise TemporalLeakageError(
                "prediction rejected — " + "; ".join(parts)
            )

    def __len__(self) -> int:
        return len(self._observations)


# --------------------------------------------------------------- prediction


@dataclass(frozen=True)
class Prediction:
    """A probability assigned to a proposition at a moment in time.

    Frozen, and deliberately containing **no outcome fields**. There is nothing
    here for a later result to overwrite. If it was 0.70 when issued, it is 0.70
    forever.
    """

    proposition: str
    predicted_probability: float
    prediction_time: int  # logical tick — the temporal boundary
    information_available_at_prediction_time: tuple[str, ...]
    evidence_ids_used: tuple[str, ...] = ()
    belief_id: str | None = None
    id: str = field(default_factory=lambda: _new_id("PRED"))
    issued_at: str = field(default_factory=_now)  # wall clock, for the audit trail
    rationale: str = ""

    def __post_init__(self) -> None:
        if not self.proposition.strip():
            raise ValueError("proposition must not be empty")
        probability = float(self.predicted_probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                f"predicted_probability must be within 0.0–1.0, got {probability!r}"
            )
        object.__setattr__(self, "predicted_probability", probability)
        object.__setattr__(
            self,
            "information_available_at_prediction_time",
            tuple(self.information_available_at_prediction_time),
        )
        object.__setattr__(self, "evidence_ids_used", tuple(self.evidence_ids_used))

    @property
    def stated_direction(self) -> str:
        """The binary reading — reported, but never a substitute for the number.

        `P(A) = 0.51` and `P(A) = 0.99` both read as "A" here, which is exactly
        why accuracy alone is not an acceptable score.
        """
        if self.predicted_probability > 0.5:
            return "true"
        if self.predicted_probability < 0.5:
            return "false"
        return "undecided"

    def to_dict(self) -> dict[str, Any]:
        return {
            "prediction_id": self.id,
            "proposition": self.proposition,
            "predicted_probability": self.predicted_probability,
            "prediction_time": self.prediction_time,
            "issued_at": self.issued_at,
            "information_available_at_prediction_time": list(
                self.information_available_at_prediction_time
            ),
            "evidence_ids_used": list(self.evidence_ids_used),
            "belief_id": self.belief_id,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Prediction":
        return cls(
            id=data["prediction_id"],
            proposition=data["proposition"],
            predicted_probability=data["predicted_probability"],
            prediction_time=data["prediction_time"],
            issued_at=data["issued_at"],
            information_available_at_prediction_time=tuple(
                data.get("information_available_at_prediction_time", [])
            ),
            evidence_ids_used=tuple(data.get("evidence_ids_used", [])),
            belief_id=data.get("belief_id"),
            rationale=data.get("rationale", ""),
        )


def issue_prediction(
    timeline: Timeline,
    proposition: str,
    predicted_probability: float,
    prediction_time: int,
    information_available: Iterable[str] | None = None,
    evidence_ids_used: Iterable[str] = (),
    belief_id: str | None = None,
    rationale: str = "",
) -> Prediction:
    """Issue a prediction, refusing it outright if it cites future information.

    `information_available` defaults to everything the timeline held before
    `prediction_time`, which is the honest answer to "what did I have at the
    time". Both it and `evidence_ids_used` are checked.
    """
    view = timeline.view_as_of(prediction_time)
    available = tuple(information_available) if information_available is not None else view.ids()

    timeline.check_available(available, before=prediction_time)
    timeline.check_available(evidence_ids_used, before=prediction_time)

    return Prediction(
        proposition=proposition,
        predicted_probability=predicted_probability,
        prediction_time=prediction_time,
        information_available_at_prediction_time=available,
        evidence_ids_used=tuple(evidence_ids_used),
        belief_id=belief_id,
        rationale=rationale,
    )


# --------------------------------------------------------------- evaluation


@dataclass(frozen=True)
class Evaluation:
    """What reality did, and how wrong the prediction turned out to be."""

    actual_outcome: bool
    outcome_label: str
    evaluation_time: int
    prediction_error: float  # |p - outcome|
    brier_score: float
    log_loss: float | None  # None when mathematically undefined
    correctly_anticipated: bool
    evaluated_at: str = field(default_factory=_now)
    id: str = field(default_factory=lambda: _new_id("EVAL"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.id,
            "actual_outcome": self.actual_outcome,
            "outcome_label": self.outcome_label,
            "evaluation_time": self.evaluation_time,
            "evaluated_at": self.evaluated_at,
            "prediction_error": self.prediction_error,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "correctly_anticipated": self.correctly_anticipated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Evaluation":
        return cls(
            id=data["evaluation_id"],
            actual_outcome=data["actual_outcome"],
            outcome_label=data["outcome_label"],
            evaluation_time=data["evaluation_time"],
            evaluated_at=data["evaluated_at"],
            prediction_error=data["prediction_error"],
            brier_score=data["brier_score"],
            log_loss=data["log_loss"],
            correctly_anticipated=data["correctly_anticipated"],
        )


def brier_score(probability: float, outcome: bool) -> float:
    """Squared error. 0 is perfect, 0.25 is what you get for always saying 0.5."""
    return (probability - (1.0 if outcome else 0.0)) ** 2


def log_loss(probability: float, outcome: bool) -> float | None:
    """Surprisal, in nats. `None` when the claim was certain and wrong.

    Returning `None` rather than a clamped number is deliberate: a confident
    error is infinitely surprising, and quietly substituting 34.5 for infinity
    would flatter exactly the predictions that deserve it least. The aggregate
    reports how many were excluded.
    """
    p = probability if outcome else 1.0 - probability
    if p <= 0.0:
        return None
    return -math.log(max(p, LOG_LOSS_EPSILON))


@dataclass(frozen=True)
class PredictionRecord:
    """A prediction, and its evaluation once reality has weighed in.

    Evaluating returns a **new** record around the *same* frozen `Prediction`.
    The original object is never touched, so the issued numbers cannot drift.
    """

    prediction: Prediction
    evaluation: Evaluation | None = None

    # --- the issued side, passed straight through ---------------------------
    @property
    def id(self) -> str:
        return self.prediction.id

    @property
    def proposition(self) -> str:
        return self.prediction.proposition

    @property
    def predicted_probability(self) -> float:
        return self.prediction.predicted_probability

    @property
    def prediction_time(self) -> int:
        return self.prediction.prediction_time

    @property
    def information_available_at_prediction_time(self) -> tuple[str, ...]:
        return self.prediction.information_available_at_prediction_time

    @property
    def evidence_ids_used(self) -> tuple[str, ...]:
        return self.prediction.evidence_ids_used

    @property
    def belief_id(self) -> str | None:
        return self.prediction.belief_id

    # --- the evaluated side, absent until reality arrives --------------------
    @property
    def evaluation_status(self) -> EvaluationStatus:
        return (
            EvaluationStatus.PENDING
            if self.evaluation is None
            else EvaluationStatus.EVALUATED
        )

    @property
    def actual_outcome(self) -> bool | None:
        return None if self.evaluation is None else self.evaluation.actual_outcome

    @property
    def evaluation_time(self) -> int | None:
        return None if self.evaluation is None else self.evaluation.evaluation_time

    @property
    def prediction_error(self) -> float | None:
        return None if self.evaluation is None else self.evaluation.prediction_error

    @property
    def is_evaluated(self) -> bool:
        return self.evaluation is not None

    def evaluate(
        self, outcome: bool, outcome_label: str, evaluation_time: int
    ) -> "PredictionRecord":
        """Return a new record carrying the outcome. Refuses a second evaluation."""
        if self.evaluation is not None:
            raise ValueError(
                f"prediction {self.id} has already been evaluated; "
                "a prediction is scored once against reality"
            )
        if evaluation_time < self.prediction.prediction_time:
            raise ValueError(
                f"evaluation time {evaluation_time} precedes prediction time "
                f"{self.prediction.prediction_time}"
            )

        probability = self.prediction.predicted_probability
        evaluation = Evaluation(
            actual_outcome=outcome,
            outcome_label=outcome_label,
            evaluation_time=evaluation_time,
            prediction_error=abs(probability - (1.0 if outcome else 0.0)),
            brier_score=brier_score(probability, outcome),
            log_loss=log_loss(probability, outcome),
            correctly_anticipated=(probability > 0.5) == outcome
            if probability != 0.5
            else False,
        )
        return PredictionRecord(prediction=self.prediction, evaluation=evaluation)

    def to_dict(self) -> dict[str, Any]:
        payload = self.prediction.to_dict()
        payload["evaluation_status"] = self.evaluation_status.value
        payload["actual_outcome"] = self.actual_outcome
        payload["evaluation_time"] = self.evaluation_time
        payload["prediction_error"] = self.prediction_error
        payload["evaluation"] = None if self.evaluation is None else self.evaluation.to_dict()
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PredictionRecord":
        evaluation = data.get("evaluation")
        return cls(
            prediction=Prediction.from_dict(data),
            evaluation=Evaluation.from_dict(evaluation) if evaluation else None,
        )


# --------------------------------------------------------------- experience


@dataclass(frozen=True)
class Experience:
    """A filed record that a prediction met reality.

    Written after every evaluation and then left alone. **Nothing reads these to
    change ECHO.** They exist so that a later system could learn from them; that
    system does not exist, and building it is not part of this challenge.
    """

    prediction_id: str
    proposition: str
    prediction: float
    outcome: bool
    outcome_label: str
    error: float
    evidence_available: tuple[str, ...]
    id: str = field(default_factory=lambda: _new_id("EXP"))
    timestamp: str = field(default_factory=_now)
    tick: int = 0

    @classmethod
    def from_record(cls, record: PredictionRecord) -> "Experience":
        if record.evaluation is None:
            raise ValueError(
                f"cannot file an experience for unevaluated prediction {record.id}"
            )
        return cls(
            prediction_id=record.id,
            proposition=record.proposition,
            prediction=record.predicted_probability,
            outcome=record.evaluation.actual_outcome,
            outcome_label=record.evaluation.outcome_label,
            error=record.evaluation.prediction_error,
            evidence_available=record.information_available_at_prediction_time,
            tick=record.evaluation.evaluation_time,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "experience_id": self.id,
            "prediction_id": self.prediction_id,
            "proposition": self.proposition,
            "prediction": self.prediction,
            "outcome": self.outcome,
            "outcome_label": self.outcome_label,
            "error": self.error,
            "evidence_available": list(self.evidence_available),
            "timestamp": self.timestamp,
            "tick": self.tick,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Experience":
        return cls(
            id=data["experience_id"],
            prediction_id=data["prediction_id"],
            proposition=data["proposition"],
            prediction=data["prediction"],
            outcome=data["outcome"],
            outcome_label=data["outcome_label"],
            error=data["error"],
            evidence_available=tuple(data.get("evidence_available", [])),
            timestamp=data["timestamp"],
            tick=data.get("tick", 0),
        )
