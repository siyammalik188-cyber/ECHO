"""Learning from prediction errors.

The rule this module exists to *not* implement is:

    if prediction_wrong: change_probability()

That is a thermostat, not learning. What happens instead, on every evaluated
prediction:

1. **What happened?** The outcome and the error are already recorded.
2. **Why was it wrong?** A failure analysis considers every explanation it has
   — random variation, unreliable evidence, a changed environment, a failing
   strategy, too little information — and is allowed to conclude `UNKNOWN`.
3. **Is this a pattern or a one-off?** A single error changes nothing. A
   detector compares recent performance to historical performance with a
   binomial test, and must fire **twice, separated in time** before anything is
   proposed. A streak is not a regime change.
4. **What should change?** Only then is a successor strategy proposed, with the
   evidence that motivated it and the condition that would reverse it.
5. **How confident should ECHO be?** Every analysis and proposal carries a
   confidence, derived from the statistics rather than asserted.

Nothing here is autonomous, and nothing rewrites code. A strategy selects among
a fixed menu of predictors; learning means choosing differently on evidence and
being able to say why.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Sequence

from .predictors import LAPLACE_WINDOW
from .strategy import Strategy, StrategyStatus

# --- detector configuration, fixed in advance --------------------------------
RECENT_WINDOW = 10  # predictions in the "recent" sample
MIN_HISTORY = 12  # historical predictions required before any detection
ALPHA = 0.01  # binomial significance threshold
MIN_EFFECT = 0.25  # recent miss rate must exceed historical by at least this
CONFIRM_GAP = 4  # trials that must separate the two firings
MIN_FOR_ANALYSIS = 8  # below this, the honest answer is "not enough information"
STRATEGY_FAILURE_BRIER = 0.35  # sustained mean Brier that indicts the strategy
EVIDENCE_DIVERGENCE = 0.15  # gap between what was reported and what happened


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def binomial_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). Exact, no dependencies.

    Answers: if nothing had changed, how surprising is this many misses?
    """
    if n <= 0:
        return 1.0
    p = min(max(p, 1e-9), 1 - 1e-9)
    return sum(
        math.comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(k, n + 1)
    )


class FailureExplanation(str, Enum):
    RANDOM_VARIATION = "RANDOM_VARIATION"
    EVIDENCE_ERROR = "EVIDENCE_ERROR"
    REGIME_CHANGE = "REGIME_CHANGE"
    STRATEGY_FAILURE = "STRATEGY_FAILURE"
    INSUFFICIENT_INFORMATION = "INSUFFICIENT_INFORMATION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ChangeSignal:
    """The output of comparing recent performance against historical performance."""

    at_trial: int
    historical_performance: dict[str, Any]
    recent_performance: dict[str, Any]
    error_rate: float
    p_value: float
    confidence: float
    suspected_regime_change: bool
    threshold_crossed: str
    confirmed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "at_trial": self.at_trial,
            "historical_performance": self.historical_performance,
            "recent_performance": self.recent_performance,
            "error_rate": round(self.error_rate, 4),
            "p_value": self.p_value,
            "confidence": round(self.confidence, 4),
            "suspected_regime_change": self.suspected_regime_change,
            "threshold_crossed": self.threshold_crossed,
            "confirmed": self.confirmed,
        }


@dataclass
class Sample:
    """One scored prediction, reduced to what the detector needs."""

    trial: int
    probability: float
    outcome: bool
    brier: float

    @property
    def missed(self) -> bool:
        """The outcome fell on the side the strategy thought less likely."""
        assigned = self.probability if self.outcome else 1.0 - self.probability
        return assigned < 0.5


def _summarise(samples: Sequence[Sample]) -> dict[str, Any]:
    if not samples:
        return {"predictions": 0, "mean_brier": None, "miss_rate": None}
    return {
        "predictions": len(samples),
        "mean_brier": round(sum(s.brier for s in samples) / len(samples), 6),
        "miss_rate": round(sum(1 for s in samples if s.missed) / len(samples), 6),
    }


class ChangeDetector:
    """Flags statistically unusual deterioration, and insists on confirmation.

    No trial number is hard-coded anywhere. The detector only ever compares a
    recent sample to the history preceding it. Requiring two firings separated
    by `confirm_gap` trials is what separates a run of bad luck — which does not
    persist — from a change in the process, which does.
    """

    def __init__(
        self,
        recent_window: int = RECENT_WINDOW,
        min_history: int = MIN_HISTORY,
        alpha: float = ALPHA,
        min_effect: float = MIN_EFFECT,
        confirm_gap: int = CONFIRM_GAP,
    ) -> None:
        self.recent_window = recent_window
        self.min_history = min_history
        self.alpha = alpha
        self.min_effect = min_effect
        self.confirm_gap = confirm_gap
        self.signals: list[ChangeSignal] = []
        self._first_firing_trial: int | None = None

    def reset_after_adaptation(self) -> None:
        """Start the comparison afresh after a strategy change.

        Without this, the pre-change history would keep making the post-change
        world look anomalous forever.
        """
        self._first_firing_trial = None

    def assess(self, samples: Sequence[Sample], at_trial: int) -> ChangeSignal | None:
        """Compare the recent window against everything before it."""
        if len(samples) < self.min_history + self.recent_window:
            return None

        recent = list(samples[-self.recent_window :])
        historical = list(samples[: -self.recent_window])

        historical_summary = _summarise(historical)
        recent_summary = _summarise(recent)

        historical_miss = historical_summary["miss_rate"] or 0.0
        recent_misses = sum(1 for s in recent if s.missed)
        recent_miss_rate = recent_misses / len(recent)

        # Floor the historical rate: a strategy that has never missed would make
        # any miss infinitely surprising, which is not a useful alarm.
        baseline = max(historical_miss, 0.05)
        p_value = binomial_tail(recent_misses, len(recent), baseline)
        effect = recent_miss_rate - historical_miss

        significant = p_value <= self.alpha and effect >= self.min_effect

        confirmed = False
        threshold = "none"
        if significant:
            if self._first_firing_trial is None:
                self._first_firing_trial = at_trial
                threshold = (
                    f"p={p_value:.2e} <= {self.alpha} and effect={effect:.2f} >= "
                    f"{self.min_effect} (first firing, awaiting confirmation)"
                )
            elif at_trial - self._first_firing_trial >= self.confirm_gap:
                confirmed = True
                threshold = (
                    f"p={p_value:.2e} <= {self.alpha} and effect={effect:.2f} >= "
                    f"{self.min_effect}, sustained for "
                    f"{at_trial - self._first_firing_trial} trials (confirmed)"
                )
            else:
                threshold = (
                    f"p={p_value:.2e}, significant but only "
                    f"{at_trial - self._first_firing_trial} trial(s) since the first "
                    f"firing (needs {self.confirm_gap})"
                )

        signal = ChangeSignal(
            at_trial=at_trial,
            historical_performance=historical_summary,
            recent_performance=recent_summary,
            error_rate=recent_miss_rate,
            p_value=p_value,
            confidence=max(0.0, min(1.0, 1.0 - p_value)) if significant else 0.0,
            suspected_regime_change=significant,
            threshold_crossed=threshold,
            confirmed=confirmed,
        )
        self.signals.append(signal)
        return signal


@dataclass(frozen=True)
class FailureAnalysis:
    """Why the prediction may have been wrong, and how sure ECHO is of that."""

    explanation: FailureExplanation
    confidence: float
    reasoning: str
    considered: tuple[tuple[str, str], ...] = ()  # (explanation, what was found)

    def to_dict(self) -> dict[str, Any]:
        return {
            "explanation": self.explanation.value,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "considered": [list(c) for c in self.considered],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FailureAnalysis":
        return cls(
            explanation=FailureExplanation(data["explanation"]),
            confidence=data["confidence"],
            reasoning=data["reasoning"],
            considered=tuple(tuple(c) for c in data.get("considered", [])),
        )


def analyse_failure(
    error: float,
    samples: Sequence[Sample],
    signal: ChangeSignal | None,
    evidence_divergence: float | None,
) -> FailureAnalysis:
    """Weigh every explanation on the evidence, and say `UNKNOWN` when unsure.

    Deliberately ordered from most-specific to least, with an explicit fallback.
    No branch is allowed to look at ground truth — the arguments are all things
    ECHO can compute from its own records.
    """
    considered: list[tuple[str, str]] = []
    n = len(samples)
    recent = list(samples[-RECENT_WINDOW:])
    recent_brier = sum(s.brier for s in recent) / len(recent) if recent else 0.0

    if n < MIN_FOR_ANALYSIS:
        considered.append(
            (
                FailureExplanation.INSUFFICIENT_INFORMATION.value,
                f"only {n} evaluated prediction(s); too few to distinguish anything",
            )
        )
        return FailureAnalysis(
            explanation=FailureExplanation.INSUFFICIENT_INFORMATION,
            confidence=0.8,
            reasoning=(
                f"Only {n} prediction(s) have been evaluated. Any explanation for a "
                "single error would be a guess dressed as an inference."
            ),
            considered=tuple(considered),
        )

    # 1. A confirmed, sustained deterioration is the strongest available signal.
    if signal is not None and signal.confirmed:
        considered.append(
            (
                FailureExplanation.REGIME_CHANGE.value,
                f"deterioration confirmed: {signal.threshold_crossed}",
            )
        )
        return FailureAnalysis(
            explanation=FailureExplanation.REGIME_CHANGE,
            confidence=min(0.95, signal.confidence),
            reasoning=(
                f"Recent miss rate {signal.recent_performance['miss_rate']:.2f} against a "
                f"historical {signal.historical_performance['miss_rate']:.2f} "
                f"(p={signal.p_value:.2e}), sustained rather than transient. The "
                "process appears to have changed."
            ),
            considered=tuple(considered),
        )

    # 2. Reports disagreeing with outcomes points at the evidence, not the model.
    if evidence_divergence is not None and evidence_divergence >= EVIDENCE_DIVERGENCE:
        considered.append(
            (
                FailureExplanation.EVIDENCE_ERROR.value,
                f"reported rate and realised rate differ by {evidence_divergence:.2f}",
            )
        )
        return FailureAnalysis(
            explanation=FailureExplanation.EVIDENCE_ERROR,
            confidence=min(0.85, 0.4 + evidence_divergence),
            reasoning=(
                f"What was reported and what actually happened differ by "
                f"{evidence_divergence:.2f}. The predictions may be a reasonable "
                "reading of unreliable information."
            ),
            considered=tuple(considered),
        )

    # 3. Significant but unconfirmed: something is off; saying what would be a guess.
    if signal is not None and signal.suspected_regime_change and not signal.confirmed:
        considered.append(
            (
                FailureExplanation.REGIME_CHANGE.value,
                "deterioration is significant but not yet sustained",
            )
        )
        considered.append(
            (
                FailureExplanation.RANDOM_VARIATION.value,
                "a run of bad luck would look the same at this point",
            )
        )
        return FailureAnalysis(
            explanation=FailureExplanation.UNKNOWN,
            confidence=0.4,
            reasoning=(
                f"Recent performance is significantly worse than history "
                f"(p={signal.p_value:.2e}), but it has not persisted long enough to "
                "separate a changed process from an unlucky run. Both remain live."
            ),
            considered=tuple(considered),
        )

    # 4. Sustained poor scores with no regime signal and no bad evidence.
    if recent_brier >= STRATEGY_FAILURE_BRIER and len(recent) >= RECENT_WINDOW:
        considered.append(
            (
                FailureExplanation.STRATEGY_FAILURE.value,
                f"recent mean Brier {recent_brier:.3f} with no change signal",
            )
        )
        return FailureAnalysis(
            explanation=FailureExplanation.STRATEGY_FAILURE,
            confidence=0.55,
            reasoning=(
                f"Recent mean Brier is {recent_brier:.3f} — persistently poor — "
                "without evidence of a changed process or unreliable reports. The "
                "strategy itself may be a bad fit."
            ),
            considered=tuple(considered),
        )

    # 5. An individual miss with none of the above is what noise looks like.
    considered.append(
        (
            FailureExplanation.RANDOM_VARIATION.value,
            f"error {error:.2f} with recent mean Brier {recent_brier:.3f} and no signal",
        )
    )
    return FailureAnalysis(
        explanation=FailureExplanation.RANDOM_VARIATION,
        confidence=0.6,
        reasoning=(
            f"A single miss (error {error:.2f}) against otherwise unremarkable recent "
            f"performance (mean Brier {recent_brier:.3f}). Probabilistic predictions "
            "are supposed to be wrong sometimes; this looks like one of those times."
        ),
        considered=tuple(considered),
    )


@dataclass(frozen=True)
class LearningExperience:
    """An immutable record that ECHO analysed one prediction error.

    The prediction it refers to is untouched — this sits beside it.
    """

    prediction_id: str
    observed_outcome: bool
    prediction_error: float
    relevant_evidence: tuple[str, ...]
    previous_strategy: str
    failure_analysis: FailureAnalysis
    proposed_update: dict[str, Any] | None
    confidence: float
    id: str = field(default_factory=lambda: _new_id("LEXP"))
    created_at: str = field(default_factory=_now)
    trial: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "learning_experience_id": self.id,
            "prediction_id": self.prediction_id,
            "observed_outcome": self.observed_outcome,
            "prediction_error": self.prediction_error,
            "relevant_evidence": list(self.relevant_evidence),
            "previous_strategy": self.previous_strategy,
            "failure_analysis": self.failure_analysis.to_dict(),
            "proposed_update": self.proposed_update,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "trial": self.trial,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LearningExperience":
        return cls(
            id=data["learning_experience_id"],
            prediction_id=data["prediction_id"],
            observed_outcome=data["observed_outcome"],
            prediction_error=data["prediction_error"],
            relevant_evidence=tuple(data.get("relevant_evidence", [])),
            previous_strategy=data["previous_strategy"],
            failure_analysis=FailureAnalysis.from_dict(data["failure_analysis"]),
            proposed_update=data.get("proposed_update"),
            confidence=data["confidence"],
            created_at=data["created_at"],
            trial=data.get("trial", 0),
        )


@dataclass(frozen=True)
class StrategyRevision:
    """A recorded decision to change strategy, and what became of it.

    Answers, permanently: what changed, why, what triggered it, whether it
    helped, and what would make ECHO reverse it.
    """

    at_trial: int
    from_strategy_id: str
    to_strategy_id: str
    reason: str
    triggering_experience_ids: tuple[str, ...]
    change_signal: dict[str, Any] | None
    reversal_condition: str
    confidence: float
    id: str = field(default_factory=lambda: _new_id("REV"))
    created_at: str = field(default_factory=_now)
    outcome: str = "pending"  # pending | improved | no_improvement
    measured_before: dict[str, Any] | None = None
    measured_after: dict[str, Any] | None = None

    def concluded(
        self, outcome: str, before: dict[str, Any], after: dict[str, Any]
    ) -> "StrategyRevision":
        """Return a new revision carrying the verdict. The original is unchanged."""
        return StrategyRevision(
            id=self.id,
            at_trial=self.at_trial,
            from_strategy_id=self.from_strategy_id,
            to_strategy_id=self.to_strategy_id,
            reason=self.reason,
            triggering_experience_ids=self.triggering_experience_ids,
            change_signal=self.change_signal,
            reversal_condition=self.reversal_condition,
            confidence=self.confidence,
            created_at=self.created_at,
            outcome=outcome,
            measured_before=before,
            measured_after=after,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision_id": self.id,
            "at_trial": self.at_trial,
            "from_strategy_id": self.from_strategy_id,
            "to_strategy_id": self.to_strategy_id,
            "reason": self.reason,
            "triggering_experience_ids": list(self.triggering_experience_ids),
            "change_signal": self.change_signal,
            "reversal_condition": self.reversal_condition,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "outcome": self.outcome,
            "measured_before": self.measured_before,
            "measured_after": self.measured_after,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "StrategyRevision":
        return cls(
            id=data["revision_id"],
            at_trial=data["at_trial"],
            from_strategy_id=data["from_strategy_id"],
            to_strategy_id=data["to_strategy_id"],
            reason=data["reason"],
            triggering_experience_ids=tuple(data.get("triggering_experience_ids", [])),
            change_signal=data.get("change_signal"),
            reversal_condition=data["reversal_condition"],
            confidence=data["confidence"],
            created_at=data["created_at"],
            outcome=data.get("outcome", "pending"),
            measured_before=data.get("measured_before"),
            measured_after=data.get("measured_after"),
        )


def propose_successor(current: Strategy, signal: ChangeSignal) -> Strategy:
    """Propose a strategy whose assumptions match what the evidence suggests.

    The current strategy assumes stationarity. A confirmed, sustained
    deterioration is evidence against exactly that assumption, so the successor
    drops it: a bounded window weights recent evidence and lets old evidence
    fall away.

    The window is not tuned to any environment — it is the detector's own recent
    window, the same span over which the deterioration was established.
    """
    return current.descendant(
        description=(
            f"Laplace-smoothed frequency over the most recent {RECENT_WINDOW} "
            "observations, discarding older evidence."
        ),
        kind=LAPLACE_WINDOW,
        params={"alpha": 1.0, "window": RECENT_WINDOW},
        assumptions=(
            "The process may change; recent evidence is more informative than old.",
            "A bounded window trades statistical precision for responsiveness.",
        ),
        applicable_conditions=(
            "Processes suspected of changing regime.",
            "Any setting where sustained deterioration has been observed.",
        ),
        confidence=min(0.8, signal.confidence),
    )
