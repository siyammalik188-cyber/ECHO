"""Proper scoring for probabilistic predictions.

Accuracy is deliberately not the headline. "P(A) = 0.51" and "P(A) = 0.99" are
very different claims that accuracy scores identically; a proper scoring rule is
the only way to tell a lucky guess from a well-calibrated one. Accuracy is still
reported — it is what people ask for — but it is reported alongside the scores
that can actually distinguish those two claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

DEFAULT_BUCKETS = 10


@dataclass(frozen=True)
class Bucket:
    low: float
    high: float
    count: int
    mean_predicted: float
    observed_frequency: float

    @property
    def gap(self) -> float:
        """Signed miscalibration. Positive means overconfident in the proposition."""
        return self.mean_predicted - self.observed_frequency

    @property
    def label(self) -> str:
        return f"{self.low:.1f}–{self.high:.1f}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "range": self.label,
            "count": self.count,
            "mean_predicted": round(self.mean_predicted, 4),
            "observed_frequency": round(self.observed_frequency, 4),
            "gap": round(self.gap, 4),
        }


def brier(probabilities: Sequence[float], outcomes: Sequence[bool]) -> float | None:
    """Mean squared error. 0 perfect; 0.25 is the score for always saying 0.5."""
    if not probabilities:
        return None
    return sum(
        (p - (1.0 if o else 0.0)) ** 2 for p, o in zip(probabilities, outcomes)
    ) / len(probabilities)


def mean_log_loss(
    probabilities: Sequence[float], outcomes: Sequence[bool]
) -> tuple[float | None, int]:
    """Mean surprisal in nats, and how many predictions had to be excluded.

    A prediction of 0.0 on something that happened has infinite log loss. Rather
    than clamp it into a finite number — which would quietly rescue the worst
    possible prediction — it is excluded and counted. The count is reported.
    """
    import math

    kept, excluded = [], 0
    for p, o in zip(probabilities, outcomes):
        q = p if o else 1.0 - p
        if q <= 0.0:
            excluded += 1
            continue
        kept.append(-math.log(q))
    if not kept:
        return None, excluded
    return sum(kept) / len(kept), excluded


def accuracy(probabilities: Sequence[float], outcomes: Sequence[bool]) -> float | None:
    """Fraction where the >0.5 side matched. Reported, never relied on alone."""
    if not probabilities:
        return None
    hits = sum(1 for p, o in zip(probabilities, outcomes) if p != 0.5 and (p > 0.5) == o)
    return hits / len(probabilities)


def base_rate(outcomes: Sequence[bool]) -> float | None:
    if not outcomes:
        return None
    return sum(1 for o in outcomes if o) / len(outcomes)


def reference_brier(outcomes: Sequence[bool]) -> dict[str, float | None]:
    """What trivial strategies would have scored, for context.

    A Brier score means little in isolation. Beating "always predict the base
    rate" is the bar that matters, and it is a harder bar than beating 0.5.
    """
    if not outcomes:
        return {"always_half": None, "always_base_rate": None}
    rate = base_rate(outcomes) or 0.0
    return {
        "always_half": brier([0.5] * len(outcomes), outcomes),
        "always_base_rate": brier([rate] * len(outcomes), outcomes),
    }


def calibration_buckets(
    probabilities: Sequence[float],
    outcomes: Sequence[bool],
    buckets: int = DEFAULT_BUCKETS,
) -> list[Bucket]:
    """Group predictions by stated probability and compare to what happened.

    Well calibrated means: of everything you called 70%, about 70% happened.
    Empty buckets are omitted — reporting a bucket with no predictions in it
    would imply a measurement that was never made.
    """
    width = 1.0 / buckets
    grouped: list[list[tuple[float, bool]]] = [[] for _ in range(buckets)]

    for p, o in zip(probabilities, outcomes):
        # The epsilon matters. In binary floating point 0.7 / 0.1 is
        # 6.999999999999999, so a naive int() puts a stated 0.7 in the 0.6–0.7
        # bucket — quietly misreporting calibration at every boundary, and
        # always in the same direction.
        index = min(int(p * buckets + 1e-9), buckets - 1)  # 1.0 lands in the top bucket
        grouped[index].append((p, o))

    out: list[Bucket] = []
    for index, pairs in enumerate(grouped):
        if not pairs:
            continue
        out.append(
            Bucket(
                low=index * width,
                high=(index + 1) * width,
                count=len(pairs),
                mean_predicted=sum(p for p, _ in pairs) / len(pairs),
                observed_frequency=sum(1 for _, o in pairs if o) / len(pairs),
            )
        )
    return out


def expected_calibration_error(buckets: Sequence[Bucket]) -> float | None:
    """Count-weighted mean |stated − observed| across buckets. 0 is perfect."""
    total = sum(b.count for b in buckets)
    if not total:
        return None
    return sum(b.count * abs(b.gap) for b in buckets) / total


def summarise(
    probabilities: Sequence[float],
    outcomes: Sequence[bool],
    buckets: int = DEFAULT_BUCKETS,
) -> dict[str, Any]:
    """Every score in one place, plus the references needed to read them."""
    bucket_list = calibration_buckets(probabilities, outcomes, buckets)
    loss, excluded = mean_log_loss(probabilities, outcomes)
    return {
        "count": len(probabilities),
        "brier": brier(probabilities, outcomes),
        "log_loss": loss,
        "log_loss_excluded": excluded,
        "accuracy": accuracy(probabilities, outcomes),
        "base_rate": base_rate(outcomes),
        "reference_brier": reference_brier(outcomes),
        "expected_calibration_error": expected_calibration_error(bucket_list),
        "buckets": [b.to_dict() for b in bucket_list],
    }
