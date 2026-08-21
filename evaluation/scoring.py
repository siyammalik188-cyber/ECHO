"""Turn raw extractor output into precision, recall, and calibration figures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .matching import band_deviation, in_band, matches


@dataclass
class Judgement:
    """What became of one proposed memory, or one unmet expectation."""

    kind: str  # true_positive | false_positive | false_negative | tolerated
    case_id: str
    label: str | None = None  # the expectation or forbidden-pattern label
    content: str | None = None  # the proposed memory text, when there was one
    memory_type: str | None = None
    confidence: float | None = None
    importance: float | None = None
    detail: str = ""


@dataclass
class CalibrationSample:
    case_id: str
    label: str
    field: str  # confidence | importance
    expected_band: tuple[float, float]
    actual: float

    @property
    def inside(self) -> bool:
        return in_band(self.actual, self.expected_band)

    @property
    def deviation(self) -> float:
        return band_deviation(self.actual, self.expected_band)


@dataclass
class CaseResult:
    case_id: str
    requirement: str
    title: str
    proposed: list[dict[str, Any]]
    judgements: list[Judgement] = field(default_factory=list)
    calibration: list[CalibrationSample] = field(default_factory=list)
    type_hits: int = 0
    type_misses: int = 0
    error: str | None = None

    def count(self, kind: str) -> int:
        return sum(1 for j in self.judgements if j.kind == kind)

    @property
    def passed(self) -> bool:
        """A case passes when every expectation was met and nothing wrong was produced."""
        return (
            self.error is None
            and self.count("false_negative") == 0
            and self.count("false_positive") == 0
        )


def score_case(case: dict[str, Any], proposed: list[dict[str, Any]]) -> CaseResult:
    """Match one case's proposals against its labels.

    Assignment is greedy and one-to-one: each expectation consumes at most one
    proposal, and each proposal satisfies at most one expectation.
    """
    result = CaseResult(
        case_id=case["id"],
        requirement=case["requirement"],
        title=case["title"],
        proposed=proposed,
    )

    unclaimed = list(range(len(proposed)))

    # 1. Expectations first — a proposal that satisfies one is a true positive.
    for expectation in case.get("expected", []):
        hit_index = None
        for index in unclaimed:
            if matches(expectation["must_include"], proposed[index]["content"]):
                hit_index = index
                break

        if hit_index is None:
            result.judgements.append(
                Judgement(
                    kind="false_negative",
                    case_id=case["id"],
                    label=expectation["label"],
                    detail="no proposed memory matched this expectation",
                )
            )
            continue

        unclaimed.remove(hit_index)
        memory = proposed[hit_index]
        result.judgements.append(
            Judgement(
                kind="true_positive",
                case_id=case["id"],
                label=expectation["label"],
                content=memory["content"],
                memory_type=memory.get("memory_type"),
                confidence=memory.get("confidence"),
                importance=memory.get("importance"),
            )
        )

        # Type accuracy is tracked separately from precision/recall.
        if memory.get("memory_type") in expectation.get("memory_type", []):
            result.type_hits += 1
        else:
            result.type_misses += 1

        for field_name in ("confidence", "importance"):
            band = expectation.get(field_name)
            value = memory.get(field_name)
            if band is not None and value is not None:
                result.calibration.append(
                    CalibrationSample(
                        case_id=case["id"],
                        label=expectation["label"],
                        field=field_name,
                        expected_band=(band[0], band[1]),
                        actual=float(value),
                    )
                )

    # 2. Whatever is left is either explicitly forbidden, explicitly tolerated,
    #    or an unlabelled surplus. All but 'tolerated' count against precision.
    for index in unclaimed:
        memory = proposed[index]
        content = memory["content"]

        forbidden = next(
            (
                rule
                for rule in case.get("forbidden", [])
                if matches(rule["must_include"], content)
            ),
            None,
        )
        if forbidden is not None:
            result.judgements.append(
                Judgement(
                    kind="false_positive",
                    case_id=case["id"],
                    label=forbidden["label"],
                    content=content,
                    memory_type=memory.get("memory_type"),
                    confidence=memory.get("confidence"),
                    importance=memory.get("importance"),
                    detail="matched a forbidden pattern",
                )
            )
            continue

        permitted = next(
            (
                rule
                for rule in case.get("tolerated", [])
                if matches(rule["must_include"], content)
            ),
            None,
        )
        if permitted is not None:
            result.judgements.append(
                Judgement(
                    kind="tolerated",
                    case_id=case["id"],
                    label=permitted["label"],
                    content=content,
                    memory_type=memory.get("memory_type"),
                    confidence=memory.get("confidence"),
                    importance=memory.get("importance"),
                    detail="allowed but not required",
                )
            )
            continue

        result.judgements.append(
            Judgement(
                kind="false_positive",
                case_id=case["id"],
                label=None,
                content=content,
                memory_type=memory.get("memory_type"),
                confidence=memory.get("confidence"),
                importance=memory.get("importance"),
                detail="did not match any expectation, forbidden pattern, or tolerance",
            )
        )

    return result


@dataclass
class Totals:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    tolerated: int = 0
    type_hits: int = 0
    type_misses: int = 0
    cases_passed: int = 0
    cases_total: int = 0
    errors: int = 0

    @staticmethod
    def _ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    @property
    def precision(self) -> float | None:
        """Tolerated extractions excluded from the denominator."""
        return self._ratio(
            self.true_positives, self.true_positives + self.false_positives
        )

    @property
    def strict_precision(self) -> float | None:
        """Tolerated extractions counted as false positives."""
        return self._ratio(
            self.true_positives,
            self.true_positives + self.false_positives + self.tolerated,
        )

    @property
    def recall(self) -> float | None:
        return self._ratio(
            self.true_positives, self.true_positives + self.false_negatives
        )

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or (p + r) == 0:
            return None
        return 2 * p * r / (p + r)

    @property
    def type_accuracy(self) -> float | None:
        return self._ratio(self.type_hits, self.type_hits + self.type_misses)


def aggregate(results: list[CaseResult]) -> Totals:
    totals = Totals(cases_total=len(results))
    for result in results:
        totals.true_positives += result.count("true_positive")
        totals.false_positives += result.count("false_positive")
        totals.false_negatives += result.count("false_negative")
        totals.tolerated += result.count("tolerated")
        totals.type_hits += result.type_hits
        totals.type_misses += result.type_misses
        totals.cases_passed += 1 if result.passed else 0
        totals.errors += 1 if result.error else 0
    return totals


def calibration_summary(
    results: list[CaseResult], field_name: str
) -> dict[str, Any]:
    """In-band rate and average miss distance for confidence or importance."""
    samples = [s for r in results for s in r.calibration if s.field == field_name]
    if not samples:
        return {"samples": 0}

    inside = [s for s in samples if s.inside]
    outside = [s for s in samples if not s.inside]
    return {
        "samples": len(samples),
        "in_band": len(inside),
        "in_band_rate": len(inside) / len(samples),
        "mean_deviation": sum(s.deviation for s in samples) / len(samples),
        "mean_deviation_when_outside": (
            sum(s.deviation for s in outside) / len(outside) if outside else 0.0
        ),
        "over_band": sum(1 for s in outside if s.actual > s.expected_band[1]),
        "under_band": sum(1 for s in outside if s.actual < s.expected_band[0]),
        "worst": (
            max(
                (
                    {
                        "case_id": s.case_id,
                        "label": s.label,
                        "expected_band": list(s.expected_band),
                        "actual": s.actual,
                        "deviation": s.deviation,
                    }
                    for s in samples
                ),
                key=lambda d: d["deviation"],
            )
            if outside
            else None
        ),
    }
