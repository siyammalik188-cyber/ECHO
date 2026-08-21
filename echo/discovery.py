"""Hypothesis search: propose, screen, fit, rank, confirm, and only then look.

The shape of the thing:

```
TRAIN            VAL_A          VAL_B           TEST
-----------> ----------> ------------> ------------->
fit the link   rank the      confirm the    reported once,
               candidates    winner against after every
                             the base rate  decision is made
```

Four rules hold it together.

1. **Generation and validation are separate.** A `CandidateSource` returns
   expressions. It cannot return a score, because the search only ever reads
   `.expression` off whatever it is handed. If a language model is wired in
   later it can propose all it likes; the data decides what survives.
2. **Ranking never touches the confirmation block.** `VAL_A` picks the winner,
   `VAL_B` checks it. A block used to choose from thousands of candidates
   cannot also be the block that says the choice was real.
3. **One candidate is confirmed.** The single top-ranked hypothesis goes to
   `VAL_B`. If it fails, the answer is *no reliable discovery* — not "try the
   next one", which would make `VAL_B` a second selection set.
4. **The holdout is sealed until every decision is made.** Reading it closes
   the search permanently; see `ChronologicalSplit`.

Everything here is deterministic. The only seed is for the random baseline,
which exists to be beaten.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Protocol, Sequence

from . import calibration
from . import expressions as ex
from .expressions import Expr
from .hypothesis import Hypothesis, HypothesisStatus
from .observations import (
    TEST,
    TRAIN,
    VAL_A,
    VAL_B,
    Block,
    ChronologicalSplit,
    ObservationSet,
)

# ------------------------------------------------------------------- settings
#
# Pre-registered. Every one of these was fixed before the first experiment ran,
# and none was changed after any test-set number was seen.

#: How many screened candidates get a full fit and a ranking. The screen is
#: cheap and approximate; this is where the real work happens.
SCREEN_KEEP = 64

#: Brier penalty per unit of expression complexity. A candidate four points
#: more complicated than another must beat it by more than 0.008 Brier to win.
COMPLEXITY_PENALTY = 0.002

#: The winner must beat the frozen base rate on the confirmation block by at
#: least this much, in absolute Brier.
MIN_EDGE = 0.01

#: ...and the paired improvement must be this many standard errors from zero.
#: With thousands of candidates screened, a threshold that a fluke clears one
#: time in three is not a threshold.
MIN_T = 2.5

#: Guards for the logistic fit.
MAX_SLOPE = 10.0
RIDGE = 1e-4
FIT_ITERATIONS = 30
PROBABILITY_FLOOR = 1e-6


@dataclass(frozen=True)
class SearchConfig:
    screen_keep: int = SCREEN_KEEP
    complexity_penalty: float = COMPLEXITY_PENALTY
    min_edge: float = MIN_EDGE
    min_t: float = MIN_T
    max_lag: int = ex.MAX_LAG
    max_window: int = ex.MAX_WINDOW
    max_depth: int = ex.MAX_DEPTH
    max_variables: int = ex.MAX_VARIABLES
    max_operations: int = ex.MAX_OPERATIONS
    baseline_seed: int = 90210

    def to_dict(self) -> dict[str, Any]:
        return {
            "screen_keep": self.screen_keep,
            "complexity_penalty": self.complexity_penalty,
            "min_edge": self.min_edge,
            "min_t": self.min_t,
            "max_lag": self.max_lag,
            "max_window": self.max_window,
            "max_depth": self.max_depth,
            "max_variables": self.max_variables,
            "max_operations": self.max_operations,
            "baseline_seed": self.baseline_seed,
        }


# ------------------------------------------------------------------- the link


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass(frozen=True)
class Fit:
    """A one-dimensional logistic link from an expression's value to a probability.

    The expression says *what to look at*. This says *how much it matters*, and
    it is fitted on the training block alone — the standardisation constants
    included. Nothing downstream refits it, so a probability produced on the
    holdout uses only numbers derived from the earliest rows in the series.
    """

    mean: float
    sd: float
    intercept: float
    slope: float

    def probability(self, value: float) -> float:
        z = (value - self.mean) / self.sd
        p = _sigmoid(self.intercept + self.slope * z)
        return min(max(p, PROBABILITY_FLOOR), 1.0 - PROBABILITY_FLOOR)

    def to_dict(self) -> dict[str, float]:
        return {
            "mean": self.mean,
            "sd": self.sd,
            "intercept": self.intercept,
            "slope": self.slope,
        }


class DegenerateFit(ValueError):
    """The candidate has no variation to fit on, so it carries no information."""


def fit_logistic(values: Sequence[float], outcomes: Sequence[bool], block: Block) -> Fit:
    rows = list(block.indices())
    n = len(rows)
    if n < 20:
        raise DegenerateFit("not enough training rows to fit a link")

    mean = sum(values[t] for t in rows) / n
    variance = sum((values[t] - mean) ** 2 for t in rows) / n
    sd = math.sqrt(variance)
    if sd < 1e-12:
        raise DegenerateFit("candidate is constant over the training block")

    z = [(values[t] - mean) / sd for t in rows]
    y = [1.0 if outcomes[t] else 0.0 for t in rows]

    positives = sum(y)
    if positives == 0 or positives == n:
        raise DegenerateFit("training outcomes are all one value")

    base = positives / n
    intercept = math.log(base / (1.0 - base))
    slope = 0.0

    for _ in range(FIT_ITERATIONS):
        g0 = g1 = 0.0
        h00 = h01 = h11 = 0.0
        for zi, yi in zip(z, y):
            p = _sigmoid(intercept + slope * zi)
            residual = yi - p
            w = max(p * (1.0 - p), 1e-9)
            g0 += residual
            g1 += residual * zi
            h00 += w
            h01 += w * zi
            h11 += w * zi * zi
        g0 -= RIDGE * intercept
        g1 -= RIDGE * slope
        h00 += RIDGE
        h11 += RIDGE
        determinant = h00 * h11 - h01 * h01
        if abs(determinant) < 1e-12:
            break
        step0 = (h11 * g0 - h01 * g1) / determinant
        step1 = (h00 * g1 - h01 * g0) / determinant
        intercept += step0
        slope += step1
        slope = max(-MAX_SLOPE, min(MAX_SLOPE, slope))
        if abs(step0) < 1e-9 and abs(step1) < 1e-9:
            break

    return Fit(mean=mean, sd=sd, intercept=intercept, slope=slope)


def probabilities(fit: Fit, values: Sequence[float], block: Block) -> list[float]:
    return [fit.probability(values[t]) for t in block.indices()]


def score(
    fit: Fit, values: Sequence[float], outcomes: Sequence[bool], block: Block
) -> dict[str, Any]:
    predicted = probabilities(fit, values, block)
    truth = [outcomes[t] for t in block.indices()]
    summary = calibration.summarise(predicted, truth)
    summary["block"] = block.name
    return summary


# ------------------------------------------------------------------ screening


def screening_statistic(
    values: Sequence[float], outcomes: Sequence[bool], block: Block
) -> float:
    """|point-biserial correlation| between a candidate's value and the outcome.

    Cheap enough to run on every candidate in the space, and computed on the
    training block only. It decides which candidates are worth fitting; it does
    not decide anything else.
    """
    rows = list(block.indices())
    n = len(rows)
    if n == 0:
        return 0.0
    total = total_sq = positive_total = 0.0
    positives = 0
    for t in rows:
        v = values[t]
        total += v
        total_sq += v * v
        if outcomes[t]:
            positive_total += v
            positives += 1
    if positives == 0 or positives == n:
        return 0.0
    mean = total / n
    variance = total_sq / n - mean * mean
    if variance <= 1e-18:
        return 0.0
    mean_positive = positive_total / positives
    mean_negative = (total - positive_total) / (n - positives)
    proportion = positives / n
    r = (mean_positive - mean_negative) * math.sqrt(proportion * (1 - proportion))
    r /= math.sqrt(variance)
    return abs(r)


# ------------------------------------------------------------ candidate source


class CandidateSource(Protocol):
    """Proposes expressions. Note the return type: expressions, not verdicts.

    A source is free to be a clever heuristic, an enumerator, or eventually a
    language model. What it is not free to do is claim a hypothesis is any good
    — the search reads the expression and throws away everything else, then
    measures it against the data itself.
    """

    name: str

    def propose(self, observations: ObservationSet, split: ChronologicalSplit) -> Iterable[Any]:
        ...


def unary_terms(
    *,
    max_lag: int = ex.MAX_LAG,
    max_window: int = ex.MAX_WINDOW,
    variable_names: Sequence[str] = ex.VARIABLE_NAMES,
) -> list[Expr]:
    """Every single-variable term the language can express, in a fixed order."""
    terms: list[Expr] = []
    for variable in variable_names:
        terms.append(ex.Feature(variable))
        for steps in range(1, max_lag + 1):
            terms.append(ex.Lag(variable, steps))
        terms.append(ex.Change(variable))
        for window in range(2, max_window + 1):
            terms.append(ex.Mean(variable, window))
            terms.append(ex.Sum(variable, window))
    return terms


class EnumerationSource:
    """Exhaustive over the bounded space: every term, and every pair of terms.

    Exhaustive rather than greedy on purpose. A search that keeps only the
    terms that look promising on their own can never find an interaction whose
    parts are individually worthless — which is exactly the kind of
    relationship this challenge asks about. So every pair is tried, and no
    prior about which variables matter enters anywhere.

    `DIFFERENCE(a, b)` is enumerated once per unordered pair: the fitted link
    has a free sign, so `DIFFERENCE(b, a)` is the same hypothesis. `RATIO` is
    enumerated both ways round, because it is not.
    """

    name = "exhaustive-enumeration"

    def __init__(self, *, max_lag: int = ex.MAX_LAG, max_window: int = ex.MAX_WINDOW) -> None:
        self.max_lag = max_lag
        self.max_window = max_window

    def propose(
        self, observations: ObservationSet, split: ChronologicalSplit
    ) -> Iterable[Expr]:
        terms = unary_terms(
            max_lag=self.max_lag,
            max_window=self.max_window,
            variable_names=observations.variable_names,
        )
        yield from terms
        for i, left in enumerate(terms):
            for j, right in enumerate(terms):
                if j >= i:
                    yield ex.Product(left, right)
                if j > i:
                    yield ex.Difference(left, right)
                if j != i:
                    yield ex.Ratio(left, right)


def _as_expression(proposal: Any) -> Expr:
    """Take the expression and nothing else.

    A source that hands back a `Hypothesis` complete with glowing scores gets
    its expression extracted and its claims dropped on the floor. This function
    is the whole enforcement of 'the model may propose, the data disposes'.
    """
    if isinstance(proposal, Expr):
        return proposal
    expression = getattr(proposal, "expression", None)
    if isinstance(expression, Expr):
        return expression
    raise ex.ExpressionError(
        f"a candidate source returned {type(proposal).__name__}, which is not an expression"
    )


# ------------------------------------------------------------------- baselines


@dataclass(frozen=True)
class Baseline:
    name: str
    kind: str
    detail: str
    predictions: dict[str, list[float]]  # block name -> probabilities

    def score(self, observations: ObservationSet, block: Block) -> dict[str, Any]:
        truth = [observations.outcomes[t] for t in block.indices()]
        summary = calibration.summarise(self.predictions[block.name], truth)
        summary["block"] = block.name
        return summary


def build_baselines(
    observations: ObservationSet,
    split: ChronologicalSplit,
    blocks: Sequence[Block],
    config: SearchConfig,
) -> list[Baseline]:
    """Random, base rate, and each raw feature on its own.

    The base rate is frozen on the training block: it is what a system that has
    noticed nothing at all would predict, forever.
    """
    train = split.block(TRAIN)
    train_rows = list(train.indices())
    base_rate = sum(1 for t in train_rows if observations.outcomes[t]) / len(train_rows)

    rng = random.Random(config.baseline_seed)
    random_predictions = {
        block.name: [rng.random() for _ in block.indices()] for block in blocks
    }

    out = [
        Baseline(
            name="BASELINE 1 — random",
            kind="random",
            detail=f"a uniform draw in (0, 1) per row, seed {config.baseline_seed}",
            predictions=random_predictions,
        ),
        Baseline(
            name="BASELINE 2 — historical base rate",
            kind="base_rate",
            detail=f"P = {base_rate:.4f}, the frequency of the outcome on TRAIN, frozen",
            predictions={
                block.name: [base_rate] * len(block) for block in blocks
            },
        ),
    ]

    mapping = observations.as_mapping()
    for variable in observations.variable_names:
        expr = ex.Feature(variable)
        values = ex.evaluate(expr, mapping)
        try:
            fit = fit_logistic(values, observations.outcomes, train)
        except DegenerateFit:
            continue
        out.append(
            Baseline(
                name=f"BASELINE 3 — {expr.text()}",
                kind="raw_feature",
                detail=f"a logistic link on {variable} alone, fitted on TRAIN",
                predictions={
                    block.name: probabilities(fit, values, block) for block in blocks
                },
            )
        )
    return out


def base_rate_baseline(baselines: Sequence[Baseline]) -> Baseline:
    for baseline in baselines:
        if baseline.kind == "base_rate":
            return baseline
    raise KeyError("no base-rate baseline was built")


# --------------------------------------------------------------- confirmation


@dataclass(frozen=True)
class ConfirmationResult:
    """Did the winner beat the do-nothing baseline on a block it did not pick?"""

    block: str
    rows: int
    candidate_brier: float
    baseline_brier: float
    edge: float
    t_statistic: float
    min_edge: float
    min_t: float

    @property
    def passed(self) -> bool:
        return self.edge >= self.min_edge and self.t_statistic >= self.min_t

    def why(self) -> str:
        if self.passed:
            return (
                f"beat the frozen base rate on {self.block} by {self.edge:.4f} Brier "
                f"(t = {self.t_statistic:.2f}), clearing the pre-registered floor of "
                f"{self.min_edge:.4f} at t ≥ {self.min_t}"
            )
        parts = []
        if self.edge < self.min_edge:
            parts.append(
                f"edge {self.edge:+.4f} is below the required {self.min_edge:.4f}"
            )
        if self.t_statistic < self.min_t:
            parts.append(f"t = {self.t_statistic:.2f} is below the required {self.min_t}")
        return f"failed on {self.block}: " + "; and ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "block": self.block,
            "rows": self.rows,
            "candidate_brier": self.candidate_brier,
            "baseline_brier": self.baseline_brier,
            "edge": self.edge,
            "t_statistic": self.t_statistic,
            "min_edge": self.min_edge,
            "min_t": self.min_t,
            "passed": self.passed,
        }


def confirm(
    candidate: Sequence[float],
    baseline: Sequence[float],
    truth: Sequence[bool],
    *,
    block_name: str,
    min_edge: float,
    min_t: float,
) -> ConfirmationResult:
    """A paired comparison, row by row, against what a naive system would say."""
    differences = [
        (b - (1.0 if y else 0.0)) ** 2 - (c - (1.0 if y else 0.0)) ** 2
        for c, b, y in zip(candidate, baseline, truth)
    ]
    n = len(differences)
    mean = sum(differences) / n if n else 0.0
    if n > 1:
        variance = sum((d - mean) ** 2 for d in differences) / (n - 1)
        standard_error = math.sqrt(variance / n) if variance > 0 else 0.0
    else:
        standard_error = 0.0
    t = mean / standard_error if standard_error > 0 else 0.0

    candidate_brier = sum(
        (c - (1.0 if y else 0.0)) ** 2 for c, y in zip(candidate, truth)
    ) / n
    baseline_brier = sum(
        (b - (1.0 if y else 0.0)) ** 2 for b, y in zip(baseline, truth)
    ) / n
    return ConfirmationResult(
        block=block_name,
        rows=n,
        candidate_brier=candidate_brier,
        baseline_brier=baseline_brier,
        edge=mean,
        t_statistic=t,
        min_edge=min_edge,
        min_t=min_t,
    )


# ---------------------------------------------------------------- the search


@dataclass
class Ranked:
    """A fully evaluated candidate, with everything the ranking used."""

    hypothesis: Hypothesis
    fit: Fit
    values: list[float]
    train_score: dict[str, Any]
    val_a_score: dict[str, Any]
    penalised: float

    @property
    def raw(self) -> float:
        return self.val_a_score["brier"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "expression": self.hypothesis.expression.text(),
            "complexity": self.hypothesis.complexity,
            "screening_statistic": self.hypothesis.screening_statistic,
            "train_brier": self.train_score["brier"],
            "raw_val_a_brier": self.raw,
            "penalised_val_a_brier": self.penalised,
            "fit": self.fit.to_dict(),
        }


@dataclass
class SearchOutcome:
    observations_id: str
    config: SearchConfig
    source_name: str
    proposed: int
    screened: int
    fitted: int
    rejected_degenerate: int
    ranking: list[Ranked] = field(default_factory=list)
    winner: Ranked | None = None
    confirmation: ConfirmationResult | None = None
    promoted: Hypothesis | None = None
    rejected: list[Hypothesis] = field(default_factory=list)
    best_by_training: Ranked | None = None
    baselines: list[Baseline] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def discovered(self) -> bool:
        return self.promoted is not None

    def alternatives(self, limit: int = 5) -> list[Ranked]:
        return [r for r in self.ranking[: limit + 1] if r is not self.winner][:limit]


class DiscoverySearch:
    """Runs one search over one observation set. Deterministic end to end."""

    def __init__(
        self,
        observations: ObservationSet,
        split: ChronologicalSplit,
        *,
        config: SearchConfig | None = None,
        source: CandidateSource | None = None,
    ) -> None:
        self.observations = observations
        self.split = split
        self.config = config or SearchConfig()
        self.source = source or EnumerationSource(
            max_lag=self.config.max_lag, max_window=self.config.max_window
        )

    # ------------------------------------------------------------------- run

    def run(self, *, tick_start: int = 0) -> SearchOutcome:
        split = self.split
        split.require_open_search("running a hypothesis search")

        started = time.perf_counter()
        observations = self.observations
        mapping = observations.as_mapping()
        outcomes = observations.outcomes
        train = split.block(TRAIN)
        val_a = split.block(VAL_A)
        val_b = split.block(VAL_B)

        # -- propose and screen ------------------------------------------
        #
        # Screening reads TRAIN and nothing else. Note what is absent: no
        # ordering by variable name, no prior on which features matter, no
        # short-list of "interesting" combinations.
        scored: list[tuple[float, str, Expr]] = []
        proposed = 0
        for proposal in self.source.propose(observations, split):
            expression = _as_expression(proposal)
            ex.validate(
                expression,
                max_depth=self.config.max_depth,
                max_variables=self.config.max_variables,
                max_operations=self.config.max_operations,
            )
            proposed += 1
            values = ex.evaluate(expression, mapping)
            statistic = screening_statistic(values, outcomes, train)
            scored.append((statistic, expression.text(), expression))

        # Sorted by strength, then by printed form, so ties break the same way
        # on every machine and every run.
        scored.sort(key=lambda item: (-item[0], item[1]))
        keep = scored[: self.config.screen_keep]

        # -- fit and rank -------------------------------------------------
        ranking: list[Ranked] = []
        rejected: list[Hypothesis] = []
        degenerate = 0
        tick = tick_start
        for statistic, _text, expression in keep:
            tick += 1
            hypothesis = Hypothesis.propose(
                expression,
                created_at=tick,
                evidence_window=(train.start, train.stop),
                discovery_reason=(
                    f"screened at |r| = {statistic:.4f} on {train.name} "
                    f"({self.source.name})"
                ),
                screening_statistic=statistic,
            )
            values = ex.evaluate(expression, mapping)
            try:
                fit = fit_logistic(values, outcomes, train)
            except DegenerateFit as exc:
                degenerate += 1
                rejected.append(
                    hypothesis.with_status(HypothesisStatus.REJECTED, f"degenerate: {exc}")
                )
                continue

            train_score = score(fit, values, outcomes, train)
            val_a_score = score(fit, values, outcomes, val_a)
            penalised = (
                val_a_score["brier"] + self.config.complexity_penalty * hypothesis.complexity
            )
            hypothesis = hypothesis.with_status(
                HypothesisStatus.TESTING, "fitted on TRAIN, ranked on VAL_A"
            ).with_scores(
                training=train_score,
                validation=val_a_score,
                penalised=penalised,
            )
            ranking.append(
                Ranked(
                    hypothesis=hypothesis,
                    fit=fit,
                    values=values,
                    train_score=train_score,
                    val_a_score=val_a_score,
                    penalised=penalised,
                )
            )

        # Lower penalised score wins; the simpler expression wins a tie; the
        # printed form breaks anything still level.
        ranking.sort(
            key=lambda r: (
                r.penalised,
                r.hypothesis.complexity,
                r.hypothesis.expression.text(),
            )
        )

        baselines = build_baselines(
            observations, split, (train, val_a, val_b), self.config
        )
        outcome = SearchOutcome(
            observations_id=observations.id,
            config=self.config,
            source_name=self.source.name,
            proposed=proposed,
            screened=len(scored),
            fitted=len(ranking),
            rejected_degenerate=degenerate,
            ranking=ranking,
            rejected=rejected,
            baselines=baselines,
        )
        if ranking:
            outcome.best_by_training = min(ranking, key=lambda r: r.train_score["brier"])

        # -- confirm exactly one candidate --------------------------------
        if not ranking:
            outcome.seconds = time.perf_counter() - started
            return outcome

        winner = ranking[0]
        outcome.winner = winner
        base = base_rate_baseline(baselines)
        truth = [outcomes[t] for t in val_b.indices()]
        confirmation = confirm(
            probabilities(winner.fit, winner.values, val_b),
            base.predictions[val_b.name],
            truth,
            block_name=val_b.name,
            min_edge=self.config.min_edge,
            min_t=self.config.min_t,
        )
        outcome.confirmation = confirmation

        if confirmation.passed:
            validated = winner.hypothesis.with_status(
                HypothesisStatus.VALIDATED, confirmation.why()
            )
            outcome.promoted = validated.with_status(
                HypothesisStatus.ACTIVE,
                "promoted on TRAIN/VAL_A/VAL_B evidence, before the holdout was read",
            )
            winner.hypothesis = outcome.promoted
        else:
            rejected_winner = winner.hypothesis.with_status(
                HypothesisStatus.REJECTED, confirmation.why()
            )
            winner.hypothesis = rejected_winner
            outcome.rejected.append(rejected_winner)

        outcome.seconds = time.perf_counter() - started
        return outcome


# ------------------------------------------------------------ final holdout


@dataclass(frozen=True)
class HoldoutResult:
    hypothesis: Hypothesis | None
    test_score: dict[str, Any] | None
    baseline_scores: list[tuple[str, dict[str, Any]]]
    survived: bool | None
    overfitting_control: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis": self.hypothesis.to_dict() if self.hypothesis else None,
            "test_score": self.test_score,
            "baselines": [
                {"name": name, "score": score} for name, score in self.baseline_scores
            ],
            "survived": self.survived,
            "overfitting_control": self.overfitting_control,
        }


def evaluate_holdout(
    outcome: SearchOutcome,
    observations: ObservationSet,
    split: ChronologicalSplit,
) -> HoldoutResult:
    """Open the sealed block, once, and report what is there.

    Called only after promotion has already been decided. Whatever this returns
    cannot change the decision — it can only describe how well the decision
    turned out, which is the only honest thing an out-of-sample number is for.
    """
    test = split.unlock_holdout()
    outcomes = observations.outcomes
    truth = [outcomes[t] for t in test.indices()]

    baseline_scores: list[tuple[str, dict[str, Any]]] = []
    mapping = observations.as_mapping()
    train = split.block(TRAIN)
    rng = random.Random(outcome.config.baseline_seed + 1)
    train_rows = list(train.indices())
    base_rate = sum(1 for t in train_rows if outcomes[t]) / len(train_rows)

    baseline_scores.append(
        (
            "BASELINE 1 — random",
            calibration.summarise([rng.random() for _ in truth], truth),
        )
    )
    baseline_scores.append(
        (
            "BASELINE 2 — historical base rate",
            calibration.summarise([base_rate] * len(truth), truth),
        )
    )
    for variable in observations.variable_names:
        expr = ex.Feature(variable)
        values = ex.evaluate(expr, mapping)
        try:
            fit = fit_logistic(values, outcomes, train)
        except DegenerateFit:
            continue
        baseline_scores.append(
            (
                f"BASELINE 3 — {expr.text()}",
                calibration.summarise(probabilities(fit, values, test), truth),
            )
        )

    # The overfitting control: whatever fitted TRAIN best, scored where it
    # cannot have memorised anything. Recorded whether or not it was promoted.
    control: dict[str, Any] | None = None
    if outcome.best_by_training is not None:
        best = outcome.best_by_training
        control_test = score(best.fit, best.values, outcomes, test)
        control = {
            "expression": best.hypothesis.expression.text(),
            "complexity": best.hypothesis.complexity,
            "train_brier": best.train_score["brier"],
            "val_a_brier": best.val_a_score["brier"],
            "test_brier": control_test["brier"],
            "status": (
                best.hypothesis.status.value
                if best.hypothesis is not (outcome.winner.hypothesis if outcome.winner else None)
                else best.hypothesis.status.value
            ),
            "was_promoted": bool(
                outcome.promoted
                and outcome.promoted.hypothesis_id == best.hypothesis.hypothesis_id
            ),
        }

    if outcome.promoted is None or outcome.winner is None:
        return HoldoutResult(None, None, baseline_scores, None, control)

    winner = outcome.winner
    test_score = score(winner.fit, winner.values, outcomes, test)
    base_test = dict(baseline_scores[1][1])
    survived = test_score["brier"] < base_test["brier"]
    final = outcome.promoted.with_scores(
        test=test_score,
        reason="holdout read once, after promotion",
    )
    if not survived:
        final = final.with_status(
            HypothesisStatus.RETIRED,
            f"did not beat the base rate on the holdout "
            f"({test_score['brier']:.4f} vs {base_test['brier']:.4f})",
        )
    outcome.promoted = final
    return HoldoutResult(final, test_score, baseline_scores, survived, control)
