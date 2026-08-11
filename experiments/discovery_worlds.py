"""TRUTH. The generators, and the only place the hidden relationships exist.

Nothing under `echo/` imports this module, and a test enforces that. The
discovery system receives an `ObservationSet` — six columns and an outcome — and
has no path back to the process that produced it. The evaluator (this file, and
the report that reads it) may know everything. ECHO may not.

The hidden relationships are written here in plain arithmetic rather than in
ECHO's hypothesis language. That is deliberate. If the generator were built out
of the same primitives the searcher enumerates, it would be fair to ask whether
the answer had been handed over in the shape of the question. It is written the
way any other simulation would be written, and whether the relationship happens
to be expressible in ECHO's language is a fact about the language, not a favour.

The variables are named X1…X6 and nothing else. There is no `X_the_important_one`.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Sequence

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo.observations import ObservationSet  # noqa: E402

#: How the six observed series behave. Autocorrelation is what makes lags and
#: moving averages meaningful rather than noise; the values differ per variable
#: so that no single one is obviously special.
AR_COEFFICIENTS = (0.55, 0.80, 0.65, 0.35, 0.70, 0.20)
SCALES = (1.0, 1.4, 0.8, 1.2, 1.0, 1.6)

ROWS = 960
TRAIN_STOP = 480
VAL_A_STOP = 640
VAL_B_STOP = 800


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _standardise(series: Sequence[float]) -> list[float]:
    n = len(series)
    mean = sum(series) / n
    variance = sum((v - mean) ** 2 for v in series) / n
    sd = math.sqrt(variance) if variance > 0 else 1.0
    return [(v - mean) / sd for v in series]


@dataclass
class World:
    """A generating process. The `signal` function is the secret."""

    id: str
    name: str
    #: What ECHO is told about the data. Never mentions the relationship.
    observable_description: str
    #: What is actually going on. Report only; never reaches the search.
    hidden_process: str
    #: True if there is a genuine, persistent relationship to be found.
    has_relationship: bool
    seed: int
    #: Maps the six generated columns to an unstandardised signal per row.
    signal: Callable[[list[list[float]], int], float]
    beta: float = 2.0
    intercept: float = 0.0
    rows: int = ROWS
    #: Rows on which the signal is switched off. Used by the trap world.
    active_until: int | None = None
    notes: str = ""

    def generate(self) -> ObservationSet:
        rng = random.Random(self.seed)
        columns: list[list[float]] = []
        for index, (phi, scale) in enumerate(zip(AR_COEFFICIENTS, SCALES)):
            series: list[float] = []
            value = 0.0
            innovation_sd = scale * math.sqrt(1.0 - phi * phi)
            for _ in range(self.rows):
                value = phi * value + rng.gauss(0.0, innovation_sd)
                series.append(value)
            columns.append(series)

        raw = [self.signal(columns, t) for t in range(self.rows)]
        z = _standardise(raw)

        outcomes: list[bool] = []
        for t in range(self.rows):
            active = self.active_until is None or t < self.active_until
            logit = self.intercept + (self.beta * z[t] if active else 0.0)
            outcomes.append(rng.random() < _sigmoid(logit))

        return ObservationSet(
            id=self.id,
            variable_names=("X1", "X2", "X3", "X4", "X5", "X6"),
            columns=tuple(tuple(column) for column in columns),
            outcomes=tuple(outcomes),
        )


# --------------------------------------------------------------- the secrets
#
# Each of these is a hidden generating process. None of them is written in
# ECHO's hypothesis language, and none of their names appears in any prompt,
# rule, threshold, or comment that the discovery system can reach.


def _lagged_third(columns: list[list[float]], t: int) -> float:
    """X3, two steps back."""
    return columns[2][t - 2] if t >= 2 else 0.0


def _change_times_lag(columns: list[list[float]], t: int) -> float:
    """How fast X2 is moving, scaled by where X4 was three steps ago.

    Neither part carries any signal by itself. A search that ranks single
    variables and then combines the winners cannot find this; only trying the
    combinations can.
    """
    if t < 3:
        return 0.0
    change = columns[1][t] - columns[1][t - 1]
    return change * columns[3][t - 3]


def _nothing(columns: list[list[float]], t: int) -> float:
    return 0.0


def _fifth(columns: list[list[float]], t: int) -> float:
    return columns[4][t]


WORLDS: list[World] = [
    World(
        id="DISC-A-lagged",
        name="World A — a relationship exists",
        observable_description=(
            "Six observed series and a binary outcome. The series are "
            "autocorrelated to differing degrees. At each step the six values "
            "are revealed first and the outcome second, so anything observed at "
            "or before time t is fair game for predicting the outcome at t."
        ),
        hidden_process=(
            "P(outcome) = sigmoid(2.0 · z(X3 at t-2)). One variable, one lag, "
            "nothing else. X3's contemporaneous value is not used; a system "
            "that only looks at the present will not find this."
        ),
        has_relationship=True,
        seed=515151,
        signal=_lagged_third,
        beta=2.0,
    ),
    World(
        id="DISC-B-interaction",
        name="World B — a relationship exists, and it is not obvious",
        observable_description=(
            "Six observed series and a binary outcome, generated the same way "
            "as the others. Nothing distinguishes any variable by name, scale, "
            "or position."
        ),
        hidden_process=(
            "P(outcome) = sigmoid(2.4 · z((X2 at t − X2 at t-1) · (X4 at t-3))). "
            "An interaction between how fast one series is moving and where a "
            "different series was three steps earlier. Neither factor predicts "
            "the outcome on its own: their individual correlations with it are "
            "approximately zero by construction."
        ),
        has_relationship=True,
        seed=828282,
        signal=_change_times_lag,
        beta=2.4,
    ),
    World(
        id="DISC-N-null",
        name="World N — nothing to find",
        observable_description=(
            "Six observed series and a binary outcome, generated the same way "
            "as the others."
        ),
        hidden_process=(
            "The outcome is a coin flip with P = sigmoid(0.20) ≈ 0.55, "
            "independent of every observed variable at every lag. There is no "
            "relationship of any kind. The correct result is NO RELIABLE "
            "DISCOVERY; anything else is a false discovery."
        ),
        has_relationship=False,
        seed=373737,
        signal=_nothing,
        beta=0.0,
        intercept=0.20,
    ),
    World(
        id="DISC-T-trap",
        name="World T — a relationship that stops",
        observable_description=(
            "Six observed series and a binary outcome, generated the same way "
            "as the others."
        ),
        hidden_process=(
            "P(outcome) = sigmoid(3.0 · z(X5 at t)) for the first 480 rows, and "
            "sigmoid(0) = 0.5 for every row after that. A strong, simple, "
            "contemporaneous relationship that holds throughout training and "
            "then is not there any more. Fitting it beautifully is the wrong "
            "answer; the correct result is NO RELIABLE DISCOVERY."
        ),
        has_relationship=False,
        seed=646464,
        signal=_fifth,
        beta=3.0,
        active_until=TRAIN_STOP,
        notes="the overfitting control",
    ),
]


def by_id(world_id: str) -> World:
    for world in WORLDS:
        if world.id == world_id:
            return world
    raise KeyError(f"no world {world_id!r}")
