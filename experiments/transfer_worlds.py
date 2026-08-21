"""TRUTH for the transfer experiment. Source and targets, and nothing shared.

The same wall as ECHO 5: nothing under `echo/` imports this module, and a test
enforces it. What is new here is a second wall, between the *source* and the
*target*. ECHO discovers something in the source world, and the only thing
allowed across is an abstract `StructuralPattern` — never a variable name, never
the winning expression, never the source data or outcomes, and never this file.

The worlds are deliberately unalike on every axis that is not the point:

| | names | scale | noise | base rate | AR |
| --- | --- | --- | --- | --- | --- |
| source | `X1…X6` | ~1 | none | ~0.5 | mixed |
| targets | `Z1…Z6` | 0.02 → 10⁶ | 0–35% | 0.30–0.72 | different |

The variable names carry no meaning. There is no `temperature`, no `pressure`,
no metadata describing what a series "is", because in this experiment those
concepts do not exist. `Z3` is the third column and that is all it is.

Which target shares the source's structure, which shares half of it, and which
shares none is recorded here for the evaluator and reaches ECHO nowhere.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo.observations import ObservationSet  # noqa: E402

ROWS = 960
TRAIN_STOP = 480
VAL_A_STOP = 640
VAL_B_STOP = 800


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _standardise(series: list[float]) -> list[float]:
    n = len(series)
    mean = sum(series) / n
    variance = sum((v - mean) ** 2 for v in series) / n
    sd = math.sqrt(variance) if variance > 0 else 1.0
    return [(v - mean) / sd for v in series]


@dataclass
class TransferWorld:
    id: str
    name: str
    variable_names: tuple[str, ...]
    #: What ECHO is told. Shape, scale and noise only — never the relationship.
    observable_description: str
    #: The evaluator's copy of the secret.
    hidden_process: str
    #: Does this world share the source's structure? Report only.
    shares_structure: str  # "full" | "partial" | "none"
    seed: int
    ar: tuple[float, ...]
    scales: tuple[float, ...]
    offsets: tuple[float, ...]
    signal: Callable[[list[list[float]], int], float]
    beta: float
    intercept: float = 0.0
    observation_noise: float = 0.0
    rows: int = ROWS
    notes: str = ""

    def generate(self) -> ObservationSet:
        rng = random.Random(self.seed)
        columns: list[list[float]] = []
        for phi, scale, offset in zip(self.ar, self.scales, self.offsets):
            series: list[float] = []
            value = 0.0
            innovation_sd = math.sqrt(max(1.0 - phi * phi, 1e-9))
            for _ in range(self.rows):
                value = phi * value + rng.gauss(0.0, innovation_sd)
                series.append(offset + scale * value)
            columns.append(series)

        raw = [self.signal(columns, t) for t in range(self.rows)]
        z = _standardise(raw)

        outcomes: list[bool] = []
        for t in range(self.rows):
            probability = _sigmoid(self.intercept + self.beta * z[t])
            outcome = rng.random() < probability
            if self.observation_noise and rng.random() < self.observation_noise:
                outcome = not outcome
            outcomes.append(outcome)

        return ObservationSet(
            id=self.id,
            variable_names=self.variable_names,
            columns=tuple(tuple(column) for column in columns),
            outcomes=tuple(outcomes),
        )


# --------------------------------------------------------------- the secrets
#
# Written in plain arithmetic, not in ECHO's primitives. Which column each one
# reaches for differs between worlds, and no world's signal is reachable from
# any other.


def _source_signal(columns: list[list[float]], t: int) -> float:
    """Source: how fast the second series moves, times where the fourth was."""
    if t < 3:
        return 0.0
    return (columns[1][t] - columns[1][t - 1]) * columns[3][t - 3]


def _target_a_signal(columns: list[list[float]], t: int) -> float:
    if t < 3:
        return 0.0
    return (columns[4][t] - columns[4][t - 1]) * columns[0][t - 3]


def _target_b_signal(columns: list[list[float]], t: int) -> float:
    if t < 2:
        return 0.0
    return (columns[1][t] - columns[1][t - 1]) * columns[5][t - 2]


def _target_c_signal(columns: list[list[float]], t: int) -> float:
    if t < 3:
        return 0.0
    return (columns[2][t] - columns[2][t - 1]) * columns[4][t - 3]


def _target_p_signal(columns: list[list[float]], t: int) -> float:
    """Only the first half of the source's shape survives here."""
    if t < 1:
        return 0.0
    return columns[3][t] - columns[3][t - 1]


def _target_n_signal(columns: list[list[float]], t: int) -> float:
    """A real relationship of an entirely different shape.

    Squaring is the point. For a zero-mean series, the outcome depends on how
    far the value is from centre rather than on which side of centre it is, so
    a linear read of that series — or of any delay of it, or of its change —
    carries essentially nothing. The transferred shape has no purchase here.
    """
    return columns[1][t] * columns[1][t]


def _target_f_signal(columns: list[list[float]], t: int) -> float:
    return 0.0


SOURCE = TransferWorld(
    id="SRC-source",
    name="Source world",
    variable_names=("X1", "X2", "X3", "X4", "X5", "X6"),
    observable_description=(
        "Six autocorrelated series on a scale of roughly ±1, and a binary "
        "outcome that occurs about half the time. Observations are clean."
    ),
    hidden_process=(
        "P(outcome) = sigmoid(2.4 · z((X2 at t − X2 at t-1) · (X4 at t-3)))."
    ),
    shares_structure="source",
    seed=1123581,
    ar=(0.55, 0.80, 0.65, 0.35, 0.70, 0.20),
    scales=(1.0, 1.4, 0.8, 1.2, 1.0, 1.6),
    offsets=(0.0,) * 6,
    signal=_source_signal,
    beta=2.4,
)


TARGETS: list[TransferWorld] = [
    TransferWorld(
        id="TGT-A",
        name="Target A — different names, scales and base rate",
        variable_names=("Z1", "Z2", "Z3", "Z4", "Z5", "Z6"),
        observable_description=(
            "Six autocorrelated series with an offset baseline and mixed "
            "magnitudes, and a binary outcome that occurs about seven times in "
            "ten. Observations are clean."
        ),
        hidden_process=(
            "P(outcome) = sigmoid(0.9 + 2.2 · z((Z5 at t − Z5 at t-1) · "
            "(Z1 at t-3))). Structurally the same shape as the source, through "
            "different columns, at a different scale, with a different base rate."
        ),
        shares_structure="full",
        seed=2244668,
        ar=(0.30, 0.75, 0.50, 0.85, 0.45, 0.60),
        scales=(18.0, 3.5, 42.0, 0.6, 7.5, 25.0),
        offsets=(0.0, -20.0, 5.0, 40.0, 250.0, -8.0),
        signal=_target_a_signal,
        beta=2.2,
        intercept=0.9,
    ),
    TransferWorld(
        id="TGT-B",
        name="Target B — same shape, a different delay, and noisy outcomes",
        variable_names=("Z1", "Z2", "Z3", "Z4", "Z5", "Z6"),
        observable_description=(
            "Six autocorrelated series on small magnitudes, and a binary "
            "outcome occurring about a third of the time. A share of recorded "
            "outcomes are wrong."
        ),
        hidden_process=(
            "P(outcome) = sigmoid(-0.7 + 2.6 · z((Z2 at t − Z2 at t-1) · "
            "(Z6 at t-2))), then 12% of outcomes are recorded flipped. The "
            "delay is two steps, not three: a pattern that had memorised the "
            "source's exact lag cannot fire here."
        ),
        shares_structure="full",
        seed=3355779,
        ar=(0.65, 0.40, 0.20, 0.55, 0.80, 0.25),
        scales=(0.02, 0.05, 0.11, 0.03, 0.07, 0.04),
        offsets=(0.0, 0.5, -0.2, 0.0, 1.0, 0.0),
        signal=_target_b_signal,
        beta=2.6,
        intercept=-0.7,
        observation_noise=0.12,
    ),
    TransferWorld(
        id="TGT-C",
        name="Target C — the same shape, six orders of magnitude away",
        variable_names=("Z1", "Z2", "Z3", "Z4", "Z5", "Z6"),
        observable_description=(
            "Six autocorrelated series whose values run from roughly ten "
            "thousand to a million, and a binary outcome occurring a little "
            "under half the time."
        ),
        hidden_process=(
            "P(outcome) = sigmoid(-0.15 + 2.3 · z((Z3 at t − Z3 at t-1) · "
            "(Z5 at t-3))). The same shape as the source with values around "
            "10^5, so the product term reaches roughly 10^10. Nothing is "
            "normalised for this specifically; the standardisation inside the "
            "logistic fit is the same code every condition uses."
        ),
        shares_structure="full",
        seed=4466880,
        ar=(0.50, 0.35, 0.70, 0.45, 0.60, 0.30),
        scales=(85_000.0, 120_000.0, 240_000.0, 60_000.0, 310_000.0, 95_000.0),
        offsets=(500_000.0, 250_000.0, 800_000.0, 60_000.0, 0.0, 12_000.0),
        signal=_target_c_signal,
        beta=2.3,
        intercept=-0.15,
    ),
    TransferWorld(
        id="TGT-P",
        name="Target P — half the shape holds",
        variable_names=("Z1", "Z2", "Z3", "Z4", "Z5", "Z6"),
        observable_description=(
            "Six autocorrelated series at moderate magnitudes, and a binary "
            "outcome occurring a little over half the time."
        ),
        hidden_process=(
            "P(outcome) = sigmoid(0.2 + 2.5 · z(Z4 at t − Z4 at t-1)). The "
            "first half of the source's shape — a step-to-step change — is "
            "predictive. The second half, a delayed second series multiplied "
            "in, is not: it only adds noise. The correct response is to keep "
            "the component that works and weaken the rest."
        ),
        shares_structure="partial",
        seed=5577991,
        ar=(0.45, 0.60, 0.35, 0.50, 0.70, 0.40),
        scales=(2.5, 6.0, 1.2, 3.3, 0.9, 4.4),
        offsets=(10.0, 0.0, -5.0, 2.0, 0.0, 30.0),
        signal=_target_p_signal,
        beta=2.5,
        intercept=0.2,
    ),
    TransferWorld(
        id="TGT-N",
        name="Target N — a real relationship, of the wrong shape",
        variable_names=("Z1", "Z2", "Z3", "Z4", "Z5", "Z6"),
        observable_description=(
            "Six autocorrelated series at moderate magnitudes, and a binary "
            "outcome occurring about half the time."
        ),
        hidden_process=(
            "P(outcome) = sigmoid(2.4 · z((Z2 at t)^2)). There is a strong "
            "relationship here, and an exhaustive search can express it — but "
            "it is a squared level, not a product of a change and a delay. "
            "Because the series is centred, a linear read of Z2, of any delay "
            "of it, or of its change carries almost nothing, so no grounding "
            "of the transferred shape has any purchase. The pattern should be "
            "rejected; a system that trusts prior knowledge regardless would "
            "carry it anyway."
        ),
        shares_structure="none",
        seed=6688113,
        ar=(0.75, 0.30, 0.55, 0.40, 0.65, 0.50),
        scales=(4.0, 1.1, 2.8, 5.5, 0.7, 3.2),
        offsets=(-3.0, 0.0, 0.0, 7.0, 0.0, -1.0),
        signal=_target_n_signal,
        beta=2.4,
    ),
    TransferWorld(
        id="TGT-F",
        name="Target F — false analogy: the same statistics, no shared structure",
        variable_names=("Z1", "Z2", "Z3", "Z4", "Z5", "Z6"),
        observable_description=(
            "Six autocorrelated series with an offset baseline and mixed "
            "magnitudes, and a binary outcome that occurs about seven times in "
            "ten. Observations are clean."
        ),
        hidden_process=(
            "Deliberately built to look like Target A from the outside: the "
            "same autocorrelations, the same scales, the same offsets, the "
            "same base rate, the same description. The outcome is independent "
            "of every observed series at every lag. Superficial similarity is "
            "all there is, and a system that transfers on resemblance rather "
            "than on evidence will be caught here."
        ),
        shares_structure="none",
        seed=7799224,
        ar=(0.30, 0.75, 0.50, 0.85, 0.45, 0.60),
        scales=(18.0, 3.5, 42.0, 0.6, 7.5, 25.0),
        offsets=(0.0, -20.0, 5.0, 40.0, 250.0, -8.0),
        signal=_target_f_signal,
        beta=0.0,
        intercept=0.9,
        notes="statistically matched to TGT-A",
    ),
]


def by_id(world_id: str) -> TransferWorld:
    if world_id == SOURCE.id:
        return SOURCE
    for world in TARGETS:
        if world.id == world_id:
            return world
    raise KeyError(f"no transfer world {world_id!r}")
