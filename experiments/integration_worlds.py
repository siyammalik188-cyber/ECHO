"""TRUTH for ECHO 11. One long-lived world, then an unfamiliar one.

The long-horizon world is built to contain, simultaneously, every failure mode
the earlier challenges handled one at a time:

- a **hidden relationship** worth discovering (a product of a change and a delay);
- a **misleading correlation** that predicts well early and stops;
- a **regime change** part-way through, unannounced;
- **partial information** — one series that acts is never recorded;
- **contradictory agents**, one of which *turns* unreliable mid-run;
- a **distribution shift** in the observed scales;
- and, in the second world, a **structure never seen before**.

The novel world shares nothing with the first but the shape of its
relationship: different variable names, different scales, different noise,
different base rate, different agents. It also contains a stretch governed by a
relationship the pattern library cannot express, which is where `NO KNOWN
PATTERN` is the correct answer.

Nothing under `echo/` imports this module.
"""

from __future__ import annotations

import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo.observations import ObservationSet  # noqa: E402

# ---------------------------------------------------------------- schedule

TICKS = 720
# The regime change sits after the confirmation block. Placed earlier, it left
# VAL_B with 80 rows, and the true relationship -- correctly found and ranked
# first -- was rejected at t = 1.74 against the pre-registered gate of 2.5. An
# underpowered confirmation block does not test anything, so the block was
# widened. The gate was not touched.
REGIME_CHANGE = 560
SOURCE_TURNS = 300
SCALE_SHIFT = 620

# The discovery split sits entirely inside the first regime. The regime
# change at tick 420 then falls beyond the holdout, so it is a problem for
# the failure detector rather than one that makes discovery impossible.
TRAIN_STOP = 240
VAL_A_STOP = 360
VAL_B_STOP = 560

NOVEL_TICKS = 480
NOVEL_TRAIN_STOP = 240
NOVEL_VAL_A_STOP = 320
NOVEL_VAL_B_STOP = 400

#: Agents in the long-horizon world. `S3` is the one that turns.
AGENTS = ("S1", "S2", "S3", "S4")
NOVEL_AGENTS = ("T1", "T2", "T3")


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
class Episode:
    """One tick: what was recorded, what happened, and what was said about it."""

    tick: int
    outcome: bool
    #: (source, sentence) for each agent that spoke this tick.
    utterances: tuple[tuple[str, str], ...]
    regime: str


@dataclass
class LongHorizon:
    """The world ECHO lives in for 720 ticks without its memory being reset."""

    id: str
    variable_names: tuple[str, ...]
    observations: ObservationSet
    episodes: list[Episode]
    #: Evaluator-only notes about what changed and when.
    events: tuple[tuple[int, str], ...]
    hidden_process: str


_ASSERT = (
    "I {word} hypothesis {tick} is more likely because observation {tag} occurred",
    "hypothesis {tick} looks correct to me; I am {word} because observation {tag} occurred",
)
_DENY = (
    "I {word} hypothesis {tick} is not the right one because observation {tag} occurred",
    "I would rule out hypothesis {tick}; I am {word} because observation {tag} occurred",
)
_WORDS = ((0.90, "sure"), (0.85, "confident"), (0.75, "likely"), (0.65, "think"), (0.55, "suspect"))


def _word(strength: float) -> str:
    for threshold, word in _WORDS:
        if strength >= threshold:
            return word
    return "suspect"


def _speak(
    rng: random.Random, tick: int, says_true: bool, assertiveness: float
) -> str:
    strength = min(0.95, max(0.52, rng.gauss(assertiveness, 0.04)))
    template = rng.choice(_ASSERT if says_true else _DENY)
    return template.format(
        word=_word(strength), tick=tick, tag=f"{rng.choice('XYZW')}{rng.randint(1, 9)}"
    )


def build_long_horizon(seed: int = 771177) -> LongHorizon:
    """Six recorded series, one latent, and a world that changes underneath."""
    rng = random.Random(seed)
    names = ("Za", "Zb", "Zc", "Zd", "Ze", "Zf")
    ar = (0.55, 0.70, 0.40, 0.35, 0.65, 0.25)
    scales = [1.0, 1.3, 0.9, 1.1, 1.0, 1.4]

    # The underlying series and the recorded series are kept apart. A
    # distribution shift changes how a variable is *written down*, not what it
    # does — collapsing the two would mean the shift silently altered the
    # relationship, and the decoy below would vanish for the wrong reason.
    base: list[list[float]] = []
    columns: list[list[float]] = []
    for index, (phi, scale) in enumerate(zip(ar, scales)):
        underlying: list[float] = []
        recorded: list[float] = []
        value = 0.0
        innovation = math.sqrt(max(1 - phi * phi, 1e-9))
        for tick in range(TICKS):
            value = phi * value + rng.gauss(0.0, innovation)
            underlying.append(scale * value)
            # From SCALE_SHIFT onward two series are recorded on a 40x scale.
            factor = 40.0 if (tick >= SCALE_SHIFT and index in (2, 5)) else 1.0
            recorded.append(factor * scale * value)
        base.append(underlying)
        columns.append(recorded)

    # The latent series. It acts on the outcome and is never recorded, which is
    # what makes the early correlation misleading.
    latent: list[float] = []
    value = 0.0
    for _ in range(TICKS):
        value = 0.6 * value + rng.gauss(0.0, 0.8)
        latent.append(value)

    # The real relationship: a change times a three-step delay. After the
    # regime change it runs the other way and weakens.
    raw = []
    for tick in range(TICKS):
        if tick < 3:
            raw.append(0.0)
            continue
        signal = (base[1][tick] - base[1][tick - 1]) * base[3][tick - 3]
        raw.append(signal)
    z = _standardise(raw)

    # A decoy: Zc predicts well before the regime change and not after. It is
    # the "previously useful pattern that stops working".
    decoy = _standardise(base[2])

    outcomes: list[bool] = []
    for tick in range(TICKS):
        if tick < REGIME_CHANGE:
            logit = 2.2 * z[tick] + 0.9 * decoy[tick] + 0.35 * latent[tick]
        else:
            logit = -1.6 * z[tick] + 0.30 * latent[tick]
        outcomes.append(rng.random() < _sigmoid(logit))

    observations = ObservationSet(
        id="LONG-HORIZON",
        variable_names=names,
        columns=tuple(tuple(c) for c in columns),
        outcomes=tuple(outcomes),
    )

    # -- the agents --------------------------------------------------------
    profiles = {
        "S1": (0.82, 0.80),  # good, moderately firm
        "S2": (0.56, 0.90),  # near chance, very firm
        "S3": (0.88, 0.75),  # good... until SOURCE_TURNS
        "S4": (0.22, 0.85),  # systematically wrong throughout
    }

    episodes: list[Episode] = []
    for tick in range(TICKS):
        truth = outcomes[tick]
        utterances: list[tuple[str, str]] = []
        for source in AGENTS:
            accuracy, assertiveness = profiles[source]
            if source == "S3" and tick >= SOURCE_TURNS:
                accuracy = 0.25  # the turn: a trusted source becomes unreliable
            says_true = truth if rng.random() < accuracy else not truth
            utterances.append((source, _speak(rng, tick, says_true, assertiveness)))
        episodes.append(
            Episode(
                tick=tick,
                outcome=truth,
                utterances=tuple(utterances),
                regime="early" if tick < REGIME_CHANGE else "late",
            )
        )

    return LongHorizon(
        id="LONG-HORIZON",
        variable_names=names,
        observations=observations,
        episodes=episodes,
        events=(
            (SOURCE_TURNS, "S3 stops being reliable (0.88 -> 0.25), unannounced"),
            (REGIME_CHANGE, "the relationship reverses sign and weakens; the Zc decoy stops working"),
            (SCALE_SHIFT, "Zc and Zf are recorded on a 40x scale from here on"),
        ),
        hidden_process=(
            "Before tick 420: P = sigmoid(2.2·z((Zb_t − Zb_{t-1})·Zd_{t-3}) "
            "+ 0.9·z(Zc) + 0.35·latent). After: P = sigmoid(−1.6·z(same product) "
            "+ 0.30·latent). The latent series is never recorded."
        ),
    )


# --------------------------------------------------------------- novel world


@dataclass
class NovelWorld:
    id: str
    variable_names: tuple[str, ...]
    observations: ObservationSet
    episodes: list[Episode]
    hidden_process: str
    #: Ticks governed by a structure the pattern language cannot express.
    unpatterned_from: int


def build_novel(seed: int = 313373) -> NovelWorld:
    """A world sharing only the *shape* of the first one's relationship.

    New names, new scales, new noise, new base rate, new agents. The second
    half is governed by a parity relationship, which nothing in ECHO's
    hypothesis language can express — the correct response there is
    `NO KNOWN PATTERN`, not the nearest available explanation.
    """
    rng = random.Random(seed)
    names = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
    ar = (0.25, 0.60, 0.80, 0.45, 0.35, 0.70)
    scales = (2400.0, 15.0, 0.004, 780.0, 0.09, 320.0)
    offsets = (0.0, 900.0, 0.0, -50.0, 0.0, 4000.0)

    columns: list[list[float]] = []
    for phi, scale, offset in zip(ar, scales, offsets):
        series: list[float] = []
        value = 0.0
        innovation = math.sqrt(max(1 - phi * phi, 1e-9))
        for _ in range(NOVEL_TICKS):
            value = phi * value + rng.gauss(0.0, innovation)
            series.append(offset + scale * value)
        columns.append(series)

    # Same shape as the source world, through different columns and at a
    # wildly different scale: a change times a two-step delay.
    raw = []
    for tick in range(NOVEL_TICKS):
        if tick < 2:
            raw.append(0.0)
            continue
        raw.append((columns[4][tick] - columns[4][tick - 1]) * columns[0][tick - 2])
    z = _standardise(raw)

    unpatterned_from = NOVEL_VAL_B_STOP
    outcomes: list[bool] = []
    for tick in range(NOVEL_TICKS):
        if tick < unpatterned_from:
            logit = 0.6 + 2.3 * z[tick]
        else:
            # Parity of the signs of two series: expressible with no
            # combination of the approved primitives.
            parity = (columns[1][tick] > 900.0) != (columns[5][tick] > 4000.0)
            logit = 2.0 if parity else -2.0
        outcomes.append(rng.random() < _sigmoid(logit))

    observations = ObservationSet(
        id="NOVEL",
        variable_names=names,
        columns=tuple(tuple(c) for c in columns),
        outcomes=tuple(outcomes),
    )

    profiles = {"T1": (0.75, 0.85), "T2": (0.60, 0.70), "T3": (0.30, 0.90)}
    episodes: list[Episode] = []
    for tick in range(NOVEL_TICKS):
        truth = outcomes[tick]
        utterances = []
        for source in NOVEL_AGENTS:
            accuracy, assertiveness = profiles[source]
            says_true = truth if rng.random() < accuracy else not truth
            utterances.append((source, _speak(rng, tick, says_true, assertiveness)))
        episodes.append(
            Episode(
                tick=tick,
                outcome=truth,
                utterances=tuple(utterances),
                regime="patterned" if tick < unpatterned_from else "unpatterned",
            )
        )

    return NovelWorld(
        id="NOVEL",
        variable_names=names,
        observations=observations,
        episodes=episodes,
        hidden_process=(
            "Ticks 0-399: P = sigmoid(0.6 + 2.3·z((Q5_t − Q5_{t-1})·Q1_{t-2})) — "
            "the source world's shape, through different columns, at scales from "
            "0.004 to 4000. Ticks 400+: P depends on the parity of the signs of "
            "two series, which the hypothesis language cannot express at all."
        ),
        unpatterned_from=unpatterned_from,
    )
