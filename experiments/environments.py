"""Three deterministic environments that generate outcomes ECHO must predict.

A hidden process emits `OUTCOME_A` or `OUTCOME_B` on each trial. **The
underlying probability is never shown to ECHO.** It sees only the sequence of
observations that have already happened, and must state a probability for the
next one before it occurs.

Every environment is seeded, so the whole experiment is reproducible: the same
seed yields the same outcome sequence, the same predictions, and the same
scores. Nothing here touches the network or a model.

- **A — Stable.** A stationary process. The honest ceiling for any predictor is
  the true rate; the question is whether ECHO converges on it.
- **B — Changing.** The probability shifts sharply half way through, with no
  announcement. A frequency estimator over all history *must* lag here, and
  measuring how badly is the point of including it.
- **C — Noisy.** The outcomes ECHO *observes* are corrupted: a fraction are
  reported flipped. It is scored against what actually happened, not against
  what it was told. This separates "wrong" from "misinformed".
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

OUTCOME_A = "OUTCOME_A"
OUTCOME_B = "OUTCOME_B"

PROPOSITION = "The next outcome will be OUTCOME_A."


@dataclass(frozen=True)
class Trial:
    index: int
    true_outcome: bool  # True means OUTCOME_A actually occurred
    observed_outcome: bool  # what ECHO is shown; differs from truth only in the noisy env

    @property
    def true_label(self) -> str:
        return OUTCOME_A if self.true_outcome else OUTCOME_B

    @property
    def observed_label(self) -> str:
        return OUTCOME_A if self.observed_outcome else OUTCOME_B

    @property
    def corrupted(self) -> bool:
        return self.true_outcome != self.observed_outcome


@dataclass
class Environment:
    id: str
    name: str
    description: str  # what ECHO could reasonably infer
    hidden_process: str  # the truth, never shown to ECHO
    trials: int
    seed: int
    observation_noise: float = 0.0  # chance an observation is reported flipped
    schedule: list[tuple[int, float]] = field(default_factory=list)  # (from_index, P(A))

    def probability_at(self, index: int) -> float:
        """The hidden P(A) at a given trial. Used to generate, never revealed."""
        current = self.schedule[0][1]
        for start, probability in self.schedule:
            if index >= start:
                current = probability
        return current

    def generate(self) -> list[Trial]:
        """Produce the full outcome sequence. Deterministic for a given seed."""
        rng = random.Random(self.seed)
        out: list[Trial] = []
        for index in range(self.trials):
            true_outcome = rng.random() < self.probability_at(index)
            observed = true_outcome
            if self.observation_noise and rng.random() < self.observation_noise:
                observed = not observed
            out.append(
                Trial(index=index, true_outcome=true_outcome, observed_outcome=observed)
            )
        return out


ENVIRONMENTS: list[Environment] = [
    Environment(
        id="ENV-A-stable",
        name="Environment A — Stable",
        description=(
            "A stationary process. Nothing about it changes for the duration of "
            "the run."
        ),
        hidden_process="P(OUTCOME_A) = 0.70, constant for all 60 trials.",
        trials=60,
        seed=20260810,
        schedule=[(0, 0.70)],
    ),
    Environment(
        id="ENV-B-changing",
        name="Environment B — Changing",
        description=(
            "A process that is stationary for a while and then is not. No signal "
            "is given when it changes."
        ),
        hidden_process=(
            "P(OUTCOME_A) = 0.85 for trials 0–29, then drops to 0.20 for trials "
            "30–59. The change is abrupt and unannounced."
        ),
        trials=60,
        seed=11235813,
        schedule=[(0, 0.85), (30, 0.20)],
    ),
    Environment(
        id="ENV-C-noisy",
        name="Environment C — Noisy",
        description=(
            "A stationary process whose observations are unreliable: a "
            "substantial fraction of what ECHO is told is simply wrong."
        ),
        hidden_process=(
            "P(OUTCOME_A) = 0.65, constant. Each observation shown to ECHO has a "
            "30% chance of being reported as the opposite of what happened. "
            "Predictions are scored against the truth, not the report."
        ),
        trials=60,
        seed=31415926,
        observation_noise=0.30,
        schedule=[(0, 0.65)],
    ),
]


def by_id(environment_id: str) -> Environment:
    for environment in ENVIRONMENTS:
        if environment.id == environment_id:
            return environment
    raise KeyError(f"no environment {environment_id!r}")


# ------------------------------------------------------------------ predictor


LAPLACE_ALPHA = 1.0


def laplace_predictor(view, alpha: float = LAPLACE_ALPHA) -> tuple[float, str]:
    """Estimate P(next = OUTCOME_A) from the observations available so far.

    Laplace-smoothed relative frequency: `(a + α) / (n + 2α)`. With no history
    it says 0.5 — genuine ignorance rather than a coin-flip guess dressed up as
    knowledge — and it never states 0 or 1, so it is never infinitely surprised.

    It weights all history equally, which is correct for a stationary process
    and wrong for a changing one. Environment B exists to measure exactly that
    weakness rather than to hide it.

    Takes a `TimelineView`, which cannot reach past its horizon. The predictor
    has no access to the future even in principle.
    """
    observed = view.outcomes()
    n = len(observed)
    a = sum(1 for value in observed if value)
    probability = (a + alpha) / (n + 2 * alpha)
    rationale = (
        f"Laplace-smoothed frequency over {n} prior observation(s): "
        f"{a} of {n} were OUTCOME_A, giving ({a} + {alpha}) / ({n} + {2 * alpha}) "
        f"= {probability:.4f}."
    )
    return probability, rationale
