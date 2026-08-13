"""TRUTH for ECHO 9. How good ECHO actually is at each kind of question.

Every domain has a hidden **skill** — the probability its answers are right —
and a hidden **assertiveness** — how strongly it states them. The two are set
independently on purpose, because the interesting failure is not being bad, it
is being bad *and confident*: `causal_inference` here answers barely better than
chance while asserting 0.85, which is exactly the profile a self-assessment
layer exists to catch.

ECHO is told neither number. It sees only its own outcomes, and any competence
band, meta-confidence or abstention decision has to be derived from those.

Nothing under `echo/` imports this module.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo.metacognition import DOMAINS  # noqa: E402


@dataclass(frozen=True)
class DomainTruth:
    """The hidden profile of one capability area."""

    domain: str
    #: P(ECHO's answer is on the correct side).
    skill: float
    #: How strongly ECHO states its answers, right or wrong.
    assertiveness: float
    #: The evaluator's note. Never reaches ECHO.
    note: str

    @property
    def is_overconfident(self) -> bool:
        return self.assertiveness > self.skill + 0.10


PROFILES: dict[str, DomainTruth] = {
    "experimentation": DomainTruth(
        "experimentation",
        skill=0.92,
        assertiveness=0.88,
        note="genuinely strong, and states itself at about the right strength",
    ),
    "prediction": DomainTruth(
        "prediction",
        skill=0.86,
        assertiveness=0.84,
        note="strong and roughly calibrated",
    ),
    "discovery": DomainTruth(
        "discovery",
        skill=0.74,
        assertiveness=0.78,
        note="middling; slight overstatement",
    ),
    "transfer": DomainTruth(
        "transfer",
        skill=0.66,
        assertiveness=0.88,
        note="mediocre and overconfident — states 0.88, delivers 0.66",
    ),
    "causal_inference": DomainTruth(
        "causal_inference",
        skill=0.53,
        assertiveness=0.85,
        note=(
            "barely better than chance while asserting 0.85. The domain the "
            "abstention mechanism exists for."
        ),
    ),
}

assert set(PROFILES) == set(DOMAINS)


@dataclass(frozen=True)
class Task:
    """One binary question, its true answer, and how ECHO would answer it."""

    domain: str
    index: int
    truth: bool
    #: ECHO's stated probability that the proposition is true.
    belief: float

    @property
    def answer(self) -> bool:
        return self.belief > 0.5

    @property
    def correct(self) -> bool:
        return self.answer == self.truth

    @property
    def proposition(self) -> str:
        return f"{self.domain} task {self.index}"


def generate(domain: str, count: int, rng: random.Random) -> list[Task]:
    """Draw a stream of tasks for one domain from its hidden profile."""
    profile = PROFILES[domain]
    tasks: list[Task] = []
    for index in range(count):
        truth = rng.random() < 0.5
        on_target = rng.random() < profile.skill
        says_true = truth if on_target else not truth
        # Assertiveness is jittered a little so the record is not degenerate.
        strength = min(0.99, max(0.51, rng.gauss(profile.assertiveness, 0.05)))
        belief = strength if says_true else 1.0 - strength
        tasks.append(Task(domain=domain, index=index, truth=truth, belief=belief))
    return tasks
