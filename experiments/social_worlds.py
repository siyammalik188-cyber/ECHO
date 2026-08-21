"""TRUTH for ECHO 10. Four agents, and none of them wears a label.

Each has a hidden accuracy, a hidden area of expertise where it does better, and
a hidden bias in how strongly it speaks. One is systematically wrong — not noisy,
*anti-correlated* — and ECHO is not told which. It has to fall out of outcomes.

The minority-correct rounds are the point of the whole exercise. In those, the
three weaker agents agree and the strongest disagrees, and the truth is with the
lone dissenter. A system that counted votes would get every one of them wrong.

Nothing under `echo/` imports this module.
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROPOSITIONS = tuple(f"hypothesis-{index}" for index in range(1, 41))

#: Rounds where the majority is wrong and one agent is right. Chosen in advance
#: and spread through the run, not clustered at the end.
MINORITY_ROUNDS = (7, 13, 19, 24, 29, 33, 36, 38)


@dataclass(frozen=True)
class AgentTruth:
    """One agent's hidden profile."""

    name: str
    #: P(this agent's assertion matches the truth) outside its speciality.
    accuracy: float
    #: The topic it is unusually good at, and how good.
    speciality: str
    speciality_accuracy: float
    #: How firmly it speaks, regardless of whether it is right.
    assertiveness: float
    #: The evaluator's note. Never reaches ECHO.
    note: str

    def accuracy_for(self, proposition: str) -> float:
        return (
            self.speciality_accuracy
            if proposition in SPECIALITIES[self.speciality]
            else self.accuracy
        )


#: Which propositions belong to which topic. ECHO never sees this grouping; it
#: exists so the agents can differ in *what* they are good at, not just how
#: often they are right.
SPECIALITIES: dict[str, set[str]] = {
    "odd": {p for i, p in enumerate(PROPOSITIONS) if i % 2 == 0},
    "even": {p for i, p in enumerate(PROPOSITIONS) if i % 2 == 1},
    "none": set(),
}


AGENTS: tuple[AgentTruth, ...] = (
    AgentTruth(
        name="AGENT-A",
        accuracy=0.80,
        speciality="odd",
        speciality_accuracy=0.92,
        assertiveness=0.80,
        note="good, and better still on its speciality",
    ),
    AgentTruth(
        name="AGENT-B",
        accuracy=0.55,
        speciality="none",
        speciality_accuracy=0.55,
        assertiveness=0.90,
        note="barely better than chance, and says everything firmly",
    ),
    AgentTruth(
        name="AGENT-C",
        accuracy=0.92,
        speciality="even",
        speciality_accuracy=0.95,
        assertiveness=0.70,
        note="the most reliable, and the most hedged — the trap for a system that reads confidence as competence",
    ),
    AgentTruth(
        name="AGENT-D",
        accuracy=0.18,
        speciality="none",
        speciality_accuracy=0.18,
        assertiveness=0.88,
        note=(
            "systematically wrong: right less than a fifth of the time while "
            "asserting 0.88. Not noise — anti-correlated with the truth, which "
            "makes it informative in reverse once that is noticed."
        ),
    ),
)

AGENTS_BY_NAME = {agent.name: agent for agent in AGENTS}


#: Phrasings the agents use. Varied so parsing is doing real work, and all
#: within the closed vocabulary `parse_claim` recognises.
_ASSERT_TEMPLATES = (
    "I {word} {proposition} is more likely because {evidence}",
    "{proposition} looks correct to me; I am {word} because {evidence}",
    "I am {word} that {proposition} holds, because {evidence}",
)
_DENY_TEMPLATES = (
    "I {word} {proposition} is not the right one because {evidence}",
    "I would rule out {proposition}; I am {word} because {evidence}",
    "{proposition} is unlikely, and I am {word} of that because {evidence}",
)

_WORDS_BY_STRENGTH = (
    (0.90, "sure"),
    (0.85, "confident"),
    (0.75, "likely"),
    (0.65, "think"),
    (0.55, "suspect"),
)


def _word_for(strength: float) -> str:
    for threshold, word in _WORDS_BY_STRENGTH:
        if strength >= threshold:
            return word
    return "suspect"


@dataclass(frozen=True)
class Round:
    """One proposition, its truth, and what each agent said about it."""

    index: int
    proposition: str
    truth: bool
    utterances: tuple[tuple[str, str], ...]  # (agent name, sentence)
    minority_round: bool


def generate(seed: int = 90210) -> list[Round]:
    """Build the whole conversation deterministically."""
    rng = random.Random(seed)
    rounds: list[Round] = []

    for index, proposition in enumerate(PROPOSITIONS):
        truth = rng.random() < 0.5
        minority = index in MINORITY_ROUNDS

        says: dict[str, bool] = {}
        if minority:
            # The strongest agent alone is right; everyone else agrees and is
            # wrong. Constructed, not drawn, so the case is guaranteed present.
            for agent in AGENTS:
                says[agent.name] = truth if agent.name == "AGENT-C" else not truth
        else:
            for agent in AGENTS:
                on_target = rng.random() < agent.accuracy_for(proposition)
                says[agent.name] = truth if on_target else not truth

        utterances: list[tuple[str, str]] = []
        for agent in AGENTS:
            strength = min(0.95, max(0.52, rng.gauss(agent.assertiveness, 0.04)))
            word = _word_for(strength)
            evidence = f"observation {rng.choice('XYZW')}{rng.randint(1, 9)} occurred"
            templates = _ASSERT_TEMPLATES if says[agent.name] else _DENY_TEMPLATES
            sentence = rng.choice(templates).format(
                word=word, proposition=proposition, evidence=evidence
            )
            utterances.append((agent.name, sentence))

        rounds.append(
            Round(
                index=index,
                proposition=proposition,
                truth=truth,
                utterances=tuple(utterances),
                minority_round=minority,
            )
        )
    return rounds
