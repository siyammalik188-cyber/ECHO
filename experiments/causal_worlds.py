"""TRUTH for ECHO 8. Four causal structures, three of them indistinguishable.

The classic problem, built exactly: a chain, a fork, a reversed chain and a
collider over three recorded variables. The first three form a **Markov
equivalence class** — same skeleton, no v-structure — so they induce the
*identical* joint distribution and no observational method can order them. Their
parameters are derived from one another by Bayes' rule rather than fitted, so
the equality is exact to floating point.

The collider is deliberately *not* equivalent. Observation can rule it out, and
should. Including it keeps the report honest: observational data is not useless
in general, it is insufficient for a specific and identifiable reason.

Nothing under `echo/` imports this module.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo.causal import CausalModel  # noqa: E402
from echo.experiment import ExperimentOption, Intervention  # noqa: E402

VARIABLES = ("V1", "V2", "V3")

# The chain's parameters. Everything else is derived from these so that the
# equivalence class matches exactly rather than approximately.
P_V1 = 0.5
P_V2_GIVEN_V1 = (0.15, 0.85)  # indexed by V1
P_V3_GIVEN_V2 = (0.20, 0.80)  # indexed by V2

# Derived by Bayes from the chain. P(V2=1) = 0.5 and P(V3=1) = 0.5 by symmetry,
# so the reversed conditionals come out as clean numbers.
P_V1_GIVEN_V2 = (0.15, 0.85)
P_V2_GIVEN_V3 = (0.20, 0.80)


def _model(model_id: str, parents, cpts, description) -> CausalModel:
    return CausalModel(
        id=model_id,
        variables=VARIABLES,
        parents=parents,
        cpts=cpts,
        description=description,
    )


#: V1 -> V2 -> V3. A causes B causes C.
CHAIN = _model(
    "M1-chain",
    parents={"V1": (), "V2": ("V1",), "V3": ("V2",)},
    cpts={
        "V1": {(): P_V1},
        "V2": {(0,): P_V2_GIVEN_V1[0], (1,): P_V2_GIVEN_V1[1]},
        "V3": {(0,): P_V3_GIVEN_V2[0], (1,): P_V3_GIVEN_V2[1]},
    },
    description="V1 -> V2 -> V3",
)

#: V2 -> V1 and V2 -> V3. A <- B -> C: the middle one drives both ends.
FORK = _model(
    "M2-fork",
    parents={"V1": ("V2",), "V2": (), "V3": ("V2",)},
    cpts={
        "V1": {(0,): P_V1_GIVEN_V2[0], (1,): P_V1_GIVEN_V2[1]},
        "V2": {(): 0.5},
        "V3": {(0,): P_V3_GIVEN_V2[0], (1,): P_V3_GIVEN_V2[1]},
    },
    description="V2 -> V1, V2 -> V3",
)

#: V3 -> V2 -> V1. The chain, running the other way.
REVERSE_CHAIN = _model(
    "M3-reverse-chain",
    parents={"V1": ("V2",), "V2": ("V3",), "V3": ()},
    cpts={
        "V1": {(0,): P_V1_GIVEN_V2[0], (1,): P_V1_GIVEN_V2[1]},
        "V2": {(0,): P_V2_GIVEN_V3[0], (1,): P_V2_GIVEN_V3[1]},
        "V3": {(): 0.5},
    },
    description="V3 -> V2 -> V1",
)

#: V1 -> V2 <- V3. A collider: the ends are independent until you look at V2.
COLLIDER = _model(
    "M4-collider",
    parents={"V1": (), "V2": ("V1", "V3"), "V3": ()},
    cpts={
        "V1": {(): 0.5},
        "V2": {(0, 0): 0.05, (0, 1): 0.50, (1, 0): 0.50, (1, 1): 0.95},
        "V3": {(): 0.5},
    },
    description="V1 -> V2 <- V3",
)

WORLDS = (CHAIN, FORK, REVERSE_CHAIN, COLLIDER)

#: The three that observation cannot order. Their joints are equal by
#: construction, so any method that claims to separate them from observational
#: data alone is wrong.
EQUIVALENCE_CLASS = (CHAIN, FORK, REVERSE_CHAIN)


def causal_menu(samples: int = 12) -> list[ExperimentOption]:
    """Interventions available in the causal worlds, with their prices."""
    return [
        ExperimentOption(
            id="OBS",
            intervention=Intervention.observation(),
            cost=1.0,
            risk=0.0,
            time=1.0,
            samples=samples,
            description="Watch. Rules out a collider; cannot order the rest.",
        ),
        ExperimentOption(
            id="DO-V1",
            intervention=Intervention.of(V1=1),
            cost=3.0,
            risk=0.1,
            time=2.0,
            samples=samples,
            description="Set V1.",
        ),
        ExperimentOption(
            id="DO-V2",
            intervention=Intervention.of(V2=1),
            cost=3.0,
            risk=0.1,
            time=2.0,
            samples=samples,
            description="Set V2.",
        ),
        ExperimentOption(
            id="DO-V3",
            intervention=Intervention.of(V3=1),
            cost=3.0,
            risk=0.1,
            time=2.0,
            samples=samples,
            description="Set V3.",
        ),
    ]


@dataclass
class CausalScenario:
    id: str
    name: str
    briefing: str
    truth_id: str
    #: How many interventional experiments the campaign is allowed.
    steps: int = 3
    samples: int = 12
    #: Rows of purely observational data seen before any intervention.
    observational_rows: int = 200

    def truth(self) -> CausalModel:
        for model in WORLDS:
            if model.id == self.truth_id:
                return model
        raise KeyError(self.truth_id)


SCENARIOS: list[CausalScenario] = [
    CausalScenario(
        id="CAUSAL-chain",
        name="The truth is a chain",
        briefing=(
            "Three recorded binary variables, all pairwise associated. Four "
            "structures are on the table."
        ),
        truth_id="M1-chain",
    ),
    CausalScenario(
        id="CAUSAL-fork",
        name="The truth is a fork",
        briefing="The same variables, the same four structures, the same associations.",
        truth_id="M2-fork",
    ),
    CausalScenario(
        id="CAUSAL-reverse",
        name="The truth is the chain, reversed",
        briefing="The same variables, the same four structures, the same associations.",
        truth_id="M3-reverse-chain",
    ),
    CausalScenario(
        id="CAUSAL-collider",
        name="The truth is a collider",
        briefing="The same variables and the same four structures.",
        truth_id="M4-collider",
    ),
    CausalScenario(
        id="CAUSAL-starved",
        name="Too little evidence to settle it",
        briefing=(
            "The same setup, but only one intervention is affordable and it is "
            "run on very little data."
        ),
        truth_id="M1-chain",
        steps=1,
        samples=3,
    ),
]


def by_id(scenario_id: str) -> CausalScenario:
    for scenario in SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(scenario_id)
