"""TRUTH for ECHO 7. Competing explanations, and one of them is right.

Each scenario hands ECHO a set of causal models it cannot tell apart from
watching alone, plus a menu of things it could do about that. Which model is
actually generating the data is recorded here and reaches ECHO nowhere — it is
used to draw samples and to score the result afterwards, never to choose.

The menu is built to contain every trap the challenge asks for:

- a **cheap, highly informative** experiment;
- an **expensive, uninformative** one;
- one whose result is **inconclusive** by construction — every hypothesis
  predicts exactly the same distribution, so its expected information gain is
  identically zero and no amount of budget changes that;
- one that **contradicts** whatever the observational data made look obvious;
- and the observational data itself, which is **misleading** on purpose.

Nothing under `echo/` imports this module.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo.causal import CausalModel  # noqa: E402
from echo.experiment import ExperimentOption, Intervention  # noqa: E402

VARIABLES = ("V1", "V2", "V3")

#: Only V1 and V2 are recorded. V3 acts but is never seen — which is what
#: lets a confounded world be indistinguishable from a direct one by watching.
OBSERVED = ("V1", "V2")

#: The edge strength that makes the confounded model reproduce the direct
#: model's observed joint exactly. Derived, not guessed: it solves
#: 0.5c^2 + 0.5(1-c)^2 = P(V1=1, V2=1) = 0.425.
CONFOUNDER_STRENGTH = (2 + math.sqrt(4 - 1.2)) / 4


def _model(
    model_id: str,
    parents: dict[str, tuple[str, ...]],
    cpts: dict[str, dict[tuple[int, ...], float]],
    description: str,
) -> CausalModel:
    return CausalModel(
        id=model_id,
        variables=VARIABLES,
        parents=parents,
        cpts=cpts,
        description=description,
        observed=OBSERVED,
    )


# --------------------------------------------------------------- the models
#
# Two explanations of the same visible fact: V1 and V2 move together.

#: V1 causes V2. V3 drifts along on its own.
DIRECT = _model(
    "H1-direct",
    parents={"V1": (), "V2": ("V1",), "V3": ()},
    cpts={
        "V1": {(): 0.5},
        "V2": {(0,): 0.15, (1,): 0.85},
        "V3": {(): 0.5},
    },
    description="V1 -> V2; V3 stands alone",
)

#: V3 causes both. V1 and V2 never touch, but they move together anyway.
CONFOUNDED = _model(
    "H2-confounded",
    parents={"V1": ("V3",), "V2": ("V3",), "V3": ()},
    cpts={
        "V1": {(0,): 1 - CONFOUNDER_STRENGTH, (1,): CONFOUNDER_STRENGTH},
        "V2": {(0,): 1 - CONFOUNDER_STRENGTH, (1,): CONFOUNDER_STRENGTH},
        "V3": {(): 0.5},
    },
    description="V3 -> V1 and V3 -> V2; V1 and V2 are not connected",
)

#: V2 causes V1 — the same association, running the other way.
REVERSED = _model(
    "H3-reversed",
    parents={"V1": ("V2",), "V2": (), "V3": ()},
    cpts={
        "V1": {(0,): 0.15, (1,): 0.85},
        "V2": {(): 0.5},
        "V3": {(): 0.5},
    },
    description="V2 -> V1; V3 stands alone",
)

COMPETING = (DIRECT, CONFOUNDED, REVERSED)


# ----------------------------------------------------------------- the menu


def experiment_menu(samples: int = 8) -> list[ExperimentOption]:
    """What ECHO may do. Costs and budgets are fixed here, not by ECHO.

    Note that no option is labelled *informative* or *useless*. Which is which
    depends on the current posterior and is recomputed every step; the labels
    below describe price and effort, which are facts about the world.
    """
    return [
        ExperimentOption(
            id="OBS",
            intervention=Intervention.observation(),
            cost=1.0,
            risk=0.0,
            time=1.0,
            samples=samples,
            description="Watch without touching anything. Cheap.",
        ),
        ExperimentOption(
            id="DO-V1",
            intervention=Intervention.of(V1=1),
            cost=2.0,
            risk=0.1,
            time=2.0,
            samples=samples,
            description="Set V1 and watch the rest. Cheap.",
        ),
        ExperimentOption(
            id="DO-V2",
            intervention=Intervention.of(V2=1),
            cost=12.0,
            risk=0.5,
            time=6.0,
            samples=samples,
            description="Set V2 and watch the rest. Expensive and slow.",
        ),
        ExperimentOption(
            id="DO-V3",
            intervention=Intervention.of(V3=1),
            cost=6.0,
            risk=0.2,
            time=3.0,
            samples=samples,
            description="Set V3 and watch the rest. Mid-priced.",
        ),
        ExperimentOption(
            id="DO-V1V2",
            intervention=Intervention.of(V1=1, V2=1),
            cost=9.0,
            risk=0.6,
            time=5.0,
            samples=samples,
            description=(
                "Set V1 and V2 together. Only V3 is left free, and every "
                "hypothesis says the same thing about V3 — so this can return "
                "no information at all, at a high price."
            ),
        ),
    ]


@dataclass
class Scenario:
    """One controlled situation: a hypothesis set, a menu, and a hidden truth."""

    id: str
    name: str
    #: What ECHO is told about the situation. Never says which model is right.
    briefing: str
    models: tuple[CausalModel, ...]
    truth_id: str
    #: The evaluator's note on why the observational picture is misleading.
    trap: str
    samples: int = 8
    steps: int = 4

    def truth(self) -> CausalModel:
        for model in self.models:
            if model.id == self.truth_id:
                return model
        raise KeyError(f"no model {self.truth_id!r} in scenario {self.id}")

    def menu(self) -> list[ExperimentOption]:
        return experiment_menu(self.samples)


SCENARIOS: list[Scenario] = [
    Scenario(
        id="EXP-confounded",
        name="Scenario 1 — the association is real, the story is not",
        briefing=(
            "Three binary variables. V1 and V2 are strongly associated in the "
            "observational record. Three explanations are on the table and the "
            "observational data does not separate them."
        ),
        models=COMPETING,
        truth_id="H2-confounded",
        trap=(
            "V1 and V2 correlate at about 0.7 whichever explanation is true, so "
            "watching harder cannot help. Only setting V1 or V3 breaks the tie: "
            "under the true model, forcing V1 leaves V2 alone."
        ),
        samples=8,
        steps=4,
    ),
    Scenario(
        id="EXP-direct",
        name="Scenario 2 — the obvious story happens to be right",
        briefing=(
            "The same three variables and the same three explanations. The "
            "observational record again shows V1 and V2 moving together."
        ),
        models=COMPETING,
        truth_id="H1-direct",
        trap=(
            "Included so the setup cannot be passed by always concluding "
            "'confounded'. Here the direct explanation is correct, and the same "
            "procedure has to arrive at it."
        ),
        samples=8,
        steps=4,
    ),
    Scenario(
        id="EXP-reversed",
        name="Scenario 3 — the association runs the other way",
        briefing=(
            "The same three variables and the same three explanations. The "
            "observational record is, once more, an association between V1 and "
            "V2."
        ),
        models=COMPETING,
        truth_id="H3-reversed",
        trap=(
            "The third possibility. Distinguishing it from the direct model "
            "requires intervening on V1 and finding V2 unmoved, then on V2 and "
            "finding V1 follows — a two-step conclusion no single experiment "
            "delivers."
        ),
        samples=8,
        steps=4,
    ),
]


def by_id(scenario_id: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(f"no scenario {scenario_id!r}")
