"""A small fictional world whose true state is hidden from ECHO.

**The Thornwood Station incident.** On the night of 12 March the rare orchid
collection in the station greenhouse died. Four things were in play:

- **Dr Vance**, the night botanist, who runs unscheduled fertiliser trials.
- **The heating system**, which has a history of intermittent faults.
- **The irrigation supply**, drawn from an unfiltered rainwater cistern.
- **An inspector**, visiting that week, who dislikes Vance.

Each scenario below fixes a hidden `ground_truth` for its proposition. **ECHO
never receives it.** It sees only `Evidence` records — a description, a source,
a reliability, a relevance, and a stance — and must arrive at a confidence from
those alone. Ground truth is used exactly once, by the scorer, after the run, to
decide whether ECHO ended up in the right place.

Beliefs always start at 0.5 (genuine ignorance) and move only through evidence.
Nothing is ever handed the answer, and no scenario tells ECHO that a previous
belief was wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from echo.belief import Evidence, EvidenceStance

SUPPORTS = EvidenceStance.SUPPORTS
CONTRADICTS = EvidenceStance.CONTRADICTS

VANCE_PROPOSITION = (
    "Dr Vance's unscheduled fertiliser application caused the orchid deaths."
)
CISTERN_PROPOSITION = (
    "Contaminated cistern water caused the orchid deaths."
)


def ev(
    eid: str,
    description: str,
    source: str,
    reliability: float,
    relevance: float,
    stance: EvidenceStance,
) -> Evidence:
    return Evidence(
        id=eid,
        description=description,
        source=source,
        reliability=reliability,
        relevance=relevance,
        stance=stance,
    )


@dataclass
class Scenario:
    id: str
    title: str
    proposition: str
    ground_truth: bool  # HIDDEN from ECHO; used only by the scorer
    hidden_state: str  # what actually happened, for the report's appendix
    initial_evidence: list[Evidence]  # A, B, C — forms the hypothesis
    later_evidence: list[Evidence] = field(default_factory=list)  # D, E — arrives after
    purpose: str = ""

    @property
    def all_evidence(self) -> list[Evidence]:
        return [*self.initial_evidence, *self.later_evidence]


# The three ambiguous observations every orchid scenario opens with. Identical
# across scenarios on purpose: the initial belief is therefore identical, and
# any later divergence is caused by the later evidence and nothing else.
def _opening_evidence(suffix: str) -> list[Evidence]:
    return [
        ev(
            f"EV-A-{suffix}",
            "Station log records an unscheduled fertiliser application at 23:40, "
            "logged by Vance.",
            "station activity log",
            reliability=0.90,
            relevance=0.50,
            stance=SUPPORTS,
        ),
        ev(
            f"EV-B-{suffix}",
            "A lab technician noticed an unidentified white residue on the soil "
            "surface the next morning.",
            "lab technician, visual inspection",
            reliability=0.60,
            relevance=0.45,
            stance=SUPPORTS,
        ),
        ev(
            f"EV-C-{suffix}",
            "Keycard records show Vance was the only person to enter the "
            "greenhouse overnight.",
            "keycard access system",
            reliability=0.85,
            relevance=0.35,
            stance=SUPPORTS,
        ),
    ]


SCENARIOS: list[Scenario] = [
    Scenario(
        id="S1-belief-correct",
        title="The original hypothesis is right, and later evidence confirms it",
        proposition=VANCE_PROPOSITION,
        ground_truth=True,
        hidden_state=(
            "Vance's experimental compound was phytotoxic at the concentration "
            "applied. The heater ran normally all night. The anonymous note was "
            "written by the inspector, who disliked Vance and was guessing."
        ),
        initial_evidence=_opening_evidence("S1"),
        later_evidence=[
            ev(
                "EV-D-S1",
                "An anonymous note left at reception claims the heating failed "
                "that night.",
                "anonymous note",
                reliability=0.20,
                relevance=0.50,
                stance=CONTRADICTS,
            ),
            ev(
                "EV-E-S1",
                "Mass-spectrometry assay identifies the residue as Vance's "
                "compound at four times the phytotoxic threshold.",
                "accredited external laboratory",
                reliability=0.95,
                relevance=0.95,
                stance=SUPPORTS,
            ),
        ],
        purpose=(
            "The reversal control. Contradictory evidence arrives, but it is weak. "
            "A system that lowers confidence whenever it is contradicted fails here."
        ),
    ),
    Scenario(
        id="S2-belief-incorrect",
        title="The original hypothesis is wrong, and strong evidence overturns it",
        proposition=VANCE_PROPOSITION,
        ground_truth=False,
        hidden_state=(
            "The heating system failed for six hours and the greenhouse dropped "
            "to -4 °C, which killed the orchids. Vance did apply fertiliser, but "
            "it was an inert calcium carbonate control batch."
        ),
        initial_evidence=_opening_evidence("S2"),
        later_evidence=[
            ev(
                "EV-D-S2",
                "Heating telemetry shows a six-hour outage with the greenhouse "
                "reaching -4 °C, corroborated by an independent sensor.",
                "building telemetry, dual-sensor",
                reliability=0.95,
                relevance=0.90,
                stance=CONTRADICTS,
            ),
            ev(
                "EV-E-S2",
                "Mass-spectrometry assay identifies the residue as inert calcium "
                "carbonate with no phytotoxic activity.",
                "accredited external laboratory",
                reliability=0.90,
                relevance=0.85,
                stance=CONTRADICTS,
            ),
        ],
        purpose=(
            "The mirror of S1. Identical opening evidence and identical initial "
            "belief; only the later evidence differs. Divergence here is caused by "
            "evidence quality, not by any hint about which world we are in."
        ),
    ),
    Scenario(
        id="S3-weak-noise",
        title="A well-supported belief under a barrage of weak contradiction",
        proposition=VANCE_PROPOSITION,
        ground_truth=True,
        hidden_state=(
            "Same as S1: Vance's compound did it. The station rumour mill produced "
            "four separate unfounded counter-theories."
        ),
        initial_evidence=_opening_evidence("S3"),
        later_evidence=[
            ev(
                "EV-D1-S3",
                "A junior intern says they 'heard the heating was playing up'.",
                "second-hand hearsay",
                reliability=0.15,
                relevance=0.45,
                stance=CONTRADICTS,
            ),
            ev(
                "EV-D2-S3",
                "A canteen rumour blames the cistern water.",
                "canteen rumour",
                reliability=0.12,
                relevance=0.40,
                stance=CONTRADICTS,
            ),
            ev(
                "EV-D3-S3",
                "The inspector asserts, without evidence, that Vance is being "
                "scapegoated.",
                "inspector, unsupported assertion",
                reliability=0.25,
                relevance=0.45,
                stance=CONTRADICTS,
            ),
            ev(
                "EV-D4-S3",
                "An unsigned forum post claims a similar die-off happened without "
                "any fertiliser involved.",
                "anonymous forum post",
                reliability=0.10,
                relevance=0.50,
                stance=CONTRADICTS,
            ),
        ],
        purpose=(
            "Volume is not weight. Four contradictions in a row must not add up to "
            "a reversal when every one of them is untrustworthy."
        ),
    ),
    Scenario(
        id="S4-irrelevant-contradiction",
        title="Contradictory evidence that does not bear on the proposition",
        proposition=VANCE_PROPOSITION,
        ground_truth=True,
        hidden_state=(
            "Same as S1. The budget dispute and the car-park incident are real "
            "events with nothing to do with the orchids."
        ),
        initial_evidence=_opening_evidence("S4"),
        later_evidence=[
            ev(
                "EV-D1-S4",
                "Vance disputed the greenhouse budget allocation in February.",
                "finance committee minutes",
                reliability=0.95,
                relevance=0.05,
                stance=CONTRADICTS,
            ),
            ev(
                "EV-D2-S4",
                "A vehicle reversed into the greenhouse loading bay on 3 March, "
                "nine days before the incident.",
                "site incident report",
                reliability=0.90,
                relevance=0.10,
                stance=CONTRADICTS,
            ),
        ],
        purpose=(
            "Highly reliable and entirely beside the point. Both must be recorded "
            "and neither applied — confidence should not move at all."
        ),
    ),
    Scenario(
        id="S5-gradual-accumulation",
        title="Several moderate contradictions accumulate into a reversal",
        proposition=VANCE_PROPOSITION,
        ground_truth=False,
        hidden_state=(
            "The cistern was contaminated with a herbicide run-off. Vance's "
            "fertiliser was irrelevant. No single observation is decisive, but "
            "together they point away from Vance."
        ),
        initial_evidence=_opening_evidence("S5"),
        later_evidence=[
            ev(
                "EV-D1-S5",
                "Two orchids in a separate room that received no fertiliser also "
                "died.",
                "greenhouse inventory audit",
                reliability=0.75,
                relevance=0.55,
                stance=CONTRADICTS,
            ),
            ev(
                "EV-D2-S5",
                "The same compound was applied in January with no ill effects.",
                "historical trial records",
                reliability=0.70,
                relevance=0.50,
                stance=CONTRADICTS,
            ),
            ev(
                "EV-D3-S5",
                "Cistern water sampled on 13 March shows elevated herbicide "
                "residue.",
                "water quality sampling",
                reliability=0.80,
                relevance=0.60,
                stance=CONTRADICTS,
            ),
        ],
        purpose=(
            "No single item here would overturn the belief. The question is "
            "whether they compound correctly rather than each being shrugged off."
        ),
    ),
    Scenario(
        id="S6-alternative-proposition",
        title="A different proposition about the same night",
        proposition=CISTERN_PROPOSITION,
        ground_truth=True,
        hidden_state=(
            "The same world as S5, viewed from the other side: the cistern really "
            "was the cause. Evidence that contradicted the Vance hypothesis "
            "supports this one."
        ),
        initial_evidence=[
            ev(
                "EV-A-S6",
                "Cistern water sampled on 13 March shows elevated herbicide "
                "residue.",
                "water quality sampling",
                reliability=0.80,
                relevance=0.60,
                stance=SUPPORTS,
            ),
            ev(
                "EV-B-S6",
                "Orchids in rooms with no fertiliser exposure also died; all were "
                "on the same irrigation line.",
                "greenhouse inventory audit",
                reliability=0.75,
                relevance=0.65,
                stance=SUPPORTS,
            ),
            ev(
                "EV-C-S6",
                "Groundskeeping sprayed herbicide upslope of the cistern on 11 "
                "March.",
                "groundskeeping work order",
                reliability=0.85,
                relevance=0.55,
                stance=SUPPORTS,
            ),
        ],
        later_evidence=[
            ev(
                "EV-D-S6",
                "The cistern filter was replaced on 1 March and certified clean.",
                "maintenance certificate",
                reliability=0.70,
                relevance=0.40,
                stance=CONTRADICTS,
            ),
        ],
        purpose=(
            "A belief that should end up strongly held, tested with a moderate "
            "contradiction that should dent it without dislodging it."
        ),
    ),
]


def by_id(scenario_id: str) -> Scenario:
    for scenario in SCENARIOS:
        if scenario.id == scenario_id:
            return scenario
    raise KeyError(f"no scenario {scenario_id!r}")
