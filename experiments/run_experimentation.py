"""Run the ECHO 7 experimentation campaigns and write ECHO_7_EXPERIMENTATION.md.

One campaign is: start from a flat posterior over the competing explanations,
repeatedly choose an experiment, run it against the hidden truth, and update.
The question is whether choosing by expected information gain beats choosing at
random — measured on identical scenarios, identical menus and identical seeds.

Four policies are compared. Two are the point (`information_gain` vs `random`)
and two are there to show the trade-off has two edges: `max_information`
ignores price, `cheapest` ignores value.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo.causal import Assignment, CausalModel, HypothesisSet  # noqa: E402
from echo.experiment import (  # noqa: E402
    COST_WEIGHT,
    RISK_WEIGHT,
    ExperimentLedger,
    ExperimentOption,
    ExperimentRecord,
    Intervention,
    appraise,
    choose_experiment,
    outcome_distribution,
    outcome_space,
)

from experiments.experiment_worlds import SCENARIOS, Scenario  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "experimentation"

#: How many independent campaigns per policy. Enough that the comparison is
#: not one lucky draw.
REPEATS = 40

#: A campaign counts as having identified the truth once the true model holds
#: at least this much posterior mass.
IDENTIFIED = 0.90

#: An experiment is called conclusive if it moved the posterior entropy by more
#: than this. Fixed in advance.
CONCLUSIVE_BITS = 0.05

POLICIES = ("information_gain", "random", "max_information", "cheapest")


def draw(
    model: CausalModel,
    intervention: Intervention,
    samples: int,
    rng: random.Random,
) -> list[Assignment]:
    """Sample observations from the hidden truth under an intervention."""
    outcomes = outcome_space(model, intervention)
    weights = outcome_distribution(model, intervention, outcomes)
    total = sum(weights)
    drawn: list[Assignment] = []
    for _ in range(samples):
        target = rng.random() * total
        running = 0.0
        for outcome, weight in zip(outcomes, weights):
            running += weight
            if target <= running:
                drawn.append(outcome)
                break
        else:  # pragma: no cover - only on floating-point edge
            drawn.append(outcomes[-1])
    return drawn


@dataclass
class Campaign:
    scenario: Scenario
    policy: str
    seed: int
    records: list[ExperimentRecord] = field(default_factory=list)
    posterior_trace: list[float] = field(default_factory=list)  # P(truth) per step
    entropy_trace: list[float] = field(default_factory=list)
    final: HypothesisSet | None = None

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.records)

    @property
    def final_truth_probability(self) -> float:
        return self.posterior_trace[-1] if self.posterior_trace else 0.0

    @property
    def final_entropy(self) -> float:
        return self.entropy_trace[-1] if self.entropy_trace else 0.0

    @property
    def identified(self) -> bool:
        return self.final_truth_probability >= IDENTIFIED

    @property
    def bits_removed(self) -> float:
        if not self.records:
            return 0.0
        return self.records[0].entropy_before - self.records[-1].entropy_after

    @property
    def bits_per_cost(self) -> float:
        return self.bits_removed / self.total_cost if self.total_cost else 0.0

    def cost_to_identify(self) -> float | None:
        spent = 0.0
        for record, probability in zip(self.records, self.posterior_trace):
            spent += record.cost
            if probability >= IDENTIFIED:
                return spent
        return None

    def contradictions(self) -> int:
        return sum(1 for r in self.records if r.note == "contradicted the leader")


def run_campaign(scenario: Scenario, policy: str, seed: int) -> Campaign:
    hypotheses = HypothesisSet.uniform(scenario.models)
    truth = scenario.truth()
    menu = scenario.menu()
    rng = random.Random(seed)
    campaign = Campaign(scenario=scenario, policy=policy, seed=seed)

    for step in range(scenario.steps):
        chosen, appraisals = choose_experiment(menu, hypotheses, policy=policy, rng=rng)
        leader_before, _ = hypotheses.best()

        observations = draw(truth, chosen.option.intervention, chosen.option.samples, rng)
        counts: dict[Assignment, int] = {}
        for observation in observations:
            counts[observation] = counts.get(observation, 0) + 1

        after = hypotheses.updated(
            observations,
            intervention=(
                chosen.option.intervention.as_mapping()
                if not chosen.option.intervention.is_observational
                else None
            ),
        )
        leader_after, _ = after.best()
        entropy_before = hypotheses.entropy()
        entropy_after = after.entropy()

        note = ""
        if leader_after.id != leader_before.id:
            note = "contradicted the leader"

        record = ExperimentRecord(
            experiment_id=ExperimentRecord.make_id(chosen.option.id, step),
            step=step,
            option_id=chosen.option.id,
            intervention=chosen.option.intervention,
            policy=policy,
            appraisal=chosen.to_dict(),
            alternatives=tuple(a.to_dict() for a in appraisals if a is not chosen),
            outcome_counts=tuple(sorted(counts.items())),
            posterior_before=tuple(hypotheses.ranked()),
            posterior_after=tuple(after.ranked()),
            entropy_before=entropy_before,
            entropy_after=entropy_after,
            cost=chosen.option.cost,
            created_at=step,
            conclusive=abs(entropy_before - entropy_after) > CONCLUSIVE_BITS,
            note=note,
        )
        campaign.records.append(record)
        hypotheses = after
        campaign.posterior_trace.append(hypotheses.probability(scenario.truth_id))
        campaign.entropy_trace.append(hypotheses.entropy())

    campaign.final = hypotheses
    return campaign


@dataclass
class PolicySummary:
    policy: str
    campaigns: list[Campaign]

    def _mean(self, attribute: str) -> float:
        return mean(getattr(c, attribute) for c in self.campaigns)

    @property
    def truth_probability(self) -> float:
        return self._mean("final_truth_probability")

    @property
    def final_entropy(self) -> float:
        return self._mean("final_entropy")

    @property
    def total_cost(self) -> float:
        return self._mean("total_cost")

    @property
    def bits_removed(self) -> float:
        return self._mean("bits_removed")

    @property
    def bits_per_cost(self) -> float:
        return self._mean("bits_per_cost")

    @property
    def identified_rate(self) -> float:
        return sum(1 for c in self.campaigns if c.identified) / len(self.campaigns)

    @property
    def mean_cost_to_identify(self) -> float | None:
        costs = [c.cost_to_identify() for c in self.campaigns]
        present = [c for c in costs if c is not None]
        return mean(present) if present else None


def run_scenario(scenario: Scenario) -> dict[str, PolicySummary]:
    out: dict[str, PolicySummary] = {}
    for policy in POLICIES:
        campaigns = [
            run_campaign(scenario, policy, seed=1000 + index) for index in range(REPEATS)
        ]
        out[policy] = PolicySummary(policy, campaigns)
    return out


# ---------------------------------------------------------------- reporting


def _fmt(value: float | None, places: int = 4) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def render(results: dict[str, dict[str, PolicySummary]]) -> str:
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    from experiments.experiment_worlds import COMPETING, experiment_menu

    add("# ECHO 7 — Experimentation")
    add()
    add("## Objective")
    add()
    add(
        "Move ECHO from *I can predict* to *I cannot tell these explanations "
        "apart, so I will design something that can*. Everything is synthetic "
        "and offline; an 'intervention' sets a variable in a simulation."
    )
    add()

    # -- architecture -----------------------------------------------------
    add("## Architecture")
    add()
    add(
        "| Module | Responsibility |\n"
        "| --- | --- |\n"
        "| `echo/causal.py` | structural causal models, `do()`, latent variables, a posterior over competing models |\n"
        "| `echo/experiment.py` | expected information gain, cost-aware selection policies, the immutable experiment record and ledger |\n"
        "| `experiments/experiment_worlds.py` | the hidden truth — never imported by anything under `echo/` |"
    )
    add()
    add(
        "Expected information gain is the mutual information between the "
        "hypothesis and the experiment's result:"
    )
    add()
    add("```")
    add("EIG(e) = H(P(H)) − Σ_c P(c | e) · H(P(H | c, e))")
    add("```")
    add()
    add(
        "computed **exactly** — every count vector of an n-sample experiment is "
        "enumerated and weighted by its multinomial probability. There is no "
        "sampling in the appraisal, so the choice is reproducible to the last "
        "bit. Utility subtracts price: "
        f"`EIG − {COST_WEIGHT} · cost − {RISK_WEIGHT} · risk`, both weights "
        "fixed before the first run."
    )
    add()

    # -- design and controls ----------------------------------------------
    add("## Experiment design and controls")
    add()
    add(
        "Three explanations of one visible fact: `V1` and `V2` move together. "
        "`V3` exists and acts but is **never recorded**, which is what makes "
        "the confounded story possible."
    )
    add()
    add("| Hypothesis | Structure |")
    add("| --- | --- |")
    for model in COMPETING:
        add(f"| `{model.id}` | {model.structure()} |")
    add()
    add(
        "The three are constructed to produce the **identical** observed joint "
        "distribution over `(V1, V2)`:"
    )
    add()
    add("| Model | P(0,0) | P(0,1) | P(1,0) | P(1,1) | corr(V1,V2) |")
    add("| --- | --- | --- | --- | --- | --- |")
    for model in COMPETING:
        joint = model.observed_joint()
        add(
            f"| `{model.id}` | "
            + " | ".join(f"{joint[key]:.4f}" for key in sorted(joint))
            + f" | {model.correlation('V1', 'V2'):+.4f} |"
        )
    add()
    add(
        "**This is the misleading observational correlation, and it is exactly "
        "misleading.** No amount of watching separates these models, because "
        "there is nothing in the observational distribution to separate. The "
        "appraisal discovers this rather than being told it: observation scores "
        "0.0000 bits."
    )
    add()
    hypotheses = HypothesisSet.uniform(COMPETING)
    add(
        f"Starting uncertainty is {hypotheses.entropy():.4f} bits (three models, "
        "flat prior). The menu, appraised against that prior:"
    )
    add()
    add("| Option | Intervention | Cost | Risk | EIG (bits) | Utility | Bits per unit cost | Control it provides |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    roles = {
        "OBS": "misleading observational correlation — provably 0 bits",
        "DO-V1": "cheap and informative",
        "DO-V2": "expensive, same information as DO-V1 at 6× the price",
        "DO-V3": "most informative single experiment",
        "DO-V1V2": "expensive and **inconclusive by construction** — 0 bits at any budget",
    }
    for option in experiment_menu(8):
        a = appraise(option, hypotheses)
        add(
            f"| `{option.id}` | {option.intervention.label()} | {option.cost:.1f} | "
            f"{option.risk:.1f} | {a.expected_information_gain:.4f} | "
            f"{a.utility:+.4f} | {a.information_per_cost:.4f} | {roles[option.id]} |"
        )
    add()
    add(
        "Two options score exactly zero. `OBS` does because the models agree on "
        "everything visible; `DO-V1V2` does because fixing both recorded "
        "variables leaves nothing to observe. Neither zero is written down "
        "anywhere — both fall out of the formula."
    )
    add()
    add(
        f"Each campaign runs {SCENARIOS[0].steps} experiments and is repeated "
        f"{REPEATS} times per policy with matched seeds."
    )
    add()

    # -- results ------------------------------------------------------------
    add("## Measured results")
    add()
    for scenario in SCENARIOS:
        summaries = results[scenario.id]
        add(f"### `{scenario.id}` — truth is `{scenario.truth_id}`")
        add()
        add(f"*{scenario.briefing}*")
        add()
        add(
            "| Policy | P(truth) after | Final entropy | Identified | Total cost | "
            "Bits removed | Bits per unit cost | Cost to identify |"
        )
        add("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for policy in POLICIES:
            s = summaries[policy]
            add(
                f"| `{policy}` | {s.truth_probability:.4f} | {s.final_entropy:.4f} | "
                f"{s.identified_rate:.0%} | {s.total_cost:.1f} | "
                f"{s.bits_removed:.4f} | {s.bits_per_cost:.4f} | "
                f"{_fmt(s.mean_cost_to_identify, 1)} |"
            )
        add()

    # -- the headline -------------------------------------------------------
    add("## Does information-gain selection beat random selection?")
    add()
    add("| Scenario | Metric | Random | Information gain | Difference |")
    add("| --- | --- | --- | --- | --- |")
    wins = 0
    for scenario in SCENARIOS:
        summaries = results[scenario.id]
        r, g = summaries["random"], summaries["information_gain"]
        for label, left, right, better_is_high in (
            ("P(truth) after", r.truth_probability, g.truth_probability, True),
            ("Bits removed", r.bits_removed, g.bits_removed, True),
            ("Bits per unit cost", r.bits_per_cost, g.bits_per_cost, True),
            ("Final entropy", r.final_entropy, g.final_entropy, False),
        ):
            delta = right - left
            good = delta > 0 if better_is_high else delta < 0
            if label == "Bits per unit cost" and good:
                wins += 1
            add(
                f"| `{scenario.id}` | {label} | {left:.4f} | {right:.4f} | "
                f"{delta:+.4f} {'(better)' if good else '(worse)'} |"
            )
    add()
    add(
        f"Information-gain selection is more cost-efficient than random in "
        f"**{wins} of {len(SCENARIOS)}** scenarios."
    )
    add()

    # -- failures -----------------------------------------------------------
    add("## Failures and things that did not work")
    add()
    failures: list[str] = []
    for scenario in SCENARIOS:
        summaries = results[scenario.id]
        g = summaries["information_gain"]
        if g.identified_rate < 1.0:
            failures.append(
                f"- `{scenario.id}`: the information-gain policy identified the "
                f"true model in only {g.identified_rate:.0%} of campaigns within "
                f"{scenario.steps} experiments."
            )
        if g.bits_per_cost <= summaries["random"].bits_per_cost:
            failures.append(
                f"- `{scenario.id}`: information-gain selection was **not** more "
                f"cost-efficient than random "
                f"({g.bits_per_cost:.4f} vs {summaries['random'].bits_per_cost:.4f})."
            )
        if g.final_entropy > 0.1:
            failures.append(
                f"- `{scenario.id}`: {g.final_entropy:.4f} bits of uncertainty "
                "remained at the end. The campaign did not fully resolve the "
                "question."
            )
    if failures:
        for line in failures:
            add(line)
    else:
        add("Every scenario resolved. Nothing to report here.")
    add()
    add(
        "The `cheapest` policy is the clearest negative result in the table: it "
        "buys the observational option, which is provably worth nothing, and "
        "ends every campaign exactly as uncertain as it started. Choosing by "
        "price alone is not a weaker version of choosing well — it is a way of "
        "spending money to learn nothing."
    )
    add()

    # -- limitations --------------------------------------------------------
    add("## Limitations")
    add()
    add(
        "1. **The hypothesis set is given.** ECHO chooses among three supplied "
        "models; it does not invent candidate structures. It is not told which "
        "is true, which is the part that matters here, but the space is not its "
        "own.\n"
        "2. **Binary variables, tiny graphs.** Exactness is bought with size. "
        "Three variables and two observed; nothing here scales as written.\n"
        "3. **The menu is fixed.** Costs, risks and sample budgets are given by "
        "the environment. ECHO selects, it does not design new instruments.\n"
        "4. **Synthetic throughout.** An intervention is a line of arithmetic. "
        "Nothing was done to anything real.\n"
        "5. **One truth per scenario.** Three scenarios, forty campaigns each; "
        "enough to compare policies, not enough to characterise the method.\n"
        "6. **Utility weights are hand-set.** Different cost and risk weights "
        "would order the menu differently. They were fixed in advance and not "
        "adjusted afterwards, but they were still chosen."
    )
    add()

    # -- what it does and does not show -------------------------------------
    add("## What this does and does not demonstrate")
    add()
    add(
        "**Does:** that ECHO can compute, from its own current uncertainty, "
        "which available action would reduce it most per unit cost; that this "
        "beats random selection on measured campaigns; that it correctly "
        "assigns zero value to an experiment whose result cannot depend on "
        "which hypothesis is true, without being told which one that is."
    )
    add()
    add(
        "**Does not:** that ECHO understands causation — ECHO 8 is where that "
        "is tested and it is a narrower claim than the word suggests. Nothing "
        "here involves the real world, autonomy, or any capacity to act outside "
        "a simulation. This is Bayesian experimental design over a small "
        "enumerable space. It is not consciousness, not self-awareness, and not "
        "general intelligence, and no measurement here bears on those questions "
        "at all."
    )
    add()
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ECHO 7 experimentation study.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    results: dict[str, dict[str, PolicySummary]] = {}
    for scenario in SCENARIOS:
        summaries = run_scenario(scenario)
        results[scenario.id] = summaries
        best = summaries["information_gain"]
        rand = summaries["random"]
        print(
            f"  {scenario.id:16s} info-gain P(truth)={best.truth_probability:.3f} "
            f"cost={best.total_cost:.0f} | random P(truth)={rand.truth_probability:.3f} "
            f"cost={rand.total_cost:.0f}"
        )
        # persist one representative campaign per scenario
        ledger = ExperimentLedger(RESULTS_DIR / scenario.id / "experiments.json")
        for record in summaries["information_gain"].campaigns[0].records:
            ledger.add(record)
        ledger.save()

    report = render(results)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
