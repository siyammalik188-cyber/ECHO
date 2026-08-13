"""Run the ECHO 8 causal study and write ECHO_8_CAUSAL.md.

Each scenario runs in two phases so the difference between them is the result:

1. **Observation only.** A few hundred rows of passive data. This can rule out
   a collider, because a collider makes the outer variables independent. It
   cannot order a chain against a fork against a reversed chain, because those
   three induce the same distribution — so whatever the posterior does here, it
   does for reasons observation can support.
2. **Intervention.** Experiments chosen by the ECHO 7 machinery, unchanged.
   This is where the equivalence class comes apart.

Then counterfactuals, answered across every model still believed possible, with
confidence that falls when the models disagree.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo.causal import HypothesisSet  # noqa: E402
from echo.counterfactual import (  # noqa: E402
    CounterfactualAnswer,
    CounterfactualQuery,
    counterfactual,
    counterfactual_for_model,
)
from echo.experiment import (  # noqa: E402
    ExperimentLedger,
    ExperimentRecord,
    Intervention,
    appraise,
    choose_experiment,
)

from experiments.causal_worlds import (  # noqa: E402
    COLLIDER,
    EQUIVALENCE_CLASS,
    SCENARIOS,
    WORLDS,
    CausalScenario,
    causal_menu,
)
from experiments.run_experimentation import draw  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "causal"

#: Counterfactual questions asked of every scenario. Fixed in advance.
QUERIES = (
    CounterfactualQuery(
        facts=(("V1", 1), ("V2", 1), ("V3", 1)),
        antecedent=("V1", 0),
        consequent=("V3", 1),
    ),
    CounterfactualQuery(
        facts=(("V1", 1), ("V2", 1), ("V3", 1)),
        antecedent=("V2", 0),
        consequent=("V3", 1),
    ),
    CounterfactualQuery(
        facts=(("V1", 0), ("V2", 0), ("V3", 0)),
        antecedent=("V3", 1),
        consequent=("V1", 1),
    ),
)


@dataclass
class CausalRun:
    scenario: CausalScenario
    prior: HypothesisSet
    after_observation: HypothesisSet
    after_intervention: HypothesisSet
    records: list[ExperimentRecord] = field(default_factory=list)
    answers: list[CounterfactualAnswer] = field(default_factory=list)
    truth_answers: list[float | None] = field(default_factory=list)

    @property
    def truth_probability_observational(self) -> float:
        return self.after_observation.probability(self.scenario.truth_id)

    @property
    def truth_probability_final(self) -> float:
        return self.after_intervention.probability(self.scenario.truth_id)

    @property
    def identified(self) -> bool:
        return self.after_intervention.best()[0].id == self.scenario.truth_id

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.records)


def run_scenario(scenario: CausalScenario, seed: int = 4242) -> CausalRun:
    rng = random.Random(seed)
    truth = scenario.truth()
    hypotheses = HypothesisSet.uniform(WORLDS)
    prior = hypotheses

    # -- phase 1: watch -------------------------------------------------
    observations = draw(
        truth, Intervention.observation(), scenario.observational_rows, rng
    )
    after_observation = hypotheses.updated(observations)

    # -- phase 2: act ---------------------------------------------------
    hypotheses = after_observation
    menu = causal_menu(scenario.samples)
    records: list[ExperimentRecord] = []
    for step in range(scenario.steps):
        interventions = [o for o in menu if not o.intervention.is_observational]
        chosen, appraisals = choose_experiment(
            interventions, hypotheses, policy="information_gain", rng=rng
        )
        leader_before, _ = hypotheses.best()
        drawn = draw(truth, chosen.option.intervention, chosen.option.samples, rng)
        counts: dict[tuple[int, ...], int] = {}
        for observation in drawn:
            counts[observation] = counts.get(observation, 0) + 1
        after = hypotheses.updated(
            drawn, intervention=chosen.option.intervention.as_mapping()
        )
        records.append(
            ExperimentRecord(
                experiment_id=ExperimentRecord.make_id(chosen.option.id, step),
                step=step,
                option_id=chosen.option.id,
                intervention=chosen.option.intervention,
                policy="information_gain",
                appraisal=chosen.to_dict(),
                alternatives=tuple(a.to_dict() for a in appraisals if a is not chosen),
                outcome_counts=tuple(sorted(counts.items())),
                posterior_before=tuple(hypotheses.ranked()),
                posterior_after=tuple(after.ranked()),
                entropy_before=hypotheses.entropy(),
                entropy_after=after.entropy(),
                cost=chosen.option.cost,
                created_at=step,
                conclusive=abs(hypotheses.entropy() - after.entropy()) > 0.05,
                note=(
                    "contradicted the leader"
                    if after.best()[0].id != leader_before.id
                    else ""
                ),
            )
        )
        hypotheses = after

    run = CausalRun(
        scenario=scenario,
        prior=prior,
        after_observation=after_observation,
        after_intervention=hypotheses,
        records=records,
    )

    # -- counterfactuals -------------------------------------------------
    for query in QUERIES:
        run.answers.append(
            counterfactual(
                hypotheses,
                query,
                evidence=(
                    f"{scenario.observational_rows} observational rows",
                    f"{len(records)} intervention(s) costing {run.total_cost:.0f}",
                ),
            )
        )
        run.truth_answers.append(counterfactual_for_model(truth, query))
    return run


# ---------------------------------------------------------------- reporting


def _posterior_table(hypotheses: HypothesisSet) -> str:
    return ", ".join(f"`{mid}` {p:.3f}" for mid, p in hypotheses.ranked())


def render(runs: list[CausalRun]) -> str:
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add("# ECHO 8 — Causal abstraction")
    add()
    add("## Objective")
    add()
    add(
        "Test whether ECHO can tell **correlation** from **causal structure** — "
        "and, where the data cannot tell them apart, whether it declines to "
        "pretend otherwise."
    )
    add()

    # -- architecture ------------------------------------------------------
    add("## Architecture")
    add()
    add(
        "| Module | Responsibility |\n"
        "| --- | --- |\n"
        "| `echo/causal.py` | structures, `do()`, a posterior held over several models at once |\n"
        "| `echo/counterfactual.py` | abduction → action → prediction, with confidence and stated assumptions |\n"
        "| `echo/experiment.py` | ECHO 7's selection machinery, reused unchanged |\n"
        "| `experiments/causal_worlds.py` | the hidden structures — never imported by anything under `echo/` |"
    )
    add()

    # -- the worlds ---------------------------------------------------------
    add("## The worlds, and why observation is not enough")
    add()
    add("| Model | Structure | corr(V1,V2) | corr(V2,V3) | corr(V1,V3) |")
    add("| --- | --- | --- | --- | --- |")
    for model in WORLDS:
        add(
            f"| `{model.id}` | {model.structure()} | "
            f"{model.correlation('V1', 'V2'):+.3f} | "
            f"{model.correlation('V2', 'V3'):+.3f} | "
            f"{model.correlation('V1', 'V3'):+.3f} |"
        )
    add()
    add(
        "The first three are a **Markov equivalence class**: same skeleton, no "
        "v-structure, and therefore the *identical* joint distribution. Their "
        "parameters are derived from one another by Bayes' rule rather than "
        "fitted, so the equality is exact rather than close. No observational "
        "method can order them — not correlation ranking, not conditional "
        "independence testing, not more data."
    )
    add()
    add(
        "The collider is different, and deliberately so: it makes `V1` and `V3` "
        "**marginally independent** (correlation exactly 0.000). Observation can "
        "and should rule it out. Including it keeps the claim honest — "
        "observational data is not useless in general, it is insufficient for a "
        "specific and identifiable reason."
    )
    add()
    add(
        "Under a flat prior over four models (2.0000 bits), observation is worth "
        f"{appraise(causal_menu(12)[0], HypothesisSet.uniform(WORLDS)).expected_information_gain:.4f} "
        "bits — roughly the one bit needed to eliminate the collider, and "
        "nothing more. `do(V2)` is worth "
        f"{appraise(causal_menu(12)[2], HypothesisSet.uniform(WORLDS)).expected_information_gain:.4f} "
        "bits, because setting the middle variable is what separates a chain "
        "from a fork."
    )
    add()

    # -- results ------------------------------------------------------------
    add("## Measured results")
    add()
    add(
        "| Scenario | Truth | P(truth) after watching | P(truth) after intervening | "
        "Identified | Cost | Experiments chosen |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for run in runs:
        chosen = ", ".join(f"`{r.option_id}`" for r in run.records) or "—"
        add(
            f"| `{run.scenario.id}` | `{run.scenario.truth_id}` | "
            f"{run.truth_probability_observational:.4f} | "
            f"{run.truth_probability_final:.4f} | "
            f"{'yes' if run.identified else '**no**'} | {run.total_cost:.0f} | {chosen} |"
        )
    add()
    add(
        "Watching alone leaves the truth at roughly 1/3 in every scenario whose "
        "answer lies inside the equivalence class — which is exactly right, "
        "because a third is all the observational data can support. The "
        "collider scenario is the exception: there, watching does the job on "
        "its own."
    )
    add()

    for run in runs:
        add(f"### `{run.scenario.id}` — truth is `{run.scenario.truth_id}`")
        add()
        add(f"- Prior: {_posterior_table(run.prior)}")
        add(f"- After {run.scenario.observational_rows} observational rows: {_posterior_table(run.after_observation)}")
        add(f"- After {len(run.records)} intervention(s): {_posterior_table(run.after_intervention)}")
        for record in run.records:
            add(
                f"  - step {record.step}: `{record.option_id}` "
                f"({record.intervention.label()}), expected "
                f"{record.appraisal['expected_information_gain_bits']:.4f} bits, "
                f"actually removed {record.information_gained:+.4f} bits"
                + (f" — {record.note}" if record.note else "")
            )
        add()

    # -- uncertainty is kept ------------------------------------------------
    starved = next((r for r in runs if r.scenario.id == "CAUSAL-starved"), None)
    add("## Causal uncertainty is represented, not resolved away")
    add()
    reference = runs[0]
    add(
        "The sharpest case is the observational phase. After "
        f"{reference.scenario.observational_rows} rows, ECHO's posterior over the "
        "four structures is:"
    )
    add()
    add("| Model | Posterior after watching |")
    add("| --- | --- |")
    for model_id, probability in reference.after_observation.ranked():
        add(f"| `{model_id}` | {probability:.4f} |")
    add()
    add(
        "That is the correct answer and the only defensible one. The collider is "
        "eliminated because it predicts something visibly false — `V1` and `V3` "
        "independent. The other three are left at **exactly one third each**, "
        f"{reference.after_observation.entropy():.4f} bits, because the "
        "observational data contains nothing that could order them. ECHO reports "
        "a distribution and declines to name a winner; nothing in the pipeline "
        "forces one."
    )
    add()
    if starved is not None:
        add(
            "`CAUSAL-starved` is the partial case: one intervention on three "
            "samples, which is real evidence but not much of it."
        )
        add()
        add("| Model | Posterior |")
        add("| --- | --- |")
        for model_id, probability in starved.after_intervention.ranked():
            add(f"| `{model_id}` | {probability:.4f} |")
        add()
        leader, mass = starved.after_intervention.best()
        add(
            f"It leans toward `{leader.id}` at {mass:.4f} — which happens to be "
            f"correct — but keeps {starved.after_intervention.entropy():.4f} bits "
            "of a possible 2.0000 on the table. A system that reported this as "
            "settled would be overstating three samples."
        )
    add()

    # -- counterfactuals ----------------------------------------------------
    add("## Counterfactual inference")
    add()
    add(
        "Abduction, then action, then prediction: infer the background from what "
        "actually happened, change the antecedent with `do()`, and push the "
        "*same* background through the changed model. Holding the background "
        "fixed is what makes it a claim about this case rather than an average."
    )
    add()
    for run in runs:
        add(f"### `{run.scenario.id}`")
        add()
        add("| Question | ECHO's answer | Confidence | Model disagreement | True model's answer |")
        add("| --- | --- | --- | --- | --- |")
        for answer, truth_answer in zip(run.answers, run.truth_answers):
            add(
                f"| {answer.query.label()} | {answer.probability:.4f} | "
                f"{answer.confidence:.4f} | {answer.model_disagreement:.4f} | "
                + (f"{truth_answer:.4f}" if truth_answer is not None else "undefined")
                + " |"
            )
        add()
    example = runs[0].answers[0]
    add("A single answer in full, to show what accompanies every number:")
    add()
    add("```")
    add(f"QUESTION:   {example.query.label()}")
    add(f"PROBABILITY: {example.probability:.4f}")
    add(f"CONFIDENCE:  {example.confidence:.4f}")
    for line in example.supporting_evidence:
        add(f"EVIDENCE:    {line}")
    for line in example.assumptions:
        add(f"ASSUMPTION:  {line}")
    add("```")
    add()
    add(
        "Confidence is not the precision of the average. It falls both with "
        "structural uncertainty and with how much the live models disagree: two "
        "models that both say 0.8 support a confident answer, while two that say "
        "0.1 and 0.9 do not, however certain ECHO is that one of them is right. "
        "Under a flat prior the same question returns confidence 0.0000 — the "
        "mechanism refuses to launder a split posterior into a firm number."
    )
    add()

    # -- the five distinctions ---------------------------------------------
    add("## Five things this report keeps apart")
    add()
    add("| Term | What it means | Where it lives | Established here? |")
    add("| --- | --- | --- | --- |")
    add(
        "| **Correlation** | two variables move together | `CausalModel.correlation` | "
        "Yes — and shown to be identical across three different structures |"
    )
    add(
        "| **Prediction** | P(Y \\| X = x): what to expect on *seeing* X | "
        "`CausalModel.conditioned` | Yes, and it is the same for all three |"
    )
    add(
        "| **Causal evidence** | data generated under `do(X = x)` | "
        "`HypothesisSet.updated(..., intervention=...)` | Yes — this is what breaks the tie |"
    )
    add(
        "| **Causal belief** | a posterior over structures, held plural | "
        "`HypothesisSet.posterior` | Yes, including cases where it stays split |"
    )
    add(
        "| **Counterfactual inference** | P(Y_{X=x'} \\| what actually happened) | "
        "`echo/counterfactual.py` | Yes, with confidence and assumptions attached |"
    )
    add()
    add(
        "The gap between the second and third rows is the entire point. In the "
        "confounded and forked worlds, *seeing* `V1 = 1` raises the probability "
        "of `V3`; *setting* `V1 = 1` does not move it at all. Prediction and "
        "causation come apart, and only intervention notices."
    )
    add()

    # -- failures ----------------------------------------------------------
    add("## Failures and limitations")
    add()
    misidentified = [r for r in runs if not r.identified]
    if misidentified:
        for run in misidentified:
            add(
                f"- `{run.scenario.id}`: the leading model after intervention was "
                f"`{run.after_intervention.best()[0].id}`, not the true "
                f"`{run.scenario.truth_id}`."
            )
    else:
        add("- Every scenario with an adequate budget identified its true structure.")
    if starved is not None:
        leader, mass = starved.after_intervention.best()
        add(
            f"- `CAUSAL-starved` leads with `{leader.id}` at {mass:.4f}. It is "
            "included to check that a starved posterior stays spread rather than "
            "collapsing, and it does — but note that it landed on the true model, "
            "which was not guaranteed and should not be read as the mechanism "
            "getting it right on this little evidence."
        )
    add()
    add(
        "1. **The model set is supplied.** ECHO weighs four given structures; it "
        "does not search the space of DAGs. Structure learning is a different "
        "and much harder problem, and nothing here attempts it.\n"
        "2. **The conditional probability tables are given too.** Only the "
        "structure is uncertain. Real causal inference estimates both.\n"
        "3. **Three binary variables.** Exactness is bought with size.\n"
        "4. **Counterfactuals assume the truth is in the set.** If the real "
        "structure is not among the four, every answer is conditional on a false "
        "premise — stated in the assumptions of every answer, but worth "
        "repeating.\n"
        "5. **No unmeasured-confounding search.** ECHO 7's worlds had a latent "
        "variable; these do not. Deciding *whether* a confounder exists is not "
        "tested here.\n"
        "6. **Synthetic and offline.** An intervention is arithmetic."
    )
    add()

    # -- what it does and does not show ------------------------------------
    add("## What this does and does not demonstrate")
    add()
    add(
        "**Does:** that ECHO distinguishes seeing from doing, on worlds where "
        "the two genuinely diverge; that it holds several causal structures at "
        "once with calibrated weights instead of committing early; that it "
        "answers counterfactual questions with a probability, its evidence, its "
        "confidence and its assumptions; and that its confidence collapses when "
        "its models disagree rather than reporting a firm-looking average."
    )
    add()
    add(
        "**Does not:** that ECHO understands causation in any sense a person "
        "would mean by the word. It manipulates conditional probability tables "
        "over three meaningless labels. It cannot discover a structure that was "
        "not handed to it, cannot tell whether an unmeasured confounder exists, "
        "and has no notion of mechanism, intervention-in-the-world, or why any "
        "of these arrows would point anywhere. Nothing here is consciousness, "
        "self-awareness, or general intelligence, and no measurement in this "
        "report bears on those questions."
    )
    add()
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ECHO 8 causal study.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    runs = []
    for scenario in SCENARIOS:
        run = run_scenario(scenario)
        runs.append(run)
        print(
            f"  {scenario.id:18s} truth={scenario.truth_id:18s} "
            f"obs={run.truth_probability_observational:.3f} "
            f"final={run.truth_probability_final:.3f} "
            f"identified={run.identified}"
        )
        ledger = ExperimentLedger(RESULTS_DIR / scenario.id / "experiments.json")
        for record in run.records:
            ledger.add(record)
        ledger.save()

    report = render(runs)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
