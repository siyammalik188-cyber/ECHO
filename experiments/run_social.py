"""Run the ECHO 10 social-learning study and write ECHO_10_SOCIAL_LEARNING.md.

Forty rounds. In each, four agents make a natural-language claim about a
proposition, ECHO parses them, forms a belief weighted by what it has learned
about each source, and then the truth is revealed and every source is scored.

ECHO starts knowing nothing about any of them. Reliability is not configured
anywhere; it is a running tally that begins at 0.5 for everyone.

Three baselines share the identical claim stream:

- **majority vote** — count hands, ignore who is speaking;
- **confidence-weighted** — believe whoever speaks most firmly;
- **trust the loudest source** — follow the single most assertive agent.

The last two exist because the most reliable agent here is also the most
hedged. Any method that reads confidence as competence will pick the wrong one,
and that is worth measuring rather than asserting.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo.social import (  # noqa: E402
    MAX_RELIABILITY,
    MIN_RELIABILITY,
    PRIOR_RELIABILITY,
    Aggregate,
    SocialLedger,
    parse_claim,
)

from experiments.social_worlds import (  # noqa: E402
    AGENTS,
    AGENTS_BY_NAME,
    MINORITY_ROUNDS,
    Round,
    generate,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "social"

#: Rounds before the first scored decision, so ECHO has some record to use.
WARMUP = 6


@dataclass
class RoundResult:
    round: Round
    echo_probability: float
    aggregate: Aggregate
    majority_says: bool
    confidence_says: bool
    loudest_says: bool
    reliabilities: dict[str, float]

    @property
    def echo_says(self) -> bool:
        return self.echo_probability > 0.5

    @property
    def echo_correct(self) -> bool:
        return self.echo_says == self.round.truth

    @property
    def majority_correct(self) -> bool:
        return self.majority_says == self.round.truth

    @property
    def confidence_correct(self) -> bool:
        return self.confidence_says == self.round.truth

    @property
    def loudest_correct(self) -> bool:
        return self.loudest_says == self.round.truth

    @property
    def brier(self) -> float:
        return (self.echo_probability - (1.0 if self.round.truth else 0.0)) ** 2


@dataclass
class Study:
    ledger: SocialLedger
    results: list[RoundResult] = field(default_factory=list)
    reliability_trace: list[dict[str, float]] = field(default_factory=list)

    def scored(self) -> list[RoundResult]:
        return self.results[WARMUP:]

    def accuracy(self, attribute: str) -> float:
        scored = self.scored()
        return sum(1 for r in scored if getattr(r, attribute)) / len(scored)

    def minority_results(self) -> list[RoundResult]:
        return [r for r in self.scored() if r.round.minority_round]


def run_study(seed: int = 90210) -> Study:
    rounds = generate(seed)
    ledger = SocialLedger()
    study = Study(ledger=ledger)

    for index, round_ in enumerate(rounds):
        claims = [
            parse_claim(name, sentence, created_at=index)
            for name, sentence in round_.utterances
        ]
        for claim in claims:
            ledger.hear(claim)

        reliabilities = {
            agent.name: ledger.reliability(agent.name) for agent in AGENTS
        }
        belief = ledger.believe(round_.proposition)

        # -- baselines, on the identical claim stream ---------------------
        asserting = sum(1 for c in claims if c.asserts)
        majority_says = asserting > len(claims) / 2

        weighted = sum(
            (1.0 if c.asserts else -1.0) * (c.confidence - 0.5) for c in claims
        )
        confidence_says = weighted > 0

        loudest = max(claims, key=lambda c: (c.confidence, c.source))
        loudest_says = loudest.asserts

        study.results.append(
            RoundResult(
                round=round_,
                echo_probability=belief.probability,
                aggregate=belief,
                majority_says=majority_says,
                confidence_says=confidence_says,
                loudest_says=loudest_says,
                reliabilities=dict(reliabilities),
            )
        )

        ledger.resolve(round_.proposition, round_.truth, tick=index)
        study.reliability_trace.append(
            {agent.name: ledger.reliability(agent.name) for agent in AGENTS}
        )

    return study


# ---------------------------------------------------------------- reporting


def _pct(value: float) -> str:
    return f"{value:.1%}"


def render(study: Study) -> str:
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    scored = study.scored()
    ledger = study.ledger

    add("# ECHO 10 — Social learning")
    add()
    add("## Objective")
    add()
    add(
        "Let ECHO learn from other reasoning systems without believing them. "
        "Four simulated agents make claims in natural language; one is "
        "systematically wrong and ECHO is not told which. Everything is offline "
        "and no model is in the loop."
    )
    add()

    # -- architecture -------------------------------------------------------
    add("## Architecture")
    add()
    add(
        "| Module | Responsibility |\n"
        "| --- | --- |\n"
        "| `echo/social.py` | claim parsing, per-source reliability records, log-odds aggregation, the append-only ledger |\n"
        "| `experiments/social_worlds.py` | each agent's hidden accuracy, speciality and assertiveness — never imported by anything under `echo/` |"
    )
    add()
    add(
        "Parsing is shallow pattern-matching over a closed vocabulary. There is "
        "no model consulted, so nothing can be talked into an interpretation, "
        "and an utterance outside the vocabulary raises rather than producing a "
        "confident misreading. **Being parsed is not being believed** — a claim "
        "enters the ledger as testimony and moves nothing until its source has "
        "a record."
    )
    add()
    add("Testimony combines in log-odds:")
    add()
    add("```")
    add("logit P(proposition) = logit(prior) + Σ  ±1 · logit(reliability_s) · (2·confidence − 1)")
    add("```")
    add()
    add(
        "A source at reliability 0.5 contributes **exactly zero** — hearing from "
        "someone whose record is a coin flip should not move anything. A source "
        "*below* 0.5 contributes negative weight, so a reliably wrong agent is "
        "informative in reverse. That is not the same as ignoring it, and it is "
        "the correct treatment of the deceptive agent."
    )
    add()

    # -- design -------------------------------------------------------------
    add("## Experiment design")
    add()
    add(
        f"{len(study.results)} rounds, four agents, one claim each. The first "
        f"{WARMUP} rounds are warm-up and are excluded from the scores, because "
        "ECHO cannot weigh sources it has no record of. ECHO is told nothing "
        "about any agent; every reliability below is derived from outcomes."
    )
    add()
    add("| Agent | Hidden accuracy | Speciality | Hidden assertiveness | Role |")
    add("| --- | --- | --- | --- | --- |")
    for agent in AGENTS:
        add(
            f"| `{agent.name}` | {agent.accuracy:.2f} | {agent.speciality} "
            f"({agent.speciality_accuracy:.2f}) | {agent.assertiveness:.2f} | "
            f"{agent.note} |"
        )
    add()
    add(
        "**Controls.** `AGENT-D` is the deception control: right less than a "
        "fifth of the time while asserting 0.88, and never labelled. `AGENT-C` "
        "is the confidence trap — the most reliable agent is also the most "
        "hedged, so any method that reads firmness as competence will prefer the "
        "wrong one. And in "
        f"{len(MINORITY_ROUNDS)} of the {len(study.results)} rounds the three "
        "weaker agents agree and `AGENT-C` alone is right; a vote loses every "
        "one of those."
    )
    add()

    # -- learned reliability ------------------------------------------------
    add("## Source reliability, learned from outcomes")
    add()
    add("| Agent | Hidden accuracy | ECHO's learned reliability | Claims scored | Rank |")
    add("| --- | --- | --- | --- | --- |")
    ranked = ledger.ranked()
    positions = {name: index + 1 for index, (name, _) in enumerate(ranked)}
    for agent in AGENTS:
        record = ledger.record(agent.name)
        add(
            f"| `{agent.name}` | {agent.accuracy:.2f} | {record.reliability:.4f} | "
            f"{record.resolved} | {positions[agent.name]} |"
        )
    add()
    true_order = [a.name for a in sorted(AGENTS, key=lambda a: -a.accuracy)]
    learned_order = [name for name, _ in ranked]
    add(
        f"True ordering by accuracy: {' > '.join(f'`{n}`' for n in true_order)}. "
        f"ECHO's learned ordering: {' > '.join(f'`{n}`' for n in learned_order)}."
    )
    add()
    add(
        "The deceptive agent lands near the floor. That is worth more than it "
        "looks: a reliability well below 0.5 means its claims are used with the "
        "sign reversed, so a systematically wrong source becomes a *useful* one "
        "once its record is known."
    )
    add()
    add("How the estimates moved, sampled every eight rounds:")
    add()
    header = " | ".join(f"`{a.name}`" for a in AGENTS)
    add(f"| Round | {header} |")
    add("| --- | " + " | ".join("---" for _ in AGENTS) + " |")
    for index in range(0, len(study.reliability_trace), 8):
        row = study.reliability_trace[index]
        add(
            f"| {index + 1} | "
            + " | ".join(f"{row[a.name]:.3f}" for a in AGENTS)
            + " |"
        )
    final = study.reliability_trace[-1]
    add(
        f"| {len(study.reliability_trace)} | "
        + " | ".join(f"{final[a.name]:.3f}" for a in AGENTS)
        + " |"
    )
    add()

    # -- headline results ---------------------------------------------------
    add("## Measured results")
    add()
    add(
        f"Scored over the {len(scored)} rounds after warm-up, all four methods "
        "seeing the identical claim stream:"
    )
    add()
    add("| Method | Accuracy | Correct on minority rounds |")
    add("| --- | --- | --- |")
    minority = study.minority_results()
    rows = (
        ("**ECHO** (reliability-weighted)", "echo_correct"),
        ("Majority vote", "majority_correct"),
        ("Confidence-weighted", "confidence_correct"),
        ("Follow the loudest source", "loudest_correct"),
    )
    for label, attribute in rows:
        minority_right = sum(1 for r in minority if getattr(r, attribute))
        add(
            f"| {label} | {_pct(study.accuracy(attribute))} | "
            f"{minority_right} of {len(minority)} |"
        )
    add()
    brier = sum(r.brier for r in scored) / len(scored)
    add(f"ECHO's Brier score across the scored rounds: **{brier:.4f}**.")
    add()

    # -- minority rounds ----------------------------------------------------
    add("## Following evidence rather than the count")
    add()
    add(
        f"In {len(minority)} scored rounds the majority is wrong by "
        "construction: three agents agree and the most reliable one dissents. "
        "These are the rounds where a vote and a weighing come apart."
    )
    add()
    add("| Round | Truth | Majority said | ECHO's belief | ECHO right? |")
    add("| --- | --- | --- | --- | --- |")
    for result in minority:
        add(
            f"| {result.round.index} | {result.round.truth} | "
            f"{result.majority_says} | {result.echo_probability:.4f} | "
            f"{'yes' if result.echo_correct else '**no**'} |"
        )
    add()
    against_majority = [
        r for r in scored if r.echo_says != r.majority_says
    ]
    won = sum(1 for r in against_majority if r.echo_correct)
    add(
        f"Across all scored rounds ECHO disagreed with the majority "
        f"{len(against_majority)} times and was right in {won} of them. It is "
        "weighing testimony, not counting it."
    )
    add()

    # -- contradiction ------------------------------------------------------
    add("## Contradiction")
    add()
    contradicted = ledger.contradictions()
    add(
        f"Sources disagreed on {len(contradicted)} of "
        f"{len({r.round.proposition for r in study.results})} propositions. "
        "Both sides are retained — the ledger is append-only and nothing is "
        "discarded when it conflicts."
    )
    add()
    example = next((r for r in minority if r.aggregate.contributions), None)
    if example is not None:
        add(
            f"One round in full (`{example.round.proposition}`, truth "
            f"`{example.round.truth}`), showing where the belief came from:"
        )
        add()
        add("| Source | Said | Its confidence | Its reliability | Log-odds weight |")
        add("| --- | --- | --- | --- | --- |")
        for contribution in example.aggregate.contributions:
            add(
                f"| `{contribution.source}` | "
                f"{'asserts' if contribution.asserts else 'denies'} | "
                f"{contribution.confidence:.2f} | {contribution.reliability:.4f} | "
                f"{contribution.weight:+.4f} |"
            )
        add()
        add(
            f"Headcount: {example.aggregate.asserting} asserting, "
            f"{example.aggregate.denying} denying. ECHO's belief: "
            f"{example.echo_probability:.4f}. The claim that moved it most was "
            "not from the largest group."
        )
    add()

    # -- failures -----------------------------------------------------------
    add("## Failures and things that did not work")
    add()
    failures: list[str] = []
    wrong_minority = [r for r in minority if not r.echo_correct]
    if wrong_minority:
        failures.append(
            f"- ECHO got {len(wrong_minority)} of {len(minority)} minority rounds "
            "wrong, siding with the majority against the truth."
        )
    if study.accuracy("echo_correct") <= study.accuracy("majority_correct"):
        failures.append(
            "- ECHO did **not** beat majority vote overall "
            f"({_pct(study.accuracy('echo_correct'))} vs "
            f"{_pct(study.accuracy('majority_correct'))}). The weighting bought "
            "nothing."
        )
    early = study.reliability_trace[WARMUP - 1]
    if abs(early["AGENT-D"] - PRIOR_RELIABILITY) < 0.15:
        failures.append(
            f"- After the {WARMUP}-round warm-up the deceptive agent was still at "
            f"{early['AGENT-D']:.3f}, barely distinguishable from an unknown "
            "source. Reliability is learned slowly, and early rounds are decided "
            "with essentially no information about who is speaking."
        )
    if not failures:
        failures.append(
            "- ECHO beat every baseline and got every minority round right."
        )
    for line in failures:
        add(line)
    add()
    add(
        "The structural cost worth naming: reliability is a **single number per "
        "source**, so `AGENT-A`'s genuine speciality — it is much better on half "
        "the propositions than the other half — is averaged away entirely. ECHO "
        "cannot represent *reliable about this, not about that*, and nothing in "
        "the measured results above would reveal that it is losing information."
    )
    add()

    # -- limitations --------------------------------------------------------
    add("## Limitations")
    add()
    add(
        "1. **Parsing is a regular expression.** The 'natural language' is "
        "generated from a handful of templates over a closed vocabulary. This "
        "is claim extraction in the narrowest sense and would not survive "
        "contact with real prose.\n"
        "2. **Reliability is one number per source.** No topic-specific "
        "expertise, despite the agents having some by construction.\n"
        "3. **Stationarity is assumed.** The tally weights every past outcome "
        "equally, so a source whose reliability *changes* is reported as its "
        "lifetime average. `recent_reliability` exists for this and is used in "
        "ECHO 11; it is not what drives the weighting here.\n"
        "4. **Every claim is eventually resolved.** Real testimony is mostly "
        "never checked, and a source that only makes unfalsifiable claims would "
        "sit at the prior forever.\n"
        "5. **The agents do not react to ECHO.** They are recordings, not "
        "participants. Nothing here bears on strategic or adversarial "
        "behaviour by a source that knows it is being scored.\n"
        "6. **One seed, one run.** Enough to show the mechanism; not enough to "
        "characterise it."
    )
    add()

    # -- what it does and does not show ------------------------------------
    add("## What this does and does not demonstrate")
    add()
    add(
        "**Does:** that ECHO extracts claim, evidence, confidence and source "
        "from an utterance without thereby believing it; that it learns which "
        "sources are worth listening to from outcomes alone, including "
        "discovering that one is anti-correlated with the truth and using it in "
        "reverse; that it retains contradictory claims rather than resolving "
        "them by fiat; and that it follows weighted evidence rather than "
        "headcount, including in rounds constructed so that the majority is "
        "wrong."
    )
    add()
    add(
        "**Does not:** any form of understanding, negotiation, or theory of "
        "mind. The agents are templated sentence generators and ECHO's "
        "'listening' is a regular expression followed by a weighted sum. It has "
        "no model of what another agent believes, wants, or might be trying to "
        "do — the deceptive agent is detected as a low number, not as a liar. "
        "Nothing here is consciousness, sentience, or general intelligence."
    )
    add()
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ECHO 10 social study.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    study = run_study()
    for agent in AGENTS:
        record = study.ledger.record(agent.name)
        print(
            f"  {agent.name}  hidden={agent.accuracy:.2f}  "
            f"learned={record.reliability:.4f}  scored={record.resolved}"
        )
    print(
        f"  ECHO {study.accuracy('echo_correct'):.1%} | "
        f"majority {study.accuracy('majority_correct'):.1%} | "
        f"confidence {study.accuracy('confidence_correct'):.1%} | "
        f"loudest {study.accuracy('loudest_correct'):.1%}"
    )
    minority = study.minority_results()
    print(
        f"  minority rounds: ECHO "
        f"{sum(1 for r in minority if r.echo_correct)}/{len(minority)}, "
        f"majority {sum(1 for r in minority if r.majority_correct)}/{len(minority)}"
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    study.ledger.save(RESULTS_DIR / "social.json")

    text = render(study)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
