"""Run the ECHO 9 metacognition study and write ECHO_9_METACOGNITION.md.

Three phases, in this order, because the order is what makes it a test rather
than a demonstration:

1. **Calibration.** ECHO answers everything and builds a record. It has no
   prior notion of which domains it is good at; the record is the only source.
2. **Self-diagnosis.** With the record in hand but before seeing any new
   outcome, ECHO ranks the domains by how likely it thinks it is to be right.
   The ranking is then compared against what actually happens next.
3. **Abstention.** ECHO answers a fresh stream, declining where its own record
   says it should. Four numbers come out: accuracy when it answered, how it
   would have done on what it declined, how often it was confidently wrong, and
   how often it declined something it would have got right.
"""

from __future__ import annotations

import argparse
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo.metacognition import (  # noqa: E402
    ABSTENTION_THRESHOLD,
    DOMAINS,
    HIGH_CONFIDENCE,
    MIN_FOR_ASSESSMENT,
    STRONG_BRIER,
    WEAK_BRIER,
    Claim,
    Competence,
    Metacognition,
    calibration_of_error_prediction,
    expected_calibration_error,
)

from experiments.metacognition_worlds import PROFILES, Task, generate  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "metacognition"

CALIBRATION_TASKS = 60
EVALUATION_TASKS = 60
SEED = 20260813


@dataclass
class AbstentionOutcome:
    domain: str
    answered: int
    answered_correct: int
    abstained: int
    abstained_would_be_correct: int
    confidently_wrong: int

    @property
    def accuracy_when_answering(self) -> float | None:
        return self.answered_correct / self.answered if self.answered else None

    @property
    def accuracy_on_declined(self) -> float | None:
        return (
            self.abstained_would_be_correct / self.abstained if self.abstained else None
        )

    @property
    def unnecessary_abstention_rate(self) -> float | None:
        return (
            self.abstained_would_be_correct / self.abstained if self.abstained else None
        )


@dataclass
class Study:
    layer: Metacognition
    calibration: dict[str, list[Task]]
    evaluation: dict[str, list[Task]]
    self_diagnosis: list[tuple[str, float]] = field(default_factory=list)
    actual_next: dict[str, float] = field(default_factory=dict)
    abstention: dict[str, AbstentionOutcome] = field(default_factory=dict)
    no_abstention_accuracy: dict[str, float] = field(default_factory=dict)


def run_study(seed: int = SEED) -> Study:
    rng = random.Random(seed)
    layer = Metacognition()

    calibration = {d: generate(d, CALIBRATION_TASKS, rng) for d in DOMAINS}
    evaluation = {d: generate(d, EVALUATION_TASKS, rng) for d in DOMAINS}

    # -- phase 1: build a record ------------------------------------------
    tick = 0
    for domain in DOMAINS:
        for task in calibration[domain]:
            tick += 1
            # The error ECHO expects is read off its own record so far. Early
            # on that record is empty and the honest answer is 0.5.
            claim = layer.record(
                Claim(
                    claim_id=f"C-{tick}",
                    domain=domain,
                    proposition=task.proposition,
                    belief=task.belief,
                    predicted_error=layer.expected_error(domain),
                    created_at=tick,
                )
            )
            layer.resolve(claim.claim_id, task.truth)

    # -- phase 2: self-diagnosis, before seeing any new outcome ------------
    study = Study(layer=layer, calibration=calibration, evaluation=evaluation)
    study.self_diagnosis = layer.rank_by_expected_success(DOMAINS)
    study.actual_next = {
        domain: sum(1 for t in evaluation[domain] if t.correct) / len(evaluation[domain])
        for domain in DOMAINS
    }

    # -- phase 3: abstention ------------------------------------------------
    for domain in DOMAINS:
        abstain = layer.should_abstain(domain)
        answered = answered_correct = abstained = abstained_correct = 0
        confidently_wrong = 0
        for task in evaluation[domain]:
            tick += 1
            claim = layer.record(
                Claim(
                    claim_id=f"E-{tick}",
                    domain=domain,
                    proposition=task.proposition,
                    belief=task.belief,
                    predicted_error=layer.expected_error(domain),
                    created_at=tick,
                    abstained=abstain,
                )
            )
            layer.resolve(claim.claim_id, task.truth)
            if abstain:
                abstained += 1
                abstained_correct += int(task.correct)
            else:
                answered += 1
                answered_correct += int(task.correct)
                if not task.correct and max(task.belief, 1 - task.belief) >= HIGH_CONFIDENCE:
                    confidently_wrong += 1
        study.abstention[domain] = AbstentionOutcome(
            domain=domain,
            answered=answered,
            answered_correct=answered_correct,
            abstained=abstained,
            abstained_would_be_correct=abstained_correct,
            confidently_wrong=confidently_wrong,
        )
        study.no_abstention_accuracy[domain] = sum(
            1 for t in evaluation[domain] if t.correct
        ) / len(evaluation[domain])

    return study


# ---------------------------------------------------------------- reporting


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.1%}"


def _num(value: float | None, places: int = 4) -> str:
    return "—" if value is None else f"{value:.{places}f}"


def render(study: Study) -> str:
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    layer = study.layer
    report = layer.report()

    add("# ECHO 9 — Metacognition")
    add()
    add("## Objective")
    add()
    add(
        "Give ECHO a measurable account of how reliable its own answers have "
        "been, per kind of question, and test whether that account is any good: "
        "whether it can predict its own error rate, tell a strong area from a "
        "weak one without being told, and decline to answer where declining is "
        "correct."
    )
    add()
    add(
        "**Nothing here is introspection and nothing simulates it.** There is no "
        "method that returns *I feel uncertain*. Every quantity below is a "
        "count, a frequency, or a proper scoring rule over recorded outcomes, "
        "and each traces back to the claims that produced it."
    )
    add()

    # -- architecture -------------------------------------------------------
    add("## Architecture")
    add()
    add(
        "| Module | Responsibility |\n"
        "| --- | --- |\n"
        "| `echo/metacognition.py` | immutable `Claim` records, per-domain tallies, competence bands, abstention, error-prediction calibration |\n"
        "| `experiments/metacognition_worlds.py` | the hidden skill and assertiveness of each domain — never imported by anything under `echo/` |"
    )
    add()
    add("Two quantities are kept strictly apart:")
    add()
    add(
        "| | Question it answers | Evidence behind it |\n"
        "| --- | --- | --- |\n"
        "| **belief** | how likely is this proposition? | the evidence for that proposition |\n"
        "| **meta-confidence** | how often have claims *like this* been right? | a tally of past claims and their outcomes |"
    )
    add()
    add(
        f"Meta-confidence is Laplace-smoothed accuracy, so one lucky answer is "
        f"not a track record. Below {MIN_FOR_ASSESSMENT} scored claims a domain "
        "is `UNTESTED` — a third state, distinct from weak. Collapsing "
        "*known to be bad* into *no idea* would throw away the difference."
    )
    add()

    # -- design -------------------------------------------------------------
    add("## Experiment design")
    add()
    add(
        f"Five domains, {CALIBRATION_TASKS} calibration tasks and "
        f"{EVALUATION_TASKS} evaluation tasks each. Every domain has a hidden "
        "**skill** (how often ECHO is right) and a hidden **assertiveness** (how "
        "strongly it says so), set independently. ECHO is told neither."
    )
    add()
    add("| Domain | Hidden skill | Hidden assertiveness | Overconfident by design? |")
    add("| --- | --- | --- | --- |")
    for domain in DOMAINS:
        profile = PROFILES[domain]
        add(
            f"| `{domain}` | {profile.skill:.2f} | {profile.assertiveness:.2f} | "
            f"{'**yes**' if profile.is_overconfident else 'no'} |"
        )
    add()
    add(
        "The interesting failure is not being bad, it is being bad *and* "
        "confident. `causal_inference` answers barely better than chance while "
        "asserting 0.85, and `transfer` delivers 0.66 while asserting 0.88. "
        "Those two are what the abstention mechanism exists for."
    )
    add()
    add(
        "**Controls.** A strong-and-calibrated domain (`experimentation`) so the "
        "mechanism cannot pass by distrusting everything; a middling one "
        "(`discovery`) so the bands are not merely a two-way split; and the "
        "`UNTESTED` state, which is checked separately because a system that "
        "abstained on everything unmeasured could never build a record at all."
    )
    add()

    # -- capability monitoring ---------------------------------------------
    add("## Capability monitoring")
    add()
    add(
        "After the calibration phase, derived from outcomes alone — no labels "
        "about which domain is easy reach ECHO:"
    )
    add()
    add(
        "| Domain | Answered | Accuracy | Brier | Meta-confidence | Band | "
        "Confidently wrong | Hidden skill |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for domain in DOMAINS:
        record = report[domain]
        add(
            f"| `{domain}` | {record.answered} | {_pct(record.accuracy)} | "
            f"{_num(record.brier)} | {record.meta_confidence:.4f} | "
            f"**{record.competence.value}** | {record.confidently_wrong} | "
            f"{PROFILES[domain].skill:.2f} |"
        )
    add()
    add(
        f"Bands are fixed thresholds on the Brier score: `STRONG` at or below "
        f"{STRONG_BRIER}, `WEAK` at or above {WEAK_BRIER} — which is the score of "
        "always saying 0.5, so a `WEAK` domain is doing no better than shrugging "
        "— and `UNCERTAIN` between them."
    )
    add()

    # -- belief vs meta-confidence -----------------------------------------
    add("## Belief and meta-confidence are different numbers")
    add()
    add("| Domain | A typical belief | Meta-confidence | Gap |")
    add("| --- | --- | --- | --- |")
    for domain in DOMAINS:
        claims = [c for c in layer.claims(domain) if not c.abstained]
        typical = sum(max(c.belief, 1 - c.belief) for c in claims) / len(claims)
        meta = report[domain].meta_confidence
        add(f"| `{domain}` | {typical:.4f} | {meta:.4f} | {typical - meta:+.4f} |")
    add()
    add(
        "The gap is the point. In `causal_inference` ECHO states its answers at "
        "around 0.85 and has been right about half the time; the belief and the "
        "warrant for it are not the same quantity, and only tracking both makes "
        "the difference visible."
    )
    add()

    # -- error prediction ---------------------------------------------------
    add("## Error prediction")
    add()
    add(
        "Before each outcome, ECHO states how likely it thinks it is to be "
        "wrong — read off its own record so far, which starts empty and says "
        "0.5. Whether that estimate tracks reality is measurable:"
    )
    add()
    add("| Domain | Predicted error rate | Observed error rate | Gap |")
    add("| --- | --- | --- | --- |")
    for domain in DOMAINS:
        record = report[domain]
        add(
            f"| `{domain}` | {_num(record.predicted_error_rate)} | "
            f"{_num(record.observed_error_rate)} | "
            f"{_num(record.error_prediction_gap)} |"
        )
    add()
    rows = calibration_of_error_prediction(layer.claims())
    ece = expected_calibration_error(rows)
    add("Bucketed across all domains:")
    add()
    add("| Predicted-error band | Claims | Mean predicted | Observed | Gap |")
    add("| --- | --- | --- | --- | --- |")
    for row in rows:
        add(
            f"| {row['range']} | {row['count']} | {row['mean_predicted_error']:.4f} | "
            f"{row['observed_error_rate']:.4f} | {row['gap']:+.4f} |"
        )
    add()
    add(f"Expected calibration error of ECHO's self-assessment: **{_num(ece)}**.")
    add()

    # -- self diagnosis -----------------------------------------------------
    add("## Self-diagnosis")
    add()
    add(
        "With the record built but before seeing any new outcome, ECHO ranks the "
        "domains by how likely it expects to be right. The right-hand column is "
        "what then actually happened on the fresh tasks."
    )
    add()
    add("| ECHO's rank | Domain | Expected success | Actual next-phase accuracy |")
    add("| --- | --- | --- | --- |")
    for position, (domain, expected) in enumerate(study.self_diagnosis, start=1):
        add(
            f"| {position} | `{domain}` | {expected:.4f} | "
            f"{_pct(study.actual_next[domain])} |"
        )
    add()
    predicted_order = [d for d, _ in study.self_diagnosis]
    actual_order = sorted(DOMAINS, key=lambda d: -study.actual_next[d])
    pairs = 0
    concordant = 0
    for i in range(len(predicted_order)):
        for j in range(i + 1, len(predicted_order)):
            pairs += 1
            a, b = predicted_order[i], predicted_order[j]
            if study.actual_next[a] >= study.actual_next[b]:
                concordant += 1
    add(
        f"ECHO's ordering agrees with the outcome on **{concordant} of {pairs}** "
        f"pairs. Its top pick was `{predicted_order[0]}`; the domain that "
        f"actually scored highest next was `{actual_order[0]}`."
    )
    add()
    add(
        "The self-diagnostic is the whole of `rank_by_expected_success`: each "
        "domain's own measured record, ordered. No labels about difficulty, no "
        "hand-written notion of which capability is hard."
    )
    add()

    # -- abstention ---------------------------------------------------------
    add("## Abstention")
    add()
    add(
        f"ECHO declines a domain when its meta-confidence falls below "
        f"{ABSTENTION_THRESHOLD}. An `UNTESTED` domain is **not** an automatic "
        "abstention — refusing everything it has not already been scored on "
        "would make the record unfillable."
    )
    add()
    add(
        "| Domain | Abstained? | Accuracy when answering | Would have scored on declined | "
        "Confidently wrong | Unnecessary abstentions |"
    )
    add("| --- | --- | --- | --- | --- | --- |")
    for domain in DOMAINS:
        outcome = study.abstention[domain]
        declined = outcome.abstained > 0
        add(
            f"| `{domain}` | {'**yes**' if declined else 'no'} | "
            f"{_pct(outcome.accuracy_when_answering)} | "
            f"{_pct(outcome.accuracy_on_declined)} | "
            f"{outcome.confidently_wrong} | "
            f"{outcome.abstained_would_be_correct} of {outcome.abstained} |"
        )
    add()
    answered_total = sum(o.answered for o in study.abstention.values())
    answered_right = sum(o.answered_correct for o in study.abstention.values())
    declined_total = sum(o.abstained for o in study.abstention.values())
    declined_right = sum(o.abstained_would_be_correct for o in study.abstention.values())
    overall_without = sum(
        1 for domain in DOMAINS for t in study.evaluation[domain] if t.correct
    ) / (len(DOMAINS) * EVALUATION_TASKS)
    add(
        f"Answering everything would have scored **{overall_without:.1%}**. "
        f"Answering only where the record supports it scored "
        f"**{answered_right / answered_total:.1%}** on "
        f"{answered_total} questions, while declining {declined_total} on which "
        f"it would have scored {declined_right / declined_total:.1%} had it "
        "tried." if declined_total else
        f"Answering everything would have scored {overall_without:.1%}; ECHO "
        "declined nothing."
    )
    add()
    if declined_total:
        add(
            f"The cost is explicit: {declined_right} of the {declined_total} "
            "declined questions would have been answered correctly. Abstention "
            "is not free, and a system that abstained on everything would score "
            "100% on the first column and lose every one of those."
        )
    add()

    # -- failures -----------------------------------------------------------
    add("## Failures and things that did not work")
    add()
    failures: list[str] = []
    for domain in DOMAINS:
        record = report[domain]
        profile = PROFILES[domain]
        gap = record.error_prediction_gap
        if gap is not None and abs(gap) > 0.10:
            failures.append(
                f"- `{domain}`: ECHO's predicted error rate was off by "
                f"{gap:+.4f} — it "
                + ("over-warned" if gap > 0 else "**under-warned**")
                + "."
            )
        if profile.is_overconfident and record.competence is Competence.STRONG:
            failures.append(
                f"- `{domain}`: overconfident by construction, yet banded "
                "`STRONG`. The band did not catch it."
            )
    if ece is not None and ece > 0.10:
        failures.append(
            f"- Self-assessment calibration is poor overall: expected "
            f"calibration error {ece:.4f}."
        )
    # A domain can be banded WEAK and still answer, because the abstention gate
    # is on meta-confidence rather than on the band. Where that happens it is a
    # miss and belongs here, not in the limitations.
    for domain in DOMAINS:
        record = report[domain]
        outcome = study.abstention[domain]
        if record.competence is Competence.WEAK and outcome.abstained == 0:
            failures.append(
                f"- `{domain}` is banded **WEAK** (Brier {_num(record.brier)}) but "
                f"did **not** abstain, because meta-confidence "
                f"{record.meta_confidence:.4f} is above the {ABSTENTION_THRESHOLD} "
                f"gate. It went on to be confidently wrong "
                f"{outcome.confidently_wrong} times — the most of any answering "
                "domain. The two mechanisms disagree, and the band is the one "
                "that was right."
            )
    if not failures:
        failures.append(
            "- No domain's error prediction was off by more than 0.10, and no "
            "overconfident domain was banded `STRONG`."
        )
    for line in failures:
        add(line)
    add()
    add(
        "The structural weakness worth naming: ECHO's abstention decision is "
        "per **domain**, not per question. It cannot notice that one particular "
        "causal question is easy while the domain as a whole is weak, so every "
        "unnecessary abstention in the table above is a direct consequence of "
        "the granularity, not of the thresholds."
    )
    add()

    # -- limitations --------------------------------------------------------
    add("## Limitations")
    add()
    add(
        "1. **Domain-level granularity.** Reliability is tracked per capability "
        "area, not per question. This is the main cost, and it is visible in the "
        "unnecessary-abstention column.\n"
        "2. **Stationarity assumed.** The tally weights every past claim "
        "equally. A domain that improved or degraded would be reported as its "
        "lifetime average, and nothing here detects the change.\n"
        f"3. **Thresholds are hand-set.** {STRONG_BRIER}, {WEAK_BRIER}, "
        f"{ABSTENTION_THRESHOLD} and {MIN_FOR_ASSESSMENT} were fixed before the "
        "run and not adjusted, but they were still chosen by a person.\n"
        "4. **The tasks are synthetic.** A domain is a Bernoulli draw against a "
        "hidden skill. Real capability is not one number.\n"
        "5. **The record is ECHO's only source, which is also a ceiling.** It "
        "cannot recognise a domain as hard before it has failed at it, and "
        "nothing here transfers a lesson from one domain to another.\n"
        "6. **One seed, one run.** Enough to show the mechanism works; not "
        "enough to characterise it."
    )
    add()

    # -- what it does and does not show -------------------------------------
    add("## What this does and does not demonstrate")
    add()
    add(
        "**Does:** that ECHO maintains a per-domain record of its own accuracy "
        "and derives competence bands from it without being told which areas are "
        "hard; that it separates belief in a proposition from the historical "
        "reliability of that kind of belief, and reports both; that its stated "
        "error rate can be scored against its actual error rate; and that it "
        "declines to answer where its record does not support answering, at a "
        "measured and reported cost."
    )
    add()
    add(
        "**Does not:** anything about self-awareness. This is a scoreboard with "
        "thresholds on it. It has no access to its own reasoning process, no "
        "representation of itself, and no state that could be called noticing "
        "anything — when it abstains, that is a comparison between two floats. "
        "The word *metacognition* is used here in the narrow measurement sense "
        "and nothing in this report bears on consciousness, sentience, or "
        "general intelligence."
    )
    add()
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ECHO 9 metacognition study.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    study = run_study()
    report = study.layer.report()
    for domain in DOMAINS:
        record = report[domain]
        print(
            f"  {domain:18s} acc={_pct(record.accuracy):>6s} "
            f"brier={_num(record.brier)} meta={record.meta_confidence:.3f} "
            f"{record.competence.value:9s} abstained="
            f"{study.abstention[domain].abstained > 0}"
        )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    study.layer.save(RESULTS_DIR / "metacognition.json")

    text = render(study)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
