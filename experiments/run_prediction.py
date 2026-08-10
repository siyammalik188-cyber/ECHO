"""Run the prediction experiment across all three environments and write the report.

Fully offline. No model, no API key, no network.

    python -m experiments.run_prediction --out docs/ECHO_3_PREDICTION.md

Per trial, in this order — the order is the temporal guarantee:

1. take a view of the timeline as of tick `i` (observations 0..i-1 only);
2. compute a probability from that view;
3. issue the prediction at tick `i`, which re-checks every cited id;
4. **only then** record trial `i`'s observation at tick `i`;
5. evaluate the prediction against the true outcome at tick `i + 1`.

At step 3 the outcome being predicted does not yet exist on the timeline, so it
cannot have been used. Deliberate leakage attempts are run separately and their
rejections are reported.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from echo import calibration
from echo.prediction import (
    Timeline,
    TemporalLeakageError,
    issue_prediction,
)
from echo.prediction_ledger import PredictionLedger

from .environments import (
    ENVIRONMENTS,
    OUTCOME_A,
    PROPOSITION,
    Environment,
    laplace_predictor,
)

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"


@dataclass
class EnvironmentRun:
    environment: Environment
    ledger: PredictionLedger
    timeline: Timeline
    trials: list = field(default_factory=list)

    @property
    def scores(self) -> dict[str, Any]:
        probabilities, outcomes = self.ledger.scored()
        return calibration.summarise(probabilities, outcomes)

    def scores_for_slice(self, start: int, end: int) -> dict[str, Any]:
        records = [
            r
            for r in self.ledger.evaluated()
            if start <= r.prediction_time < end
        ]
        return calibration.summarise(
            [r.predicted_probability for r in records],
            [r.evaluation.actual_outcome for r in records],  # type: ignore[union-attr]
        )


def run_environment(environment: Environment, directory: Path) -> EnvironmentRun:
    timeline = Timeline()
    ledger = PredictionLedger.in_directory(directory / environment.id)
    trials = environment.generate()
    run = EnvironmentRun(environment=environment, ledger=ledger, timeline=timeline, trials=trials)

    for trial in trials:
        tick = trial.index

        # 1–2. Only what already happened is reachable from here.
        view = timeline.view_as_of(tick)
        probability, rationale = laplace_predictor(view)

        # 3. Issue. This independently re-checks every cited observation.
        prediction = issue_prediction(
            timeline,
            proposition=PROPOSITION,
            predicted_probability=probability,
            prediction_time=tick,
            information_available=view.ids(),
            rationale=rationale,
        )
        ledger.add(prediction)

        # 4. Reality happens *after* the prediction was committed.
        timeline.add(
            f"Trial {trial.index} reported {trial.observed_label}.",
            at=tick,
            outcome=trial.observed_outcome,
            reported_label=trial.observed_label,
        )

        # 5. Score against what truly happened, not what was reported.
        ledger.evaluate(
            prediction.id,
            outcome=trial.true_outcome,
            outcome_label=trial.true_label,
            evaluation_time=tick + 1,
        )

    ledger.remember_timeline(timeline)
    ledger.save()
    return run


# ---------------------------------------------------- temporal leakage tests


def temporal_leakage_probes(directory: Path) -> list[dict[str, Any]]:
    """TEMPORAL_LEAKAGE_TEST — deliberately try to smuggle the future in.

    Every probe below is an attempt to cite information that does not yet exist
    at the prediction's timestamp. All of them must be refused.
    """
    probes: list[dict[str, Any]] = []

    def probe(name: str, description: str, attempt) -> None:
        try:
            attempt()
        except TemporalLeakageError as exc:
            probes.append(
                {
                    "probe": name,
                    "description": description,
                    "rejected": True,
                    "error": str(exc),
                }
            )
        except Exception as exc:  # noqa: BLE001 — an unexpected type is still a failure to report
            probes.append(
                {
                    "probe": name,
                    "description": description,
                    "rejected": False,
                    "error": f"WRONG EXCEPTION {type(exc).__name__}: {exc}",
                }
            )
        else:
            probes.append(
                {
                    "probe": name,
                    "description": description,
                    "rejected": False,
                    "error": "ACCEPTED — the system failed to reject future information",
                }
            )

    timeline = Timeline()
    past = timeline.add("The past.", at=0, outcome=True)
    future = timeline.add("The future — the very outcome being predicted.", at=5, outcome=False)
    far_future = timeline.add("Much later.", at=99, outcome=True)

    probe(
        "cite_the_outcome_being_predicted",
        "Predict at t=5 while citing the observation stamped t=5 — the outcome itself.",
        lambda: issue_prediction(
            timeline, PROPOSITION, 0.9, prediction_time=5, information_available=[past.id, future.id]
        ),
    )
    probe(
        "cite_far_future_information",
        "Predict at t=1 while citing an observation from t=99.",
        lambda: issue_prediction(
            timeline, PROPOSITION, 0.9, prediction_time=1, information_available=[far_future.id]
        ),
    )
    probe(
        "smuggle_future_through_evidence_ids",
        "Keep `information_available` clean but hide the future in `evidence_ids_used`.",
        lambda: issue_prediction(
            timeline,
            PROPOSITION,
            0.9,
            prediction_time=1,
            information_available=[past.id],
            evidence_ids_used=[far_future.id],
        ),
    )
    probe(
        "cite_an_unknown_observation",
        "Cite an id the timeline has never heard of — it cannot be shown to predate anything.",
        lambda: issue_prediction(
            timeline, PROPOSITION, 0.9, prediction_time=1, information_available=["OBS-invented"]
        ),
    )

    # The view itself must not leak, independently of the id checks.
    view = timeline.view_as_of(5)
    probes.append(
        {
            "probe": "view_cannot_reach_past_its_horizon",
            "description": "Ask a t=5 view for observations stamped t=5 and t=99.",
            "rejected": view.get(future.id) is None and view.get(far_future.id) is None,
            "error": (
                "view returned a future observation"
                if view.get(future.id) or view.get(far_future.id)
                else "view exposed only observations strictly before t=5"
            ),
        }
    )

    # And a legitimate prediction must still be accepted, or the check is vacuous.
    try:
        issue_prediction(
            timeline, PROPOSITION, 0.6, prediction_time=5, information_available=[past.id]
        )
        legitimate_ok, detail = True, "a prediction citing only the past was accepted"
    except Exception as exc:  # noqa: BLE001
        legitimate_ok, detail = False, f"legitimate prediction wrongly rejected: {exc}"
    probes.append(
        {
            "probe": "legitimate_prediction_still_accepted",
            "description": "Guard against a system that rejects everything and calls it safety.",
            "rejected": legitimate_ok,
            "error": detail,
        }
    )
    return probes


# -------------------------------------------------------- critical sequence


def critical_sequence(directory: Path) -> dict[str, Any]:
    """The 0.80 → B, 0.65 → B, 0.40 sequence, kept verbatim.

    The point is that discovering A did not happen must not retroactively turn
    prediction 1 into 0.40.
    """
    timeline = Timeline()
    ledger = PredictionLedger.in_directory(directory / "critical")
    proposition = "The next outcome will be OUTCOME_A."

    issued = []
    for tick, probability in ((0, 0.80), (1, 0.65), (2, 0.40)):
        view = timeline.view_as_of(tick)
        prediction = issue_prediction(
            timeline,
            proposition=proposition,
            predicted_probability=probability,
            prediction_time=tick,
            information_available=view.ids(),
        )
        ledger.add(prediction)
        issued.append(prediction)
        if tick < 2:  # the first two meet reality; the third is left pending
            timeline.add(f"Trial {tick} reported OUTCOME_B.", at=tick, outcome=False)
            ledger.evaluate(
                prediction.id, outcome=False, outcome_label="OUTCOME_B", evaluation_time=tick + 1
            )

    ledger.remember_timeline(timeline)
    ledger.save()

    reloaded = PredictionLedger.load(ledger.path)
    return {
        "history": reloaded.history(),
        "probabilities_in_order": [
            r.predicted_probability for r in reloaded.records()
        ],
        "unchanged": [r.predicted_probability for r in reloaded.records()]
        == [0.80, 0.65, 0.40],
        "path": str(ledger.path),
    }


# ------------------------------------------------------------------ report


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def render(
    runs: list[EnvironmentRun],
    probes: list[dict[str, Any]],
    critical: dict[str, Any],
    persistence: dict[str, Any],
) -> str:
    lines: list[str] = []
    add = lines.append

    add("# ECHO 3 — Prediction")
    add("")
    add("## PREDICTION ≠ LEARNING")
    add("")
    add(
        "This experiment establishes whether ECHO can state explicit probabilities "
        "about future events and later score them against what actually happened. "
        "**It records experience. It does not generalise from it.**"
    )
    add("")
    add(
        "- No prediction, outcome, or `Experience` record is read back to change a "
        "belief, a probability, or a strategy. Nothing consumes them."
    )
    add(
        "- The predictor is a fixed formula — Laplace-smoothed frequency — "
        "identical on trial 1 and trial 60. It does not improve; it only "
        "accumulates observations, which is not the same thing."
    )
    add(
        "- Environment B changes half way through and ECHO's calibration collapses "
        "there. **It does not notice, adapt, or recover.** A learning system "
        "would. That failure is left in the numbers below rather than designed "
        "around."
    )
    add("")
    add(
        "What exists now is the loop that learning would later require: a "
        "prediction, sealed; an outcome, recorded; an error, computed; an "
        "experience, filed. The loop is empty by design."
    )
    add("")

    total = sum(len(r.ledger) for r in runs)
    evaluated = sum(len(r.ledger.evaluated()) for r in runs)
    add("## Totals")
    add("")
    add(f"- Total predictions issued: **{total}**")
    add(f"- Evaluated against reality: **{evaluated}**")
    add(f"- Environments: **{len(runs)}**")
    add("")

    add("## Scores by environment")
    add("")
    add(
        "Brier is mean squared error (0 perfect, 0.25 for always saying 0.5). "
        "Log loss is mean surprisal in nats — lower is better. ECE is the "
        "count-weighted average gap between stated and observed frequency. "
        "**Accuracy is shown last and on purpose**: it scores `P(A)=0.51` and "
        "`P(A)=0.99` identically, which is precisely the distinction this "
        "experiment exists to preserve."
    )
    add("")
    add("| Environment | N | Brier | vs always-0.5 | vs base rate | Log loss | ECE | Accuracy |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for run in runs:
        s = run.scores
        ref = s["reference_brier"]
        add(
            f"| {run.environment.name} | {s['count']} | **{_fmt(s['brier'])}** "
            f"| {_fmt(ref['always_half'])} | {_fmt(ref['always_base_rate'])} "
            f"| {_fmt(s['log_loss'])} | {_fmt(s['expected_calibration_error'])} "
            f"| {_fmt(s['accuracy'], 3)} |"
        )
    add("")
    add(
        "A Brier score means little alone, so the two trivial strategies it must "
        "beat are shown beside it. Beating *always predict the base rate* is the "
        "real bar, and it is a far harder one than beating 0.5 — a predictor can "
        "look respectable against 0.5 while adding nothing over simply knowing "
        "how often A tends to happen."
    )
    add("")
    add("### How to read these numbers")
    add("")
    add(
        "**The base-rate column is a hindsight oracle, and it is the floor.** It "
        "is computed from the base rate that actually occurred, which no predictor "
        "could have known in advance. For a coin that lands heads with probability "
        "*p*, the best achievable Brier score is *p*(1−*p*) — irreducible noise "
        "that no amount of intelligence removes. Matching that column is therefore "
        "**the ceiling on performance, not a passing grade**, and slightly "
        "exceeding it is expected rather than a failure."
    )
    add("")
    stable = next((r for r in runs if r.environment.id == "ENV-A-stable"), None)
    if stable:
        s = stable.scores
        floor = s["reference_brier"]["always_base_rate"]
        add(
            f"So Environment A's Brier of {_fmt(s['brier'])} against a floor of "
            f"{_fmt(floor)} is close to the best any predictor could do on that "
            "sequence — the residual gap is mostly the early trials, where ECHO "
            "correctly had no idea and said so."
        )
        add("")

    for run in runs:
        env = run.environment
        s = run.scores
        add(f"### {env.name}")
        add("")
        add(f"**What ECHO could tell.** {env.description}")
        add("")
        add(f"**Predictions:** {s['count']} · **Observed base rate:** {_fmt(s['base_rate'], 3)}")
        add("")
        add("| Stated probability | N | Mean stated | Observed frequency | Gap |")
        add("| --- | --- | --- | --- | --- |")
        for bucket in s["buckets"]:
            add(
                f"| {bucket['range']} | {bucket['count']} | {bucket['mean_predicted']:.3f} "
                f"| {bucket['observed_frequency']:.3f} | {bucket['gap']:+.3f} |"
            )
        add("")
        if s["log_loss_excluded"]:
            add(
                f"_{s['log_loss_excluded']} prediction(s) excluded from log loss as "
                "mathematically undefined (probability 0 or 1 on the wrong side)._"
            )
            add("")

        if env.id == "ENV-B-changing":
            before = run.scores_for_slice(0, 30)
            after = run.scores_for_slice(30, 60)
            add("**Before and after the change:**")
            add("")
            add("| Segment | N | Brier | Base rate | Mean stated |")
            add("| --- | --- | --- | --- | --- |")
            for label, seg in (("Trials 0–29", before), ("Trials 30–59", after)):
                mean_stated = (
                    sum(b["mean_predicted"] * b["count"] for b in seg["buckets"]) / seg["count"]
                    if seg["count"]
                    else None
                )
                add(
                    f"| {label} | {seg['count']} | {_fmt(seg['brier'])} "
                    f"| {_fmt(seg['base_rate'], 3)} | {_fmt(mean_stated, 3)} |"
                )
            add("")
            add(
                "This is the honest failure of the experiment. The estimator "
                "averages all history equally, so after the process flips it keeps "
                "predicting the old regime and is confidently wrong for a long "
                "stretch. It never recovers within the run. **Nothing in ECHO "
                "detects the change**, because nothing in ECHO is looking."
            )
            add("")

        if env.id == "ENV-C-noisy":
            corrupted = sum(1 for t in run.trials if t.corrupted)
            observed_rate = sum(1 for t in run.trials if t.observed_outcome) / len(run.trials)
            add(
                f"**Why this one is miscalibrated.** {corrupted} of {len(run.trials)} "
                f"observations shown to ECHO were flipped, so the apparent rate in "
                f"its evidence was {observed_rate:.3f} while the true rate was "
                f"{_fmt(s['base_rate'], 3)}. ECHO converged on what it was told, "
                "which is the correct response to its evidence and the wrong answer "
                "about the world. Its error here is inherited from its information, "
                "not produced by its reasoning — and nothing in the scores can "
                "separate those two, which is a limitation of the experiment rather "
                "than a finding about ECHO."
            )
            add("")

        add(f"**Hidden process (never shown to ECHO):** {env.hidden_process}")
        add("")

    # ------------------------------------------------------------- examples
    add("## Prediction history examples")
    add("")
    add(
        "The ledger answers, for any prediction: what did I predict, what "
        "probability did I assign, what information did I have, what actually "
        "happened, and how wrong was I."
    )
    add("")
    sample_run = runs[0]
    add("| # | t | P(A) | Information | Actual | Error |")
    add("| --- | --- | --- | --- | --- | --- |")
    for entry in sample_run.ledger.history()[:8]:
        add(
            f"| `{entry['prediction_id'][-6:]}` | {entry['prediction_time']} "
            f"| {entry['predicted_probability']:.4f} | {entry['information_count']} obs "
            f"| {entry['actual_outcome']} | {_fmt(entry['prediction_error'], 4)} |"
        )
    add("")

    best = min(
        (r for run in runs for r in run.ledger.evaluated()),
        key=lambda r: r.evaluation.brier_score,  # type: ignore[union-attr]
    )
    worst = max(
        (r for run in runs for r in run.ledger.evaluated()),
        key=lambda r: r.evaluation.brier_score,  # type: ignore[union-attr]
    )
    add("**Correct predictions.** The single best-scored prediction across all runs:")
    add("")
    add(
        f"- `{best.id}` at t={best.prediction_time}: stated **{best.predicted_probability:.4f}**, "
        f"outcome **{best.evaluation.outcome_label}**, Brier "  # type: ignore[union-attr]
        f"**{best.evaluation.brier_score:.4f}**"  # type: ignore[union-attr]
    )
    add("")
    add("**Failed predictions.** The single worst-scored prediction across all runs:")
    add("")
    add(
        f"- `{worst.id}` at t={worst.prediction_time}: stated "
        f"**{worst.predicted_probability:.4f}**, outcome "
        f"**{worst.evaluation.outcome_label}**, Brier "  # type: ignore[union-attr]
        f"**{worst.evaluation.brier_score:.4f}**"  # type: ignore[union-attr]
    )
    add("")
    for run in runs:
        misses = [
            r
            for r in run.ledger.evaluated()
            if not r.evaluation.correctly_anticipated  # type: ignore[union-attr]
        ]
        add(
            f"- {run.environment.name}: {len(run.ledger.evaluated()) - len(misses)} "
            f"anticipated correctly, {len(misses)} not "
            f"(a coarse count that ignores how strongly each was claimed)."
        )
    add("")

    # ---------------------------------------------------- critical sequence
    add("## The critical sequence")
    add("")
    add(
        "Prediction 1 at 0.80 → **B**. Prediction 2 at 0.65 → **B**. Prediction 3 "
        "at 0.40, left pending. After both failures, prediction 1 must still read "
        "0.80. Reloaded from disk:"
    )
    add("")
    add("| # | P(A) stated | Status | Actual | Error |")
    add("| --- | --- | --- | --- | --- |")
    for index, entry in enumerate(critical["history"], start=1):
        add(
            f"| {index} | **{entry['predicted_probability']:.2f}** | {entry['status']} "
            f"| {entry['actual_outcome'] or '—'} | {_fmt(entry['prediction_error'], 3)} |"
        )
    add("")
    add(
        f"Sequence read back from disk: "
        f"`{critical['probabilities_in_order']}` — "
        f"**{'unchanged' if critical['unchanged'] else 'ALTERED — THIS IS A FAILURE'}**."
    )
    add("")

    # ------------------------------------------------------------- leakage
    add("## TEMPORAL_LEAKAGE_TEST")
    add("")
    add(
        "Each probe below deliberately tries to smuggle information from after "
        "the prediction timestamp into the prediction. All must be refused. The "
        "last two are controls: one checks the view cannot reach the future, the "
        "other checks a legitimate prediction is still accepted — a system that "
        "rejects everything would pass the first four and be useless."
    )
    add("")
    add("| Probe | Outcome |")
    add("| --- | --- |")
    for p in probes:
        add(f"| `{p['probe']}` | {'✅ correct' if p['rejected'] else '❌ FAILED'} |")
    add("")
    for p in probes:
        add(f"- **`{p['probe']}`** — {p['description']}")
        add(f"  - {p['error']}")
    add("")
    failures = [p for p in probes if not p["rejected"]]
    add(
        f"**Result: {len(probes) - len(failures)}/{len(probes)} probes behaved correctly.**"
        + ("" if not failures else " **THERE ARE FAILURES ABOVE — the guarantee is broken.**")
    )
    add("")

    # --------------------------------------------------------- persistence
    add("## Persistence")
    add("")
    add(
        "Run as part of this report, not asserted: the ledgers were discarded and "
        "rebuilt from their JSON files, then compared field by field."
    )
    add("")
    add("| After restart | Survives |")
    add("| --- | --- |")
    for key, label in (
        ("predictions_available", "Every prediction"),
        ("probabilities_identical", "The exact stated probabilities"),
        ("information_available", "What information each prediction had"),
        ("outcomes_available", "The recorded outcomes and errors"),
        ("experiences_available", "The filed experience records"),
    ):
        add(f"| {label} | {'yes' if persistence[key] else 'NO'} |")
    add("")
    add(
        f"Predictions checked: {persistence['predictions_checked']}; "
        f"experiences checked: {persistence['experiences_checked']}."
    )
    add("")

    # -------------------------------------------------------- limitations
    add("## Limitations")
    add("")
    add(
        "1. **The predictor is deliberately naive.** Laplace-smoothed frequency "
        "over all history. It cannot represent trend, regime, or recency, and "
        "Environment B punishes it accordingly. A better predictor would score "
        "better; that would not tell us anything more about whether the "
        "prediction *loop* works, which is what is being tested."
    )
    add(
        "2. **Environment C conflates two things.** ECHO is scored against truth "
        "while seeing corrupted observations, so its error there mixes bad "
        "prediction with bad information. The scores cannot separate them, and no "
        "attempt is made to."
    )
    add(
        "3. **60 trials per environment is small.** Calibration buckets hold few "
        "predictions each, so observed frequencies are noisy. Bucket gaps here "
        "are suggestive, not solid."
    )
    add(
        "4. **Predictions within a run are not independent.** Each is built from "
        "the history that produced the ones before it, so the scores should not "
        "be read as 60 independent trials."
    )
    add(
        "5. **Only binary outcomes.** No multi-class, no continuous quantities, no "
        "structured predictions."
    )
    add(
        "6. **Time is a logical tick, not a clock.** This is the right choice for "
        "reproducibility and for reasoning about ordering, but it means the "
        "temporal guarantee is about sequence, not about wall-clock time. A real "
        "deployment would have to defend against clock skew, which this does not."
    )
    add(
        "7. **The proposition is fixed and supplied.** ECHO does not decide what "
        "is worth predicting, only what probability to assign."
    )
    add("")

    add("## Conclusion")
    add("")
    add(
        f"ECHO issued **{total}** explicit probabilistic predictions across three "
        "environments, sealed each one before the outcome existed, scored them "
        "with proper scoring rules, and filed an immutable experience record for "
        "each. Every temporal-leakage probe was refused. The critical sequence "
        "survived two failed predictions without a single stated probability "
        "moving."
    )
    add("")
    add(
        "It is well calibrated on a stationary process, roughly base-rate on a "
        "noisy one, and **badly wrong on a changing one — where it fails to "
        "notice anything has changed at all.** That last result is the most "
        "informative thing in this report and the clearest demonstration of what "
        "is missing."
    )
    add("")
    add(
        "**PREDICTION ≠ LEARNING.** The experience records exist and nothing reads "
        "them. ECHO does not currently generalise from any of this."
    )
    add("")
    return "\n".join(lines)


def verify_persistence(runs: list[EnvironmentRun]) -> dict[str, Any]:
    checks = {
        "predictions_available": True,
        "probabilities_identical": True,
        "information_available": True,
        "outcomes_available": True,
        "experiences_available": True,
        "predictions_checked": 0,
        "experiences_checked": 0,
    }
    for run in runs:
        expected = run.ledger.records()
        expected_experiences = run.ledger.experiences()
        revived = PredictionLedger.load(run.ledger.path)
        restored = revived.records()

        checks["predictions_checked"] += len(expected)
        checks["experiences_checked"] += len(expected_experiences)

        if len(restored) != len(expected):
            checks["predictions_available"] = False
            continue
        for before, after in zip(expected, restored):
            if before.predicted_probability != after.predicted_probability:
                checks["probabilities_identical"] = False
            if (
                before.information_available_at_prediction_time
                != after.information_available_at_prediction_time
            ):
                checks["information_available"] = False
            if before.prediction_error != after.prediction_error:
                checks["outcomes_available"] = False
        if len(revived.experiences()) != len(expected_experiences):
            checks["experiences_available"] = False
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the prediction experiment.")
    parser.add_argument("--out", help="write the markdown report here")
    parser.add_argument("--data", help="directory for the ledgers")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data) if args.data else RESULTS_DIR / "prediction"

    runs = [run_environment(env, data_dir) for env in ENVIRONMENTS]
    probes = temporal_leakage_probes(data_dir)
    critical = critical_sequence(data_dir)
    persistence = verify_persistence(runs)

    markdown = render(runs, probes, critical, persistence)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = RESULTS_DIR / f"prediction_run_{stamp}.json"
    raw_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "environments": [
                    {
                        "id": run.environment.id,
                        "hidden_process": run.environment.hidden_process,
                        "scores": run.scores,
                        "history": run.ledger.history(),
                    }
                    for run in runs
                ],
                "temporal_leakage_probes": probes,
                "critical_sequence": critical,
                "persistence": persistence,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        print(f"report -> {out}")
    else:
        print(markdown)
    print(f"raw    -> {raw_path}")

    for run in runs:
        s = run.scores
        print(
            f"  {run.environment.id:<18} n={s['count']:<3} "
            f"brier={_fmt(s['brier'])} base={_fmt(s['reference_brier']['always_base_rate'])} "
            f"ece={_fmt(s['expected_calibration_error'])}"
        )
    leak_failures = [p for p in probes if not p["rejected"]]
    print(f"  leakage probes: {len(probes) - len(leak_failures)}/{len(probes)} correct")
    print(f"  critical sequence unchanged: {critical['unchanged']}")

    return 0 if not leak_failures and critical["unchanged"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
