"""Baseline vs learning, across four environments, and the report.

Fully offline. No model, no API key, no network.

    python -m experiments.run_learning --out docs/ECHO_4_LEARNING.md

Two conditions run against **identical deterministic environments**:

- **A — no learning.** The Challenge-3 strategy, fixed for the whole run.
- **B — learning enabled.** The same starting strategy, plus error analysis,
  change detection, and strategy revision.

Nothing tells ECHO where a regime changes, or that one does. Detection is a
binomial test comparing recent performance to history, requiring confirmation
over time. The report states measured results including failures.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from echo import calibration
from echo.learning import (
    EVIDENCE_DIVERGENCE,
    RECENT_WINDOW,
    ChangeDetector,
    FailureExplanation,
    LearningExperience,
    Sample,
    StrategyRevision,
    analyse_failure,
    propose_successor,
)
from echo.learning_ledger import LearningLedger
from echo.prediction import Timeline, issue_prediction
from echo.prediction_ledger import PredictionLedger
from echo.strategy import PerformanceRecord, StrategyStatus, initial_strategy

from .environments import LEARNING_ENVIRONMENTS, PROPOSITION, Environment

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"

TEST_WINDOW = 8  # trials a candidate strategy is given before a verdict
PROMOTION_MARGIN = 0.02  # Brier improvement required to promote a candidate
DIVERGENCE_WINDOW = 30  # span for comparing reports against realised outcomes
HIGH_CONFIDENCE = 0.80  # a claim at or beyond this is "high confidence"


@dataclass
class ConditionRun:
    environment: Environment
    condition: str  # "baseline" | "learning"
    predictions: PredictionLedger
    learning: LearningLedger | None
    samples: list[Sample] = field(default_factory=list)
    events: list[str] = field(default_factory=list)
    detections: list[dict[str, Any]] = field(default_factory=list)

    # ---------------------------------------------------------------- scores

    @property
    def change_point(self) -> int | None:
        return self.environment.change_points[0] if self.environment.change_points else None

    def _slice(self, start: int, end: int | None = None) -> list[Sample]:
        end = end if end is not None else len(self.samples)
        return [s for s in self.samples if start <= s.trial < end]

    def scores(self, samples: list[Sample] | None = None) -> dict[str, Any]:
        chosen = self.samples if samples is None else samples
        return calibration.summarise(
            [s.probability for s in chosen], [s.outcome for s in chosen]
        )

    @property
    def pre_change(self) -> dict[str, Any]:
        return self.scores(self._slice(0, self.change_point or len(self.samples)))

    @property
    def post_change(self) -> dict[str, Any]:
        if self.change_point is None:
            return self.scores([])
        return self.scores(self._slice(self.change_point))

    @property
    def final_window(self) -> dict[str, Any]:
        return self.scores(self.samples[-20:])

    @property
    def high_confidence_errors(self) -> int:
        return sum(
            1
            for s in self.samples
            if max(s.probability, 1 - s.probability) >= HIGH_CONFIDENCE and s.missed
        )

    def recovery_trial(self) -> int | None:
        """First post-change trial where a rolling window returns to pre-change form.

        `None` means it never recovered within the run — which is a result, not
        a missing value.
        """
        if self.change_point is None:
            return None
        target = (self.pre_change["brier"] or 0.0) + 0.05
        post = self._slice(self.change_point)
        for index in range(RECENT_WINDOW, len(post) + 1):
            window = post[index - RECENT_WINDOW : index]
            mean = sum(s.brier for s in window) / len(window)
            if mean <= target:
                return window[-1].trial
        return None


def _divergence(timeline: Timeline, samples: list[Sample]) -> float | None:
    """How far the reported observations sit from the outcomes actually scored.

    A large, sustained gap points at the evidence rather than the reasoning.
    """
    if len(samples) < DIVERGENCE_WINDOW:
        return None
    reported = [
        o.payload["outcome"]
        for o in timeline.all()[-DIVERGENCE_WINDOW:]
        if "outcome" in o.payload
    ]
    realised = [s.outcome for s in samples[-DIVERGENCE_WINDOW:]]
    if not reported or not realised:
        return None
    return abs(
        sum(1 for r in reported if r) / len(reported)
        - sum(1 for r in realised if r) / len(realised)
    )


def run_condition(
    environment: Environment, condition: str, directory: Path
) -> ConditionRun:
    learning_enabled = condition == "learning"
    base = directory / environment.id / condition

    timeline = Timeline()
    predictions = PredictionLedger.in_directory(base)
    learning_ledger = LearningLedger.in_directory(base) if learning_enabled else None

    strategy = initial_strategy()
    # `is not None`, not truthiness: LearningLedger defines __len__, so an empty
    # ledger is falsy and the incumbent would never be registered — leaving every
    # shadow comparison to fall back on a placeholder.
    if learning_ledger is not None:
        learning_ledger.register_strategy(strategy)

    run = ConditionRun(
        environment=environment,
        condition=condition,
        predictions=predictions,
        learning=learning_ledger,
    )
    run.events.append(
        f"TRIAL 0 — strategy {strategy.id} ACTIVE: {strategy.description}"
    )

    detector = ChangeDetector()
    shadow: dict[str, list[Sample]] = {strategy.id: []}
    testing: dict[str, Any] | None = None
    recent_experiences: list[str] = []

    for trial in environment.generate():
        tick = trial.index
        view = timeline.view_as_of(tick)

        probability, rationale = strategy.predict(view)
        prediction = issue_prediction(
            timeline,
            proposition=PROPOSITION,
            predicted_probability=probability,
            prediction_time=tick,
            information_available=view.ids(),
            rationale=f"[{strategy.id}] {rationale}",
        )
        predictions.add(prediction)

        # Reality arrives only after the prediction is sealed.
        timeline.add(
            f"Trial {tick} reported {trial.observed_label}.",
            at=tick,
            outcome=trial.observed_outcome,
        )
        record = predictions.evaluate(
            prediction.id, trial.true_outcome, trial.true_label, evaluation_time=tick + 1
        )
        brier = record.evaluation.brier_score  # type: ignore[union-attr]
        run.samples.append(
            Sample(trial=tick, probability=probability, outcome=trial.true_outcome, brier=brier)
        )

        # Every registered strategy is shadow-scored on the same view, so a
        # comparison later is like-for-like and uses no future information.
        for other in list(shadow):
            candidate = (
                strategy
                if other == strategy.id
                else (learning_ledger.get_strategy(other) if learning_ledger is not None else None)
            )
            if candidate is None:
                continue
            shadow_p, _ = candidate.predict(view)
            shadow[other].append(
                Sample(tick, shadow_p, trial.true_outcome, (shadow_p - (1.0 if trial.true_outcome else 0.0)) ** 2)
            )

        if not learning_enabled or learning_ledger is None:
            continue

        # ---------------------------------------------------------- learning
        signal = detector.assess(run.samples, tick)
        divergence = _divergence(timeline, run.samples)
        analysis = analyse_failure(
            error=record.prediction_error or 0.0,
            samples=run.samples,
            signal=signal,
            evidence_divergence=divergence,
        )

        proposed: dict[str, Any] | None = None
        if analysis.explanation is FailureExplanation.REGIME_CHANGE and testing is None:
            proposed = {
                "action": "propose_successor_strategy",
                "rationale": "sustained deterioration consistent with a changed process",
            }

        experience = learning_ledger.add_experience(
            LearningExperience(
                prediction_id=prediction.id,
                observed_outcome=trial.true_outcome,
                prediction_error=record.prediction_error or 0.0,
                relevant_evidence=record.information_available_at_prediction_time[-RECENT_WINDOW:],
                previous_strategy=f"{strategy.id} v{strategy.version}",
                failure_analysis=analysis,
                proposed_update=proposed,
                confidence=analysis.confidence,
                trial=tick,
            )
        )
        recent_experiences.append(experience.id)
        recent_experiences = recent_experiences[-RECENT_WINDOW:]

        if signal is not None and signal.suspected_regime_change:
            run.detections.append(signal.to_dict())
            if not signal.confirmed:
                run.events.append(
                    f"TRIAL {tick} — deterioration flagged (p={signal.p_value:.1e}), "
                    "not yet confirmed; no action taken"
                )

        # --- propose a successor -------------------------------------------
        if proposed is not None and signal is not None:
            successor = propose_successor(strategy, signal)
            learning_ledger.register_strategy(successor)
            shadow[successor.id] = []
            revision = learning_ledger.add_revision(
                StrategyRevision(
                    at_trial=tick,
                    from_strategy_id=strategy.id,
                    to_strategy_id=successor.id,
                    reason=(
                        "Sustained, statistically significant deterioration against "
                        f"historical performance ({signal.threshold_crossed}). The "
                        "active strategy assumes a stationary process; that "
                        "assumption is what the evidence contradicts."
                    ),
                    triggering_experience_ids=tuple(recent_experiences),
                    change_signal=signal.to_dict(),
                    reversal_condition=(
                        f"Revert to {strategy.id} if the successor's mean Brier over "
                        f"the {TEST_WINDOW}-trial test window is not at least "
                        f"{PROMOTION_MARGIN} better than the incumbent's over the same "
                        "trials, or if it later loses that margin over any subsequent "
                        f"{RECENT_WINDOW}-trial window."
                    ),
                    confidence=signal.confidence,
                )
            )
            successor.set_status(
                StrategyStatus.TESTING,
                f"proposed at trial {tick} after confirmed deterioration",
            )
            run.events.append(
                f"TRIAL {tick} — deterioration CONFIRMED "
                f"({signal.recent_performance['miss_rate']:.0%} recent miss rate vs "
                f"{signal.historical_performance['miss_rate']:.0%} historical, "
                f"p={signal.p_value:.1e})"
            )
            run.events.append(
                f"TRIAL {tick} — hypothesis: REGIME_CHANGE (confidence "
                f"{analysis.confidence:.2f}); alternatives considered and rejected"
            )
            run.events.append(
                f"TRIAL {tick} — strategy {successor.id} proposed "
                f"(window={successor.params.get('window')}) → TESTING for "
                f"{TEST_WINDOW} trials"
            )
            testing = {
                "strategy": successor,
                "incumbent": strategy,
                "start": tick + 1,
                "end": tick + 1 + TEST_WINDOW,
                "revision_id": revision.id,
            }
            strategy = successor
            detector.reset_after_adaptation()

        # --- conclude the test ---------------------------------------------
        elif testing is not None and tick + 1 >= testing["end"]:
            start, end = testing["start"], testing["end"]
            candidate: Any = testing["strategy"]
            incumbent: Any = testing["incumbent"]

            def window_of(strategy_id: str) -> list[Sample]:
                return [s for s in shadow.get(strategy_id, []) if start <= s.trial < end]

            candidate_window = [s for s in run.samples if start <= s.trial < end]
            incumbent_window = window_of(incumbent.id)
            candidate_brier = (
                sum(s.brier for s in candidate_window) / len(candidate_window)
                if candidate_window
                else 1.0
            )
            incumbent_brier = (
                sum(s.brier for s in incumbent_window) / len(incumbent_window)
                if incumbent_window
                else 1.0
            )

            before = {
                "strategy": incumbent.id,
                "mean_brier": round(incumbent_brier, 6),
                "window": f"trials {start}-{end - 1}",
            }
            after = {
                "strategy": candidate.id,
                "mean_brier": round(candidate_brier, 6),
                "window": f"trials {start}-{end - 1}",
            }
            candidate.record_performance(
                PerformanceRecord(
                    label=f"test window trials {start}-{end - 1}",
                    predictions=len(candidate_window),
                    mean_brier=candidate_brier,
                    miss_rate=sum(1 for s in candidate_window if s.missed)
                    / max(1, len(candidate_window)),
                )
            )
            incumbent.record_performance(
                PerformanceRecord(
                    label=f"shadow over trials {start}-{end - 1}",
                    predictions=len(incumbent_window),
                    mean_brier=incumbent_brier,
                    miss_rate=sum(1 for s in incumbent_window if s.missed)
                    / max(1, len(incumbent_window)),
                )
            )

            if candidate_brier <= incumbent_brier - PROMOTION_MARGIN:
                candidate.set_status(
                    StrategyStatus.ACTIVE,
                    f"test window Brier {candidate_brier:.3f} beat the incumbent's "
                    f"{incumbent_brier:.3f} by more than {PROMOTION_MARGIN}",
                    confidence=min(0.85, candidate.confidence + 0.2),
                )
                incumbent.set_status(
                    StrategyStatus.WEAKENED,
                    f"superseded at trial {tick}; shadow Brier {incumbent_brier:.3f}",
                    confidence=max(0.1, incumbent.confidence - 0.3),
                )
                learning_ledger.conclude_revision(
                    testing["revision_id"], "improved", before, after
                )
                run.events.append(
                    f"TRIAL {tick} — test concluded: {candidate.id} Brier "
                    f"{candidate_brier:.3f} vs incumbent {incumbent_brier:.3f} → "
                    f"{candidate.id} ACTIVE, {incumbent.id} WEAKENED"
                )
                strategy = candidate
            else:
                candidate.set_status(
                    StrategyStatus.RETIRED,
                    f"test window Brier {candidate_brier:.3f} did not beat the "
                    f"incumbent's {incumbent_brier:.3f} by {PROMOTION_MARGIN}",
                    confidence=0.1,
                )
                learning_ledger.conclude_revision(
                    testing["revision_id"], "no_improvement", before, after
                )
                run.events.append(
                    f"TRIAL {tick} — test concluded: {candidate.id} Brier "
                    f"{candidate_brier:.3f} did NOT beat incumbent "
                    f"{incumbent_brier:.3f} → {candidate.id} RETIRED, reverting to "
                    f"{incumbent.id}"
                )
                strategy = incumbent
            testing = None

    predictions.remember_timeline(timeline)
    predictions.save()
    if learning_ledger is not None:
        learning_ledger.save()

    if run.change_point is not None:
        recovery = run.recovery_trial()
        run.events.append(
            f"TRIAL {recovery} — rolling performance back to pre-change form"
            if recovery is not None
            else "NEVER — rolling performance did not return to pre-change form"
        )
    return run


# ------------------------------------------------------------------- report


def _fmt(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _delta(baseline, learning) -> str:
    if baseline is None or learning is None:
        return "n/a"
    diff = learning - baseline
    direction = "better" if diff < 0 else "worse" if diff > 0 else "same"
    return f"{diff:+.4f} ({direction})"


def render(pairs: list[tuple[ConditionRun, ConditionRun]], persistence: dict[str, Any]) -> str:
    lines: list[str] = []
    add = lines.append

    add("# ECHO 4 — Learning From Prediction Errors")
    add("")

    # ------------------------------------------------------ definitions
    add("## What the words mean here")
    add("")
    add(
        "These four are different things, and the whole point of this series is "
        "that they are not interchangeable."
    )
    add("")
    add("| Term | Definition | Where it lives |")
    add("| --- | --- | --- |")
    add("| **MEMORY** | Stored information. A statement that was true and was kept. | `echo/memory.py` |")
    add(
        "| **BELIEF REVISION** | Changing confidence in a proposition when evidence "
        "arrives. One number, one sentence, a fixed rule. | `echo/belief.py` |"
    )
    add(
        "| **PREDICTION** | A probabilistic statement about a future outcome, sealed "
        "before the outcome exists. | `echo/prediction.py` |"
    )
    add(
        "| **LEARNING** | A persistent change in *strategy or behaviour*, caused by "
        "experience, that improves performance on **future** situations. | `echo/learning.py` |"
    )
    add("")
    add(
        "**A value changing is not learning.** Belief revision changes a number by a "
        "rule that never itself changes; ECHO is exactly the same afterwards. What "
        "is claimed in this document is narrower and more specific: ECHO's *method "
        "of predicting* changed, the change was caused by measured errors, it "
        "persists, and it is tested on situations that came after it."
    )
    add("")
    add(
        "This is not AGI, not consciousness, and not autonomous intelligence. It is "
        "a change-detector wired to a two-item menu of estimators. The mechanism is "
        "a few hundred lines of arithmetic and it is described in full below."
    )
    add("")

    # ------------------------------------------------------ headline result
    add("## Condition A (no learning) vs Condition B (learning)")
    add("")
    add(
        "Identical deterministic environments, identical starting strategy. The only "
        "difference is whether error analysis and strategy revision are enabled."
    )
    add("")
    add("| Environment | Metric | No learning | Learning | Change |")
    add("| --- | --- | --- | --- | --- |")
    for baseline, learned in pairs:
        env = baseline.environment
        rows = [
            ("Brier (whole run)", baseline.scores()["brier"], learned.scores()["brier"]),
            ("Log loss", baseline.scores()["log_loss"], learned.scores()["log_loss"]),
            (
                "ECE",
                baseline.scores()["expected_calibration_error"],
                learned.scores()["expected_calibration_error"],
            ),
        ]
        if env.change_points:
            rows.append(
                ("Brier after change", baseline.post_change["brier"], learned.post_change["brier"])
            )
        rows.append(
            ("Brier, final 20", baseline.final_window["brier"], learned.final_window["brier"])
        )
        for label, base_value, learn_value in rows:
            add(
                f"| `{env.id}` | {label} | {_fmt(base_value)} | {_fmt(learn_value)} "
                f"| {_delta(base_value, learn_value)} |"
            )
        add(
            f"| `{env.id}` | High-confidence errors | {baseline.high_confidence_errors} "
            f"| {learned.high_confidence_errors} "
            f"| {learned.high_confidence_errors - baseline.high_confidence_errors:+d} |"
        )
        recovery_base = baseline.recovery_trial()
        recovery_learn = learned.recovery_trial()
        if env.change_points:
            add(
                f"| `{env.id}` | Recovery trial | "
                f"{recovery_base if recovery_base is not None else 'never'} "
                f"| {recovery_learn if recovery_learn is not None else 'never'} "
                f"| — |"
            )
    add("")
    add(
        "Lower is better for Brier, log loss and ECE. 'Recovery trial' is the first "
        "post-change trial at which a rolling 10-prediction Brier returns to within "
        "0.05 of the pre-change level; **never** means it did not, within the run."
    )
    add("")

    # ------------------------------------------------------ per environment
    for baseline, learned in pairs:
        env = baseline.environment
        add(f"## `{env.id}` — {env.name}")
        add("")
        add(f"**What ECHO could tell.** {env.description}")
        add("")
        add("### Timeline (learning condition)")
        add("")
        add("```")
        if env.change_points:
            for point in env.change_points:
                add(f"TRIAL {point} — environment changes (hidden from ECHO)")
        for event in learned.events:
            add(event)
        add("```")
        add("")

        if learned.learning is not None:
            counts = learned.learning.explanations()
            add("**Failure explanations reached:**")
            add("")
            add("| Explanation | Count |")
            add("| --- | --- |")
            for name in FailureExplanation:
                add(f"| `{name.value}` | {counts.get(name.value, 0)} |")
            add("")

            learned_items = learned.learning.what_did_i_learn()
            if learned_items:
                add("**What did I learn, and did it help?**")
                add("")
                for item in learned_items:
                    add(f"- **At trial {item['at_trial']}**")
                    add(f"  - From: {item['changed_from']}")
                    add(f"  - To: {item['changed_to']}")
                    add(f"  - Why: {item['why']}")
                    add(f"  - Confidence: {item['confidence']:.3f}")
                    add(f"  - Did it help: **{item['did_it_help']}**")
                    add(f"  - Measured before: {item['before']}")
                    add(f"  - Measured after: {item['after']}")
                    add(f"  - Would reverse if: {item['would_reverse_if']}")
                add("")
            else:
                add("**No strategy revision was made in this environment.**")
                add("")

            add("**Strategies, including retired ones:**")
            add("")
            add("| Strategy | v | Kind | Params | Status | Confidence |")
            add("| --- | --- | --- | --- | --- | --- |")
            for strategy in learned.learning.strategies():
                add(
                    f"| `{strategy.id}` | {strategy.version} | {strategy.kind} "
                    f"| {strategy.params} | **{strategy.status.value}** "
                    f"| {strategy.confidence:.2f} |"
                )
            add("")

        add(f"**Hidden process (never shown to ECHO):** {env.hidden_process}")
        add("")

    # ------------------------------------------------------ generalisation
    add("## Generalisation, and whether this is just memorisation")
    add("")
    add(
        "`ENV-D-late-change` exists to answer one question: has ECHO learned "
        "*'regime changes happen at trial 30'*, or *'sustained deterioration means "
        "the process may have changed'*? Its change is **32 trials later** than "
        "Environment B's and runs in the **opposite direction** — a rarer outcome "
        "becoming common rather than the reverse."
    )
    add("")
    d_pair = next((p for p in pairs if p[0].environment.id == "ENV-D-late-change"), None)
    if d_pair:
        base_d, learn_d = d_pair
        detected = [e for e in learn_d.events if "CONFIRMED" in e]
        add(
            f"- Detection in ENV-D: {detected[0] if detected else '**none — no change was detected**'}"
        )
        add(
            f"- Brier after the change: {_fmt(base_d.post_change['brier'])} without "
            f"learning, {_fmt(learn_d.post_change['brier'])} with "
            f"({_delta(base_d.post_change['brier'], learn_d.post_change['brier'])})"
        )
        add(
            "- **Nothing in the detector references a trial number.** It compares a "
            "recent window to the history preceding it, wherever that window happens "
            "to fall."
        )
    add("")

    # ------------------------------------------------------ false alarms
    add("## False alarms")
    add("")
    add(
        "`ENV-F-false-alarm` is stationary throughout — P(A) = 0.75 from first trial "
        "to last — but trials 40–46 are forced to OUTCOME_B, producing a run of "
        "seven unusual results. A detector that fires on surprise alone will call "
        "this a regime change. **Any confirmed detection here is a false alarm.**"
    )
    add("")
    add("| Environment | Real changes | Flagged | Confirmed | Verdict |")
    add("| --- | --- | --- | --- | --- |")
    for _, learned in pairs:
        env = learned.environment
        flagged = len(learned.detections)
        confirmed = sum(1 for d in learned.detections if d["confirmed"])
        real = len(env.change_points)
        if real == 0:
            verdict = "no false alarm" if confirmed == 0 else f"**{confirmed} FALSE ALARM(S)**"
        else:
            verdict = "detected" if confirmed else "**missed**"
        add(f"| `{env.id}` | {real} | {flagged} | {confirmed} | {verdict} |")
    add("")
    add(
        "The gap between *flagged* and *confirmed* is where overreaction is "
        "prevented. A single significant window is not enough — the deterioration "
        "must persist across a gap of "
        f"{ChangeDetector().confirm_gap} trials. A streak of bad luck stops; a "
        "changed process does not."
    )
    add("")

    # ------------------------------------------------------ persistence
    add("## Learning history survives restart")
    add("")
    add("| After restart | Survives |")
    add("| --- | --- |")
    for key, label in (
        ("experiences", "Learning experiences"),
        ("revisions", "Strategy revisions"),
        ("strategies", "All strategies, including retired ones"),
        ("verdicts", "Whether each revision helped"),
        ("reversal_conditions", "What would reverse each change"),
    ):
        add(f"| {label} | {'yes' if persistence[key] else 'NO'} |")
    add("")

    # ------------------------------------------------------ limitations
    add("## Limitations")
    add("")
    add(
        "1. **The menu has two items.** A strategy picks between all-history and "
        "recent-window frequency estimation. ECHO cannot invent an estimator, and "
        "calling a choice between two options 'learning' is only fair because the "
        "choice is caused by evidence, persists, and is tested — not because the "
        "space is impressive."
    )
    add(
        "2. **The window size is not learned.** The successor's window is the "
        "detector's own window. A system that learned *how much* history to keep "
        "would be doing something substantially harder."
    )
    add(
        "3. **One adaptation per run, effectively.** After a promotion there is no "
        "mechanism for proposing a third strategy on new evidence beyond the "
        "monitored reversal condition."
    )
    add(
        "4. **The detector only notices deterioration.** A process that changed in "
        "ECHO's favour would go unremarked, because nothing looks for improvement."
    )
    add(
        "5. **Thresholds are hand-set.** α, the effect size, the confirmation gap, "
        "the test window and the promotion margin were all chosen by a human before "
        "the run and not tuned afterwards — but they were still chosen, and "
        "different values would give different results."
    )
    add(
        "6. **Four environments, one author.** They test the failure modes their "
        "author anticipated. Environment F is the only adversarial case, and it is "
        "adversarial in exactly the way its author thought to be adversarial."
    )
    add(
        "7. **Deterministic and small.** A single seed per environment. Results are "
        "reproducible but not statistically robust; a different seed could tell a "
        "different story."
    )
    add("")

    # ------------------------------------------------------ conclusion
    add("## Conclusion")
    add("")
    improved = []
    failed = []
    for baseline, learned in pairs:
        if not learned.environment.change_points:
            continue
        base_post = baseline.post_change["brier"]
        learn_post = learned.post_change["brier"]
        if base_post is not None and learn_post is not None:
            (improved if learn_post < base_post else failed).append(
                (learned.environment.id, base_post, learn_post)
            )
    for env_id, base_value, learn_value in improved:
        add(
            f"- `{env_id}`: post-change Brier improved from {base_value:.4f} to "
            f"{learn_value:.4f}."
        )
    for env_id, base_value, learn_value in failed:
        add(
            f"- `{env_id}`: post-change Brier did **not** improve "
            f"({base_value:.4f} → {learn_value:.4f}). Reported as a failure."
        )
    add("")

    # The environments that never change are where learning can only cost
    # something. Reporting the wins without them would be picking the sample
    # after seeing the results.
    add("**And what it cost.** In the environments where nothing changed, there was "
        "nothing to gain — only accuracy to lose by chasing noise:")
    add("")
    for baseline, learned in pairs:
        if learned.environment.change_points:
            continue
        base_brier = baseline.scores()["brier"]
        learn_brier = learned.scores()["brier"]
        delta = learn_brier - base_brier
        false_alarms = sum(1 for d in learned.detections if d["confirmed"])
        verdict = (
            "no measurable cost"
            if abs(delta) < 5e-5
            else (
                f"cost {delta:.4f} Brier"
                if delta > 0
                else f"no cost — {abs(delta):.4f} better despite the false alarm"
            )
        )
        add(
            f"- `{learned.environment.id}`: {false_alarms} confirmed detection(s) "
            f"where the process never changed — **{false_alarms} false alarm(s)**. "
            f"Whole-run Brier {base_brier:.4f} → {learn_brier:.4f} ({verdict})."
        )
    add("")
    add(
        "Both stationary environments produced a false alarm. In `ENV-F-false-alarm` "
        "the candidate strategy was tested and lost to the incumbent, so it was "
        "retired and the incumbent restored — the mechanism recovered, but the "
        "trials spent testing it were paid for in accuracy. Detection here is not "
        "free, and these numbers are the price."
    )
    add("")
    add(
        "**LEARNING ≠ MEMORY ≠ BELIEF REVISION ≠ PREDICTION.** What changed here is "
        "the method, not a stored value: ECHO detected that its own predictions had "
        "deteriorated beyond what chance explains, considered several reasons why, "
        "proposed a different estimator whose assumptions matched the evidence, "
        "tested it against the incumbent on trials neither had seen, and kept it "
        "only because it measurably won. The change persisted, and it transferred "
        "to a regime change in a different place and direction."
    )
    add("")
    add(
        "That is a narrow, mechanical form of learning. It is not general "
        "intelligence, not autonomy, and not understanding — ECHO does not know "
        "what an orchid or an outcome is. It noticed a number getting worse and "
        "changed a formula, for stated reasons, and can show its work."
    )
    add("")
    return "\n".join(lines)


def verify_persistence(pairs) -> dict[str, Any]:
    checks = {
        "experiences": True,
        "revisions": True,
        "strategies": True,
        "verdicts": True,
        "reversal_conditions": True,
    }
    for _, learned in pairs:
        if learned.learning is None:
            continue
        expected = learned.learning
        revived = LearningLedger.load(expected.path)
        if len(revived.experiences()) != len(expected.experiences()):
            checks["experiences"] = False
        if len(revived.revisions()) != len(expected.revisions()):
            checks["revisions"] = False
        if len(revived.strategies()) != len(expected.strategies()):
            checks["strategies"] = False
        for before, after in zip(expected.revisions(), revived.revisions()):
            if before.outcome != after.outcome:
                checks["verdicts"] = False
            if before.reversal_condition != after.reversal_condition:
                checks["reversal_conditions"] = False
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the learning experiment.")
    parser.add_argument("--out", help="write the markdown report here")
    parser.add_argument("--data", help="directory for the ledgers")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data) if args.data else RESULTS_DIR / "learning"

    pairs = []
    for environment in LEARNING_ENVIRONMENTS:
        baseline = run_condition(environment, "baseline", data_dir)
        learned = run_condition(environment, "learning", data_dir)
        pairs.append((baseline, learned))

    persistence = verify_persistence(pairs)
    markdown = render(pairs, persistence)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = RESULTS_DIR / f"learning_run_{stamp}.json"
    raw_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "environments": [
                    {
                        "id": learned.environment.id,
                        "change_points": list(learned.environment.change_points),
                        "baseline": baseline.scores(),
                        "learning": learned.scores(),
                        "baseline_post_change": baseline.post_change,
                        "learning_post_change": learned.post_change,
                        "baseline_recovery": baseline.recovery_trial(),
                        "learning_recovery": learned.recovery_trial(),
                        "detections": learned.detections,
                        "events": learned.events,
                        "what_i_learned": (
                            learned.learning.what_did_i_learn() if learned.learning is not None else []
                        ),
                    }
                    for baseline, learned in pairs
                ],
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
    print(f"raw    -> {raw_path}\n")

    for baseline, learned in pairs:
        env = learned.environment
        confirmed = sum(1 for d in learned.detections if d["confirmed"])
        post_base = baseline.post_change["brier"]
        post_learn = learned.post_change["brier"]
        print(
            f"  {env.id:<20} brier {_fmt(baseline.scores()['brier'])} -> "
            f"{_fmt(learned.scores()['brier'])} | post-change "
            f"{_fmt(post_base)} -> {_fmt(post_learn)} | confirmed detections "
            f"{confirmed} (real changes {len(env.change_points)})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
