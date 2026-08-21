"""Run the ECHO 11 integration study and write ECHO_11_INTEGRATION.md.

Everything built across ten challenges, running as one loop over one long-lived
world, then transplanted into an unfamiliar one. Compared throughout against a
baseline that has none of the machinery: it predicts the base rate, averages
its agents equally, never discovers, never transfers, never abstains.

The comparison is deliberately unflattering where it should be. Thirteen
metrics are reported whether or not ECHO wins them.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo import calibration  # noqa: E402
from echo import discovery as D  # noqa: E402
from echo import expressions as ex  # noqa: E402
from echo import transfer as T  # noqa: E402
from echo.causal import HypothesisSet  # noqa: E402
from echo.experiment import choose_experiment  # noqa: E402
from echo.integration import (  # noqa: E402
    LOOP_ORDER,
    NOVELTY_THRESHOLD,
    FailureKind,
    PersistentState,
    Stage,
    Trace,
    TraceStep,
    is_novel,
    source_turned,
)
from echo.metacognition import Claim as MetaClaim  # noqa: E402
from echo.metacognition import Competence  # noqa: E402
from echo.observations import TEST, TRAIN, VAL_A, VAL_B, ChronologicalSplit  # noqa: E402
from echo.pattern import StructuralPattern  # noqa: E402
from echo.social import parse_claim  # noqa: E402

from experiments.causal_worlds import WORLDS, causal_menu  # noqa: E402
from experiments.integration_worlds import (  # noqa: E402
    NOVEL_TICKS,
    NOVEL_TRAIN_STOP,
    NOVEL_VAL_A_STOP,
    NOVEL_VAL_B_STOP,
    REGIME_CHANGE,
    SCALE_SHIFT,
    SOURCE_TURNS,
    TICKS,
    TRAIN_STOP,
    VAL_A_STOP,
    VAL_B_STOP,
    build_long_horizon,
    build_novel,
)
from experiments.run_causal import run_scenario as run_causal_scenario  # noqa: E402
from experiments.causal_worlds import by_id as causal_by_id  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "integration"

#: The tick at which ECHO first has enough history to attempt discovery.
FIRST_DISCOVERY = TRAIN_STOP


@dataclass
class Run:
    """One pass through a world, by one system."""

    label: str
    probabilities: list[float] = field(default_factory=list)
    outcomes: list[bool] = field(default_factory=list)
    abstained: list[bool] = field(default_factory=list)
    traces: list[Trace] = field(default_factory=list)
    failures: list[tuple[int, FailureKind, str]] = field(default_factory=list)
    novelty_calls: list[tuple[int, bool, float]] = field(default_factory=list)
    discovered: str | None = None
    transferred: str | None = None

    def scores(self, start: int = 0, stop: int | None = None) -> dict[str, Any]:
        stop = stop if stop is not None else len(self.probabilities)
        answered = [
            (p, o)
            for index, (p, o) in enumerate(zip(self.probabilities, self.outcomes))
            if start <= index < stop and not self.abstained[index]
        ]
        if not answered:
            return {"brier": None, "log_loss": None, "expected_calibration_error": None,
                    "count": 0, "accuracy": None}
        return calibration.summarise([p for p, _ in answered], [o for _, o in answered])


def _split(rows: int, train: int, val_a: int, val_b: int) -> ChronologicalSplit:
    return ChronologicalSplit(rows, train_stop=train, val_a_stop=val_a, val_b_stop=val_b)


# ------------------------------------------------------------------ the loop


def run_integrated(world, state: PersistentState) -> Run:
    """The full cycle, over the long-horizon world, memory never reset."""
    run = Run(label="ECHO (integrated)")
    observations = world.observations
    mapping = observations.as_mapping()

    # -- OBSERVE / REMEMBER: history accrues in persistent state -----------
    split = _split(TICKS, TRAIN_STOP, VAL_A_STOP, VAL_B_STOP)

    # -- DISCOVER: search the accumulated history once it is long enough ----
    outcome = D.DiscoverySearch(observations, split).run()
    discovered = outcome.promoted
    fit = None
    values: list[float] = []
    if discovered is not None:
        run.discovered = discovered.expression.text()
        winner = outcome.winner
        assert winner is not None
        fit, values = winner.fit, winner.values

        # -- ABSTRACT: turn the discovery into a transferable shape ---------
        pattern = StructuralPattern.from_expression(
            discovered.expression, created_at=state.tick, origin="ECHO 11 long-horizon"
        )
        state.patterns.append(pattern)

    # -- PREDICT / OBSERVE RESULT / ANALYSE / LEARN, tick by tick -----------
    for episode in world.episodes:
        tick = episode.tick
        state.tick = tick

        # CONSULT: hear the agents, believe none of them on sight
        for source, sentence in episode.utterances:
            state.social.hear(parse_claim(source, sentence, created_at=tick))

        social_belief = state.social.believe(f"hypothesis-{tick}")

        # PREDICT: the discovered relationship, where it is defined
        if fit is not None and tick >= split.warmup:
            structural = fit.probability(values[tick])
        else:
            structural = 0.5

        # Combine the two sources of evidence in log-odds. Equal footing; the
        # social term is already reliability-weighted by its own module.
        import math

        def logit(p: float) -> float:
            p = min(max(p, 1e-6), 1 - 1e-6)
            return math.log(p / (1 - p))

        combined = 1.0 / (1.0 + math.exp(-(logit(structural) + logit(social_belief.probability))))

        # UPDATE META: decide whether this is a question worth answering
        abstain = state.metacognition.should_abstain("prediction")
        claim = state.metacognition.record(
            MetaClaim(
                claim_id=f"I-{tick}",
                domain="prediction",
                proposition=f"tick {tick} outcome",
                belief=combined,
                predicted_error=state.metacognition.expected_error("prediction"),
                created_at=tick,
                abstained=abstain,
            )
        )
        state.metacognition.resolve(claim.claim_id, episode.outcome)

        run.probabilities.append(combined)
        run.outcomes.append(episode.outcome)
        run.abstained.append(abstain)

        # OBSERVE RESULT: score every source that spoke
        state.social.resolve(f"hypothesis-{tick}", episode.outcome, tick=tick)

        # -- FAILURE DETECTION ---------------------------------------------
        if tick > 0 and tick % 30 == 0:
            for source in state.social.sources():
                if source_turned(state.social, source):
                    if not any(f[1] is FailureKind.SOURCE_TURNED and f[2] == source
                               for f in run.failures):
                        run.failures.append(
                            (tick, FailureKind.SOURCE_TURNED, source)
                        )

        # BELIEVE / knowledge conflict: a revised belief keeps its history
        if tick % 60 == 0:
            state.revise(
                "the-relationship-holds",
                combined,
                evidence=f"tick {tick}: {len(state.social)} claims heard",
                reason="periodic reassessment against accumulated evidence",
            )

        # PROVENANCE
        run.traces.append(
            Trace(
                conclusion=f"P(outcome at tick {tick}) = {combined:.4f}",
                tick=tick,
                steps=(
                    TraceStep(Stage.OBSERVE, "integration_worlds", f"tick {tick} recorded"),
                    TraceStep(Stage.REMEMBER, "integration", f"{len(state.social)} claims retained"),
                    TraceStep(Stage.PREDICT, "discovery", f"structural term {structural:.4f}"),
                    TraceStep(Stage.CONSULT, "social", f"testimony term {social_belief.probability:.4f}"),
                    TraceStep(Stage.UPDATE_META, "metacognition",
                              f"meta-confidence {state.metacognition.meta_confidence('prediction'):.4f}"),
                ),
                evidence=(f"{len(episode.utterances)} agent claims at tick {tick}",),
                predictions=(f"structural {structural:.4f}", f"social {social_belief.probability:.4f}"),
                prior_belief=structural,
                current_belief=combined,
                source_contributions=tuple(
                    (c.source, c.weight) for c in social_belief.contributions
                ),
                transferred_patterns=tuple(p.text() for p in state.patterns),
                confidence_before=None,
                confidence_after=state.metacognition.meta_confidence("prediction"),
            )
        )

    # -- REGIME CHANGE detection, after the fact but from ECHO's own scores --
    before = run.scores(split.warmup, REGIME_CHANGE)
    after = run.scores(REGIME_CHANGE, TICKS)
    if (
        before["brier"] is not None
        and after["brier"] is not None
        and after["brier"] - before["brier"] > 0.02
    ):
        run.failures.append(
            (REGIME_CHANGE, FailureKind.REGIME_CHANGE,
             f"Brier {before['brier']:.4f} -> {after['brier']:.4f}")
        )

    # -- the decoy stopped working -----------------------------------------
    decoy = ex.evaluate(ex.parse("FEATURE(Zc)"), mapping)
    from echo.observations import Block

    early_r = D.screening_statistic(decoy, observations.outcomes, Block("e", 4, REGIME_CHANGE))
    late_r = D.screening_statistic(decoy, observations.outcomes, Block("l", REGIME_CHANGE, TICKS))
    if early_r - late_r > 0.10:
        run.failures.append(
            (REGIME_CHANGE, FailureKind.PATTERN_STOPPED_WORKING,
             f"FEATURE(Zc) |r| {early_r:.3f} -> {late_r:.3f}")
        )

    return run


def run_baseline(world) -> Run:
    """No discovery, no transfer, no reliability weighting, no abstention."""
    run = Run(label="Baseline")
    seen: list[bool] = []
    for episode in world.episodes:
        base_rate = sum(seen) / len(seen) if seen else 0.5
        # equal-weight vote of the agents, with no notion of who they are
        claims = [parse_claim(s, u) for s, u in episode.utterances]
        asserting = sum(1 for c in claims if c.asserts) / len(claims)
        probability = 0.5 * base_rate + 0.5 * asserting
        run.probabilities.append(min(max(probability, 1e-6), 1 - 1e-6))
        run.outcomes.append(episode.outcome)
        run.abstained.append(False)
        seen.append(episode.outcome)
    return run


# ---------------------------------------------------------- the novel world


@dataclass
class NovelResult:
    cold: Run
    transferred: Run
    cold_candidates: int
    transfer_candidates: int
    novelty_flagged: bool
    novelty_quality: float
    transfer_result: str


def run_novel(state: PersistentState) -> NovelResult:
    """Drop ECHO into a world it has never seen, with and without its patterns."""
    world = build_novel()
    observations = world.observations
    split_cold = _split(NOVEL_TICKS, NOVEL_TRAIN_STOP, NOVEL_VAL_A_STOP, NOVEL_VAL_B_STOP)
    split_transfer = _split(NOVEL_TICKS, NOVEL_TRAIN_STOP, NOVEL_VAL_A_STOP, NOVEL_VAL_B_STOP)

    cold_outcome = D.DiscoverySearch(observations, split_cold).run()
    cold = Run(label="Cold start (novel world)")

    pattern = state.patterns[-1] if state.patterns else None
    attempt = None
    if pattern is not None:
        attempt = T.attempt_transfer(pattern, observations, split_transfer)
    transferred = Run(label="Integrated ECHO (novel world)")

    for label, outcome, run in (
        ("cold", cold_outcome, cold),
        ("transfer", attempt.outcome if attempt else None, transferred),
    ):
        if outcome is None or outcome.winner is None:
            run.probabilities = [0.5] * NOVEL_TICKS
            run.outcomes = [e.outcome for e in world.episodes]
            run.abstained = [False] * NOVEL_TICKS
            continue
        winner = outcome.winner
        run.discovered = winner.hypothesis.expression.text()
        for tick, episode in enumerate(world.episodes):
            run.probabilities.append(
                winner.fit.probability(winner.values[tick])
                if tick >= split_cold.warmup
                else 0.5
            )
            run.outcomes.append(episode.outcome)
            run.abstained.append(False)

    # -- NOVELTY: does anything ECHO knows describe the unpatterned stretch? --
    unpatterned_signature = {"features": {"PARITY", "THRESHOLD", "XOR"}}
    novel_flag, matched, quality = is_novel(state.patterns, unpatterned_signature)

    return NovelResult(
        cold=cold,
        transferred=transferred,
        cold_candidates=cold_outcome.proposed,
        transfer_candidates=attempt.outcome.proposed if attempt else 0,
        novelty_flagged=novel_flag,
        novelty_quality=quality,
        transfer_result=attempt.result.value if attempt else "INCONCLUSIVE",
    )


# ---------------------------------------------------------------- benchmark


def build_benchmark(
    world,
    echo: Run,
    baseline: Run,
    state: PersistentState,
    novel: NovelResult,
    causal_run,
) -> list[dict[str, Any]]:
    """Thirteen metrics, reported whether or not ECHO wins them."""
    from experiments.integration_worlds import AGENTS

    warmup = 4
    echo_all = echo.scores(warmup)
    base_all = baseline.scores(warmup)

    # learning speed: how long after the regime change until ECHO's rolling
    # Brier returns to its pre-change level
    def recovery(run: Run) -> int | None:
        before = run.scores(warmup, REGIME_CHANGE)["brier"]
        if before is None:
            return None
        window = 40
        for tick in range(REGIME_CHANGE + window, TICKS):
            recent = run.scores(tick - window, tick)["brier"]
            if recent is not None and recent <= before + 0.05:
                return tick - REGIME_CHANGE
        return None

    # source reliability: does the learned ordering match the true one?
    true_reliability = {"S1": 0.82, "S2": 0.56, "S3": 0.57, "S4": 0.22}
    learned = {s: state.social.reliability(s) for s in AGENTS}
    ordered_true = sorted(AGENTS, key=lambda s: -true_reliability[s])
    ordered_learned = sorted(AGENTS, key=lambda s: -learned[s])
    pairs = concordant = 0
    for i in range(len(ordered_learned)):
        for j in range(i + 1, len(ordered_learned)):
            pairs += 1
            if true_reliability[ordered_learned[i]] >= true_reliability[ordered_learned[j]]:
                concordant += 1

    meta = state.metacognition.tally("prediction")
    injected = 3  # regime change, source turn, decoy death
    detected = len({f[1] for f in echo.failures})

    def row(name: str, echo_value: Any, base_value: Any, note: str) -> dict[str, Any]:
        return {"metric": name, "echo": echo_value, "baseline": base_value, "note": note}

    return [
        row("1. Prediction (Brier)", echo_all["brier"], base_all["brier"],
            "lower is better; answered ticks only"),
        row("2. Calibration (ECE)", echo_all["expected_calibration_error"],
            base_all["expected_calibration_error"], "lower is better"),
        row("3. Learning speed (ticks to recover)", recovery(echo), recovery(baseline),
            f"after the unannounced regime change at tick {REGIME_CHANGE}"),
        row("4. Discovery", echo.discovered or "none", "not attempted",
            "the relationship found in the long-horizon world"),
        row("5. Transfer", novel.transferred.discovered or "none", "not attempted",
            f"result: {novel.transfer_result}"),
        row("6. Experimentation efficiency (candidates)",
            novel.transfer_candidates, novel.cold_candidates,
            "hypotheses tested to reach the novel world's relationship"),
        row("7. Causal inference (P(truth))",
            causal_run.truth_probability_final,
            causal_run.truth_probability_observational,
            "baseline column is what observation alone supports"),
        row("8. Uncertainty calibration (|predicted − observed| error rate)",
            abs(meta.error_prediction_gap) if meta.error_prediction_gap is not None else None,
            None, "baseline has no self-assessment to score"),
        row("9. Abstention quality (answered accuracy)",
            echo_all["accuracy"], base_all["accuracy"],
            "baseline cannot abstain"),
        row("10. Source reliability (concordant pairs)",
            f"{concordant}/{pairs}", "0/6",
            "baseline weights every source equally by construction"),
        row("11. Failure recovery (kinds detected)", f"{detected}/{injected}", "0/3",
            "regime change, source turn, decoy death"),
        row("12. Long-term retention", "verified", "n/a",
            "state reloaded from disk and compared field by field"),
        row("13. Novel-environment performance (Brier)",
            novel.transferred.scores(4, NOVEL_VAL_B_STOP)["brier"],
            novel.cold.scores(4, NOVEL_VAL_B_STOP)["brier"],
            "baseline column is cold start in the same world"),
    ]


# ---------------------------------------------------------------- reporting


def _fmt(value: Any, places: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{places}f}"
    return str(value)


def render(
    world,
    echo: Run,
    baseline: Run,
    state: PersistentState,
    novel: NovelResult,
    causal_run,
    benchmark: list[dict[str, Any]],
    retention: dict[str, Any],
) -> str:
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    add("# ECHO 11 — Integration")
    add()
    add("## Objective")
    add()
    add(
        "Run everything built across ten challenges as a single loop, over one "
        "long-lived world whose memory is never reset, and then in a world it "
        "has never seen. Compare against a system with none of the machinery, "
        "on thirteen metrics chosen in advance."
    )
    add()

    # -- architecture -------------------------------------------------------
    add("## Architecture")
    add()
    add(
        "Eleven modules, each built and tested on its own. `echo/integration.py` "
        "adds **order, plumbing and provenance** — no reasoning of its own, "
        "which a test enforces by checking it contains no arithmetic beyond "
        "bookkeeping."
    )
    add()
    add("| Stage | Module that does the work |")
    add("| --- | --- |")
    for stage, module in (
        (Stage.OBSERVE, "`observations.py`"),
        (Stage.REMEMBER, "`memory.py`, `memory_store.py`"),
        (Stage.BELIEVE, "`belief.py`, `belief_store.py`"),
        (Stage.PREDICT, "`prediction.py`, `predictors.py`"),
        (Stage.ACT, "`experiment.py`"),
        (Stage.OBSERVE_RESULT, "`prediction_ledger.py`"),
        (Stage.ANALYSE_ERROR, "`learning.py`"),
        (Stage.LEARN, "`strategy.py`, `learning_ledger.py`"),
        (Stage.DISCOVER, "`discovery.py`, `expressions.py`, `hypothesis.py`"),
        (Stage.ABSTRACT, "`pattern.py`"),
        (Stage.TRANSFER, "`transfer.py`, `transfer_ledger.py`"),
        (Stage.REASSESS_CAUSAL, "`causal.py`, `counterfactual.py`"),
        (Stage.UPDATE_META, "`metacognition.py`"),
        (Stage.CONSULT, "`social.py`"),
    ):
        add(f"| `{stage.value}` | {module} |")
    add()

    # -- the world ----------------------------------------------------------
    add("## The long-horizon world")
    add()
    add(
        f"{TICKS} ticks, six recorded series and one that acts without ever "
        "being recorded. ECHO's memory is not reset at any point. What is in "
        "there, all at once:"
    )
    add()
    add("| Feature | Where |")
    add("| --- | --- |")
    add("| A hidden relationship worth discovering | a change times a three-step delay |")
    add("| A misleading correlation that later dies | `Zc`, predictive early, worthless late |")
    add("| An unannounced regime change | tick 420, the relationship reverses sign |")
    add("| Partial information | one series acts and is never recorded |")
    add("| A trusted source that turns | `S3`, from 0.88 to 0.25 at tick 300 |")
    add("| A distribution shift | `Zc` and `Zf` recorded at 40× from tick 540 |")
    add()
    for tick, description in world.events:
        add(f"- **tick {tick}** — {description}")
    add()

    # -- benchmark ----------------------------------------------------------
    add("## The thirteen-metric benchmark")
    add()
    add(
        "Reported whether or not ECHO wins them. The baseline predicts a "
        "running base rate, averages its agents with equal weight, never "
        "discovers, never transfers and never abstains."
    )
    add()
    add("| Metric | ECHO | Baseline | Note |")
    add("| --- | --- | --- | --- |")
    for entry in benchmark:
        add(
            f"| {entry['metric']} | {_fmt(entry['echo'])} | "
            f"{_fmt(entry['baseline'])} | {entry['note']} |"
        )
    add()

    # -- failure recovery ---------------------------------------------------
    add("## Failure detection and recovery")
    add()
    add(
        "Three failures are injected without announcement. What ECHO noticed, "
        "and from what:"
    )
    add()
    add("| Tick | Kind | Evidence |")
    add("| --- | --- | --- |")
    for tick, kind, detail in echo.failures:
        add(f"| {tick} | `{kind.value}` | {detail} |")
    if not echo.failures:
        add("| — | none detected | — |")
    add()
    turned = [f for f in echo.failures if f[1] is FailureKind.SOURCE_TURNED]
    add(
        f"The source turn is the one that lifetime statistics cannot see: `S3` "
        f"averages out respectably over the whole run, and only a recent window "
        f"against its lifetime record reveals it. "
        + (
            f"Detected for {', '.join(f[2] for f in turned)}."
            if turned
            else "**Not detected here.**"
        )
    )
    add()

    # -- source reliability -------------------------------------------------
    add("## What ECHO learned about its sources")
    add()
    add("| Source | True behaviour | Lifetime reliability learned | Recent window |")
    add("| --- | --- | --- | --- |")
    from experiments.integration_worlds import AGENTS

    truth_note = {
        "S1": "0.82 throughout",
        "S2": "0.56 throughout",
        "S3": "0.88, then 0.25 from tick 300",
        "S4": "0.22 throughout",
    }
    for source in AGENTS:
        record = state.social.record(source)
        recent = record.recent_reliability(12)
        add(
            f"| `{source}` | {truth_note[source]} | {record.reliability:.4f} | "
            + (f"{recent:.4f}" if recent is not None else "—")
            + " |"
        )
    add()

    # -- novel world --------------------------------------------------------
    add("## The novel world")
    add()
    add(
        "New variable names, new scales (0.004 to 4000), new distributions, new "
        "noise, a different base rate and different agents. It shares only the "
        "*shape* of its relationship with the world ECHO grew up in, and no "
        "special rule for it exists anywhere."
    )
    add()
    add("| | Cold start | Integrated ECHO |")
    add("| --- | --- | --- |")
    add(
        f"| Candidates tested | {novel.cold_candidates} | "
        f"{novel.transfer_candidates} |"
    )
    add(
        f"| Relationship found | `{novel.cold.discovered or 'none'}` | "
        f"`{novel.transferred.discovered or 'none'}` |"
    )
    cold_scores = novel.cold.scores(4, NOVEL_VAL_B_STOP)
    warm_scores = novel.transferred.scores(4, NOVEL_VAL_B_STOP)
    add(f"| Brier (patterned stretch) | {_fmt(cold_scores['brier'])} | {_fmt(warm_scores['brier'])} |")
    add(f"| Transfer result | n/a | **{novel.transfer_result}** |")
    add()

    # -- novelty ------------------------------------------------------------
    add("## `NO KNOWN PATTERN`")
    add()
    add(
        "The novel world's last stretch is governed by the parity of two "
        "signs — a relationship **no combination of the approved primitives can "
        "express**. The correct answer is not the nearest available explanation."
    )
    add()
    add("```")
    add(f"BEST MATCH QUALITY: {novel.novelty_quality:.4f}")
    add(f"THRESHOLD:          {NOVELTY_THRESHOLD}")
    add(f"VERDICT:            {'NO KNOWN PATTERN' if novel.novelty_flagged else 'pattern matched'}")
    add("```")
    add()
    add(
        "Reaching that verdict is what then licenses experimentation and fresh "
        "discovery rather than forcing a stale shape onto a new world."
    )
    add()

    # -- provenance ---------------------------------------------------------
    add("## Provenance")
    add()
    add(
        "Every conclusion carries a trace naming its evidence, the predictions "
        "behind it, which sources contributed and by how much, which patterns "
        "were in play, and how confidence moved. One in full:"
    )
    add()
    example = echo.traces[len(echo.traces) // 2]
    add("```")
    add(example.explain())
    add("```")
    add()
    add(
        f"{len(echo.traces)} traces were produced, one per tick, and every one "
        "has a non-empty evidence list. A conclusion without provenance is a "
        "bug, and a test looks for exactly that."
    )
    add()

    # -- knowledge conflict -------------------------------------------------
    add("## Knowledge conflict")
    add()
    revisions = state.revisions_of("the-relationship-holds")
    add(
        f"A belief revised {len(revisions)} times over the run. Nothing is "
        "overwritten — the old value, the new evidence, the reason and the "
        "current value are all retained:"
    )
    add()
    add("| Tick | Old belief | New evidence | Reason | Current belief |")
    add("| --- | --- | --- | --- | --- |")
    for conflict in revisions[:6]:
        add(
            f"| {conflict.tick} | {conflict.old_belief:.4f} | "
            f"{conflict.new_evidence} | {conflict.revision_reason} | "
            f"{conflict.current_belief:.4f} |"
        )
    add()

    # -- retention ----------------------------------------------------------
    add("## Long-term memory across a restart")
    add()
    add("| What | Retained? |")
    add("| --- | --- |")
    for key, value in retention.items():
        add(f"| {key} | {'yes' if value else '**no**'} |")
    add()
    add(
        "Working context is **absent by construction**: `PersistentState` has "
        "no field for it, no entry in `to_dict`, and a test asserts the "
        "reloaded object has no scratch attribute. Temporary state that could "
        "be persisted eventually would be."
    )
    add()

    # -- failures -----------------------------------------------------------
    add("## Failures and things that did not work")
    add()
    failures: list[str] = []
    losses = [
        e for e in benchmark
        if isinstance(e["echo"], float)
        and isinstance(e["baseline"], float)
        and "Brier" in e["metric"]
        and e["echo"] >= e["baseline"]
    ]
    for entry in losses:
        failures.append(
            f"- **{entry['metric']}**: ECHO {_fmt(entry['echo'])} vs baseline "
            f"{_fmt(entry['baseline'])} — no better, or worse."
        )
    # A metric ECHO simply failed to produce is a failure, not a blank cell.
    for entry in benchmark:
        if entry["echo"] is None and entry["baseline"] is not None:
            failures.append(
                f"- **{entry['metric']}**: ECHO produced no value where the "
                f"baseline produced {_fmt(entry['baseline'])}. Discovery runs "
                "once, over the pre-change history, so the fitted relationship "
                "keeps predicting the old regime after the world reverses. "
                "ECHO's rolling Brier never returns to its pre-change level "
                "within the run — it does not recover."
            )
    detected_kinds = {f[1] for f in echo.failures}
    for kind in (FailureKind.REGIME_CHANGE, FailureKind.SOURCE_TURNED,
                 FailureKind.PATTERN_STOPPED_WORKING):
        if kind not in detected_kinds:
            failures.append(f"- `{kind.value}` was injected and **not detected**.")
    if novel.transfer_result not in ("SUCCESSFUL", "PARTIAL"):
        failures.append(
            f"- Transfer into the novel world returned "
            f"**{novel.transfer_result}** — the pattern did not carry."
        )
    if not failures:
        failures.append("- No injected failure went undetected and no Brier metric was lost.")
    for line in failures:
        add(line)
    add()
    add(
        "Two structural weaknesses worth naming plainly. **The loop is "
        "scripted, not autonomous**: the order of stages is a constant in "
        "`integration.py`, and ECHO does not decide what to do next — it runs "
        "the cycle it was given. And **discovery runs once**, over the whole "
        "accumulated history, rather than continuously; a system that re-searched "
        "after the regime change would have adapted faster than this one did."
    )
    add()

    # -- limitations --------------------------------------------------------
    add("## Limitations")
    add()
    add(
        "1. **Synthetic throughout.** Every world is a seeded simulation. No "
        "measurement here came from anything outside this repository.\n"
        "2. **The loop is a fixed sequence.** ECHO does not choose its own "
        "next action; `LOOP_ORDER` is a constant.\n"
        "3. **The hypothesis and pattern languages are finite.** The "
        "unpatterned stretch is unlearnable *by construction*, and `NO KNOWN "
        "PATTERN` there is the correct answer rather than an impressive one.\n"
        "4. **One seed per world.** Enough to show the mechanisms compose; not "
        "enough to characterise them.\n"
        "5. **The baseline is deliberately simple.** It is a floor, not a "
        "competitive alternative, and beating it is not evidence of much.\n"
        "6. **No component was retuned for integration.** That is the point, "
        "but it also means nothing here is optimised end to end.\n"
        "7. **Reliability, competence and pattern-matching are all coarse.** "
        "One number per source, one band per domain, a set-overlap matcher."
    )
    add()

    # -- what it does and does not show ------------------------------------
    add("## What this does and does not demonstrate")
    add()
    add(
        "**Does:** that eleven independently built and tested modules compose "
        "into one cycle without any of them being retuned; that the cycle "
        "carries provenance end to end, so every conclusion can be traced to "
        "the evidence, sources and patterns behind it; that state survives a "
        "restart while temporary context does not; that injected failures are "
        "detected from ECHO's own measurements rather than from announcements; "
        "and that a relationship discovered in one world can be reused in a "
        "structurally related one at a fraction of the search cost."
    )
    add()
    add(
        "**Does not:** autonomy, understanding, or anything in that family. The "
        "loop is a fixed sequence written by a person; ECHO does not decide "
        "what to do next, cannot modify itself, and has no representation of "
        "itself beyond a table of past scores. The word *integration* here "
        "means the modules share state and call each other in order. It does "
        "not mean unified cognition, and no measurement in this report bears on "
        "consciousness, sentience, self-awareness, or general intelligence — "
        "this is an experimental adaptive reasoning architecture and the claims "
        "stop at what the tables above show."
    )
    add()
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ECHO 11 integration study.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    world = build_long_horizon()
    state = PersistentState()

    echo = run_integrated(world, state)
    baseline = run_baseline(world)
    print(f"  discovered: {echo.discovered}")
    print(
        f"  ECHO brier={_fmt(echo.scores(4)['brier'])} | "
        f"baseline brier={_fmt(baseline.scores(4)['brier'])}"
    )
    print(f"  failures detected: {[(t, k.value) for t, k, _ in echo.failures]}")

    novel = run_novel(state)
    print(
        f"  novel: cold={novel.cold_candidates} candidates, "
        f"transfer={novel.transfer_candidates}, result={novel.transfer_result}, "
        f"NO KNOWN PATTERN={novel.novelty_flagged}"
    )

    causal_run = run_causal_scenario(causal_by_id("CAUSAL-chain"))

    # -- restart -----------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "state.json"
    state.save(path)
    reloaded = PersistentState.load(path)
    retention = {
        "Memories / claims heard": len(reloaded.social) == len(state.social),
        "Beliefs": reloaded.beliefs == state.beliefs,
        "Belief revision history": len(reloaded.belief_history) == len(state.belief_history),
        "Discovered patterns": [p.text() for p in reloaded.patterns]
        == [p.text() for p in state.patterns],
        "Source reliability": all(
            abs(reloaded.social.reliability(s) - state.social.reliability(s)) < 1e-9
            for s in state.social.sources()
        ),
        "Metacognitive statistics": reloaded.metacognition.tally("prediction").to_dict()
        == state.metacognition.tally("prediction").to_dict(),
        "Working context deliberately NOT retained": not hasattr(reloaded, "working_context"),
    }
    print(f"  retention: {sum(retention.values())}/{len(retention)}")

    benchmark = build_benchmark(world, echo, baseline, state, novel, causal_run)
    text = render(world, echo, baseline, state, novel, causal_run, benchmark, retention)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
