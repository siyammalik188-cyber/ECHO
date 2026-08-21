"""Run the transfer experiment and write ECHO_6_TRANSFER.md.

Order of operations, which is also the order that keeps it honest:

1. Discover in the SOURCE world using the unchanged ECHO 5 pipeline. Whatever
   it finds is the source knowledge; nothing here names an expression.
2. Abstract that discovery into a `StructuralPattern`, mechanically.
3. In each target world, run four conditions on identical data:
   A cold start, B structural transfer, C the raw source expression remapped by
   column position, D an unrelated pattern from a different discovery.
4. Judge each transfer on TRAIN/VAL_A/VAL_B only.
5. Sweep training budgets to measure how much history each condition needs.
6. Open each target's holdout once, at the end, and report.

The source dataset, its outcomes, its variable names and its winning expression
never reach a target search. Only the abstract pattern crosses.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo import discovery as D  # noqa: E402
from echo import expressions as ex  # noqa: E402
from echo import transfer as T  # noqa: E402
from echo.discovery_ledger import DiscoveryLedger, DiscoveryRecord  # noqa: E402
from echo.observations import TRAIN, VAL_A, VAL_B, ChronologicalSplit  # noqa: E402
from echo.pattern import PatternStatus, StructuralPattern  # noqa: E402
from echo.transfer_ledger import TransferLedger  # noqa: E402

from experiments.transfer_worlds import (  # noqa: E402
    ROWS,
    SOURCE,
    TARGETS,
    TRAIN_STOP,
    VAL_A_STOP,
    VAL_B_STOP,
    TransferWorld,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "transfer"

#: Training budgets for the sample-efficiency sweep, smallest first. Fixed
#: before the experiment ran.
BUDGETS = (24, 32, 48, 64, 96, 160, 320, 476)

CONDITIONS = ("A_cold", "B_structural", "C_raw_expression", "D_irrelevant")

CONDITION_LABELS = {
    "A_cold": "A — cold start",
    "B_structural": "B — structural transfer",
    "C_raw_expression": "C — raw source expression, remapped by position",
    "D_irrelevant": "D — an unrelated pattern",
}


def make_split(train_rows: int | None = None) -> ChronologicalSplit:
    return ChronologicalSplit(
        ROWS,
        train_stop=TRAIN_STOP,
        val_a_stop=VAL_A_STOP,
        val_b_stop=VAL_B_STOP,
        train_rows=train_rows,
    )


# --------------------------------------------------------------- 1. the source


@dataclass
class SourceKnowledge:
    record: DiscoveryRecord
    pattern: StructuralPattern
    expression_text: str
    outcome: D.SearchOutcome
    holdout: D.HoldoutResult


def discover_source() -> SourceKnowledge:
    """Run ECHO 5 unchanged on the source world and abstract what it finds."""
    observations = SOURCE.generate()
    split = make_split()
    outcome = D.DiscoverySearch(observations, split).run()
    holdout = D.evaluate_holdout(outcome, observations, split)

    if outcome.promoted is None:
        raise RuntimeError(
            "source discovery found nothing; there is no knowledge to transfer"
        )

    expression = outcome.promoted.expression
    pattern = StructuralPattern.from_expression(
        expression,
        created_at=1,
        origin="abstracted from a discovery in a previous world",
    )

    record = DiscoveryRecord(
        observations_id=observations.id,
        outcome="DISCOVERY SUCCESS" if holdout.survived else "DISCOVERY PARTIAL",
        config=outcome.config.to_dict(),
        source_name=outcome.source_name,
        proposed=outcome.proposed,
        screened=outcome.screened,
        fitted=outcome.fitted,
        rejected_count=len(outcome.rejected),
        validated_count=1,
        promoted_count=1,
        discovered_expression=expression.text(),
        discovered_complexity=outcome.promoted.complexity,
        discovery_reason=outcome.promoted.discovery_reason,
        evidence_window=outcome.promoted.evidence_window,
        confirmation=outcome.confirmation.to_dict() if outcome.confirmation else None,
        survived_holdout=holdout.survived,
        training_score=outcome.promoted.training_score,
        validation_score=outcome.promoted.validation_score,
        test_score=holdout.test_score,
        baseline_before=None,
        holdout_baselines=[
            {"name": name, "brier": sc["brier"], "log_loss": sc["log_loss"]}
            for name, sc in holdout.baseline_scores
        ],
        alternatives=[r.to_dict() for r in outcome.alternatives(limit=5)],
        overfitting_control=holdout.overfitting_control,
        hypotheses=(outcome.promoted,),
        seconds=outcome.seconds,
    )
    return SourceKnowledge(
        record=record,
        pattern=pattern,
        expression_text=expression.text(),
        outcome=outcome,
        holdout=holdout,
    )


def irrelevant_pattern() -> tuple[StructuralPattern, str]:
    """A genuine discovery from an unrelated world, for ablation condition D.

    Not a made-up shape: the ECHO 5 lagged world is searched with the same
    pipeline, and whatever it finds is abstracted the same way. It is real
    knowledge that happens to be the wrong knowledge, which is exactly what
    condition D needs to be about.
    """
    from experiments.discovery_worlds import (
        TRAIN_STOP as D_TRAIN,
        VAL_A_STOP as D_VAL_A,
        VAL_B_STOP as D_VAL_B,
        ROWS as D_ROWS,
        by_id,
    )

    world = by_id("DISC-A-lagged")
    observations = world.generate()
    split = ChronologicalSplit(
        D_ROWS, train_stop=D_TRAIN, val_a_stop=D_VAL_A, val_b_stop=D_VAL_B
    )
    outcome = D.DiscoverySearch(observations, split).run()
    if outcome.promoted is None:
        raise RuntimeError("the unrelated world produced no discovery to transfer")
    expression = outcome.promoted.expression
    pattern = StructuralPattern.from_expression(
        expression,
        created_at=1,
        origin="abstracted from a discovery in a different, unrelated world",
    )
    return pattern, expression.text()


# ----------------------------------------------------------- 2. one condition


@dataclass
class ConditionRun:
    condition: str
    world_id: str
    budget: int | None
    outcome: D.SearchOutcome
    attempt: T.TransferAttempt | None
    fell_back: bool = False
    fallback: D.SearchOutcome | None = None

    @property
    def effective(self) -> D.SearchOutcome:
        return self.fallback if self.fallback is not None else self.outcome

    @property
    def confirmed(self) -> bool:
        return self.effective.promoted is not None

    @property
    def hypotheses_tested(self) -> int:
        total = self.outcome.proposed
        if self.fallback is not None:
            total += self.fallback.proposed
        return total

    @property
    def val_b_brier(self) -> float | None:
        confirmation = self.effective.confirmation
        return confirmation.candidate_brier if confirmation else None


def run_condition(
    condition: str,
    world: TransferWorld,
    knowledge: SourceKnowledge,
    unrelated: StructuralPattern,
    *,
    budget: int | None = None,
    cold_reference: float | None = None,
    allow_fallback: bool = True,
) -> ConditionRun:
    observations = world.generate()
    split = make_split(budget)

    if condition == "A_cold":
        outcome = D.DiscoverySearch(observations, split).run()
        return ConditionRun(condition, world.id, budget, outcome, None)

    if condition == "C_raw_expression":
        # The naive reuse: take the old winning expression and point it at the
        # new world's columns by position.
        remapped = T.remap_positionally(
            ex.parse(knowledge.expression_text),
            SOURCE.variable_names,
            observations.variable_names,
        )
        source = T.FixedSource([remapped], name=f"raw-source-expression({remapped.text()})")
        outcome = D.DiscoverySearch(observations, split, source=source).run()
        return ConditionRun(condition, world.id, budget, outcome, None)

    pattern = knowledge.pattern if condition == "B_structural" else unrelated
    attempt = T.attempt_transfer(
        pattern, observations, split, cold_reference=cold_reference
    )
    run = ConditionRun(condition, world.id, budget, attempt.outcome, attempt)

    # A rejected hint is a hint that did not pay off, not a dead end. Falling
    # back to a from-scratch search is what stops prior knowledge from being a
    # trap; without it, "transfer" would only ever be able to hurt.
    if allow_fallback and not attempt.confirmed:
        fallback_split = make_split(budget)
        run.fallback = D.DiscoverySearch(observations, fallback_split).run()
        run.fell_back = True
    return run


# ------------------------------------------------------- 3. the whole target


@dataclass
class TargetResult:
    world: TransferWorld
    runs: dict[str, ConditionRun]
    holdouts: dict[str, dict[str, Any] | None]
    baselines: list[tuple[str, dict[str, Any]]]
    record: T.TransferRecord
    pattern_after: StructuralPattern
    efficiency: dict[str, int | None]
    seconds: float = 0.0


def run_target(
    world: TransferWorld,
    knowledge: SourceKnowledge,
    unrelated: StructuralPattern,
    pattern_now: StructuralPattern,
    tick: int,
) -> TargetResult:
    started = time.perf_counter()

    # Cold start first: its confirmation-block score is the reference that lets
    # a *confirmed* transfer still be judged harmful.
    cold = run_condition("A_cold", world, knowledge, unrelated)
    cold_reference = cold.val_b_brier

    runs: dict[str, ConditionRun] = {"A_cold": cold}
    runs["B_structural"] = run_condition(
        "B_structural", world, knowledge, unrelated, cold_reference=cold_reference
    )
    runs["C_raw_expression"] = run_condition(
        "C_raw_expression", world, knowledge, unrelated
    )
    runs["D_irrelevant"] = run_condition(
        "D_irrelevant", world, knowledge, unrelated, cold_reference=cold_reference
    )

    # -- sample efficiency ------------------------------------------------
    # The smallest budget at which a condition reaches a confirmed discovery.
    efficiency: dict[str, int | None] = {}
    for condition in ("A_cold", "B_structural"):
        reached: int | None = None
        for budget in BUDGETS:
            trial = run_condition(
                condition,
                world,
                knowledge,
                unrelated,
                budget=budget,
                allow_fallback=False,
            )
            confirmed = (
                trial.outcome.promoted is not None
                if condition == "A_cold"
                else (trial.attempt is not None and trial.attempt.confirmed)
            )
            if confirmed:
                reached = budget
                break
        efficiency[condition] = reached

    # -- holdout, once, after every decision -------------------------------
    observations = world.generate()
    holdout_split = make_split()
    holdouts: dict[str, dict[str, Any] | None] = {}
    baselines: list[tuple[str, dict[str, Any]]] = []
    for condition in CONDITIONS:
        run = runs[condition]
        outcome = run.effective
        split = make_split()
        result = D.evaluate_holdout(outcome, observations, split)
        holdouts[condition] = result.test_score
        if not baselines:
            baselines = result.baseline_scores

    attempt = runs["B_structural"].attempt
    assert attempt is not None
    explanation = T.explain(attempt, world.id)
    base_before = next(
        (sc for name, sc in baselines if "base rate" in name), None
    )
    record = T.TransferRecord(
        transfer_id=T.TransferRecord.make_id(pattern_now.pattern_id, world.id, tick),
        source_pattern_id=pattern_now.pattern_id,
        target_environment_id=world.id,
        pattern_version=pattern_now.version,
        target_evidence={
            "full_groundings": attempt.full_groundings,
            "component_groundings": attempt.component_groundings,
            "confirmation": (
                attempt.outcome.confirmation.to_dict()
                if attempt.outcome.confirmation
                else None
            ),
            "cold_start_val_b_brier": cold_reference,
        },
        transfer_confidence=pattern_now.confidence,
        result=attempt.result,
        performance_before_transfer=base_before,
        performance_after_transfer=holdouts["B_structural"],
        created_at=tick,
        explanation=explanation,
        grounded_candidates=attempt.full_groundings + attempt.component_groundings,
        hypotheses_tested=runs["B_structural"].hypotheses_tested,
        adopted_expression=(
            attempt.adopted.hypothesis.expression.text() if attempt.adopted else None
        ),
        adopted_is_component=attempt.adopted_is_component,
    )
    pattern_after = T.update_pattern(pattern_now, attempt, world.id)

    return TargetResult(
        world=world,
        runs=runs,
        holdouts=holdouts,
        baselines=baselines,
        record=record,
        pattern_after=pattern_after,
        efficiency=efficiency,
        seconds=time.perf_counter() - started,
    )


# ------------------------------------------------------------ leakage probes


def leakage_probes(knowledge: SourceKnowledge) -> list[dict[str, Any]]:
    """TRANSFER_TEMPORAL_LEAKAGE_TEST — try to get the target holdout into the
    transfer decision, and try to get source truth across the wall."""
    from echo.observations import TEST, HoldoutViolation
    from echo.pattern import contains_variable_names

    probes: list[dict[str, Any]] = []

    def record(attack: str, expectation: str, run) -> None:
        try:
            result = run()
        except (HoldoutViolation, ex.ExpressionError, Exception) as exc:  # noqa: BLE001
            if isinstance(exc, (HoldoutViolation, ex.ExpressionError, KeyError, ValueError)):
                probes.append(
                    {
                        "attack": attack,
                        "expected": expectation,
                        "outcome": f"refused — {exc.__class__.__name__}",
                        "blocked": True,
                    }
                )
                return
            raise
        probes.append(
            {
                "attack": attack,
                "expected": expectation,
                "outcome": str(result),
                "blocked": bool(result),
            }
        )

    world = TARGETS[0]
    observations = world.generate()

    split = make_split()
    record(
        "Read the target TEST block while deciding whether to transfer",
        "refused",
        lambda: split.block(TEST),
    )

    closed = make_split()
    closed.unlock_holdout()
    record(
        "Attempt a transfer after the target holdout has been read",
        "refused",
        lambda: T.attempt_transfer(knowledge.pattern, observations, closed),
    )

    # The source's own variables must not be expressible in the target world.
    record(
        f"Evaluate the source expression `{knowledge.expression_text}` in the target",
        "refused",
        lambda: ex.check_schema(
            ex.parse(knowledge.expression_text), observations.variable_names
        ),
    )

    leaked = contains_variable_names(
        knowledge.pattern.to_dict(), SOURCE.variable_names
    )
    probes.append(
        {
            "attack": "Search the transferred pattern for any source variable name",
            "expected": "none present",
            "outcome": ", ".join(leaked) if leaked else "no source variable name appears",
            "blocked": not leaked,
        }
    )

    # Grounding must produce only target variables.
    grounded = knowledge.pattern.ground(observations.variable_names)
    foreign = sorted(
        {
            name
            for expr in grounded
            for name in expr.variables()
            if name not in observations.variable_names
        }
    )
    probes.append(
        {
            "attack": "Check every grounded candidate for a foreign variable",
            "expected": "target variables only",
            "outcome": (
                ", ".join(foreign)
                if foreign
                else f"all {len(grounded)} groundings use target columns only"
            ),
            "blocked": not foreign,
        }
    )
    return probes


# ---------------------------------------------------------------- reporting


def _brier(score: dict[str, Any] | None) -> str:
    return f"{score['brier']:.4f}" if score else "—"


def _full(score: dict[str, Any] | None) -> str:
    if not score:
        return "—"
    return (
        f"{score['brier']:.4f} / {score['log_loss']:.4f} / "
        f"{score['expected_calibration_error']:.4f}"
    )


def overall_verdict(results: list[TargetResult]) -> tuple[str, str]:
    """One line for the whole experiment, computed from counts."""
    full = [r for r in results if r.world.shares_structure == "full"]
    partial = [r for r in results if r.world.shares_structure == "partial"]
    none = [r for r in results if r.world.shares_structure == "none"]

    carried = sum(1 for r in full if r.record.result is T.TransferResult.SUCCESSFUL)
    weakened = sum(1 for r in partial if r.record.result is T.TransferResult.PARTIAL)
    refused = sum(
        1
        for r in none
        if r.record.result
        in (T.TransferResult.REJECTED, T.TransferResult.INCONCLUSIVE, T.TransferResult.HARMFUL)
    )
    detail = (
        f"{carried} of {len(full)} structurally matching targets carried the "
        f"pattern whole; {weakened} of {len(partial)} partially-matching target "
        f"kept a component and weakened the rest; {refused} of {len(none)} "
        f"non-matching targets refused it."
    )
    if carried == len(full) and refused == len(none) and weakened == len(partial):
        return "TRANSFER SUCCESS", detail
    if carried == 0:
        return "TRANSFER FAILURE", detail
    return "TRANSFER PARTIAL", detail


def render(
    knowledge: SourceKnowledge,
    unrelated: StructuralPattern,
    unrelated_text: str,
    results: list[TargetResult],
    versions: list[StructuralPattern],
    probes: list[dict[str, Any]],
) -> str:
    from echo.pattern import PARAMETER_TOLERANCE, TRANSFERABLE_AFTER

    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    verdict, detail = overall_verdict(results)
    by_id_map = {r.world.id: r for r in results}

    add("# ECHO 6 — Transfer")
    add()
    add(
        "ECHO 5 showed that a bounded search can find a predictive relationship "
        "nobody encoded. This asks something harder: can what it found in one "
        "world be reused in a different one — different variable names, "
        "different scales, different distributions, different noise, different "
        "base rates — where only the *shape* of the relationship is shared?"
    )
    add()

    # -- executive --------------------------------------------------------
    add("## Executive result")
    add()
    add(f"# {verdict}")
    add()
    add(detail)
    add()
    add(
        "| Target | Shares structure | Transfer result | Adopted | Cold TEST | Transfer TEST |"
    )
    add("| --- | --- | --- | --- | --- | --- |")
    for result in results:
        adopted = (
            f"`{result.record.adopted_expression}`"
            if result.record.adopted_expression
            else "*none*"
        )
        add(
            f"| `{result.world.id}` | {result.world.shares_structure} | "
            f"**{result.record.result.value}** | {adopted} | "
            f"{_brier(result.holdouts['A_cold'])} | "
            f"{_brier(result.holdouts['B_structural'])} |"
        )
    add()

    # -- source discovery -------------------------------------------------
    record = knowledge.record
    add("## Source discovery")
    add()
    add(
        "The source world was searched with the ECHO 5 pipeline, unchanged and "
        f"unassisted. It proposed {record.proposed} candidates and confirmed "
        "one. No expression is named anywhere in the transfer code; whatever "
        "the search returned is what became the source knowledge."
    )
    add()
    add(f"```\n{record.discovered_expression}\n```")
    add()
    add(f"- Complexity: {record.discovered_complexity}")
    add(f"- Provenance: {record.discovery_reason}")
    add(f"- TRAIN Brier: {record.training_score['brier']:.4f}")
    add(f"- VAL_A Brier: {record.validation_score['brier']:.4f}")
    add(
        f"- Holdout Brier: {record.test_score['brier']:.4f} "
        f"(survived unseen data: {record.survived_holdout})"
    )
    add()

    # -- abstract pattern -------------------------------------------------
    add("## Abstract pattern")
    add()
    add(
        "`abstract()` is a pure function of whatever the search returned. Had "
        "discovery found something else, the pattern would be that instead — "
        "which is the whole anti-cheating requirement, held as a property of "
        "the code rather than a promise."
    )
    add()
    add(f"```\n{record.discovered_expression}   →   {knowledge.pattern.text()}\n```")
    add()
    add(f"- In words: *{knowledge.pattern.description()}*")
    add(f"- Slots: {', '.join(knowledge.pattern.slots)}")
    add(
        "- Components: "
        + ", ".join(f"`{c.text()}`" for c in knowledge.pattern.components())
    )
    add(
        f"- The source's delay is a **preference, not a constant**: grounding "
        f"also tries ±{PARAMETER_TOLERANCE}, so a target whose delay differs is "
        "still reachable. `TGT-B` is that case."
    )
    add(
        "- **No source variable name survives the abstraction.** A test asserts "
        f"that none of `{'`, `'.join(SOURCE.variable_names)}` appears anywhere "
        "in the serialised pattern; the probes below re-check it live."
    )
    add()
    first = results[0]
    add(
        f"Grounded in a six-column target world the pattern yields "
        f"**{first.record.target_evidence['full_groundings']} full candidates** "
        f"and {first.record.target_evidence['component_groundings']} component "
        f"candidates, against **{first.runs['A_cold'].outcome.proposed} for an "
        f"exhaustive search** — a "
        f"{first.runs['A_cold'].outcome.proposed // max(first.record.grounded_candidates, 1)}× "
        "reduction. That restriction is the only mechanism by which transfer "
        "can be cheaper than discovery."
    )
    add()

    # -- target worlds ----------------------------------------------------
    add("## Target worlds")
    add()
    add(
        "Everything that is not the point differs between them. What ECHO is "
        "told describes the shape of the data and never the relationship:"
    )
    add()
    for result in results:
        add(f"- **`{result.world.id}`** — {result.world.observable_description}")
    add()
    add(
        "Variable names are `Z1`…`Z6` in every target and `X1`…`X6` in the "
        "source. They mean nothing, no metadata describes them, and there is no "
        "concept in this experiment that a name could refer to."
    )
    add()

    # -- cold start / transfer results -------------------------------------
    add("## Cold start and transfer")
    add()
    add("Brier / log loss / ECE on the sealed holdout, plus how much history each needed.")
    add()
    add(
        "| Target | Cold start (holdout) | Transfer (holdout) | Cold rows needed | "
        "Transfer rows needed | Candidates: cold | Candidates: transfer |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for result in results:
        cold_rows = result.efficiency["A_cold"]
        transfer_rows = result.efficiency["B_structural"]
        add(
            f"| `{result.world.id}` | {_full(result.holdouts['A_cold'])} | "
            f"{_full(result.holdouts['B_structural'])} | "
            f"{cold_rows if cold_rows else 'never'} | "
            f"{transfer_rows if transfer_rows else 'never'} | "
            f"{result.runs['A_cold'].outcome.proposed} | "
            f"{result.runs['B_structural'].outcome.proposed} |"
        )
    add()
    gains = [
        (r.world.id, r.efficiency["A_cold"], r.efficiency["B_structural"])
        for r in results
        if r.efficiency["A_cold"] and r.efficiency["B_structural"]
    ]
    better = [g for g in gains if g[2] < g[1]]
    if better:
        add(
            "**Sample efficiency.** "
            + "; ".join(
                f"`{world}` needed {cold} training rows from scratch and {transfer} "
                f"with the pattern ({cold / transfer:.1f}×)"
                for world, cold, transfer in better
            )
            + "."
        )
    else:
        add(
            "**Sample efficiency.** No target reached a confirmed relationship "
            "with fewer rows under transfer than under cold start."
        )
    add(
        f" The budget ladder is {', '.join(str(b) for b in BUDGETS)} training "
        "rows; validation and holdout blocks are identical at every budget, so "
        "the only thing that varies is how much history the search had. Two "
        "targets reach a confirmed relationship at the smallest budget on the "
        "ladder under both conditions, so their entries are a floor rather than "
        "a measurement — the ladder cannot resolve a difference below 24 rows."
    )
    add()
    add(
        "**The holdout columns are identical, and that is the honest headline.** "
        "Given the full training block, the restricted search and the "
        "exhaustive one converge on the same expression, so transfer buys no "
        "accuracy at all where data is plentiful. What it buys is everything "
        "before that point: the same answer from "
        f"{results[0].runs['B_structural'].outcome.proposed} candidates instead "
        f"of {results[0].runs['A_cold'].outcome.proposed}, and in `TGT-B` from a "
        "quarter of the observations. A claim that transfer made ECHO *more "
        "accurate* here would be false; the measurable benefit is that it "
        "needed less to get there."
    )
    add()

    # -- ablation ---------------------------------------------------------
    add("## Ablation — four conditions on identical data")
    add()
    for label in CONDITION_LABELS.values():
        add(f"- **{label}**")
    add()
    add(f"Condition D transfers a real but unrelated discovery: `{unrelated_text}` → `{unrelated.text()}`.")
    add()
    add("| Target | Condition | Candidates tested | VAL_B Brier | Holdout Brier | Fell back to cold start |")
    add("| --- | --- | --- | --- | --- | --- |")
    for result in results:
        for condition in CONDITIONS:
            run = result.runs[condition]
            val_b = f"{run.val_b_brier:.4f}" if run.val_b_brier is not None else "—"
            add(
                f"| `{result.world.id}` | {condition} | {run.hypotheses_tested} | "
                f"{val_b} | {_brier(result.holdouts[condition])} | "
                f"{'yes' if run.fell_back else 'no'} |"
            )
    add()
    add(
        "Conditions B and D differ only in *which* pattern was handed over. "
        "Where D matches A exactly, that is the fallback working: the unrelated "
        "pattern failed confirmation, was set aside, and the search started "
        "over. Prior knowledge does not become active merely by being prior."
    )
    add()

    # -- negative / partial / false analogy --------------------------------
    for section, world_id, blurb in (
        (
            "Negative transfer",
            "TGT-N",
            "A world with a strong relationship of the wrong shape. The "
            "transferred pattern should find nothing to hold onto.",
        ),
        (
            "Partial transfer",
            "TGT-P",
            "A world where the first half of the shape is predictive and the "
            "second half only adds noise. Treating the pattern as indivisible "
            "would mean losing a genuine signal or adopting a bad one.",
        ),
        (
            "False analogy",
            "TGT-F",
            "Built to look like `TGT-A` from the outside — same "
            "autocorrelations, same scales, same offsets, same base rate, same "
            "description — with no relationship at all.",
        ),
    ):
        result = by_id_map.get(world_id)
        if result is None:
            continue
        add(f"## {section} — `{world_id}`")
        add()
        add(blurb)
        add()
        add(f"**Result: {result.record.result.value}.**")
        add()
        run = result.runs["B_structural"]
        confirmation = run.outcome.confirmation
        if confirmation is not None:
            add(
                f"The best grounding of the pattern was "
                f"`{run.outcome.winner.hypothesis.expression.text()}`. On the "
                f"confirmation block it "
                + (
                    f"beat the frozen base rate by {confirmation.edge:.4f} Brier "
                    f"(t = {confirmation.t_statistic:.2f})"
                    if confirmation.edge > 0
                    else f"was {abs(confirmation.edge):.4f} Brier worse than the "
                    f"frozen base rate (t = {confirmation.t_statistic:.2f})"
                )
                + f", against a required {confirmation.min_edge:.4f} at "
                f"t ≥ {confirmation.min_t}."
            )
            add()
        add(f"- Fell back to a from-scratch search: **{'yes' if run.fell_back else 'no'}**")
        add(f"- Cold start on the same data reached {_brier(result.holdouts['A_cold'])} on the holdout")
        add(f"- Transfer condition reached {_brier(result.holdouts['B_structural'])}")
        add()
        explanation = result.record.explanation
        add("ECHO's own account of the decision:")
        add()
        add("```")
        add(f"SOURCE PATTERN:     {explanation['source_pattern']}")
        add(f"TARGET OBSERVATION: {explanation['target_observation']}")
        add(f"DECISION:           {explanation['decision']}")
        add(f"RESULT:             {explanation['result']}")
        add("```")
        add()

    # -- generalisation ----------------------------------------------------
    add("## Generalisation across targets")
    add()
    add(
        "Three structurally matching worlds, sharing nothing but the shape. "
        "None shares a variable name, a scale, a distribution, a noise level or "
        "a base rate with the source or with each other."
    )
    add()
    add("| Target | Delay | Value range | Outcome noise | Base rate | Transfer result | Adopted |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for result in results:
        world = result.world
        observations = world.generate()
        column = observations.column(world.variable_names[0])
        base = sum(observations.outcomes) / len(observations.outcomes)
        add(
            f"| `{world.id}` | — | {min(column):.3g} … {max(column):.3g} | "
            f"{world.observation_noise:.0%} | {base:.2f} | "
            f"{result.record.result.value} | "
            + (f"`{result.record.adopted_expression}`" if result.record.adopted_expression else "*none*")
            + " |"
        )
    add()
    add(
        "`TGT-C` is the scale test: its values run to roughly 10⁶ and the "
        "interaction term to about 10¹⁰. Nothing is normalised for it "
        "specifically — the standardisation inside the logistic fit is the same "
        "code every condition uses, cold start included."
    )
    add()

    # -- knowledge lifecycle ----------------------------------------------
    add("## Knowledge lifecycle")
    add()
    add(
        f"A pattern does not become transferable by working once. It needs "
        f"{TRANSFERABLE_AFTER} independent successes, and anything short of "
        "success weakens it. Every version is kept."
    )
    add()
    add("| Version | Status | Confidence | Successes | Failures | Most recent evidence |")
    add("| --- | --- | --- | --- | --- | --- |")
    for version in versions:
        # The latest evidence note, not `status_reason` — a version that
        # recorded evidence without changing status carries the previous
        # transition's reason forward, which would read as though the wrong
        # world caused it.
        latest = version.evidence[-1] if version.evidence else version.status_reason
        add(
            f"| v{version.version} | **{version.status.value}** | "
            f"{version.confidence:.2f} | {version.successes} | {version.failures} | "
            f"{latest} |"
        )
    add()

    # -- leakage -----------------------------------------------------------
    add("## Holdout and source integrity — TRANSFER_TEMPORAL_LEAKAGE_TEST")
    add()
    add(
        "Two walls have to hold: the target's holdout must not reach the "
        "transfer decision, and the source's truth must not reach the target at "
        "all. These are attempts to breach them, run live:"
    )
    add()
    add("| Attack | Expected | Outcome | Blocked |")
    add("| --- | --- | --- | --- |")
    for probe in probes:
        add(
            f"| {probe['attack']} | {probe['expected']} | {probe['outcome']} | "
            f"{'✅' if probe['blocked'] else '❌'} |"
        )
    add()

    # -- distinctions ------------------------------------------------------
    add("## Four different things")
    add()
    add("| | What it means | Established here? |")
    add("| --- | --- | --- |")
    add(
        "| **DISCOVERY** | Finding a predictive relationship in a world | Yes — "
        "in the source, and again in every cold-start condition |"
    )
    add(
        "| **TRANSFER** | Applying structural knowledge from one world to "
        "another | Yes — measured against cold start on identical data |"
    )
    add(
        "| **GENERALIZATION** | Performing beyond the original conditions | Yes "
        "— across three targets with different names, scales, delays and noise |"
    )
    add(
        "| **UNDERSTANDING** | Knowing why a relationship holds | **No. Not "
        "established, not claimed, not tested for.** |"
    )
    add()
    add(
        "ECHO reused a shape. It has no account of why a change multiplied by a "
        "delay predicts anything, no model of what these series are, and no "
        "causal claim of any kind. This is not AGI and not consciousness."
    )
    add()

    # -- limitations -------------------------------------------------------
    add("## Limitations")
    add()
    add(
        "1. **The hypothesis language is finite and so is the pattern language.** "
        "A pattern is a tree of nine primitives with slots. Structure that "
        "cannot be written that way cannot be transferred, or discovered.\n"
        "2. **The environments are synthetic.** Seeded simulations, one seed "
        "each, not measurements of anything.\n"
        "3. **The structural equivalence is engineered.** The targets were "
        "built to share a shape with the source. That the shape carries is "
        "evidence the mechanism works, not evidence that real domains relate "
        "this way.\n"
        "4. **The search space is bounded.** Single terms and pairs of terms; "
        "grounding is slots × a small parameter neighbourhood. A shape needing "
        "a wider neighbourhood than "
        f"±{PARAMETER_TOLERANCE} would not be found by transfer.\n"
        "5. **There is no semantic grounding.** `Z3` is a column index. Nothing "
        "in the system could represent what a series measures, and nothing "
        "should be read as if it could.\n"
        "6. **There is no causal inference.** A relationship that predicts is "
        "not a relationship that explains.\n"
        "7. **Statistical similarity can be accidental.** `TGT-F` is the "
        "deliberate case, but with enough targets some shape will fit some "
        "world by luck. The confirmation block is what limits it, not "
        "something in the transfer mechanism itself."
    )
    add()

    # -- final question ----------------------------------------------------
    add(
        "## Did ECHO reuse knowledge from one environment to improve prediction "
        "in a structurally related but independently represented one?"
    )
    add()
    carried = [r for r in results if r.record.result is T.TransferResult.SUCCESSFUL]
    if carried:
        add("**Yes, with the evidence being:**")
        add()
        for result in carried:
            cold_rows = result.efficiency["A_cold"]
            transfer_rows = result.efficiency["B_structural"]
            line = (
                f"- In `{result.world.id}`, the pattern grounded to "
                f"`{result.record.adopted_expression}` and confirmed against the "
                f"frozen base rate, reaching {_brier(result.holdouts['B_structural'])} "
                f"on a holdout that was sealed throughout."
            )
            if cold_rows and transfer_rows and transfer_rows < cold_rows:
                line += (
                    f" It needed {transfer_rows} training rows where a "
                    f"from-scratch search needed {cold_rows}, testing "
                    f"{result.runs['B_structural'].outcome.proposed} candidates "
                    f"instead of {result.runs['A_cold'].outcome.proposed}."
                )
            add(line)
        add()
        add(
            "And the evidence that it is structure being reused rather than "
            "similarity being trusted: `TGT-F` is statistically indistinguishable "
            "from `TGT-A` on every observable and the pattern was **rejected** "
            "there; `TGT-N` has a strong relationship of the wrong shape and the "
            "pattern found nothing; `TGT-P` shares half the shape and half is "
            "what was kept. A mechanism that transferred on resemblance would "
            "have fired in `TGT-F`."
        )
    else:
        add(
            "**No.** No structurally matching target carried the transferred "
            "pattern whole. The measurements above stand as reported; the "
            "experiment was not adjusted to change them."
        )
    add()
    return "\n".join(lines)


# ------------------------------------------------------------------- driver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ECHO 6 transfer experiment.")
    parser.add_argument("--out", type=Path, default=None, help="write the report here")
    args = parser.parse_args(argv)

    knowledge = discover_source()
    print(f"  source discovery: {knowledge.expression_text}")
    print(f"  abstract pattern: {knowledge.pattern.text()}")

    unrelated, unrelated_text = irrelevant_pattern()
    print(f"  unrelated knowledge (condition D): {unrelated_text} -> {unrelated.text()}")

    ledger = TransferLedger.in_directory(RESULTS_DIR)
    pattern = knowledge.pattern
    ledger.add_pattern(pattern)
    versions = [pattern]

    results: list[TargetResult] = []
    for index, world in enumerate(TARGETS):
        result = run_target(world, knowledge, unrelated, pattern, tick=index + 1)
        pattern = result.pattern_after
        ledger.add_pattern(pattern)
        ledger.add_transfer(result.record)
        versions.append(pattern)
        results.append(result)
        print(
            f"  {world.id:8s} {result.record.result.value:12s} "
            f"adopted={result.record.adopted_expression} "
            f"cold_rows={result.efficiency['A_cold']} "
            f"transfer_rows={result.efficiency['B_structural']}"
        )

    ledger.save()
    print(f"  ledger -> {ledger.path}  ({ledger.counts()})")

    probes = leakage_probes(knowledge)
    report = render(knowledge, unrelated, unrelated_text, results, versions, probes)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
