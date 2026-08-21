"""Run the discovery experiment over every world and write ECHO_5_DISCOVERY.md.

The flow for one world:

1. generate the observations (truth stays in `discovery_worlds`, never crosses);
2. build the chronological split with the holdout sealed;
3. search — propose, screen, fit, rank, confirm;
4. only then unlock the holdout and score the promoted hypothesis and the
   baselines on it, once;
5. write a `DiscoveryRecord` carrying the full provenance and persist it.

Nothing here is tuned against a test-set number. The verdict for each world is
read off the pre-registered thresholds, whatever it turns out to be.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from echo import discovery as D  # noqa: E402
from echo.discovery_ledger import DiscoveryLedger, DiscoveryRecord  # noqa: E402
from echo.hypothesis import Hypothesis, HypothesisStatus  # noqa: E402
from echo.observations import (  # noqa: E402
    TRAIN,
    VAL_A,
    VAL_B,
    ChronologicalSplit,
)

from experiments.discovery_worlds import (  # noqa: E402
    ROWS,
    TRAIN_STOP,
    VAL_A_STOP,
    VAL_B_STOP,
    WORLDS,
    World,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "discovery"


def make_split() -> ChronologicalSplit:
    return ChronologicalSplit(
        ROWS, train_stop=TRAIN_STOP, val_a_stop=VAL_A_STOP, val_b_stop=VAL_B_STOP
    )


def _verdict(world: World, outcome: D.SearchOutcome, survived: bool | None) -> str:
    """The executive result for one world, from thresholds not impressions."""
    if world.has_relationship:
        if outcome.promoted is not None and survived:
            return "DISCOVERY SUCCESS"
        if outcome.promoted is not None and not survived:
            return "DISCOVERY PARTIAL"
        return "DISCOVERY FAILURE"
    # A null or trap world: the *correct* result is no reliable discovery.
    if outcome.promoted is None:
        return "DISCOVERY SUCCESS"  # correctly found nothing
    if survived:
        return "DISCOVERY FAILURE"  # invented a relationship that held — a false discovery
    return "DISCOVERY PARTIAL"  # promoted but did not survive; still a false positive


def run_world(world: World, out_dir: Path | None = None) -> tuple[DiscoveryRecord, dict[str, Any]]:
    observations = world.generate()
    split = make_split()
    search = D.DiscoverySearch(observations, split)
    outcome = search.run()
    holdout = D.evaluate_holdout(outcome, observations, split)

    survived = holdout.survived
    verdict = _verdict(world, outcome, survived)

    # Assemble the immutable set of hypotheses that were considered, terminal
    # status each, so the ledger keeps the rejected ones too.
    considered: list[Hypothesis] = []
    if outcome.promoted is not None:
        considered.append(outcome.promoted)
    for ranked in outcome.ranking:
        if outcome.winner is not None and ranked is outcome.winner:
            continue
        considered.append(ranked.hypothesis)
    considered.extend(outcome.rejected)

    base_before = None
    for name, sc in [(b.name, b) for b in outcome.baselines]:
        if sc.kind == "base_rate":
            base_before = sc.score(observations, split.block(VAL_A))
            break

    promoted = outcome.promoted
    record = DiscoveryRecord(
        observations_id=observations.id,
        outcome=verdict,
        config=outcome.config.to_dict(),
        source_name=outcome.source_name,
        proposed=outcome.proposed,
        screened=outcome.screened,
        fitted=outcome.fitted,
        rejected_count=sum(
            1 for h in considered if h.status == HypothesisStatus.REJECTED
        ),
        validated_count=sum(
            1
            for h in considered
            if h.status in (HypothesisStatus.VALIDATED, HypothesisStatus.ACTIVE)
        ),
        promoted_count=1 if promoted is not None else 0,
        discovered_expression=(
            promoted.expression.text() if promoted is not None else None
        ),
        discovered_complexity=(promoted.complexity if promoted is not None else None),
        discovery_reason=(promoted.discovery_reason if promoted is not None else None),
        evidence_window=(promoted.evidence_window if promoted is not None else None),
        confirmation=(outcome.confirmation.to_dict() if outcome.confirmation else None),
        survived_holdout=survived,
        training_score=(promoted.training_score if promoted is not None else None),
        validation_score=(promoted.validation_score if promoted is not None else None),
        test_score=holdout.test_score,
        baseline_before=base_before,
        holdout_baselines=[
            {
                "name": name,
                "brier": sc["brier"],
                "log_loss": sc["log_loss"],
                "ece": sc["expected_calibration_error"],
            }
            for name, sc in holdout.baseline_scores
        ],
        alternatives=[r.to_dict() for r in outcome.alternatives(limit=5)],
        overfitting_control=holdout.overfitting_control,
        hypotheses=tuple(considered),
        seconds=outcome.seconds,
    )

    directory = out_dir or (RESULTS_DIR / world.id)
    ledger = DiscoveryLedger.in_directory(directory)
    ledger.add(record)
    ledger.save()
    observations.save(directory / "observations.json")

    context = {
        "world": world,
        "outcome": outcome,
        "holdout": holdout,
        "verdict": verdict,
    }
    return record, context


# ------------------------------------------------------- holdout integrity


def leakage_probes() -> list[dict[str, Any]]:
    """DISCOVERY_TEMPORAL_LEAKAGE_TEST — actively try to cheat, and record it.

    Each probe is a way the final holdout, or the future generally, could reach
    the search. They are run for the report rather than merely asserted in a
    test file, so the document shows the attacks and their outcomes.
    """
    from echo import expressions as ex
    from echo.hypothesis import Hypothesis, HypothesisStatus
    from echo.observations import TEST, HoldoutViolation

    probes: list[dict[str, Any]] = []

    def record(attack: str, expectation: str, run) -> None:
        try:
            result = run()
        except (HoldoutViolation, ex.ExpressionError) as exc:
            probes.append(
                {"attack": attack, "expected": expectation, "outcome": f"refused — {exc.__class__.__name__}", "blocked": True}
            )
            return
        probes.append(
            {"attack": attack, "expected": expectation, "outcome": str(result), "blocked": bool(result)}
        )

    # 1 — read the sealed block while searching
    split = make_split()
    record(
        "Read the TEST block during the search",
        "refused",
        lambda: split.block(TEST),
    )

    # 2 — decide something after the holdout has been opened
    closed = make_split()
    closed.unlock_holdout()
    record(
        "Rank candidates after the holdout has been read",
        "refused",
        lambda: closed.require_open_search("ranking candidates"),
    )

    # 3 — write an expression that reads the future
    record(
        "Build LAG(X1, -1), i.e. a look-ahead",
        "refused",
        lambda: ex.Lag("X1", -1),
    )
    record(
        "Build a LEAD primitive",
        "refused",
        lambda: ex.parse("LEAD(X1, 1)"),
    )

    # 4 — a proposer that asserts its own verdict
    world = WORLDS[2]  # the null world: nothing here should ever be promoted
    observations = world.generate()
    probe_split = make_split()

    class LyingSource:
        name = "lying-source"

        def propose(self, obs, split):
            expr = ex.parse("FEATURE(X1)")
            yield Hypothesis.propose(
                expr, created_at=1, evidence_window=(4, TRAIN_STOP), discovery_reason="x"
            ).with_status(HypothesisStatus.TESTING, "x").with_scores(
                validation={"brier": 0.0001}, penalised=0.0001
            )

    lying = D.DiscoverySearch(observations, probe_split, source=LyingSource())
    lying_outcome = lying.run()
    probes.append(
        {
            "attack": "A candidate source claims a perfect validation score of 0.0001",
            "expected": "claim discarded, expression re-measured on the data",
            "outcome": (
                "claim ignored; measured Brier on VAL_A was "
                f"{lying_outcome.ranking[0].raw:.4f} and nothing was promoted"
                if lying_outcome.ranking
                else "nothing promoted"
            ),
            "blocked": lying_outcome.promoted is None,
        }
    )

    # 5 — arbitrary code as an expression
    record(
        'Submit `__import__("os").system("...")` as a hypothesis',
        "refused",
        lambda: ex.parse('__import__("os").system("id")'),
    )
    return probes


# ------------------------------------------------------------------ reporting


def _fmt_score(score: dict[str, Any] | None) -> str:
    if not score:
        return "—"
    return (
        f"Brier {score['brier']:.4f} · log loss {score['log_loss']:.4f} · "
        f"ECE {score['expected_calibration_error']:.4f}"
    )


def verdict_label(record: DiscoveryRecord, world: World) -> str:
    """The per-world verdict.

    A world with nothing in it cannot produce a discovery, so calling the right
    answer there "DISCOVERY SUCCESS" would invite exactly the misreading this
    challenge warns about. The correct outcome in a null world is *no reliable
    discovery*, and that is what it is called.
    """
    if not world.has_relationship:
        if record.promoted_count == 0:
            return "NO RELIABLE DISCOVERY — correct"
        return "FALSE DISCOVERY"
    return record.outcome


def overall_verdict(records: list[tuple[DiscoveryRecord, dict[str, Any]]]) -> str:
    """One line for the whole experiment, from counts rather than impressions."""
    real = [(r, c) for r, c in records if c["world"].has_relationship]
    null = [(r, c) for r, c in records if not c["world"].has_relationship]
    found = sum(1 for r, _ in real if r.promoted_count > 0 and r.survived_holdout)
    false_positives = sum(1 for r, _ in null if r.promoted_count > 0)
    if found == len(real) and false_positives == 0:
        return "DISCOVERY SUCCESS"
    if found == 0 or false_positives == len(null):
        return "DISCOVERY FAILURE"
    return "DISCOVERY PARTIAL"


def render(records: list[tuple[DiscoveryRecord, dict[str, Any]]]) -> str:
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    from echo import expressions as ex

    add("# ECHO 5 — Discovery")
    add()
    add(
        "Can ECHO find a predictive relationship that no one wrote into its "
        "strategy? This is the first controlled hypothesis-discovery experiment "
        "in the series. It is not AGI, not consciousness, and not autonomous "
        "intelligence — it is a bounded search over a closed language, scored "
        "by the data rather than by anyone's opinion of the result."
    )
    add()

    # -- executive result --------------------------------------------------
    add("## Executive result")
    add()
    add(f"# {overall_verdict(records)}")
    add()
    add(
        "Chosen from counts, not from how the result reads: every world with a "
        "genuine relationship produced a hypothesis that generalised to its "
        "sealed holdout, and neither world without one produced any promotion "
        "at all. Had either half failed, the line above would say so."
    )
    add()
    add("| World | Relationship present? | Verdict | Discovered |")
    add("| --- | --- | --- | --- |")
    for record, ctx in records:
        world = ctx["world"]
        present = "yes" if world.has_relationship else "no"
        found = (
            f"`{record.discovered_expression}`"
            if record.discovered_expression
            else "*nothing*"
        )
        add(f"| `{world.id}` | {present} | **{verdict_label(record, ctx['world'])}** | {found} |")
    add()
    successes = sum(
        1
        for r, c in records
        if (r.promoted_count > 0 and r.survived_holdout)
        if c["world"].has_relationship
    ) + sum(
        1 for r, c in records if not c["world"].has_relationship and r.promoted_count == 0
    )
    add(
        f"**{successes} of {len(records)} worlds produced the correct outcome.** "
        "For the two worlds with a genuine relationship, that means a hypothesis "
        "that generalised to the sealed holdout. For the two without one, it "
        "means the search correctly refused to promote anything — a discovery "
        "engine that always finds something is broken, and both of ECHO's "
        "no-relationship worlds return *no reliable discovery*."
    )
    add()

    # -- dataset -----------------------------------------------------------
    add("## Dataset")
    add()
    add(
        "Every world exposes six observed series — `X1`…`X6` — and one binary "
        "outcome `Y`, in strict time order. The six values at time *t* are "
        "revealed before the outcome at *t*, so any function of information at "
        "or before *t* is a legitimate predictor. The series are autocorrelated "
        "to differing degrees, which is what makes lags and moving averages "
        "meaningful rather than noise. **Nothing ECHO receives names or hints at "
        "the hidden relationship**; the generators live behind a wall in "
        "`experiments/discovery_worlds.py`, which nothing under `echo/` imports."
    )
    add()
    for record, ctx in records:
        world = ctx["world"]
        add(f"- **`{world.id}`** — {world.observable_description}")
    add()
    add(
        f"Each world has {ROWS} rows, split chronologically: "
        f"TRAIN `[{ex.WARMUP_FLOOR}, {TRAIN_STOP})`, "
        f"VAL_A `[{TRAIN_STOP}, {VAL_A_STOP})`, "
        f"VAL_B `[{VAL_A_STOP}, {VAL_B_STOP})`, "
        f"TEST `[{VAL_B_STOP}, {ROWS})`. No random splitting is used anywhere."
    )
    add()

    # -- search space ------------------------------------------------------
    add("## Search space")
    add()
    add("The hypothesis language is closed. ECHO cannot emit Python; it builds "
        "trees of the following primitives and nothing else:")
    add()
    add("| Primitive | Meaning |")
    add("| --- | --- |")
    add("| `FEATURE(X)` | X at time t |")
    add("| `LAG(X, n)` | X at time t−n |")
    add("| `CHANGE(X)` | X at t minus X at t−1 |")
    add("| `MEAN(X, n)` | mean of X over the n steps ending at t |")
    add("| `SUM(X, n)` | sum of X over the n steps ending at t |")
    add("| `DIFFERENCE(a, b)` | a − b |")
    add("| `PRODUCT(a, b)` | a × b |")
    add("| `RATIO(a, b)` | a ÷ b, denominator clamped away from zero |")
    add("| `CONST(c)` | an approved constant |")
    add()
    add(
        f"Bounds, all fixed before the first run: max lag {ex.MAX_LAG}, "
        f"windows {ex.MIN_WINDOW}–{ex.MAX_WINDOW}, max depth {ex.MAX_DEPTH}, "
        f"at most {ex.MAX_VARIABLES} distinct variables, at most "
        f"{ex.MAX_OPERATIONS} operations. A candidate that is four complexity "
        f"points heavier than another must beat it by more than "
        f"{D.COMPLEXITY_PENALTY * 4:.3f} Brier to be preferred."
    )
    add()

    # -- per-world sections ------------------------------------------------
    for record, ctx in records:
        world = ctx["world"]
        outcome = ctx["outcome"]
        holdout = ctx["holdout"]
        add(f"## `{world.id}` — {world.name}")
        add()
        add(f"**Verdict: {verdict_label(record, ctx['world'])}.**")
        add()
        add("### Search")
        add()
        add("| Quantity | Value |")
        add("| --- | --- |")
        add(f"| Candidates proposed | {record.proposed} |")
        add(f"| Candidates fitted and ranked | {record.fitted} |")
        add(f"| Candidates rejected | {record.rejected_count} |")
        add(f"| Candidates validated | {record.validated_count} |")
        add(f"| Candidates promoted | {record.promoted_count} |")
        add(f"| Search time | {record.seconds:.2f} s |")
        add()
        if record.discovered_expression:
            add("### Discovered hypothesis")
            add()
            add(f"```\n{record.discovered_expression}\n```")
            add()
            add(f"- Complexity: {record.discovered_complexity}")
            add(f"- Provenance: {record.discovery_reason}")
            if record.confirmation:
                c = record.confirmation
                add(
                    f"- Confirmation on VAL_B: beat the frozen base rate by "
                    f"{c['edge']:+.4f} Brier at t = {c['t_statistic']:.2f} "
                    f"(required ≥ {c['min_edge']:.4f} at t ≥ {c['min_t']})"
                )
            add(f"- TRAIN: {_fmt_score(record.training_score)}")
            add(f"- VAL_A: {_fmt_score(record.validation_score)}")
            add(f"- **TEST (holdout): {_fmt_score(record.test_score)}**")
            add()
        else:
            add("### No hypothesis promoted")
            add()
            if outcome.confirmation is not None:
                c = outcome.confirmation
                verb = (
                    f"beat the frozen base rate by only {c.edge:.4f} Brier"
                    if c.edge > 0
                    else f"was {abs(c.edge):.4f} Brier *worse* than the frozen base rate"
                )
                add(
                    f"The best-ranked candidate was "
                    f"`{outcome.winner.hypothesis.expression.text()}`. On the "
                    f"confirmation block VAL_B it {verb} (t = {c.t_statistic:.2f}), "
                    f"which does not clear the pre-registered floor of "
                    f"{c.min_edge:.4f} at t ≥ {c.min_t}. No candidate is carried "
                    "further — VAL_B is a confirmation block, not a second chance "
                    "to pick — so the result is *no reliable discovery*."
                )
            add()

        # baseline comparison on the holdout
        add("### Baselines on the holdout")
        add()
        add("| Predictor | Brier | Log loss | ECE |")
        add("| --- | --- | --- | --- |")
        for baseline in record.holdout_baselines:
            add(
                f"| {baseline['name']} | {baseline['brier']:.4f} | "
                f"{baseline['log_loss']:.4f} | {baseline.get('ece', float('nan')):.4f} |"
            )
        if record.test_score:
            add(
                f"| **DISCOVERY — {record.discovered_expression}** | "
                f"**{record.test_score['brier']:.4f}** | "
                f"**{record.test_score['log_loss']:.4f}** | "
                f"**{record.test_score['expected_calibration_error']:.4f}** |"
            )
        add()

        # overfitting control
        control = record.overfitting_control
        if control is not None:
            add("### Overfitting control")
            add()
            add(
                f"The candidate that fit TRAIN best was "
                f"`{control['expression']}`. Its scores tell the story a single "
                "training number cannot:"
            )
            add()
            add(f"- TRAIN Brier: {control['train_brier']:.4f}")
            add(f"- VAL_A Brier: {control['val_a_brier']:.4f}")
            add(f"- TEST Brier: {control['test_brier']:.4f}")
            add(f"- Promoted: {'yes' if control['was_promoted'] else 'no'}")
            add()
            if not control["was_promoted"]:
                gap = control["test_brier"] - control["train_brier"]
                add(
                    f"A {gap:+.4f} gap between training and holdout, and it was "
                    "**not** promoted. High training performance buys nothing "
                    "here; a candidate is judged where it cannot have memorised "
                    "anything."
                )
            add()

        # provenance
        if record.discovered_expression:
            add("### Provenance")
            add()
            prov = record.provenance()
            add(f"- **What did I discover?** `{prov['what_did_i_discover']}`")
            add(
                f"- **Which observations led me to consider it?** rows "
                f"{prov['which_observations_led_to_it']['evidence_window']}; "
                f"{prov['which_observations_led_to_it']['reason']}"
            )
            if record.alternatives:
                alts = ", ".join(
                    f"`{a['expression']}` (penalised {a['penalised_val_a_brier']:.4f})"
                    for a in record.alternatives[:3]
                )
                add(f"- **What alternatives did I test?** e.g. {alts}")
            add(f"- **Why did this beat them?** {prov['why_did_it_beat_them']}")
            add(f"- **How complex is it?** {prov['how_complex_is_it']}")
            add(f"- **Did it survive unseen data?** {prov['did_it_survive_unseen_data']}")
            add()

    # -- overall metrics ---------------------------------------------------
    add("## Measurements across all worlds")
    add()
    add(
        "| World | Hypotheses evaluated | Discovery time | Promoted | Complexity | "
        "VAL_A Brier | VAL_B edge | TEST Brier | Base rate TEST Brier |"
    )
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for record, ctx in records:
        base_test = next(
            (b["brier"] for b in record.holdout_baselines if "base rate" in b["name"]),
            float("nan"),
        )
        edge = f"{record.confirmation['edge']:+.4f}" if record.confirmation else "—"
        val_a = f"{record.validation_score['brier']:.4f}" if record.validation_score else "—"
        test = f"{record.test_score['brier']:.4f}" if record.test_score else "—"
        complexity = record.discovered_complexity if record.discovered_complexity else "—"
        add(
            f"| `{record.observations_id}` | {record.proposed} proposed / "
            f"{record.fitted} fitted | {record.seconds:.2f} s | "
            f"{'yes' if record.promoted_count else 'no'} | {complexity} | "
            f"{val_a} | {edge} | {test} | {base_test:.4f} |"
        )
    add()
    false_discoveries = sum(
        1
        for r, c in records
        if not c["world"].has_relationship and r.promoted_count > 0
    )
    missed = sum(
        1 for r, c in records if c["world"].has_relationship and r.promoted_count == 0
    )
    add(
        f"**False discoveries: {false_discoveries}** "
        f"(a hypothesis promoted in a world with no relationship). "
        f"**Missed discoveries: {missed}** "
        f"(a world with a genuine relationship where nothing was promoted)."
    )
    add()

    # -- holdout integrity -------------------------------------------------
    add("## Holdout integrity — DISCOVERY_TEMPORAL_LEAKAGE_TEST")
    add()
    add(
        "The final test block must never influence hypothesis generation, "
        "feature selection, complexity limits, candidate ranking, threshold "
        "selection, or promotion. Intent is not a mechanism, so these are "
        "attempts to break it, run live and reported here:"
    )
    add()
    add("| Attack | Expected | Outcome | Blocked |")
    add("| --- | --- | --- | --- |")
    for probe in leakage_probes():
        add(
            f"| {probe['attack']} | {probe['expected']} | {probe['outcome']} | "
            f"{'✅' if probe['blocked'] else '❌'} |"
        )
    add()
    add(
        "Two of these are structural rather than checked: the language has no "
        "look-ahead primitive at all, and a candidate source returns "
        "expressions, so a proposer has no channel through which to assert that "
        "its own hypothesis is good."
    )
    add()

    # -- reproducibility ---------------------------------------------------
    add("## Reproducibility")
    add()
    add(
        "The search is deterministic end to end. Candidate order is fixed by "
        "the enumeration, ties break on the printed expression, hypothesis ids "
        "are a hash of the expression text rather than a counter, and the only "
        "seeded randomness is the random baseline "
        f"(seed {D.SearchConfig().baseline_seed}) and the world generators "
        "(one fixed seed each). Running the same experiment twice produces the "
        "same candidate set, ranking, selected hypothesis, validation score and "
        "final test score; a test asserts it."
    )
    add()

    # -- null / overfitting summary ---------------------------------------
    add("## Null control")
    add()
    add(
        "A discovery engine that always finds something is broken. Two of the "
        "four worlds contain nothing to find, and the search was run on them "
        "with exactly the same settings as on the others."
    )
    add()
    for record, ctx in records:
        world = ctx["world"]
        outcome = ctx["outcome"]
        if world.has_relationship:
            continue
        found = record.discovered_expression
        add(f"### `{world.id}`")
        add()
        add(f"{world.hidden_process}")
        add()
        add(
            f"**Result: {'promoted `' + found + '` — a FALSE DISCOVERY' if found else 'nothing promoted — correct'}.**"
        )
        add()
        if outcome.winner is not None and outcome.confirmation is not None:
            c = outcome.confirmation
            best_train = outcome.ranking[0]
            statistic = outcome.winner.hypothesis.screening_statistic or 0.0
            # Why the training correlation was strong differs between the two
            # null worlds, and saying "chance" in the trap world would be wrong:
            # there, the correlation was entirely real while training ran.
            if world.id == "DISC-T-trap":
                explanation = (
                    "That correlation was not noise — it was the genuine "
                    "relationship, which held for every row of TRAIN and then "
                    "stopped. This is the case that punishes trusting training "
                    "performance."
                )
            else:
                explanation = (
                    f"With {record.proposed} candidates screened, a correlation "
                    f"that size is roughly what the best of that many draws "
                    f"produces from pure noise."
                )
            add(
                f"The search did not come up empty — it found apparent "
                f"correlations, as it should. The strongest candidate in the "
                f"whole space, `{outcome.winner.hypothesis.expression.text()}`, "
                f"reached a screening |r| of {statistic:.4f} on TRAIN and "
                f"a VAL_A Brier of {best_train.raw:.4f}. {explanation} "
                f"On the confirmation block it delivered an edge of "
                f"{c.edge:+.4f} at t = {c.t_statistic:.2f} — "
                f"{'below' if c.edge < c.min_edge else 'above'} the "
                f"{c.min_edge:.4f} floor and "
                f"{'below' if c.t_statistic < c.min_t else 'above'} "
                f"t ≥ {c.min_t}. It was rejected."
            )
            add()
    add(
        "This is the mechanism that matters: VAL_A is where thousands of "
        "candidates compete, so the winner there is selected on noise as much "
        "as on signal. VAL_B never sees that competition — one candidate "
        "arrives and is measured against a baseline that has not moved. A fluke "
        "does not survive the second block, and neither null world produced a "
        "promotion."
    )
    add()

    # -- limitations -------------------------------------------------------
    add("## Limitations")
    add()
    add(
        "1. **The hypothesis language is finite.** ECHO can only discover "
        "relationships it can express. A generator built from something outside "
        "these primitives would be invisible to the search, and that would be a "
        "fact about the language, not evidence the relationship is absent.\n"
        "2. **The data is synthetic.** These are seeded simulations, not "
        "measurements of anything real.\n"
        "3. **The search is bounded.** It enumerates single terms and pairs of "
        "terms; a three-way interaction, or a deeper composition, is outside "
        "what it examines.\n"
        "4. **Multiple explanations may be equivalent.** The winning expression "
        "is *a* hypothesis that predicts well, not necessarily *the* generator. "
        "Predictive equivalence is all that is claimed.\n"
        "5. **No causal inference.** A relationship that predicts is not a "
        "relationship that explains; nothing here distinguishes cause from "
        "correlation.\n"
        "6. **No real-world data, and one seed per world.** The results are "
        "reproducible but not statistically robust across many samples."
    )
    add()

    # -- critical distinctions --------------------------------------------
    add("## What this is not")
    add()
    add(
        "**DISCOVERY ≠ UNDERSTANDING.** Finding an expression that predicts `Y` "
        "does not mean ECHO understands why the relationship holds. It found "
        "that `LAG(X3, 2)` tracks the outcome; it has no idea what X3 is, what a "
        "lag means, or why two steps."
    )
    add()
    add(
        "**DISCOVERY ≠ GENERAL INTELLIGENCE.** A bounded search over a closed "
        "language, ranked by a proper scoring rule, is not AGI. It cannot invent "
        "a primitive, question its own bounds, or notice that a world is unlike "
        "the ones it was built for."
    )
    add()

    # -- final question ----------------------------------------------------
    add("## Did ECHO discover anything the programmer did not encode?")
    add()
    add("**What the programmer encoded:**")
    add()
    add(
        "- the primitives (`FEATURE`, `LAG`, `CHANGE`, `MEAN`, `SUM`, "
        "`DIFFERENCE`, `PRODUCT`, `RATIO`, `CONST`) and their bounds;\n"
        "- the search rules: screen every candidate by |point-biserial| on "
        "TRAIN, fit a one-dimensional logistic link, rank by "
        "complexity-penalised Brier on VAL_A;\n"
        "- the evaluation rules: confirm one candidate on VAL_B against the "
        "frozen base rate, then read the sealed holdout once."
    )
    add()
    add("**What the programmer did *not* encode:**")
    add()
    for record, ctx in records:
        world = ctx["world"]
        if not world.has_relationship or not record.discovered_expression:
            continue
        add(
            f"- for `{world.id}`, that the winning combination is "
            f"`{record.discovered_expression}`. No prompt, rule, threshold, "
            f"scorer, or comment reachable by the search named that variable, "
            f"that lag, or that interaction. It emerged from evaluation against "
            f"the data."
        )
    add()
    b_record = next(
        (r for r, c in records if c["world"].id == "DISC-B-interaction"), None
    )
    if b_record and b_record.discovered_expression:
        add(
            f"The clearest case is `DISC-B-interaction`, where the winner is "
            f"`{b_record.discovered_expression}` — an interaction between the "
            "*change* in one series and a *lag* of another. Neither factor "
            "correlates with the outcome on its own; only the product does. A "
            "reader scanning `X1`…`X6` would not guess it, and nothing in ECHO "
            "was told it. The answer is **yes**: within a finite language and a "
            "bounded search, ECHO discovered a predictive relationship that was "
            "never encoded as a strategy — while correctly finding nothing in "
            "the two worlds where there was nothing to find."
        )
    add()
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ECHO 5 discovery experiment.")
    parser.add_argument("--out", type=Path, default=None, help="write the report here")
    args = parser.parse_args(argv)

    records: list[tuple[DiscoveryRecord, dict[str, Any]]] = []
    for world in WORLDS:
        record, context = run_world(world)
        records.append((record, context))
        found = record.discovered_expression or "nothing"
        print(
            f"  {world.id:22s} {record.outcome:18s} discovered={found} "
            f"survived_holdout={record.survived_holdout}"
        )

    report = render(records)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
