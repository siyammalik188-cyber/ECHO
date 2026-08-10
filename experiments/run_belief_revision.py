"""Run the belief-revision experiment and write the report.

Fully offline. No model, no API key, no network — belief revision is arithmetic
over evidence weights, so there is nothing to generate.

    python -m experiments.run_belief_revision
    python -m experiments.run_belief_revision --out docs/ECHO_2_BELIEF_REVISION.md

Each scenario: create the belief at 0.5, feed the opening evidence to form a
hypothesis, then feed the later evidence. Ground truth is consulted only by the
scorer, after the fact.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from echo.belief import (
    EVIDENCE_SCALE,
    MIN_RELEVANCE,
    Belief,
    EvidenceStance,
    logit,
)
from echo.belief_store import BeliefStore

from .world import SCENARIOS, Scenario

HERE = Path(__file__).parent
RESULTS_DIR = HERE / "results"

# --- thresholds used to classify behaviour, fixed before running -------------
DECISION_THRESHOLD = 0.5  # above this, ECHO holds the proposition true
WEAK_WEIGHT = 0.25  # evidence at or below this is "weak"
STRONG_WEIGHT = 0.60  # evidence at or above this is "strong"
OVERREACTION_DELTA = 0.15  # a weak item moving confidence more than this
RESISTANCE_DELTA = 0.10  # a strong item moving confidence less than this
MATERIAL_DELTA = 0.10  # what counts as having "updated" at all


@dataclass
class ScenarioResult:
    scenario: Scenario
    belief: Belief
    initial_confidence: float = 0.0  # after opening evidence, before later evidence
    findings: list[str] = field(default_factory=list)

    @property
    def final_confidence(self) -> float:
        return self.belief.confidence

    @property
    def believes_proposition(self) -> bool:
        return self.final_confidence > DECISION_THRESHOLD

    @property
    def initially_believed(self) -> bool:
        return self.initial_confidence > DECISION_THRESHOLD

    @property
    def correct(self) -> bool:
        """Did ECHO end up on the right side of the decision threshold?"""
        return self.believes_proposition == self.scenario.ground_truth

    @property
    def moved_toward_truth(self) -> bool:
        target = 1.0 if self.scenario.ground_truth else 0.0
        return abs(self.final_confidence - target) < abs(self.initial_confidence - target)

    @property
    def later_delta(self) -> float:
        return self.final_confidence - self.initial_confidence

    @property
    def failed_to_update(self) -> bool:
        """Truth contradicted the hypothesis, strong evidence arrived, nothing moved."""
        if self.scenario.ground_truth == self.initially_believed:
            return False  # nothing needed overturning
        had_strong = any(
            r.applied and r.evidence_snapshot["weight"] >= STRONG_WEIGHT
            for r in self._later_revisions()
        )
        return had_strong and abs(self.later_delta) < MATERIAL_DELTA

    def _later_revisions(self):
        opening = {e.id for e in self.scenario.initial_evidence}
        return [r for r in self.belief.revision_history if r.evidence_id not in opening]

    @property
    def overreactions(self) -> list[dict[str, Any]]:
        out = []
        for r in self.belief.revision_history:
            weight = r.evidence_snapshot["weight"]
            if r.applied and weight <= WEAK_WEIGHT and abs(r.delta) > OVERREACTION_DELTA:
                out.append(
                    {
                        "evidence_id": r.evidence_id,
                        "weight": weight,
                        "delta": round(r.delta, 4),
                    }
                )
        return out

    @property
    def resistances(self) -> list[dict[str, Any]]:
        out = []
        for r in self.belief.revision_history:
            snapshot = r.evidence_snapshot
            if (
                r.applied
                and snapshot["stance"] == EvidenceStance.CONTRADICTS.value
                and snapshot["weight"] >= STRONG_WEIGHT
                and abs(r.delta) < RESISTANCE_DELTA
            ):
                out.append(
                    {
                        "evidence_id": r.evidence_id,
                        "weight": snapshot["weight"],
                        "delta": round(r.delta, 4),
                        "logit_shift": round(
                            logit(r.new_confidence) - logit(r.previous_confidence), 4
                        ),
                    }
                )
        return out


def run_scenario(scenario: Scenario, store: BeliefStore) -> ScenarioResult:
    """Form a hypothesis from the opening evidence, then meet the later evidence."""
    belief = store.add_belief(Belief(proposition=scenario.proposition, confidence=0.5))
    result = ScenarioResult(scenario=scenario, belief=belief)

    for evidence in scenario.initial_evidence:
        store.consider(belief.id, evidence)
    result.initial_confidence = belief.confidence

    for evidence in scenario.later_evidence:
        store.consider(belief.id, evidence)

    return result


def run_all(directory: Path | str) -> tuple[list[ScenarioResult], BeliefStore]:
    store = BeliefStore.in_directory(directory)
    results = [run_scenario(scenario, store) for scenario in SCENARIOS]
    store.save()
    return results, store


def verify_persistence(store: BeliefStore, results: list[ScenarioResult]) -> dict[str, Any]:
    """Reload the store from disk and check the four things that must survive.

    Run for real, not asserted: the in-memory objects are discarded and every
    check below is made against a store rebuilt from the JSON file.
    """
    expected = {
        r.belief.id: {
            "proposition": r.belief.proposition,
            "confidence": r.belief.confidence,
            "history": len(r.belief.revision_history),
            "applied_evidence": [
                rev.evidence_id for rev in r.belief.revision_history if rev.applied
            ],
        }
        for r in results
    }

    revived = BeliefStore.load(store.path)
    checks = {
        "original_belief_available": True,
        "revision_history_available": True,
        "final_confidence_available": True,
        "revision_evidence_available": True,
        "beliefs_checked": len(expected),
    }

    for belief_id, want in expected.items():
        restored = revived.get_belief(belief_id)
        if restored is None or restored.proposition != want["proposition"]:
            checks["original_belief_available"] = False
            continue
        if len(restored.revision_history) != want["history"]:
            checks["revision_history_available"] = False
        if restored.confidence != want["confidence"]:
            checks["final_confidence_available"] = False
        for evidence_id in want["applied_evidence"]:
            if revived.get_evidence(evidence_id) is None:
                checks["revision_evidence_available"] = False

    checks["all_passed"] = all(v for k, v in checks.items() if isinstance(v, bool))
    return checks


def aggregate(results: list[ScenarioResult]) -> dict[str, Any]:
    return {
        "scenarios": len(results),
        "correct_belief_updates": sum(1 for r in results if r.correct),
        "incorrect_belief_updates": sum(1 for r in results if not r.correct),
        "failures_to_update": sum(1 for r in results if r.failed_to_update),
        "overreactions_to_weak_evidence": sum(len(r.overreactions) for r in results),
        "resistance_to_strong_contradiction": sum(len(r.resistances) for r in results),
        "moved_toward_truth": sum(1 for r in results if r.moved_toward_truth),
        "total_revisions_applied": sum(r.belief.revision_count for r in results),
        "total_evidence_considered": sum(r.belief.considered_count for r in results),
    }


# ------------------------------------------------------------------- report


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def render(
    results: list[ScenarioResult],
    totals: dict[str, Any],
    store_path: Path,
    persistence: dict[str, Any] | None = None,
) -> str:
    lines: list[str] = []
    add = lines.append

    add("# ECHO 2 — Belief Revision")
    add("")
    add("## BELIEF REVISION ≠ GENERAL LEARNING")
    add("")
    add(
        "This experiment tests one narrow thing: **whether ECHO can hold an "
        "uncertain belief, take in new evidence, work out whether that evidence "
        "bears on the belief, and revise it in a measurable, reproducible way.**"
    )
    add("")
    add("It is **not** learning, and the distinction is not a technicality:")
    add("")
    add(
        "- Nothing **generalises.** Evidence about the orchids moves the orchid "
        "belief and nothing else. There is no transfer to a new proposition."
    )
    add(
        "- Nothing about ECHO **changes.** The update rule is a fixed constant "
        "(`EVIDENCE_SCALE`), identical on the first revision and the thousandth. "
        "ECHO does not get better at weighing evidence by weighing evidence."
    )
    add(
        "- There is no **model, no parameters, no training.** This runs entirely "
        "offline; the arithmetic is a log-odds sum you could do on paper."
    )
    add(
        "- ECHO does not decide **what is worth believing** — propositions are "
        "given, and evidence arrives labelled with its stance."
    )
    add("")
    add(
        "What moved is one number attached to one sentence, by a rule that was "
        "fixed in advance. Calling that learning would be a category error."
    )
    add("")

    add("## How a belief moves")
    add("")
    add("```")
    add("logit(confidence) += ±EVIDENCE_SCALE × reliability × relevance")
    add("```")
    add("")
    add(
        f"`EVIDENCE_SCALE = {EVIDENCE_SCALE}`, and evidence with relevance below "
        f"`MIN_RELEVANCE = {MIN_RELEVANCE}` is recorded but **not applied** — the "
        "belief is left alone and the decision is written into the history with "
        "its reason."
    )
    add("")
    add(
        "`reliability × relevance` is the evidence's **weight**. A rumour about "
        "the right topic and a forensic report about the wrong topic both move a "
        "belief very little. Only evidence that is both trustworthy and pertinent "
        "moves it much. This is why nothing here mechanically lowers confidence "
        "whenever it is contradicted."
    )
    add("")

    add("## Results")
    add("")
    add("| Measure | Count |")
    add("| --- | --- |")
    add(f"| Scenarios | {totals['scenarios']} |")
    add(f"| Correct belief updates | {totals['correct_belief_updates']} |")
    add(f"| Incorrect belief updates | {totals['incorrect_belief_updates']} |")
    add(f"| Failures to update | {totals['failures_to_update']} |")
    add(f"| Overreactions to weak evidence | {totals['overreactions_to_weak_evidence']} |")
    add(
        f"| Resistance to strong contradictory evidence | "
        f"{totals['resistance_to_strong_contradiction']} |"
    )
    add(f"| Ended closer to the truth than it started | {totals['moved_toward_truth']}/{totals['scenarios']} |")
    add(f"| Revisions applied | {totals['total_revisions_applied']} |")
    add(f"| Evidence considered | {totals['total_evidence_considered']} |")
    add("")
    add(
        "Definitions, fixed before the run: a **correct update** ends on the right "
        f"side of {DECISION_THRESHOLD}; a **failure to update** is a belief that "
        "needed overturning, met strong evidence against it, and moved less than "
        f"{MATERIAL_DELTA}; an **overreaction** is evidence of weight ≤ "
        f"{WEAK_WEIGHT} moving confidence more than {OVERREACTION_DELTA}; "
        f"**resistance** is contradicting evidence of weight ≥ {STRONG_WEIGHT} "
        f"moving it less than {RESISTANCE_DELTA}."
    )
    add("")
    add(
        f"**On the {totals['moved_toward_truth']}/{totals['scenarios']} 'ended closer "
        "to the truth' figure** — this is not a failure count, and it is worth being "
        "precise about why. In S3, S4 and S6 the belief was already on the correct "
        "side and the later evidence was genuinely contradictory, just weak. Moving "
        "slightly *away* from the truth in response to real if unconvincing "
        "counter-evidence is the correct behaviour: a system that ignored weak "
        "contradiction entirely would score better on this metric and be worse at "
        "its job. The figure to read for correctness is the 'correct belief "
        "updates' row; this one measures something narrower and is reported "
        "because leaving it out would flatter the result."
    )
    add("")

    add("| Scenario | Truth | Initial | Final | Δ | Verdict |")
    add("| --- | --- | --- | --- | --- | --- |")
    for r in results:
        add(
            f"| `{r.scenario.id}` | {'true' if r.scenario.ground_truth else 'false'} "
            f"| {_fmt(r.initial_confidence)} | {_fmt(r.final_confidence)} "
            f"| {r.later_delta:+.3f} | {'correct' if r.correct else 'INCORRECT'} |"
        )
    add("")

    add("## Scenario detail")
    add("")
    for r in results:
        s = r.scenario
        add(f"### `{s.id}` — {s.title}")
        add("")
        add(f"**Proposition.** {s.proposition}")
        add("")
        add(f"**Why this scenario exists.** {s.purpose}")
        add("")
        add(
            f"**Initial belief.** Created at 0.500 with no evidence, then raised to "
            f"**{_fmt(r.initial_confidence)}** by the opening evidence alone. The "
            "hypothesis was formed, not supplied."
        )
        add("")
        add("**Evidence and revisions, in order:**")
        add("")
        add("| Evidence | Source | Rel. | Relv. | Weight | Stance | Applied | Confidence |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for rev in r.belief.revision_history:
            snap = rev.evidence_snapshot
            add(
                f"| `{snap['id']}` | {snap['source']} | {snap['reliability']:.2f} "
                f"| {snap['relevance']:.2f} | {snap['weight']:.2f} | {snap['stance']} "
                f"| {'yes' if rev.applied else 'no'} "
                f"| {_fmt(rev.previous_confidence)} → {_fmt(rev.new_confidence)} "
                f"({rev.delta:+.3f}) |"
            )
        add("")
        add("**Why the belief moved as it did:**")
        add("")
        for rev in r.belief.revision_history:
            if rev.evidence_id in {e.id for e in s.initial_evidence}:
                continue
            add(f"- {rev.reason}")
        add("")
        add(
            f"**Final confidence:** {_fmt(r.final_confidence)} "
            f"({'holds' if r.believes_proposition else 'rejects'} the proposition). "
            f"Ground truth: **{s.ground_truth}**. "
            f"{'Correct.' if r.correct else 'INCORRECT — this is a real failure.'}"
        )
        add("")
        if r.overreactions:
            add(f"- ⚠️ Overreaction to weak evidence: {r.overreactions}")
            add("")
        if r.resistances:
            add(f"- ⚠️ Resistance to strong contradictory evidence: {r.resistances}")
            add("")
        add(f"**Hidden state (never shown to ECHO):** {s.hidden_state}")
        add("")

    add("## The reversal test")
    add("")
    s1 = next(r for r in results if r.scenario.id == "S1-belief-correct")
    s2 = next(r for r in results if r.scenario.id == "S2-belief-incorrect")
    add(
        "`S1` and `S2` open with **identical evidence** and therefore an identical "
        f"initial belief of {_fmt(s1.initial_confidence)}. Only the later evidence "
        "differs, and ECHO is never told which world it is in."
    )
    add("")
    add("| | S1 (hypothesis true) | S2 (hypothesis false) |")
    add("| --- | --- | --- |")
    add(f"| Initial | {_fmt(s1.initial_confidence)} | {_fmt(s2.initial_confidence)} |")
    add(f"| Final | {_fmt(s1.final_confidence)} | {_fmt(s2.final_confidence)} |")
    add(f"| Δ | {s1.later_delta:+.3f} | {s2.later_delta:+.3f} |")
    add("")
    add(
        "Both scenarios contain contradictory evidence. In S1 it is an anonymous "
        "note (weight 0.10) and the belief barely moves; in S2 it is dual-sensor "
        "telemetry (weight 0.86) and the belief collapses. The difference is "
        "produced by evidence quality alone — there is no branch anywhere in the "
        "code that consults ground truth."
    )
    add("")

    add("## Persistence")
    add("")
    add(
        f"Every belief, its full revision history, and every evidence record are "
        f"written to `{store_path.name}`. The check below was **run as part of this "
        "report**, not asserted: the in-memory objects were discarded and the store "
        "rebuilt from the JSON file on disk."
    )
    add("")
    if persistence:
        add("| After restart | Survives |")
        add("| --- | --- |")
        add(f"| The original belief | {'yes' if persistence['original_belief_available'] else 'NO'} |")
        add(f"| The revision history | {'yes' if persistence['revision_history_available'] else 'NO'} |")
        add(f"| The final confidence | {'yes' if persistence['final_confidence_available'] else 'NO'} |")
        add(
            f"| The evidence responsible for each revision | "
            f"{'yes' if persistence['revision_evidence_available'] else 'NO'} |"
        )
        add("")
        add(f"Beliefs checked: {persistence['beliefs_checked']}.")
        add("")

    add("## Limitations — read before trusting any of this")
    add("")
    add(
        "1. **Stance is supplied, not inferred.** Every evidence record arrives "
        "labelled `supports` or `contradicts`. ECHO decides *how much* an "
        "observation matters, not *whether it disagrees*. Reading contradiction "
        "out of raw natural language is a different and much harder problem, and "
        "it is not solved here."
    )
    add(
        "2. **Reliability and relevance are hand-assigned.** A human author set "
        "every number in `experiments/world.py`. In a real system those would "
        "themselves be uncertain judgements, and errors in them would propagate "
        "directly into confidence."
    )
    add(
        "3. **Evidence is treated as independent.** Log-odds addition assumes it. "
        "Three reports derived from the same original source would be triple-"
        "counted here, which is a well-known way to become overconfident."
    )
    add(
        "4. **Six scenarios written by the same author as the system.** They probe "
        "the failure modes their author thought of. That is a weak guarantee."
    )
    add(
        "5. **Saturation caveat.** Near 0 or 1, a large log-odds shift produces a "
        "small change in probability. A strong contradiction against a 0.99 belief "
        "can therefore look like 'resistance' when measured as a probability "
        "delta, even though the update was applied at full strength — the "
        "`logit_shift` field is reported alongside so this is visible."
    )
    add(
        "6. **Ordering does not matter, and arguably should.** Log-odds addition "
        "is commutative, so the same evidence in any order yields the same "
        "confidence. Real inquiry is not always order-independent."
    )
    add("")

    add("## Conclusion")
    add("")
    correct = totals["correct_belief_updates"]
    total = totals["scenarios"]
    add(
        f"ECHO reached the correct side of the decision threshold in **{correct} of "
        f"{total}** scenarios, with **{totals['overreactions_to_weak_evidence']}** "
        f"overreactions to weak evidence and "
        f"**{totals['resistance_to_strong_contradiction']}** cases of resistance to "
        "strong contradictory evidence."
    )
    add("")
    add(
        "The mechanism does what was asked of it: it holds an uncertain belief, "
        "distinguishes evidence that bears on the proposition from evidence that "
        "does not, weighs what remains by how much it deserves to be trusted, "
        "revises in a direction and magnitude caused by that weighing, and keeps "
        "an unbroken record of every step. The reversal test shows it is not "
        "merely lowering confidence on contact with disagreement."
    )
    add("")
    add(
        "That is the whole claim. **It is not learning**, it does not generalise, "
        "and it says nothing about whether ECHO could form these propositions or "
        "assess this evidence on its own — both of which were handed to it."
    )
    add("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the belief-revision experiment.")
    parser.add_argument("--out", help="write the markdown report here")
    parser.add_argument("--data", help="directory for the belief store")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data) if args.data else RESULTS_DIR / "store"

    results, store = run_all(data_dir)
    totals = aggregate(results)
    persistence = verify_persistence(store, results)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = RESULTS_DIR / f"belief_run_{stamp}.json"
    raw_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "evidence_scale": EVIDENCE_SCALE,
                "min_relevance": MIN_RELEVANCE,
                "totals": totals,
                "persistence": persistence,
                "scenarios": [
                    {
                        "id": r.scenario.id,
                        "ground_truth": r.scenario.ground_truth,
                        "initial_confidence": r.initial_confidence,
                        "final_confidence": r.final_confidence,
                        "correct": r.correct,
                        "belief": r.belief.to_dict(),
                    }
                    for r in results
                ],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    markdown = render(results, totals, store.path, persistence)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(markdown, encoding="utf-8")
        print(f"report -> {out}")
    else:
        print(markdown)
    print(f"raw    -> {raw_path}")
    print(f"store  -> {store.path}")

    for r in results:
        flag = "ok " if r.correct else "BAD"
        print(
            f"  [{flag}] {r.scenario.id:<28} "
            f"{r.initial_confidence:.3f} -> {r.final_confidence:.3f} "
            f"(truth={r.scenario.ground_truth})"
        )
    return 0 if totals["incorrect_belief_updates"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
