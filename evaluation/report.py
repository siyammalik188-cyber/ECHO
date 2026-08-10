"""Render a raw results file into the markdown evaluation report.

    python -m evaluation.report evaluation/results/raw_<stamp>.json \
        --out docs/ECHO_1_5_MEMORY_EVALUATION.md

Kept separate from the runner so the report can be regenerated from saved model
output without re-paying for the API calls.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .run_evaluation import load_cases
from .scoring import CaseResult, aggregate, calibration_summary, score_case


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def num(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def score_pass(cases: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[CaseResult]:
    by_id = {c["id"]: c for c in cases}
    results = []
    for record in records:
        case = by_id[record["case_id"]]
        result = score_case(case, record.get("proposed", []))
        result.error = record.get("error")
        results.append(result)
    return results


def render(payload: dict[str, Any], dataset: dict[str, Any]) -> str:
    cases = dataset["cases"]
    lines: list[str] = []
    add = lines.append

    add("# ECHO 1.5 — Memory Extraction Evaluation")
    add("")
    add(
        "Measures one thing only: **whether ECHO can selectively form persistent "
        "memories from conversation.** ECHO does not learn from this experiment. "
        "Nothing here is fed back into the system, no weights change, no prompt "
        "adapts, and no memory persists past the run."
    )
    add("")
    add("## Run")
    add("")
    add(f"- Model: `{payload['model']}`")
    add(f"- Dataset version: {payload['dataset_version']} ({payload['case_count']} cases)")
    add(f"- Passes: {payload['runs']}")
    add(f"- Started: {payload.get('started_at')}")
    add(f"- Finished: {payload.get('finished_at', 'incomplete')}")
    add(
        f"- Store thresholds: confidence ≥ {payload['thresholds']['min_confidence']}, "
        f"importance ≥ {payload['thresholds']['min_importance']}"
    )
    add("")

    for pass_payload in payload["passes"]:
        results = score_pass(cases, pass_payload["records"])
        totals = aggregate(results)
        run_label = (
            f"Pass {pass_payload['run'] + 1}" if payload["runs"] > 1 else "Results"
        )

        add(f"## {run_label}")
        add("")
        add("| Metric | Value |")
        add("| --- | --- |")
        add(f"| Precision | {pct(totals.precision)} |")
        add(f"| Precision (strict — tolerated counted as FP) | {pct(totals.strict_precision)} |")
        add(f"| Recall | {pct(totals.recall)} |")
        add(f"| F1 | {num(totals.f1)} |")
        add(f"| True positives | {totals.true_positives} |")
        add(f"| False positives | {totals.false_positives} |")
        add(f"| False negatives | {totals.false_negatives} |")
        add(f"| Tolerated extractions | {totals.tolerated} |")
        add(f"| Memory-type accuracy (on true positives) | {pct(totals.type_accuracy)} |")
        add(f"| Cases fully clean | {totals.cases_passed}/{totals.cases_total} |")
        add(f"| API errors | {totals.errors} |")
        add("")

        for field_name in ("confidence", "importance"):
            summary = calibration_summary(results, field_name)
            add(f"### {field_name.title()} calibration")
            add("")
            if not summary.get("samples"):
                add("_No true positives, so nothing to calibrate._")
                add("")
                continue
            add(f"- Samples: {summary['samples']}")
            add(f"- Inside the pre-registered band: {summary['in_band']} ({pct(summary['in_band_rate'])})")
            add(f"- Above band: {summary['over_band']} · Below band: {summary['under_band']}")
            add(f"- Mean deviation from band: {num(summary['mean_deviation'])}")
            add(f"- Mean deviation when outside: {num(summary['mean_deviation_when_outside'])}")
            if summary.get("worst"):
                w = summary["worst"]
                add(
                    f"- Worst miss: `{w['case_id']}` — expected {w['expected_band']}, "
                    f"got {w['actual']} (off by {num(w['deviation'])})"
                )
            add("")

        add("### Per case")
        add("")
        add("| Case | Requirement | Proposed | TP | FP | FN | Tol | Clean |")
        add("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for result in results:
            add(
                f"| `{result.case_id}` | {result.requirement} | {len(result.proposed)} "
                f"| {result.count('true_positive')} | {result.count('false_positive')} "
                f"| {result.count('false_negative')} | {result.count('tolerated')} "
                f"| {'yes' if result.passed else 'NO'} |"
            )
        add("")

        # Incorrect extractions
        add("### Examples of incorrect extraction (false positives)")
        add("")
        false_positives = [
            j for r in results for j in r.judgements if j.kind == "false_positive"
        ]
        if not false_positives:
            add("_None._")
        else:
            for j in false_positives:
                label = j.label or "unlabelled surplus"
                add(f"- **`{j.case_id}`** — {label}")
                add(f"  - proposed: “{j.content}”")
                add(
                    f"  - type `{j.memory_type}`, confidence {j.confidence}, "
                    f"importance {j.importance}"
                )
                add(f"  - why counted: {j.detail}")
        add("")

        # Missed memories
        add("### Examples of missed memories (false negatives)")
        add("")
        false_negatives = [
            j for r in results for j in r.judgements if j.kind == "false_negative"
        ]
        if not false_negatives:
            add("_None._")
        else:
            for j in false_negatives:
                add(f"- **`{j.case_id}`** — expected but not produced: {j.label}")
        add("")

        # Tolerated
        tolerated = [j for r in results for j in r.judgements if j.kind == "tolerated"]
        if tolerated:
            add("### Tolerated extractions (allowed, not required)")
            add("")
            for j in tolerated:
                add(f"- **`{j.case_id}`** — {j.label}: “{j.content}”")
            add("")

        # Errors
        errored = [r for r in results if r.error]
        if errored:
            add("### API errors")
            add("")
            for result in errored:
                add(f"- `{result.case_id}`: {result.error}")
            add("")

    add("## Full model output")
    add("")
    add("Every proposal, verbatim, including ones that scored as failures.")
    add("")
    for pass_payload in payload["passes"]:
        for record in pass_payload["records"]:
            add(f"#### `{record['case_id']}`")
            add("")
            if record.get("error"):
                add(f"- ERROR: {record['error']}")
                add("")
                continue
            if not record["proposed"]:
                add("_No memories proposed._")
                add("")
                continue
            for memory in record["proposed"]:
                survives = memory in record.get("survives_threshold", [])
                add(
                    f"- [{memory['memory_type']}] “{memory['content']}” "
                    f"— confidence {memory['confidence']}, importance {memory['importance']}"
                    f"{'' if survives else '  _(filtered out by the store threshold)_'}"
                )
            add("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render an evaluation report.")
    parser.add_argument("results", help="path to a raw results JSON file")
    parser.add_argument("--out", help="write markdown here instead of stdout")
    args = parser.parse_args(argv)

    payload = json.loads(Path(args.results).read_text(encoding="utf-8"))
    markdown = render(payload, load_cases())

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        print(f"wrote {path}")
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
