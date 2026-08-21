"""Run ECHO's real extraction prompt against a real model and score the result.

This calls the live API. It requires credentials (`ANTHROPIC_API_KEY`, or an
`ant auth login` profile) and it costs money. It imports the extractor from the
`echo` package rather than reimplementing it, so what is measured is exactly
what ships.

    python -m evaluation.run_evaluation                 # one pass, default model
    python -m evaluation.run_evaluation --runs 3        # repeat, to see variance
    python -m evaluation.run_evaluation --dry-run       # no API calls; verify wiring

Raw model output is written to evaluation/results/ so a scored report can be
regenerated later without paying for the calls again.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from echo.conversation import Conversation
from echo.extraction import (
    EXTRACTION_SYSTEM,
    MIN_CONFIDENCE,
    MIN_IMPORTANCE,
    LLMMemoryExtractor,
)
from echo.llm import DEFAULT_MODEL, AnthropicLLM

HERE = Path(__file__).parent
CASES_PATH = HERE / "cases.json"
RESULTS_DIR = HERE / "results"


def load_cases(path: Path = CASES_PATH) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_conversation(case: dict[str, Any]) -> Conversation:
    conversation = Conversation(id=case["id"].replace("-", "_"))
    for turn in case["conversation"]:
        conversation.add(turn["role"], turn["content"])
    return conversation


def run_case(
    case: dict[str, Any], extractor: LLMMemoryExtractor
) -> dict[str, Any]:
    """Extract from one case. Errors are recorded, never swallowed silently."""
    record: dict[str, Any] = {"case_id": case["id"], "proposed": [], "error": None}
    try:
        candidates = extractor.extract(build_conversation(case))
    except Exception as exc:  # noqa: BLE001 — an API failure is a result, not a crash
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
        return record

    record["proposed"] = [
        {
            "content": c.content,
            "memory_type": c.memory_type.value,
            "confidence": c.confidence,
            "importance": c.importance,
        }
        for c in candidates
    ]
    # What would actually reach the store, after ECHO's threshold gate.
    record["survives_threshold"] = [
        m
        for m in record["proposed"]
        if m["confidence"] >= MIN_CONFIDENCE and m["importance"] >= MIN_IMPORTANCE
    ]
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate ECHO memory extraction.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--runs", type=int, default=1, help="passes over the dataset")
    parser.add_argument("--only", help="run a single case id")
    parser.add_argument("--dry-run", action="store_true", help="no API calls")
    parser.add_argument("--out", help="output path for the raw results JSON")
    args = parser.parse_args(argv)

    dataset = load_cases()
    cases = dataset["cases"]
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
        if not cases:
            print(f"no case with id {args.only!r}", file=sys.stderr)
            return 1

    started = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "started_at": started.isoformat(),
        "model": args.model,
        "runs": args.runs,
        "dataset_version": dataset["dataset_version"],
        "case_count": len(cases),
        "thresholds": {
            "min_confidence": MIN_CONFIDENCE,
            "min_importance": MIN_IMPORTANCE,
        },
        # Recorded verbatim so a later reader can tell whether the prompt that
        # produced these numbers is the prompt still in the codebase.
        "extraction_system_prompt": EXTRACTION_SYSTEM,
        "dry_run": args.dry_run,
        "passes": [],
    }

    if args.dry_run:
        print(f"[dry run] would extract from {len(cases)} cases x {args.runs} run(s)")
        for case in cases:
            print(f"  {case['id']:<26} {case['requirement']}")
        return 0

    try:
        llm = AnthropicLLM(model=args.model)
    except Exception as exc:  # noqa: BLE001
        print(f"could not build a model client: {exc}", file=sys.stderr)
        print(
            "This evaluation requires real credentials. Set ANTHROPIC_API_KEY "
            "(or run `ant auth login`) and try again.",
            file=sys.stderr,
        )
        return 2
    extractor = LLMMemoryExtractor(llm)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    out_path = Path(args.out) if args.out else RESULTS_DIR / f"raw_{stamp}.json"

    for run_index in range(args.runs):
        records = []
        for case in cases:
            print(f"[run {run_index + 1}/{args.runs}] {case['id']} ... ", end="", flush=True)
            record = run_case(case, extractor)
            if record["error"]:
                print(f"ERROR {record['error']}")
            else:
                print(f"{len(record['proposed'])} proposed")
            records.append(record)
            # Write after every case so a mid-run failure loses nothing.
            payload_snapshot = dict(payload)
            payload_snapshot["passes"] = payload["passes"] + [{"run": run_index, "records": records}]
            out_path.write_text(
                json.dumps(payload_snapshot, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        payload["passes"].append({"run": run_index, "records": records})

    payload["finished_at"] = datetime.now(timezone.utc).isoformat()
    out_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\nraw results -> {out_path}")

    attempted = sum(len(p["records"]) for p in payload["passes"])
    failed = sum(1 for p in payload["passes"] for r in p["records"] if r["error"])
    if failed:
        print(f"{failed}/{attempted} case runs failed with an error.", file=sys.stderr)
    if failed == attempted:
        # Every call failed — usually missing credentials. Do not let this look
        # like a completed evaluation to a caller checking the exit code.
        print(
            "No case produced output; there is nothing to score. "
            "Check credentials (ANTHROPIC_API_KEY, or `ant auth login`).",
            file=sys.stderr,
        )
        return 2

    print("now render the report:")
    print(f"  python -m evaluation.report {out_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
