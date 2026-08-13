"""What ECHO 11 has to be true for: the modules compose, and say why.

The properties that matter are that the coordinator adds no reasoning of its
own, that every conclusion carries provenance, that persistent state survives a
restart while temporary context does not, that a revised belief keeps its
history, that injected failures are detected from ECHO's own measurements, and
that `NO KNOWN PATTERN` is reachable.
"""

from __future__ import annotations

import pytest

from echo.integration import (
    LOOP_ORDER,
    NOVELTY_THRESHOLD,
    RECENT_WINDOW,
    RELIABILITY_DROP,
    TURNED_CEILING,
    FailureKind,
    KnowledgeConflict,
    PersistentState,
    Stage,
    Trace,
    TraceStep,
    best_match,
    is_novel,
    match_quality,
    source_turned,
)
from echo.pattern import PatternStatus, StructuralPattern
from echo.social import SocialLedger, parse_claim
from echo import expressions as ex

from source_audit import code_without_docs, imports_of

from experiments import run_integration as R
from experiments.integration_worlds import (
    NOVEL_VAL_B_STOP,
    REGIME_CHANGE,
    SOURCE_TURNS,
    build_long_horizon,
    build_novel,
)


@pytest.fixture(scope="module")
def session():
    world = build_long_horizon()
    state = PersistentState()
    echo = R.run_integrated(world, state)
    baseline = R.run_baseline(world)
    novel = R.run_novel(state)
    return {
        "world": world,
        "state": state,
        "echo": echo,
        "baseline": baseline,
        "novel": novel,
    }


# ------------------------------------------------- 11.1 the loop is complete


def test_1_the_loop_covers_every_stage():
    assert len(LOOP_ORDER) == len(set(LOOP_ORDER))
    assert set(LOOP_ORDER) == set(Stage)
    assert LOOP_ORDER[0] is Stage.OBSERVE
    assert LOOP_ORDER[-1] is Stage.CONSULT


def test_1b_the_coordinator_owns_no_reasoning():
    """It adds order, plumbing and provenance — not arithmetic."""
    import echo.integration as module

    code = code_without_docs(module.__file__)
    # No scoring rules, no probability machinery, no fitting.
    for forbidden in ("math.exp", "math.log", "def fit", "def predict", "brier"):
        assert forbidden not in code, f"{forbidden!r} in the coordinator"


def test_1c_the_coordinator_delegates_to_the_tested_modules():
    import echo.integration as module

    imports = " ".join(imports_of(module.__file__))
    for expected in ("metacognition", "pattern", "social"):
        assert expected in imports


def test_1d_no_truth_generator_is_reachable():
    import echo.integration as module

    for line in imports_of(module.__file__):
        assert "integration_worlds" not in line
        assert "experiments" not in line


# ------------------------------------------- 11.2 long horizon, no resets


def test_2_the_run_is_long_and_memory_is_never_reset(session):
    echo = session["echo"]
    state = session["state"]
    assert len(echo.probabilities) >= 700
    # claims accumulate across the whole run rather than per-episode
    assert len(state.social) == len(echo.probabilities) * 4


def test_2b_echo_beats_the_baseline_on_prediction(session):
    echo = session["echo"].scores(4)
    baseline = session["baseline"].scores(4)
    assert echo["brier"] < baseline["brier"]


def test_2c_the_hidden_relationship_is_discovered(session):
    """Found from the data, not supplied."""
    assert session["echo"].discovered is not None
    expression = ex.parse(session["echo"].discovered)
    assert expression.variables()  # a real expression over the world's columns


# ------------------------------------------------- 11.3 the novel situation


def test_3_the_novel_world_shares_nothing_but_shape():
    long_horizon = build_long_horizon()
    novel = build_novel()
    assert set(long_horizon.variable_names).isdisjoint(novel.variable_names)
    # wildly different scales
    spread = max(
        max(abs(v) for v in novel.observations.column(name))
        for name in novel.variable_names
    )
    assert spread > 1000


def test_3b_transfer_costs_far_less_than_cold_start(session):
    novel = session["novel"]
    assert novel.transfer_candidates > 0
    assert novel.transfer_candidates < novel.cold_candidates / 50


def test_3c_no_special_rule_exists_for_the_novel_world():
    import echo.integration as module

    code = code_without_docs(module.__file__)
    for name in build_novel().variable_names:
        assert name not in code


# --------------------------------------------- 11.4 failure detection


def test_4_injected_failures_are_detected(session):
    kinds = {f[1] for f in session["echo"].failures}
    assert FailureKind.SOURCE_TURNED in kinds
    assert FailureKind.REGIME_CHANGE in kinds
    assert FailureKind.PATTERN_STOPPED_WORKING in kinds


def test_4b_the_source_that_turned_is_the_one_flagged(session):
    turned = [f[2] for f in session["echo"].failures if f[1] is FailureKind.SOURCE_TURNED]
    assert turned == ["S3"]


def test_4c_a_turn_is_detected_after_it_happens_not_before(session):
    ticks = [f[0] for f in session["echo"].failures if f[1] is FailureKind.SOURCE_TURNED]
    assert all(tick > SOURCE_TURNS for tick in ticks)


def test_4d_a_steady_source_is_not_flagged():
    """The detector must not fire on an ordinary run of bad luck."""
    ledger = SocialLedger()
    for tick in range(80):
        ledger.hear(
            parse_claim("STEADY", f"I am sure hypothesis {tick} holds because X")
        )
        # right 80% of the time, in a fixed repeating pattern
        ledger.resolve(f"hypothesis-{tick}", tick % 5 != 0, tick=tick)
    assert source_turned(ledger, "STEADY") is False


def test_4e_a_real_turn_is_detected():
    ledger = SocialLedger()
    for tick in range(60):
        ledger.hear(parse_claim("TURNS", f"I am sure hypothesis {tick} holds because X"))
        ledger.resolve(f"hypothesis-{tick}", True, tick=tick)
    assert source_turned(ledger, "TURNS") is False
    for tick in range(60, 60 + RECENT_WINDOW):
        ledger.hear(parse_claim("TURNS", f"I am sure hypothesis {tick} holds because X"))
        ledger.resolve(f"hypothesis-{tick}", False, tick=tick)
    assert source_turned(ledger, "TURNS") is True


# ------------------------------------------------------ 11.5 provenance


def test_5_every_conclusion_has_a_trace(session):
    echo = session["echo"]
    assert len(echo.traces) == len(echo.probabilities)
    for trace in echo.traces:
        assert trace.conclusion
        assert trace.evidence, f"tick {trace.tick} has no evidence"
        assert trace.steps


def test_5b_a_trace_answers_why(session):
    trace = session["echo"].traces[-1]
    text = trace.explain()
    for expected in ("CONCLUSION:", "EVIDENCE:", "SOURCE:"):
        assert expected in text
    assert trace.source_contributions
    assert trace.current_belief is not None


def test_5c_traces_name_the_modules_that_did_the_work(session):
    modules = {
        step.module for trace in session["echo"].traces for step in trace.steps
    }
    assert {"discovery", "social", "metacognition"} <= modules


def test_5d_a_trace_is_immutable():
    trace = Trace(conclusion="x", tick=1, steps=(TraceStep(Stage.OBSERVE, "m", "s"),))
    with pytest.raises(Exception):
        trace.conclusion = "y"  # type: ignore[misc]


# ------------------------------------------------------ 11.6 retention


def test_6_state_survives_a_restart(tmp_path, session):
    state = session["state"]
    path = tmp_path / "state.json"
    state.save(path)
    reloaded = PersistentState.load(path)

    assert reloaded.beliefs == state.beliefs
    assert len(reloaded.belief_history) == len(state.belief_history)
    assert [p.text() for p in reloaded.patterns] == [p.text() for p in state.patterns]
    assert len(reloaded.social) == len(state.social)
    for source in state.social.sources():
        assert reloaded.social.reliability(source) == pytest.approx(
            state.social.reliability(source)
        )
    assert reloaded.metacognition.tally("prediction").to_dict() == (
        state.metacognition.tally("prediction").to_dict()
    )


def test_6b_temporary_context_is_not_persisted(tmp_path, session):
    """There is no field for it, so it cannot leak into the file."""
    path = tmp_path / "state.json"
    session["state"].save(path)
    payload = path.read_text(encoding="utf-8")
    for forbidden in ("working_context", "scratch", "temporary"):
        assert forbidden not in payload
    reloaded = PersistentState.load(path)
    assert not hasattr(reloaded, "working_context")


def test_6c_a_reloaded_pattern_still_grounds_identically(session, tmp_path):
    state = session["state"]
    if not state.patterns:
        pytest.skip("nothing was discovered to abstract")
    path = tmp_path / "state.json"
    state.save(path)
    reloaded = PersistentState.load(path)
    names = ("Q1", "Q2", "Q3", "Q4", "Q5", "Q6")
    assert [e.text() for e in reloaded.patterns[-1].ground(names)] == [
        e.text() for e in state.patterns[-1].ground(names)
    ]


# ------------------------------------------------ 11.7 knowledge conflict


def test_7_a_revised_belief_keeps_its_history():
    state = PersistentState()
    assert state.revise("p", 0.7, "first sighting", "initial") is None
    conflict = state.revise("p", 0.3, "contradicting evidence", "reassessment")
    assert conflict is not None
    assert conflict.old_belief == 0.7
    assert conflict.current_belief == 0.3
    assert conflict.new_evidence == "contradicting evidence"
    assert state.beliefs["p"] == 0.3
    assert len(state.revisions_of("p")) == 1


def test_7b_nothing_is_silently_overwritten():
    state = PersistentState()
    state.revise("p", 0.1, "e0", "r0")
    for index, value in enumerate((0.2, 0.3, 0.4, 0.5)):
        state.revise("p", value, f"e{index + 1}", f"r{index + 1}")
    revisions = state.revisions_of("p")
    assert len(revisions) == 4
    assert [r.old_belief for r in revisions] == [0.1, 0.2, 0.3, 0.4]


def test_7c_the_long_run_revised_a_belief(session):
    assert session["state"].revisions_of("the-relationship-holds")


# ------------------------------------------------------- 11.8 novelty


def test_8_no_known_pattern_is_reachable(session):
    novel = session["novel"]
    assert novel.novelty_flagged is True
    assert novel.novelty_quality < NOVELTY_THRESHOLD


def test_8b_a_matching_situation_is_not_called_novel():
    pattern = StructuralPattern.from_expression(
        ex.parse("PRODUCT(CHANGE(X2), LAG(X4, 3))"), created_at=1
    )
    signature = {"features": {"PRODUCT", "CHANGE", "LAG"}}
    flagged, matched, quality = is_novel([pattern], signature)
    assert flagged is False
    assert matched is pattern
    assert quality == pytest.approx(1.0)


def test_8c_an_empty_library_is_always_novel():
    flagged, matched, quality = is_novel([], {"features": {"PRODUCT"}})
    assert flagged is True
    assert matched is None
    assert quality == 0.0


def test_8d_retired_patterns_are_not_matched():
    pattern = StructuralPattern.from_expression(
        ex.parse("PRODUCT(CHANGE(X2), LAG(X4, 3))"), created_at=1
    )
    retired = (
        pattern.with_status(PatternStatus.TESTING, "x")
        .with_status(PatternStatus.WEAKENED, "y")
        .with_status(PatternStatus.RETIRED, "z")
    )
    matched, quality = best_match([retired], {"features": {"PRODUCT"}})
    assert matched is None
    assert quality == 0.0


# ------------------------------------------------- 11.9 the benchmark


def test_9_the_benchmark_has_thirteen_metrics(session):
    from experiments.causal_worlds import by_id as causal_by_id
    from experiments.run_causal import run_scenario

    benchmark = R.build_benchmark(
        session["world"],
        session["echo"],
        session["baseline"],
        session["state"],
        session["novel"],
        run_scenario(causal_by_id("CAUSAL-chain")),
    )
    assert len(benchmark) == 13
    for entry in benchmark:
        assert set(entry) == {"metric", "echo", "baseline", "note"}
        assert entry["note"]


def test_9b_the_benchmark_reports_metrics_echo_loses(session):
    """No cherry-picking: a metric where ECHO does not win must still appear."""
    from experiments.causal_worlds import by_id as causal_by_id
    from experiments.run_causal import run_scenario

    benchmark = R.build_benchmark(
        session["world"],
        session["echo"],
        session["baseline"],
        session["state"],
        session["novel"],
        run_scenario(causal_by_id("CAUSAL-chain")),
    )
    # the novel-world Brier is a genuine tie or loss and must be present
    novel_entry = next(e for e in benchmark if e["metric"].startswith("13."))
    assert novel_entry["echo"] is not None
    assert novel_entry["baseline"] is not None


def test_9c_the_study_is_reproducible():
    first_state = PersistentState()
    second_state = PersistentState()
    world = build_long_horizon()
    first = R.run_integrated(world, first_state)
    second = R.run_integrated(build_long_horizon(), second_state)
    assert first.discovered == second.discovered
    assert first.probabilities == second.probabilities
    assert [f[1] for f in first.failures] == [f[1] for f in second.failures]


# ------------------------------------------------------------- safety


def test_no_network_or_shell_anywhere_in_echo():
    import pathlib

    import echo

    root = pathlib.Path(echo.__file__).resolve().parent
    for path in sorted(root.glob("*.py")):
        code = code_without_docs(path)
        for forbidden in (
            "subprocess",
            "socket",
            "urllib",
            "requests",
            "os.system",
            "eval(",
            "exec(",
            "__import__(",
        ):
            assert forbidden not in code, f"{forbidden!r} in {path.name}"
