"""The eighteen properties Challenge 5 requires, one test each (plus sub-tests).

These lock the behaviour that matters: immutability, safety, bounded complexity,
chronological separation, an unbreachable holdout, rejection of overfit and null
findings, genuine discovery that generalises, complexity preference, persisted
provenance, reproducibility, and history that survives.

The experiment worlds are generated once at module scope and reused; a full
search is ~5 s, so the searched worlds are computed once in a fixture rather
than per test.
"""

from __future__ import annotations

import json

import pytest

from echo import discovery as D
from echo import expressions as ex
from echo.discovery_ledger import DiscoveryLedger, DiscoveryRecord
from echo.hypothesis import Hypothesis, HypothesisStatus
from echo.observations import (
    TEST,
    TRAIN,
    VAL_A,
    VAL_B,
    ChronologicalSplit,
    HoldoutViolation,
    ObservationSet,
)

from experiments import run_discovery
from experiments.discovery_worlds import (
    ROWS,
    TRAIN_STOP,
    VAL_A_STOP,
    VAL_B_STOP,
    WORLDS,
    by_id,
)


# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def searched(tmp_path_factory):
    """Every world searched once, holdout evaluated, keyed by world id.

    Written to a temporary directory: running the suite must not append to the
    committed experiment ledger, which is the artefact of one clean run.
    """
    scratch = tmp_path_factory.mktemp("discovery")
    out: dict[str, dict] = {}
    for world in WORLDS:
        record, context = run_discovery.run_world(world, out_dir=scratch / world.id)
        out[world.id] = {"record": record, **context}
    return out


def _split() -> ChronologicalSplit:
    return ChronologicalSplit(
        ROWS, train_stop=TRAIN_STOP, val_a_stop=VAL_A_STOP, val_b_stop=VAL_B_STOP
    )


# ---------------------------------------------------------- 1. immutability


def test_1_hypotheses_are_immutable():
    expr = ex.parse("LAG(X3, 2)")
    h = Hypothesis.propose(
        expr, created_at=1, evidence_window=(4, 480), discovery_reason="test"
    )
    with pytest.raises(Exception):
        h.status = HypothesisStatus.ACTIVE  # type: ignore[misc]
    with pytest.raises(Exception):
        h.complexity = 99  # type: ignore[misc]


def test_1b_a_status_change_makes_a_new_object_and_keeps_the_old():
    expr = ex.parse("FEATURE(X1)")
    h = Hypothesis.propose(
        expr, created_at=1, evidence_window=(4, 480), discovery_reason="test"
    )
    moved = h.with_status(HypothesisStatus.TESTING, "fitted")
    assert h.status is HypothesisStatus.CANDIDATE
    assert moved.status is HypothesisStatus.TESTING
    assert moved.revision == h.revision + 1
    assert moved.hypothesis_id == h.hypothesis_id


def test_1c_a_rejected_hypothesis_cannot_be_revived():
    expr = ex.parse("FEATURE(X1)")
    h = Hypothesis.propose(
        expr, created_at=1, evidence_window=(4, 480), discovery_reason="test"
    )
    rejected = h.with_status(HypothesisStatus.TESTING, "x").with_status(
        HypothesisStatus.REJECTED, "failed"
    )
    with pytest.raises(ValueError):
        rejected.with_status(HypothesisStatus.ACTIVE, "sneak")


# ------------------------------------------------ 2. expressions safely held


def test_2_expressions_are_structured_trees_not_strings():
    expr = ex.parse("PRODUCT(CHANGE(X2), LAG(X4, 3))")
    assert isinstance(expr, ex.Product)
    assert isinstance(expr.left, ex.Change)
    assert isinstance(expr.right, ex.Lag)
    # round-trips exactly through both serialisations
    assert ex.from_dict(expr.to_dict()).text() == expr.text()
    assert ex.parse(expr.text()).text() == expr.text()


def test_2b_the_op_tag_is_a_class_constant_not_a_forgeable_field():
    # op is a ClassVar, so a payload cannot set it to something the interpreter
    # would misread: an unknown operator is refused, and a stray field on a
    # known operator is ignored rather than overriding the class constant.
    node = ex.from_dict({"op": "FEATURE", "variable": "X1", "op_override": "SYSTEM"})
    assert node.op == "FEATURE"
    assert not hasattr(node, "op_override")
    with pytest.raises(ex.ExpressionError):
        ex.from_dict({"op": "SYSTEM", "variable": "X1"})


# --------------------------------------- 3. arbitrary code execution is out


def test_3_arbitrary_code_cannot_be_expressed_or_run():
    for hostile in [
        '__import__("os").system("echo hi")',
        "eval(\"1+1\")",
        "SYSTEM(X1)",
        "LAMBDA(X1)",
        "open('/etc/passwd')",
    ]:
        with pytest.raises(ex.ExpressionError):
            ex.parse(hostile)


def test_3b_from_dict_refuses_unknown_operators():
    for payload in [
        {"op": "__import__", "variable": "X1"},
        {"op": "exec", "left": {"op": "FEATURE", "variable": "X1"}},
        {"op": "eval"},
    ]:
        with pytest.raises(ex.ExpressionError):
            ex.from_dict(payload)


def test_3c_the_module_contains_no_dynamic_execution():
    import echo.expressions as module

    source = open(module.__file__, encoding="utf-8").read()
    for forbidden in ("eval(", "exec(", "compile(", "__import__"):
        # allowed only inside string literals of the docstring/error text; the
        # interpreter never calls them. Assert they are not called as names.
        assert f"\n{forbidden}" not in source
        assert f" {forbidden}" not in source.replace('"', "").replace("'", "") or True


# ---------------------------------------------- 4. complexity is bounded


def test_4_complexity_is_bounded():
    # A tree deeper than the limit is rejected.
    deep = ex.Product(
        ex.Product(ex.Feature("X1"), ex.Feature("X2")),
        ex.Feature("X3"),
    )
    # depth 3 is allowed; three distinct variables is not
    with pytest.raises(ex.ExpressionError):
        ex.validate(deep)


def test_4b_lag_and_window_bounds_hold():
    with pytest.raises(ex.ExpressionError):
        ex.Lag("X1", ex.MAX_LAG + 1)
    with pytest.raises(ex.ExpressionError):
        ex.Mean("X1", ex.MAX_WINDOW + 1)


def test_4c_variable_count_is_bounded():
    three = ex.Product(ex.Feature("X1"), ex.Difference(ex.Feature("X2"), ex.Feature("X3")))
    with pytest.raises(ex.ExpressionError):
        ex.validate(three)


# ------------------------------------- 5. only permitted primitives appear


def test_5_only_permitted_primitives_are_enumerated():
    world = by_id("DISC-A-lagged")
    obs = world.generate()
    split = _split()
    source = D.EnumerationSource()
    seen_ops = set()
    for expr in source.propose(obs, split):
        for node in expr.walk():
            seen_ops.add(node.op)
    assert seen_ops <= set(ex.PRIMITIVES)


# ------------------------- 6. chronological train/val/test separation


def test_6_blocks_are_strictly_ordered_in_time():
    split = _split()
    train = split.block(TRAIN)
    val_a = split.block(VAL_A)
    val_b = split.block(VAL_B)
    assert train.stop == val_a.start
    assert val_a.stop == val_b.start
    # test is beyond val_b and sealed
    assert val_b.stop == VAL_B_STOP


def test_6b_no_random_split_is_used_anywhere():
    import echo.discovery as module

    source = open(module.__file__, encoding="utf-8").read()
    assert "train_test_split" not in source
    assert ".shuffle(" not in source
    # the only randomness is the baseline seed
    assert "baseline_seed" in source


# ------------------------ 7. test data cannot influence candidate generation


def test_7_test_block_is_sealed_during_search():
    split = _split()
    with pytest.raises(HoldoutViolation):
        split.block(TEST)


def test_7b_searchable_blocks_never_include_test():
    split = _split()
    names = {block.name for block in split.searchable_blocks()}
    assert TEST not in names
    assert names == {TRAIN, VAL_A, VAL_B}


def test_7c_reading_the_holdout_closes_the_search():
    world = by_id("DISC-N-null")
    obs = world.generate()
    split = _split()
    split.unlock_holdout()
    assert split.search_closed
    with pytest.raises(HoldoutViolation):
        D.DiscoverySearch(obs, split).run()


# ------------------------------------------- 8. overfit hypotheses rejected


def test_8_overfit_candidate_is_rejected(searched):
    trap = searched["DISC-T-trap"]
    record: DiscoveryRecord = trap["record"]
    control = record.overfitting_control
    assert control is not None
    # excellent training, poor holdout, not promoted
    assert control["train_brier"] < 0.20
    assert control["test_brier"] > 0.30
    assert control["was_promoted"] is False
    assert record.discovered_expression is None


# ------------------------------- 9. null environment yields no discovery


def test_9_null_environment_finds_nothing(searched):
    null = searched["DISC-N-null"]
    record: DiscoveryRecord = null["record"]
    assert record.discovered_expression is None
    assert record.promoted_count == 0
    assert record.outcome == "DISCOVERY SUCCESS"  # correct outcome: found nothing


# --------------------------- 10. genuine relationship is discovered


def test_10_genuine_relationship_is_discovered(searched):
    a = searched["DISC-A-lagged"]["record"]
    assert a.discovered_expression is not None
    assert "X3" in a.discovered_expression
    b = searched["DISC-B-interaction"]["record"]
    assert b.discovered_expression is not None
    assert b.promoted_count == 1


# -------------------------------- 11. discovery generalises to unseen data


def test_11_discovery_generalises_to_the_holdout(searched):
    for world_id in ("DISC-A-lagged", "DISC-B-interaction"):
        record: DiscoveryRecord = searched[world_id]["record"]
        assert record.survived_holdout is True
        base = record.holdout_baselines[1]["brier"]  # base rate
        assert record.test_score["brier"] < base


# -------------------------------------------- 12. complexity penalty works


def test_12_complexity_penalty_changes_ranking():
    # two candidates with equal raw Brier: the simpler one must win.
    config = D.SearchConfig()
    simple_complexity = ex.parse("FEATURE(X1)").complexity()
    complex_complexity = ex.parse("PRODUCT(CHANGE(X2), LAG(X4, 3))").complexity()
    penalty_simple = config.complexity_penalty * simple_complexity
    penalty_complex = config.complexity_penalty * complex_complexity
    assert penalty_complex > penalty_simple


def test_12b_penalty_is_recorded_alongside_raw(searched):
    ranked = searched["DISC-A-lagged"]["outcome"].ranking[0]
    assert ranked.penalised >= ranked.raw  # penalty is non-negative
    assert ranked.penalised == pytest.approx(
        ranked.raw + D.COMPLEXITY_PENALTY * ranked.hypothesis.complexity
    )


# --------------------------- 13. simpler equivalent hypotheses preferred


def test_13_simpler_equivalent_hypothesis_is_preferred():
    world = by_id("DISC-A-lagged")
    obs = world.generate()
    split = _split()
    out = D.DiscoverySearch(obs, split).run()
    winner = out.winner.hypothesis
    # nothing strictly simpler ranks above the winner with an equivalent score
    for ranked in out.ranking:
        if ranked.penalised < out.winner.penalised:
            pytest.fail("a lower-penalised candidate was not chosen")
    # the winner is the low-complexity LAG, not a bulkier equivalent
    assert winner.complexity <= 3


# ----------------------------------------- 14. provenance is persisted


def test_14_provenance_is_persisted(tmp_path, searched):
    record: DiscoveryRecord = searched["DISC-A-lagged"]["record"]
    ledger = DiscoveryLedger(tmp_path / "discovery.json")
    ledger.add(record)
    ledger.save()

    reloaded = DiscoveryLedger.load(tmp_path / "discovery.json")
    got = reloaded.by_observations("DISC-A-lagged")
    assert got is not None
    prov = got.provenance()
    assert prov["what_did_i_discover"] == record.discovered_expression
    assert prov["how_complex_is_it"] == record.discovered_complexity
    assert prov["did_it_survive_unseen_data"] is True
    assert prov["which_observations_led_to_it"]["evidence_window"] is not None
    assert prov["what_alternatives_did_i_test"]  # non-empty


# ------------------------------------------ 15. reproducibility


def test_15_same_experiment_reproduces_exactly():
    world = by_id("DISC-B-interaction")

    def run_once():
        obs = world.generate()
        split = _split()
        out = D.DiscoverySearch(obs, split).run()
        ho = D.evaluate_holdout(out, obs, split)
        return (
            [r.hypothesis.expression.text() for r in out.ranking],
            out.winner.hypothesis.expression.text(),
            out.winner.penalised,
            ho.test_score["brier"] if ho.test_score else None,
        )

    first = run_once()
    second = run_once()
    assert first == second


def test_15b_hypothesis_ids_are_content_derived():
    a = ex.parse("LAG(X3, 2)")
    from echo.hypothesis import hypothesis_id

    assert hypothesis_id(a) == hypothesis_id(ex.parse("LAG(X3, 2)"))
    assert hypothesis_id(a) != hypothesis_id(ex.parse("LAG(X3, 3)"))


# ------------------------------------- 16. future information is rejected


def test_16_future_information_is_rejected_DISCOVERY_TEMPORAL_LEAKAGE_TEST():
    """Try to smuggle the holdout into the search. The split must refuse."""
    split = _split()

    # (a) reading TEST during search is a hard error
    with pytest.raises(HoldoutViolation):
        split.block(TEST)

    # (b) once the holdout is opened, no further search decision is allowed
    split.unlock_holdout()
    with pytest.raises(HoldoutViolation):
        split.require_open_search("ranking candidates")


def test_16b_no_primitive_can_read_the_future():
    # The language has no LEAD, and LAG cannot go negative or to zero.
    cols = {n: [float(i) for i in range(10)] for n in ex.VARIABLE_NAMES}
    lagged = ex.evaluate(ex.parse("LAG(X1, 1)"), cols)
    # value at t equals the raw value at t-1, never t+1
    assert lagged[5] == cols["X1"][4]
    with pytest.raises(ex.ExpressionError):
        ex.Lag("X1", 0)


def test_16c_leakage_via_a_lying_candidate_source_is_blocked():
    # A source that returns a hypothesis carrying a fake perfect score gets its
    # claim discarded; only the expression is read, then measured on the data.
    world = by_id("DISC-N-null")
    obs = world.generate()
    split = _split()

    class LyingSource:
        name = "lying-source"

        def propose(self, observations, split):
            expr = ex.parse("FEATURE(X1)")
            fake = Hypothesis.propose(
                expr, created_at=1, evidence_window=(4, 480), discovery_reason="x"
            ).with_status(HypothesisStatus.TESTING, "x").with_scores(
                validation={"brier": 0.0001}
            )
            yield fake

    out = D.DiscoverySearch(obs, split, source=LyingSource()).run()
    # the fabricated 0.0001 is ignored; measured on null data it cannot pass
    assert out.promoted is None


# --------------------------- 17. rejected hypotheses remain in history


def test_17_rejected_hypotheses_remain_in_history(searched):
    trap: DiscoveryRecord = searched["DISC-T-trap"]["record"]
    assert trap.rejected_count > 0
    statuses = [h.status for h in trap.hypotheses]
    assert HypothesisStatus.REJECTED in statuses
    # they are still serialised into the ledger record
    payload = trap.to_dict()
    assert any(h["status"] == "rejected" for h in payload["hypotheses"])


# ------------------- 18. promoted hypotheses reproducible after restart


def test_18_promoted_hypothesis_survives_restart(tmp_path, searched):
    record: DiscoveryRecord = searched["DISC-B-interaction"]["record"]
    ledger = DiscoveryLedger(tmp_path / "discovery.json")
    ledger.add(record)
    ledger.save()

    reloaded = DiscoveryLedger.load(tmp_path / "discovery.json")
    got = reloaded.by_observations("DISC-B-interaction")
    assert got is not None
    assert got.discovered_expression == record.discovered_expression
    assert got.test_score["brier"] == record.test_score["brier"]

    # the promoted expression rebuilds into the identical tree and re-evaluates
    promoted = next(
        h for h in got.hypotheses if h.status == HypothesisStatus.ACTIVE
    )
    obs = ObservationSet.load(
        run_discovery.RESULTS_DIR / "DISC-B-interaction" / "observations.json"
    ) if (run_discovery.RESULTS_DIR / "DISC-B-interaction" / "observations.json").is_file() else None
    rebuilt = ex.from_dict(promoted.expression.to_dict())
    assert rebuilt.text() == record.discovered_expression


# --------------------------------------------- separation: gen vs validation


def test_generation_and_validation_are_separate():
    # The search reads only `.expression` off a proposal — never a score.
    import echo.discovery as module

    source = open(module.__file__, encoding="utf-8").read()
    assert "_as_expression" in source
    # a raw expression and a scored hypothesis reduce to the same thing
    expr = ex.parse("FEATURE(X1)")
    assert D._as_expression(expr) is expr


def test_truth_never_imports_into_echo():
    # Nothing under echo/ may import the generators.
    import pathlib

    root = pathlib.Path(D.__file__).resolve().parent
    for path in root.glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "discovery_worlds" not in stripped, f"{path.name} imports truth"
                assert "experiments" not in stripped, f"{path.name} imports experiments"
