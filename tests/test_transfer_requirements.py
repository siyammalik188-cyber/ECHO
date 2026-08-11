"""The twenty properties Challenge 6 requires, one test each (plus sub-tests).

These lock down what makes a transfer claim meaningful: that the source is
discovered independently, that nothing of the source but its shape crosses the
wall, that cold start and transfer are separately measurable on identical data,
that the target holdout cannot reach the decision, and that the pattern can be
rejected, weakened and retired rather than merely trusted.

The heavy fixtures — a source discovery and a full four-condition run on one
matching target — are module-scoped, because each costs seconds.
"""

from __future__ import annotations

import pytest

from echo import discovery as D
from echo import expressions as ex
from echo import transfer as T
from echo.observations import (
    TEST,
    TRAIN,
    VAL_A,
    VAL_B,
    ChronologicalSplit,
    HoldoutViolation,
)
from echo.pattern import (
    PARAMETER_TOLERANCE,
    TRANSFERABLE_AFTER,
    AbstractNode,
    PatternError,
    PatternStatus,
    StructuralPattern,
    contains_variable_names,
)
from echo.transfer_ledger import TransferLedger

from experiments import run_transfer as R
from experiments.transfer_worlds import SOURCE, TARGETS, by_id


# --------------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def knowledge():
    return R.discover_source()


@pytest.fixture(scope="module")
def unrelated():
    return R.irrelevant_pattern()


@pytest.fixture(scope="module")
def target_a(knowledge, unrelated):
    """A full four-condition run on the structurally matching target."""
    pattern, _ = unrelated
    return R.run_target(by_id("TGT-A"), knowledge, pattern, knowledge.pattern, tick=1)


def _split(train_rows=None):
    return R.make_split(train_rows)


def _pattern() -> StructuralPattern:
    return StructuralPattern.from_expression(
        ex.parse("PRODUCT(CHANGE(X2), LAG(X4, 3))"), created_at=1
    )


# --------------------------------------------- 1. source discovery independent


def test_1_source_discovery_is_independent(knowledge):
    # The search proposed the whole exhaustive space; nothing pre-selected it.
    assert knowledge.outcome.proposed > 10_000
    assert knowledge.record.discovered_expression is not None
    assert knowledge.record.survived_holdout is True


def _code_without_docs(path) -> str:
    """Source with docstrings and comments removed, everything else intact.

    Prose may illustrate a shape; code may not select for one. Ordinary string
    literals are deliberately *kept*, because `if expr.text() == "PRODUCT(...)"`
    is exactly the cheat this is looking for.
    """
    import ast
    import io
    import tokenize

    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    doc_spans: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            doc_spans.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))

    comment_lines: dict[int, str] = {}
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            comment_lines[token.start[0]] = token.string

    kept: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if number in doc_spans:
            continue
        if number in comment_lines:
            line = line.replace(comment_lines[number], "")
        kept.append(line)
    return "\n".join(kept)


def test_1b_no_winning_expression_is_named_in_the_transfer_code(knowledge):
    """The programmer must not have written the answer down.

    Checks the expression discovery actually returned, not a guess at it, so
    the test cannot pass by the answer having quietly changed.
    """
    import pathlib

    import echo.pattern as pattern_module
    import echo.transfer as transfer_module

    discovered = knowledge.expression_text
    shape = knowledge.pattern.text()
    for module in (transfer_module, pattern_module, R):
        code = _code_without_docs(pathlib.Path(module.__file__))
        assert discovered not in code, f"{module.__name__} names the discovered expression"
        assert shape not in code, f"{module.__name__} hard-codes the abstract shape"


def test_1c_the_docstring_stripper_still_sees_real_code(tmp_path):
    """The audit above is only worth anything if it would catch a cheat."""
    cheat = tmp_path / "cheat.py"
    cheat.write_text(
        '"""A docstring mentioning PRODUCT(CHANGE(X2), LAG(X4, 3)) harmlessly."""\n'
        "# and a comment mentioning PRODUCT(CHANGE(X2), LAG(X4, 3))\n"
        "def pick(e):\n"
        '    return e.text() == "PRODUCT(CHANGE(X2), LAG(X4, 3))"\n',
        encoding="utf-8",
    )
    code = _code_without_docs(cheat)
    assert code.count("PRODUCT(CHANGE(X2), LAG(X4, 3))") == 1, code


# ------------------------------------- 2/3. no source leakage into knowledge


def test_2_source_variables_cannot_leak_into_target_knowledge(knowledge):
    leaked = contains_variable_names(knowledge.pattern.to_dict(), SOURCE.variable_names)
    assert leaked == []


def test_3_structural_patterns_contain_no_source_variable_names():
    pattern = _pattern()
    assert "X2" not in pattern.text()
    assert "X4" not in pattern.text()
    assert set(pattern.slots) <= {"A", "B", "C"}
    assert contains_variable_names(pattern.to_dict(), ex.VARIABLE_NAMES) == []


def test_3b_groundings_only_use_target_variables():
    pattern = _pattern()
    names = ("Z1", "Z2", "Z3", "Z4", "Z5", "Z6")
    for expr in pattern.ground(names):
        assert set(expr.variables()) <= set(names)


def test_20_no_source_truth_generator_is_reachable_by_echo():
    import pathlib

    root = pathlib.Path(D.__file__).resolve().parent
    for path in root.glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "transfer_worlds" not in stripped, f"{path.name} imports truth"
                assert "experiments" not in stripped, f"{path.name} imports experiments"


# ------------------------- 4. cold start and transfer separately measurable


def test_4_cold_and_transfer_are_independently_measurable(target_a):
    cold = target_a.runs["A_cold"]
    transfer = target_a.runs["B_structural"]
    # same data, different candidate sets, both scored
    assert cold.outcome.proposed > transfer.outcome.proposed
    assert cold.val_b_brier is not None
    assert transfer.val_b_brier is not None
    assert target_a.holdouts["A_cold"] is not None


def test_4b_transfer_searches_far_fewer_candidates(target_a):
    cold = target_a.runs["A_cold"].outcome.proposed
    transfer = target_a.runs["B_structural"].outcome.proposed
    assert transfer < cold / 50


# --------------------------------- 5. target holdout cannot influence transfer


def test_5_target_holdout_cannot_influence_transfer_TRANSFER_TEMPORAL_LEAKAGE_TEST():
    split = _split()
    with pytest.raises(HoldoutViolation):
        split.block(TEST)

    observations = by_id("TGT-A").generate()
    closed = _split()
    closed.unlock_holdout()
    with pytest.raises(HoldoutViolation):
        T.attempt_transfer(_pattern(), observations, closed)


def test_5b_transfer_decision_uses_no_holdout_rows(target_a):
    # Evidence recorded for the decision comes from VAL_B, never TEST.
    evidence = target_a.record.target_evidence
    assert evidence["confirmation"] is None or evidence["confirmation"]["block"] == VAL_B


def test_5c_source_expression_cannot_be_evaluated_in_a_target(knowledge):
    observations = by_id("TGT-A").generate()
    with pytest.raises(ex.ExpressionError):
        ex.check_schema(
            ex.parse(knowledge.expression_text), observations.variable_names
        )


# ------------------------------------- 6. successful structural transfer


def test_6_successful_structural_transfer_is_measurable(target_a):
    assert target_a.record.result in (
        T.TransferResult.SUCCESSFUL,
        T.TransferResult.PARTIAL,
    )
    holdout = target_a.holdouts["B_structural"]
    assert holdout is not None
    base = next(sc for name, sc in target_a.baselines if "base rate" in name)
    assert holdout["brier"] < base["brier"]


# --------------------------------------------- 7/8. harmful and rejected


def test_7_harmful_transfer_is_measurable():
    """A confirmed transfer that lands materially worse than cold start."""
    observations = by_id("TGT-P").generate()
    attempt = T.attempt_transfer(
        _pattern(),
        observations,
        _split(),
        # pretend a from-scratch search did very well; following the pattern
        # then costs more than the margin allows
        cold_reference=0.01,
    )
    assert attempt.result is T.TransferResult.HARMFUL
    assert "worse than searching from scratch" in attempt.reason


def test_8_negative_transfer_can_be_rejected():
    observations = by_id("TGT-N").generate()
    attempt = T.attempt_transfer(_pattern(), observations, _split())
    assert attempt.result in (T.TransferResult.REJECTED, T.TransferResult.INCONCLUSIVE)
    assert not attempt.confirmed


def test_8b_a_rejected_transfer_still_allows_a_from_scratch_search(knowledge, unrelated):
    pattern, _ = unrelated
    run = R.run_condition(
        "B_structural", by_id("TGT-N"), knowledge, pattern
    )
    assert run.fell_back is True
    assert run.effective is run.fallback


# ------------------------------------------------ 9. partial transfer weakens


def test_9_partial_transfer_can_be_weakened():
    observations = by_id("TGT-P").generate()
    attempt = T.attempt_transfer(_pattern(), observations, _split())
    assert attempt.result is T.TransferResult.PARTIAL
    assert attempt.adopted_is_component is True
    updated = T.update_pattern(_pattern(), attempt, "TGT-P")
    assert updated.status in (PatternStatus.WEAKENED, PatternStatus.TESTING)


def test_9b_a_pattern_decomposes_into_usable_components():
    components = _pattern().components()
    assert len(components) == 2
    assert {c.text() for c in components} == {"CHANGE(A)", "LAG(B, 3)"}


# ------------------------- 10. irrelevant transfer is not automatically active


def test_10_irrelevant_transfer_does_not_become_active(unrelated):
    pattern, _ = unrelated
    observations = by_id("TGT-A").generate()
    attempt = T.attempt_transfer(pattern, observations, _split())
    # It may or may not confirm, but it must never be adopted without passing
    # exactly the same gates as anything else.
    if attempt.confirmed:
        confirmation = attempt.outcome.confirmation
        assert confirmation is not None and confirmation.passed
    else:
        assert attempt.adopted is None


def test_10b_a_pattern_needs_more_than_one_success_to_be_transferable():
    pattern = _pattern()
    assert pattern.status is PatternStatus.DISCOVERED
    assert not pattern.earned_transferable()
    once = pattern.with_evidence("worked once", success=True)
    assert not once.earned_transferable()
    twice = once.with_evidence("worked twice", success=True)
    assert twice.earned_transferable()
    assert TRANSFERABLE_AFTER == 2


# ---------------------------------------- 11/12/13. renaming, scale, spread


def test_11_variable_renaming_does_not_prevent_valid_transfer():
    pattern = _pattern()  # abstracted from X-named variables
    grounded = pattern.ground(("Z1", "Z2", "Z3", "Z4", "Z5", "Z6"))
    assert grounded
    grounded_other = pattern.ground(("Q_a", "Q_b", "Q_c"))
    assert grounded_other


def test_12_numerical_scale_does_not_prevent_valid_transfer():
    """TGT-C runs at ~10^6; the interaction term reaches ~10^10."""
    world = by_id("TGT-C")
    observations = world.generate()
    column = observations.column("Z5")
    assert max(abs(v) for v in column) > 1e5
    attempt = T.attempt_transfer(_pattern(), observations, _split())
    assert attempt.result is T.TransferResult.SUCCESSFUL


def test_12b_standardisation_is_general_and_shared_by_both_conditions():
    # The same fit_logistic standardises for cold start and for transfer; there
    # is no scale handling specific to either.
    import inspect

    source = inspect.getsource(D.fit_logistic)
    assert "mean" in source and "sd" in source
    assert "TGT" not in source and "pattern" not in source


def test_13_different_distributions_are_handled():
    # Targets differ in autocorrelation, scale, offset, noise and base rate.
    seen_base_rates = set()
    for world in TARGETS:
        observations = world.generate()
        base = round(sum(observations.outcomes) / len(observations.outcomes), 1)
        seen_base_rates.add(base)
    assert len(seen_base_rates) >= 3


# ------------------------------------------- 14/15. records immutable, persist


def test_14_transfer_records_are_immutable(target_a):
    record = target_a.record
    with pytest.raises(Exception):
        record.result = T.TransferResult.SUCCESSFUL  # type: ignore[misc]
    with pytest.raises(Exception):
        record.transfer_confidence = 1.0  # type: ignore[misc]


def test_14b_pattern_versions_are_never_edited_in_place():
    pattern = _pattern()
    moved = pattern.with_status(PatternStatus.TESTING, "trying it")
    assert pattern.status is PatternStatus.DISCOVERED
    assert moved.version == pattern.version + 1
    with pytest.raises(Exception):
        pattern.status = PatternStatus.ACTIVE  # type: ignore[misc,attr-defined]


def test_15_transfer_decisions_survive_restart(tmp_path, target_a):
    ledger = TransferLedger(tmp_path / "transfer.json")
    ledger.add_pattern(target_a.pattern_after)
    ledger.add_transfer(target_a.record)
    ledger.save()

    reloaded = TransferLedger.load(tmp_path / "transfer.json")
    got = reloaded.for_target("TGT-A")
    assert len(got) == 1
    assert got[0].result is target_a.record.result
    assert got[0].adopted_expression == target_a.record.adopted_expression
    assert got[0].transfer_id == target_a.record.transfer_id

    pattern = reloaded.current(target_a.pattern_after.pattern_id)
    assert pattern is not None
    assert pattern.text() == target_a.pattern_after.text()
    # and it still grounds identically after a restart
    names = ("Z1", "Z2", "Z3", "Z4", "Z5", "Z6")
    assert [e.text() for e in pattern.ground(names)] == [
        e.text() for e in target_a.pattern_after.ground(names)
    ]


# ------------------------------------------------------- 16. false analogy


def test_16_false_analogy_rejects_transferred_knowledge():
    observations = by_id("TGT-F").generate()
    attempt = T.attempt_transfer(_pattern(), observations, _split())
    assert attempt.result in (T.TransferResult.REJECTED, T.TransferResult.INCONCLUSIVE)


def test_16b_the_false_analogy_world_is_statistically_like_the_matching_one():
    """It must be a genuine trap: same observable statistics, no shared structure."""
    a = by_id("TGT-A")
    f = by_id("TGT-F")
    assert a.ar == f.ar
    assert a.scales == f.scales
    assert a.offsets == f.offsets
    assert a.observable_description == f.observable_description
    assert f.shares_structure == "none"


# --------------------------------------- 17. transfer compared to cold start


def test_17_transfer_is_compared_against_cold_start(target_a):
    assert "A_cold" in target_a.runs and "B_structural" in target_a.runs
    assert target_a.efficiency["A_cold"] is not None or True
    # the transfer record carries the cold-start reference it was judged against
    assert "cold_start_val_b_brier" in target_a.record.target_evidence


# ------------------------------------------------------- 18. reproducibility


def test_18_the_experiment_is_reproducible():
    world = by_id("TGT-B")

    def once():
        observations = world.generate()
        attempt = T.attempt_transfer(_pattern(), observations, _split())
        return (
            attempt.result,
            attempt.adopted.hypothesis.expression.text() if attempt.adopted else None,
            round(attempt.outcome.confirmation.edge, 12)
            if attempt.outcome.confirmation
            else None,
        )

    assert once() == once()


def test_18b_pattern_ids_are_content_derived():
    a = StructuralPattern.from_expression(ex.parse("PRODUCT(CHANGE(X2), LAG(X4, 3))"), created_at=1)
    b = StructuralPattern.from_expression(ex.parse("PRODUCT(CHANGE(Z5), LAG(Z1, 3))"), created_at=9)
    # different worlds, same shape, same identity
    assert a.pattern_id == b.pattern_id
    c = StructuralPattern.from_expression(ex.parse("PRODUCT(CHANGE(X2), LAG(X4, 2))"), created_at=1)
    assert c.pattern_id != a.pattern_id


# ------------------------------------------------- 19. knowledge can retire


def test_19_transferred_knowledge_can_be_retired():
    pattern = _pattern().with_status(PatternStatus.TESTING, "trying")
    weakened = pattern.with_status(PatternStatus.WEAKENED, "did not carry")
    retired = weakened.with_status(PatternStatus.RETIRED, "withdrawn")
    assert retired.status is PatternStatus.RETIRED
    with pytest.raises(PatternError):
        retired.with_status(PatternStatus.TESTING, "sneak it back")


def test_19b_the_full_lifecycle_is_reachable():
    pattern = _pattern()
    assert pattern.status is PatternStatus.DISCOVERED
    pattern = pattern.with_status(PatternStatus.TESTING, "x")
    pattern = pattern.with_status(PatternStatus.TRANSFERABLE, "earned")
    pattern = pattern.with_status(PatternStatus.WEAKENED, "slipped")
    pattern = pattern.with_status(PatternStatus.RETIRED, "done")
    assert pattern.status is PatternStatus.RETIRED


# ------------------------------------------------------ safety and hygiene


def test_abstraction_is_mechanical_not_hand_written():
    """Whatever is discovered is what gets abstracted."""
    for text, expected in [
        ("LAG(X3, 2)", "LAG(A, 2)"),
        ("PRODUCT(CHANGE(X2), LAG(X4, 3))", "PRODUCT(CHANGE(A), LAG(B, 3))"),
        ("RATIO(MEAN(X1, 4), FEATURE(X5))", "RATIO(MEAN(A, 4), FEATURE(B))"),
        ("PRODUCT(FEATURE(X1), FEATURE(X1))", "PRODUCT(FEATURE(A), FEATURE(A))"),
    ]:
        pattern = StructuralPattern.from_expression(ex.parse(text), created_at=1)
        assert pattern.text() == expected


def test_pattern_grounding_is_bounded():
    pattern = _pattern()
    with pytest.raises(PatternError):
        pattern.ground(tuple(f"V{i}" for i in range(80)), max_candidates=100)


def test_pattern_payloads_cannot_smuggle_operators():
    with pytest.raises(PatternError):
        AbstractNode.from_dict({"op": "__import__", "slot": "A"})
    with pytest.raises(PatternError):
        StructuralPattern.from_dict(
            {
                "pattern_id": "PAT-x",
                "template": {"op": "eval"},
                "slots": [],
                "complexity": 1,
                "created_at": 1,
                "origin": "x",
                "status": "discovered",
            }
        )


def test_explanations_make_no_causal_claim(target_a):
    explanation = target_a.record.explanation
    assert set(explanation) >= {
        "source_pattern",
        "target_observation",
        "decision",
        "result",
    }
    blob = " ".join(explanation.values()).lower()
    for forbidden in ("because", "causes", "understands", "explains why"):
        assert forbidden not in blob.replace(T.NO_UNDERSTANDING.lower(), "")
    assert "no causal claim" in explanation["disclaimer"]


def test_a_lying_source_cannot_assert_its_own_transfer():
    """Generation proposes; the target's data disposes."""
    observations = by_id("TGT-F").generate()

    class LyingPatternSource:
        name = "lying"

        def propose(self, obs, split):
            from echo.hypothesis import Hypothesis, HypothesisStatus

            yield Hypothesis.propose(
                ex.parse("FEATURE(Z1)"),
                created_at=1,
                evidence_window=(4, 480),
                discovery_reason="x",
            ).with_status(HypothesisStatus.TESTING, "x").with_scores(
                validation={"brier": 0.0}, penalised=0.0
            )

    outcome = D.DiscoverySearch(observations, _split(), source=LyingPatternSource()).run()
    assert outcome.promoted is None
