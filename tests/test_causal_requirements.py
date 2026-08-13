"""What ECHO 8 has to be true for: seeing is not doing, and neither is knowing why.

The load-bearing properties are that the Markov-equivalent worlds are *exactly*
indistinguishable observationally (so the problem is real rather than merely
hard), that intervention breaks the tie, that the posterior is allowed to stay
plural, and that a counterfactual answer never carries more confidence than the
models behind it support.
"""

from __future__ import annotations

import pytest

from echo.causal import CausalError, HypothesisSet
from echo.counterfactual import (
    STANDING_ASSUMPTIONS,
    CounterfactualQuery,
    counterfactual,
    counterfactual_for_model,
)
from echo.experiment import Intervention, appraise

from experiments import run_causal as R
from experiments.causal_worlds import (
    CHAIN,
    COLLIDER,
    EQUIVALENCE_CLASS,
    FORK,
    REVERSE_CHAIN,
    SCENARIOS,
    WORLDS,
    by_id,
    causal_menu,
)


@pytest.fixture(scope="module")
def flat():
    return HypothesisSet.uniform(WORLDS)


@pytest.fixture(scope="module")
def runs():
    return {s.id: R.run_scenario(s) for s in SCENARIOS}


# ------------------------------------------ 8.1/8.2 observational ambiguity


def test_1_the_equivalence_class_is_exactly_indistinguishable():
    """Not approximately. If watching could order them, there is no problem here."""
    reference = sorted(CHAIN.observed_joint().items())
    for model in EQUIVALENCE_CLASS[1:]:
        for (key_a, p_a), (key_b, p_b) in zip(reference, sorted(model.observed_joint().items())):
            assert key_a == key_b
            assert p_a == pytest.approx(p_b, abs=1e-15)


def test_1b_correlation_ranking_cannot_solve_it():
    """Every pairwise correlation is identical across the class."""
    for first, second in (("V1", "V2"), ("V2", "V3"), ("V1", "V3")):
        values = [m.correlation(first, second) for m in EQUIVALENCE_CLASS]
        for value in values[1:]:
            assert value == pytest.approx(values[0], abs=1e-15)


def test_1c_the_worlds_are_genuinely_different_structures():
    assert len({m.structure() for m in WORLDS}) == len(WORLDS)


def test_2_the_collider_is_observationally_separable():
    """Observation is insufficient, not useless — and the difference is identifiable."""
    assert COLLIDER.correlation("V1", "V3") == pytest.approx(0.0, abs=1e-12)
    for model in EQUIVALENCE_CLASS:
        assert model.correlation("V1", "V3") > 0.3
    reference = sorted(CHAIN.observed_joint().items())
    collider = sorted(COLLIDER.observed_joint().items())
    assert any(
        abs(a - b) > 1e-6 for (_, a), (_, b) in zip(reference, collider)
    )


def test_2b_watching_leaves_the_equivalence_class_at_exactly_one_third(runs):
    for scenario_id in ("CAUSAL-chain", "CAUSAL-fork", "CAUSAL-reverse"):
        run = runs[scenario_id]
        for model in EQUIVALENCE_CLASS:
            assert run.after_observation.probability(model.id) == pytest.approx(
                1 / 3, abs=1e-9
            )
        assert run.after_observation.probability("M4-collider") == pytest.approx(
            0.0, abs=1e-9
        )


# ------------------------------------------------- 8.3 intervention evidence


def test_3_intervention_changes_causal_beliefs(runs):
    for scenario_id in ("CAUSAL-chain", "CAUSAL-fork", "CAUSAL-reverse"):
        run = runs[scenario_id]
        assert run.truth_probability_observational == pytest.approx(1 / 3, abs=1e-9)
        assert run.truth_probability_final > 0.95
        assert run.identified


def test_3b_the_intervention_machinery_is_echo_sevens(flat):
    """Reused, not reimplemented."""
    import echo.experiment as experiment_module

    assert R.choose_experiment is experiment_module.choose_experiment
    assert R.appraise is experiment_module.appraise


def test_3c_setting_the_middle_variable_is_the_informative_one(flat):
    gains = {
        option.id: appraise(option, flat).expected_information_gain
        for option in causal_menu(12)
    }
    assert gains["DO-V2"] > gains["DO-V1"]
    assert gains["DO-V2"] > gains["DO-V3"]
    assert gains["DO-V2"] > gains["OBS"]


def test_3d_seeing_and_doing_come_apart():
    """The single fact the whole challenge rests on."""
    index = FORK.variables.index("V3")
    seen = sum(p for a, p in FORK.conditioned("V1", 1).items() if a[index] == 1)
    done = FORK.intervened({"V1": 1}).marginal("V3")
    assert seen > 0.6  # seeing V1 predicts V3
    assert done == pytest.approx(0.5, abs=1e-9)  # setting V1 does nothing to V3
    # ...whereas in the chain, setting V1 does move V3
    assert CHAIN.intervened({"V1": 1}).marginal("V3") > 0.6


# ---------------------------------------------------- 8.4 counterfactuals


def test_4_a_counterfactual_carries_probability_evidence_confidence_assumptions(runs):
    answer = runs["CAUSAL-chain"].answers[0]
    assert 0.0 <= answer.probability <= 1.0
    assert answer.supporting_evidence
    assert 0.0 <= answer.confidence <= 1.0
    assert answer.assumptions
    for assumption in STANDING_ASSUMPTIONS:
        assert assumption in answer.assumptions


def test_4b_confidence_collapses_when_models_disagree(flat):
    query = CounterfactualQuery(
        facts=(("V1", 1), ("V2", 1), ("V3", 1)),
        antecedent=("V1", 0),
        consequent=("V3", 1),
    )
    spread = counterfactual(flat, query)
    assert spread.model_disagreement > 0.1
    assert spread.confidence < 0.2

    settled = counterfactual(flat.with_posterior([0.97, 0.01, 0.01, 0.01]), query)
    assert settled.confidence > spread.confidence


def test_4c_no_certainty_is_claimed_that_the_data_cannot_support(flat):
    """A flat posterior must never yield a confident counterfactual."""
    for query in R.QUERIES:
        answer = counterfactual(flat, query)
        assert answer.confidence < 0.5


def test_4d_counterfactuals_differ_by_structure():
    """Same question, same facts, different worlds, different answers."""
    query = CounterfactualQuery(
        facts=(("V1", 1), ("V2", 1), ("V3", 1)),
        antecedent=("V1", 0),
        consequent=("V3", 1),
    )
    answers = {m.id: counterfactual_for_model(m, query) for m in EQUIVALENCE_CLASS}
    assert len(set(round(v, 6) for v in answers.values() if v is not None)) > 1
    # In the chain V1 is upstream of V3, so removing it matters...
    assert answers["M1-chain"] < 0.3
    # ...in the fork it is not, so it does not.
    assert answers["M2-fork"] > 0.7


def test_4e_per_model_answers_are_reported_not_only_the_average(flat):
    answer = counterfactual(flat, R.QUERIES[0])
    assert len(answer.per_model) == len(WORLDS)
    assert answer.spread > 0.5


def test_4f_impossible_facts_are_refused_rather_than_guessed():
    query = CounterfactualQuery(
        facts=(("V1", 1),), antecedent=("V1", 0), consequent=("V3", 1)
    )
    # a model in which the fact has zero probability contributes nothing
    impossible = CHAIN.intervened({"V1": 0})
    assert counterfactual_for_model(impossible, query) is None


# ------------------------------------------------ 8.5 uncertainty is kept


def test_5_multiple_causal_models_are_held_simultaneously(runs):
    run = runs["CAUSAL-starved"]
    posterior = dict(run.after_intervention.ranked())
    live = [p for p in posterior.values() if p > 0.01]
    assert len(live) >= 2
    assert run.after_intervention.entropy() > 0.3


def test_5b_nothing_forces_a_single_explanation(runs):
    """The observational posterior is a distribution, not a winner."""
    run = runs["CAUSAL-chain"]
    ranked = run.after_observation.ranked()
    top = ranked[0][1]
    assert top == pytest.approx(1 / 3, abs=1e-9)
    assert sum(p for _, p in ranked) == pytest.approx(1.0)


def test_5c_the_posterior_always_sums_to_one(runs):
    for run in runs.values():
        for stage in (run.prior, run.after_observation, run.after_intervention):
            assert sum(stage.posterior) == pytest.approx(1.0, abs=1e-9)


def test_5d_a_hypothesis_set_refuses_a_malformed_posterior():
    with pytest.raises(CausalError):
        HypothesisSet(WORLDS, (0.5, 0.5, 0.5, 0.5))


# ------------------------------------------------------ hygiene and safety


def test_no_truth_leaks_into_the_reasoning_modules():
    import pathlib

    import echo.causal as causal_module
    import echo.counterfactual as counterfactual_module
    import echo.experiment as experiment_module

    for module in (causal_module, counterfactual_module, experiment_module):
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        for line in source.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                assert "causal_worlds" not in stripped
                assert "experiments" not in stripped
        for model in WORLDS:
            assert model.id not in source


def test_the_study_is_reproducible():
    first = R.run_scenario(by_id("CAUSAL-fork"))
    second = R.run_scenario(by_id("CAUSAL-fork"))
    assert first.after_intervention.ranked() == second.after_intervention.ranked()
    assert [r.option_id for r in first.records] == [r.option_id for r in second.records]


def test_experiment_records_are_immutable(runs):
    record = runs["CAUSAL-chain"].records[0]
    with pytest.raises(Exception):
        record.cost = 0.0  # type: ignore[misc]


def test_intervening_on_an_unknown_variable_is_refused():
    with pytest.raises(CausalError):
        CHAIN.intervened({"V9": 1})
    with pytest.raises(CausalError):
        CHAIN.intervened({"V1": 7})


def test_counterfactual_on_an_unknown_variable_is_refused():
    query = CounterfactualQuery(
        facts=(("V1", 1),), antecedent=("V9", 0), consequent=("V3", 1)
    )
    with pytest.raises(CausalError):
        counterfactual_for_model(CHAIN, query)
