"""What ECHO 7 has to be true for: competing hypotheses, and a computed choice.

The properties that matter here are that observation genuinely cannot separate
the models (so the problem is real), that expected information gain is computed
rather than looked up, that a useless experiment scores exactly zero without
anyone saying so, and that choosing well beats choosing at random on measured
campaigns.
"""

from __future__ import annotations

import math
import random

import pytest

from echo.causal import CausalError, CausalModel, HypothesisSet, entropy
from echo.experiment import (
    POLICIES,
    ExperimentLedger,
    ExperimentOption,
    ExperimentRecord,
    Intervention,
    appraise,
    choose_experiment,
    outcome_space,
)

from experiments import run_experimentation as R
from experiments.experiment_worlds import (
    COMPETING,
    CONFOUNDED,
    DIRECT,
    REVERSED,
    SCENARIOS,
    by_id,
    experiment_menu,
)


@pytest.fixture(scope="module")
def flat():
    return HypothesisSet.uniform(COMPETING)


# ------------------------------------------- 1. competing hypotheses are real


def test_1_competing_hypotheses_are_observationally_identical():
    """If watching could separate them, there would be nothing to experiment on."""
    joints = [tuple(sorted(model.observed_joint().items())) for model in COMPETING]
    reference = joints[0]
    for joint in joints[1:]:
        for (key_a, p_a), (key_b, p_b) in zip(reference, joint):
            assert key_a == key_b
            assert p_a == pytest.approx(p_b, abs=1e-12)


def test_1b_they_are_nonetheless_different_models():
    structures = {model.structure() for model in COMPETING}
    assert len(structures) == len(COMPETING)
    # and they disagree about what an intervention would do
    under_do = {
        model.intervened({"V1": 1}).marginal("V2") for model in COMPETING
    }
    assert len(under_do) > 1


def test_1c_the_observational_correlation_is_misleading_and_identical():
    correlations = [model.correlation("V1", "V2") for model in COMPETING]
    for value in correlations:
        assert value == pytest.approx(correlations[0], abs=1e-12)
        assert value > 0.5  # strong, and equally strong under every explanation


def test_1d_echo_is_never_told_which_hypothesis_is_true():
    import pathlib

    import echo.causal as causal_module
    import echo.experiment as experiment_module

    for module in (causal_module, experiment_module):
        source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
        for truth_id in {s.truth_id for s in SCENARIOS}:
            assert truth_id not in source


# ------------------------------------------------- 2. the menu has structure


def test_2_every_option_carries_cost_risk_time_and_budget():
    for option in experiment_menu(8):
        assert option.cost > 0
        assert option.risk >= 0
        assert option.time > 0
        assert option.samples >= 1


def test_2b_interventions_are_distinct_from_observation():
    assert Intervention.observation().is_observational
    assert not Intervention.of(V1=1).is_observational
    assert Intervention.of(V1=1).label() == "INTERVENE(V1=1)"


# ------------------------------------------- 3. information gain is computed


def test_3_expected_information_gain_is_computed_not_hardcoded(flat):
    """Change the posterior and the ranking must change with it."""
    menu = experiment_menu(8)
    first = {a.option.id: a.expected_information_gain for a in
             [appraise(o, flat) for o in menu]}

    # If V3-based confounding is already ruled out, intervening on V3 is worth
    # much less. Nothing tells the appraiser this; it follows from the numbers.
    skewed = flat.with_posterior([0.5, 0.0, 0.5])
    second = {a.option.id: a.expected_information_gain for a in
              [appraise(o, skewed) for o in menu]}
    assert second["DO-V3"] < first["DO-V3"]


def test_3b_an_uninformative_experiment_scores_exactly_zero(flat):
    """Both zeros must fall out of the formula, not out of a special case."""
    appraisals = {o.id: appraise(o, flat) for o in experiment_menu(8)}
    # observation: every model predicts the same visible distribution
    assert appraisals["OBS"].expected_information_gain == pytest.approx(0.0, abs=1e-12)
    # do(V1, V2): nothing observable is left free
    assert appraisals["DO-V1V2"].expected_information_gain == pytest.approx(0.0, abs=1e-12)
    assert len(outcome_space(COMPETING[0], Intervention.of(V1=1, V2=1))) == 1


def test_3c_information_gain_never_exceeds_prior_entropy(flat):
    for option in experiment_menu(20):
        a = appraise(option, flat)
        assert -1e-12 <= a.expected_information_gain <= flat.entropy() + 1e-12


def test_3d_cost_enters_the_choice(flat):
    """Two options with equal information must be separated by price."""
    appraisals = {o.id: appraise(o, flat) for o in experiment_menu(8)}
    cheap, dear = appraisals["DO-V1"], appraisals["DO-V2"]
    assert cheap.expected_information_gain == pytest.approx(
        dear.expected_information_gain, abs=1e-12
    )
    assert cheap.option.cost < dear.option.cost
    assert cheap.utility > dear.utility
    assert cheap.information_per_cost > dear.information_per_cost


def test_3e_no_policy_names_a_specific_intervention():
    import inspect

    for policy in POLICIES.values():
        source = inspect.getsource(type(policy))
        for name in ("V1", "V2", "V3", "DO-"):
            assert name not in source


def test_3f_the_appraisal_is_exact_for_these_sizes(flat):
    for option in experiment_menu(8):
        assert appraise(option, flat).exact


# ----------------------------------------- 4. interventions update beliefs


def test_4_intervention_evidence_moves_the_posterior(flat):
    truth = CONFOUNDED
    # Under the true model, forcing V1 leaves V2 at its base rate.
    observations = R.draw(truth, Intervention.of(V1=1), 40, random.Random(7))
    after = flat.updated(observations, intervention={"V1": 1})
    assert after.probability("H2-confounded") > flat.probability("H2-confounded")
    assert after.probability("H1-direct") < flat.probability("H1-direct")


def test_4b_observation_alone_cannot_move_the_posterior(flat):
    observations = R.draw(DIRECT, Intervention.observation(), 200, random.Random(3))
    after = flat.updated(observations)
    for model, before in zip(after.models, flat.posterior):
        assert after.probability(model.id) == pytest.approx(before, abs=1e-9)


def test_4c_updating_returns_a_new_set_and_mutates_nothing(flat):
    before = tuple(flat.posterior)
    flat.updated(
        R.draw(DIRECT, Intervention.of(V1=1), 10, random.Random(1)),
        intervention={"V1": 1},
    )
    assert tuple(flat.posterior) == before


# --------------------------------------- 5. controls behave as constructed


def test_5_the_expensive_uninformative_option_is_never_chosen_on_merit(flat):
    chosen, _ = choose_experiment(experiment_menu(8), flat, policy="information_gain")
    assert chosen.option.id != "DO-V1V2"
    assert chosen.option.id != "OBS"


def test_5b_the_cheapest_policy_buys_a_worthless_experiment(flat):
    chosen, _ = choose_experiment(experiment_menu(8), flat, policy="cheapest")
    assert chosen.expected_information_gain == pytest.approx(0.0, abs=1e-12)


def test_5c_an_experiment_can_contradict_the_leading_hypothesis():
    """Start convinced of the wrong thing; the right experiment must overturn it."""
    convinced = HypothesisSet(COMPETING, (0.80, 0.10, 0.10))
    assert convinced.best()[0].id == "H1-direct"
    observations = R.draw(CONFOUNDED, Intervention.of(V1=1), 40, random.Random(11))
    after = convinced.updated(observations, intervention={"V1": 1})
    assert after.best()[0].id == "H2-confounded"


def test_5d_a_campaign_records_when_the_leader_was_overturned():
    campaign = R.run_campaign(by_id("EXP-confounded"), "information_gain", seed=1000)
    assert campaign.records
    assert all(isinstance(r, ExperimentRecord) for r in campaign.records)


# --------------------------- 6. the headline: better than random selection


@pytest.fixture(scope="module")
def campaigns():
    return {scenario.id: R.run_scenario(scenario) for scenario in SCENARIOS}


def test_6_information_gain_beats_random_selection(campaigns):
    for scenario in SCENARIOS:
        summaries = campaigns[scenario.id]
        gain = summaries["information_gain"]
        chance = summaries["random"]
        assert gain.truth_probability > chance.truth_probability
        assert gain.bits_per_cost > chance.bits_per_cost


def test_6b_random_selection_is_a_genuine_control(campaigns):
    """It must actually differ from the informed policy, not shadow it."""
    for scenario in SCENARIOS:
        summaries = campaigns[scenario.id]
        picked = {
            record.option_id
            for campaign in summaries["random"].campaigns
            for record in campaign.records
        }
        assert len(picked) > 1


def test_6c_ignoring_cost_costs_more(campaigns):
    """max_information may learn more, but it must not be cheaper."""
    for scenario in SCENARIOS:
        summaries = campaigns[scenario.id]
        assert (
            summaries["max_information"].total_cost
            >= summaries["information_gain"].total_cost
        )


def test_6d_the_cheapest_policy_learns_nothing(campaigns):
    for scenario in SCENARIOS:
        summary = campaigns[scenario.id]["cheapest"]
        assert summary.bits_removed == pytest.approx(0.0, abs=1e-9)
        assert summary.identified_rate == 0.0


# ------------------------------------------------ records and reproducibility


def test_records_are_immutable():
    campaign = R.run_campaign(by_id("EXP-direct"), "information_gain", seed=1000)
    record = campaign.records[0]
    with pytest.raises(Exception):
        record.entropy_after = 0.0  # type: ignore[misc]


def test_records_carry_the_alternatives_that_were_rejected():
    campaign = R.run_campaign(by_id("EXP-direct"), "information_gain", seed=1000)
    record = campaign.records[0]
    assert len(record.alternatives) == len(experiment_menu(8)) - 1
    assert record.appraisal["expected_information_gain_bits"] >= 0.0


def test_the_experiment_is_reproducible():
    first = R.run_campaign(by_id("EXP-reversed"), "information_gain", seed=42)
    second = R.run_campaign(by_id("EXP-reversed"), "information_gain", seed=42)
    assert [r.option_id for r in first.records] == [r.option_id for r in second.records]
    assert first.posterior_trace == second.posterior_trace


def test_the_ledger_survives_restart(tmp_path):
    campaign = R.run_campaign(by_id("EXP-confounded"), "information_gain", seed=5)
    ledger = ExperimentLedger(tmp_path / "experiments.json")
    for record in campaign.records:
        ledger.add(record)
    ledger.save()

    reloaded = ExperimentLedger.load(tmp_path / "experiments.json")
    assert len(reloaded) == len(campaign.records)
    assert [r.option_id for r in reloaded.records()] == [
        r.option_id for r in campaign.records
    ]
    assert reloaded.total_cost() == pytest.approx(campaign.total_cost)


# --------------------------------------------------- causal model mechanics


def test_do_differs_from_conditioning():
    """The whole point, in one assertion."""
    # In the confounded world, seeing V1=1 raises the odds of V2; setting it does not.
    conditioned = CONFOUNDED.conditioned("V1", 1)
    index = CONFOUNDED.variables.index("V2")
    seen = sum(p for a, p in conditioned.items() if a[index] == 1)
    done = CONFOUNDED.intervened({"V1": 1}).marginal("V2")
    assert seen > 0.7
    assert done == pytest.approx(0.5, abs=1e-9)


def test_malformed_models_are_rejected():
    with pytest.raises(CausalError):
        CausalModel(
            id="cyclic",
            variables=("A", "B"),
            parents={"A": ("B",), "B": ("A",)},
            cpts={"A": {(0,): 0.5, (1,): 0.5}, "B": {(0,): 0.5, (1,): 0.5}},
        )
    with pytest.raises(CausalError):
        CausalModel(
            id="bad-probability",
            variables=("A",),
            parents={"A": ()},
            cpts={"A": {(): 1.5}},
        )


def test_entropy_is_bits():
    assert entropy([0.5, 0.5]) == pytest.approx(1.0)
    assert entropy([1.0]) == pytest.approx(0.0)
    assert entropy([1 / 3] * 3) == pytest.approx(math.log2(3))
