"""Unit tests for the prediction records, the timeline, and the environments."""

import pytest

from echo.calibration import accuracy, mean_log_loss, reference_brier, summarise
from echo.prediction import (
    EvaluationStatus,
    Observation,
    Prediction,
    PredictionRecord,
    TemporalLeakageError,
    Timeline,
    brier_score,
    issue_prediction,
    log_loss,
)
from experiments.environments import ENVIRONMENTS, OUTCOME_A, laplace_predictor

PROPOSITION = "The next outcome will be OUTCOME_A."


# ------------------------------------------------------------------ timeline


def test_observations_are_frozen():
    observation = Observation(content="x", at=1)
    with pytest.raises(Exception):
        observation.at = 5


def test_a_negative_tick_is_rejected():
    with pytest.raises(ValueError):
        Observation(content="x", at=-1)


def test_an_observation_cannot_be_recorded_twice():
    timeline = Timeline()
    observation = timeline.add("x", at=0)
    with pytest.raises(ValueError, match="already on the timeline"):
        timeline.record(observation)


def test_a_view_is_strictly_before_its_horizon():
    timeline = Timeline()
    for tick in range(5):
        timeline.add(f"t{tick}", at=tick)

    view = timeline.view_as_of(3)

    assert [o.at for o in view] == [0, 1, 2]
    assert view.get(timeline.all()[3].id) is None


def test_a_view_at_zero_is_empty_not_an_error():
    timeline = Timeline()
    timeline.add("t0", at=0)
    assert len(timeline.view_as_of(0)) == 0


def test_views_are_ordered_deterministically():
    timeline = Timeline()
    for tick in (4, 1, 3, 0, 2):
        timeline.add(f"t{tick}", at=tick)

    assert [o.at for o in timeline.view_as_of(9)] == [0, 1, 2, 3, 4]


def test_check_available_accepts_the_genuine_past():
    timeline = Timeline()
    past = timeline.add("t0", at=0)
    timeline.check_available([past.id], before=1)  # must not raise


def test_observation_round_trips_through_dict():
    observation = Observation(content="x", at=3, payload={"outcome": True})
    assert Observation.from_dict(observation.to_dict()).to_dict() == observation.to_dict()


# ---------------------------------------------------------------- prediction


def test_a_probability_outside_zero_and_one_is_rejected():
    timeline = Timeline()
    timeline.add("t0", at=0)
    for bad in (-0.1, 1.4):
        with pytest.raises(ValueError, match="predicted_probability"):
            issue_prediction(timeline, PROPOSITION, bad, prediction_time=1)


def test_an_empty_proposition_is_rejected():
    timeline = Timeline()
    with pytest.raises(ValueError):
        Prediction(
            proposition="  ",
            predicted_probability=0.5,
            prediction_time=0,
            information_available_at_prediction_time=(),
        )


def test_information_available_defaults_to_everything_before_the_tick():
    timeline = Timeline()
    for tick in range(5):
        timeline.add(f"t{tick}", at=tick)

    prediction = issue_prediction(timeline, PROPOSITION, 0.5, prediction_time=3)

    assert len(prediction.information_available_at_prediction_time) == 3


def test_stated_direction_does_not_replace_the_probability():
    """P(A)=0.51 and P(A)=0.99 read the same as a direction — that is the point."""
    timeline = Timeline()
    timeline.add("t0", at=0)

    hedged = issue_prediction(timeline, PROPOSITION, 0.51, prediction_time=1)
    confident = issue_prediction(timeline, PROPOSITION, 0.99, prediction_time=1)

    assert hedged.stated_direction == confident.stated_direction == "true"
    assert hedged.predicted_probability != confident.predicted_probability


def test_an_exact_half_is_undecided_rather_than_forced():
    timeline = Timeline()
    timeline.add("t0", at=0)
    prediction = issue_prediction(timeline, PROPOSITION, 0.5, prediction_time=1)

    assert prediction.stated_direction == "undecided"


def test_a_half_prediction_is_never_scored_as_correctly_anticipated():
    timeline = Timeline()
    timeline.add("t0", at=0)
    record = PredictionRecord(issue_prediction(timeline, PROPOSITION, 0.5, prediction_time=1))

    for outcome in (True, False):
        evaluated = PredictionRecord(record.prediction).evaluate(outcome, "x", 2)
        assert evaluated.evaluation.correctly_anticipated is False


def test_a_pending_record_reports_pending_and_no_outcome():
    timeline = Timeline()
    timeline.add("t0", at=0)
    record = PredictionRecord(issue_prediction(timeline, PROPOSITION, 0.7, prediction_time=1))

    assert record.evaluation_status is EvaluationStatus.PENDING
    assert record.actual_outcome is None
    assert record.prediction_error is None
    assert record.evaluation_time is None
    assert not record.is_evaluated


def test_evaluating_returns_a_new_record_and_leaves_the_old_one_pending():
    timeline = Timeline()
    timeline.add("t0", at=0)
    record = PredictionRecord(issue_prediction(timeline, PROPOSITION, 0.7, prediction_time=1))

    evaluated = record.evaluate(True, OUTCOME_A, 2)

    assert evaluated is not record
    assert record.evaluation_status is EvaluationStatus.PENDING  # the old view is untouched
    assert evaluated.evaluation_status is EvaluationStatus.EVALUATED
    assert evaluated.prediction is record.prediction  # same frozen prediction


def test_record_round_trips_through_dict():
    timeline = Timeline()
    timeline.add("t0", at=0)
    record = PredictionRecord(
        issue_prediction(timeline, PROPOSITION, 0.7, prediction_time=1)
    ).evaluate(True, OUTCOME_A, 2)

    assert PredictionRecord.from_dict(record.to_dict()).to_dict() == record.to_dict()


def test_the_serialised_shape_uses_the_required_field_names():
    timeline = Timeline()
    timeline.add("t0", at=0)
    record = PredictionRecord(issue_prediction(timeline, PROPOSITION, 0.7, prediction_time=1))
    payload = record.to_dict()

    for key in (
        "prediction_id",
        "proposition",
        "predicted_probability",
        "prediction_time",
        "information_available_at_prediction_time",
        "evidence_ids_used",
        "belief_id",
        "evaluation_status",
        "actual_outcome",
        "evaluation_time",
        "prediction_error",
    ):
        assert key in payload, key


def test_a_prediction_can_reference_a_belief():
    timeline = Timeline()
    timeline.add("t0", at=0)
    prediction = issue_prediction(
        timeline, PROPOSITION, 0.7, prediction_time=1, belief_id="BEL-123"
    )
    assert prediction.belief_id == "BEL-123"


# ------------------------------------------------------------------ scoring


def test_brier_and_log_loss_agree_with_their_definitions():
    import math

    assert brier_score(0.7, True) == pytest.approx(0.09)
    assert brier_score(0.7, False) == pytest.approx(0.49)
    assert log_loss(0.7, True) == pytest.approx(-math.log(0.7))
    assert log_loss(0.0, True) is None


def test_mean_log_loss_excludes_and_counts_the_undefined():
    loss, excluded = mean_log_loss([1.0, 0.5], [False, True])
    assert excluded == 1
    assert loss == pytest.approx(-__import__("math").log(0.5))


def test_mean_log_loss_is_none_when_everything_is_undefined():
    loss, excluded = mean_log_loss([1.0], [False])
    assert loss is None and excluded == 1


def test_accuracy_ignores_exact_halves():
    assert accuracy([0.5, 0.9], [True, True]) == pytest.approx(0.5)


def test_reference_brier_reports_both_baselines():
    references = reference_brier([True, True, True, False])
    assert references["always_half"] == pytest.approx(0.25)
    assert references["always_base_rate"] == pytest.approx(0.75 * 0.25)


def test_summarise_returns_everything_needed_to_read_a_result():
    summary = summarise([0.8, 0.2, 0.6], [True, False, True])
    for key in (
        "count",
        "brier",
        "log_loss",
        "accuracy",
        "base_rate",
        "reference_brier",
        "expected_calibration_error",
        "buckets",
    ):
        assert key in summary, key


def test_empty_input_yields_none_rather_than_zero():
    summary = summarise([], [])
    assert summary["brier"] is None
    assert summary["accuracy"] is None


# ------------------------------------------------------------- environments


def test_environments_are_deterministic():
    for environment in ENVIRONMENTS:
        first = [t.true_outcome for t in environment.generate()]
        second = [t.true_outcome for t in environment.generate()]
        assert first == second, environment.id


def test_the_changing_environment_actually_changes():
    changing = next(e for e in ENVIRONMENTS if e.id == "ENV-B-changing")
    trials = changing.generate()

    before = sum(1 for t in trials[:30] if t.true_outcome) / 30
    after = sum(1 for t in trials[30:] if t.true_outcome) / 30

    assert before > 0.65
    assert after < 0.40


def test_the_noisy_environment_corrupts_some_observations():
    noisy = next(e for e in ENVIRONMENTS if e.id == "ENV-C-noisy")
    trials = noisy.generate()

    corrupted = [t for t in trials if t.corrupted]
    assert 5 < len(corrupted) < len(trials) // 2 + 10
    # The clean environments corrupt nothing.
    stable = next(e for e in ENVIRONMENTS if e.id == "ENV-A-stable")
    assert not any(t.corrupted for t in stable.generate())


def test_the_predictor_says_half_when_it_knows_nothing():
    timeline = Timeline()
    probability, rationale = laplace_predictor(timeline.view_as_of(0))

    assert probability == pytest.approx(0.5)
    assert "0 prior observation" in rationale


def test_the_predictor_never_states_zero_or_one():
    timeline = Timeline()
    for tick in range(50):
        timeline.add(f"t{tick}", at=tick, outcome=True)

    probability, _ = laplace_predictor(timeline.view_as_of(50))

    assert 0.0 < probability < 1.0


def test_the_predictor_converges_toward_the_observed_rate():
    timeline = Timeline()
    for tick in range(100):
        timeline.add(f"t{tick}", at=tick, outcome=tick % 4 != 0)  # 75% True

    probability, _ = laplace_predictor(timeline.view_as_of(100))

    assert probability == pytest.approx(0.75, abs=0.02)


def test_the_predictor_explains_its_arithmetic():
    timeline = Timeline()
    timeline.add("t0", at=0, outcome=True)
    timeline.add("t1", at=1, outcome=False)

    _, rationale = laplace_predictor(timeline.view_as_of(2))

    assert "2 prior observation" in rationale
    assert "1 of 2" in rationale
