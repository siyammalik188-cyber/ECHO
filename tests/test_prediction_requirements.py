"""The twelve properties prediction was asked to demonstrate.

Written to be read as evidence. Each test maps to one numbered requirement.
Everything runs offline — no model, no API key, no network.
"""

import math

import pytest

from echo.calibration import brier, calibration_buckets, expected_calibration_error
from echo.prediction import (
    EvaluationStatus,
    Experience,
    Prediction,
    PredictionRecord,
    TemporalLeakageError,
    Timeline,
    issue_prediction,
)
from echo.prediction_ledger import PredictionLedger

PROPOSITION = "The next outcome will be OUTCOME_A."


def a_timeline() -> Timeline:
    timeline = Timeline()
    timeline.add("Trial 0 was A.", at=0, outcome=True)
    timeline.add("Trial 1 was B.", at=1, outcome=False)
    timeline.add("Trial 2 was A.", at=2, outcome=True)
    return timeline


# 1 ------------------------------------------------- predictions are immutable


def test_1_predictions_are_immutable():
    timeline = a_timeline()
    prediction = issue_prediction(timeline, PROPOSITION, 0.70, prediction_time=3)

    for field, value in (
        ("predicted_probability", 0.40),
        ("proposition", "something else"),
        ("prediction_time", 99),
        ("id", "PRED-forged"),
    ):
        with pytest.raises(Exception):
            setattr(prediction, field, value)

    assert prediction.predicted_probability == 0.70


def test_1b_a_prediction_has_no_outcome_fields_to_overwrite():
    """Structural: there is nowhere on a Prediction for a result to be written."""
    timeline = a_timeline()
    prediction = issue_prediction(timeline, PROPOSITION, 0.70, prediction_time=3)

    for absent in ("actual_outcome", "evaluation_time", "prediction_error", "evaluation"):
        assert not hasattr(prediction, absent), absent


def test_1c_the_information_lists_are_tuples_not_mutable_lists():
    timeline = a_timeline()
    prediction = issue_prediction(timeline, PROPOSITION, 0.70, prediction_time=3)

    assert isinstance(prediction.information_available_at_prediction_time, tuple)
    assert isinstance(prediction.evidence_ids_used, tuple)


# 2 --------------------------------------------- timestamps are preserved


def test_2_prediction_timestamps_are_preserved(tmp_path):
    timeline = a_timeline()
    ledger = PredictionLedger.in_directory(tmp_path)
    prediction = issue_prediction(timeline, PROPOSITION, 0.70, prediction_time=3)
    ledger.add(prediction)
    issued_at, tick = prediction.issued_at, prediction.prediction_time

    timeline.add("Trial 3 was B.", at=3, outcome=False)
    ledger.evaluate(prediction.id, outcome=False, outcome_label="OUTCOME_B", evaluation_time=4)
    ledger.save()

    restored = PredictionLedger.load(ledger.path).get(prediction.id)
    assert restored.prediction_time == tick == 3
    assert restored.prediction.issued_at == issued_at
    assert restored.evaluation_time == 4  # a separate field, not a replacement


def test_2b_an_evaluation_cannot_predate_its_prediction():
    timeline = a_timeline()
    record = PredictionRecord(issue_prediction(timeline, PROPOSITION, 0.7, prediction_time=5))

    with pytest.raises(ValueError, match="precedes prediction time"):
        record.evaluate(True, "OUTCOME_A", evaluation_time=2)


# 3 ------------------------------------ future information is rejected


def test_3_future_information_is_rejected():
    """TEMPORAL_LEAKAGE_TEST — the outcome being predicted must be unreachable."""
    timeline = a_timeline()
    future = timeline.add("Trial 5 was A — the very thing being predicted.", at=5, outcome=True)

    with pytest.raises(TemporalLeakageError, match="at or after the prediction time"):
        issue_prediction(
            timeline,
            PROPOSITION,
            0.99,
            prediction_time=5,
            information_available=[future.id],
        )


def test_3b_future_information_cannot_be_smuggled_through_evidence_ids():
    timeline = a_timeline()
    future = timeline.add("Later.", at=9, outcome=True)

    with pytest.raises(TemporalLeakageError):
        issue_prediction(
            timeline,
            PROPOSITION,
            0.99,
            prediction_time=4,
            information_available=[],  # clean...
            evidence_ids_used=[future.id],  # ...but the future hides here
        )


def test_3c_unknown_observation_ids_are_rejected():
    """An id that cannot be resolved cannot be shown to predate anything."""
    timeline = a_timeline()

    with pytest.raises(TemporalLeakageError, match="unknown"):
        issue_prediction(
            timeline, PROPOSITION, 0.7, prediction_time=3, information_available=["OBS-invented"]
        )


def test_3d_a_legitimate_prediction_is_still_accepted():
    """Guard against a system that rejects everything and calls it safety."""
    timeline = a_timeline()
    prediction = issue_prediction(timeline, PROPOSITION, 0.7, prediction_time=3)

    assert len(prediction.information_available_at_prediction_time) == 3


# 4 ------------------------- outcomes cannot modify historical predictions


def test_4_outcomes_cannot_modify_historical_predictions(tmp_path):
    timeline = a_timeline()
    ledger = PredictionLedger.in_directory(tmp_path)
    prediction = issue_prediction(timeline, PROPOSITION, 0.80, prediction_time=3)
    ledger.add(prediction)
    before = prediction.to_dict()

    timeline.add("Trial 3 was B.", at=3, outcome=False)
    evaluated = ledger.evaluate(
        prediction.id, outcome=False, outcome_label="OUTCOME_B", evaluation_time=4
    )

    # The frozen prediction is the identical object, unchanged in every field.
    assert evaluated.prediction is prediction
    assert prediction.to_dict() == before
    assert evaluated.predicted_probability == 0.80  # not revised toward the outcome
    # Only evaluation fields were added.
    assert evaluated.evaluation_status is EvaluationStatus.EVALUATED
    assert evaluated.actual_outcome is False
    assert evaluated.prediction_error == pytest.approx(0.80)


# 5 ------------------------------------ prediction errors are correct


@pytest.mark.parametrize(
    "probability,outcome,error,brier_value",
    [
        (0.80, False, 0.80, 0.64),
        (0.80, True, 0.20, 0.04),
        (0.50, True, 0.50, 0.25),
        (1.00, True, 0.00, 0.00),
        (0.00, False, 0.00, 0.00),
    ],
)
def test_5_prediction_errors_are_calculated_correctly(probability, outcome, error, brier_value):
    timeline = Timeline()
    timeline.add("past", at=0, outcome=True)
    record = PredictionRecord(
        issue_prediction(timeline, PROPOSITION, probability, prediction_time=1)
    )

    evaluated = record.evaluate(outcome, "label", evaluation_time=2)

    assert evaluated.prediction_error == pytest.approx(error)
    assert evaluated.evaluation.brier_score == pytest.approx(brier_value)


def test_5b_log_loss_is_none_when_certain_and_wrong():
    """Infinite surprise is reported as undefined, not quietly clamped."""
    timeline = Timeline()
    timeline.add("past", at=0, outcome=True)
    record = PredictionRecord(issue_prediction(timeline, PROPOSITION, 1.0, prediction_time=1))

    evaluated = record.evaluate(False, "OUTCOME_B", evaluation_time=2)

    assert evaluated.evaluation.log_loss is None
    assert evaluated.evaluation.brier_score == pytest.approx(1.0)


def test_5c_log_loss_matches_the_definition():
    timeline = Timeline()
    timeline.add("past", at=0, outcome=True)
    record = PredictionRecord(issue_prediction(timeline, PROPOSITION, 0.25, prediction_time=1))

    evaluated = record.evaluate(True, "OUTCOME_A", evaluation_time=2)

    assert evaluated.evaluation.log_loss == pytest.approx(-math.log(0.25))


# 6 --------------------------------------- experiences are persisted


def test_6_prediction_experiences_are_persisted(tmp_path):
    timeline = a_timeline()
    ledger = PredictionLedger.in_directory(tmp_path)
    prediction = issue_prediction(timeline, PROPOSITION, 0.70, prediction_time=3)
    ledger.add(prediction)
    timeline.add("Trial 3 was A.", at=3, outcome=True)
    ledger.evaluate(prediction.id, outcome=True, outcome_label="OUTCOME_A", evaluation_time=4)
    ledger.remember_timeline(timeline)
    ledger.save()

    revived = PredictionLedger.load(ledger.path)
    (experience,) = revived.experiences()

    assert experience.prediction_id == prediction.id
    assert experience.prediction == 0.70
    assert experience.outcome is True
    assert experience.error == pytest.approx(0.30)
    assert experience.evidence_available == prediction.information_available_at_prediction_time
    assert experience.timestamp


def test_6b_an_experience_cannot_be_filed_for_an_unevaluated_prediction():
    timeline = a_timeline()
    record = PredictionRecord(issue_prediction(timeline, PROPOSITION, 0.7, prediction_time=3))

    with pytest.raises(ValueError, match="unevaluated"):
        Experience.from_record(record)


def test_6c_experiences_are_immutable():
    timeline = a_timeline()
    record = PredictionRecord(issue_prediction(timeline, PROPOSITION, 0.7, prediction_time=3))
    experience = Experience.from_record(record.evaluate(True, "OUTCOME_A", 4))

    with pytest.raises(Exception):
        experience.error = 0.0


# 7 ------------------------------------------- predictions survive restart


def test_7_predictions_survive_restart(tmp_path):
    timeline = a_timeline()
    ledger = PredictionLedger.in_directory(tmp_path)
    for tick, probability in ((3, 0.70), (4, 0.55)):
        prediction = issue_prediction(timeline, PROPOSITION, probability, prediction_time=tick)
        ledger.add(prediction)
        timeline.add(f"Trial {tick} was A.", at=tick, outcome=True)
        ledger.evaluate(prediction.id, True, "OUTCOME_A", evaluation_time=tick + 1)
    ledger.remember_timeline(timeline)
    ledger.save()

    expected = [r.to_dict() for r in ledger.records()]
    del ledger, timeline  # only the file survives

    revived = PredictionLedger.in_directory(tmp_path)

    assert [r.to_dict() for r in revived.records()] == expected
    assert [r.predicted_probability for r in revived.records()] == [0.70, 0.55]
    # "What information did I have?" still answerable after a restart.
    first = revived.records()[0]
    assert len(revived.information_at(first.id)) == len(
        first.information_available_at_prediction_time
    )


# 8 ------------------------------------------- Brier score is correct


def test_8_brier_score_is_calculated_correctly():
    # Worked by hand: (0.8-1)^2 = 0.04, (0.3-0)^2 = 0.09, (0.5-1)^2 = 0.25
    assert brier([0.8, 0.3, 0.5], [True, False, True]) == pytest.approx(
        (0.04 + 0.09 + 0.25) / 3
    )


def test_8b_always_saying_half_scores_a_quarter():
    assert brier([0.5] * 4, [True, False, True, False]) == pytest.approx(0.25)


def test_8c_a_perfect_forecaster_scores_zero():
    assert brier([1.0, 0.0, 1.0], [True, False, True]) == pytest.approx(0.0)


def test_8d_brier_distinguishes_confident_from_hedged_when_both_are_right():
    """The whole reason accuracy alone is not acceptable."""
    confident = brier([0.99], [True])
    hedged = brier([0.51], [True])

    assert confident < hedged  # same "correct" call, very different claims


def test_8e_brier_punishes_confident_errors_hardest():
    assert brier([0.99], [False]) > brier([0.51], [False])


# 9 ---------------------------------- calibration metrics are correct


def test_9_calibration_buckets_group_and_measure_correctly():
    # Six predictions at 0.7, four of which come true → observed 0.667 in that bucket.
    probabilities = [0.7] * 6 + [0.2] * 5
    outcomes = [True, True, True, True, False, False] + [True, False, False, False, False]

    buckets = calibration_buckets(probabilities, outcomes)
    by_range = {b.label: b for b in buckets}

    seventies = by_range["0.7–0.8"]
    assert seventies.count == 6
    assert seventies.mean_predicted == pytest.approx(0.7)
    assert seventies.observed_frequency == pytest.approx(4 / 6)
    assert seventies.gap == pytest.approx(0.7 - 4 / 6)

    twenties = by_range["0.2–0.3"]
    assert twenties.count == 5
    assert twenties.observed_frequency == pytest.approx(0.2)
    assert twenties.gap == pytest.approx(0.0)  # perfectly calibrated bucket


def test_9b_empty_buckets_are_omitted_rather_than_reported_as_zero():
    buckets = calibration_buckets([0.75, 0.75], [True, False])
    assert [b.label for b in buckets] == ["0.7–0.8"]


def test_9c_expected_calibration_error_is_count_weighted():
    # 8 predictions at 0.5 perfectly calibrated, 2 at 0.9 that never happen.
    probabilities = [0.5] * 8 + [0.9] * 2
    outcomes = [True, True, True, True, False, False, False, False, False, False]

    buckets = calibration_buckets(probabilities, outcomes)
    ece = expected_calibration_error(buckets)

    # 0.5 bucket gap 0.0 (weight 8); 0.9 bucket gap 0.9 (weight 2) -> 0.18
    assert ece == pytest.approx((8 * 0.0 + 2 * 0.9) / 10)


def test_9d_a_perfectly_calibrated_forecaster_has_zero_ece():
    probabilities = [0.5] * 10
    outcomes = [True] * 5 + [False] * 5

    assert expected_calibration_error(calibration_buckets(probabilities, outcomes)) == pytest.approx(0.0)


def test_9e_probability_one_lands_in_the_top_bucket():
    buckets = calibration_buckets([1.0], [True])
    assert buckets[0].label == "0.9–1.0"


# 10 ------------------------------------ a prediction is evaluated once


def test_10_the_same_prediction_cannot_be_evaluated_twice(tmp_path):
    timeline = a_timeline()
    ledger = PredictionLedger.in_directory(tmp_path)
    prediction = issue_prediction(timeline, PROPOSITION, 0.70, prediction_time=3)
    ledger.add(prediction)
    ledger.evaluate(prediction.id, True, "OUTCOME_A", evaluation_time=4)

    with pytest.raises(ValueError, match="already been evaluated"):
        ledger.evaluate(prediction.id, False, "OUTCOME_B", evaluation_time=5)

    # And the first evaluation stands, unaltered.
    assert ledger.get(prediction.id).actual_outcome is True


def test_10b_a_record_refuses_a_second_evaluation_directly(tmp_path):
    timeline = a_timeline()
    record = PredictionRecord(issue_prediction(timeline, PROPOSITION, 0.7, prediction_time=3))
    evaluated = record.evaluate(True, "OUTCOME_A", 4)

    with pytest.raises(ValueError, match="already been evaluated"):
        evaluated.evaluate(False, "OUTCOME_B", 5)


def test_10c_a_prediction_cannot_be_added_to_the_ledger_twice(tmp_path):
    timeline = a_timeline()
    ledger = PredictionLedger.in_directory(tmp_path)
    prediction = issue_prediction(timeline, PROPOSITION, 0.7, prediction_time=3)
    ledger.add(prediction)

    with pytest.raises(ValueError, match="already in the ledger"):
        ledger.add(prediction)


# 11 ------------------- information after the prediction cannot influence it


def test_11_information_after_the_prediction_cannot_influence_it():
    """The view handed to a predictor physically cannot reach the future."""
    timeline = a_timeline()
    timeline.add("Trial 3 — the outcome being predicted.", at=3, outcome=True)
    timeline.add("Trial 4 — even later.", at=4, outcome=True)

    view = timeline.view_as_of(3)

    assert len(view) == 3  # ticks 0, 1, 2 only
    assert all(o.at < 3 for o in view)
    assert view.outcomes() == (True, False, True)  # the future is simply absent


def test_11b_a_predictor_reading_only_the_view_is_unaffected_by_later_events():
    from experiments.environments import laplace_predictor

    timeline = a_timeline()
    before, _ = laplace_predictor(timeline.view_as_of(3))

    # Reality happens — loudly, and all in one direction.
    for tick in range(3, 20):
        timeline.add(f"Trial {tick} was A.", at=tick, outcome=True)

    after, _ = laplace_predictor(timeline.view_as_of(3))

    assert before == after, "a t=3 prediction changed because of t>3 events"


def test_11c_the_recorded_information_set_excludes_the_outcome():
    timeline = a_timeline()
    prediction = issue_prediction(timeline, PROPOSITION, 0.7, prediction_time=3)
    outcome = timeline.add("Trial 3 was A.", at=3, outcome=True)

    assert outcome.id not in prediction.information_available_at_prediction_time


# 12 ------------------------------------- history cannot be rewritten


def test_12_prediction_history_cannot_be_rewritten(tmp_path):
    """The critical sequence: 0.80 → B, 0.65 → B, 0.40. Nothing may be restated."""
    timeline = Timeline()
    ledger = PredictionLedger.in_directory(tmp_path)

    issued = []
    for tick, probability in ((0, 0.80), (1, 0.65), (2, 0.40)):
        prediction = issue_prediction(timeline, PROPOSITION, probability, prediction_time=tick)
        ledger.add(prediction)
        issued.append(prediction)
        if tick < 2:
            timeline.add(f"Trial {tick} was B.", at=tick, outcome=False)
            ledger.evaluate(prediction.id, False, "OUTCOME_B", evaluation_time=tick + 1)

    ledger.remember_timeline(timeline)
    ledger.save()

    revived = PredictionLedger.load(ledger.path)

    # The complete sequence, in issue order, with the original numbers.
    assert [r.predicted_probability for r in revived.records()] == [0.80, 0.65, 0.40]
    # Prediction 1 was not revised toward the outcome after A failed to occur.
    assert revived.records()[0].predicted_probability == 0.80
    assert revived.records()[0].prediction_error == pytest.approx(0.80)
    # The third is still open — an outcome it has not met cannot have changed it.
    assert revived.records()[2].evaluation_status is EvaluationStatus.PENDING
    assert revived.records()[2].actual_outcome is None


def test_12b_the_ledger_hands_out_copies_not_its_own_storage(tmp_path):
    timeline = a_timeline()
    ledger = PredictionLedger.in_directory(tmp_path)
    ledger.add(issue_prediction(timeline, PROPOSITION, 0.7, prediction_time=3))

    records = ledger.records()
    assert isinstance(records, tuple)
    assert isinstance(ledger.experiences(), tuple)


def test_12c_issue_order_is_preserved_and_not_re_sorted(tmp_path):
    """History is a sequence, not a set — the order it happened in is part of it."""
    timeline = Timeline()
    for tick in range(6):
        timeline.add(f"Trial {tick}.", at=tick, outcome=tick % 2 == 0)
    ledger = PredictionLedger.in_directory(tmp_path)

    probabilities = [0.9, 0.1, 0.5, 0.7, 0.3, 0.6]
    for tick, probability in enumerate(probabilities, start=6):
        ledger.add(issue_prediction(timeline, PROPOSITION, probability, prediction_time=tick))
    ledger.save()

    assert [r.predicted_probability for r in ledger.records()] == probabilities
    revived = PredictionLedger.load(ledger.path)
    assert [r.predicted_probability for r in revived.records()] == probabilities


def test_12d_there_is_no_api_for_changing_a_stated_probability(tmp_path):
    """No setter, no update method, no revise. The absence is the guarantee."""
    ledger = PredictionLedger.in_directory(tmp_path)

    for forbidden in ("update", "revise", "amend", "delete", "remove", "set_probability"):
        assert not hasattr(ledger, forbidden), f"ledger exposes {forbidden}()"
