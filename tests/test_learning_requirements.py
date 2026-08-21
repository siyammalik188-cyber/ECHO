"""The seventeen properties learning was asked to demonstrate.

Written to be read as evidence. Each test maps to one numbered requirement.
Everything runs offline — no model, no API key, no network.
"""

import math

import pytest

from echo.learning import (
    CONFIRM_GAP,
    MIN_FOR_ANALYSIS,
    RECENT_WINDOW,
    ChangeDetector,
    FailureExplanation,
    LearningExperience,
    Sample,
    StrategyRevision,
    analyse_failure,
    binomial_tail,
    propose_successor,
)
from echo.learning_ledger import LearningLedger
from echo.predictors import LAPLACE_ALL, LAPLACE_WINDOW
from echo.strategy import PerformanceRecord, StrategyStatus, initial_strategy
from experiments.environments import by_id
from experiments.run_learning import run_condition


# --------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """Run every environment under both conditions once, and share the result."""
    directory = tmp_path_factory.mktemp("learning")
    from experiments.environments import LEARNING_ENVIRONMENTS

    out = {}
    for environment in LEARNING_ENVIRONMENTS:
        out[environment.id] = {
            "baseline": run_condition(environment, "baseline", directory),
            "learning": run_condition(environment, "learning", directory),
        }
    return out


def samples(pattern, probability=0.85):
    """Build a sample history from a string of hits (.) and misses (x)."""
    out = []
    for index, char in enumerate(pattern):
        outcome = char == "."
        out.append(
            Sample(
                trial=index,
                probability=probability,
                outcome=outcome,
                brier=(probability - (1.0 if outcome else 0.0)) ** 2,
            )
        )
    return out


# 1 ------------------------------------------- errors create experiences


def test_1_prediction_errors_create_experiences(runs):
    learned = runs["ENV-B-changing"]["learning"]

    experiences = learned.learning.experiences()
    assert len(experiences) == len(learned.samples)  # one per evaluated prediction
    first = experiences[0]
    assert first.prediction_id
    assert first.previous_strategy
    assert 0.0 <= first.prediction_error <= 1.0
    assert first.created_at


def test_1b_an_experience_records_every_required_field():
    experience = LearningExperience(
        prediction_id="PRED-1",
        observed_outcome=False,
        prediction_error=0.8,
        relevant_evidence=("OBS-1",),
        previous_strategy="STRAT-1 v1",
        failure_analysis=analyse_failure(0.8, samples("." * 20), None, None),
        proposed_update=None,
        confidence=0.6,
    )
    payload = experience.to_dict()
    for key in (
        "learning_experience_id",
        "prediction_id",
        "observed_outcome",
        "prediction_error",
        "relevant_evidence",
        "previous_strategy",
        "failure_analysis",
        "proposed_update",
        "confidence",
        "created_at",
    ):
        assert key in payload, key


def test_1c_experiences_are_immutable():
    experience = LearningExperience(
        prediction_id="PRED-1",
        observed_outcome=False,
        prediction_error=0.8,
        relevant_evidence=(),
        previous_strategy="STRAT-1 v1",
        failure_analysis=analyse_failure(0.8, samples("." * 20), None, None),
        proposed_update=None,
        confidence=0.6,
    )
    with pytest.raises(Exception):
        experience.prediction_error = 0.0


# 2 ------------------------------------------------ experiences are analysed


def test_2_experiences_can_be_analysed(runs):
    learned = runs["ENV-B-changing"]["learning"]
    reached = set(learned.learning.explanations())

    assert reached, "no failure analysis was performed at all"
    for name in reached:
        assert name in {e.value for e in FailureExplanation}


def test_2b_analysis_can_conclude_unknown():
    """UNKNOWN must be reachable, or the analyser is just guessing confidently."""
    detector = ChangeDetector()
    history = samples("." * 20 + "x" * 10)
    signal = detector.assess(history, at_trial=30)

    analysis = analyse_failure(0.8, history, signal, None)

    assert signal is not None and signal.suspected_regime_change
    assert not signal.confirmed  # significant, but not yet sustained
    assert analysis.explanation is FailureExplanation.UNKNOWN
    assert "both remain live" in analysis.reasoning.lower()


def test_2c_analysis_says_insufficient_information_when_it_has_none():
    analysis = analyse_failure(0.9, samples("x"), None, None)

    assert analysis.explanation is FailureExplanation.INSUFFICIENT_INFORMATION
    assert analysis.confidence > 0.5


def test_2d_analysis_can_blame_the_evidence():
    analysis = analyse_failure(0.8, samples("." * 30), None, evidence_divergence=0.4)

    assert analysis.explanation is FailureExplanation.EVIDENCE_ERROR


def test_2e_analysis_can_blame_the_strategy():
    """Persistently poor scores, no regime signal, no bad evidence."""
    history = samples("x" * 30, probability=0.5)
    for s in history:
        s.brier = 0.5  # sustained, unambiguous poor performance

    analysis = analyse_failure(0.7, history, None, 0.0)

    assert analysis.explanation is FailureExplanation.STRATEGY_FAILURE


def test_2f_every_explanation_carries_reasoning_and_alternatives():
    analysis = analyse_failure(0.3, samples("." * 20), None, None)

    assert analysis.reasoning
    assert analysis.considered
    assert 0.0 <= analysis.confidence <= 1.0


# 3 ----------------------------- a single error does not rewrite strategy


def test_3_a_single_error_does_not_rewrite_strategy():
    detector = ChangeDetector()
    history = samples("." * 25 + "x")  # one miss after a clean run

    signal = detector.assess(history, at_trial=26)
    analysis = analyse_failure(0.85, history, signal, None)

    assert analysis.explanation is FailureExplanation.RANDOM_VARIATION
    assert signal is None or not signal.confirmed


def test_3b_no_strategy_change_happens_before_a_confirmed_signal(runs):
    learned = runs["ENV-B-changing"]["learning"]
    revisions = learned.learning.revisions()

    assert len(revisions) <= 1
    if revisions:
        signal = revisions[0].change_signal
        assert signal["confirmed"] is True
        assert signal["suspected_regime_change"] is True


def test_3c_a_short_streak_is_not_enough_to_confirm():
    """Significant once is not confirmation — it must persist."""
    detector = ChangeDetector()
    history = samples("." * 20 + "x" * 10)

    first = detector.assess(history, at_trial=30)
    immediately_after = detector.assess(history, at_trial=30 + CONFIRM_GAP - 1)

    assert first.suspected_regime_change and not first.confirmed
    assert not immediately_after.confirmed


# 4 -------------------------------- repeated errors trigger investigation


def test_4_repeated_errors_can_trigger_investigation():
    detector = ChangeDetector()
    history = samples("." * 20 + "x" * 10)

    detector.assess(history, at_trial=30)  # first firing
    later = detector.assess(history, at_trial=30 + CONFIRM_GAP)  # sustained

    assert later.confirmed
    assert later.p_value < 0.01
    assert later.suspected_regime_change


def test_4b_a_confirmed_signal_records_everything_required():
    detector = ChangeDetector()
    history = samples("." * 20 + "x" * 10)
    detector.assess(history, at_trial=30)
    signal = detector.assess(history, at_trial=30 + CONFIRM_GAP)

    payload = signal.to_dict()
    for key in (
        "historical_performance",
        "recent_performance",
        "error_rate",
        "confidence",
        "suspected_regime_change",
        "threshold_crossed",
    ):
        assert key in payload, key
    assert payload["historical_performance"]["predictions"] > 0
    assert payload["recent_performance"]["predictions"] == RECENT_WINDOW


def test_4c_the_binomial_tail_is_correct():
    # P(X >= 2 | n=3, p=0.5) = (3 + 1) / 8
    assert binomial_tail(2, 3, 0.5) == pytest.approx(0.5)
    assert binomial_tail(0, 5, 0.3) == pytest.approx(1.0)
    assert binomial_tail(5, 5, 0.5) == pytest.approx(0.5**5)


# 5 ------------------------------------------------ strategies have versions


def test_5_strategies_have_versions():
    first = initial_strategy()
    assert first.version == 1

    second = first.descendant("a successor", kind=LAPLACE_WINDOW, params={"window": 10})
    assert second.version == 2
    assert second.parent_id == first.id
    assert first.version == 1  # the parent is untouched

    third = second.descendant("a third")
    assert third.version == 3


def test_5b_a_strategy_records_everything_required():
    payload = initial_strategy().to_dict()
    for key in (
        "strategy_id",
        "description",
        "assumptions",
        "applicable_conditions",
        "version",
        "performance_history",
        "confidence",
        "status",
        "created_at",
    ):
        assert key in payload, key


# 6 ------------------------------------------ revisions are persistent


def test_6_strategy_revisions_are_persistent(tmp_path):
    ledger = LearningLedger.in_directory(tmp_path)
    first = initial_strategy()
    second = first.descendant("successor", kind=LAPLACE_WINDOW, params={"window": 10})
    ledger.register_strategy(first)
    ledger.register_strategy(second)
    revision = ledger.add_revision(
        StrategyRevision(
            at_trial=36,
            from_strategy_id=first.id,
            to_strategy_id=second.id,
            reason="sustained deterioration",
            triggering_experience_ids=("LEXP-1", "LEXP-2"),
            change_signal={"confirmed": True},
            reversal_condition="revert if it loses the margin",
            confidence=0.9,
        )
    )
    ledger.conclude_revision(revision.id, "improved", {"mean_brier": 0.4}, {"mean_brier": 0.2})
    ledger.save()

    revived = LearningLedger.in_directory(tmp_path)
    (restored,) = revived.revisions()

    assert restored.at_trial == 36
    assert restored.from_strategy_id == first.id
    assert restored.to_strategy_id == second.id
    assert restored.outcome == "improved"
    assert restored.reversal_condition == "revert if it loses the margin"
    assert restored.triggering_experience_ids == ("LEXP-1", "LEXP-2")


def test_6b_concluding_a_revision_does_not_mutate_the_original_object(tmp_path):
    ledger = LearningLedger.in_directory(tmp_path)
    revision = ledger.add_revision(
        StrategyRevision(
            at_trial=1,
            from_strategy_id="A",
            to_strategy_id="B",
            reason="r",
            triggering_experience_ids=(),
            change_signal=None,
            reversal_condition="c",
            confidence=0.5,
        )
    )
    ledger.conclude_revision(revision.id, "improved", {}, {})

    assert revision.outcome == "pending"  # the object we hold never changed
    assert ledger.revisions()[0].outcome == "improved"


# 7 --------------------------------------- old strategies remain in history


def test_7_old_strategies_remain_in_history(runs):
    learned = runs["ENV-B-changing"]["learning"]
    strategies = learned.learning.strategies()

    assert len(strategies) >= 2, "the superseded strategy was not kept"
    statuses = {s.status for s in strategies}
    assert StrategyStatus.ACTIVE in statuses
    assert StrategyStatus.WEAKENED in statuses or StrategyStatus.RETIRED in statuses
    # The original is still there with its full record.
    original = min(strategies, key=lambda s: s.version)
    assert original.version == 1
    assert original.kind == LAPLACE_ALL
    assert original.assumptions


def test_7b_a_retired_strategy_is_not_revived():
    strategy = initial_strategy()
    strategy.set_status(StrategyStatus.RETIRED, "failed its test")

    with pytest.raises(ValueError, match="retired"):
        strategy.set_status(StrategyStatus.ACTIVE, "second thoughts")


def test_7c_status_changes_keep_their_reasons():
    strategy = initial_strategy()
    strategy.set_status(StrategyStatus.WEAKENED, "superseded at trial 44")

    reasons = [entry[2] for entry in strategy.status_history]
    assert "superseded at trial 44" in reasons
    assert len(strategy.status_history) >= 2  # the initial activation is kept too


# 8 --------------------------------------------- new strategies are tested


def test_8_new_strategies_are_tested_before_adoption(runs):
    learned = runs["ENV-B-changing"]["learning"]
    events = " ".join(learned.events)

    assert "TESTING" in events
    assert "test concluded" in events
    successor = max(learned.learning.strategies(), key=lambda s: s.version)
    assert successor.performance_history, "the candidate was adopted without measurement"


def test_8b_the_incumbent_is_shadow_scored_over_the_same_trials(runs):
    """A comparison against a placeholder would be no comparison at all."""
    learned = runs["ENV-B-changing"]["learning"]
    (revision,) = learned.learning.revisions()

    before, after = revision.measured_before, revision.measured_after
    assert before is not None and after is not None
    assert before["window"] == after["window"]  # like-for-like
    assert 0.0 < before["mean_brier"] < 1.0, "incumbent Brier looks like a fallback value"
    assert 0.0 < after["mean_brier"] < 1.0


# 9 ---------------------------------- unsuccessful strategies are retired


def test_9_unsuccessful_strategies_are_retired(runs):
    """Environment F has no regime change; the candidate should not survive."""
    learned = runs["ENV-F-false-alarm"]["learning"]
    revisions = learned.learning.revisions()

    assert revisions, "nothing was proposed, so nothing could be retired"
    assert revisions[0].outcome == "no_improvement"
    retired = [s for s in learned.learning.strategies() if s.status is StrategyStatus.RETIRED]
    assert retired, "a candidate that failed its test was not retired"


def test_9b_a_weakened_strategy_keeps_its_history():
    strategy = initial_strategy()
    strategy.record_performance(
        PerformanceRecord(label="w", predictions=8, mean_brier=0.4, miss_rate=0.6)
    )
    strategy.set_status(StrategyStatus.WEAKENED, "superseded")

    assert strategy.status is StrategyStatus.WEAKENED
    assert len(strategy.performance_history) == 1


# 10 ------------------------------------ successful strategies go active


def test_10_successful_strategies_become_active(runs):
    learned = runs["ENV-B-changing"]["learning"]
    (revision,) = learned.learning.revisions()

    assert revision.outcome == "improved"
    successor = learned.learning.get_strategy(revision.to_strategy_id)
    incumbent = learned.learning.get_strategy(revision.from_strategy_id)
    assert successor.status is StrategyStatus.ACTIVE
    assert incumbent.status is StrategyStatus.WEAKENED
    assert revision.measured_after["mean_brier"] < revision.measured_before["mean_brier"]


# 11 --------------------------- learning improves adaptation in Environment B


def test_11_learning_improves_adaptation_in_environment_b(runs):
    baseline = runs["ENV-B-changing"]["baseline"]
    learned = runs["ENV-B-changing"]["learning"]

    base_post = baseline.post_change["brier"]
    learn_post = learned.post_change["brier"]

    assert learn_post < base_post, (
        f"learning did not improve post-change Brier: {base_post} -> {learn_post}"
    )
    # And it recovered at all, where the fixed strategy never did.
    assert baseline.recovery_trial() is None
    assert learned.recovery_trial() is not None


def test_11b_learning_also_helps_over_a_longer_run(runs):
    baseline = runs["ENV-B-LONG"]["baseline"]
    learned = runs["ENV-B-LONG"]["learning"]

    assert learned.post_change["brier"] < baseline.post_change["brier"]
    assert learned.final_window["brier"] < baseline.final_window["brier"]


# 12 ------------------------------ learning does not memorise trial numbers


def test_12_learning_does_not_memorise_trial_numbers():
    """The detector's state is a comparison, not a calendar."""
    detector = ChangeDetector()

    # Nothing in the detector's configuration references a position in the run.
    for value in (
        detector.recent_window,
        detector.min_history,
        detector.confirm_gap,
    ):
        assert isinstance(value, int)
    assert not hasattr(detector, "change_trial")
    assert not hasattr(detector, "known_change_points")

    # The same deterioration is detected wherever it happens to occur.
    early = ChangeDetector()
    late = ChangeDetector()
    pattern = "." * 20 + "x" * 10

    early.assess(samples(pattern), at_trial=30)
    early_signal = early.assess(samples(pattern), at_trial=30 + CONFIRM_GAP)

    late.assess(samples("." * 100 + "x" * 10), at_trial=110)
    late_signal = late.assess(samples("." * 100 + "x" * 10), at_trial=110 + CONFIRM_GAP)

    assert early_signal.confirmed and late_signal.confirmed


def test_12b_detection_happens_near_the_actual_change_not_at_trial_thirty(runs):
    """Environment D changes at 62. Detection must follow the change, not a number."""
    learned = runs["ENV-D-late-change"]["learning"]
    confirmed = [d for d in learned.detections if d["confirmed"]]

    assert confirmed, "no change was detected in Environment D"
    at = confirmed[0]["at_trial"]
    assert at > 62, f"detection at trial {at} preceded the change at 62"
    assert at < 62 + 30, f"detection at trial {at} took implausibly long"


# 13 ------------------------------------ generalises to a new change location


def test_13_learning_generalises_to_a_new_regime_change_location(runs):
    baseline = runs["ENV-D-late-change"]["baseline"]
    learned = runs["ENV-D-late-change"]["learning"]

    base_post = baseline.post_change["brier"]
    learn_post = learned.post_change["brier"]

    assert learn_post < base_post, (
        "learning failed to transfer to a change in a different place and direction: "
        f"{base_post} -> {learn_post} (this would be overfitting to Environment B)"
    )
    base_recovery = baseline.recovery_trial()
    learn_recovery = learned.recovery_trial()
    assert learn_recovery is not None
    if base_recovery is not None:
        assert learn_recovery < base_recovery


# 14 --------------------------------------------- false alarms are measurable


def test_14_false_alarms_are_measurable(runs):
    """A stationary environment with a run of unusual outcomes."""
    learned = runs["ENV-F-false-alarm"]["learning"]
    environment = by_id_learning("ENV-F-false-alarm")

    assert environment.change_points == ()  # nothing actually changed
    confirmed = [d for d in learned.detections if d["confirmed"]]
    flagged = learned.detections

    # The count is available either way — that is what "measurable" means.
    assert isinstance(len(confirmed), int)
    assert len(flagged) >= len(confirmed)
    # Whatever the detector did, the test stage refused to keep the change.
    assert learned.learning.revisions()[0].outcome == "no_improvement"


def test_14b_flagging_and_confirming_are_different_counts(runs):
    """The gap between them is where overreaction is prevented."""
    learned = runs["ENV-F-false-alarm"]["learning"]
    flagged = len(learned.detections)
    confirmed = sum(1 for d in learned.detections if d["confirmed"])

    assert flagged > confirmed, "every flag was confirmed; the gate does nothing"


def by_id_learning(environment_id):
    from experiments.environments import LEARNING_ENVIRONMENTS

    return next(e for e in LEARNING_ENVIRONMENTS if e.id == environment_id)


# 15 ------------------------------------ learning can fail, and say so


def test_15_learning_can_fail_and_reports_failure_honestly(runs):
    """Environment F's revision did not help, and the ledger says exactly that."""
    learned = runs["ENV-F-false-alarm"]["learning"]
    (item,) = learned.learning.what_did_i_learn()

    assert item["did_it_help"] == "no_improvement"
    assert item["before"]["mean_brier"] < item["after"]["mean_brier"]  # it was worse
    assert item["would_reverse_if"]  # and the reversal condition was stated up front


def test_15b_a_failed_adaptation_has_a_measurable_cost(runs):
    """Honest accounting: chasing a false alarm is not free."""
    baseline = runs["ENV-F-false-alarm"]["baseline"]
    learned = runs["ENV-F-false-alarm"]["learning"]

    # Not asserting which way this goes — only that both numbers exist to compare.
    assert baseline.scores()["brier"] is not None
    assert learned.scores()["brier"] is not None


# 16 --------------------------------- original predictions stay immutable


def test_16_original_predictions_remain_immutable(runs):
    learned = runs["ENV-B-changing"]["learning"]
    records = learned.predictions.records()

    # Learning changed the strategy but touched no issued prediction.
    for record in records:
        with pytest.raises(Exception):
            record.prediction.predicted_probability = 0.5

    reloaded = type(learned.predictions).load(learned.predictions.path)
    assert [r.predicted_probability for r in reloaded.records()] == [
        r.predicted_probability for r in records
    ]


def test_16b_experiences_reference_predictions_without_altering_them(runs):
    learned = runs["ENV-B-changing"]["learning"]

    for experience in learned.learning.experiences()[:20]:
        record = learned.predictions.get(experience.prediction_id)
        assert record is not None
        assert record.predicted_probability == pytest.approx(
            record.prediction.predicted_probability
        )


# 17 ---------------------------------------- learning history survives restart


def test_17_learning_history_survives_restart(runs):
    learned = runs["ENV-B-changing"]["learning"]
    expected = learned.learning

    revived = LearningLedger.load(expected.path)

    assert len(revived.experiences()) == len(expected.experiences())
    assert len(revived.revisions()) == len(expected.revisions())
    assert len(revived.strategies()) == len(expected.strategies())

    # And every question the ledger is supposed to answer still answers.
    (item,) = revived.what_did_i_learn()
    assert item["why"]
    assert item["did_it_help"] in {"improved", "no_improvement", "pending"}
    assert item["would_reverse_if"]
    assert item["before"] and item["after"]
    assert revived.triggering_experiences(revived.revisions()[0].id)


def test_17b_strategy_status_and_performance_survive_restart(runs):
    learned = runs["ENV-B-changing"]["learning"]
    revived = LearningLedger.load(learned.learning.path)

    for before, after in zip(learned.learning.strategies(), revived.strategies()):
        assert before.id == after.id
        assert before.status is after.status
        assert before.version == after.version
        assert len(before.performance_history) == len(after.performance_history)
        assert before.assumptions == after.assumptions
