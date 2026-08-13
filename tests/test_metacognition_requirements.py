"""What ECHO 9 has to be true for: a scoreboard, not a feeling.

The properties that matter are that belief and meta-confidence are genuinely
separate quantities, that competence bands come from outcomes rather than
labels, that `UNTESTED` is its own state, that abstention has a measured cost,
and that nothing anywhere reports an inner life.
"""

from __future__ import annotations

import pytest

from echo.metacognition import (
    ABSTENTION_THRESHOLD,
    DOMAINS,
    MIN_FOR_ASSESSMENT,
    STRONG_BRIER,
    WEAK_BRIER,
    Claim,
    Competence,
    Metacognition,
    calibration_of_error_prediction,
    expected_calibration_error,
)

from source_audit import code_without_docs

from experiments import run_metacognition as R
from experiments.metacognition_worlds import PROFILES


@pytest.fixture(scope="module")
def study():
    return R.run_study()


def _claim(domain: str, index: int, belief: float, predicted_error: float = 0.3) -> Claim:
    return Claim(
        claim_id=f"{domain}-{index}",
        domain=domain,
        proposition=f"p{index}",
        belief=belief,
        predicted_error=predicted_error,
        created_at=index,
    )


def _fill(layer: Metacognition, domain: str, right: int, wrong: int, belief: float = 0.9):
    index = 0
    for _ in range(right):
        index += 1
        claim = layer.record(_claim(domain, index, belief))
        layer.resolve(claim.claim_id, True)  # belief > 0.5 and it was true
    for _ in range(wrong):
        index += 1
        claim = layer.record(_claim(domain, index + 10_000, belief))
        layer.resolve(claim.claim_id, False)  # belief > 0.5 but it was false
    return layer


# --------------------------------- 9.1 belief and meta-confidence differ


def test_1_belief_and_meta_confidence_are_different_quantities():
    layer = Metacognition()
    _fill(layer, "prediction", right=5, wrong=5, belief=0.82)
    record = layer.tally("prediction")
    # every belief was 0.82; the track record is nothing like 0.82
    assert record.meta_confidence == pytest.approx(6 / 12)
    assert record.meta_confidence != pytest.approx(0.82, abs=0.2)


def test_1b_a_claim_records_the_meta_confidence_it_was_made_under():
    layer = Metacognition()
    _fill(layer, "prediction", right=10, wrong=0)
    claim = layer.record(_claim("prediction", 99, 0.7))
    assert claim.meta_confidence_at_claim is not None
    assert claim.meta_confidence_at_claim > 0.8


def test_1c_brier_scores_the_belief_against_the_proposition_not_against_being_right():
    """The bug this field exists to prevent."""
    # A confident 'false' call on a proposition that was false is an excellent
    # call, and must score near zero.
    claim = _claim("prediction", 1, belief=0.12).resolved(False)
    assert claim.correct is True
    assert claim.brier == pytest.approx(0.12**2)
    # A confident 'true' call on a proposition that was false is a bad one.
    other = _claim("prediction", 2, belief=0.95).resolved(False)
    assert other.correct is False
    assert other.brier == pytest.approx(0.95**2)


# ----------------------------------------- 9.2 capability monitoring


def test_2_competence_bands_come_from_outcomes(study):
    report = study.layer.report()
    # The hidden skills are ordered; the bands must respect that ordering.
    strong = report["experimentation"]
    weak = report["causal_inference"]
    assert strong.competence is Competence.STRONG
    assert weak.competence is Competence.WEAK
    assert strong.brier < weak.brier


def test_2b_all_four_bands_are_reachable():
    layer = Metacognition()
    _fill(layer, "prediction", right=40, wrong=0, belief=0.95)
    assert layer.competence("prediction") is Competence.STRONG
    _fill(layer, "transfer", right=20, wrong=20, belief=0.95)
    assert layer.competence("transfer") is Competence.WEAK
    _fill(layer, "discovery", right=30, wrong=10, belief=0.75)
    assert layer.competence("discovery") is Competence.UNCERTAIN
    assert layer.competence("experimentation") is Competence.UNTESTED


def test_2c_untested_is_its_own_state_not_low_confidence():
    layer = Metacognition()
    _fill(layer, "prediction", right=MIN_FOR_ASSESSMENT - 1, wrong=0)
    assert layer.competence("prediction") is Competence.UNTESTED
    # ...and an untested domain still answers, or the record could never fill
    assert layer.should_abstain("prediction") is False


def test_2d_domains_are_tracked_separately(study):
    report = study.layer.report()
    briers = {d: report[d].brier for d in DOMAINS}
    assert len({round(b, 3) for b in briers.values() if b is not None}) > 1


def test_2e_no_difficulty_labels_reach_the_metacognition_layer():
    import pathlib

    import echo.metacognition as module

    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(("import ", "from ")):
            assert "metacognition_worlds" not in stripped
            assert "experiments" not in stripped
    for token in ("skill", "assertiveness", "easy", "hard"):
        assert f"self.{token}" not in source


# --------------------------------------------- 9.3 error prediction


def test_3_error_prediction_is_scored_against_actual_errors(study):
    rows = calibration_of_error_prediction(study.layer.claims())
    assert rows
    for row in rows:
        assert 0.0 <= row["mean_predicted_error"] <= 1.0
        assert 0.0 <= row["observed_error_rate"] <= 1.0
    assert expected_calibration_error(rows) is not None


def test_3b_predicted_error_tracks_observed_error(study):
    report = study.layer.report()
    for domain in DOMAINS:
        gap = report[domain].error_prediction_gap
        assert gap is not None
        assert abs(gap) < 0.25, f"{domain} self-assessment off by {gap}"


def test_3c_with_no_record_the_expected_error_is_one_half():
    layer = Metacognition()
    assert layer.expected_error("prediction") == pytest.approx(0.5)


def test_3d_expected_error_falls_as_the_record_improves():
    layer = Metacognition()
    before = layer.expected_error("prediction")
    _fill(layer, "prediction", right=30, wrong=2)
    assert layer.expected_error("prediction") < before


# ------------------------------------------------- 9.4 abstention


def test_4_abstention_happens_where_the_record_is_poor(study):
    assert study.abstention["causal_inference"].abstained > 0
    assert study.abstention["experimentation"].abstained == 0


def test_4b_abstention_improves_accuracy_when_answering(study):
    answered = sum(o.answered for o in study.abstention.values())
    answered_right = sum(o.answered_correct for o in study.abstention.values())
    everything = sum(
        1 for d in DOMAINS for t in study.evaluation[d] if t.correct
    ) / (len(DOMAINS) * R.EVALUATION_TASKS)
    assert answered_right / answered > everything


def test_4c_the_cost_of_abstention_is_measured(study):
    """Unnecessary abstentions must be counted, not hidden."""
    declined = study.abstention["causal_inference"]
    assert declined.abstained > 0
    assert declined.abstained_would_be_correct > 0
    assert declined.accuracy_on_declined is not None


def test_4d_all_four_abstention_metrics_are_available(study):
    outcome = study.abstention["causal_inference"]
    assert outcome.accuracy_when_answering is None or 0 <= outcome.accuracy_when_answering <= 1
    assert 0 <= outcome.accuracy_on_declined <= 1
    assert outcome.confidently_wrong >= 0
    assert outcome.abstained_would_be_correct >= 0


def test_4e_confident_wrongness_is_counted():
    layer = Metacognition()
    claim = layer.record(_claim("transfer", 1, belief=0.95))
    layer.resolve(claim.claim_id, False)
    assert layer.tally("transfer").confidently_wrong == 1
    # a hedged wrong answer is not "confidently wrong"
    hedged = layer.record(_claim("transfer", 2, belief=0.55))
    layer.resolve(hedged.claim_id, False)
    assert layer.tally("transfer").confidently_wrong == 1


def test_4f_the_abstention_threshold_is_what_drives_the_decision():
    layer = Metacognition()
    _fill(layer, "causal_inference", right=10, wrong=10)
    record = layer.tally("causal_inference")
    assert record.meta_confidence < ABSTENTION_THRESHOLD
    assert layer.should_abstain("causal_inference") is True
    assert layer.should_abstain("causal_inference", threshold=0.1) is False


# ------------------------------------------- 9.5 the self-diagnostic


def test_5_echo_ranks_its_own_domains_without_labels(study):
    ranking = study.self_diagnosis
    assert [d for d, _ in ranking] == sorted(
        DOMAINS, key=lambda d: -study.layer.tally(d).meta_confidence
    )


def test_5b_the_ranking_agrees_with_what_happens_next(study):
    """A rank order that did not predict anything would be worthless."""
    order = [d for d, _ in study.self_diagnosis]
    pairs = concordant = 0
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            pairs += 1
            if study.actual_next[order[i]] >= study.actual_next[order[j]]:
                concordant += 1
    assert concordant / pairs >= 0.8


def test_5c_the_strongest_and_weakest_are_correctly_separated(study):
    order = [d for d, _ in study.self_diagnosis]
    assert PROFILES[order[0]].skill > PROFILES[order[-1]].skill


# ---------------------------------------- 9.6 no simulated self-awareness


def test_6_nothing_reports_a_feeling():
    """No simulated inner life — checked against code, not against prose.

    The module's own docstring says there is no method returning *I feel
    uncertain*, so a raw grep would fail on the sentence promising the opposite
    of the violation. Docstrings and comments are stripped; ordinary string
    literals are kept, because a literal is where such a thing would live.
    """
    import echo.metacognition as module

    code = code_without_docs(module.__file__).lower()
    for phrase in (
        "i feel",
        "i am aware",
        "self-aware",
        "conscious",
        "sentient",
        "i sense",
        "my mind",
    ):
        assert phrase not in code, f"{phrase!r} appears in the metacognition code"


def test_6c_the_source_audit_would_catch_a_real_violation(tmp_path):
    """The audit above is worthless if it cannot fail."""
    cheat = tmp_path / "cheat.py"
    cheat.write_text(
        '"""A docstring promising never to say I feel uncertain."""\n'
        "# nor in a comment: I feel uncertain\n"
        "def describe():\n"
        '    return "I feel uncertain"\n',
        encoding="utf-8",
    )
    code = code_without_docs(cheat).lower()
    assert code.count("i feel uncertain") == 1


def test_6b_every_reported_quantity_is_numeric(study):
    for domain in DOMAINS:
        payload = study.layer.tally(domain).to_dict()
        for key, value in payload.items():
            assert isinstance(value, (int, float, str, type(None))), key
        assert isinstance(payload["meta_confidence"], float)
        assert payload["competence"] in {c.value for c in Competence}


# ------------------------------------------------- records and persistence


def test_claims_are_immutable():
    claim = _claim("prediction", 1, 0.8)
    with pytest.raises(Exception):
        claim.belief = 0.2  # type: ignore[misc]
    resolved = claim.resolved(True)
    assert claim.outcome is None
    assert resolved.outcome is True


def test_an_outcome_cannot_be_overwritten():
    claim = _claim("prediction", 1, 0.8).resolved(True)
    with pytest.raises(ValueError):
        claim.resolved(False)


def test_the_record_survives_restart(tmp_path, study):
    path = tmp_path / "metacognition.json"
    study.layer.save(path)
    reloaded = Metacognition.load(path)
    assert len(reloaded) == len(study.layer)
    for domain in DOMAINS:
        before = study.layer.tally(domain)
        after = reloaded.tally(domain)
        assert after.answered == before.answered
        assert after.meta_confidence == pytest.approx(before.meta_confidence)
        assert after.competence is before.competence


def test_the_study_is_reproducible():
    first = R.run_study()
    second = R.run_study()
    for domain in DOMAINS:
        assert first.layer.tally(domain).to_dict() == second.layer.tally(domain).to_dict()
    assert first.self_diagnosis == second.self_diagnosis
