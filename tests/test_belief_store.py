"""Persistence tests for beliefs and their evidence, plus the scenario run."""

import json

import pytest

from echo.belief import Belief, Evidence, EvidenceStance
from echo.belief_store import BeliefStore
from experiments.run_belief_revision import aggregate, run_all
from experiments.world import SCENARIOS


def evidence(eid="EV-1", stance=EvidenceStance.SUPPORTS, **overrides) -> Evidence:
    fields = dict(
        id=eid,
        description="Something was observed.",
        source="station log",
        reliability=0.9,
        relevance=0.8,
        stance=stance,
    )
    fields.update(overrides)
    return Evidence(**fields)


# ------------------------------------------------------------------- store


def test_a_missing_file_is_an_empty_store(tmp_path):
    store = BeliefStore.load(tmp_path / "nothing.json")
    assert len(store) == 0
    assert store.beliefs() == []


def test_save_and_load_round_trip(tmp_path):
    store = BeliefStore.in_directory(tmp_path)
    belief = store.add_belief(Belief("p", confidence=0.5))
    store.consider(belief.id, evidence("EV-A"))
    store.consider(belief.id, evidence("EV-B", stance=EvidenceStance.CONTRADICTS))
    store.save()

    reloaded = BeliefStore.in_directory(tmp_path)

    assert len(reloaded) == 1
    assert reloaded.get_belief(belief.id).to_dict() == belief.to_dict()
    assert len(reloaded.evidence()) == 2


def test_the_file_is_readable_json_with_a_schema_version(tmp_path):
    store = BeliefStore.in_directory(tmp_path)
    belief = store.add_belief(Belief("p", confidence=0.5))
    store.consider(belief.id, evidence("EV-A"))
    path = store.save()

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["beliefs"][0]["proposition"] == "p"
    assert payload["evidence"][0]["id"] == "EV-A"


def test_an_unknown_schema_version_is_rejected(tmp_path):
    path = tmp_path / "beliefs.json"
    path.write_text(json.dumps({"schema_version": 99, "beliefs": []}), encoding="utf-8")

    with pytest.raises(ValueError):
        BeliefStore.load(path)


def test_saving_leaves_no_temp_file(tmp_path):
    store = BeliefStore.in_directory(tmp_path)
    store.add_belief(Belief("p", confidence=0.5))
    store.save()
    store.save()

    assert list(tmp_path.glob("*.tmp")) == []


def test_a_belief_cannot_be_added_twice(tmp_path):
    store = BeliefStore.in_directory(tmp_path)
    belief = store.add_belief(Belief("p", confidence=0.5))

    with pytest.raises(ValueError):
        store.add_belief(belief)


def test_considering_against_an_unknown_belief_raises(tmp_path):
    store = BeliefStore.in_directory(tmp_path)

    with pytest.raises(KeyError):
        store.consider("BEL-nope", evidence("EV-A"))


def test_evidence_is_registered_automatically_when_considered(tmp_path):
    store = BeliefStore.in_directory(tmp_path)
    belief = store.add_belief(Belief("p", confidence=0.5))

    store.consider(belief.id, evidence("EV-A"))

    assert store.get_evidence("EV-A") is not None


def test_evidence_for_returns_what_the_belief_actually_saw(tmp_path):
    store = BeliefStore.in_directory(tmp_path)
    belief = store.add_belief(Belief("p", confidence=0.5))
    store.consider(belief.id, evidence("EV-A"))
    store.consider(belief.id, evidence("EV-B", stance=EvidenceStance.CONTRADICTS))

    assert [e.id for e in store.evidence_for(belief.id)] == ["EV-A", "EV-B"]


def test_the_evidence_behind_a_revision_survives_a_restart(tmp_path):
    """Requirement 8 of the challenge, at the store level."""
    store = BeliefStore.in_directory(tmp_path)
    belief = store.add_belief(Belief("p", confidence=0.5))
    store.consider(
        belief.id,
        evidence("EV-decisive", stance=EvidenceStance.CONTRADICTS, source="telemetry"),
    )
    store.save()
    belief_id = belief.id
    del store, belief

    revived = BeliefStore.in_directory(tmp_path)
    restored = revived.get_belief(belief_id)
    (revision,) = restored.revision_history

    responsible = revived.get_evidence(revision.evidence_id)
    assert responsible is not None
    assert responsible.source == "telemetry"
    assert responsible.stance is EvidenceStance.CONTRADICTS


# ---------------------------------------------------------------- scenarios


def test_every_scenario_reaches_the_correct_side_of_the_threshold(tmp_path):
    results, _ = run_all(tmp_path)
    totals = aggregate(results)

    wrong = [r.scenario.id for r in results if not r.correct]
    assert wrong == [], f"scenarios landed on the wrong side of 0.5: {wrong}"
    assert totals["correct_belief_updates"] == len(SCENARIOS)


def test_no_scenario_overreacts_to_weak_evidence(tmp_path):
    results, _ = run_all(tmp_path)
    offenders = {r.scenario.id: r.overreactions for r in results if r.overreactions}
    assert offenders == {}


def test_the_reversal_pair_starts_identical_and_ends_apart(tmp_path):
    """S1 and S2 differ only in their later evidence, never in ground truth hints."""
    results, _ = run_all(tmp_path)
    s1 = next(r for r in results if r.scenario.id == "S1-belief-correct")
    s2 = next(r for r in results if r.scenario.id == "S2-belief-incorrect")

    assert s1.initial_confidence == pytest.approx(s2.initial_confidence)
    assert s1.final_confidence > 0.8
    assert s2.final_confidence < 0.4


def test_the_irrelevant_scenario_moves_not_one_bit(tmp_path):
    results, _ = run_all(tmp_path)
    s4 = next(r for r in results if r.scenario.id == "S4-irrelevant-contradiction")

    assert s4.final_confidence == s4.initial_confidence
    assert all(not r.applied for r in s4.belief.revision_history[3:])


def test_weak_noise_dents_but_does_not_reverse(tmp_path):
    results, _ = run_all(tmp_path)
    s3 = next(r for r in results if r.scenario.id == "S3-weak-noise")

    assert s3.final_confidence < s3.initial_confidence  # it did register
    assert s3.final_confidence > 0.5  # but did not capitulate


def test_gradual_accumulation_eventually_flips_the_belief(tmp_path):
    results, _ = run_all(tmp_path)
    s5 = next(r for r in results if r.scenario.id == "S5-gradual-accumulation")

    assert s5.initial_confidence > 0.5
    assert s5.final_confidence < 0.5
    # No single item did it — each contradiction is individually moderate.
    later = [r for r in s5.belief.revision_history if r.applied][3:]
    assert all(abs(r.delta) < 0.15 for r in later)


def test_the_whole_experiment_is_reproducible(tmp_path):
    first, _ = run_all(tmp_path / "a")
    second, _ = run_all(tmp_path / "b")

    assert [r.final_confidence for r in first] == [r.final_confidence for r in second]


def test_ground_truth_is_never_visible_to_the_belief(tmp_path):
    """Nothing ECHO stores should contain the answer."""
    _, store = run_all(tmp_path)
    serialised = json.dumps(
        json.loads(store.path.read_text(encoding="utf-8"))
    ).lower()

    assert "ground_truth" not in serialised
    assert "hidden_state" not in serialised
