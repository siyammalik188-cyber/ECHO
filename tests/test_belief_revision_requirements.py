"""The nine properties belief revision was asked to demonstrate.

Written to be read as evidence. Each test maps to one numbered requirement.
Everything here runs offline — no model, no API key, no network.
"""

import pytest

from echo.belief import Belief, Evidence, EvidenceStance
from echo.belief_store import BeliefStore

PROPOSITION = "Dr Vance's fertiliser caused the orchid deaths."


def evidence(
    eid="EV-1",
    stance=EvidenceStance.SUPPORTS,
    reliability=0.9,
    relevance=0.8,
    source="station log",
    description="Something was observed.",
) -> Evidence:
    return Evidence(
        id=eid,
        description=description,
        source=source,
        reliability=reliability,
        relevance=relevance,
        stance=stance,
    )


def a_belief(confidence=0.5) -> Belief:
    return Belief(proposition=PROPOSITION, confidence=confidence)


# 1 ---------------------------------------------------------- beliefs exist


def test_1_beliefs_can_be_created():
    belief = a_belief(0.5)

    for attribute in (
        "id",
        "proposition",
        "confidence",
        "supporting_evidence",
        "contradicting_evidence",
        "created_at",
        "updated_at",
        "revision_count",
        "revision_history",
    ):
        assert hasattr(belief, attribute), attribute

    assert belief.id
    assert belief.proposition == PROPOSITION
    assert belief.confidence == 0.5
    assert belief.revision_count == 0
    assert belief.revision_history == ()
    assert belief.supporting_evidence == ()
    assert belief.contradicting_evidence == ()


def test_1b_a_belief_is_formed_from_evidence_not_assigned():
    """Starts at genuine ignorance and moves only because evidence arrived."""
    belief = a_belief(0.5)
    for i, (rel, relv) in enumerate([(0.9, 0.5), (0.6, 0.45), (0.85, 0.35)]):
        belief.consider(evidence(eid=f"EV-{i}", reliability=rel, relevance=relv))

    assert belief.confidence > 0.5
    assert belief.revision_count == 3
    assert belief.initial_confidence == 0.5


# 2 -------------------------------------------------------- survives restart


def test_2_beliefs_survive_restart(tmp_path):
    store = BeliefStore.in_directory(tmp_path)
    belief = store.add_belief(a_belief(0.5))
    store.consider(belief.id, evidence("EV-A"))
    store.consider(belief.id, evidence("EV-B", stance=EvidenceStance.CONTRADICTS))
    store.save()

    belief_id = belief.id
    expected_confidence = belief.confidence
    expected_history = len(belief.revision_history)
    del store, belief  # nothing in memory survives; only the file does

    revived = BeliefStore.in_directory(tmp_path)
    restored = revived.get_belief(belief_id)

    assert restored is not None
    assert restored.proposition == PROPOSITION
    assert restored.confidence == expected_confidence  # final confidence survives
    assert len(restored.revision_history) == expected_history  # history survives
    assert restored.revision_count == 2
    # The evidence responsible for each revision is still resolvable.
    for revision in restored.revision_history:
        assert revived.get_evidence(revision.evidence_id) is not None
        assert revision.evidence_snapshot["source"]


# 3 ------------------------------------------------------- evidence supports


def test_3_evidence_can_support_a_belief():
    belief = a_belief(0.5)
    before = belief.confidence

    revision = belief.consider(evidence("EV-S", stance=EvidenceStance.SUPPORTS))

    assert revision.applied
    assert belief.confidence > before
    assert "EV-S" in belief.supporting_evidence
    assert "EV-S" not in belief.contradicting_evidence


# 4 ---------------------------------------------------- evidence contradicts


def test_4_evidence_can_contradict_a_belief():
    belief = a_belief(0.7)
    before = belief.confidence

    revision = belief.consider(evidence("EV-C", stance=EvidenceStance.CONTRADICTS))

    assert revision.applied
    assert belief.confidence < before
    assert "EV-C" in belief.contradicting_evidence


# 5 ------------------------------------------- contradiction causes revision


def test_5_contradictory_evidence_causes_a_recorded_revision():
    belief = a_belief(0.5)
    belief.consider(evidence("EV-A", reliability=0.9, relevance=0.5))
    initial = belief.confidence

    belief.consider(
        evidence(
            "EV-D",
            stance=EvidenceStance.CONTRADICTS,
            reliability=0.95,
            relevance=0.90,
            source="dual-sensor telemetry",
        )
    )

    assert belief.confidence < initial
    revision = belief.revision_history[-1]
    assert revision.evidence_id == "EV-D"
    assert revision.previous_confidence == initial  # previous state preserved
    assert revision.new_confidence == belief.confidence
    assert revision.delta < 0
    # And it says why, naming the evidence and its weighting.
    assert "EV-D" in revision.reason
    assert "contradicts" in revision.reason
    assert "0.95" in revision.reason and "0.90" in revision.reason


# 6 ------------------------------------------------ history cannot be destroyed


def test_6a_revision_history_is_a_copy_that_cannot_be_mutated():
    belief = a_belief(0.5)
    belief.consider(evidence("EV-A"))

    history = belief.revision_history
    assert isinstance(history, tuple)  # no append, no clear
    with pytest.raises((AttributeError, TypeError)):
        history.append("nonsense")  # type: ignore[attr-defined]

    assert len(belief.revision_history) == 1


def test_6b_confidence_cannot_be_set_directly():
    """The only route to a new confidence is consider(), which records why."""
    belief = a_belief(0.5)

    with pytest.raises(AttributeError):
        belief.confidence = 0.99  # type: ignore[misc]

    assert belief.confidence == 0.5


def test_6c_a_revision_entry_is_frozen():
    belief = a_belief(0.5)
    belief.consider(evidence("EV-A"))
    revision = belief.revision_history[0]

    with pytest.raises(Exception):
        revision.new_confidence = 0.1  # type: ignore[misc]


def test_6d_history_only_ever_grows():
    belief = a_belief(0.5)
    lengths = []
    for i in range(5):
        belief.consider(evidence(eid=f"EV-{i}"))
        lengths.append(len(belief.revision_history))

    assert lengths == [1, 2, 3, 4, 5]


def test_6e_the_same_evidence_cannot_be_counted_twice():
    belief = a_belief(0.5)
    item = evidence("EV-A")
    belief.consider(item)

    with pytest.raises(ValueError, match="already been considered"):
        belief.consider(item)

    assert len(belief.revision_history) == 1


def test_6f_history_survives_a_save_and_reload_intact(tmp_path):
    store = BeliefStore.in_directory(tmp_path)
    belief = store.add_belief(a_belief(0.5))
    for i in range(3):
        store.consider(belief.id, evidence(eid=f"EV-{i}"))
    before = [r.to_dict() for r in belief.revision_history]
    store.save()

    reloaded = BeliefStore.in_directory(tmp_path).get_belief(belief.id)

    assert [r.to_dict() for r in reloaded.revision_history] == before


# 7 ------------------------------- weak contradiction does not reverse a belief


def test_7_weak_contradictory_evidence_does_not_reverse_a_strong_belief():
    belief = a_belief(0.5)
    for i, (rel, relv) in enumerate([(0.9, 0.5), (0.6, 0.45), (0.85, 0.35)]):
        belief.consider(evidence(eid=f"EV-{i}", reliability=rel, relevance=relv))
    strong = belief.confidence
    assert strong > 0.7

    # Four separate weak contradictions: rumour, hearsay, anonymous, unsupported.
    for i, (rel, relv) in enumerate([(0.15, 0.45), (0.12, 0.40), (0.25, 0.45), (0.10, 0.50)]):
        belief.consider(
            evidence(
                eid=f"EV-W{i}",
                stance=EvidenceStance.CONTRADICTS,
                reliability=rel,
                relevance=relv,
            )
        )

    # It dented the belief, but nothing like a reversal.
    assert belief.confidence < strong
    assert belief.confidence > 0.5, "four weak rumours must not overturn the belief"
    assert strong - belief.confidence < 0.15


def test_7b_a_single_weak_item_barely_moves_anything():
    belief = a_belief(0.75)

    belief.consider(
        evidence(
            "EV-weak",
            stance=EvidenceStance.CONTRADICTS,
            reliability=0.2,
            relevance=0.5,
        )
    )

    assert abs(belief.confidence - 0.75) < 0.03


def test_7c_irrelevant_evidence_does_not_move_the_belief_at_all():
    belief = a_belief(0.75)

    revision = belief.consider(
        evidence(
            "EV-irrelevant",
            stance=EvidenceStance.CONTRADICTS,
            reliability=0.95,  # highly reliable...
            relevance=0.05,  # ...and entirely beside the point
        )
    )

    assert not revision.applied
    assert belief.confidence == 0.75
    assert revision.delta == 0.0
    assert "below the" in revision.reason  # and it says why
    assert belief.revision_count == 0  # considered, but not a revision
    assert belief.considered_count == 1  # yet still on the record


# 8 --------------------------- strong contradiction substantially moves belief


def test_8_strong_contradictory_evidence_substantially_changes_confidence():
    belief = a_belief(0.5)
    for i, (rel, relv) in enumerate([(0.9, 0.5), (0.6, 0.45), (0.85, 0.35)]):
        belief.consider(evidence(eid=f"EV-{i}", reliability=rel, relevance=relv))
    strong = belief.confidence

    belief.consider(
        evidence(
            "EV-telemetry",
            stance=EvidenceStance.CONTRADICTS,
            reliability=0.95,
            relevance=0.90,
            source="dual-sensor telemetry",
        )
    )
    belief.consider(
        evidence(
            "EV-assay",
            stance=EvidenceStance.CONTRADICTS,
            reliability=0.90,
            relevance=0.85,
            source="accredited laboratory",
        )
    )

    assert strong - belief.confidence > 0.30, "strong evidence must actually bite"
    assert belief.confidence < 0.5, "the belief should have flipped"


def test_8b_strong_and_weak_contradiction_are_not_treated_alike():
    """The whole point: magnitude tracks evidence quality, not mere disagreement."""

    def drop_from(reliability, relevance):
        belief = a_belief(0.75)
        belief.consider(
            evidence(
                "EV-x",
                stance=EvidenceStance.CONTRADICTS,
                reliability=reliability,
                relevance=relevance,
            )
        )
        return 0.75 - belief.confidence

    weak = drop_from(0.2, 0.5)
    strong = drop_from(0.95, 0.9)

    assert strong > weak * 5


# 9 ------------------------------ different evidence sequences, different beliefs


def test_9_two_evidence_sequences_produce_different_final_beliefs():
    """Same opening, same initial belief, different later evidence, different ends."""

    def opening() -> Belief:
        belief = a_belief(0.5)
        for i, (rel, relv) in enumerate([(0.9, 0.5), (0.6, 0.45), (0.85, 0.35)]):
            belief.consider(evidence(eid=f"EV-{i}", reliability=rel, relevance=relv))
        return belief

    world_a, world_b = opening(), opening()
    assert world_a.confidence == world_b.confidence  # identical starting point

    # World A: the contradiction is anonymous; then a strong corroboration.
    world_a.consider(
        evidence("EV-note", stance=EvidenceStance.CONTRADICTS, reliability=0.2, relevance=0.5)
    )
    world_a.consider(
        evidence("EV-assay-a", stance=EvidenceStance.SUPPORTS, reliability=0.95, relevance=0.95)
    )

    # World B: the contradiction is instrumented and corroborated.
    world_b.consider(
        evidence("EV-telemetry", stance=EvidenceStance.CONTRADICTS, reliability=0.95, relevance=0.9)
    )
    world_b.consider(
        evidence("EV-assay-b", stance=EvidenceStance.CONTRADICTS, reliability=0.9, relevance=0.85)
    )

    assert world_a.confidence > 0.8
    assert world_b.confidence < 0.4
    assert world_a.confidence - world_b.confidence > 0.4


def test_9b_the_same_sequence_is_reproducible():
    """Revision is deterministic: no sampling, no randomness, no clock dependence."""

    def run():
        belief = a_belief(0.5)
        for i, (rel, relv, stance) in enumerate(
            [
                (0.9, 0.5, EvidenceStance.SUPPORTS),
                (0.95, 0.9, EvidenceStance.CONTRADICTS),
                (0.7, 0.6, EvidenceStance.SUPPORTS),
            ]
        ):
            belief.consider(
                evidence(eid=f"EV-{i}", reliability=rel, relevance=relv, stance=stance)
            )
        return belief.confidence

    assert run() == run()
