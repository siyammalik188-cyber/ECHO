"""Unit tests for the belief record, evidence record, and the update rule."""

import math

import pytest

from echo.belief import (
    MAX_CONFIDENCE,
    MIN_CONFIDENCE,
    Belief,
    Evidence,
    EvidenceStance,
    Revision,
    logit,
    sigmoid,
)


def evidence(**overrides) -> Evidence:
    fields = dict(
        description="Something was observed.",
        source="station log",
        reliability=0.9,
        relevance=0.8,
        stance=EvidenceStance.SUPPORTS,
    )
    fields.update(overrides)
    return Evidence(**fields)


# ------------------------------------------------------------------ evidence


def test_evidence_carries_every_required_field():
    item = evidence()
    for attribute in ("id", "description", "source", "reliability", "relevance", "stance"):
        assert hasattr(item, attribute), attribute


def test_weight_is_reliability_times_relevance():
    assert evidence(reliability=0.5, relevance=0.4).weight == pytest.approx(0.2)


@pytest.mark.parametrize("field", ["reliability", "relevance"])
@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_evidence_scores_must_be_within_zero_and_one(field, value):
    with pytest.raises(ValueError, match=field):
        evidence(**{field: value})


def test_evidence_requires_a_description_and_a_source():
    with pytest.raises(ValueError):
        evidence(description="  ")
    with pytest.raises(ValueError):
        evidence(source="")


def test_a_string_stance_is_coerced():
    assert evidence(stance="contradicts").stance is EvidenceStance.CONTRADICTS


def test_evidence_is_frozen():
    item = evidence()
    with pytest.raises(Exception):
        item.reliability = 0.1  # type: ignore[misc]


def test_evidence_round_trips_through_dict():
    item = evidence()
    assert Evidence.from_dict(item.to_dict()).to_dict() == item.to_dict()


# --------------------------------------------------------------- the maths


def test_logit_and_sigmoid_are_inverses():
    for p in (0.1, 0.25, 0.5, 0.75, 0.9):
        assert sigmoid(logit(p)) == pytest.approx(p)


def test_confidence_never_reaches_zero_or_one():
    """A belief can become very unlikely, never impossible or unfalsifiable."""
    belief = Belief("p", confidence=0.5)
    for i in range(40):
        belief.consider(evidence(id=f"EV-down-{i}", stance=EvidenceStance.CONTRADICTS))
    assert belief.confidence >= MIN_CONFIDENCE
    assert belief.confidence > 0.0

    rising = Belief("p", confidence=0.5)
    for i in range(40):
        rising.consider(evidence(id=f"EV-up-{i}", stance=EvidenceStance.SUPPORTS))
    assert rising.confidence <= MAX_CONFIDENCE
    assert rising.confidence < 1.0


def test_evidence_of_equal_weight_and_opposite_stance_cancels_out():
    belief = Belief("p", confidence=0.6)
    start = belief.confidence

    belief.consider(evidence(id="EV-up", stance=EvidenceStance.SUPPORTS, reliability=0.8, relevance=0.7))
    belief.consider(evidence(id="EV-dn", stance=EvidenceStance.CONTRADICTS, reliability=0.8, relevance=0.7))

    assert belief.confidence == pytest.approx(start, abs=1e-9)


def test_order_does_not_change_the_destination():
    """Log-odds addition is commutative — noted as a limitation in the report."""

    def run(order):
        belief = Belief("p", confidence=0.5)
        for eid, stance, rel, relv in order:
            belief.consider(
                evidence(id=eid, stance=stance, reliability=rel, relevance=relv)
            )
        return belief.confidence

    items = [
        ("a", EvidenceStance.SUPPORTS, 0.9, 0.6),
        ("b", EvidenceStance.CONTRADICTS, 0.7, 0.8),
        ("c", EvidenceStance.SUPPORTS, 0.5, 0.5),
    ]
    assert run(items) == pytest.approx(run(list(reversed(items))))


# ------------------------------------------------------------------- belief


def test_a_belief_needs_a_proposition():
    with pytest.raises(ValueError):
        Belief("   ")


def test_initial_confidence_must_be_a_probability():
    with pytest.raises(ValueError, match="confidence"):
        Belief("p", confidence=1.4)


def test_revision_count_excludes_considerations_that_did_not_apply():
    belief = Belief("p", confidence=0.5)
    belief.consider(evidence(id="EV-applied"))
    belief.consider(evidence(id="EV-ignored", relevance=0.01))

    assert belief.considered_count == 2
    assert belief.revision_count == 1


def test_updated_at_advances_but_created_at_does_not():
    belief = Belief("p", confidence=0.5)
    created, first_update = belief.created_at, belief.updated_at
    belief.consider(evidence(id="EV-1"))

    assert belief.created_at == created
    assert belief.updated_at >= first_update


def test_initial_confidence_reports_the_pre_evidence_value():
    belief = Belief("p", confidence=0.4)
    belief.consider(evidence(id="EV-1"))
    belief.consider(evidence(id="EV-2"))

    assert belief.initial_confidence == 0.4
    assert belief.confidence != 0.4


def test_evidence_lists_are_copies():
    belief = Belief("p", confidence=0.5)
    belief.consider(evidence(id="EV-1"))

    assert isinstance(belief.supporting_evidence, tuple)
    assert isinstance(belief.contradicting_evidence, tuple)


def test_a_revision_snapshot_keeps_the_evidence_readable_without_a_lookup():
    belief = Belief("p", confidence=0.5)
    belief.consider(evidence(id="EV-1", source="dual-sensor telemetry"))

    snapshot = belief.revision_history[0].evidence_snapshot
    assert snapshot["source"] == "dual-sensor telemetry"
    assert snapshot["weight"] == pytest.approx(0.72)


def test_the_explanation_names_the_weighting_and_the_movement():
    belief = Belief("p", confidence=0.5)
    revision = belief.consider(
        evidence(id="EV-1", reliability=0.95, relevance=0.90, stance=EvidenceStance.CONTRADICTS)
    )

    reason = revision.reason
    assert "EV-1" in reason
    assert "contradicts" in reason
    assert "reliable" in reason or "pertinent" in reason
    assert "0.500" in reason  # where it came from
    assert "→" in reason  # where it went


def test_weak_source_and_weak_relevance_are_explained_differently():
    unreliable = Belief("p", 0.6).consider(
        evidence(id="EV-a", reliability=0.2, relevance=0.9)
    )
    tangential = Belief("p", 0.6).consider(
        evidence(id="EV-b", reliability=0.9, relevance=0.25)
    )

    assert "source is weak" in unreliable.reason
    assert "glances at the proposition" in tangential.reason


def test_belief_round_trips_through_dict():
    belief = Belief("p", confidence=0.5)
    belief.consider(evidence(id="EV-1"))
    belief.consider(evidence(id="EV-2", stance=EvidenceStance.CONTRADICTS))
    belief.consider(evidence(id="EV-3", relevance=0.01))

    restored = Belief.from_dict(belief.to_dict())

    assert restored.to_dict() == belief.to_dict()
    assert restored.confidence == belief.confidence
    assert restored.revision_count == belief.revision_count
    assert restored.supporting_evidence == belief.supporting_evidence
    assert restored.contradicting_evidence == belief.contradicting_evidence


def test_the_serialised_shape_uses_the_required_field_names():
    belief = Belief("p", confidence=0.5)
    belief.consider(evidence(id="EV-1"))
    payload = belief.to_dict()

    for key in (
        "belief_id",
        "proposition",
        "confidence",
        "supporting_evidence",
        "contradicting_evidence",
        "created_at",
        "updated_at",
        "revision_count",
        "revision_history",
    ):
        assert key in payload, key


def test_revision_round_trips_through_dict():
    belief = Belief("p", confidence=0.5)
    belief.consider(evidence(id="EV-1"))
    original = belief.revision_history[0]

    assert Revision.from_dict(original.to_dict()).to_dict() == original.to_dict()
