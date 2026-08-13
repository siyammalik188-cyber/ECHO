"""What ECHO 10 has to be true for: listening without believing.

The load-bearing properties are that a parsed claim is not an accepted one,
that reliability is derived from outcomes and stays revisable, that an
anti-correlated source is used in reverse rather than ignored, and that a
majority can be outvoted by evidence.
"""

from __future__ import annotations

import pytest

from echo.social import (
    MAX_RELIABILITY,
    MIN_RELIABILITY,
    PRIOR_RELIABILITY,
    Claim,
    SocialError,
    SocialLedger,
    SourceRecord,
    aggregate,
    parse_claim,
)

from source_audit import imports_of

from experiments import run_social as R
from experiments.social_worlds import AGENTS, AGENTS_BY_NAME, MINORITY_ROUNDS, generate


@pytest.fixture(scope="module")
def study():
    return R.run_study()


# ------------------------------------------ 10.1 claims are parsed, not believed


def test_1_a_claim_is_parsed_into_its_four_parts():
    claim = parse_claim(
        "AGENT-A",
        "I think hypothesis 2 is more likely because observation X occurred",
    )
    assert claim.source == "AGENT-A"
    assert claim.proposition == "hypothesis-2"
    assert claim.asserts is True
    assert claim.confidence == pytest.approx(0.65)
    assert claim.evidence == ("observation X occurred",)


def test_1b_denial_is_distinguished_from_assertion():
    denial = parse_claim(
        "AGENT-B", "I am sure hypothesis 3 is not the right one because Y happened"
    )
    assert denial.asserts is False
    assert denial.confidence == pytest.approx(0.90)


def test_1c_negation_inside_the_evidence_does_not_flip_the_claim():
    """'because X did not occur' is a reason, not a denial of the claim."""
    claim = parse_claim(
        "AGENT-A", "I am confident hypothesis 4 holds, because observation X did not occur"
    )
    assert claim.asserts is True


def test_1d_an_unparseable_utterance_is_refused_not_guessed():
    for bad in ("", "   ", "the weather is nice today", "I think so"):
        with pytest.raises(SocialError):
            parse_claim("AGENT-A", bad)


def test_1e_parsing_does_not_believe(study=None):
    ledger = SocialLedger()
    claim = parse_claim("AGENT-X", "I am sure hypothesis 9 is more likely because Z")
    ledger.hear(claim)
    # heard, recorded, and worth exactly nothing until the source has a record
    assert ledger.reliability("AGENT-X") == pytest.approx(PRIOR_RELIABILITY)
    assert ledger.believe("hypothesis-9").probability == pytest.approx(0.5, abs=1e-9)


def test_1f_no_model_is_consulted_by_the_parser():
    import echo.social as module

    for line in imports_of(module.__file__):
        for forbidden in ("anthropic", "llm", "openai", "requests", "urllib"):
            assert forbidden not in line.lower()


# ------------------------------------------------ 10.2 reliability is learned


def test_2_reliability_is_learned_from_outcomes(study):
    ledger = study.ledger
    for agent in AGENTS:
        learned = ledger.reliability(agent.name)
        assert abs(learned - agent.accuracy) < 0.12, f"{agent.name}: {learned}"


def test_2b_the_learned_ordering_matches_the_true_ordering(study):
    learned = [name for name, _ in study.ledger.ranked()]
    true_order = [a.name for a in sorted(AGENTS, key=lambda a: -a.accuracy)]
    assert learned == true_order


def test_2c_an_unknown_source_sits_at_the_prior():
    ledger = SocialLedger()
    assert ledger.reliability("NEVER-HEARD-OF") == pytest.approx(PRIOR_RELIABILITY)


def test_2d_reliability_stays_revisable():
    """A source that turns must be trackable, not locked in by its history."""
    record = SourceRecord("AGENT-Z")
    for tick in range(30):
        record = record.with_outcome(tick, True)
    assert record.reliability > 0.9
    for tick in range(30, 60):
        record = record.with_outcome(tick, False)
    assert record.reliability < 0.6
    assert record.recent_reliability(window=20) == pytest.approx(0.0)


def test_2e_reliability_is_clamped_away_from_certainty():
    record = SourceRecord("AGENT-P")
    for tick in range(500):
        record = record.with_outcome(tick, True)
    assert record.reliability <= MAX_RELIABILITY
    other = SourceRecord("AGENT-Q")
    for tick in range(500):
        other = other.with_outcome(tick, False)
    assert other.reliability >= MIN_RELIABILITY


def test_2f_reliability_cannot_be_asserted_only_earned():
    ledger = SocialLedger()
    assert not hasattr(ledger, "set_reliability")
    record = ledger.record("AGENT-A")
    with pytest.raises(Exception):
        record.correct = 99  # type: ignore[misc]


# ---------------------------------------------------- 10.3 contradiction


def test_3_contradictory_claims_are_both_retained():
    ledger = SocialLedger()
    ledger.hear(parse_claim("AGENT-A", "I am sure hypothesis 5 holds because X"))
    ledger.hear(parse_claim("AGENT-B", "I am sure hypothesis 5 is not right because Y"))
    assert "hypothesis-5" in ledger.contradictions()
    assert len(ledger.claims("hypothesis-5")) == 2


def test_3b_a_coin_flip_source_moves_nothing():
    claims = [parse_claim("AGENT-M", "I am sure hypothesis 6 holds because X")]
    result = aggregate(claims, {"AGENT-M": 0.5})
    assert result.probability == pytest.approx(0.5, abs=1e-9)


def test_3c_one_reliable_source_can_outweigh_three_poor_ones():
    claims = [
        parse_claim("A", "I am sure hypothesis 7 holds because X"),
        parse_claim("B", "I am sure hypothesis 7 holds because X"),
        parse_claim("C", "I am sure hypothesis 7 holds because X"),
        parse_claim("D", "I am sure hypothesis 7 is not right because X"),
    ]
    result = aggregate(claims, {"A": 0.55, "B": 0.55, "C": 0.55, "D": 0.97})
    assert result.asserting == 3 and result.denying == 1
    assert result.majority_asserts is True
    assert result.probability < 0.5
    assert result.followed_majority is False


# -------------------------------------------------- 10.4 deception control


def test_4_the_deceptive_agent_is_discovered(study):
    """Never labelled; found from outcomes."""
    learned = study.ledger.reliability("AGENT-D")
    assert learned < 0.3
    assert learned < min(
        study.ledger.reliability(a.name) for a in AGENTS if a.name != "AGENT-D"
    )


def test_4b_an_anti_correlated_source_is_used_in_reverse():
    """Below 0.5 means negative weight — informative, not merely discounted."""
    claims = [parse_claim("LIAR", "I am sure hypothesis 8 holds because X")]
    result = aggregate(claims, {"LIAR": 0.05})
    assert result.probability < 0.2
    assert result.contributions[0].weight < 0


def test_4c_no_agent_name_or_profile_reaches_the_social_module():
    import pathlib

    import echo.social as module

    source = pathlib.Path(module.__file__).read_text(encoding="utf-8")
    for agent in AGENTS:
        assert agent.name not in source
    for line in imports_of(module.__file__):
        assert "social_worlds" not in line
        assert "experiments" not in line


# ------------------------------------------ 10.5 evidence beats headcount


def test_5_minority_correct_rounds_are_won(study):
    minority = study.minority_results()
    assert len(minority) >= 5
    echo_right = sum(1 for r in minority if r.echo_correct)
    majority_right = sum(1 for r in minority if r.majority_correct)
    assert majority_right == 0  # by construction
    assert echo_right > len(minority) / 2


def test_5b_echo_beats_every_baseline(study):
    echo = study.accuracy("echo_correct")
    for baseline in ("majority_correct", "confidence_correct", "loudest_correct"):
        assert echo > study.accuracy(baseline), baseline


def test_5c_following_the_loudest_is_actively_harmful(study):
    """The most reliable agent is the most hedged; firmness is not competence."""
    assert study.accuracy("loudest_correct") < 0.5
    loudest = max(AGENTS, key=lambda a: a.assertiveness)
    best = max(AGENTS, key=lambda a: a.accuracy)
    assert loudest.name != best.name


def test_5d_echo_disagrees_with_the_majority_and_is_usually_right(study):
    against = [r for r in study.scored() if r.echo_says != r.majority_says]
    assert against
    assert sum(1 for r in against if r.echo_correct) / len(against) > 0.8


# ---------------------------------------- 10.6 combining with the rest


def test_6_the_belief_records_where_it_came_from(study):
    result = study.minority_results()[0]
    assert len(result.aggregate.contributions) == len(AGENTS)
    for contribution in result.aggregate.contributions:
        assert contribution.source in AGENTS_BY_NAME
        assert 0.0 <= contribution.reliability <= 1.0
    payload = result.aggregate.to_dict()
    assert "contributions" in payload and "followed_majority" in payload


def test_6b_the_ledger_survives_restart(tmp_path, study):
    path = tmp_path / "social.json"
    study.ledger.save(path)
    reloaded = SocialLedger.load(path)
    assert len(reloaded) == len(study.ledger)
    for agent in AGENTS:
        assert reloaded.reliability(agent.name) == pytest.approx(
            study.ledger.reliability(agent.name)
        )
    assert reloaded.contradictions() == study.ledger.contradictions()


def test_6c_the_study_is_reproducible():
    first = R.run_study()
    second = R.run_study()
    assert first.ledger.ranked() == second.ledger.ranked()
    assert [r.echo_probability for r in first.results] == [
        r.echo_probability for r in second.results
    ]


def test_6d_claims_are_immutable():
    claim = parse_claim("AGENT-A", "I am sure hypothesis 1 holds because X")
    with pytest.raises(Exception):
        claim.confidence = 0.1  # type: ignore[misc]


def test_6e_every_generated_utterance_parses():
    rounds = generate()
    for round_ in rounds:
        for name, sentence in round_.utterances:
            claim = parse_claim(name, sentence)
            assert claim.proposition == round_.proposition
            assert claim.source == name
