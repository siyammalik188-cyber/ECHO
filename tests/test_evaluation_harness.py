"""Tests for the evaluation harness itself.

The harness is a measuring instrument. If the matcher is lenient in the model's
favour, every number it produces is worthless — so these tests check that it
counts misses as misses and surplus as surplus.
"""

import json
from pathlib import Path

from evaluation.matching import band_deviation, in_band, matches, normalise
from evaluation.run_evaluation import CASES_PATH, build_conversation, load_cases
from evaluation.scoring import aggregate, calibration_summary, score_case


def memory(content, memory_type="fact", confidence=0.9, importance=0.8):
    return {
        "content": content,
        "memory_type": memory_type,
        "confidence": confidence,
        "importance": importance,
    }


# ------------------------------------------------------------------ matching


def test_normalise_strips_punctuation_and_case():
    assert normalise("ACME-4471!") == "acme4471"
    assert normalise("  Paediatric   Nurse.  ") == "paediatric nurse"


def test_every_group_must_contribute_a_hit():
    groups = [["dhaka"], ["lives", "resides"]]
    assert matches(groups, "The user lives in Dhaka.")
    assert not matches(groups, "The user lives in Berlin.")  # first group fails
    assert not matches(groups, "Dhaka is hot.")  # second group fails


def test_any_alternative_within_a_group_suffices():
    assert matches([["chose", "decided", "picked"]], "The user decided on Postgres.")


def test_an_empty_alternative_matches_anything():
    """Used by the negative cases to forbid producing any memory at all."""
    assert matches([[""]], "literally anything")


def test_no_groups_never_matches():
    assert not matches([], "anything")


def test_bands():
    assert in_band(0.5, (0.4, 0.6))
    assert not in_band(0.7, (0.4, 0.6))
    assert band_deviation(0.5, (0.4, 0.6)) == 0.0
    assert round(band_deviation(0.9, (0.4, 0.6)), 6) == 0.3
    assert round(band_deviation(0.1, (0.4, 0.6)), 6) == 0.3


# ------------------------------------------------------------------- scoring


def a_case(**overrides):
    case = {
        "id": "test-case",
        "requirement": "test",
        "title": "test",
        "expected": [
            {
                "label": "user lives in Dhaka",
                "memory_type": ["fact", "identity"],
                "must_include": [["dhaka"]],
                "confidence": [0.8, 1.0],
                "importance": [0.4, 1.0],
            }
        ],
        "forbidden": [{"label": "the train remark", "must_include": [["train"]]}],
        "tolerated": [{"label": "heat tolerance", "must_include": [["heat"]]}],
    }
    case.update(overrides)
    return case


def test_a_matching_proposal_is_a_true_positive():
    result = score_case(a_case(), [memory("The user lives in Dhaka.")])

    assert result.count("true_positive") == 1
    assert result.count("false_negative") == 0
    assert result.passed


def test_a_missing_expectation_is_a_false_negative():
    result = score_case(a_case(), [])

    assert result.count("false_negative") == 1
    assert not result.passed


def test_a_forbidden_proposal_is_a_labelled_false_positive():
    result = score_case(
        a_case(), [memory("The user lives in Dhaka."), memory("The user is on a train.")]
    )

    (fp,) = [j for j in result.judgements if j.kind == "false_positive"]
    assert fp.label == "the train remark"
    assert not result.passed


def test_an_unlabelled_surplus_proposal_is_still_a_false_positive():
    """Silence about a proposal is not permission for it."""
    result = score_case(
        a_case(),
        [memory("The user lives in Dhaka."), memory("The user enjoys long walks.")],
    )

    (fp,) = [j for j in result.judgements if j.kind == "false_positive"]
    assert fp.label is None
    assert "did not match" in fp.detail


def test_a_tolerated_proposal_is_neither_credited_nor_penalised():
    result = score_case(
        a_case(),
        [memory("The user lives in Dhaka."), memory("The user is used to the heat.")],
    )

    assert result.count("tolerated") == 1
    assert result.count("false_positive") == 0
    assert result.passed


def test_each_expectation_consumes_at_most_one_proposal():
    """Two near-identical proposals cannot both be credited for one expectation."""
    result = score_case(
        a_case(),
        [memory("The user lives in Dhaka."), memory("The user resides in Dhaka.")],
    )

    assert result.count("true_positive") == 1
    assert result.count("false_positive") == 1


def test_a_negative_case_is_clean_only_when_nothing_is_produced():
    negative = {
        "id": "neg",
        "requirement": "nothing",
        "title": "nothing",
        "expected": [],
        "forbidden": [{"label": "anything at all", "must_include": [[""]]}],
        "tolerated": [],
    }

    assert score_case(negative, []).passed
    assert not score_case(negative, [memory("Anything.")]).passed


def test_memory_type_accuracy_is_tracked_apart_from_precision():
    right = score_case(a_case(), [memory("Lives in Dhaka.", memory_type="fact")])
    wrong = score_case(a_case(), [memory("Lives in Dhaka.", memory_type="goal")])

    assert (right.type_hits, right.type_misses) == (1, 0)
    assert (wrong.type_hits, wrong.type_misses) == (0, 1)
    # A wrong type does not turn a true positive into a false positive.
    assert wrong.count("true_positive") == 1


# --------------------------------------------------------------- aggregation


def test_precision_and_recall_are_computed_from_the_judgements():
    results = [
        score_case(a_case(), [memory("Lives in Dhaka.")]),  # 1 TP
        score_case(a_case(), []),  # 1 FN
        score_case(a_case(), [memory("On a train.")]),  # 1 FP + 1 FN
    ]
    totals = aggregate(results)

    assert (totals.true_positives, totals.false_positives, totals.false_negatives) == (
        1,
        1,
        2,
    )
    assert totals.precision == 0.5
    assert totals.recall == 1 / 3


def test_strict_precision_counts_tolerated_against_the_model():
    results = [
        score_case(
            a_case(),
            [memory("Lives in Dhaka."), memory("Used to the heat.")],
        )
    ]
    totals = aggregate(results)

    assert totals.precision == 1.0
    assert totals.strict_precision == 0.5


def test_metrics_are_none_rather_than_zero_when_undefined():
    totals = aggregate([score_case({"id": "x", "requirement": "", "title": ""}, [])])
    assert totals.precision is None
    assert totals.recall is None


def test_calibration_flags_an_overconfident_extraction():
    case = a_case(
        expected=[
            {
                "label": "hedged move",
                "memory_type": ["fact"],
                "must_include": [["berlin"]],
                "confidence": [0.15, 0.7],
                "importance": [0.2, 0.8],
            }
        ],
        forbidden=[],
        tolerated=[],
    )
    result = score_case(case, [memory("Might move to Berlin.", confidence=0.95, importance=0.5)])

    summary = calibration_summary([result], "confidence")
    assert summary["in_band"] == 0
    assert summary["over_band"] == 1
    assert round(summary["mean_deviation"], 6) == 0.25


# ------------------------------------------------------------------- dataset


def test_the_committed_dataset_covers_all_ten_required_scenarios():
    cases = load_cases()["cases"]
    requirements = " ".join(c["requirement"] for c in cases)

    for n in range(1, 11):
        assert f"{n}." in requirements, f"requirement {n} is not covered"


def test_every_case_is_well_formed():
    for case in load_cases()["cases"]:
        assert case["conversation"], case["id"]
        assert case["conversation"][0]["role"] == "user", case["id"]
        for expectation in case.get("expected", []):
            assert expectation["must_include"], case["id"]
            assert len(expectation["confidence"]) == 2, case["id"]
            assert len(expectation["importance"]) == 2, case["id"]
            assert expectation["confidence"][0] <= expectation["confidence"][1]
            assert expectation["importance"][0] <= expectation["importance"][1]


def test_cases_convert_into_real_conversations():
    for case in load_cases()["cases"]:
        conversation = build_conversation(case)
        assert len(conversation) == len(case["conversation"])
        assert conversation.to_api_messages()[0]["role"] == "user"


def test_the_dataset_file_is_valid_json_on_disk():
    json.loads(Path(CASES_PATH).read_text(encoding="utf-8"))
