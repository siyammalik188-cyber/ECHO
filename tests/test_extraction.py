"""Tests for the deliberate extraction step and the thresholds behind it."""

import pytest

from echo.conversation import Conversation
from echo.extraction import (
    EXTRACTION_SCHEMA,
    LLMMemoryExtractor,
    MemoryCandidate,
    consolidate,
    render_transcript,
)
from echo.memory import MemoryType
from echo.memory_store import MemoryStore
from fakes import FakeExtractor, FakeStructuredLLM, candidate


def a_conversation() -> Conversation:
    conversation = Conversation()
    conversation.add_user("I live in Dhaka.")
    conversation.add_assistant("Noted.")
    return conversation


# ------------------------------------------------------------- the candidate


@pytest.mark.parametrize("field", ["confidence", "importance"])
def test_candidate_scores_are_validated(field):
    with pytest.raises(ValueError, match=field):
        candidate("Sam lives in Dhaka.", **{field: 1.5})


def test_candidate_rejects_empty_content():
    with pytest.raises(ValueError):
        MemoryCandidate(
            content="", memory_type=MemoryType.FACT, confidence=0.9, importance=0.9
        )


# --------------------------------------------------------- the LLM extractor


def test_extraction_is_a_separate_call_with_its_own_prompt_and_schema():
    llm = FakeStructuredLLM({"memories": []})

    LLMMemoryExtractor(llm).extract(a_conversation())

    (messages, schema, system) = llm.calls[0]
    assert schema == EXTRACTION_SCHEMA
    assert "extract" in (system or "").lower()
    # The transcript is handed over as material to review, not continued.
    assert "<transcript>" in messages[0]["content"]
    assert messages[0]["role"] == "user"


def test_extractor_builds_candidates_from_the_payload():
    llm = FakeStructuredLLM(
        {
            "memories": [
                {
                    "content": "Sam lives in Dhaka.",
                    "memory_type": "fact",
                    "confidence": 0.95,
                    "importance": 0.8,
                }
            ]
        }
    )

    (result,) = LLMMemoryExtractor(llm).extract(a_conversation())

    assert result.content == "Sam lives in Dhaka."
    assert result.memory_type is MemoryType.FACT
    assert result.confidence == 0.95


def test_an_empty_extraction_is_a_normal_result():
    llm = FakeStructuredLLM({"memories": []})
    assert LLMMemoryExtractor(llm).extract(a_conversation()) == []


def test_malformed_rows_are_dropped_not_stored_badly():
    llm = FakeStructuredLLM(
        {
            "memories": [
                {"content": "missing its scores", "memory_type": "fact"},
                {
                    "content": "bad type",
                    "memory_type": "nonsense",
                    "confidence": 0.9,
                    "importance": 0.9,
                },
                {
                    "content": "out of range",
                    "memory_type": "fact",
                    "confidence": 5,
                    "importance": 0.9,
                },
                {
                    "content": "Sam lives in Dhaka.",
                    "memory_type": "fact",
                    "confidence": 0.95,
                    "importance": 0.8,
                },
            ]
        }
    )

    results = LLMMemoryExtractor(llm).extract(a_conversation())

    assert [r.content for r in results] == ["Sam lives in Dhaka."]


def test_an_empty_conversation_is_not_sent_to_the_model():
    llm = FakeStructuredLLM({"memories": []})

    assert LLMMemoryExtractor(llm).extract(Conversation()) == []
    assert llm.calls == []


def test_transcript_rendering_labels_every_turn():
    assert render_transcript(a_conversation()) == (
        "user: I live in Dhaka.\nassistant: Noted."
    )


# ------------------------------------------------------------- the threshold


def test_consolidate_stores_candidates_that_clear_the_thresholds(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    conversation = a_conversation()
    extractor = FakeExtractor([candidate("Sam lives in Dhaka.")])

    created = consolidate(conversation, extractor, store)

    assert [m.content for m in created] == ["Sam lives in Dhaka."]
    assert len(store) == 1


def test_the_stored_memory_points_back_at_its_conversation(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    conversation = a_conversation()

    (created,) = consolidate(
        conversation, FakeExtractor([candidate("Sam lives in Dhaka.")]), store
    )

    assert created.source_conversation == conversation.id


def test_low_confidence_candidates_are_rejected(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    extractor = FakeExtractor([candidate("Sam might live in Dhaka.", confidence=0.2)])

    assert consolidate(a_conversation(), extractor, store) == []
    assert len(store) == 0


def test_low_importance_candidates_are_rejected(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    extractor = FakeExtractor([candidate("Sam said hello.", importance=0.05)])

    assert consolidate(a_conversation(), extractor, store) == []
    assert len(store) == 0


def test_thresholds_are_applied_here_not_left_to_the_extractor(tmp_path):
    """A confident, important-looking candidate still fails a raised bar."""
    store = MemoryStore(tmp_path / "memories.json")
    extractor = FakeExtractor([candidate("Sam lives in Dhaka.", importance=0.5)])

    assert consolidate(a_conversation(), extractor, store, min_importance=0.9) == []
    assert len(store) == 0


def test_rerunning_consolidation_does_not_duplicate(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    conversation = a_conversation()
    extractor = FakeExtractor([candidate("Sam lives in Dhaka.")])

    first = consolidate(conversation, extractor, store)
    second = consolidate(conversation, extractor, store)

    assert len(first) == 1
    assert second == []  # already held; nothing new was created
    assert len(store) == 1
