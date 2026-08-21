"""The four properties the memory system was asked to demonstrate.

Each test below maps to one requirement and is written to be read as evidence
rather than as coverage.
"""

from echo.agent import Agent
from echo.conversation import Conversation
from echo.memory import MemoryType
from echo.memory_store import MemoryStore
from fakes import FakeExtractor, FakeLLM, candidate


def a_talkative_agent(tmp_path, extractor=None) -> Agent:
    return Agent(
        FakeLLM(["Got it."]),
        Conversation(),
        directory=tmp_path,
        extractor=extractor,
    )


# 1 ------------------------------------------------ a conversation can create a memory


def test_1_a_conversation_can_create_a_memory(tmp_path):
    extractor = FakeExtractor(
        [candidate("Sam lives in Dhaka.", memory_type=MemoryType.FACT)]
    )
    agent = a_talkative_agent(tmp_path, extractor)

    agent.send("I live in Dhaka.")
    created = agent.remember()

    assert len(created) == 1
    memory = created[0]
    assert memory.content == "Sam lives in Dhaka."
    assert memory.memory_type is MemoryType.FACT
    assert memory.source_conversation == agent.id  # traceable to where it came from
    assert extractor.seen == [agent.id]  # extraction actually ran over this conversation
    assert len(agent.memories) == 1


# 2 -------------------------------------------- the memory survives a program restart


def test_2_memory_survives_restarting_the_program(tmp_path):
    extractor = FakeExtractor([candidate("Sam lives in Dhaka.")])
    agent = a_talkative_agent(tmp_path, extractor)
    agent.send("I live in Dhaka.")
    agent.remember()
    agent.save()

    # Simulate a restart: drop every object holding state. Only the files remain.
    conversation_id = agent.id
    del agent, extractor

    revived = MemoryStore.in_directory(tmp_path)

    assert len(revived) == 1
    memory = revived.all()[0]
    assert memory.content == "Sam lives in Dhaka."
    assert memory.source_conversation == conversation_id
    assert memory.confidence == 0.9
    assert memory.importance == 0.8


def test_2b_a_memory_outlives_the_conversation_that_produced_it(tmp_path):
    """Persistent memory is cross-conversation — that is what makes it persistent."""
    agent = a_talkative_agent(tmp_path, FakeExtractor([candidate("Sam lives in Dhaka.")]))
    agent.send("I live in Dhaka.")
    agent.remember()
    agent.save()
    first_conversation_id = agent.id
    del agent

    # A brand new conversation, sharing nothing but the directory.
    later = Agent(FakeLLM(["Sure."]), Conversation(), directory=tmp_path)

    assert later.id != first_conversation_id
    assert len(later.conversation) == 0  # no history carried over
    assert len(later.memories) == 1  # but the memory is there
    assert later.memories.all()[0].source_conversation == first_conversation_id


# 3 ----------------------------------------------------- the memory can be retrieved


def test_3_the_memory_can_be_retrieved_later(tmp_path):
    agent = a_talkative_agent(tmp_path, FakeExtractor([candidate("Sam lives in Dhaka.")]))
    agent.send("I live in Dhaka.")
    agent.remember()
    agent.save()
    del agent

    # New process, new conversation, asked about something recorded earlier.
    later = Agent(FakeLLM(["Dhaka."]), Conversation(), directory=tmp_path)
    found = later.recall("where does Sam live")

    assert [m.content for m in found] == ["Sam lives in Dhaka."]
    assert found[0].access_count == 1  # retrieval was recorded on the record
    assert found[0].last_accessed is not None


def test_3b_a_recalled_memory_reaches_the_model_without_becoming_history(tmp_path):
    agent = a_talkative_agent(tmp_path, FakeExtractor([candidate("Sam lives in Dhaka.")]))
    agent.send("I live in Dhaka.")
    agent.remember()

    llm = FakeLLM(["Dhaka."])
    later = Agent(llm, Conversation(system="Be brief."), directory=tmp_path)
    later.recall("where does Sam live")
    later.send("Where do I live?")

    messages, system = llm.calls[0]
    # The memory shaped the request...
    assert "Sam lives in Dhaka." in system
    assert "Be brief." in system
    # ...but it is not a message, so it never became conversation history.
    assert [m["content"] for m in messages] == ["Where do I live?"]
    assert all(
        "Sam lives in Dhaka." not in m.content for m in later.conversation.messages
    )


# 4 ------------------------------- ordinary conversation text is not automatic memory


def test_4_ordinary_conversation_does_not_automatically_become_memory(tmp_path):
    extractor = FakeExtractor([candidate("Sam lives in Dhaka.")])
    agent = a_talkative_agent(tmp_path, extractor)

    agent.send("Hi there.")
    agent.send("What's the weather like?")
    agent.send("Thanks, bye.")
    agent.save()

    # Three exchanges are on the transcript...
    assert len(agent.conversation) == 6
    # ...and none of it became memory. send() never calls the extractor.
    assert len(agent.memories) == 0
    assert extractor.seen == []

    # Not even after a save-and-reload, which persists the transcript only.
    assert len(MemoryStore.in_directory(tmp_path)) == 0


def test_4b_extraction_over_small_talk_produces_nothing(tmp_path):
    """Running extraction is not the same as agreeing to remember."""
    silent_extractor = FakeExtractor([])  # the model found nothing worth keeping
    agent = a_talkative_agent(tmp_path, silent_extractor)

    agent.send("Hi there.")
    created = agent.remember()

    assert created == []
    assert silent_extractor.seen == [agent.id]  # it ran; it just declined
    assert len(agent.memories) == 0


def test_4c_low_value_candidates_are_rejected_by_echo_not_by_the_extractor(tmp_path):
    """The threshold gate is ECHO's own code, applied to whatever is proposed."""
    eager_extractor = FakeExtractor(
        [
            candidate("Sam said hello.", importance=0.05),
            candidate("Sam may possibly own a bicycle.", confidence=0.2),
        ]
    )
    agent = a_talkative_agent(tmp_path, eager_extractor)

    agent.send("Hi there, I think I might own a bicycle?")
    created = agent.remember()

    assert created == []
    assert len(agent.memories) == 0


def test_4d_creating_a_memory_requires_an_extractor(tmp_path):
    """There is no default path that quietly turns conversation into memory."""
    agent = a_talkative_agent(tmp_path, extractor=None)
    agent.send("I live in Dhaka.")

    try:
        agent.remember()
    except RuntimeError as exc:
        assert "extractor" in str(exc)
    else:
        raise AssertionError("expected remember() to refuse without an extractor")

    assert len(agent.memories) == 0
