"""Temporary context is the third kind of state — and the one that must not persist."""

import json

from echo.agent import Agent
from echo.context import WorkingContext
from echo.conversation import Conversation
from echo.memory import Memory, MemoryType
from fakes import FakeExtractor, FakeLLM, candidate


def a_memory(content: str = "Sam lives in Dhaka.") -> Memory:
    return Memory(
        content=content,
        memory_type=MemoryType.FACT,
        source_conversation="conv123",
        confidence=0.9,
        importance=0.8,
    )


def test_an_empty_context_renders_as_nothing():
    context = WorkingContext()
    assert context.is_empty()
    assert context.render() is None


def test_notes_and_memories_are_rendered_in_labelled_sections():
    context = WorkingContext()
    context.add_memories([a_memory()])
    context.add_note("The user is in a hurry.")

    rendered = context.render()

    assert "Sam lives in Dhaka." in rendered
    assert "The user is in a hurry." in rendered
    assert "remember" in rendered.lower()


def test_blank_notes_are_ignored():
    context = WorkingContext()
    context.add_note("   ")
    assert context.is_empty()


def test_the_same_memory_is_not_staged_twice():
    context = WorkingContext()
    memory = a_memory()

    context.add_memories([memory])
    context.add_memories([memory])

    assert len(context.recalled) == 1


def test_clear_empties_it():
    context = WorkingContext()
    context.add_note("something")
    context.add_memories([a_memory()])

    context.clear()

    assert context.is_empty()


def test_context_has_no_serialization_at_all():
    """Not an oversight — it is what keeps temporary state from becoming durable."""
    context = WorkingContext()
    assert not hasattr(context, "to_dict")
    assert not hasattr(context, "save")


def test_saving_an_agent_writes_no_context_anywhere(tmp_path):
    agent = Agent(FakeLLM(), Conversation(), directory=tmp_path)
    agent.context.add_note("do not persist me")
    agent.context.add_memories([a_memory("secret scratch note")])

    agent.save()

    written = "\n".join(
        path.read_text(encoding="utf-8") for path in tmp_path.rglob("*.json")
    )
    assert "do not persist me" not in written
    assert "secret scratch note" not in written


def test_context_does_not_survive_a_restart(tmp_path):
    agent = Agent(FakeLLM(), Conversation(), directory=tmp_path)
    agent.context.add_note("this turn only")
    agent.save()
    conversation_id = agent.id
    del agent

    revived = Agent.resume(conversation_id, FakeLLM(), directory=tmp_path)

    assert revived.context.is_empty()


def test_context_is_per_turn_and_not_recorded_in_the_transcript(tmp_path):
    llm = FakeLLM(["Sure."])
    agent = Agent(llm, Conversation(), directory=tmp_path)
    agent.context.add_note("The user is in a hurry.")

    agent.send("Hello")

    messages, system = llm.calls[0]
    assert "The user is in a hurry." in system
    assert [m["content"] for m in messages] == ["Hello"]

    path = agent.save()
    assert "The user is in a hurry." not in path.read_text(encoding="utf-8")


def test_the_three_kinds_of_state_stay_separate(tmp_path):
    """History, memory, and context each hold their own thing and nothing else."""
    agent = Agent(
        FakeLLM(["Noted."]),
        Conversation(),
        directory=tmp_path,
        extractor=FakeExtractor([candidate("Sam lives in Dhaka.")]),
    )
    agent.send("I live in Dhaka.")
    agent.remember()
    agent.context.add_note("scratch")
    agent.save()

    # 1. History holds the verbatim exchange, and no memory objects.
    assert [m.content for m in agent.conversation.messages] == [
        "I live in Dhaka.",
        "Noted.",
    ]

    # 2. Memory holds the extracted statement, not the transcript.
    (memory,) = agent.memories.all()
    assert memory.content == "Sam lives in Dhaka."
    assert memory.content not in [m.content for m in agent.conversation.messages]

    # 3. Context holds the scratch note, and it is on neither of the other two.
    assert agent.context.notes == ["scratch"]
    on_disk = json.dumps(
        [json.loads(p.read_text(encoding="utf-8")) for p in tmp_path.rglob("*.json")]
    )
    assert "scratch" not in on_disk
