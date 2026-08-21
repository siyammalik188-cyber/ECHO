import pytest

from echo import storage
from echo.agent import Agent
from echo.conversation import Conversation
from fakes import ExplodingLLM, FakeLLM


def test_send_records_both_turns_and_returns_the_reply(tmp_path):
    agent = Agent(FakeLLM(["hi there"]), directory=tmp_path)

    reply = agent.send("hello")

    assert reply == "hi there"
    assert [(m.role, m.content) for m in agent.conversation.messages] == [
        ("user", "hello"),
        ("assistant", "hi there"),
    ]


def test_full_history_is_sent_on_every_turn(tmp_path):
    llm = FakeLLM(["one", "two"])
    agent = Agent(llm, directory=tmp_path)

    agent.send("first")
    agent.send("second")

    first_call, second_call = (messages for messages, _ in llm.calls)
    assert first_call == [{"role": "user", "content": "first"}]
    assert second_call == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "second"},
    ]


def test_system_prompt_is_passed_separately(tmp_path):
    llm = FakeLLM()
    agent = Agent(llm, Conversation(system="be terse"), directory=tmp_path)

    agent.send("hello")

    _, system = llm.calls[0]
    assert system == "be terse"


def test_a_failed_call_rolls_back_the_user_turn(tmp_path):
    agent = Agent(ExplodingLLM(), directory=tmp_path)

    with pytest.raises(RuntimeError):
        agent.send("hello")

    assert agent.conversation.messages == []


def test_save_and_resume_continues_the_same_conversation(tmp_path):
    agent = Agent(FakeLLM(["one"]), directory=tmp_path)
    agent.send("first")
    agent.save()

    llm = FakeLLM(["two"])
    resumed = Agent.resume(agent.id, llm, directory=tmp_path)

    assert resumed.id == agent.id
    assert len(resumed.conversation) == 2

    resumed.send("second")

    sent, _ = llm.calls[0]
    assert sent == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "one"},
        {"role": "user", "content": "second"},
    ]
    assert len(resumed.conversation) == 4


def test_resume_after_a_process_restart_sees_what_was_on_disk(tmp_path):
    agent = Agent(FakeLLM(["one"]), directory=tmp_path)
    agent.send("first")
    agent.save()
    del agent  # nothing in memory survives; only the file does

    ids = storage.list_ids(storage.conversations_dir(tmp_path))
    assert len(ids) == 1

    resumed = Agent.resume(ids[0], FakeLLM(), directory=tmp_path)
    assert [m.content for m in resumed.conversation.messages] == ["first", "one"]


def test_listing_conversations_never_picks_up_the_memory_store(tmp_path):
    """The two kinds of durable state do not share a filename namespace."""
    agent = Agent(FakeLLM(["one"]), directory=tmp_path)
    agent.send("first")
    agent.save()

    assert (tmp_path / "memories.json").is_file()
    assert storage.list_ids(storage.conversations_dir(tmp_path)) == [agent.id]
