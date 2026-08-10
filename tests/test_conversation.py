import pytest

from echo.conversation import SCHEMA_VERSION, Conversation, Message


def test_messages_accumulate_in_order():
    conversation = Conversation()
    conversation.add_user("hello")
    conversation.add_assistant("hi")
    conversation.add_user("still there?")

    assert len(conversation) == 3
    assert [m.role for m in conversation.messages] == ["user", "assistant", "user"]
    assert conversation.messages[0].content == "hello"


def test_invalid_role_rejected():
    with pytest.raises(ValueError):
        Message(role="system", content="nope")


def test_api_messages_exclude_the_system_prompt():
    conversation = Conversation(system="be terse")
    conversation.add_user("hello")

    payload = conversation.to_api_messages()

    assert payload == [{"role": "user", "content": "hello"}]
    assert all(set(m) == {"role", "content"} for m in payload)


def test_round_trip_through_dict_preserves_everything():
    original = Conversation(system="be terse")
    original.add_user("hello")
    original.add_assistant("hi")

    restored = Conversation.from_dict(original.to_dict())

    assert restored.id == original.id
    assert restored.system == original.system
    assert restored.created_at == original.created_at
    assert [m.to_dict() for m in restored.messages] == [
        m.to_dict() for m in original.messages
    ]


def test_unknown_schema_version_is_rejected():
    payload = Conversation().to_dict()
    payload["schema_version"] = SCHEMA_VERSION + 99

    with pytest.raises(ValueError):
        Conversation.from_dict(payload)


def test_ids_are_unique():
    assert Conversation().id != Conversation().id
