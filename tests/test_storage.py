import json

import pytest

from echo import storage
from echo.conversation import Conversation


def test_save_then_load_returns_an_equal_conversation(tmp_path):
    original = Conversation(system="be terse")
    original.add_user("hello")
    original.add_assistant("hi")

    path = storage.save(original, tmp_path)
    restored = storage.load(original.id, tmp_path)

    assert path.exists()
    assert restored.to_dict() == original.to_dict()


def test_saved_file_is_readable_json(tmp_path):
    conversation = Conversation()
    conversation.add_user("hello")

    path = storage.save(conversation, tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["id"] == conversation.id
    assert payload["messages"][0]["content"] == "hello"


def test_save_creates_missing_directories(tmp_path):
    target = tmp_path / "deep" / "nested"
    storage.save(Conversation(), target)
    assert target.is_dir()


def test_saving_twice_overwrites_and_leaves_no_temp_file(tmp_path):
    conversation = Conversation()
    conversation.add_user("first")
    storage.save(conversation, tmp_path)

    conversation.add_assistant("second")
    storage.save(conversation, tmp_path)

    assert storage.list_ids(tmp_path) == [conversation.id]
    assert list(tmp_path.glob("*.tmp")) == []
    assert len(storage.load(conversation.id, tmp_path)) == 2


def test_list_ids_is_sorted_and_empty_for_missing_directory(tmp_path):
    assert storage.list_ids(tmp_path / "nothing-here") == []

    for cid in ("bbb", "aaa", "ccc"):
        storage.save(Conversation(id=cid), tmp_path)

    assert storage.list_ids(tmp_path) == ["aaa", "bbb", "ccc"]


def test_loading_an_unknown_id_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        storage.load("nope", tmp_path)


def test_exists(tmp_path):
    conversation = Conversation()
    assert not storage.exists(conversation.id, tmp_path)
    storage.save(conversation, tmp_path)
    assert storage.exists(conversation.id, tmp_path)


@pytest.mark.parametrize("bad_id", ["../escape", "with/slash", "sp ace", ""])
def test_ids_that_would_escape_the_directory_are_rejected(bad_id, tmp_path):
    with pytest.raises(ValueError):
        storage.path_for(bad_id, tmp_path)
