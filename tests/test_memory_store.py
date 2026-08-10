import json

import pytest

from echo.memory import Memory, MemoryType
from echo.memory_store import MemoryStore


def make(content: str, **overrides) -> Memory:
    fields = dict(
        content=content,
        memory_type=MemoryType.FACT,
        source_conversation="conv123",
        confidence=0.9,
        importance=0.7,
    )
    fields.update(overrides)
    return Memory(**fields)


def test_a_missing_file_is_an_empty_store_not_an_error(tmp_path):
    store = MemoryStore.load(tmp_path / "nothing.json")
    assert len(store) == 0
    assert store.all() == []


def test_save_then_load_round_trips(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    store.add(make("Sam lives in Dhaka."))
    store.add(make("Sam prefers short answers.", memory_type=MemoryType.PREFERENCE))
    store.save()

    reopened = MemoryStore.load(tmp_path / "memories.json")

    assert len(reopened) == 2
    assert {m.content for m in reopened} == {
        "Sam lives in Dhaka.",
        "Sam prefers short answers.",
    }


def test_saved_file_is_readable_json_with_a_schema_version(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    store.add(make("Sam lives in Dhaka."))
    path = store.save()

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["memories"][0]["content"] == "Sam lives in Dhaka."


def test_an_unknown_schema_version_is_rejected(tmp_path):
    path = tmp_path / "memories.json"
    path.write_text(json.dumps({"schema_version": 99, "memories": []}), encoding="utf-8")

    with pytest.raises(ValueError):
        MemoryStore.load(path)


def test_saving_leaves_no_temp_file(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    store.add(make("Sam lives in Dhaka."))
    store.save()
    store.save()

    assert list(tmp_path.glob("*.tmp")) == []


def test_identical_content_is_not_stored_twice(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")

    assert store.add(make("Sam lives in Dhaka.")) is True
    assert store.add(make("  sam LIVES in dhaka.  ")) is False
    assert len(store) == 1


def test_search_finds_by_word_overlap(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    store.add(make("Sam lives in Dhaka."))
    store.add(make("Sam is allergic to peanuts."))

    hits = store.search("Where does Sam live?")

    assert [m.content for m in hits][0] == "Sam lives in Dhaka."


def test_search_ranks_stronger_overlap_first(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    store.add(make("Sam works on a Python project."))
    store.add(make("Sam prefers Python over Ruby for scripting."))

    hits = store.search("python scripting preference")

    assert hits[0].content == "Sam prefers Python over Ruby for scripting."


def test_search_returns_nothing_for_an_unrelated_query(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    store.add(make("Sam lives in Dhaka."))

    assert store.search("quantum chromodynamics") == []


def test_search_ignores_stopword_only_queries(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    store.add(make("Sam lives in Dhaka."))

    assert store.search("the and of") == []


def test_search_respects_the_limit(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    for i in range(5):
        store.add(make(f"Sam owns cat number {i}."))

    assert len(store.search("Sam cat", limit=2)) == 2


def test_retrieval_counts_as_an_access(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    store.add(make("Sam lives in Dhaka."))

    (found,) = store.search("Dhaka")

    assert found.access_count == 1
    assert found.last_accessed is not None


def test_access_counts_survive_a_save_and_reload(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    store.add(make("Sam lives in Dhaka."))
    store.search("Dhaka")
    store.search("Dhaka")
    store.save()

    (reloaded,) = MemoryStore.load(tmp_path / "memories.json").all()

    assert reloaded.access_count == 2


def test_listing_and_peeking_do_not_count_as_accesses(tmp_path):
    store = MemoryStore(tmp_path / "memories.json")
    store.add(make("Sam lives in Dhaka."))

    store.all()
    store.search("Dhaka", touch=False)

    assert store.all()[0].access_count == 0


def test_in_directory_uses_the_conventional_filename(tmp_path):
    store = MemoryStore.in_directory(tmp_path)
    store.add(make("Sam lives in Dhaka."))
    store.save()

    assert (tmp_path / "memories.json").is_file()
