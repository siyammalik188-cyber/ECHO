import pytest

from echo.memory import Memory, MemoryType


def make(**overrides) -> Memory:
    fields = dict(
        content="Sam lives in Dhaka.",
        memory_type=MemoryType.FACT,
        source_conversation="conv123",
        confidence=0.9,
        importance=0.7,
    )
    fields.update(overrides)
    return Memory(**fields)


def test_record_carries_every_required_field():
    memory = make()

    for attribute in (
        "id",
        "content",
        "created_at",
        "source_conversation",
        "memory_type",
        "confidence",
        "importance",
        "last_accessed",
        "access_count",
    ):
        assert hasattr(memory, attribute), attribute

    assert memory.id
    assert memory.created_at
    assert memory.source_conversation == "conv123"
    assert memory.memory_type is MemoryType.FACT
    assert memory.last_accessed is None  # never retrieved yet
    assert memory.access_count == 0


def test_ids_are_unique():
    assert make().id != make().id


def test_a_string_memory_type_is_coerced_to_the_enum():
    assert make(memory_type="preference").memory_type is MemoryType.PREFERENCE


@pytest.mark.parametrize("field", ["confidence", "importance"])
@pytest.mark.parametrize("value", [-0.1, 1.1, 42])
def test_scores_outside_zero_to_one_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        make(**{field: value})


def test_empty_content_is_rejected():
    with pytest.raises(ValueError):
        make(content="   ")


def test_touch_records_an_access():
    memory = make()

    memory.touch()
    assert memory.access_count == 1
    assert memory.last_accessed is not None

    first = memory.last_accessed
    memory.touch(when="2030-01-01T00:00:00+00:00")
    assert memory.access_count == 2
    assert memory.last_accessed != first


def test_round_trip_through_dict_preserves_everything():
    original = make()
    original.touch()

    restored = Memory.from_dict(original.to_dict())

    assert restored.to_dict() == original.to_dict()
    assert restored.memory_type is MemoryType.FACT
    assert restored.access_count == 1


def test_serialized_type_is_a_plain_string():
    assert make().to_dict()["memory_type"] == "fact"
