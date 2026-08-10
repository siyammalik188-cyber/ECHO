"""Persistence: one conversation per JSON file on the local filesystem."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .conversation import Conversation

DEFAULT_DIR = Path("conversations")

# Conversation ids become filenames, so keep them boring.
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def _check_id(conversation_id: str) -> str:
    if not _SAFE_ID.match(conversation_id):
        raise ValueError(
            f"conversation id {conversation_id!r} must match [A-Za-z0-9_-]+"
        )
    return conversation_id


def path_for(conversation_id: str, directory: Path | str = DEFAULT_DIR) -> Path:
    return Path(directory) / f"{_check_id(conversation_id)}.json"


def save(conversation: Conversation, directory: Path | str = DEFAULT_DIR) -> Path:
    """Write the conversation to `<directory>/<id>.json` and return the path.

    The write goes to a temp file first and is then renamed, so an interrupted
    save cannot leave a half-written conversation behind.
    """
    target = path_for(conversation.id, directory)
    target.parent.mkdir(parents=True, exist_ok=True)

    tmp = target.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(conversation.to_dict(), handle, indent=2, ensure_ascii=False)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(target)
    return target


def load(conversation_id: str, directory: Path | str = DEFAULT_DIR) -> Conversation:
    """Read a conversation back from disk. Raises FileNotFoundError if absent."""
    target = path_for(conversation_id, directory)
    with target.open("r", encoding="utf-8") as handle:
        return Conversation.from_dict(json.load(handle))


def list_ids(directory: Path | str = DEFAULT_DIR) -> list[str]:
    """Every saved conversation id in `directory`, sorted. Empty if no directory."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p.stem for p in directory.glob("*.json"))


def exists(conversation_id: str, directory: Path | str = DEFAULT_DIR) -> bool:
    return path_for(conversation_id, directory).is_file()
