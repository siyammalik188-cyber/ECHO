"""Agent: ties the conversation, the LLM, and the disk together.

There is no loop, no planning, no tools. `send()` is one round trip.
"""

from __future__ import annotations

from pathlib import Path

from . import storage
from .conversation import Conversation
from .llm import LLMClient


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        conversation: Conversation | None = None,
        directory: Path | str = storage.DEFAULT_DIR,
    ) -> None:
        self.llm = llm
        # `is None`, not `or` — an empty Conversation is falsy (it defines
        # __len__), and `or` would silently throw away a caller's system prompt.
        self.conversation = Conversation() if conversation is None else conversation
        self.directory = Path(directory)

    @property
    def id(self) -> str:
        return self.conversation.id

    def send(self, text: str) -> str:
        """Append `text` as a user turn, ask the model, append the reply, return it.

        If the model call fails the user turn is rolled back, so the conversation
        never keeps a question that was never answered.
        """
        self.conversation.add_user(text)
        try:
            reply = self.llm.complete(
                self.conversation.to_api_messages(),
                system=self.conversation.system,
            )
        except Exception:
            self.conversation.messages.pop()
            raise
        self.conversation.add_assistant(reply)
        return reply

    def save(self) -> Path:
        return storage.save(self.conversation, self.directory)

    @classmethod
    def resume(
        cls,
        conversation_id: str,
        llm: LLMClient,
        directory: Path | str = storage.DEFAULT_DIR,
    ) -> "Agent":
        """Load a saved conversation from disk and continue it."""
        conversation = storage.load(conversation_id, directory)
        return cls(llm=llm, conversation=conversation, directory=directory)
