"""Agent: ties conversation history, persistent memory, temporary context,
the LLM, and the disk together.

There is no loop, no planning, no tools, and no automatic memory formation.
`send()` is one round trip; memories are only ever created by `remember()`,
which you call on purpose.
"""

from __future__ import annotations

from pathlib import Path

from . import storage
from .context import WorkingContext
from .conversation import Conversation
from .extraction import MemoryExtractor, consolidate
from .llm import LLMClient
from .memory import Memory
from .memory_store import MemoryStore


class Agent:
    def __init__(
        self,
        llm: LLMClient,
        conversation: Conversation | None = None,
        directory: Path | str = storage.DEFAULT_ROOT,
        memories: MemoryStore | None = None,
        extractor: MemoryExtractor | None = None,
    ) -> None:
        self.llm = llm
        # `is None`, not `or` — an empty Conversation is falsy (it defines
        # __len__), and `or` would silently throw away a caller's system prompt.
        self.conversation = Conversation() if conversation is None else conversation

        # `directory` is the data root; transcripts live in a subdirectory of it
        # so that listing conversations never sees the memory store.
        self.directory = Path(directory)
        self.conversations_dir = storage.conversations_dir(self.directory)

        # The three kinds of state, side by side:
        #   self.conversation — this conversation's transcript   (durable, per-conversation)
        #   self.memories     — extracted knowledge              (durable, cross-conversation)
        #   self.context      — scratch for the current turn      (never persisted)
        self.memories = (
            MemoryStore.in_directory(self.directory) if memories is None else memories
        )
        self.extractor = extractor
        self.context = WorkingContext()

    @property
    def id(self) -> str:
        return self.conversation.id

    # ------------------------------------------------------------ conversing

    def send(self, text: str) -> str:
        """Append `text` as a user turn, ask the model, append the reply, return it.

        Working context is folded into the system prompt for this one call and
        goes no further: it is a request parameter, never a message, so it never
        becomes conversation history. This call creates no memories — see
        `remember()`.

        If the model call fails the user turn is rolled back, so the conversation
        never keeps a question that was never answered.
        """
        self.conversation.add_user(text)
        try:
            reply = self.llm.complete(
                self.conversation.to_api_messages(),
                system=self._system_for_this_turn(),
            )
        except Exception:
            self.conversation.messages.pop()
            raise
        self.conversation.add_assistant(reply)
        return reply

    def _system_for_this_turn(self) -> str | None:
        base = self.conversation.system
        extra = self.context.render()
        if not extra:
            return base
        return f"{base}\n\n{extra}" if base else extra

    # -------------------------------------------------------------- memories

    def recall(self, query: str, limit: int = 5, load: bool = True) -> list[Memory]:
        """Search persistent memory and, by default, load hits into working context.

        Retrieval counts as an access, so `last_accessed` and `access_count`
        change here. Pass `load=False` to look without staging for the next turn.
        """
        found = self.memories.search(query, limit=limit)
        if load:
            self.context.add_memories(found)
        return found

    def remember(self) -> list[Memory]:
        """Run memory extraction over this conversation. The only path to a memory.

        Returns the memories that were actually created — an empty list is a
        normal, common outcome. Requires an extractor.
        """
        if self.extractor is None:
            raise RuntimeError(
                "Agent has no extractor; pass one to create memories, e.g. "
                "Agent(..., extractor=LLMMemoryExtractor(llm))"
            )
        created = consolidate(self.conversation, self.extractor, self.memories)
        if created:
            self.memories.save()
        return created

    # -------------------------------------------------------------- the disk

    def save(self) -> Path:
        """Persist the conversation transcript and the memory store.

        Working context is intentionally not written anywhere.
        """
        self.memories.save()
        return storage.save(self.conversation, self.conversations_dir)

    @classmethod
    def resume(
        cls,
        conversation_id: str,
        llm: LLMClient,
        directory: Path | str = storage.DEFAULT_ROOT,
        extractor: MemoryExtractor | None = None,
    ) -> "Agent":
        """Load a saved conversation from disk and continue it."""
        conversation = storage.load(
            conversation_id, storage.conversations_dir(directory)
        )
        return cls(
            llm=llm,
            conversation=conversation,
            directory=directory,
            extractor=extractor,
        )
