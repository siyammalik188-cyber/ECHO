"""ECHO — a minimal conversational agent foundation."""

from .agent import Agent
from .context import WorkingContext
from .conversation import Conversation, Message
from .extraction import (
    LLMMemoryExtractor,
    MemoryCandidate,
    MemoryExtractor,
    consolidate,
)
from .llm import AnthropicLLM, LLMClient, LLMError, StructuredLLMClient
from .memory import Memory, MemoryType
from .memory_store import MemoryStore

__version__ = "0.2.0"

__all__ = [
    # conversation history
    "Conversation",
    "Message",
    # persistent memory
    "Memory",
    "MemoryType",
    "MemoryStore",
    # the deliberate step between them
    "MemoryCandidate",
    "MemoryExtractor",
    "LLMMemoryExtractor",
    "consolidate",
    # temporary context
    "WorkingContext",
    # plumbing
    "Agent",
    "AnthropicLLM",
    "LLMClient",
    "StructuredLLMClient",
    "LLMError",
]
