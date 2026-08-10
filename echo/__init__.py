"""ECHO — a minimal conversational agent foundation."""

from .agent import Agent
from .conversation import Conversation, Message
from .llm import AnthropicLLM, LLMClient, LLMError

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Conversation",
    "Message",
    "AnthropicLLM",
    "LLMClient",
    "LLMError",
]
