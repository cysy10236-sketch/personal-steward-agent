"""LLM provider interfaces: live OpenAI-compatible client and deterministic offline mock."""

from .provider import (
    LLMCompletion,
    LLMProviderProtocol,
    MockLLMProvider,
    OpenAICompatibleProvider,
    ToolCallRequest,
    build_llm_provider,
)

__all__ = [
    "LLMCompletion",
    "LLMProviderProtocol",
    "MockLLMProvider",
    "OpenAICompatibleProvider",
    "ToolCallRequest",
    "build_llm_provider",
]
