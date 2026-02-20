"""
ZENO LLM Runtime - Phase 2

Local LLM abstraction layer providing offline inference via Ollama.
"""

from .local_llm import (
    LocalLLM,
    LLMResponse,
    OllamaError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
    OllamaInvalidResponseError,
    QWEN_3B_INSTRUCT,
    LLAMA_1B_INSTRUCT,
    LLAMA3_3B,
    MISTRAL_7B,
    QWEN_CODER
)

__all__ = [
    'LocalLLM',
    'LLMResponse',
    'OllamaError',
    'OllamaConnectionError',
    'OllamaModelNotFoundError',
    'OllamaTimeoutError',
    'OllamaInvalidResponseError',
    'QWEN_3B_INSTRUCT',
    'LLAMA_1B_INSTRUCT',
    'LLAMA3_3B',
    'MISTRAL_7B',
    'QWEN_CODER',
]