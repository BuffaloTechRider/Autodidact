"""Per-provider chat backend adapters.

Each adapter implements the ChatBackend interface (see backend.py) for one
provider. LLMClient holds one adapter and forwards public method calls to
it. This eliminates the per-provider `if self.config.provider == "x"`
branching that used to live inside every public method.

Public surface preserved on LLMClient:
  - chat / chat_with_logprobs / chat_stream / embed (provider-agnostic)
  - chat_stream_ollama / chat_stream_ollama_no_logprobs
  - chat_stream_openai / chat_stream_bedrock
  - _get_openai_client / _get_bedrock_client (test patch points)

These facades route to the active backend; calling
chat_stream_bedrock on an Ollama client raises LLMClientError.
"""

from autodidact.llm.backend import (
    ChatBackend,
    _BedrockThrottleError,
    _consume_ollama_stream,
    _consume_ollama_stream_plain,
    _extract_answer,
    _with_retries,
)
from autodidact.llm.bedrock import BedrockBackend
from autodidact.llm.ollama import OllamaBackend
from autodidact.llm.openai import OpenAICompatBackend


__all__ = [
    "BedrockBackend",
    "ChatBackend",
    "OllamaBackend",
    "OpenAICompatBackend",
    "_BedrockThrottleError",
    "_consume_ollama_stream",
    "_consume_ollama_stream_plain",
    "_extract_answer",
    "_with_retries",
]
