"""Cloud provider presets — base URL, env var name, model lists, defaults.

Used by the wizard's prompts and by the AgentConfig translation in
autodidact/config.py. The wizard's interactive flow consults these to
populate dropdowns and pre-select sensible defaults; the agent uses
them only as a fallback when YAML doesn't specify ``base_url`` or
``api_key_env``.
"""

from __future__ import annotations


_CLOUD_PRESETS: dict[str, dict] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "models": [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4.1",
            "gpt-4.1-mini",
            "gpt-4.1-nano",
            "o3-mini",
            "o1",
        ],
        "default_cheap": "gpt-4o-mini",
        "default_expensive": "gpt-4o",
        "embedding_model": "text-embedding-3-small",
    },
    "anthropic": {
        # Anthropic's OpenAI-compat shim — works with our openai-provider client.
        # Direct API has quirks; OpenRouter route is also supported for Claude.
        "base_url": "https://api.anthropic.com/v1",
        "api_key_env": "ANTHROPIC_API_KEY",
        "models": [
            "claude-sonnet-4-5",
            "claude-opus-4",
            "claude-haiku-4",
            "claude-3-5-sonnet-20241022",
            "claude-3-5-haiku-20241022",
            "claude-3-opus-20240229",
        ],
        "default_cheap": "claude-haiku-4",
        "default_expensive": "claude-sonnet-4-5",
        "embedding_model": None,
    },
    "google": {
        # Google AI Studio — OpenAI-compatible endpoint. Free tier available
        # (no credit card): 500 req/day for Flash, lower for Pro.
        # Get a key at https://aistudio.google.com/apikey
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GOOGLE_API_KEY",
        "models": [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
            "gemini-2.0-flash",
        ],
        "default_cheap": "gemini-2.5-flash",
        "default_expensive": "gemini-2.5-flash",
        "embedding_model": None,
    },
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "models": [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "anthropic/claude-sonnet-4-5",
            "anthropic/claude-haiku-4",
            "google/gemini-2.5-pro",
            "google/gemini-2.5-flash",
            "meta-llama/llama-3.3-70b-instruct",
            "deepseek/deepseek-chat",
        ],
        "default_cheap": "google/gemini-2.5-flash",
        "default_expensive": "anthropic/claude-sonnet-4-5",
        "embedding_model": "openai/text-embedding-3-small",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "models": [
            "deepseek-chat",
            "deepseek-reasoner",
            "deepseek-coder",
        ],
        "default_cheap": "deepseek-chat",
        "default_expensive": "deepseek-reasoner",
        "embedding_model": None,
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "api_key_env": "MISTRAL_API_KEY",
        "models": [
            "mistral-large-latest",
            "mistral-medium-latest",
            "mistral-small-latest",
            "pixtral-large-latest",
            "codestral-latest",
        ],
        "default_cheap": "mistral-small-latest",
        "default_expensive": "mistral-large-latest",
        "embedding_model": "mistral-embed",
    },
    "groq": {
        # Fastest OpenAI-compat inference; great for the cheap slot in cloud+cloud.
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "default_cheap": "llama-3.1-8b-instant",
        "default_expensive": "llama-3.3-70b-versatile",
        "embedding_model": None,
    },
    "together": {
        "base_url": "https://api.together.xyz/v1",
        "api_key_env": "TOGETHER_API_KEY",
        "models": [
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "meta-llama/Llama-3.1-8B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "deepseek-ai/DeepSeek-V3",
            "mistralai/Mixtral-8x7B-Instruct-v0.1",
        ],
        "default_cheap": "meta-llama/Llama-3.1-8B-Instruct-Turbo",
        "default_expensive": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "embedding_model": "togethercomputer/m2-bert-80M-8k-retrieval",
    },
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "api_key_env": "FIREWORKS_API_KEY",
        "models": [
            "accounts/fireworks/models/llama-v3p3-70b-instruct",
            "accounts/fireworks/models/llama-v3p1-8b-instruct",
            "accounts/fireworks/models/qwen2p5-72b-instruct",
            "accounts/fireworks/models/deepseek-v3",
            "accounts/fireworks/models/mixtral-8x22b-instruct",
        ],
        "default_cheap": "accounts/fireworks/models/llama-v3p1-8b-instruct",
        "default_expensive": "accounts/fireworks/models/llama-v3p3-70b-instruct",
        "embedding_model": None,
    },
    "xai": {
        # xAI Grok — OpenAI-compat.
        "base_url": "https://api.x.ai/v1",
        "api_key_env": "XAI_API_KEY",
        "models": [
            "grok-4",
            "grok-3",
            "grok-3-mini",
            "grok-2-latest",
        ],
        "default_cheap": "grok-3-mini",
        "default_expensive": "grok-4",
        "embedding_model": None,
    },
    "bedrock": {
        "base_url": "",
        "api_key_env": "",
        # The Bedrock model list is *discovered at wizard time* via
        # `discover_bedrock_models()` because (a) Bedrock evolves rapidly,
        # (b) availability differs per region, and (c) some models are
        # inference-profile-only with region-specific prefixes. The static
        # entries below are only a last-resort hint shown if discovery fails
        # AND the user wants to pick from a list rather than type freely.
        "models": [],
        "default_cheap": "",
        "default_expensive": "",
        "embedding_model": None,
    },
}


def get_cloud_preset(provider: str) -> dict:
    """Get preset config for a cloud provider."""
    if provider in _CLOUD_PRESETS:
        return _CLOUD_PRESETS[provider]
    return {
        "base_url": "",
        "api_key_env": "",
        "models": [],
        "default_cheap": "",
        "default_expensive": "",
        "embedding_model": None,
    }


def list_cloud_providers() -> list[str]:
    """List available cloud provider presets, ordered for the wizard UI."""
    priority = ["google", "bedrock"]
    rest = [k for k in _CLOUD_PRESETS if k not in priority]
    return priority + rest


__all__ = [
    "_CLOUD_PRESETS",
    "get_cloud_preset",
    "list_cloud_providers",
]
