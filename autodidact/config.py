"""AgentConfig — typed YAML→Agent translation.

Replaces the two-layer mechanical translation that used to live in
``cli._agent_from_config`` and ``Agent.__init__``:

    Old: YAML dict → 11 kwargs → 80 lines of LLMConfig wiring
    New: YAML → AgentConfig (Pydantic, validated) → AgentConfig.build_agent()

Strictness: YAML loading is strict — missing required fields, ranges out
of [0, 1], API keys without explicit config or matching env var are
rejected at load time with human-readable errors. Programmatic
``Agent(local_model=..., cloud_model=..., ...)`` construction stays
loose: the kwarg shim builds an AgentConfig but allows partial state for
test fixtures and trusted programmatic callers.

The bedrock-auth wiring (``_apply_bedrock_auth``) is duplicated from
agent.py for now, since this module owns the YAML→LLMConfig translation
and agent.py owns the kwargs→LLMConfig translation. They produce
identical LLMConfig instances. A follow-up could collapse them.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal, Optional, TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

if TYPE_CHECKING:
    from autodidact.agent import Agent


# ── Provider preset registry (where to find API keys) ───────────
#
# Used to resolve api_key_env defaults when the YAML doesn't specify one.
# Mirrors the preset list in autodidact/setup_wizard.py — kept in sync via
# tests, not via runtime cross-module reading (avoid import cycle).

_PROVIDER_DEFAULT_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "groq": "GROQ_API_KEY",
    "together": "TOGETHER_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
    "xai": "XAI_API_KEY",
}

# Providers that go through the OpenAI-compatible LLMConfig backend.
_OPENAI_COMPAT_PROVIDERS = frozenset(_PROVIDER_DEFAULT_KEY_ENV.keys())


# ── Errors ──────────────────────────────────────────────────────


class ConfigError(ValueError):
    """Raised when an AgentConfig YAML file fails validation.

    Pydantic's default validation errors are noisy. ConfigError carries
    a human-readable message that names the failing fields with dotted
    paths (``cloud.provider``, ``cloud.bedrock.access_key_id``, etc.)
    so users editing YAML by hand know where to look.
    """


# ── Sub-models ──────────────────────────────────────────────────


class BedrockAuthConfig(BaseModel):
    """Bedrock authentication configuration."""

    model_config = ConfigDict(extra="ignore")

    auth_mode: Literal["default", "iam_user", "api_key"] = "default"
    region: str = "us-west-2"
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    api_key: Optional[str] = None

    @model_validator(mode="after")
    def _validate_auth_completeness(self) -> "BedrockAuthConfig":
        if self.auth_mode == "iam_user":
            missing = []
            if not self.access_key_id:
                missing.append("access_key_id")
            if not self.secret_access_key:
                missing.append("secret_access_key")
            if missing:
                raise ValueError(
                    f"bedrock.auth_mode='iam_user' requires {', '.join(missing)}"
                )
        elif self.auth_mode == "api_key":
            if not self.api_key:
                raise ValueError("bedrock.auth_mode='api_key' requires api_key")
        # 'default' uses boto3's credential chain — no fields required.
        return self


class LocalSlotConfig(BaseModel):
    """The 'local' slot: Ollama by default, OR a cheap cloud model in cloud-cloud mode.

    Distinguished from the cloud slot by its purpose (the routing target
    for the first generation attempt), not by transport.
    """

    model_config = ConfigDict(extra="ignore")

    model: str  # required — without a model, the agent has nothing to call
    embedding_model: Optional[str] = None

    # Set when the local slot is a non-Ollama cloud (cloud+cloud mode).
    # ``None`` ⇒ Ollama implicitly.
    provider: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    bedrock: Optional[BedrockAuthConfig] = None


class CloudSlotConfig(BaseModel):
    """The 'cloud' slot: the escalation target. Required for non-local-only modes."""

    model_config = ConfigDict(extra="ignore")

    provider: str  # required — no silent default
    model: str    # required
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    bedrock: Optional[BedrockAuthConfig] = None


class MemoryConfig(BaseModel):
    """SQLite + FAISS knowledge store path."""

    model_config = ConfigDict(extra="ignore")

    path: str = "~/.autodidact/memory.db"


class RoutingConfig(BaseModel):
    """Confidence threshold for memory hits."""

    model_config = ConfigDict(extra="ignore")

    confidence_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    staleness_days: float = Field(default=30.0, gt=0.0)


class GsaConfig(BaseModel):
    """Grounded self-assessment pre-gate settings."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    threshold: float = Field(default=0.55, ge=0.0, le=1.0)


# ── Top-level AgentConfig ───────────────────────────────────────


class AgentConfig(BaseModel):
    """Typed configuration for an Agent. The YAML schema."""

    model_config = ConfigDict(extra="ignore")

    local: LocalSlotConfig
    cloud: Optional[CloudSlotConfig] = None
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    gsa: GsaConfig = Field(default_factory=GsaConfig)

    # ── Loading ─────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, path: Path | str) -> "AgentConfig":
        """Load + validate a YAML config. Applies env var overrides.

        Strict: rejects missing required fields, out-of-range values, and
        OpenAI-compatible providers without explicit api_key/api_key_env
        and no matching env var.
        """
        path = Path(path)
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")

        with open(path) as f:
            raw = yaml.safe_load(f) or {}

        # Empty cloud section ⇒ local-only.
        if isinstance(raw.get("cloud"), dict) and not raw["cloud"]:
            del raw["cloud"]

        # Env-var overrides (preserved from the old _load_config behaviour).
        if os.environ.get("AUTODIDACT_MODEL"):
            raw.setdefault("local", {})["model"] = os.environ["AUTODIDACT_MODEL"]

        try:
            cfg = cls.model_validate(raw)
        except ValidationError as e:
            raise ConfigError(_format_validation_error(e)) from e

        cfg._resolve_api_keys()
        cfg._validate_api_keys_present()
        return cfg

    def _resolve_api_keys(self) -> None:
        """Fill in api_key_env from env vars when keys are otherwise missing.

        For OpenAI-compat providers, if neither api_key nor api_key_env is
        set in YAML BUT the provider's default env var is set in the
        environment, set api_key_env so the LLMClient picks it up.
        """
        for slot in (self.local, self.cloud):
            if slot is None or slot.provider is None:
                continue
            if slot.provider not in _OPENAI_COMPAT_PROVIDERS:
                continue
            if slot.api_key or slot.api_key_env:
                continue
            default_env = _PROVIDER_DEFAULT_KEY_ENV[slot.provider]
            if os.environ.get(default_env):
                slot.api_key_env = default_env

    def _validate_api_keys_present(self) -> None:
        """After resolution, fail load if an OpenAI-compat slot has no key source."""
        for slot_name, slot in (("local", self.local), ("cloud", self.cloud)):
            if slot is None or slot.provider is None:
                continue
            if slot.provider not in _OPENAI_COMPAT_PROVIDERS:
                continue
            if not (slot.api_key or slot.api_key_env):
                default_env = _PROVIDER_DEFAULT_KEY_ENV.get(slot.provider, "?")
                raise ConfigError(
                    f"{slot_name}.api_key (or {slot_name}.api_key_env) is required "
                    f"for provider '{slot.provider}'. Set api_key in YAML, set "
                    f"api_key_env to the env var name, or set the {default_env} "
                    f"environment variable."
                )

    # ── Building ────────────────────────────────────────────────

    def build_agent(self) -> "Agent":
        """Construct a fully wired Agent from this config."""
        # Lazy import to avoid a circular dependency: agent.py may import
        # AgentConfig in the future for its kwarg shim.
        from autodidact.agent import Agent

        return Agent._from_config(self)


# ── Helpers ─────────────────────────────────────────────────────


def _format_validation_error(err: ValidationError) -> str:
    """Pydantic ValidationError → human-readable message.

    Each error's ``loc`` is a tuple like ('cloud', 'provider'); we render
    it as a dotted path. Type-class errors lose Pydantic-internal noise.
    """
    lines = ["Invalid config:"]
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"])
        msg = e["msg"]
        # Strip leading "Value error, " prefix that Pydantic adds for
        # validator-raised errors — redundant in our context.
        if msg.startswith("Value error, "):
            msg = msg[len("Value error, "):]
        lines.append(f"  - {loc}: {msg}")
    return "\n".join(lines)


def apply_bedrock_auth_to_llm_kwargs(
    kwargs: dict[str, Any], auth: BedrockAuthConfig,
) -> None:
    """Translate a BedrockAuthConfig into LLMConfig kwargs.

    Mirrors agent._apply_bedrock_auth but takes a typed BedrockAuthConfig
    instead of an untyped dict. Used by build_agent().
    """
    kwargs["bedrock_auth_mode"] = auth.auth_mode
    kwargs["region"] = auth.region
    if auth.auth_mode == "iam_user":
        kwargs["bedrock_access_key_id"] = auth.access_key_id
        kwargs["bedrock_secret_access_key"] = auth.secret_access_key
        if auth.session_token:
            kwargs["bedrock_session_token"] = auth.session_token
    elif auth.auth_mode == "api_key":
        kwargs["bedrock_api_key"] = auth.api_key


__all__ = [
    "AgentConfig",
    "BedrockAuthConfig",
    "CloudSlotConfig",
    "ConfigError",
    "GsaConfig",
    "LocalSlotConfig",
    "MemoryConfig",
    "RoutingConfig",
    "apply_bedrock_auth_to_llm_kwargs",
]
