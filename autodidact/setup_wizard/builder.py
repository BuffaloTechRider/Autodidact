"""build_config — assemble the YAML-shaped config dict from wizard answers.

The only input/output type discipline here is dict-shaped (raw YAML).
The strict typed AgentConfig in autodidact/config.py validates this dict
on the *load* path; build_config is the *write* path used by the wizard.
"""

from __future__ import annotations

from typing import Optional

from autodidact.setup_wizard.presets import get_cloud_preset


def build_config(
    mode: str = "local_cloud",
    *,
    # local_cloud and local_only
    local_model: Optional[str] = None,
    embedding_model: Optional[str] = None,
    # local_cloud
    cloud_provider: Optional[str] = None,
    cloud_model: Optional[str] = None,
    cloud_api_key: Optional[str] = None,
    cloud_base_url: Optional[str] = None,
    cloud_bedrock: Optional[dict] = None,
    # cloud_cloud
    cheap_cloud_provider: Optional[str] = None,
    cheap_cloud_model: Optional[str] = None,
    cheap_cloud_api_key: Optional[str] = None,
    cheap_cloud_base_url: Optional[str] = None,
    cheap_cloud_bedrock: Optional[dict] = None,
    expensive_cloud_provider: Optional[str] = None,
    expensive_cloud_model: Optional[str] = None,
    expensive_cloud_api_key: Optional[str] = None,
    expensive_cloud_base_url: Optional[str] = None,
    expensive_cloud_bedrock: Optional[dict] = None,
    # common
    db_path: str = "~/.autodidact/memory.db",
    confidence_threshold: float = 0.7,
) -> dict:
    """Build a config dict for the given setup mode.

    Bedrock-specific auth settings (auth_mode, access_key_id, api_key, region, ...)
    are passed via the *_cloud_bedrock dicts and stored under the 'bedrock' key
    on the cloud/local section.
    """
    config: dict = {
        "routing": {"confidence_threshold": confidence_threshold},
        "memory": {"path": db_path},
    }

    if mode == "local_cloud":
        config["local"] = {
            "model": local_model or "qwen3:4b",
            "embedding_model": embedding_model or "qllama/bge-large-en-v1.5",
        }
        if cloud_provider and cloud_model:
            cloud_cfg: dict = {
                "provider": cloud_provider,
                "model": cloud_model,
            }
            if cloud_api_key:
                cloud_cfg["api_key"] = cloud_api_key
            if cloud_base_url:
                cloud_cfg["base_url"] = cloud_base_url
            if cloud_bedrock:
                cloud_cfg["bedrock"] = cloud_bedrock
            config["cloud"] = cloud_cfg

    elif mode == "cloud_cloud":
        # "Local" slot is the cheap cloud model.
        cheap_preset = get_cloud_preset(cheap_cloud_provider or "openai")
        expensive_preset = get_cloud_preset(expensive_cloud_provider or "openai")

        config["local"] = {
            "provider": cheap_cloud_provider or "openai",
            "model": cheap_cloud_model or cheap_preset.get("default_cheap", ""),
            "base_url": cheap_cloud_base_url or cheap_preset.get("base_url", ""),
            "embedding_model": cheap_preset.get("embedding_model") or "text-embedding-3-small",
        }
        if cheap_cloud_api_key:
            config["local"]["api_key"] = cheap_cloud_api_key
        if cheap_cloud_bedrock:
            config["local"]["bedrock"] = cheap_cloud_bedrock

        config["cloud"] = {
            "provider": expensive_cloud_provider or "openai",
            "model": expensive_cloud_model or expensive_preset.get("default_expensive", ""),
        }
        if expensive_cloud_base_url:
            config["cloud"]["base_url"] = expensive_cloud_base_url
        if expensive_cloud_api_key:
            config["cloud"]["api_key"] = expensive_cloud_api_key
        if expensive_cloud_bedrock:
            config["cloud"]["bedrock"] = expensive_cloud_bedrock

    elif mode == "local_local":
        config["local"] = {
            "model": local_model or "qwen3:4b",
            "embedding_model": embedding_model or "qllama/bge-large-en-v1.5",
        }
        if cloud_provider == "ollama" and cloud_model:
            config["cloud"] = {
                "provider": "ollama",
                "model": cloud_model,
            }

    elif mode == "local_only":
        config["local"] = {
            "model": local_model or "qwen3:4b",
            "embedding_model": embedding_model or "qllama/bge-large-en-v1.5",
        }

    return config


__all__ = ["build_config"]
