"""Hydra composition into a validated, immutable structured configuration."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from rahc_lora.config.schema import RootConfig
from rahc_lora.config.validation import validate_config


def default_config_dir() -> Path:
    """Return the repository configuration directory."""

    return Path(__file__).resolve().parents[3] / "configs"


def compose_config(
    *,
    config_name: str = "config",
    overrides: Sequence[str] = (),
    config_dir: Path | None = None,
) -> DictConfig:
    """Compose, type-check, validate, resolve, and freeze one run configuration."""

    selected_dir = (config_dir or default_config_dir()).resolve()
    if not selected_dir.is_dir():
        raise FileNotFoundError(f"Hydra configuration directory does not exist: {selected_dir}")

    with initialize_config_dir(version_base="1.3", config_dir=str(selected_dir)):
        raw_config = compose(config_name=config_name, overrides=list(overrides))

    structured = OmegaConf.structured(RootConfig)
    merged = cast(DictConfig, OmegaConf.merge(structured, raw_config))
    OmegaConf.resolve(merged)
    validate_config(merged)
    OmegaConf.set_readonly(merged, True)
    return merged


def config_to_dict(config: DictConfig) -> dict[str, Any]:
    """Convert a resolved configuration to a plain serializable mapping."""

    value = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    if not isinstance(value, dict):
        raise TypeError("Resolved root configuration must be a mapping")
    return cast(dict[str, Any], value)
