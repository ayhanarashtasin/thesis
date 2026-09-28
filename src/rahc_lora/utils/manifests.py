"""Reproducible, atomic run-manifest creation and validation."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from rahc_lora import __version__
from rahc_lora.config import config_to_dict

MANIFEST_SCHEMA_VERSION = 2
CORE_DISTRIBUTIONS = (
    "absl-py",
    "accelerate",
    "datasets",
    "hydra-core",
    "immutabledict",
    "langdetect",
    "nltk",
    "numpy",
    "pandas",
    "peft",
    "psutil",
    "pyarrow",
    "safetensors",
    "scikit-learn",
    "scipy",
    "torch",
    "transformers",
    "trl",
)


def dependency_versions(names: tuple[str, ...] = CORE_DISTRIBUTIONS) -> dict[str, str]:
    """Return installed distribution versions without importing heavyweight libraries."""

    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def canonical_config_hash(config: DictConfig) -> str:
    """Hash the fully resolved configuration using canonical JSON encoding."""

    payload = json.dumps(config_to_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git_state(repository: Path) -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, None
    return commit, bool(status.strip())


@dataclass(frozen=True)
class RunManifest:
    """Run identity, immutable source selections, and observable environment."""

    schema_version: int
    created_at_utc: str
    project_version: str
    python_version: str
    platform: str
    config_sha256: str
    git_commit: str | None
    git_dirty: bool | None
    dependencies: dict[str, str]
    model_identifier: str
    model_revision: str
    tokenizer_identifier: str
    tokenizer_revision: str
    dataset_fingerprints: dict[str, str]
    task_order: list[str]
    seed: int
    command_overrides: list[str]
    hardware: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RunManifest:
        """Validate and construct a manifest loaded from JSON."""

        expected = {field.name for field in cls.__dataclass_fields__.values()}
        actual = set(value)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(f"Invalid run manifest keys; missing={missing}, extra={extra}")
        if value["schema_version"] != MANIFEST_SCHEMA_VERSION:
            raise ValueError(
                "Unsupported run manifest schema version: "
                f"{value['schema_version']!r}; expected {MANIFEST_SCHEMA_VERSION}"
            )
        dependencies = value["dependencies"]
        if not isinstance(dependencies, dict) or not all(
            isinstance(key, str) and isinstance(version, str)
            for key, version in dependencies.items()
        ):
            raise ValueError("Run manifest dependencies must map strings to strings")
        return cls(**value)


def build_run_manifest(
    config: DictConfig,
    *,
    repository: Path | None = None,
    dataset_fingerprints: dict[str, str] | None = None,
    command_overrides: list[str] | None = None,
) -> RunManifest:
    """Build a run manifest from resolved configuration and observable local state."""

    repo = (repository or Path.cwd()).resolve()
    commit, dirty = _git_state(repo)
    import torch

    cuda_available = bool(torch.cuda.is_available())
    hardware: dict[str, Any] = {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "bf16_supported": bool(torch.cuda.is_bf16_supported()) if cuda_available else False,
        "gpu_names": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if cuda_available
            else []
        ),
    }
    return RunManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        created_at_utc=datetime.now(UTC).isoformat(),
        project_version=__version__,
        python_version=platform.python_version(),
        platform=platform.platform(),
        config_sha256=canonical_config_hash(config),
        git_commit=commit,
        git_dirty=dirty,
        dependencies=dependency_versions(),
        model_identifier=str(config.model.identifier),
        model_revision=str(config.model.revision),
        tokenizer_identifier=str(config.model.tokenizer_identifier),
        tokenizer_revision=str(config.model.tokenizer_revision),
        dataset_fingerprints=dict(dataset_fingerprints or {}),
        task_order=[str(task_id) for task_id in config.task_stream.tasks],
        seed=int(config.seed),
        command_overrides=list(command_overrides or []),
        hardware=hardware,
    )


def write_run_manifest(path: Path, manifest: RunManifest) -> None:
    """Atomically write a manifest as deterministic, human-readable JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            temporary = Path(temporary_name)
            if temporary.exists():
                temporary.unlink()


def read_run_manifest(path: Path) -> RunManifest:
    """Load and validate a run manifest."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Run manifest root must be a JSON object")
    return RunManifest.from_dict(value)
