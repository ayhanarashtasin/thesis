"""Atomic, checksummed checkpoints for the standard continual RL-LoRA baseline."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from rahc_lora.config.schema import RootConfig
from rahc_lora.evaluation.performance_matrix import PerformanceMatrix
from rahc_lora.models.policy import TrainableCausalPolicy
from rahc_lora.utils.atomic import write_json_atomic, write_text_atomic
from rahc_lora.utils.manifests import canonical_config_hash, dependency_versions
from rahc_lora.utils.reproducibility import capture_rng_state, restore_rng_state

CHECKPOINT_SCHEMA_VERSION = 2


@dataclass
class TrainingProgress:
    """Persistent task and optimizer position."""

    next_task_index: int = 0
    global_step: int = 0
    rollout_counter: int = 0

    def __post_init__(self) -> None:
        if min(self.next_task_index, self.global_step, self.rollout_counter) < 0:
            raise ValueError("Training progress counters must be nonnegative")


@dataclass(frozen=True)
class CheckpointIdentity:
    """Scientific identities that must match exactly on resume."""

    config_sha256: str
    model_identifier: str
    model_revision: str
    tokenizer_identifier: str
    tokenizer_revision: str
    dataset_fingerprints: dict[str, str]
    dependencies: dict[str, str]


@dataclass(frozen=True)
class CheckpointManifest:
    """Expected checkpoint files, sizes, and SHA-256 hashes."""

    schema_version: int
    checkpoint_id: str
    next_task_index: int
    global_step: int
    files: dict[str, dict[str, int | str]]


@dataclass(frozen=True)
class LoadedCheckpoint:
    """Restored progress and performance matrix."""

    progress: TrainingProgress
    matrix: PerformanceMatrix
    capability_state: dict[str, Any]
    retention_state: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _torch_save(path: Path, value: Any) -> None:
    with path.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())


def _read_manifest(path: Path) -> CheckpointManifest:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"Unsupported checkpoint manifest: {path}")
    return CheckpointManifest(**value)


class CheckpointManager:
    """Save and restore baseline state with exact compatibility checks."""

    def __init__(
        self,
        *,
        run_dir: Path,
        resolved_config: DictConfig,
        typed_config: RootConfig,
        dataset_fingerprints: dict[str, str],
    ) -> None:
        self.run_dir = run_dir.resolve()
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.resolved_config = resolved_config
        self.typed_config = typed_config
        self.identity = CheckpointIdentity(
            config_sha256=canonical_config_hash(resolved_config),
            model_identifier=typed_config.model.identifier,
            model_revision=typed_config.model.revision,
            tokenizer_identifier=typed_config.model.tokenizer_identifier,
            tokenizer_revision=typed_config.model.tokenizer_revision,
            dataset_fingerprints=dict(dataset_fingerprints),
            dependencies=dependency_versions(),
        )

    def save(
        self,
        *,
        policy: TrainableCausalPolicy,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        progress: TrainingProgress,
        matrix: PerformanceMatrix,
        rollout_generator: torch.Generator,
        capability_state: dict[str, Any],
        retention_state: dict[str, Any] | None = None,
    ) -> tuple[Path, str]:
        """Write a new checkpoint directory and return its manifest checksum."""

        checkpoint_id = f"task-{progress.next_task_index:02d}-step-{progress.global_step:08d}"
        destination = self.checkpoint_dir / checkpoint_id
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite checkpoint: {destination}")
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{checkpoint_id}.", suffix=".tmp", dir=self.checkpoint_dir)
        )
        try:
            if retention_state is None:
                retention_state = {
                    "method": self.typed_config.method.name,
                    "reference_lora_factors": {},
                    "factorized_importance": {},
                    "dual_controller": {},
                    "anchor_memory": {
                        "enabled": False,
                        "capacity": 0,
                        "records": [],
                    },
                }
                if self.typed_config.method.name == "hlora_rl":
                    retention_state["dense_path_importance"] = None
            _torch_save(
                temporary / "adapter.pt",
                {"adapter": policy.adapter_state_dict(), "value_head": None},
            )
            _torch_save(
                temporary / "optimizer.pt",
                {
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "gradient_scaler": None,
                },
            )
            _torch_save(
                temporary / "run_state.pt",
                {
                    "progress": asdict(progress),
                    "policy_version": policy.policy_version,
                    "rng_state": capture_rng_state(),
                    "rollout_generator_state": rollout_generator.get_state(),
                },
            )
            _torch_save(temporary / "retention_state.pt", retention_state)
            write_text_atomic(
                temporary / "resolved_config.yaml", OmegaConf.to_yaml(self.resolved_config)
            )
            write_json_atomic(temporary / "identity.json", asdict(self.identity))
            write_json_atomic(temporary / "capability_state.json", capability_state)
            shutil.copyfile(self.run_dir / "run_manifest.json", temporary / "run_manifest.json")
            matrix.write_csv(temporary / "performance_matrix.csv")
            files: dict[str, dict[str, int | str]] = {}
            for path in sorted(temporary.iterdir()):
                if path.is_file():
                    files[path.name] = {"size": path.stat().st_size, "sha256": _sha256(path)}
            manifest = CheckpointManifest(
                schema_version=CHECKPOINT_SCHEMA_VERSION,
                checkpoint_id=checkpoint_id,
                next_task_index=progress.next_task_index,
                global_step=progress.global_step,
                files=files,
            )
            write_json_atomic(temporary / "manifest.json", asdict(manifest))
            self.validate(temporary)
            os.replace(temporary, destination)
            pointer = {
                "checkpoint_id": checkpoint_id,
                "relative_path": destination.relative_to(self.run_dir).as_posix(),
                "manifest_sha256": _sha256(destination / "manifest.json"),
            }
            write_json_atomic(self.run_dir / "latest_checkpoint.json", pointer)
            return destination, pointer["manifest_sha256"]
        finally:
            if temporary.exists():
                resolved = temporary.resolve()
                if resolved.parent != self.checkpoint_dir.resolve():
                    raise RuntimeError(
                        "Refusing to clean checkpoint temporary directory outside root"
                    )
                shutil.rmtree(resolved)

    @staticmethod
    def validate(path: Path) -> CheckpointManifest:
        """Validate manifest membership, file sizes, and hashes."""

        manifest = _read_manifest(path / "manifest.json")
        actual = {child.name for child in path.iterdir() if child.is_file()}
        expected = {*manifest.files, "manifest.json"}
        if actual != expected:
            raise ValueError(
                f"Checkpoint file membership mismatch; missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        for name, metadata in manifest.files.items():
            file_path = path / name
            if file_path.stat().st_size != metadata["size"]:
                raise ValueError(f"Checkpoint size mismatch for {name!r}")
            if _sha256(file_path) != metadata["sha256"]:
                raise ValueError(f"Checkpoint checksum mismatch for {name!r}")
        return manifest

    def load(
        self,
        path: Path,
        *,
        policy: TrainableCausalPolicy,
        optimizer: Optimizer,
        scheduler: LRScheduler,
        rollout_generator: torch.Generator,
    ) -> LoadedCheckpoint:
        """Validate compatibility and restore every persistent baseline state owner."""

        resolved = path.resolve()
        self.validate(resolved)
        identity_value = json.loads((resolved / "identity.json").read_text(encoding="utf-8"))
        loaded_identity = CheckpointIdentity(**identity_value)
        if loaded_identity != self.identity:
            raise ValueError("Checkpoint scientific identity does not exactly match this run")
        model_state = torch.load(resolved / "adapter.pt", map_location="cpu", weights_only=False)
        if not isinstance(model_state, dict) or set(model_state) != {"adapter", "value_head"}:
            raise ValueError("Checkpoint model state must contain adapter and value_head")
        if model_state["value_head"] is not None:
            raise ValueError("Standard RL-LoRA checkpoint unexpectedly contains a value head")
        adapter = model_state["adapter"]
        if not isinstance(adapter, dict):
            raise ValueError("Checkpoint adapter state must be a mapping")
        policy.load_adapter_state_dict(adapter)
        optimizer_state = torch.load(
            resolved / "optimizer.pt", map_location=policy.device, weights_only=False
        )
        optimizer.load_state_dict(optimizer_state["optimizer"])
        scheduler.load_state_dict(optimizer_state["scheduler"])
        run_state = torch.load(resolved / "run_state.pt", map_location="cpu", weights_only=False)
        progress = TrainingProgress(**run_state["progress"])
        policy.policy_version = int(run_state["policy_version"])
        restore_rng_state(run_state["rng_state"])
        rollout_generator.set_state(run_state["rollout_generator_state"])
        retention = torch.load(
            resolved / "retention_state.pt", map_location="cpu", weights_only=False
        )
        common_retention = {
            "method": self.typed_config.method.name,
            "reference_lora_factors": {},
            "factorized_importance": {},
            "dual_controller": {},
            "anchor_memory": {"enabled": False, "capacity": 0, "records": []},
        }
        if self.typed_config.method.name == "rl_lora":
            if retention != common_retention:
                raise ValueError("Standard RL-LoRA checkpoint contains unexpected retention state")
        elif self.typed_config.method.name == "hlora_rl":
            if not isinstance(retention, dict):
                raise ValueError("HLoRA checkpoint retention state must be a mapping")
            if set(retention) != {*common_retention, "dense_path_importance"}:
                raise ValueError("HLoRA checkpoint has invalid retention-state fields")
            if {key: retention[key] for key in common_retention} != common_retention:
                raise ValueError("HLoRA checkpoint contains unexpected non-HLoRA retention state")
            dense_state = retention["dense_path_importance"]
            if self.typed_config.method.use_parameter_consolidation:
                if not isinstance(dense_state, dict):
                    raise ValueError("HLoRA checkpoint is missing dense importance state")
                from rahc_lora.consolidation.hlora import DensePathImportance

                current_effective = policy.named_effective_lora_weights(detach=True)
                current_factors = policy.named_trainable_parameters()
                validator = DensePathImportance(
                    epsilon=self.typed_config.method.parameter_importance_epsilon
                )
                validator.load_state_dict(
                    dense_state,
                    expected_effective_shapes={
                        name: tuple(value.shape) for name, value in current_effective.items()
                    },
                    expected_factor_shapes={
                        name: tuple(value.shape) for name, value in current_factors.items()
                    },
                )
            elif dense_state is not None:
                raise ValueError("Disabled HLoRA checkpoint unexpectedly contains importance")
        else:
            raise ValueError(
                f"Checkpoint loading does not support method {self.typed_config.method.name!r}"
            )
        matrix = PerformanceMatrix.read_csv(resolved / "performance_matrix.csv")
        if progress.next_task_index != matrix.completed_rows:
            raise ValueError("Checkpoint task progress disagrees with performance matrix rows")
        capability_state = json.loads(
            (resolved / "capability_state.json").read_text(encoding="utf-8")
        )
        if not isinstance(capability_state, dict):
            raise ValueError("Checkpoint capability state must be a JSON object")
        return LoadedCheckpoint(
            progress=progress,
            matrix=matrix,
            capability_state=capability_state,
            retention_state=retention,
        )
