"""Versioned dataset and split manifests with explicit membership fingerprints."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rahc_lora.config.schema import TaskDataConfig
from rahc_lora.tasks.base import DatasetRole, TaskExample, TaskSplits

DATASET_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ExampleFingerprint:
    """Non-content dataset membership record."""

    example_id: str
    content_sha256: str


@dataclass(frozen=True)
class DatasetManifest:
    """Pinned source and exact project split membership."""

    schema_version: int
    task_id: str
    canonical_dataset_name: str
    dataset_subset: str | None
    revision: str
    license: str
    citation: str
    source_splits: dict[str, str | None]
    split_seed: int
    preprocessing_version: str
    prompt_version: str
    reward_version: str
    memberships: dict[str, tuple[ExampleFingerprint, ...]]
    example_metadata: dict[str, dict[str, Any]]
    generation_metadata: dict[str, Any]

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Unsupported dataset manifest schema version")
        if not self.task_id or not self.canonical_dataset_name or not self.revision:
            raise ValueError("Dataset manifest source identity must not be empty")
        role_names = set(self.memberships)
        trainable_roles = {
            DatasetRole.TRAIN.value,
            DatasetRole.VALIDATION.value,
            DatasetRole.ANCHOR_CANDIDATE.value,
            DatasetRole.TEST.value,
        }
        evaluation_roles = {DatasetRole.EVALUATION_ONLY.value}
        if role_names not in (trainable_roles, evaluation_roles):
            raise ValueError(f"Dataset manifest has invalid role topology: {sorted(role_names)}")
        seen_ids: set[str] = set()
        seen_hashes_by_role: dict[str, str] = {}
        for role, fingerprints in self.memberships.items():
            if not fingerprints:
                raise ValueError(f"Dataset manifest role {role!r} must not be empty")
            for fingerprint in fingerprints:
                if fingerprint.example_id in seen_ids:
                    raise ValueError(
                        f"Dataset manifest example ID is duplicated: {fingerprint.example_id!r}"
                    )
                if len(fingerprint.content_sha256) != 64:
                    raise ValueError(
                        f"Dataset manifest content hash is invalid: {fingerprint.content_sha256!r}"
                    )
                if role_names == trainable_roles:
                    previous_role = seen_hashes_by_role.get(fingerprint.content_sha256)
                    if previous_role is not None:
                        raise ValueError(
                            "Dataset manifest content overlaps roles "
                            f"{previous_role!r} and {role!r}"
                        )
                    seen_hashes_by_role[fingerprint.content_sha256] = role
                seen_ids.add(fingerprint.example_id)
        if set(self.example_metadata) != seen_ids:
            missing = sorted(seen_ids.difference(self.example_metadata))
            extra = sorted(set(self.example_metadata).difference(seen_ids))
            raise ValueError(
                f"Dataset manifest metadata membership mismatch; missing={missing}, extra={extra}"
            )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DatasetManifest:
        if value.get("schema_version") != DATASET_MANIFEST_SCHEMA_VERSION:
            raise ValueError("Unsupported dataset manifest schema version")
        memberships_value = value.get("memberships")
        if not isinstance(memberships_value, dict):
            raise ValueError("Dataset manifest memberships must be an object")
        memberships = {
            role: tuple(ExampleFingerprint(**fingerprint) for fingerprint in fingerprints)
            for role, fingerprints in memberships_value.items()
        }
        copy = dict(value)
        copy["memberships"] = memberships
        return cls(**copy)


def _fingerprints(examples: tuple[TaskExample, ...]) -> tuple[ExampleFingerprint, ...]:
    return tuple(
        sorted(
            (
                ExampleFingerprint(
                    example_id=example.example_id,
                    content_sha256=example.content_sha256,
                )
                for example in examples
            ),
            key=lambda item: item.example_id,
        )
    )


def _example_metadata(*partitions: tuple[TaskExample, ...]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for examples in partitions:
        for example in examples:
            if example.example_id in metadata:
                raise ValueError(f"Duplicate example metadata ID: {example.example_id!r}")
            try:
                json.dumps(example.metadata, sort_keys=True)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Example metadata for {example.example_id!r} is not JSON serializable"
                ) from error
            metadata[example.example_id] = dict(example.metadata)
    return metadata


def build_dataset_manifest(
    config: TaskDataConfig,
    splits: TaskSplits,
    *,
    generation_metadata: dict[str, Any] | None = None,
) -> DatasetManifest:
    """Build a manifest only after TaskSplits has proven role disjointness."""

    source_splits = {
        "train": config.source_train_split,
        "validation": config.source_validation_split,
        "test": config.source_test_split,
    }
    return DatasetManifest(
        schema_version=DATASET_MANIFEST_SCHEMA_VERSION,
        task_id=config.task_id,
        canonical_dataset_name=config.dataset_name,
        dataset_subset=config.dataset_subset,
        revision=config.revision,
        license=config.license,
        citation=config.citation,
        source_splits=source_splits,
        split_seed=config.split_seed,
        preprocessing_version=config.preprocessing_version,
        prompt_version=config.prompt_version,
        reward_version=config.reward_version,
        memberships={
            DatasetRole.TRAIN.value: _fingerprints(splits.train),
            DatasetRole.VALIDATION.value: _fingerprints(splits.validation),
            DatasetRole.ANCHOR_CANDIDATE.value: _fingerprints(splits.anchor_candidates),
            DatasetRole.TEST.value: _fingerprints(splits.test),
        },
        example_metadata=_example_metadata(
            splits.train,
            splits.validation,
            splits.anchor_candidates,
            splits.test,
        ),
        generation_metadata=generation_metadata or {},
    )


def build_evaluation_dataset_manifest(
    config: TaskDataConfig,
    examples: tuple[TaskExample, ...],
    *,
    generation_metadata: dict[str, Any] | None = None,
) -> DatasetManifest:
    """Build a manifest for a dataset that can only have evaluation membership."""

    if not config.evaluation_only:
        raise ValueError(f"Dataset {config.task_id!r} is not evaluation-only")
    if any(example.role is not DatasetRole.EVALUATION_ONLY for example in examples):
        raise ValueError("Evaluation manifest received a non-evaluation example")
    fingerprints = _fingerprints(examples)
    if len({item.example_id for item in fingerprints}) != len(fingerprints):
        raise ValueError("Evaluation manifest contains duplicate example IDs")
    return DatasetManifest(
        schema_version=DATASET_MANIFEST_SCHEMA_VERSION,
        task_id=config.task_id,
        canonical_dataset_name=config.dataset_name,
        dataset_subset=config.dataset_subset,
        revision=config.revision,
        license=config.license,
        citation=config.citation,
        source_splits={"train": None, "validation": None, "test": config.source_test_split},
        split_seed=config.split_seed,
        preprocessing_version=config.preprocessing_version,
        prompt_version=config.prompt_version,
        reward_version=config.reward_version,
        memberships={DatasetRole.EVALUATION_ONLY.value: fingerprints},
        example_metadata=_example_metadata(examples),
        generation_metadata=generation_metadata or {},
    )


def write_dataset_manifest(path: Path, manifest: DatasetManifest) -> None:
    """Atomically write a deterministic dataset manifest."""

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


def read_dataset_manifest(path: Path) -> DatasetManifest:
    """Read and validate a dataset manifest."""

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Dataset manifest root must be a JSON object")
    return DatasetManifest.from_dict(value)
