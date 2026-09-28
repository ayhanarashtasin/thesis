"""Mockable dataset access and deterministic role partitioning."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

from rahc_lora.config.schema import TaskDataConfig
from rahc_lora.tasks.base import DatasetRole, TaskExample, TaskSplits, canonical_content_hash


class DatasetProvider(Protocol):
    """Lazy dataset source boundary; implementations may perform I/O only when called."""

    def load(self, config: TaskDataConfig, split: str) -> Sequence[Mapping[str, Any]]:
        """Load one pinned original source split."""


@dataclass
class InMemoryDatasetProvider:
    """Deterministic provider used by unit and integration tests."""

    datasets: dict[tuple[str, str | None, str], tuple[dict[str, Any], ...]]

    def load(self, config: TaskDataConfig, split: str) -> Sequence[Mapping[str, Any]]:
        key = (config.dataset_name, config.dataset_subset, split)
        if key not in self.datasets:
            raise KeyError(f"In-memory dataset split is unavailable: {key!r}")
        return self.datasets[key]


@dataclass(frozen=True)
class HuggingFaceDatasetProvider:
    """Lazy Hugging Face provider pinned by task configuration revision."""

    cache_dir: str | None = None

    def load(self, config: TaskDataConfig, split: str) -> Sequence[Mapping[str, Any]]:
        if config.provider != "huggingface":
            raise ValueError(f"HuggingFaceDatasetProvider cannot load provider {config.provider!r}")
        datasets = importlib.import_module("datasets")
        dataset = datasets.load_dataset(
            config.dataset_name,
            config.dataset_subset,
            revision=config.revision,
            split=split,
            cache_dir=self.cache_dir,
            trust_remote_code=False,
        )
        return cast(Sequence[Mapping[str, Any]], dataset)


def stable_source_id(config: TaskDataConfig, record: Mapping[str, Any]) -> str:
    """Create a stable non-plaintext project ID from configured source identifier fields."""

    missing = [field for field in config.id_fields if field not in record]
    if missing:
        raise ValueError(
            f"Dataset {config.task_id!r} record lacks configured ID fields: {', '.join(missing)}"
        )
    source_identity = [record[field] for field in config.id_fields]
    payload = json.dumps(source_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{config.task_id}:{digest}"


def _rank_key(seed: int, example_id: str) -> str:
    return hashlib.sha256(f"{seed}:{example_id}".encode()).hexdigest()


def _partition_count(total: int, fraction: float, reserved: int) -> int:
    if fraction == 0:
        return 0
    if total <= reserved:
        raise ValueError("Dataset is too small for all requested disjoint partitions")
    return min(total - reserved, max(1, round(total * fraction)))


def _as_example(
    config: TaskDataConfig,
    record: Mapping[str, Any],
    role: DatasetRole,
    *,
    source_split: str,
) -> TaskExample:
    fields = dict(record)
    return TaskExample(
        task_id=config.task_id,
        example_id=stable_source_id(config, record),
        role=role,
        fields=fields,
        metadata={
            "source_split": source_split,
            "source_content_sha256": canonical_content_hash(fields),
        },
    )


def build_task_splits(config: TaskDataConfig, provider: DatasetProvider) -> TaskSplits:
    """Build deterministic disjoint project roles from pinned original source splits."""

    if config.evaluation_only:
        raise ValueError(f"Evaluation-only dataset {config.task_id!r} cannot build training splits")
    if config.source_train_split is None:
        raise ValueError(f"Trainable task {config.task_id!r} has no source training split")
    source_train = provider.load(config, config.source_train_split)
    if not source_train:
        raise ValueError(f"Dataset {config.task_id!r} source training split is empty")
    training_examples = [
        _as_example(config, record, DatasetRole.TRAIN, source_split=config.source_train_split)
        for record in source_train
    ]
    training_examples.sort(key=lambda example: _rank_key(config.split_seed, example.example_id))

    has_official_validation = config.source_validation_split is not None
    validation_fraction = 0.0 if has_official_validation else config.validation_fraction
    validation_count = _partition_count(
        len(training_examples), validation_fraction, reserved=2 if config.anchor_fraction > 0 else 1
    )
    anchor_count = _partition_count(
        len(training_examples),
        config.anchor_fraction,
        reserved=validation_count + 1,
    )
    derived_validation = training_examples[:validation_count]
    derived_anchor = training_examples[validation_count : validation_count + anchor_count]
    retained_train = training_examples[validation_count + anchor_count :]

    train = tuple(
        _as_example(
            config, example.fields, DatasetRole.TRAIN, source_split=config.source_train_split
        )
        for example in retained_train
    )
    anchors = tuple(
        _as_example(
            config,
            example.fields,
            DatasetRole.ANCHOR_CANDIDATE,
            source_split=config.source_train_split,
        )
        for example in derived_anchor
    )
    if has_official_validation:
        assert config.source_validation_split is not None
        validation_records = provider.load(config, config.source_validation_split)
        validation = tuple(
            _as_example(
                config,
                record,
                DatasetRole.VALIDATION,
                source_split=config.source_validation_split,
            )
            for record in validation_records
        )
    else:
        validation = tuple(
            _as_example(
                config,
                example.fields,
                DatasetRole.VALIDATION,
                source_split=config.source_train_split,
            )
            for example in derived_validation
        )
    test_records = provider.load(config, config.source_test_split)
    test = tuple(
        _as_example(config, record, DatasetRole.TEST, source_split=config.source_test_split)
        for record in test_records
    )
    if not train or not validation or not anchors or not test:
        raise ValueError(
            f"Task {config.task_id!r} produced an empty required project split: "
            f"train={len(train)}, validation={len(validation)}, "
            f"anchor={len(anchors)}, test={len(test)}"
        )
    return TaskSplits(
        train=train,
        validation=validation,
        anchor_candidates=anchors,
        test=test,
    )


def load_evaluation_only_examples(
    config: TaskDataConfig, provider: DatasetProvider
) -> tuple[TaskExample, ...]:
    """Load a pinned capability dataset with no training-role API."""

    if not config.evaluation_only:
        raise ValueError(f"Task {config.task_id!r} is not evaluation-only")
    records = provider.load(config, config.source_test_split)
    occurrences: dict[str, int] = {}
    examples: list[TaskExample] = []
    for record in records:
        example = _as_example(
            config,
            record,
            DatasetRole.EVALUATION_ONLY,
            source_split=config.source_test_split,
        )
        occurrence = occurrences.get(example.example_id, 0)
        occurrences[example.example_id] = occurrence + 1
        if occurrence:
            example = TaskExample(
                task_id=example.task_id,
                example_id=f"{example.example_id}:occurrence-{occurrence}",
                role=example.role,
                fields=example.fields,
                metadata={**example.metadata, "source_occurrence": occurrence},
            )
        examples.append(example)
    return tuple(examples)
