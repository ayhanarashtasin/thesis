from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from rahc_lora.config.schema import TaskDataConfig
from rahc_lora.tasks.base import DatasetRole
from rahc_lora.tasks.data import (
    InMemoryDatasetProvider,
    build_task_splits,
    load_evaluation_only_examples,
)
from rahc_lora.tasks.manifests import (
    build_dataset_manifest,
    read_dataset_manifest,
    write_dataset_manifest,
)


def _config() -> TaskDataConfig:
    return TaskDataConfig(
        task_id="synthetic",
        family="test",
        provider="huggingface",
        dataset_name="test/synthetic",
        revision="a" * 40,
        license="MIT",
        citation="Synthetic test fixture",
        source_train_split="train",
        source_validation_split=None,
        source_test_split="test",
        id_fields=["id"],
        split_seed=7,
        validation_fraction=0.2,
        anchor_fraction=0.2,
        preprocessing_version="prep-v1",
        prompt_version="prompt-v1",
        reward_version="reward-v1",
    )


def _provider(reverse: bool = False) -> InMemoryDatasetProvider:
    train = [{"id": index, "question": f"question {index}"} for index in range(20)]
    if reverse:
        train.reverse()
    test = [{"id": 100 + index, "question": f"test {index}"} for index in range(4)]
    return InMemoryDatasetProvider(
        {
            ("test/synthetic", None, "train"): tuple(train),
            ("test/synthetic", None, "test"): tuple(test),
        }
    )


def test_deterministic_split_roles_are_disjoint_and_order_independent() -> None:
    first = build_task_splits(_config(), _provider())
    second = build_task_splits(_config(), _provider(reverse=True))
    assert [example.example_id for example in first.train] == [
        example.example_id for example in second.train
    ]
    all_roles = {
        DatasetRole.TRAIN: first.train,
        DatasetRole.VALIDATION: first.validation,
        DatasetRole.ANCHOR_CANDIDATE: first.anchor_candidates,
        DatasetRole.TEST: first.test,
    }
    memberships = [{example.example_id for example in examples} for examples in all_roles.values()]
    assert all(
        left.isdisjoint(right)
        for i, left in enumerate(memberships)
        for right in memberships[i + 1 :]
    )
    assert all(example.metadata["source_split"] == "train" for example in first.anchor_candidates)


def test_official_test_overlap_is_rejected() -> None:
    provider = _provider()
    overlapping = dict(provider.datasets)
    overlapping[("test/synthetic", None, "test")] = (
        overlapping[("test/synthetic", None, "train")][0],
    )
    with pytest.raises(ValueError, match="overlaps roles"):
        build_task_splits(_config(), InMemoryDatasetProvider(overlapping))


def test_dataset_manifest_round_trip_records_exact_membership(tmp_path: Path) -> None:
    config = _config()
    splits = build_task_splits(config, _provider())
    manifest = build_dataset_manifest(config, splits)
    destination = tmp_path / "synthetic.manifest.json"
    write_dataset_manifest(destination, manifest)
    loaded = read_dataset_manifest(destination)
    assert loaded == manifest
    assert len(loaded.memberships["test"]) == len(splits.test)
    assert all(len(item.content_sha256) == 64 for item in loaded.memberships["train"])
    assert set(loaded.example_metadata) == {
        item.example_id for fingerprints in loaded.memberships.values() for item in fingerprints
    }


def test_empty_required_partition_fails_instead_of_silently_skipping() -> None:
    config = replace(_config(), validation_fraction=0.0)
    with pytest.raises(ValueError, match="empty required project split"):
        build_task_splits(config, _provider())


def test_evaluation_only_duplicate_source_ids_receive_stable_occurrences() -> None:
    config = replace(
        _config(),
        task_id="capability",
        dataset_name="test/capability",
        source_train_split=None,
        source_test_split="test",
        validation_fraction=0.0,
        anchor_fraction=0.0,
        evaluation_only=True,
    )
    records = (
        {"id": 1, "text": "first"},
        {"id": 1, "text": "second"},
    )
    provider = InMemoryDatasetProvider({("test/capability", None, "test"): records})
    examples = load_evaluation_only_examples(config, provider)
    assert examples[0].example_id != examples[1].example_id
    assert examples[1].metadata["source_occurrence"] == 1
