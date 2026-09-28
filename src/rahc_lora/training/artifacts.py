"""Authoritative local run-artifact layout for continual experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from rahc_lora.config.schema import RootConfig
from rahc_lora.evaluation.capability_suite import CapabilityMetricResult
from rahc_lora.evaluation.evaluator import TaskEvaluationResult
from rahc_lora.evaluation.performance_matrix import PerformanceMatrix
from rahc_lora.utils.atomic import write_json_atomic, write_text_atomic
from rahc_lora.utils.logging import JsonlLogger
from rahc_lora.utils.manifests import build_run_manifest, write_run_manifest


def sha256_file(path: Path) -> str:
    """Hash one artifact without loading it wholly into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def collect_dataset_fingerprints(manifest_dir: Path) -> dict[str, str]:
    """Hash every checked-in dataset manifest and overlap report."""

    paths = sorted(manifest_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No dataset manifests found under {manifest_dir}")
    return {path.name: sha256_file(path) for path in paths}


class RunArtifacts:
    """Create and update one run's machine-readable artifact contract."""

    def __init__(
        self,
        resolved_config: DictConfig,
        typed_config: RootConfig,
        *,
        run_id: str,
        repository: Path,
        command_overrides: Sequence[str] = (),
    ) -> None:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id) is None:
            raise ValueError(f"Invalid run ID: {run_id!r}")
        self.resolved_config = resolved_config
        self.typed_config = typed_config
        self.repository = repository.resolve()
        self.run_dir = (
            Path(typed_config.paths.output_dir)
            / "runs"
            / typed_config.experiment.name
            / typed_config.method.name
            / typed_config.task_stream.name
            / str(typed_config.seed)
            / run_id
        ).resolve()
        if self.run_dir.exists() and any(self.run_dir.iterdir()):
            raise FileExistsError(f"Run directory already contains artifacts: {self.run_dir}")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir = Path(typed_config.data.manifest_dir).resolve()
        self.dataset_fingerprints = collect_dataset_fingerprints(manifest_dir)
        write_text_atomic(self.run_dir / "resolved_config.yaml", OmegaConf.to_yaml(resolved_config))
        write_run_manifest(
            self.run_dir / "run_manifest.json",
            build_run_manifest(
                resolved_config,
                repository=self.repository,
                dataset_fingerprints=self.dataset_fingerprints,
                command_overrides=list(command_overrides),
            ),
        )
        write_json_atomic(self.run_dir / "dataset_fingerprints.json", self.dataset_fingerprints)
        write_json_atomic(
            self.run_dir / "memory_manifest.json",
            {"enabled": False, "capacity": 0, "records": 0, "method": typed_config.method.name},
        )
        write_json_atomic(self.run_dir / "checkpoint_manifest.json", {"checkpoints": []})
        write_json_atomic(
            self.run_dir / "general_capabilities_status.json",
            {
                "enabled": typed_config.general_capabilities.enabled,
                "ifeval_scorer_repository": (
                    typed_config.general_capabilities.ifeval_scorer_repository
                ),
                "ifeval_scorer_revision": (
                    typed_config.general_capabilities.ifeval_scorer_revision
                ),
                "ifeval_scorer_seed": typed_config.general_capabilities.ifeval_scorer_seed,
                "ifeval_nltk_data_dir": (typed_config.general_capabilities.ifeval_nltk_data_dir),
                "ifeval_punkt_tab_sha256": (
                    typed_config.general_capabilities.ifeval_punkt_tab_sha256
                ),
                "max_examples_per_capability": (
                    typed_config.general_capabilities.max_examples_per_capability
                ),
                "max_perplexity_tokens": (typed_config.general_capabilities.max_perplexity_tokens),
            },
        )
        write_text_atomic(
            self.run_dir / "general_capabilities.csv",
            "after_task,capability,metric,absolute_score,frozen_baseline_score,"
            "change_from_frozen,higher_is_better,example_count\n",
        )
        pd.DataFrame(columns=["layer", "importance"]).to_parquet(
            self.run_dir / "layer_importance.parquet", index=False
        )
        pd.DataFrame(columns=["layer", "conflict"]).to_parquet(
            self.run_dir / "layer_conflicts.parquet", index=False
        )
        write_text_atomic(self.run_dir / "stdout.log", "")
        self.metrics = JsonlLogger(self.run_dir / "metrics.jsonl")
        self.system_metrics = JsonlLogger(self.run_dir / "system_metrics.jsonl")
        self._evaluation_stream = (self.run_dir / "evaluation_records.jsonl").open(
            "a", encoding="utf-8", newline="\n"
        )
        self._capability_stream = (self.run_dir / "general_capabilities.csv").open(
            "a", encoding="utf-8", newline=""
        )
        self._capability_writer = csv.writer(self._capability_stream, lineterminator="\n")

    def write_matrix(self, matrix: PerformanceMatrix) -> None:
        matrix.write_csv(self.run_dir / "performance_matrix.csv")

    def append_evaluations(
        self, after_task_id: str, results: tuple[TaskEvaluationResult, ...]
    ) -> None:
        for result in results:
            for record in result.records:
                value = {
                    "after_task": after_task_id,
                    "task_id": record.task_id,
                    "example_id": record.example_id,
                    "prompt_sha256": record.prompt_sha256,
                    "response_sha256": record.response_sha256,
                    "reward": asdict(record.reward),
                }
                self._evaluation_stream.write(
                    json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
                )
        self._evaluation_stream.flush()

    def append_capabilities(self, results: Sequence[CapabilityMetricResult]) -> None:
        """Append one or more complete capability metric groups."""

        for result in results:
            self._capability_writer.writerow(
                [
                    result.after_task,
                    result.capability,
                    result.metric,
                    format(result.absolute_score, ".17g"),
                    format(result.frozen_baseline_score, ".17g"),
                    format(result.change_from_frozen, ".17g"),
                    str(result.higher_is_better).lower(),
                    result.example_count,
                ]
            )
        self._capability_stream.flush()

    def register_checkpoint(self, checkpoint_path: Path, checksum: str) -> None:
        index_path = self.run_dir / "checkpoint_manifest.json"
        value = json.loads(index_path.read_text(encoding="utf-8"))
        checkpoints = value["checkpoints"]
        checkpoints.append(
            {
                "path": checkpoint_path.relative_to(self.run_dir).as_posix(),
                "manifest_sha256": checksum,
            }
        )
        write_json_atomic(index_path, value)

    def write_summary(self, value: dict[str, Any]) -> None:
        write_json_atomic(self.run_dir / "summary.json", value)

    def close(self) -> None:
        self.metrics.close()
        self.system_metrics.close()
        self._evaluation_stream.close()
        self._capability_stream.close()

    def __enter__(self) -> RunArtifacts:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
