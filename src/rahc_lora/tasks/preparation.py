"""Dataset preparation orchestration that writes manifests but never raw examples."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rahc_lora.config.schema import RootConfig, TaskDataConfig
from rahc_lora.tasks.base import TaskExample
from rahc_lora.tasks.constraint_generator import (
    PromptOverlapReport,
    all_constraint_examples,
    check_ifeval_prompt_overlap,
    generate_constraint_splits,
)
from rahc_lora.tasks.data import (
    DatasetProvider,
    HuggingFaceDatasetProvider,
    build_task_splits,
    load_evaluation_only_examples,
)
from rahc_lora.tasks.manifests import (
    DatasetManifest,
    build_dataset_manifest,
    build_evaluation_dataset_manifest,
    write_dataset_manifest,
)


@dataclass(frozen=True)
class DataPreparationResult:
    """Machine-readable summary of written manifests and overlap evidence."""

    manifest_paths: tuple[str, ...]
    example_counts: dict[str, int]
    ifeval_overlap: PromptOverlapReport


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
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


def _filter_capability_examples(
    config: RootConfig,
    capability: TaskDataConfig,
    examples: tuple[TaskExample, ...],
) -> tuple[TaskExample, ...]:
    if capability.task_id == "mmlu":
        selected_subjects = set(config.general_capabilities.mmlu_subjects)
        filtered = tuple(
            example for example in examples if example.fields.get("subject") in selected_subjects
        )
        observed = {str(example.fields.get("subject")) for example in filtered}
        missing = sorted(selected_subjects.difference(observed))
        if missing:
            raise ValueError(f"Pinned MMLU source is missing configured subjects: {missing}")
        return filtered
    if capability.task_id == "perplexity":
        filtered = tuple(
            example for example in examples if str(example.fields.get("text", "")).strip()
        )
        if not filtered:
            raise ValueError("Held-out perplexity corpus contains no nonempty text records")
        return filtered
    return examples


def prepare_data(
    config: RootConfig,
    *,
    provider: DatasetProvider | None = None,
    output_dir: Path | None = None,
) -> DataPreparationResult:
    """Prepare exact split manifests and verify custom prompts against official IFEval."""

    selected_provider = provider or HuggingFaceDatasetProvider(cache_dir=config.paths.cache_dir)
    manifest_dir = (output_dir or Path(config.data.manifest_dir)).resolve()
    pending_manifests: list[tuple[str, DatasetManifest]] = []
    example_counts: dict[str, int] = {}

    generated_splits = generate_constraint_splits(config.constraint_generator)
    ifeval_examples = load_evaluation_only_examples(
        config.general_capabilities.ifeval, selected_provider
    )
    official_prompts = tuple(str(example.fields["prompt"]) for example in ifeval_examples)
    overlap = check_ifeval_prompt_overlap(
        all_constraint_examples(generated_splits), official_prompts
    )

    for task_config in (config.tasks.gsm8k, config.tasks.arc_challenge, config.tasks.mbpp):
        splits = build_task_splits(task_config, selected_provider)
        manifest = build_dataset_manifest(task_config, splits)
        pending_manifests.append((f"{task_config.task_id}.manifest.json", manifest))
        example_counts[task_config.task_id] = sum(
            len(values) for values in manifest.memberships.values()
        )

    constraint_manifest = build_dataset_manifest(
        config.tasks.constraints,
        generated_splits,
        generation_metadata={
            **asdict(config.constraint_generator),
            "ifeval_dataset_revision": config.general_capabilities.ifeval.revision,
            "ifeval_overlap": asdict(overlap),
        },
    )
    pending_manifests.append(("constraints.manifest.json", constraint_manifest))
    example_counts["constraints"] = sum(
        len(values) for values in constraint_manifest.memberships.values()
    )

    if config.data.prepare_general_capabilities:
        generation_metadata = {
            "max_new_tokens": config.general_capabilities.max_new_tokens,
            "do_sample": config.general_capabilities.do_sample,
            "temperature": config.general_capabilities.temperature,
            "top_p": config.general_capabilities.top_p,
        }
        capability_configs = (
            config.general_capabilities.ifeval,
            config.general_capabilities.mmlu,
            config.general_capabilities.hellaswag,
            config.general_capabilities.arc_easy,
            config.general_capabilities.perplexity,
        )
        for capability in capability_configs:
            examples = (
                ifeval_examples
                if capability.task_id == "ifeval"
                else load_evaluation_only_examples(capability, selected_provider)
            )
            filtered = _filter_capability_examples(config, capability, examples)
            manifest = build_evaluation_dataset_manifest(
                capability,
                filtered,
                generation_metadata={
                    **generation_metadata,
                    "mmlu_subjects": (
                        config.general_capabilities.mmlu_subjects
                        if capability.task_id == "mmlu"
                        else None
                    ),
                },
            )
            pending_manifests.append((f"{capability.task_id}.manifest.json", manifest))
            example_counts[capability.task_id] = len(filtered)

    manifest_paths: list[str] = []
    for filename, manifest in pending_manifests:
        path = manifest_dir / filename
        write_dataset_manifest(path, manifest)
        manifest_paths.append(str(path))
    overlap_path = manifest_dir / "constraints_ifeval_overlap.json"
    _write_json_atomic(overlap_path, asdict(overlap))
    manifest_paths.append(str(overlap_path))

    return DataPreparationResult(
        manifest_paths=tuple(manifest_paths),
        example_counts=example_counts,
        ifeval_overlap=overlap,
    )
