from __future__ import annotations

from pathlib import Path
from typing import cast

from omegaconf import OmegaConf

from rahc_lora.config import compose_config
from rahc_lora.config.schema import RootConfig
from rahc_lora.tasks.data import InMemoryDatasetProvider
from rahc_lora.tasks.manifests import read_dataset_manifest
from rahc_lora.tasks.preparation import prepare_data


def _arc_record(index: int) -> dict[str, object]:
    return {
        "id": f"arc-{index}",
        "question": f"Science question {index}",
        "choices": {"label": ["A", "B"], "text": ["correct", "incorrect"]},
        "answerKey": "A",
    }


def _mbpp_record(index: int) -> dict[str, object]:
    return {
        "task_id": index,
        "prompt": f"Return the integer {index}",
        "test_list": [f"assert solution() == {index}"],
    }


def test_prepare_data_writes_only_manifests_and_overlap_evidence(tmp_path: Path) -> None:
    resolved = compose_config(overrides=["data.prepare_general_capabilities=false"])
    config = cast(RootConfig, OmegaConf.to_object(resolved))
    provider = InMemoryDatasetProvider(
        {
            ("openai/gsm8k", "main", "train"): tuple(
                {"question": f"Math question {index}", "answer": f"#### {index}"}
                for index in range(20)
            ),
            ("openai/gsm8k", "main", "test"): tuple(
                {"question": f"Held-out math {index}", "answer": f"#### {index}"}
                for index in range(4)
            ),
            ("allenai/ai2_arc", "ARC-Challenge", "train"): tuple(
                _arc_record(index) for index in range(20)
            ),
            ("allenai/ai2_arc", "ARC-Challenge", "validation"): tuple(
                _arc_record(100 + index) for index in range(2)
            ),
            ("allenai/ai2_arc", "ARC-Challenge", "test"): tuple(
                _arc_record(200 + index) for index in range(2)
            ),
            ("google-research-datasets/mbpp", "sanitized", "train"): tuple(
                _mbpp_record(index) for index in range(20)
            ),
            ("google-research-datasets/mbpp", "sanitized", "validation"): tuple(
                _mbpp_record(100 + index) for index in range(2)
            ),
            ("google-research-datasets/mbpp", "sanitized", "test"): tuple(
                _mbpp_record(200 + index) for index in range(2)
            ),
            ("google/IFEval", None, "train"): (
                {"key": 1, "prompt": "An unrelated official evaluation prompt."},
                {"key": 2, "prompt": "Another distinct held-out instruction."},
            ),
        }
    )
    result = prepare_data(config, provider=provider, output_dir=tmp_path)
    assert result.ifeval_overlap.exact_overlap_count == 0
    assert result.ifeval_overlap.normalized_overlap_count == 0
    assert set(result.example_counts) == {"gsm8k", "arc_challenge", "mbpp", "constraints"}
    assert {path.name for path in tmp_path.iterdir()} == {
        "gsm8k.manifest.json",
        "arc_challenge.manifest.json",
        "mbpp.manifest.json",
        "constraints.manifest.json",
        "constraints_ifeval_overlap.json",
    }
    constraints = read_dataset_manifest(tmp_path / "constraints.manifest.json")
    assert constraints.generation_metadata["ifeval_overlap"]["normalized_overlap_count"] == 0
    metadata = next(iter(constraints.example_metadata.values()))
    assert metadata["template_family"]
    assert metadata["constraint_types"]
    assert metadata["validator_version"] == "1.0.0"
    assert not any(path.suffix in {".arrow", ".parquet"} for path in tmp_path.iterdir())
