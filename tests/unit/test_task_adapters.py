from __future__ import annotations

from rahc_lora.rewards.code_tests import CodeTestsScorer, SandboxExecution
from rahc_lora.tasks.base import DatasetRole, TaskExample, TaskSplits
from rahc_lora.tasks.code_task import MBPPTask
from rahc_lora.tasks.constraint_generator import generate_constraint_splits
from rahc_lora.tasks.constraint_task import ConstraintTask
from rahc_lora.tasks.registry import default_task_registry


class PassingSandbox:
    def run(self, source: str, tests: tuple[str, ...]) -> SandboxExecution:
        return SandboxExecution(
            passed=len(tests),
            total=len(tests),
            exit_code=0,
            timed_out=False,
            output_limited=False,
            stdout="",
            stderr="",
        )


def _mbpp_splits() -> TaskSplits:
    def example(role: DatasetRole, index: int) -> TaskExample:
        return TaskExample(
            task_id="mbpp",
            example_id=f"mbpp:{role.value}",
            role=role,
            fields={
                "task_id": index,
                "prompt": f"Return {index} from solution().",
                "test_list": [f"assert solution() == {index}"],
            },
        )

    return TaskSplits(
        train=(example(DatasetRole.TRAIN, 1),),
        validation=(example(DatasetRole.VALIDATION, 2),),
        anchor_candidates=(example(DatasetRole.ANCHOR_CANDIDATE, 3),),
        test=(example(DatasetRole.TEST, 4),),
    )


def test_mbpp_prompt_hides_tests_and_scores_only_through_sandbox() -> None:
    task = MBPPTask(_mbpp_splits(), CodeTestsScorer(PassingSandbox()))
    example = task.load_test()[0]
    prompt = task.format_prompt(example)
    assert "assert solution" not in prompt
    assert task.score_response(example, "def solution(): return 4") == 1.0


def test_constraint_task_adapter_uses_generated_validator_contract() -> None:
    from rahc_lora.config.schema import ConstraintGeneratorConfig

    splits = generate_constraint_splits(
        ConstraintGeneratorConfig(
            split_sizes={
                "train": 2,
                "validation": 2,
                "anchor_candidate": 2,
                "test": 2,
            }
        )
    )
    task = ConstraintTask(splits)
    assert task.format_prompt(task.load_test()[0]).startswith("Scenario test-")


def test_default_registry_contains_exact_primary_stream() -> None:
    assert default_task_registry().task_ids == (
        "arc_challenge",
        "constraints",
        "gsm8k",
        "mbpp",
    )
