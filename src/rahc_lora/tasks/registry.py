"""Explicit task registry and construction from typed configuration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from rahc_lora.config.schema import ConstraintGeneratorConfig, TaskCatalogConfig, TaskDataConfig
from rahc_lora.rewards.code_tests import CodeSandbox, CodeTestsScorer
from rahc_lora.tasks.base import ContinualTask
from rahc_lora.tasks.code_task import MBPPTask
from rahc_lora.tasks.constraint_generator import generate_constraint_splits
from rahc_lora.tasks.constraint_task import ConstraintTask
from rahc_lora.tasks.data import DatasetProvider, build_task_splits
from rahc_lora.tasks.math_task import GSM8KTask
from rahc_lora.tasks.science_task import ARCChallengeTask


@dataclass(frozen=True)
class TaskBuildContext:
    """Dependencies permitted while constructing task adapters."""

    provider: DatasetProvider
    constraint_generator: ConstraintGeneratorConfig
    code_sandbox: CodeSandbox


TaskFactory = Callable[[TaskDataConfig, TaskBuildContext], ContinualTask]


@dataclass
class TaskRegistry:
    """No-global-state mapping from configured task IDs to adapter factories."""

    _factories: dict[str, TaskFactory] = field(default_factory=dict)

    def register(self, task_id: str, factory: TaskFactory) -> None:
        if not task_id:
            raise ValueError("Task registry ID must not be empty")
        if task_id in self._factories:
            raise ValueError(f"Task {task_id!r} is already registered")
        self._factories[task_id] = factory

    def build(self, config: TaskDataConfig, context: TaskBuildContext) -> ContinualTask:
        try:
            factory = self._factories[config.task_id]
        except KeyError as error:
            raise KeyError(f"No task adapter is registered for {config.task_id!r}") from error
        return factory(config, context)

    @property
    def task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._factories))


def default_task_registry() -> TaskRegistry:
    """Create the exact primary continual-RL registry."""

    registry = TaskRegistry()
    registry.register(
        "gsm8k",
        lambda config, context: GSM8KTask(
            build_task_splits(config, context.provider), reward_version=config.reward_version
        ),
    )
    registry.register(
        "arc_challenge",
        lambda config, context: ARCChallengeTask(
            build_task_splits(config, context.provider), reward_version=config.reward_version
        ),
    )
    registry.register(
        "mbpp",
        lambda config, context: MBPPTask(
            build_task_splits(config, context.provider),
            CodeTestsScorer(context.code_sandbox, reward_version=config.reward_version),
        ),
    )
    registry.register(
        "constraints",
        lambda config, context: ConstraintTask(
            generate_constraint_splits(context.constraint_generator),
            reward_version=config.reward_version,
        ),
    )
    return registry


def build_task_catalog(
    configs: TaskCatalogConfig,
    context: TaskBuildContext,
    registry: TaskRegistry | None = None,
) -> dict[str, ContinualTask]:
    """Build every primary task through the same registry contract."""

    selected_registry = registry or default_task_registry()
    return {
        config.task_id: selected_registry.build(config, context)
        for config in (
            configs.gsm8k,
            configs.arc_challenge,
            configs.mbpp,
            configs.constraints,
        )
    }
