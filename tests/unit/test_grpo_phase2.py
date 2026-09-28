from __future__ import annotations

import torch
from omegaconf import OmegaConf

from rahc_lora.config import ConfigurationError, compose_config
from rahc_lora.config.schema import RootConfig
from rahc_lora.models.factory import build_policy
from rahc_lora.models.policy import PolicyGenerationConfig
from rahc_lora.rewards.base import RewardResult
from rahc_lora.rl.advantages import normalize_group_advantages
from rahc_lora.rl.losses import clipped_grpo_loss
from rahc_lora.rl.rollout import RewardDiagnostics
from rahc_lora.utils.reproducibility import seed_everything


def test_group_advantages_are_finite_normalized_and_zero_safe() -> None:
    rewards = torch.tensor([[0.0, 1.0], [0.5, 0.5]], dtype=torch.float32)
    result = normalize_group_advantages(rewards, epsilon=1.0e-8, clip=5.0)
    assert torch.equal(result.values_gk, torch.tensor([[-1.0, 1.0], [0.0, 0.0]]))
    assert result.zero_variance_groups == 1
    assert result.clipped_fraction == 0.0


def test_group_advantages_reject_nonfinite_rewards() -> None:
    rewards = torch.tensor([[0.0, torch.inf]], dtype=torch.float32)
    try:
        normalize_group_advantages(rewards, epsilon=1.0e-8, clip=5.0)
    except ValueError as error:
        assert "non-finite" in str(error)
    else:
        raise AssertionError("non-finite rewards must fail")


def test_reward_diagnostics_distinguish_parse_failures_from_code_execution() -> None:
    diagnostics = RewardDiagnostics.from_results(
        (
            RewardResult(
                raw_reward=0.0,
                normalized_reward=0.0,
                parse_succeeded=False,
                parsed_response=None,
                reward_version="isolated-hidden-tests-v1",
                details={"passed": 0, "total": 2, "parsing_failure": True},
            ),
            RewardResult(
                raw_reward=0.0,
                normalized_reward=0.0,
                parse_succeeded=True,
                parsed_response="python_source",
                reward_version="isolated-hidden-tests-v1",
                details={
                    "passed": 0,
                    "total": 3,
                    "timed_out": True,
                    "output_limited": False,
                    "exit_code": 124,
                },
            ),
            RewardResult(
                raw_reward=0.5,
                normalized_reward=0.5,
                parse_succeeded=True,
                parsed_response="python_source",
                reward_version="isolated-hidden-tests-v1",
                details={
                    "passed": 1,
                    "total": 2,
                    "timed_out": False,
                    "output_limited": False,
                    "exit_code": 0,
                },
            ),
        )
    )

    assert diagnostics.as_dict() == {
        "parse_failure_count": 1,
        "code_execution_count": 2,
        "code_tests_passed": 1,
        "code_tests_total": 5,
        "code_timeout_count": 1,
        "code_output_limited_count": 0,
        "code_nonzero_exit_count": 1,
    }


def test_reward_diagnostics_reject_incomplete_code_test_details() -> None:
    result = RewardResult(
        raw_reward=0.0,
        normalized_reward=0.0,
        parse_succeeded=True,
        parsed_response="python_source",
        reward_version="isolated-hidden-tests-v1",
        details={"passed": 0},
    )

    try:
        RewardDiagnostics.from_results((result,))
    except ValueError as error:
        assert "passed and total" in str(error)
    else:
        raise AssertionError("partial code-test diagnostics must fail loudly")


def test_clipped_grpo_loss_is_finite_and_backpropagates() -> None:
    new = torch.tensor([[-0.8, -1.1], [-0.7, -1.2]], requires_grad=True)
    old = torch.tensor([[-1.0, -1.0], [-1.0, -1.0]])
    advantages = torch.tensor([1.0, -1.0])
    mask = torch.tensor([[True, True], [True, False]])
    result = clipped_grpo_loss(new, old, advantages, mask, clip_epsilon=0.2)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert new.grad is not None
    assert torch.isfinite(new.grad).all()


def test_tiny_factory_freezes_backbone_and_samples_complete_groups() -> None:
    resolved = compose_config(
        overrides=[
            "model=tiny",
            "method=rl_lora_tiny",
            "rollout.max_prompt_tokens=16",
            "rollout.max_response_tokens=2",
        ]
    )
    config = OmegaConf.to_object(resolved)
    assert isinstance(config, RootConfig)
    seed_everything(config.seed, deterministic_algorithms=True)
    bundle = build_policy(
        config.model,
        config.method.lora,
        requested_device="cpu",
    )
    policy = bundle.policy
    trainable_names = set(policy.named_trainable_parameters())
    assert trainable_names == {"q_proj.lora_A", "q_proj.lora_B"}
    assert all(
        parameter.requires_grad == (name in trainable_names)
        for name, parameter in policy.model.named_parameters()
    )
    generator = torch.Generator(device="cpu").manual_seed(7)
    samples = policy.sample(
        ("one", "two"),
        group_size=3,
        generation=PolicyGenerationConfig(
            max_prompt_tokens=16,
            max_response_tokens=2,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
        ),
        generator=generator,
    )
    assert len(samples) == 6
    logprobs, mask = policy.response_logprobs(samples)
    assert logprobs.shape == mask.shape == (6, 2)
    assert torch.isfinite(logprobs).all()
    assert mask.all()


def test_model_revisions_and_standard_baseline_retention_are_validated() -> None:
    try:
        compose_config(overrides=["model.revision=main"])
    except ConfigurationError as error:
        assert "model.revision" in str(error)
    else:
        raise AssertionError("mutable model revisions must fail")

    try:
        compose_config(overrides=["method.use_parameter_consolidation=true"])
    except ConfigurationError as error:
        assert "standard RL-LoRA" in str(error)
    else:
        raise AssertionError("standard baseline must reject hidden retention")


def test_enabled_capabilities_require_versioned_official_scorer_repository() -> None:
    try:
        compose_config(overrides=["general_capabilities.enabled=true"])
    except ConfigurationError as error:
        assert "general_capabilities.ifeval_scorer_repository" in str(error)
    else:
        raise AssertionError("enabled capability evaluation must require an official checkout")

    try:
        compose_config(overrides=["general_capabilities.max_examples_per_capability=1"])
    except ConfigurationError as error:
        assert "every fixed MMLU subject" in str(error)
    else:
        raise AssertionError("capability cap must preserve the fixed MMLU subject set")
