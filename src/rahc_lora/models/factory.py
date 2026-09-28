"""Pinned model/tokenizer loading and one-shared-LoRA policy construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from rahc_lora.config.schema import LoRAConfig, ModelConfig
from rahc_lora.models.policy import HuggingFaceTokenizer, TorchCausalPolicy
from rahc_lora.models.tiny import ByteTokenizer, TinyCausalLM


@dataclass(frozen=True)
class PolicyBundle:
    """Constructed trainable policy and immutable model identity."""

    policy: TorchCausalPolicy
    model_identifier: str
    model_revision: str
    tokenizer_identifier: str
    tokenizer_revision: str


def resolve_device(requested: str) -> torch.device:
    """Resolve an explicit launcher device without silent accelerator fallback."""

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested device {requested!r}, but CUDA is unavailable")
    return device


def _dtype(config: ModelConfig, device: torch.device) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = mapping[config.precision]
    supported = not (
        (dtype == torch.float16 and device.type == "cpu")
        or (
            dtype == torch.bfloat16 and device.type == "cuda" and not torch.cuda.is_bf16_supported()
        )
    )
    if not supported:
        if not config.allow_precision_fallback:
            raise RuntimeError(
                f"Precision {config.precision!r} is unsupported on {device}; fallback is disabled"
            )
        return torch.float32
    return dtype


def _build_transformers_policy(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    *,
    device: torch.device,
    cache_dir: str | None,
) -> TorchCausalPolicy:
    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = _dtype(model_config, device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.tokenizer_identifier,
        revision=model_config.tokenizer_revision,
        cache_dir=cache_dir,
        trust_remote_code=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.identifier,
        revision=model_config.revision,
        cache_dir=cache_dir,
        trust_remote_code=False,
        torch_dtype=dtype,
    )
    if int(model.config.vocab_size) != model_config.vocab_size:
        raise ValueError(
            "Configured vocabulary size does not match pinned model: "
            f"{model_config.vocab_size} != {model.config.vocab_size}"
        )
    model.requires_grad_(False)
    peft_config = PeftLoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.rank,
        lora_alpha=lora_config.alpha,
        lora_dropout=lora_config.dropout,
        target_modules=list(lora_config.target_modules),
        bias="none",
    )
    policy_model = get_peft_model(model, peft_config)
    return TorchCausalPolicy(
        policy_model,
        HuggingFaceTokenizer(tokenizer),
        device=device,
        max_sequence_length=model_config.max_sequence_length,
    )


def build_policy(
    model_config: ModelConfig,
    lora_config: LoRAConfig,
    *,
    requested_device: str,
    cache_dir: str | None = None,
) -> PolicyBundle:
    """Build one frozen causal backbone with exactly one shared LoRA policy."""

    device = resolve_device(requested_device)
    if model_config.backend == "tiny":
        if set(lora_config.target_modules) != {"q_proj"}:
            raise ValueError("Tiny policy requires method.lora.target_modules=[q_proj]")
        if model_config.precision != "float32":
            raise ValueError("Tiny policy supports only explicit float32 precision")
        model = TinyCausalLM(
            vocab_size=model_config.vocab_size,
            hidden_size=model_config.hidden_size,
            rank=lora_config.rank,
            alpha=lora_config.alpha,
            dropout=lora_config.dropout,
        )
        policy = TorchCausalPolicy(
            model,
            ByteTokenizer(),
            device=device,
            max_sequence_length=model_config.max_sequence_length,
        )
    elif model_config.backend == "transformers":
        policy = _build_transformers_policy(
            model_config,
            lora_config,
            device=device,
            cache_dir=str(Path(cache_dir).resolve()) if cache_dir else None,
        )
    else:
        raise ValueError(f"Unsupported model backend: {model_config.backend!r}")
    return PolicyBundle(
        policy=policy,
        model_identifier=model_config.identifier,
        model_revision=model_config.revision,
        tokenizer_identifier=model_config.tokenizer_identifier,
        tokenizer_revision=model_config.tokenizer_revision,
    )
