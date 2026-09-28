"""CPU-compatible causal model with a single project-owned LoRA adapter."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


class ByteTokenizer:
    """Versioned UTF-8 byte tokenizer used only by deterministic tiny tests."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259

    def encode(self, text: str, *, max_length: int) -> tuple[int, ...]:
        if max_length <= 0:
            raise ValueError("Tokenizer max_length must be positive")
        byte_ids = [value + 3 for value in text.encode("utf-8")]
        tokens = [self.bos_token_id, *byte_ids]
        if len(tokens) > max_length:
            tokens = (
                [self.bos_token_id]
                if max_length == 1
                else [self.bos_token_id, *tokens[-(max_length - 1) :]]
            )
        return tuple(tokens)

    def decode(self, token_ids: tuple[int, ...]) -> str:
        values = bytes(token - 3 for token in token_ids if 3 <= token < self.vocab_size)
        return values.decode("latin-1")


class LoRALinear(nn.Module):  # type: ignore[misc]
    """Frozen linear layer plus one trainable low-rank update."""

    def __init__(
        self,
        base: nn.Linear,
        *,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if rank <= 0 or alpha <= 0 or not 0 <= dropout < 1:
            raise ValueError("Invalid tiny LoRA rank, alpha, or dropout")
        self.base = base
        self.base.requires_grad_(False)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        update = (self.dropout(inputs) @ self.lora_A.transpose(0, 1)) @ self.lora_B.transpose(0, 1)
        return self.base(inputs) + update * self.scaling


@dataclass(frozen=True)
class TinyCausalLMOutput:
    """Transformers-compatible output surface."""

    logits: Tensor


class TinyCausalLM(nn.Module):  # type: ignore[misc]
    """Small frozen byte-level causal model with one shared LoRA projection."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        rank: int,
        alpha: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if vocab_size != ByteTokenizer.vocab_size:
            raise ValueError(
                f"Tiny model requires byte-tokenizer vocabulary {ByteTokenizer.vocab_size}"
            )
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.core = nn.Linear(hidden_size, hidden_size)
        self.q_proj = LoRALinear(
            nn.Linear(hidden_size, hidden_size),
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        for module in (self.embedding, self.core, self.lm_head):
            module.requires_grad_(False)

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> TinyCausalLMOutput:
        del attention_mask
        hidden = torch.tanh(self.core(self.embedding(input_ids)))
        hidden = torch.tanh(self.q_proj(hidden))
        return TinyCausalLMOutput(logits=self.lm_head(hidden))
