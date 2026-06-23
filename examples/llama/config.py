# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LLaMA-3.2 model configuration."""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class RopeScaling:
    rope_type: str = "llama3"
    factor: float = 32.0
    low_freq_factor: float = 1.0
    high_freq_factor: float = 4.0
    original_max_position_embeddings: int = 8192


@dataclass
class LlamaConfig:
    D: int = 2048  # hidden_size
    Dff: int = 8192  # intermediate_size
    H: int = 32  # num_attention_heads
    Hkv: int = 8  # num_key_value_heads
    dh: int = 64  # head_dim
    n_layers: int = 16
    vocab: int = 128256
    eps: float = 1e-5
    rope_theta: float = 500000.0
    tie_embeddings: bool = True
    rope: RopeScaling = field(default_factory=RopeScaling)

    @property
    def Dkv(self) -> int:
        return self.Hkv * self.dh

    @property
    def G(self) -> int:
        return self.H // self.Hkv

    @classmethod
    def from_pretrained(cls, path: str) -> LlamaConfig:
        import json
        import os

        with open(os.path.join(path, "config.json")) as f:
            c = json.load(f)
        rs = c.get("rope_scaling") or {}
        rope = RopeScaling(
            rope_type=rs.get("rope_type", "llama3"),
            factor=float(rs.get("factor", 32.0)),
            low_freq_factor=float(rs.get("low_freq_factor", 1.0)),
            high_freq_factor=float(rs.get("high_freq_factor", 4.0)),
            original_max_position_embeddings=int(
                rs.get("original_max_position_embeddings", 8192)
            ),
        )
        return cls(
            D=c["hidden_size"],
            Dff=c["intermediate_size"],
            H=c["num_attention_heads"],
            Hkv=c["num_key_value_heads"],
            dh=c.get("head_dim", c["hidden_size"] // c["num_attention_heads"]),
            n_layers=c["num_hidden_layers"],
            vocab=c["vocab_size"],
            eps=c.get("rms_norm_eps", 1e-5),
            rope_theta=c.get("rope_theta", 500000.0),
            tie_embeddings=c.get("tie_word_embeddings", True),
            rope=rope,
        )
