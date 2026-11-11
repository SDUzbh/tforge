"""Positional encodings.

Two kinds:
- ``EmbeddingPositional`` adds a vector to the token embedding once, at the input
  (learned, sinusoidal).
- ``AttentionPositional`` acts inside every attention layer, by rotating q/k
  (RoPE) or adding a bias to the attention scores (ALiBi). ``"none"`` (NoPE) is
  an AttentionPositional that does nothing, so it can be mixed per layer with
  RoPE or ALiBi through ``layer_overrides``.
"""
from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import Tensor, nn

from ..registry import POSITIONALS


class EmbeddingPositional(nn.Module):
    kind = "embedding"

    def forward(self, positions: Tensor) -> Tensor:  # (T,) -> (T, d_model)
        raise NotImplementedError


class AttentionPositional(nn.Module):
    kind = "attention"

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        """Transform q or k of shape (B, H, T, D) at the given absolute positions."""
        return x

    def bias(self, q_pos: Tensor, k_pos: Tensor) -> Optional[Tensor]:
        """Additive attention-score bias of shape (H, Tq, Tk), or None."""
        return None


@POSITIONALS.register("learned")
class LearnedPositional(EmbeddingPositional):
    def __init__(self, cfg):
        super().__init__()
        self.max_seq_len = cfg.max_seq_len
        self.weight = nn.Parameter(torch.empty(cfg.max_seq_len, cfg.d_model))
        nn.init.normal_(self.weight, std=cfg.init_std)

    def forward(self, positions: Tensor) -> Tensor:
        return self.weight[positions]


@POSITIONALS.register("sinusoidal")
class SinusoidalPositional(EmbeddingPositional):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model

    def forward(self, positions: Tensor) -> Tensor:
        half = self.d_model // 2
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=positions.device) / half)
        angles = positions.float()[:, None] * freqs[None, :]
        pe = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.d_model % 2:
            pe = torch.nn.functional.pad(pe, (0, 1))
        return pe


@POSITIONALS.register("none")
class NoPositional(AttentionPositional):
    def __init__(self, cfg=None):
        super().__init__()


@POSITIONALS.register("rope")
class RotaryPositional(AttentionPositional):
    """Rotary embeddings, "rotate half" convention (GPT-NeoX / Llama).

    With ``rope_fraction < 1`` only the first part of each head's dims rotate.
    """

    def __init__(self, cfg):
        super().__init__()
        rot = int(cfg.head_dim * cfg.rope_fraction)
        self.rot_dim = rot - rot % 2
        if self.rot_dim == 0:
            raise ValueError("rope_fraction * head_dim leaves no dimensions to rotate")
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, self.rot_dim, 2).float() / self.rot_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def rotate(self, x: Tensor, positions: Tensor) -> Tensor:
        angles = positions.float()[:, None] * self.inv_freq[None, :]   # (T, rot/2)
        angles = torch.cat([angles, angles], dim=-1)                   # (T, rot)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        x_rot, x_pass = x[..., : self.rot_dim], x[..., self.rot_dim:]
        x1, x2 = x_rot.chunk(2, dim=-1)
        rotated = x_rot * cos + torch.cat([-x2, x1], dim=-1) * sin
        return torch.cat([rotated, x_pass], dim=-1)


def alibi_slopes(n_heads: int) -> List[float]:
    """Head slopes from the ALiBi paper, including its rule for non-power-of-2 head counts."""

    def pow2(n):
        start = 2 ** (-8 / n)
        return [start ** (i + 1) for i in range(n)]

    if math.log2(n_heads).is_integer():
        return pow2(n_heads)
    closest = 2 ** math.floor(math.log2(n_heads))
    return pow2(closest) + pow2(2 * closest)[0::2][: n_heads - closest]


@POSITIONALS.register("alibi")
class ALiBiPositional(AttentionPositional):
    def __init__(self, cfg):
        super().__init__()
        self.register_buffer("slopes", torch.tensor(alibi_slopes(cfg.n_heads)), persistent=False)

    def bias(self, q_pos: Tensor, k_pos: Tensor) -> Tensor:
        dist = (q_pos[:, None] - k_pos[None, :]).clamp(min=0).float()  # (Tq, Tk)
        return -self.slopes[:, None, None] * dist[None]
