from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..hooks import HookPoint
from ..registry import ATTENTIONS, NORMS, POSITIONALS
from .positional import AttentionPositional, NoPositional

KV = Tuple[Tensor, Tensor]


def attention_mask(q_pos: Tensor, k_pos: Tensor, window: Optional[int]) -> Tensor:
    """Boolean (Tq, Tk) mask, True where query i may attend to key j."""
    diff = q_pos[:, None] - k_pos[None, :]
    mask = diff >= 0
    if window is not None:
        mask = mask & (diff < window)
    return mask


@ATTENTIONS.register("causal")
class CausalSelfAttention(nn.Module):
    """Causal multi-head attention covering MHA, GQA and MQA (via ``n_kv_heads``).

    Hook points (shapes for batch B, query length T, key length S):
      hook_q (B,H,T,D), hook_k / hook_v (B,Hkv,T,D) -- after qk-norm and RoPE, new tokens only
      hook_pattern (B,H,T,S) -- softmax output; hooking it switches off the fused kernel
      hook_z (B,T,H,D) -- per-head outputs before the output projection
    """

    def __init__(self, cfg, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads, self.n_kv_heads, self.d_head = cfg.n_heads, cfg.kv_heads, cfg.head_dim
        self.scale = cfg.attn_scale if cfg.attn_scale is not None else 1.0 / math.sqrt(self.d_head)
        self.window = cfg.sliding_window
        self.softcap = cfg.attn_logit_softcap
        self.use_sdpa = cfg.use_sdpa
        self.attn_dropout = cfg.attn_dropout

        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.d_head, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.d_head, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.d_head, bias=cfg.bias)
        self.o_proj = nn.Linear(self.n_heads * self.d_head, cfg.d_model, bias=cfg.bias)
        self.o_proj.is_residual_out = True

        norm = NORMS.get(cfg.norm if cfg.norm != "none" else "rmsnorm")
        self.q_norm = norm(self.d_head, cfg) if cfg.qk_norm else nn.Identity()
        self.k_norm = norm(self.d_head, cfg) if cfg.qk_norm else nn.Identity()

        pos = POSITIONALS.build(cfg.positional, cfg)
        # Embedding-level encodings are added at the input; attention then sees no positions.
        self.pos = pos if isinstance(pos, AttentionPositional) else NoPositional()

        self.hook_q, self.hook_k, self.hook_v = HookPoint(), HookPoint(), HookPoint()
        self.hook_pattern, self.hook_z = HookPoint(), HookPoint()

    def forward(self, x: Tensor, positions: Tensor, past_kv: Optional[KV] = None) -> Tuple[Tensor, KV]:
        B, T, _ = x.shape
        H, Hkv, D = self.n_heads, self.n_kv_heads, self.d_head

        q = self.q_proj(x).view(B, T, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        v = self.v_proj(x).view(B, T, Hkv, D).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        q, k = self.pos.rotate(q, positions), self.pos.rotate(k, positions)
        q, k, v = self.hook_q(q), self.hook_k(k), self.hook_v(v)

        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        new_kv = (k, v)

        S = k.shape[2]
        k_pos = torch.arange(S, device=x.device)
        mask = attention_mask(positions, k_pos, self.window)
        bias = self.pos.bias(positions, k_pos)
        if Hkv != H:
            k = k.repeat_interleave(H // Hkv, dim=1)
            v = v.repeat_interleave(H // Hkv, dim=1)
        dropout_p = self.attn_dropout if self.training else 0.0

        if self.use_sdpa and self.softcap is None and not self.hook_pattern.has_hooks:
            if bias is None and self.window is None and S == T:
                z = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=True, scale=self.scale)
            else:
                base = bias.to(q.dtype) if bias is not None else q.new_zeros(mask.shape)
                attn_mask = base.masked_fill(~mask, float("-inf"))
                z = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, scale=self.scale)
        else:
            scores = (q @ k.transpose(-2, -1)) * self.scale
            if self.softcap is not None:
                scores = self.softcap * torch.tanh(scores / self.softcap)
            if bias is not None:
                scores = scores + bias.to(scores.dtype)
            scores = scores.masked_fill(~mask, float("-inf"))
            pattern = self.hook_pattern(scores.float().softmax(dim=-1).to(q.dtype))
            z = F.dropout(pattern, dropout_p) @ v

        z = self.hook_z(z.transpose(1, 2))  # (B, T, H, D)
        return self.o_proj(z.reshape(B, T, H * D)), new_kv
