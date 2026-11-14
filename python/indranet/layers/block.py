from __future__ import annotations

from typing import Optional, Tuple

from torch import Tensor, nn

from ..hooks import HookPoint
from ..registry import ATTENTIONS, MLPS, NORMS
from .attention import KV


class Block(nn.Module):
    """One decoder layer: attention + optional MLP, with the configured norm placement.

      pre:       x = x + attn(ln1(x));             x = x + mlp(ln2(x))
      post:      x = ln1(x + attn(x));             x = ln2(x + mlp(x))
      sandwich:  x = x + ln1_post(attn(ln1(x)));   x = x + ln2_post(mlp(ln2(x)))
      parallel:  x = x + attn(ln1(x)) + mlp(ln2(x))

    Hook points (all (B,T,d_model)): hook_resid_pre, hook_attn_in, hook_attn_out,
    hook_resid_mid, hook_mlp_in, hook_mlp_out, hook_resid_post. hook_mlp_in and
    hook_mlp_out are the input/output pairs a transcoder is trained on.
    """

    def __init__(self, cfg, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm_position = cfg.norm_position
        self.parallel = cfg.parallel_residual
        sandwich = cfg.norm_position == "sandwich"

        self.ln1 = NORMS.build(cfg.norm, cfg.d_model, cfg)
        self.attn = ATTENTIONS.build(cfg.attention, cfg, layer_idx)
        self.ln1_post = NORMS.build(cfg.norm, cfg.d_model, cfg) if sandwich else None
        self.mlp = MLPS.build(cfg.mlp, cfg)
        has_mlp = self.mlp is not None
        self.ln2 = NORMS.build(cfg.norm, cfg.d_model, cfg) if has_mlp else None
        self.ln2_post = NORMS.build(cfg.norm, cfg.d_model, cfg) if has_mlp and sandwich else None
        self.drop = nn.Dropout(cfg.dropout)

        self.hook_resid_pre, self.hook_attn_in, self.hook_attn_out = HookPoint(), HookPoint(), HookPoint()
        self.hook_resid_mid, self.hook_resid_post = HookPoint(), HookPoint()
        if has_mlp:
            self.hook_mlp_in, self.hook_mlp_out = HookPoint(), HookPoint()

    def _attn(self, x: Tensor, positions: Tensor, past_kv: Optional[KV]) -> Tuple[Tensor, KV]:
        out, kv = self.attn(self.hook_attn_in(x), positions, past_kv)
        if self.ln1_post is not None:
            out = self.ln1_post(out)
        return self.hook_attn_out(self.drop(out)), kv

    def _mlp(self, x: Tensor) -> Tensor:
        out = self.mlp(self.hook_mlp_in(x))
        if self.ln2_post is not None:
            out = self.ln2_post(out)
        return self.hook_mlp_out(self.drop(out))

    def forward(self, x: Tensor, positions: Tensor, past_kv: Optional[KV] = None) -> Tuple[Tensor, KV]:
        x = self.hook_resid_pre(x)
        post = self.norm_position == "post"

        attn_out, kv = self._attn(x if post else self.ln1(x), positions, past_kv)
        if self.parallel:
            mid = self.hook_resid_mid(x + attn_out)  # mid = resid_pre + attn, for analysis only
            out = mid + self._mlp(self.ln2(x)) if self.mlp is not None else mid
            return self.hook_resid_post(out), kv

        x = x + attn_out
        if post:
            x = self.ln1(x)
        x = self.hook_resid_mid(x)
        if self.mlp is not None:
            x = x + self._mlp(x if post else self.ln2(x))
            if post:
                x = self.ln2(x)
        return self.hook_resid_post(x), kv
