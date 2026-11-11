from __future__ import annotations

from typing import Optional

from torch import Tensor, nn

from ..hooks import HookPoint
from ..registry import ACTIVATIONS, MLPS


def _round_up(x: int, multiple: int) -> int:
    return multiple * ((x + multiple - 1) // multiple)


@MLPS.register("mlp")
class MLP(nn.Module):
    """up -> activation -> down.

    Hook points: hook_pre (B,T,d_mlp) before the activation, hook_post after it.
    """

    def __init__(self, cfg, activation: Optional[str] = None):
        super().__init__()
        d_mlp = cfg.d_mlp or 4 * cfg.d_model
        self.up = nn.Linear(cfg.d_model, d_mlp, bias=cfg.bias)
        self.act = ACTIVATIONS.build(activation or cfg.activation)
        self.down = nn.Linear(d_mlp, cfg.d_model, bias=cfg.bias)
        self.down.is_residual_out = True
        self.hook_pre, self.hook_post = HookPoint(), HookPoint()

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.hook_post(self.act(self.hook_pre(self.up(x)))))


@MLPS.register("glu")
class GatedMLP(nn.Module):
    """down(act(gate(x)) * up(x)) -- the GLU family (SwiGLU, GeGLU, ...).

    The default width is 8/3 * d_model (rounded up to ``mlp_multiple_of``), which
    keeps the parameter count close to a plain 4x MLP. Hook points: hook_pre is the
    gate pre-activation, hook_post the gated hidden state fed to ``down``.
    """

    def __init__(self, cfg, activation: Optional[str] = None):
        super().__init__()
        d_mlp = cfg.d_mlp or _round_up(int(8 * cfg.d_model / 3), cfg.mlp_multiple_of)
        self.gate = nn.Linear(cfg.d_model, d_mlp, bias=cfg.bias)
        self.up = nn.Linear(cfg.d_model, d_mlp, bias=cfg.bias)
        self.act = ACTIVATIONS.build(activation or cfg.activation)
        self.down = nn.Linear(d_mlp, cfg.d_model, bias=cfg.bias)
        self.down.is_residual_out = True
        self.hook_pre, self.hook_post = HookPoint(), HookPoint()

    def forward(self, x: Tensor) -> Tensor:
        return self.down(self.hook_post(self.act(self.hook_pre(self.gate(x))) * self.up(x)))


MLPS.register("swiglu", lambda cfg: GatedMLP(cfg, activation="silu"))
MLPS.register("geglu", lambda cfg: GatedMLP(cfg, activation=cfg.activation if cfg.activation.startswith("gelu") else "gelu"))
MLPS.register("none", lambda cfg: None)  # attention-only layer
