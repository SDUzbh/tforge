from __future__ import annotations

import torch
from torch import Tensor, nn

from ..registry import ACTIVATIONS, NORMS


@NORMS.register("layernorm")
class LayerNorm(nn.LayerNorm):
    def __init__(self, dim: int, cfg):
        super().__init__(dim, eps=cfg.norm_eps, bias=cfg.bias)


@NORMS.register("rmsnorm")
class RMSNorm(nn.Module):
    def __init__(self, dim: int, cfg):
        super().__init__()
        self.eps = cfg.norm_eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


@NORMS.register("none")
def _no_norm(dim: int, cfg) -> nn.Module:
    return nn.Identity()


class ReLUSquared(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.relu(x).square()


ACTIVATIONS.register("relu", nn.ReLU)
ACTIVATIONS.register("gelu", nn.GELU)
ACTIVATIONS.register("gelu_tanh", lambda: nn.GELU(approximate="tanh"))
ACTIVATIONS.register("silu", nn.SiLU)
ACTIVATIONS.register("tanh", nn.Tanh)
ACTIVATIONS.register("relu2", ReLUSquared)
