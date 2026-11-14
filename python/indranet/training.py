"""Small training helpers; bring your own loop for anything serious."""
from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
from torch import nn


def make_optimizer(model: nn.Module, lr: float = 3e-4, weight_decay: float = 0.1,
                   betas: Tuple[float, float] = (0.9, 0.95)) -> torch.optim.AdamW:
    """AdamW with weight decay on matrices only (not biases, norms or 1-D params)."""
    decay, no_decay = [], []
    seen = set()
    for p in model.parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas)


def cosine_schedule(step: int, max_steps: int, warmup: int, lr: float, min_lr_ratio: float = 0.1) -> float:
    """Linear warmup, then cosine decay to ``min_lr_ratio * lr``."""
    if step < warmup:
        return lr * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, max_steps - warmup))
    return lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr
