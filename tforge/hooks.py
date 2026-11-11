"""Named hook points for reading and editing activations.

A ``HookPoint`` is an identity module placed wherever an activation is worth
looking at. Its full module path (e.g. ``blocks.0.hook_mlp_out``) is its name,
so you can cache activations or swap in edited ones by name; see
``DecoderLM.run_with_cache`` and ``DecoderLM.hooks``.
"""
from __future__ import annotations

from torch import Tensor, nn


class HookPoint(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = ""  # set by DecoderLM to the module path

    def forward(self, x: Tensor) -> Tensor:
        return x

    @property
    def has_hooks(self) -> bool:
        return bool(self._forward_hooks)
