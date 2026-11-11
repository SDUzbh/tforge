"""Export trained models (and reference activations) for the C++ interpretability tools.

Files are safetensors: an 8-byte little-endian header length, a JSON header, then raw
tensor bytes. The model's config is stored as JSON in the header metadata, so a single
file fully describes the model.

    from tforge.export import export_model
    export_model(model, "model.safetensors")

Then, from interp/: ``./build/tfi info model.safetensors``.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, Optional, Union

import torch
from torch import Tensor

from .model import DecoderLM

FORMAT_VERSION = "1"
_DTYPES = {torch.float32: "F32", torch.int64: "I64"}

# Components the C++ side implements. Custom registered components cannot be exported.
SUPPORTED = {
    "norm": {"rmsnorm", "layernorm", "none"},
    "positional": {"rope", "alibi", "learned", "sinusoidal", "none"},
    "attention": {"causal"},
    "mlp": {"mlp", "glu", "swiglu", "geglu", "none"},
    "activation": {"relu", "gelu", "gelu_tanh", "silu", "tanh", "relu2"},
}


def save_safetensors(tensors: Dict[str, Tensor], path: Union[str, Path],
                     metadata: Optional[Dict[str, str]] = None) -> None:
    header: Dict[str, object] = {"__metadata__": dict(metadata or {})}
    blobs, offset = [], 0
    for name, t in tensors.items():
        t = t.detach().cpu().contiguous()
        if t.dtype not in _DTYPES:
            t = t.to(torch.float32)
        data = t.numpy().tobytes()
        header[name] = {"dtype": _DTYPES[t.dtype], "shape": list(t.shape),
                        "data_offsets": [offset, offset + len(data)]}
        blobs.append(data)
        offset += len(data)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for b in blobs:
            f.write(b)


def _check_supported(model: DecoderLM) -> None:
    cfg = model.cfg
    for i in range(cfg.n_layers):
        lc = cfg.layer_config(i)
        for kind, value in [("norm", lc.norm), ("positional", lc.positional), ("attention", lc.attention),
                            ("mlp", lc.mlp), ("activation", lc.activation)]:
            if value not in SUPPORTED[kind]:
                raise ValueError(f"layer {i}: {kind} '{value}' has no C++ implementation; "
                                 f"supported: {sorted(SUPPORTED[kind])}")


def export_model(model: DecoderLM, path: Union[str, Path]) -> None:
    """Write weights (FP32) and config. Tied unembedding weights are stored once."""
    _check_supported(model)
    state = model.state_dict()
    if model.cfg.tie_embeddings:
        state.pop("unembed.weight", None)
    meta = {"format": "tforge", "format_version": FORMAT_VERSION,
            "config": json.dumps(model.cfg.to_dict())}
    save_safetensors({k: v.float() for k, v in state.items()}, path, meta)


@torch.no_grad()
def export_reference(model: DecoderLM, tokens: Tensor, path: Union[str, Path]) -> None:
    """Write tokens, logits and every hook activation for one sequence, for golden tests."""
    if tokens.dim() == 1:
        tokens = tokens[None]
    if tokens.shape[0] != 1:
        raise ValueError("reference export takes a single sequence")
    was_training = model.training
    model.eval()
    try:
        out, cache = model.run_with_cache(tokens)
    finally:
        model.train(was_training)
    tensors = {"tokens": tokens[0].long(), "logits": out.logits[0].float()}
    for name, act in cache.items():
        # hook_pos_embed is (T, d) with no batch dimension; everything else is batch-first.
        tensors[f"cache.{name}"] = (act if name == "hook_pos_embed" else act[0]).float()
    save_safetensors(tensors, path, {"format": "tforge-reference"})
