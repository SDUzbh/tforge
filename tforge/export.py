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
from typing import Dict, Optional, Sequence, Union

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


def export_model(model: DecoderLM, path: Union[str, Path], vocab: Optional[Sequence[str]] = None,
                 metadata: Optional[Dict[str, str]] = None) -> None:
    """Write weights (FP32) and config. Tied unembedding weights are stored once.

    ``vocab`` (one string per token id) lets the C++ tools take ``--text`` input and print
    tokens as text. ``metadata`` adds free-form string entries to the file header.
    """
    _check_supported(model)
    if vocab is not None and len(vocab) != model.cfg.vocab_size:
        raise ValueError(f"vocab has {len(vocab)} entries but the model's vocab_size is {model.cfg.vocab_size}")
    state = model.state_dict()
    if model.cfg.tie_embeddings:
        state.pop("unembed.weight", None)
    meta = dict(metadata or {})
    meta.update({"format": "tforge", "format_version": FORMAT_VERSION, "config": json.dumps(model.cfg.to_dict())})
    if vocab is not None:
        meta["vocab"] = json.dumps(list(vocab))
    save_safetensors({k: v.float() for k, v in state.items()}, path, meta)


def hook_gradients(model: DecoderLM, tokens: Tensor, pos: int, target: int, other: int = -1) -> Dict[str, Tensor]:
    """d metric / d activation for every hook point, by autograd, for one sequence.

    The metric is ``logits[pos, target] - logits[pos, other]``, or the log-probability of ``target``
    when ``other`` is negative (tfi's ``logit_metric``). Each hook point is its own variable: e.g.
    ``hook_resid_pre``'s gradient counts every path from it, ``hook_attn_in``'s only the path
    through attention. Shapes are those of the cached activations, without the batch dimension.
    """
    if tokens.dim() == 1:
        tokens = tokens[None]
    acts: Dict[str, Tensor] = {}

    def capture(act: Tensor, name: str) -> Tensor:
        act = act.clone()  # a separate autograd node, even where a hook point returns its input
        if act.requires_grad:
            act.retain_grad()
        else:  # e.g. sinusoidal positions, a constant buffer: make it a leaf that receives a gradient
            act.requires_grad_()
        acts[name] = act
        return act

    was_training = model.training
    model.eval()
    try:
        with torch.enable_grad(), model.hooks([(None, capture)]):
            logits = model(tokens).logits[0, pos].float()
            metric = logits[target] - logits[other] if other >= 0 else logits.log_softmax(-1)[target]
            metric.backward()
    finally:
        model.train(was_training)
        model.zero_grad(set_to_none=True)
    grads = {}
    for name, act in acts.items():
        g = act.grad if act.grad is not None else torch.zeros_like(act)
        grads[name] = (g if name == "hook_pos_embed" else g[0]).detach().float()
    return grads


def export_reference(model: DecoderLM, tokens: Tensor, path: Union[str, Path],
                     metric: Optional[Sequence[int]] = None) -> None:
    """Write tokens, logits and every hook activation for one sequence, for golden tests.

    With ``metric=(pos, target, other)`` also write each hook point's gradient of that metric
    (see ``hook_gradients``) as ``grad.<name>``.
    """
    if tokens.dim() == 1:
        tokens = tokens[None]
    if tokens.shape[0] != 1:
        raise ValueError("reference export takes a single sequence")
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            out, cache = model.run_with_cache(tokens)
    finally:
        model.train(was_training)
    tensors = {"tokens": tokens[0].long(), "logits": out.logits[0].float()}
    for name, act in cache.items():
        # hook_pos_embed is (T, d) with no batch dimension; everything else is batch-first.
        tensors[f"cache.{name}"] = (act if name == "hook_pos_embed" else act[0]).float()
    meta = {"format": "tforge-reference"}
    if metric is not None:
        pos, target, other = (int(v) for v in metric)
        for name, g in hook_gradients(model, tokens, pos, target, other).items():
            tensors[f"grad.{name}"] = g
        meta["metric"] = json.dumps([pos, target, other])
    save_safetensors(tensors, path, meta)
