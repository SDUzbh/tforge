from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Callable, Dict, Iterable, Iterator, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import ModelConfig, config_from_preset
from .hooks import HookPoint
from .layers import Block, EmbeddingPositional
from .layers.attention import KV
from .registry import NORMS, POSITIONALS

# A hook function receives (activation, hook_name) and returns a replacement or None.
HookFn = Callable[[Tensor, str], Optional[Tensor]]
NameFilter = Union[None, str, Iterable[str], Callable[[str], bool]]


class LMOutput(NamedTuple):
    logits: Tensor                       # (B, T, vocab)
    loss: Optional[Tensor] = None        # mean next-token cross-entropy, when targets are given
    kv_cache: Optional[List[KV]] = None  # one (k, v) per layer, when use_cache=True


class DecoderLM(nn.Module):
    """A decoder-only transformer language model built from a ``ModelConfig``.

    Model-level hook points: hook_embed, hook_pos_embed (embedding-level positions
    only), hook_final (after the final norm). Each block adds its own; list them all
    with ``model.hook_names()``.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        pos = POSITIONALS.build(cfg.positional, cfg)
        self.pos_embed = pos if isinstance(pos, EmbeddingPositional) else None
        for i, ov in cfg.layer_overrides.items():
            # Only an explicit per-layer setting matters; inheriting an input-level encoding is fine.
            layer_pos = ov.get("positional")
            if layer_pos is not None and isinstance(POSITIONALS.build(layer_pos, cfg), EmbeddingPositional):
                raise ValueError(f"layer {i}: '{layer_pos}' is an input-level encoding and cannot be set per layer")
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg.layer_config(i), i) for i in range(cfg.n_layers))
        # Post-norm blocks already end in a norm.
        self.ln_final = NORMS.build(cfg.norm if cfg.norm_position != "post" else "none", cfg.d_model, cfg)
        self.unembed = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.unembed.weight = self.embed.weight

        self.hook_embed, self.hook_final = HookPoint(), HookPoint()
        if self.pos_embed is not None:
            self.hook_pos_embed = HookPoint()

        self.apply(self._init_weights)
        for name, mod in self.named_modules():
            if isinstance(mod, HookPoint):
                mod.name = name

    @classmethod
    def from_preset(cls, name: str, **overrides) -> "DecoderLM":
        return cls(config_from_preset(name, **overrides))

    def _init_weights(self, module: nn.Module) -> None:
        std = self.cfg.init_std
        if isinstance(module, nn.Linear):
            if getattr(module, "is_residual_out", False) and self.cfg.scale_residual_init:
                std = std / math.sqrt(2 * self.cfg.n_layers)
            nn.init.normal_(module.weight, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=std)

    # --- forward ------------------------------------------------------------
    def forward(
        self,
        tokens: Tensor,
        targets: Optional[Tensor] = None,
        kv_cache: Optional[List[KV]] = None,
        use_cache: bool = False,
    ) -> LMOutput:
        B, T = tokens.shape
        past = kv_cache[0][0].shape[2] if kv_cache else 0
        if self.pos_embed is not None and past + T > self.cfg.max_seq_len:
            raise ValueError(f"sequence length {past + T} exceeds max_seq_len={self.cfg.max_seq_len}")
        positions = torch.arange(past, past + T, device=tokens.device)

        x = self.embed(tokens)
        if self.cfg.embed_scale:
            x = x * math.sqrt(self.cfg.d_model)
        x = self.hook_embed(x)
        if self.pos_embed is not None:
            x = x + self.hook_pos_embed(self.pos_embed(positions).to(x.dtype))
        x = self.drop(x)

        new_cache = []
        for i, block in enumerate(self.blocks):
            x, kv = block(x, positions, kv_cache[i] if kv_cache else None)
            new_cache.append(kv)

        logits = self.unembed(self.hook_final(self.ln_final(x)))
        if self.cfg.final_logit_softcap is not None:
            cap = self.cfg.final_logit_softcap
            logits = cap * torch.tanh(logits / cap)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), ignore_index=-100)
        return LMOutput(logits, loss, new_cache if use_cache else None)

    # --- generation -----------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        tokens: Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> Tensor:
        """Sample continuations of ``tokens`` (B, T). ``temperature=0`` is greedy.

        Runs in eval mode and restores the previous mode afterwards. Once the
        sequence reaches ``max_seq_len`` the cache is dropped and the model is
        re-run on the most recent ``max_seq_len`` tokens.
        """
        was_training = self.training
        self.eval()
        try:
            out, cache = tokens, None
            done = torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
            for _ in range(max_new_tokens):
                if cache is not None and cache[0][0].shape[2] >= self.cfg.max_seq_len:
                    cache = None
                if cache is None:
                    step_in = out[:, -self.cfg.max_seq_len:]
                else:
                    step_in = out[:, -1:]
                res = self(step_in, kv_cache=cache, use_cache=use_cache)
                cache = res.kv_cache
                nxt = sample_next(res.logits[:, -1], temperature, top_k, top_p)
                if eos_token_id is not None:
                    nxt = torch.where(done, torch.full_like(nxt, eos_token_id), nxt)
                    done = done | (nxt == eos_token_id)
                out = torch.cat([out, nxt[:, None]], dim=1)
                if eos_token_id is not None and bool(done.all()):
                    break
            return out
        finally:
            self.train(was_training)

    # --- hooks ------------------------------------------------------------------
    def hook_points(self) -> Dict[str, HookPoint]:
        return {name: m for name, m in self.named_modules() if isinstance(m, HookPoint)}

    def hook_names(self) -> List[str]:
        return list(self.hook_points())

    def _select(self, names: NameFilter) -> List[str]:
        points = self.hook_points()
        if names is None:
            return list(points)
        if callable(names):
            return [n for n in points if names(n)]
        wanted = [names] if isinstance(names, str) else list(names)
        missing = [n for n in wanted if n not in points]
        if missing:
            raise KeyError(f"no hook points named {missing}; see model.hook_names()")
        return wanted

    @contextmanager
    def hooks(self, fwd_hooks: Union[Dict[str, HookFn], Iterable[Tuple[NameFilter, HookFn]]]) -> Iterator["DecoderLM"]:
        """Temporarily attach hooks. Each fn(activation, name) may return a replacement.

            def zero_ablate(act, name):
                return torch.zeros_like(act)

            with model.hooks({"blocks.0.hook_mlp_out": zero_ablate}):
                logits = model(tokens).logits

        Keys may also be a list of names or a predicate, e.g. ``lambda n: n.endswith("hook_z")``.
        """
        items = fwd_hooks.items() if isinstance(fwd_hooks, dict) else fwd_hooks
        points = self.hook_points()
        handles = []
        try:
            for names, fn in items:
                for name in self._select(names):
                    handles.append(points[name].register_forward_hook(
                        lambda mod, inp, out, fn=fn, name=name: fn(out, name)))
            yield self
        finally:
            for h in handles:
                h.remove()

    def run_with_cache(self, tokens: Tensor, names: NameFilter = None, **forward_kwargs) -> Tuple[LMOutput, Dict[str, Tensor]]:
        """Run the model and return (output, {hook_name: activation}).

        ``names`` picks which hook points to store (default: all). Caching
        attention patterns uses the slower non-fused attention path.
        """
        cache: Dict[str, Tensor] = {}

        def store(act: Tensor, name: str) -> None:
            cache[name] = act.detach()

        with self.hooks([(names, store)]):
            out = self(tokens, **forward_kwargs)
        return out, cache

    # --- utilities --------------------------------------------------------------
    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        params = {id(p): p for p in self.parameters()}
        if exclude_embeddings:
            params.pop(id(self.embed.weight), None)
            params.pop(id(self.unembed.weight), None)
            if self.pos_embed is not None:
                for p in self.pos_embed.parameters():
                    params.pop(id(p), None)
        return sum(p.numel() for p in params.values())


def sample_next(logits: Tensor, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None) -> Tensor:
    """Pick one token per row from (B, vocab) logits."""
    if temperature == 0:
        return logits.argmax(dim=-1)
    logits = logits.float() / temperature
    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None:
        sorted_logits, idx = logits.sort(dim=-1, descending=True)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        drop = cum - sorted_logits.softmax(dim=-1) > top_p  # keep the token that crosses top_p
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, idx, sorted_logits)
    return torch.multinomial(logits.softmax(dim=-1), num_samples=1).squeeze(-1)
