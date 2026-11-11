"""Model configuration: one dataclass with sensible defaults, plus presets.

The defaults describe a small modern Llama-style decoder (pre-RMSNorm, RoPE,
SwiGLU, no biases, tied embeddings). Every field can be overridden, and
``layer_overrides`` changes settings for individual layers, e.g.
``{0: {"mlp": "none"}}`` makes layer 0 attention-only.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, Union

NORM_POSITIONS = ("pre", "post", "sandwich")

# Fields that describe the whole model and cannot differ between layers.
GLOBAL_FIELDS = frozenset({
    "vocab_size", "d_model", "n_layers", "max_seq_len", "tie_embeddings",
    "embed_scale", "final_logit_softcap", "init_std", "scale_residual_init",
    "layer_overrides",
})


@dataclass
class ModelConfig:
    # --- sizes -------------------------------------------------------------
    vocab_size: int = 32000
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    n_kv_heads: Optional[int] = None      # None = n_heads (MHA); 1 = MQA; otherwise GQA
    d_head: Optional[int] = None          # None = d_model // n_heads
    d_mlp: Optional[int] = None           # None = 4*d_model, or ~8/3*d_model for gated MLPs
    mlp_multiple_of: int = 64             # round the default gated d_mlp up to this
    max_seq_len: int = 2048

    # --- components (names looked up in tforge.registry) --------------------
    norm: str = "rmsnorm"                 # rmsnorm | layernorm | none
    norm_eps: float = 1e-5
    norm_position: str = "pre"            # pre | post | sandwich
    positional: str = "rope"              # rope | alibi | learned | sinusoidal | none
    rope_theta: float = 10000.0
    rope_fraction: float = 1.0            # fraction of each head's dims that get rotated
    attention: str = "causal"
    mlp: str = "swiglu"                   # mlp | glu | swiglu | geglu | none
    activation: str = "gelu"              # used by "mlp" and "glu"

    # --- attention options --------------------------------------------------
    qk_norm: bool = False                 # normalise q and k per head before the dot product
    sliding_window: Optional[int] = None  # attend only to the last N positions
    attn_scale: Optional[float] = None    # None = 1/sqrt(d_head)
    attn_logit_softcap: Optional[float] = None
    use_sdpa: bool = True                 # fused kernel when nothing needs the attention pattern

    # --- block / model options ----------------------------------------------
    parallel_residual: bool = False       # x + attn(ln1(x)) + mlp(ln2(x)), GPT-J/NeoX style
    bias: bool = False                    # biases in linear layers and LayerNorm
    tie_embeddings: bool = True
    embed_scale: bool = False             # multiply embeddings by sqrt(d_model), Gemma style
    final_logit_softcap: Optional[float] = None
    dropout: float = 0.0
    attn_dropout: float = 0.0

    # --- init ---------------------------------------------------------------
    init_std: float = 0.02
    scale_residual_init: bool = True      # scale output projections by 1/sqrt(2*n_layers)

    # --- per-layer settings ---------------------------------------------------
    layer_overrides: Dict[int, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        # JSON turns int keys into strings; accept both.
        self.layer_overrides = {int(k): dict(v) for k, v in self.layer_overrides.items()}
        for i, ov in self.layer_overrides.items():
            if not 0 <= i < self.n_layers:
                raise ValueError(f"layer_overrides has layer {i}, but n_layers={self.n_layers}")
            bad = set(ov) & GLOBAL_FIELDS
            unknown = set(ov) - {f.name for f in fields(self)}
            if unknown:
                raise ValueError(f"layer_overrides[{i}] has unknown fields: {sorted(unknown)}")
            if bad:
                raise ValueError(f"layer_overrides[{i}] sets model-wide fields: {sorted(bad)}")
        self._validate()
        for i in self.layer_overrides:
            self.layer_config(i)  # validates each overridden layer

    def _validate(self):
        if self.d_head is None and self.d_model % self.n_heads:
            raise ValueError(f"d_model={self.d_model} is not divisible by n_heads={self.n_heads}; set d_head explicitly")
        if self.n_heads % self.kv_heads:
            raise ValueError(f"n_heads={self.n_heads} must be a multiple of n_kv_heads={self.kv_heads}")
        if self.norm_position not in NORM_POSITIONS:
            raise ValueError(f"norm_position must be one of {NORM_POSITIONS}, got '{self.norm_position}'")
        if self.parallel_residual and self.norm_position != "pre":
            raise ValueError("parallel_residual requires norm_position='pre'")
        if not 0.0 < self.rope_fraction <= 1.0:
            raise ValueError("rope_fraction must be in (0, 1]")
        if self.sliding_window is not None and self.sliding_window < 1:
            raise ValueError("sliding_window must be >= 1")

    # --- derived sizes --------------------------------------------------------
    @property
    def head_dim(self) -> int:
        return self.d_head if self.d_head is not None else self.d_model // self.n_heads

    @property
    def kv_heads(self) -> int:
        return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

    def layer_config(self, i: int) -> "ModelConfig":
        """The config layer ``i`` is built from (this config plus its overrides)."""
        ov = self.layer_overrides.get(i)
        if not ov:
            return self
        return replace(self, layer_overrides={}, **ov)

    # --- serialisation --------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        unknown = set(d) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"unknown config fields: {sorted(unknown)}")
        return cls(**d)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def replace(self, **changes) -> "ModelConfig":
        return replace(self, **changes)


# --- presets -------------------------------------------------------------------
# Architecture families at small, trainable sizes. Override anything:
#   config_from_preset("llama", d_model=1024, n_layers=16)

PRESETS: Dict[str, Dict[str, Any]] = {
    # Modern default: pre-RMSNorm, RoPE, SwiGLU, GQA, no biases.
    "llama": dict(
        d_model=512, n_layers=8, n_heads=8, n_kv_heads=2,
        norm="rmsnorm", positional="rope", mlp="swiglu", bias=False,
    ),
    # GPT-2 small: pre-LayerNorm, learned positions, GELU MLP, biases, 50257 vocab.
    "gpt2": dict(
        vocab_size=50257, d_model=768, n_layers=12, n_heads=12, max_seq_len=1024,
        norm="layernorm", positional="learned", mlp="mlp", activation="gelu_tanh",
        bias=True, dropout=0.1, attn_dropout=0.1,
    ),
    # Original Transformer decoder: post-LayerNorm, sinusoidal positions, ReLU MLP.
    "vanilla": dict(
        d_model=512, n_layers=6, n_heads=8, norm="layernorm", norm_position="post",
        positional="sinusoidal", mlp="mlp", activation="relu", bias=True,
        tie_embeddings=True, dropout=0.1,
    ),
    # Gemma-2-like: sandwich norm, GeGLU, embedding scaling, softcaps,
    # alternating sliding-window (even layers) and global attention.
    "gemma2": dict(
        d_model=512, n_layers=8, n_heads=8, n_kv_heads=4, d_head=64,
        norm="rmsnorm", norm_position="sandwich", positional="rope", mlp="geglu",
        activation="gelu_tanh", embed_scale=True, attn_logit_softcap=50.0,
        final_logit_softcap=30.0,
        layer_overrides=lambda n_layers: {i: {"sliding_window": 256} for i in range(0, n_layers, 2)},
    ),
    # Small interpretability models, like TransformerLens' attn-only-2l / gelu-2l.
    "attn_only_2l": dict(
        vocab_size=4096, d_model=256, n_layers=2, n_heads=8, max_seq_len=512,
        norm="layernorm", positional="learned", mlp="none", bias=True,
    ),
    "gelu_2l": dict(
        vocab_size=4096, d_model=256, n_layers=2, n_heads=8, max_seq_len=512,
        norm="layernorm", positional="learned", mlp="mlp", activation="gelu", bias=True,
    ),
    # Tiny model for tests and quick experiments.
    "tiny": dict(
        vocab_size=256, d_model=64, n_layers=2, n_heads=4, max_seq_len=128,
    ),
}


def config_from_preset(name: str, **overrides) -> ModelConfig:
    try:
        base = PRESETS[name]
    except KeyError:
        raise KeyError(f"unknown preset '{name}'; available: {', '.join(sorted(PRESETS))}") from None
    merged = {**base, **overrides}
    # A preset may give layer_overrides as a function of the final layer count.
    if callable(merged.get("layer_overrides")):
        merged["layer_overrides"] = merged["layer_overrides"](merged.get("n_layers", ModelConfig.n_layers))
    return ModelConfig(**merged)
