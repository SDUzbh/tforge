"""indranet: IndraNet's Python side. Build and train decoder-only transformers from a config,
with the named hook points that the C++ engine (tfi) reproduces, and export them for it.

    from indranet import DecoderLM, ModelConfig

    model = DecoderLM(ModelConfig())                                  # defaults (Llama-style)
    model = DecoderLM.from_preset("gpt2", n_layers=6)                # preset + overrides
    model = DecoderLM(ModelConfig(norm="layernorm", positional="alibi",
                                  mlp="mlp", activation="relu2", n_kv_heads=1))
"""
from . import layers  # noqa: F401  (registers built-in components)
from .config import PRESETS, ModelConfig, config_from_preset
from .hooks import HookPoint
from .model import DecoderLM, LMOutput, sample_next
from .registry import available, get_registry, register

__all__ = [
    "DecoderLM", "LMOutput", "ModelConfig", "PRESETS", "config_from_preset",
    "HookPoint", "available", "get_registry", "register", "sample_next",
]
