"""Name -> component registries.

Every swappable part of the model (norms, activations, MLPs, attention,
positional encodings) is looked up by name in one of these registries, so a
config string like ``norm="rmsnorm"`` picks the implementation. Register your
own component to use it from a config:

    from indranet import register

    @register("norm", "scalenorm")
    class ScaleNorm(nn.Module):
        def __init__(self, dim, cfg): ...
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Any] = {}

    def register(self, name: str, obj: Optional[Any] = None, overwrite: bool = False):
        """Register ``obj`` under ``name``; usable as a decorator when ``obj`` is omitted."""

        def _add(o):
            if name in self._items and not overwrite:
                raise KeyError(f"{self.kind} '{name}' is already registered (pass overwrite=True to replace it)")
            self._items[name] = o
            return o

        return _add if obj is None else _add(obj)

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} '{name}'; available: {', '.join(self.names())}") from None

    def build(self, name: str, *args, **kwargs) -> Any:
        return self.get(name)(*args, **kwargs)

    def names(self) -> List[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


NORMS = Registry("norm")                # factory(dim, cfg) -> nn.Module
ACTIVATIONS = Registry("activation")    # factory() -> nn.Module
MLPS = Registry("mlp")                  # factory(cfg) -> nn.Module or None
ATTENTIONS = Registry("attention")      # factory(cfg, layer_idx) -> nn.Module
POSITIONALS = Registry("positional")    # factory(cfg) -> EmbeddingPositional | AttentionPositional

_REGISTRIES = {r.kind: r for r in (NORMS, ACTIVATIONS, MLPS, ATTENTIONS, POSITIONALS)}


def get_registry(kind: str) -> Registry:
    try:
        return _REGISTRIES[kind]
    except KeyError:
        raise KeyError(f"unknown registry '{kind}'; available: {', '.join(sorted(_REGISTRIES))}") from None


def register(kind: str, name: str, overwrite: bool = False) -> Callable:
    """Decorator: ``@register("mlp", "my_mlp")``."""
    return get_registry(kind).register(name, overwrite=overwrite)


def available() -> Dict[str, List[str]]:
    """All registered component names, by kind."""
    return {kind: r.names() for kind, r in _REGISTRIES.items()}
