"""Golden tests: the C++ forward pass (interp/) must reproduce PyTorch on every hook point.

Builds interp/ with make, exports models and reference activations, and runs tfi_golden.
Skipped when no C++ toolchain is available.
"""
import itertools
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from tforge import DecoderLM, ModelConfig, config_from_preset
from tforge.export import export_model, export_reference

INTERP = Path(__file__).resolve().parents[1] / "interp"
V = 97

pytestmark = pytest.mark.skipif(shutil.which("make") is None or shutil.which("c++") is None,
                                reason="needs make and a C++ compiler")


@pytest.fixture(scope="module")
def tools():
    subprocess.run(["make", "-s", "-C", str(INTERP), "all"], check=True)
    return INTERP / "build"


def tiny(**kw) -> ModelConfig:
    base = dict(vocab_size=V, d_model=32, n_layers=2, n_heads=4, max_seq_len=64)
    base.update(kw)
    return ModelConfig(**base)


GRID = [
    dict(norm=n, norm_position=p, positional=pos, mlp=m)
    for n, p, pos, m in itertools.product(
        ["rmsnorm", "layernorm"], ["pre", "post", "sandwich"],
        ["rope", "alibi", "learned", "sinusoidal", "none"], ["swiglu", "mlp", "none"])
]
EXTRA = [
    dict(n_kv_heads=1), dict(n_kv_heads=2), dict(qk_norm=True), dict(qk_norm=True, norm="layernorm", bias=True),
    dict(sliding_window=3), dict(sliding_window=3, positional="alibi"),
    dict(attn_logit_softcap=5.0, final_logit_softcap=10.0), dict(parallel_residual=True),
    dict(parallel_residual=True, mlp="none"), dict(rope_fraction=0.5), dict(d_head=16),
    dict(bias=True, tie_embeddings=False, embed_scale=True), dict(mlp="geglu", activation="gelu_tanh"),
    dict(mlp="glu", activation="relu2"), dict(mlp="mlp", activation="tanh"), dict(norm="none"),
    dict(attn_scale=0.3), dict(n_heads=6, d_model=36),
    dict(layer_overrides={0: {"mlp": "none"}, 1: {"positional": "none", "sliding_window": 4}}),
    dict(positional="learned", layer_overrides={1: {"positional": "rope"}}),
    dict(layer_overrides={0: {"n_heads": 2}}),
    dict(n_layers=3, layer_overrides={0: {"n_heads": 2}, 2: {"n_heads": 8, "d_head": 2}}),
    dict(positional="none", d_head=1, layer_overrides={1: {"positional": "rope", "d_head": 2}}),
    dict(norm="layernorm", layer_overrides={1: {"norm_position": "post"}}),
]
MIXED_HEADS = dict(n_layers=3, layer_overrides={0: {"n_heads": 2}, 2: {"n_heads": 8, "d_head": 2}})


def _id(c):
    return ",".join(f"{k}={v}" for k, v in c.items()) or "default"


def _golden(tools, tmp_path, model, tokens, tol="1e-4"):
    with torch.no_grad():  # sharpen attention so masking bugs show up
        for b in model.blocks:
            b.attn.q_proj.weight.mul_(10)
            b.attn.k_proj.weight.mul_(10)
    export_model(model, tmp_path / "model.safetensors")
    export_reference(model, tokens, tmp_path / "ref.safetensors")
    r = subprocess.run([str(tools / "tfi_golden"), str(tmp_path / "model.safetensors"), str(tmp_path / "ref.safetensors"), tol],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("opts", GRID + EXTRA, ids=_id)
def test_cpp_matches_pytorch(tools, tmp_path, opts):
    torch.manual_seed(0)
    _golden(tools, tmp_path, DecoderLM(tiny(**opts)), torch.randint(0, V, (12,)))


@pytest.mark.parametrize("name", ["llama", "gpt2", "vanilla", "gemma2", "attn_only_2l", "gelu_2l"])
def test_cpp_matches_pytorch_presets(tools, tmp_path, name):
    torch.manual_seed(0)
    model = DecoderLM(config_from_preset(name, vocab_size=V, max_seq_len=64, d_model=64, d_head=16, n_heads=4,
                                         n_kv_heads=None, n_layers=3))
    _golden(tools, tmp_path, model, torch.randint(0, V, (20,)))


def test_cpp_unit_tests(tools):
    r = subprocess.run([str(tools / "tfi_tests")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:]


def test_golden_rejects_nan(tools, tmp_path):
    torch.manual_seed(0)
    model = DecoderLM(tiny())
    export_reference(model, torch.randint(0, V, (12,)), tmp_path / "ref.safetensors")
    with torch.no_grad():
        model.blocks[-1].mlp.down.weight.fill_(float("nan"))
    export_model(model, tmp_path / "model.safetensors")
    r = subprocess.run([str(tools / "tfi_golden"), str(tmp_path / "model.safetensors"), str(tmp_path / "ref.safetensors")],
                       capture_output=True, text=True)
    assert r.returncode == 1, r.stdout
    assert "DIFF     logits" in r.stdout


@pytest.mark.parametrize("opts", [dict(), dict(norm="layernorm", positional="learned", mlp="mlp", bias=True),
                                  MIXED_HEADS], ids=_id)
def test_cli_commands(tools, tmp_path, opts):
    torch.manual_seed(0)
    path = tmp_path / "model.safetensors"
    export_model(DecoderLM(tiny(**opts)), path)
    toks = ",".join(str(t) for t in [5, 9, 13, 2, 7, 5, 9, 13, 2])
    corrupt = ",".join(str(t) for t in [5, 9, 40, 2, 7, 5, 9, 13, 2])
    commands = {
        "info": ([], "hook points"),
        "run": (["--tokens", toks], "mean next-token loss"),
        "lens": (["--repeat-random", "6"], "logit lens"),
        "dla": (["--tokens", toks, "--target", "7", "--other", "3"], "sum of parts"),
        "heads": (["--seg", "6", "--samples", "2"], "two-hop induction"),
        "patch": (["--clean", toks, "--corrupt", corrupt, "--target", "7", "--site", "heads"], "activation patching"),
        "ablate": (["--tokens", toks], "base loss"),
        "circuits": ([], "OV path copying"),
    }
    for cmd, (args, expected) in commands.items():
        r = subprocess.run([str(tools / "tfi"), cmd, str(path), *args], capture_output=True, text=True)
        assert r.returncode == 0, f"{cmd}: {r.stderr}"
        assert expected in r.stdout, f"{cmd}: {r.stdout}"
    bad = subprocess.run([str(tools / "tfi"), "dla", str(path), "--tokens", toks], capture_output=True, text=True)
    assert bad.returncode == 1 and "missing --target" in bad.stderr


def test_export_rejects_custom_components(tmp_path):
    from torch import nn
    from tforge import register

    @register("norm", "export_test_norm")
    class Custom(nn.LayerNorm):
        def __init__(self, dim, cfg):
            super().__init__(dim)

    with pytest.raises(ValueError, match="no C\\+\\+ implementation"):
        export_model(DecoderLM(tiny(norm="export_test_norm")), tmp_path / "m.safetensors")
