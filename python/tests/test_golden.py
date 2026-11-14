"""Golden tests: the C++ forward pass (the tfi engine) must reproduce PyTorch on every hook point, and its
backward pass must reproduce PyTorch autograd's gradient at every hook point.

Builds the C++ engine (repository root) with make, exports models and reference activations, and runs
tfi_golden.
Skipped when no C++ toolchain is available.
"""
import itertools
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from indranet import DecoderLM, ModelConfig, config_from_preset
from indranet.export import export_model, export_reference

ROOT = Path(__file__).resolve().parents[2]
V = 97

pytestmark = pytest.mark.skipif(shutil.which("make") is None or shutil.which("c++") is None,
                                reason="needs make and a C++ compiler")


@pytest.fixture(scope="module")
def tools():
    subprocess.run(["make", "-s", "-C", str(ROOT), "all"], check=True)
    return ROOT / "build"


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


def _golden(tools, tmp_path, model, tokens, metric, tol="1e-4"):
    with torch.no_grad():  # sharpen attention so masking bugs show up
        for b in model.blocks:
            b.attn.q_proj.weight.mul_(10)
            b.attn.k_proj.weight.mul_(10)
    export_model(model, tmp_path / "model.safetensors")
    export_reference(model, tokens, tmp_path / "ref.safetensors", metric=metric)
    r = subprocess.run([str(tools / "tfi_golden"), str(tmp_path / "model.safetensors"), str(tmp_path / "ref.safetensors"), tol],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize("opts", GRID + EXTRA, ids=_id)
def test_cpp_matches_pytorch(tools, tmp_path, opts):
    torch.manual_seed(0)
    # Gradients of a logit difference at an inner position (later positions get zero gradient).
    _golden(tools, tmp_path, DecoderLM(tiny(**opts)), torch.randint(0, V, (12,)), metric=(10, 3, 5))


@pytest.mark.parametrize("name", ["llama", "gpt2", "vanilla", "gemma2", "attn_only_2l", "gelu_2l"])
def test_cpp_matches_pytorch_presets(tools, tmp_path, name):
    torch.manual_seed(0)
    model = DecoderLM(config_from_preset(name, vocab_size=V, max_seq_len=64, d_model=64, d_head=16, n_heads=4,
                                         n_kv_heads=None, n_layers=3))
    _golden(tools, tmp_path, model, torch.randint(0, V, (20,)), metric=(19, 3, -1))  # log-probability metric


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
        "ablate": (["--tokens", toks, "--reference-random", "3"], "means over 3 reference sequences, one per position"),
        "circuits": ([], "OV path copying"),
    }
    for cmd, (args, expected) in commands.items():
        r = subprocess.run([str(tools / "indra"), cmd, str(path), *args], capture_output=True, text=True)
        assert r.returncode == 0, f"{cmd}: {r.stderr}"
        assert expected in r.stdout, f"{cmd}: {r.stdout}"
    bad = subprocess.run([str(tools / "indra"), "dla", str(path), "--tokens", toks], capture_output=True, text=True)
    assert bad.returncode == 1 and "missing --target" in bad.stderr
    zero = subprocess.run([str(tools / "indra"), "ablate", str(path), "--tokens", toks, "--mode", "zero"],
                          capture_output=True, text=True)
    assert zero.returncode == 0 and "zero-ablated" in zero.stdout
    # Mean ablation must not fall back to averaging over the input itself (that leaks later tokens).
    leak = subprocess.run([str(tools / "indra"), "ablate", str(path), "--tokens", toks], capture_output=True, text=True)
    assert leak.returncode == 1 and "reference sequences" in leak.stderr
    # Patching with no gap (identical inputs) explains itself and shows finite raw effects, not NaN.
    same = subprocess.run([str(tools / "indra"), "patch", str(path), "--clean", toks, "--corrupt", toks, "--target", "7"],
                          capture_output=True, text=True)
    assert same.returncode == 0 and "doesn't change this metric" in same.stdout
    table = same.stdout.split("\n\n", 1)[1]
    assert "0.000" in table and " . " not in table and "nan" not in table.lower()
    raw = subprocess.run([str(tools / "indra"), "patch", str(path), "--clean", toks, "--corrupt", corrupt,
                          "--target", "7", "--site", "heads", "--raw"], capture_output=True, text=True)
    assert raw.returncode == 0 and "raw effect of each patch" in raw.stdout
    for site in ["resid_pre", "heads"]:
        atp = subprocess.run([str(tools / "indra"), "patch", str(path), "--clean", toks, "--corrupt", corrupt, "--target", "7",
                              "--site", site, "--method", "attribution", "--steps", "3", "--compare"],
                             capture_output=True, text=True)
        assert atp.returncode == 0, atp.stderr
        assert "attribution patching" in atp.stdout and "agreement over" in atp.stdout
    mixed = subprocess.run([str(tools / "indra"), "patch", str(path), "--clean", toks, "--corrupt", corrupt, "--target", "7",
                            "--site", "heads", "--method", "mixed", "--verify", "3", "--compare"], capture_output=True, text=True)
    assert mixed.returncode == 0, mixed.stderr
    assert "mixed patching" in mixed.stdout and "the 3 largest estimates" in mixed.stdout
    assert mixed.stdout.split("exact patching for comparison")[0].count("*") == 4  # 3 marked entries + the legend
    # Swapping clean and corrupted flips the gap's sign; the summary's sizes must not depend on it.
    unverified = {}
    for order, (c, k) in {"as given": (toks, corrupt), "swapped": (corrupt, toks)}.items():
        out = subprocess.run([str(tools / "indra"), "patch", str(path), "--clean", c, "--corrupt", k, "--target", "7",
                              "--site", "heads", "--method", "mixed", "--verify", "3"], capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        gap = float(out.stdout.split("gap ")[1].split()[0].strip("),"))
        unverified[order] = (gap, float(out.stdout.split("largest estimate left unverified: ")[1].split()[0]))
    assert unverified["as given"][0] * unverified["swapped"][0] < 0
    assert all(v > 0 for _, v in unverified.values()), unverified
    # --no-gpu and TFI_NO_GPU keep patching on the CPU; this model is too small to use the GPU anyway.
    patch_args = [str(tools / "indra"), "patch", str(path), "--clean", toks, "--corrupt", corrupt, "--target", "7"]
    default = subprocess.run(patch_args, capture_output=True, text=True)
    no_gpu = subprocess.run(patch_args + ["--no-gpu"], capture_output=True, text=True)
    env_no_gpu = subprocess.run(patch_args, capture_output=True, text=True, env={**os.environ, "TFI_NO_GPU": "1"})
    assert default.returncode == no_gpu.returncode == env_no_gpu.returncode == 0, no_gpu.stderr + env_no_gpu.stderr
    assert default.stdout == no_gpu.stdout == env_no_gpu.stdout
    help_text = subprocess.run([str(tools / "indra"), "--help"], capture_output=True, text=True).stdout
    assert "--no-gpu" in help_text and "--fp16" in help_text
    fp16 = subprocess.run(patch_args + ["--fp16"], capture_output=True, text=True)
    assert fp16.returncode == 0 and "--fp16" in fp16.stdout, fp16.stderr
    # --fp16 stores the weights in fp16 even without the GPU; --fp32 is the default.
    fp16_cpu = subprocess.run(patch_args + ["--fp16", "--no-gpu"], capture_output=True, text=True)
    assert fp16_cpu.returncode == 0 and "weights are stored in fp16" in fp16_cpu.stdout, fp16_cpu.stderr
    assert fp16_cpu.stdout.split("\n\n")[-1] != default.stdout.split("\n\n")[-1]  # really rounded
    fp32 = subprocess.run(patch_args + ["--fp32"], capture_output=True, text=True)
    assert fp32.returncode == 0 and fp32.stdout == default.stdout
    both = subprocess.run(patch_args + ["--fp16", "--fp32"], capture_output=True, text=True)
    assert both.returncode == 1 and "not both" in both.stderr
    info16 = subprocess.run([str(tools / "indra"), "info", str(path), "--fp16"], capture_output=True, text=True).stdout
    info32 = subprocess.run([str(tools / "indra"), "info", str(path)], capture_output=True, text=True).stdout
    assert "weights       fp16" in info16 and "weights       float32" in info32
    run_args = [str(tools / "indra"), "run", str(path), "--tokens", toks]
    run16 = subprocess.run(run_args + ["--fp16"], capture_output=True, text=True)
    run32 = subprocess.run(run_args, capture_output=True, text=True)
    assert run16.returncode == 0 and "weights are stored in fp16" in run16.stdout
    loss = lambda out: float(out.split("mean next-token loss")[1].split()[0].strip(":"))  # noqa: E731
    assert loss(run16.stdout) == pytest.approx(loss(run32.stdout), rel=1e-2)
    ablate_args = [str(tools / "indra"), "ablate", str(path), "--tokens", toks, "--mode", "zero"]
    ablate = subprocess.run(ablate_args, capture_output=True, text=True)
    ablate_cpu = subprocess.run(ablate_args + ["--no-gpu"], capture_output=True, text=True)
    assert ablate.returncode == ablate_cpu.returncode == 0 and ablate.stdout == ablate_cpu.stdout, ablate_cpu.stderr
    ablate_fp16 = subprocess.run(ablate_args + ["--fp16", "--no-gpu"], capture_output=True, text=True)
    assert ablate_fp16.returncode == 0 and "weights are stored in fp16" in ablate_fp16.stdout
    bad_method = subprocess.run([str(tools / "indra"), "patch", str(path), "--clean", toks, "--corrupt", corrupt,
                                 "--target", "7", "--method", "fast"], capture_output=True, text=True)
    assert bad_method.returncode == 1 and "--method must be" in bad_method.stderr


def test_export_rejects_custom_components(tmp_path):
    from torch import nn
    from indranet import register

    @register("norm", "export_test_norm")
    class Custom(nn.LayerNorm):
        def __init__(self, dim, cfg):
            super().__init__(dim)

    with pytest.raises(ValueError, match="no C\\+\\+ implementation"):
        export_model(DecoderLM(tiny(norm="export_test_norm")), tmp_path / "m.safetensors")
