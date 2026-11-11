import itertools

import pytest
import torch
from torch import nn

import tforge.layers.attention as attention_module
from tforge import PRESETS, DecoderLM, ModelConfig, config_from_preset, register
from tforge.layers import ALiBiPositional, NoPositional, RotaryPositional

V = 97


def tiny(**kw) -> ModelConfig:
    base = dict(vocab_size=V, d_model=32, n_layers=2, n_heads=4, max_seq_len=64)
    base.update(kw)
    return ModelConfig(**base)


def tokens(b=2, t=12, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (b, t), generator=g)


# A spread of option combinations; each is checked for shape, causality and KV-cache agreement.
GRID = [
    dict(norm=n, norm_position=p, positional=pos, mlp=m)
    for n, p, pos, m in itertools.product(
        ["rmsnorm", "layernorm"], ["pre", "post", "sandwich"],
        ["rope", "alibi", "learned", "sinusoidal", "none"], ["swiglu", "mlp", "none"])
]
EXTRA = [
    dict(n_kv_heads=1),                                  # MQA
    dict(n_kv_heads=2),                                  # GQA
    dict(qk_norm=True),
    dict(sliding_window=3),
    dict(sliding_window=3, positional="alibi"),
    dict(attn_logit_softcap=5.0, final_logit_softcap=10.0),
    dict(parallel_residual=True),
    dict(parallel_residual=True, mlp="none"),
    dict(rope_fraction=0.5),
    dict(d_head=16),                                     # d_head * n_heads != d_model
    dict(bias=True, tie_embeddings=False, embed_scale=True),
    dict(mlp="geglu", activation="gelu_tanh"),
    dict(mlp="glu", activation="relu2"),
    dict(norm="none"),
    dict(use_sdpa=False),
    dict(layer_overrides={0: {"mlp": "none"}, 1: {"positional": "none", "sliding_window": 4}}),
]
CASES = GRID + EXTRA


def _id(c):
    return ",".join(f"{k}={v}" for k, v in c.items()) or "default"


@pytest.mark.parametrize("opts", CASES, ids=_id)
def test_forward_shapes_and_loss(opts):
    model = DecoderLM(tiny(**opts))
    x = tokens()
    out = model(x, targets=x)
    assert out.logits.shape == (2, 12, V)
    assert torch.isfinite(out.loss)
    out.loss.backward()


@pytest.mark.parametrize("opts", CASES, ids=_id)
def test_causal(opts):
    model = DecoderLM(tiny(**opts)).eval()
    x = tokens()
    y = x.clone()
    y[:, 7:] = (y[:, 7:] + 1) % V
    a, b = model(x).logits, model(y).logits
    torch.testing.assert_close(a[:, :7], b[:, :7])
    assert not torch.allclose(a[:, 7:], b[:, 7:])


@pytest.mark.parametrize("opts", CASES, ids=_id)
def test_kv_cache_matches_full_forward(opts):
    model = DecoderLM(tiny(**opts)).eval()
    x = tokens(t=10)
    full = model(x).logits
    out = model(x[:, :6], use_cache=True)
    steps = [out.logits]
    cache = out.kv_cache
    for t in range(6, 10):
        out = model(x[:, t:t + 1], kv_cache=cache, use_cache=True)
        cache = out.kv_cache
        steps.append(out.logits)
    torch.testing.assert_close(torch.cat(steps, dim=1), full, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("opts", [dict(), dict(positional="alibi"), dict(sliding_window=3), dict(n_kv_heads=1)], ids=_id)
def test_sdpa_matches_manual_attention(opts):
    a = DecoderLM(tiny(use_sdpa=True, **opts)).eval()
    with torch.no_grad():  # sharpen attention so a masking or alignment bug can't hide in near-uniform patterns
        for blk in a.blocks:
            blk.attn.q_proj.weight.mul_(30)
            blk.attn.k_proj.weight.mul_(30)
    b = DecoderLM(tiny(use_sdpa=False, **opts)).eval()
    b.load_state_dict(a.state_dict())
    x = tokens()
    _, cache = b.run_with_cache(x, names=["blocks.0.attn.hook_pattern"])
    assert cache["blocks.0.attn.hook_pattern"].max(-1).values.mean() > 0.5  # really sharp
    torch.testing.assert_close(a(x).logits, b(x).logits, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_presets(name):
    model = DecoderLM.from_preset(name, vocab_size=V, max_seq_len=64, d_model=64, d_head=16, n_heads=4,
                                  n_kv_heads=None, d_mlp=None)
    x = tokens()
    assert model(x, targets=x).logits.shape == (2, 12, V)


def test_gemma2_preset_alternates_sliding_window():
    model = DecoderLM.from_preset("gemma2")
    windows = [b.attn.window for b in model.blocks]
    assert windows == [256, None] * 4


def test_attention_only_layers_have_no_mlp():
    model = DecoderLM.from_preset("attn_only_2l")
    assert all(b.mlp is None for b in model.blocks)
    assert not any("mlp" in n for n in model.hook_names())


def test_weight_tying():
    tied = DecoderLM(tiny())
    assert tied.unembed.weight is tied.embed.weight
    untied = DecoderLM(tiny(tie_embeddings=False))
    assert untied.unembed.weight is not untied.embed.weight
    assert untied.num_parameters() == tied.num_parameters() + V * 32


def test_gqa_has_fewer_kv_params():
    mha, gqa = DecoderLM(tiny()), DecoderLM(tiny(n_kv_heads=1))
    assert gqa.blocks[0].attn.k_proj.out_features == 8
    assert gqa.num_parameters() < mha.num_parameters()


# --- config -------------------------------------------------------------------

def test_config_json_roundtrip(tmp_path):
    cfg = config_from_preset("gemma2")
    cfg.save(tmp_path / "cfg.json")
    loaded = ModelConfig.load(tmp_path / "cfg.json")
    assert loaded == cfg  # layer_overrides keys come back as ints


@pytest.mark.parametrize("bad, msg", [
    (dict(d_model=30, n_heads=4), "divisible"),
    (dict(n_kv_heads=3), "multiple"),
    (dict(norm_position="middle"), "norm_position"),
    (dict(parallel_residual=True, norm_position="post"), "parallel_residual"),
    (dict(layer_overrides={5: {"mlp": "none"}}), "n_layers"),
    (dict(layer_overrides={0: {"d_model": 64}}), "model-wide"),
    (dict(layer_overrides={0: {"nope": 1}}), "unknown"),
])
def test_config_validation(bad, msg):
    with pytest.raises(ValueError, match=msg):
        tiny(**bad)


def test_unknown_component_lists_available():
    with pytest.raises(KeyError, match="available: .*rmsnorm"):
        DecoderLM(tiny(norm="batchnorm"))


def test_embedding_positional_cannot_be_set_per_layer():
    with pytest.raises(ValueError, match="input-level"):
        DecoderLM(tiny(layer_overrides={1: {"positional": "learned"}}))


def test_learned_positions_enforce_max_seq_len():
    model = DecoderLM(tiny(positional="learned", max_seq_len=8))
    with pytest.raises(ValueError, match="max_seq_len"):
        model(tokens(t=9))


# --- registry -----------------------------------------------------------------

def test_custom_component_registration():
    @register("norm", "scalenorm_test")
    class ScaleNorm(nn.Module):
        def __init__(self, dim, cfg):
            super().__init__()
            self.g = nn.Parameter(torch.tensor(dim ** 0.5))

        def forward(self, x):
            return self.g * x / x.norm(dim=-1, keepdim=True).clamp(min=1e-5)

    @register("mlp", "double_test")
    def double(cfg):
        return nn.Sequential(nn.Linear(cfg.d_model, cfg.d_model), nn.ReLU(), nn.Linear(cfg.d_model, cfg.d_model))

    model = DecoderLM(tiny(norm="scalenorm_test", mlp="double_test"))
    assert isinstance(model.blocks[0].ln1, ScaleNorm)
    x = tokens()
    assert torch.isfinite(model(x, targets=x).loss)
    with pytest.raises(KeyError, match="already registered"):
        register("norm", "scalenorm_test")(ScaleNorm)


# --- hooks ----------------------------------------------------------------------

def test_run_with_cache_shapes():
    model = DecoderLM(tiny(n_kv_heads=2))
    out, cache = model.run_with_cache(tokens())
    assert torch.allclose(out.logits, model(tokens()).logits, atol=1e-6)
    assert cache["blocks.0.attn.hook_pattern"].shape == (2, 4, 12, 12)
    assert cache["blocks.0.attn.hook_z"].shape == (2, 12, 4, 8)
    assert cache["blocks.0.attn.hook_k"].shape == (2, 2, 12, 8)
    assert cache["blocks.1.hook_mlp_in"].shape == (2, 12, 32)
    assert cache["blocks.1.mlp.hook_post"].shape[-1] == model.blocks[1].mlp.down.in_features
    torch.testing.assert_close(cache["blocks.0.attn.hook_pattern"].sum(-1), torch.ones(2, 4, 12))


def test_residual_stream_decomposes():
    model = DecoderLM(tiny())
    _, c = model.run_with_cache(tokens())
    torch.testing.assert_close(c["blocks.0.hook_resid_mid"], c["blocks.0.hook_resid_pre"] + c["blocks.0.hook_attn_out"])
    torch.testing.assert_close(c["blocks.0.hook_resid_post"], c["blocks.0.hook_resid_mid"] + c["blocks.0.hook_mlp_out"])


def test_run_with_cache_filter():
    model = DecoderLM(tiny())
    _, cache = model.run_with_cache(tokens(), names=lambda n: n.endswith("hook_mlp_out"))
    assert sorted(cache) == ["blocks.0.hook_mlp_out", "blocks.1.hook_mlp_out"]
    with pytest.raises(KeyError, match="no hook points"):
        model.run_with_cache(tokens(), names=["blocks.9.hook_mlp_out"])


def test_hook_intervention_and_cleanup():
    model = DecoderLM(tiny()).eval()
    x = tokens()
    clean = model(x).logits

    def zero(act, name):
        return torch.zeros_like(act)

    with model.hooks({"blocks.1.hook_mlp_out": zero, "blocks.1.hook_attn_out": zero, "blocks.0.hook_mlp_out": zero,
                      "blocks.0.hook_attn_out": zero}):
        ablated = model(x).logits
    assert not torch.allclose(clean, ablated)
    # With every block's output zeroed, logits depend only on each position's own token.
    same = (x[0, 0] == x).nonzero()
    for b, t in same:
        torch.testing.assert_close(ablated[b, t], ablated[0, 0])
    torch.testing.assert_close(model(x).logits, clean)  # hooks removed on exit
    assert not any(hp.has_hooks for hp in model.hook_points().values())


def test_head_ablation_via_hook_z():
    model = DecoderLM(tiny()).eval()
    x = tokens()

    def kill_head_2(z, name):
        z = z.clone()
        z[:, :, 2] = 0
        return z

    with model.hooks({"blocks.0.attn.hook_z": kill_head_2}):
        _, cache = model.run_with_cache(x, names=["blocks.0.attn.hook_z"])
    assert cache["blocks.0.attn.hook_z"][:, :, 2].abs().sum() == 0


# --- generation -----------------------------------------------------------------

@pytest.mark.parametrize("opts", [dict(), dict(positional="learned"), dict(positional="alibi", sliding_window=4)], ids=_id)
def test_generate_cache_matches_no_cache(opts):
    model = DecoderLM(tiny(**opts))
    prompt = tokens(t=5)
    a = model.generate(prompt, 20, temperature=0)
    b = model.generate(prompt, 20, temperature=0, use_cache=False)
    assert a.shape == (2, 25)
    assert torch.equal(a, b)
    assert model.training  # mode restored


def test_generate_past_max_seq_len():
    model = DecoderLM(tiny(positional="learned", max_seq_len=16))
    out = model.generate(tokens(t=10), 20, temperature=0)
    assert out.shape == (2, 30)


def test_generate_sampling_and_eos():
    torch.manual_seed(0)
    model = DecoderLM(tiny())
    out = model.generate(tokens(t=3), 8, temperature=0.8, top_k=10, top_p=0.9)
    assert out.shape == (2, 11)
    first = model.generate(tokens(t=3), 1, temperature=0)[:, -1]
    stopped = model.generate(tokens(t=3), 8, temperature=0, eos_token_id=int(first[0]))
    assert stopped[0, 3] == first[0]
    assert (stopped[0, 3:] == first[0]).all()


# --- regressions ------------------------------------------------------------------

@pytest.mark.parametrize("pos", ["learned", "sinusoidal"])
def test_layer_overrides_with_embedding_positional(pos):
    model = DecoderLM(tiny(positional=pos, layer_overrides={0: {"mlp": "none"}, 1: {"positional": "rope"}}))
    assert model.blocks[0].mlp is None
    assert isinstance(model.blocks[0].attn.pos, NoPositional)
    assert isinstance(model.blocks[1].attn.pos, RotaryPositional)
    x = tokens()
    assert torch.isfinite(model(x, targets=x).loss)


def test_positional_frequencies_survive_bf16_cast():
    cfg = tiny(d_model=256, n_heads=4, max_seq_len=4096)  # d_head = 64
    x = torch.randn(1, 1, 1, 64)
    pos = torch.tensor([2047])
    torch.testing.assert_close(RotaryPositional(cfg).bfloat16().rotate(x, pos), RotaryPositional(cfg).rotate(x, pos))
    q_pos, k_pos = torch.arange(2040, 2048), torch.arange(2048)
    torch.testing.assert_close(ALiBiPositional(cfg).bfloat16().bias(q_pos, k_pos), ALiBiPositional(cfg).bias(q_pos, k_pos))


@pytest.mark.parametrize("pos", ["rope", "alibi"])
def test_bf16_model_runs(pos):
    model = DecoderLM(tiny(positional=pos)).bfloat16()
    assert torch.isfinite(model(tokens()).logits.float()).all()


@pytest.mark.parametrize("n", [1, 2, 4, 8, 10])
def test_gemma2_sliding_windows_follow_n_layers(n):
    model = DecoderLM.from_preset("gemma2", n_layers=n, d_model=64, d_head=16, n_heads=4, n_kv_heads=2, vocab_size=V)
    assert [b.attn.window for b in model.blocks] == [256 if i % 2 == 0 else None for i in range(n)]


def test_gemma2_explicit_layer_overrides_replace_preset_default():
    cfg = config_from_preset("gemma2", n_layers=4, layer_overrides={3: {"sliding_window": 64}})
    assert cfg.layer_overrides == {3: {"sliding_window": 64}}


def test_fused_causal_path_builds_no_mask(monkeypatch):
    calls = []
    real = attention_module.attention_mask
    monkeypatch.setattr(attention_module, "attention_mask", lambda *a: calls.append(a) or real(*a))
    DecoderLM(tiny())(tokens())
    assert calls == []
    DecoderLM(tiny(sliding_window=3))(tokens())  # masked paths still build it
    assert calls
