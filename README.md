# tforge

[![CI](https://github.com/SDUzbh/tforge/actions/workflows/ci.yml/badge.svg)](https://github.com/SDUzbh/tforge/actions/workflows/ci.yml)

Build decoder-only transformer LMs from a single config: sensible defaults, presets for
common architecture families, per-layer overrides, pluggable components, and named hook
points for interpretability work (activation caching, ablations, transcoder training data).

```python
from tforge import DecoderLM, ModelConfig

model = DecoderLM(ModelConfig())                       # defaults: small Llama-style model
model = DecoderLM.from_preset("gpt2", n_layers=6)      # preset + overrides
model = DecoderLM(ModelConfig(
    vocab_size=8000, d_model=256, n_layers=4, n_heads=8, n_kv_heads=2,
    norm="layernorm", positional="alibi", mlp="mlp", activation="relu2",
    layer_overrides={0: {"mlp": "none"}},              # layer 0 is attention-only
))

out = model(tokens, targets=targets)                   # LMOutput(logits, loss, kv_cache)
text = model.generate(prompt, max_new_tokens=50, temperature=0.8, top_p=0.9)
```

## What you can configure

| Area | Field | Options (default first) |
|---|---|---|
| Norm | `norm`, `norm_position` | `rmsnorm` / `layernorm` / `none`; `pre` / `post` / `sandwich` |
| Positions | `positional` | `rope` / `alibi` / `learned` / `sinusoidal` / `none` (NoPE); `rope_theta`, `rope_fraction` (partial rotary) |
| Attention | `n_kv_heads`, `d_head` | MHA (None), GQA (k), MQA (1); head size independent of `d_model` |
| | `qk_norm`, `sliding_window`, `attn_logit_softcap`, `attn_scale` | |
| MLP | `mlp`, `activation`, `d_mlp` | `swiglu` / `geglu` / `glu` / `mlp` / `none`; `gelu`, `gelu_tanh`, `relu`, `relu2`, `silu`, `tanh` |
| Block | `parallel_residual`, `bias`, `dropout`, `attn_dropout` | |
| Model | `tie_embeddings`, `embed_scale`, `final_logit_softcap`, `init_std`, `scale_residual_init` | |
| Per layer | `layer_overrides={i: {...}}` | any per-layer field, e.g. alternating sliding-window/global attention |

Presets: `llama`, `gpt2`, `vanilla` (original post-LN Transformer), `gemma2`,
`attn_only_2l`, `gelu_2l` (small interpretability models), `tiny`.

Configs validate on creation and round-trip through JSON (`cfg.save(path)`, `ModelConfig.load(path)`).

## Adding your own components

Every component is looked up by name in a registry, so registering one makes it usable from a config:

```python
import torch
from torch import nn
from tforge import register, DecoderLM, ModelConfig

@register("norm", "scalenorm")
class ScaleNorm(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.g = nn.Parameter(torch.tensor(dim ** 0.5))
    def forward(self, x):
        return self.g * x / x.norm(dim=-1, keepdim=True).clamp(min=1e-5)

model = DecoderLM(ModelConfig(norm="scalenorm"))
```

Registries and factory signatures: `norm(dim, cfg)`, `activation()`, `mlp(cfg)` (return `None` for
no MLP), `attention(cfg, layer_idx)`, `positional(cfg)` (subclass `EmbeddingPositional` or
`AttentionPositional`). `tforge.available()` lists everything registered.

## Hooks

Every interesting activation passes through a named `HookPoint`; `model.hook_names()` lists them.

```python
out, cache = model.run_with_cache(tokens)                        # all activations
cache["blocks.1.attn.hook_pattern"]                              # (B, H, T, T)
cache["blocks.0.hook_mlp_in"], cache["blocks.0.hook_mlp_out"]    # transcoder training pairs

def zero_head_3(z, name):                                        # z: (B, T, H, D)
    z = z.clone(); z[:, :, 3] = 0; return z

with model.hooks({"blocks.1.attn.hook_z": zero_head_3}):         # temporary intervention
    ablated = model(tokens).logits
```

Per block: `hook_resid_pre`, `hook_attn_in`, `hook_attn_out`, `hook_resid_mid`, `hook_mlp_in`,
`hook_mlp_out`, `hook_resid_post`; in attention: `hook_q`, `hook_k`, `hook_v` (after qk-norm and
RoPE), `hook_pattern`, `hook_z`; in the MLP: `hook_pre`, `hook_post`. Model level: `hook_embed`,
`hook_pos_embed`, `hook_final`.

Attention uses PyTorch's fused `scaled_dot_product_attention` unless something hooks
`hook_pattern` (or logit softcapping is on), then it computes the pattern explicitly. Both paths
are tested to give the same output.

## Example: induction heads in a 2-layer model

```bash
python examples/induction_heads.py
```

This trains the `attn_only_2l` preset (with RoPE) on sequences containing a repeated segment of
random length, then finds the circuit using the cached patterns. It checks the two-hop "virtual attention"
`P1 @ P0` from each query to the token it should copy, and confirms with head ablations. On CPU it
takes about 40 seconds. In our run the layer-0 heads settled on a lag of 9 rather than the textbook
previous-token lag of 1. RoPE makes every lag equally easy, so check the two-hop score, not just
attention to position t−1.

## Reverse engineering trained models in C++ (`interp/`)

`interp/` contains **tfi**, a dependency-free C++17 framework for post-training reverse engineering.
Export a trained model with `tforge.export.export_model(model, "model.safetensors")`, then:

```bash
cd interp && make
build/tfi heads    model.safetensors                 # previous-token / induction / two-hop head scores
build/tfi dla      model.safetensors --tokens ... --target 42   # which heads write the answer
build/tfi patch    model.safetensors --clean ... --corrupt ... --target 42 --site heads
build/tfi circuits model.safetensors                 # weight-only composition and OV copying
```

tfi re-implements the forward pass with the same hook names. It also provides a logit lens, head
ablation and a C++ API. See [interp/README.md](interp/README.md) for a full walkthrough on the
induction model.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests cover a grid of norm × position × encoding × MLP combinations plus GQA/MQA, sliding window,
softcaps, parallel residual and layer overrides. Each combination is checked for causality,
KV-cache vs full-forward equality, and fused vs explicit attention equality. When a C++ compiler is
available, `tests/test_cpp.py` also builds `interp/` and checks that its forward pass matches
PyTorch on every hook point.

## CI/CD

GitHub Actions runs on every push to `main` and every pull request
([.github/workflows/ci.yml](.github/workflows/ci.yml)):

- **python**: the full test suite on Linux and macOS × Python 3.9 and 3.12, plus an end-to-end run
  (train the induction model, export it, analyse it with `tfi`).
- **cpp**: `interp/` built with make (`-Werror -pedantic`) and CMake, using GCC and clang on Linux
  and clang on macOS.
- **sanitizers**: the C++ unit tests, golden tests and CLI run on AddressSanitizer +
  UndefinedBehaviorSanitizer builds.

To release, bump `version` in `pyproject.toml`, commit, then push a matching tag:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

[.github/workflows/release.yml](.github/workflows/release.yml) checks that the tag matches the
version and reruns CI. It then publishes a GitHub Release with the Python wheel and sdist plus `tfi`
binaries (CLI, static library, headers) for Linux x86-64 and macOS arm64. Tags with a hyphen, such
as `v0.2.0-rc1`, become pre-releases. Dependabot keeps the action versions up to date.

## Not included (yet)

Padding masks for batched prompts of different lengths, mixture-of-experts layers, a production
training loop (`tforge.training` only has an optimizer and LR-schedule helper), and loading
pretrained Hugging Face weights.
