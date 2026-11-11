# tforge

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

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests cover a grid of norm × position × encoding × MLP combinations plus GQA/MQA, sliding window,
softcaps, parallel residual and layer overrides. Each combination is checked for causality,
KV-cache vs full-forward equality, and fused vs explicit attention equality.

## Not included (yet)

Padding masks for batched prompts of different lengths, mixture-of-experts layers, a production
training loop (`tforge.training` only has an optimizer and LR-schedule helper), and loading
pretrained Hugging Face weights.
