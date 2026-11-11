# tfi — post-training reverse engineering in C++

`tfi` loads a trained tforge model and takes it apart. It re-implements tforge's forward pass in
dependency-free C++17 with the same named hook points, then adds the standard reverse-engineering
tools on top: activation-based ones that need inputs, and weight-based ones that need none.

```
tforge (PyTorch)  --train-->  model  --tforge.export.export_model-->  model.safetensors  -->  tfi (C++)
```

## Build

```bash
cd interp
make            # build/tfi (CLI), build/libtfi.a, build/tfi_tests, build/tfi_golden
make test       # C++ unit tests
```

Or with CMake: `cmake -S . -B build && cmake --build build`. On macOS, matrix multiplies use
Accelerate; elsewhere a portable loop is used.

## Export a model

```python
from tforge.export import export_model
export_model(model, "model.safetensors")   # FP32 weights + config in one file
```

Or train the example circuit and export it in one go:

```bash
python examples/induction_heads.py --export induction.safetensors
```

## Commands

| Command | Kind | What it answers |
|---|---|---|
| `info` | – | Architecture, parameter count, hook point names |
| `run` | activations | Next-token predictions and loss |
| `lens` | activations | Logit lens: what the residual stream predicts after each sublayer |
| `dla` | activations | Direct logit attribution: which heads/MLPs write the answer directly |
| `heads` | activations | Per-head pattern scores (previous-token, duplicate, induction, first-token, self) and two-hop induction pairs at any lag |
| `patch` | activations | Activation patching clean → corrupted, per layer × position or per head (normalized, or raw with `--raw`). `--method attribution` estimates every site from gradients; `--method mixed` then measures the top estimates exactly |
| `ablate` | activations | Loss increase when each head is zero-ablated, or mean-ablated with means from reference sequences |
| `circuits` | weights | Q/K/V composition scores vs. a random baseline, single-head and two-head OV copying |
| `attn` | activations | Where each head of a layer attends, query by query (top keys with weights) |

Tokens are ids: `--tokens 5,17,3`, or `--repeat-random L` for a random segment repeated twice. If
the model was exported with a vocabulary (`export_model(model, path, vocab=[...])`), you can pass
`--text "00123+00045="` instead (`--clean-text` / `--corrupt-text` for `patch`). `--target` and
`--other` then also accept a token's text, and all output shows tokens as text. Run
`build/tfi --help` for every option.

## Walkthrough: the 2-layer induction circuit

On the model from `examples/induction_heads.py` (2 layers × 4 heads, attention-only, RoPE). The
transcripts below are from a real run, trimmed.

**Which heads form the circuit?** The textbook scores (attention to t−1, to the token after the
earlier copy) stay near zero. The lag-independent two-hop score finds the circuit anyway:

```
$ tfi heads induction.safetensors --seg 20 --samples 16
previous token (t-1):
      L0  0.001  0.010  0.001  0.003
two-hop induction (reader attends to r, writer at r attends to the token to copy; any lag):
  L0H2 -> L1H3   0.920
  L0H0 -> L1H3   0.916
  L0H2 -> L1H1   0.914
```

**Where does the answer appear?** Layer 1 does the induction step:

```
$ tfi lens induction.safetensors --repeat-random 20 --seed 1 --pos 30
stage     p(target)  rank
embed     0.0000     114
L0.attn   0.0017     112
L1.attn   0.4076     0
```

**Who writes it?** The layer-1 heads write it directly. The parts sum exactly to the real logit difference:

```
$ tfi dla induction.safetensors --repeat-random 20 --seed 1 --pos 31 --target 101 --other 99
actual 7.2142   sum of parts 7.2142   (final norm scale frozen)
L1H3                2.4083  +########################################
L1H0                2.2056  +####################################
L1H1                1.8915  +###############################
```

**How does the information move?** Corrupt the source token (position 12) and patch activations back in:

```
$ tfi patch induction.safetensors --clean ... --corrupt ... --pos 31 --target 101 --other 55 --site resid_pre
              ...     12  ...     21  ...
      L0       0  1.000    0       0
      L1       0      0    0   1.000
$ tfi patch ... --site heads
             H0     H1     H2     H3
      L0  0.475 -0.009  0.549  0.001
```

The source token's information sits at position 12 before layer 0 and at position 21 = 12 + 9
before layer 1. The layer-0 heads H0 and H2 carry it with a lag of 9 (0.48 + 0.55 ≈ 1).

**Can the weights alone confirm it?** No single head copies tokens, but writer→reader OV paths do:

```
$ tfi circuits induction.safetensors
OV copying: fraction of tokens a head's direct path maps back to themselves (chance 0.0078)
      L0  0.000  0.000  0.008  0.000
      L1  0.000  0.000  0.000  0.000
two-head OV path copying (writer OV, then reader OV):
  L0H2 -> L1H0   0.969
  L0H0 -> L1H0   0.953
```

Five methods, two of them weight-only, agree on one mechanism: lag-9 heads (L0H0, L0H2) shift each
token's information forward, and layer-1 heads match on it and copy the answer.

## Attribution patching

`tfi patch` uses exact patching by default (`--method exact`): every number is measured. It runs
one forward pass per site, so its cost grows with layers × positions. Two optional methods trade
accuracy for speed. Attribution patching (Nanda 2023; Syed et al. 2023) estimates every site at once from one clean run,
one corrupted run and one backward pass:

```
effect ≈ (clean activation − corrupted activation) · ∂metric/∂activation   (gradient on the corrupted run)
```

The estimate is linear, so it is exact only where the logits depend linearly on a site. It
underestimates large changes that saturate attention or an MLP. `--steps N` averages the gradient
over N runs whose token embeddings move from the corrupted input towards the clean one. This is the
integrated-gradients variant of EAP-IG (Hanna et al. 2024), and it costs one more forward + backward
pass per step. `--compare` also runs exact patching and reports how well the two agree:

```
$ tfi patch induction.safetensors --clean ... --corrupt ... --pos 31 --target 101 --other 55 \
      --site heads --method attribution --steps 8 --compare
             H0     H1     H2     H3
      L0  0.335 -0.008  0.491 -0.002
      L1  0.219  0.260  0.128  0.258
...
agreement over 8 sites: correlation 0.980, largest error 0.140 at L0 H0 (estimate 0.335, exact 0.475)
```

Agreement with exact patching on the two example models:

| Model, site | Plain (1 step) | `--steps 8` |
|---|---|---|
| Induction, `resid_pre` (80 sites) | r = 0.989, worst error 0.64 | r = 0.994, worst error 0.18 |
| Induction, heads (8) | r = 0.737, worst error 0.42 | r = 0.980, worst error 0.14 |
| Arithmetic carry, `resid_pre` (56 sites) | r = 0.971, worst error 0.89 | r = 0.977, worst error 0.47 |
| Arithmetic carry, heads (32) | r = 0.943, worst error 0.86 | r = 0.982, worst error 0.43 |

The estimates find the right sites, but their sizes are only rough. On the arithmetic carry, plain
attribution gives the carry head L0H1 0.13 where exact patching gives 0.99. With `--steps 8` it
gives 0.56 and ranks L0H1 first.

### Mixed patching

`--method mixed` combines the two methods. Attribution estimates every site, then exact patching
re-measures the `--verify K` sites with the largest estimates (K = 10 unless given). The measured
sites are exact; every other entry is still an estimate. It costs attribution plus K forward
passes. Entries marked `*` were measured exactly:

```
$ tfi patch arith.safetensors --clean-text "58291+76834=+5" --corrupt-text "58297+76834=+1" \
      --pos 13 --target 2 --other 3 --site heads --method mixed --verify 5
* = measured by exact patching (the 5 largest estimates); other entries are attribution estimates

             H0      H1      H2      H3      H4      H5      H6      H7
      L0 -0.001   0.989*  0.132* -0.000   0.001   0.000   0.003  -0.004
      L1  0.003  -0.000  -0.007  -0.001   0.021* -0.001  -0.004   0.016*
      L2 -0.001  -0.002  -0.000   0.000  -0.005   0.003   0.013* -0.008
      L3  0.000   0.002   0.011  -0.002   0.001   0.001   0.002   0.001

the exactly measured sites' estimates were off by up to 0.855 (L0 H1: estimate 0.134, exact 0.989)
largest estimate left unverified: 0.011 (L3 H2)
```

The carry head's estimate was badly low, but it still ranked first, so it was measured exactly.
Against full exact patching, this table has a correlation of 0.999 and a largest error of 0.004.

The risk is a site whose estimate is so low that it falls below the cut. With plain attribution
and K = 3–5, that happens to induction head L0H0, estimated at 0.05 but really 0.48. `--steps 8`
ranks sites better and fixes it. Largest error against full exact patching:

| Model, site | K = 3 | K = 5 | K = 3, `--steps 8` | K = 10, `--steps 8` |
|---|---|---|---|---|
| Induction, `resid_pre` (80 sites) | 0.000 | 0.000 | 0.000 | 0.000 |
| Induction, heads (8) | 0.421 | 0.421 | 0.017 | 0.000 |
| Arithmetic carry, `resid_pre` (56) | 0.893 | 0.009 | 0.471 | 0.001 |
| Arithmetic carry, heads (32) | 0.008 | 0.004 | 0.028 | 0.012 |
| Arithmetic carry chain, `resid_pre` (60) | 0.961 | 0.006 | 0.398 | 0.014 |
| Arithmetic carry chain, heads (32) | 0.005 | 0.005 | 0.039 | 0.004 |

The output reports how far off the measured sites' estimates were and the largest estimate left
unmeasured. When the estimates were far off, unmeasured sites may be too. Exact patching (the
default) has no such risk.

The gradients come from `Model::run_with_gradients` (or `metric_gradients`): a backward pass that
produces activation gradients only, no weight gradients. It costs about one forward pass.

## C++ API

```cpp
#include "tfi/analysis.h"
using namespace tfi;

Model m = Model::load("model.safetensors");
Tokens t = {5, 17, 3, 5, 17};

ActivationCache cache;                                       // every hook point, tforge names
Tensor logits = m.run_with_cache(t, cache);
const Tensor& pattern = cache.at("blocks.1.attn.hook_pattern");   // (H, T, T)

Hooks hooks;                                                 // interventions
hooks.add("blocks.0.attn.hook_z", [](const std::string&, Tensor& z) { /* edit (T, H, D) in place */ });
Tensor edited = m.forward(t, &hooks);

DLAResult dla = direct_logit_attribution(m, t, /*pos=*/4, /*target=*/3, /*other=*/9);
Composition comp = composition_scores(m);                    // weights only

Tokens clean = t, corrupt = {5, 17, 8, 5, 17};
ActivationCache grads;                                       // d metric / d activation, every hook point
metric_gradients(m, t, /*pos=*/4, /*target=*/3, /*other=*/9, grads);
PatchResult atp = attribution_patch_heads(m, clean, corrupt, /*pos=*/4, /*target=*/3, /*other=*/9, /*steps=*/8);
PatchResult mix = mixed_patch_heads(m, clean, corrupt, 4, 3, 9, /*verify=*/10, /*steps=*/8);  // mix.exact flags measured entries
```

Link against `build/libtfi.a` with `-Iinclude` (plus `-framework Accelerate` on macOS).

## Correctness

- **Golden tests** (`tests/test_cpp.py` in the repo root) export ~120 model configurations and
  compare the C++ forward pass with PyTorch on the logits and every hook point. The configurations
  cover every norm × placement × positional encoding × MLP combination, GQA/MQA, sliding window,
  softcaps, parallel residual, per-layer overrides and all presets. Agreement is at float32
  rounding level, about 3e-7 relative. The same configurations check the backward pass: the
  gradient of a metric at every hook point must match PyTorch autograd's to 1e-4.
- **Unit tests** (`tests/unit.cpp`) check invariants on random models: DLA parts sum to the real
  logit difference, the last logit-lens stage equals the model's output, patching the only
  changed input restores the clean run exactly, the fast low-rank composition formula equals the
  direct d×d computation, plus causality and hooks. Gradients at every hook point are checked
  against finite differences. Attribution patching must equal exact patching where the model is
  linear in a site. Mixed patching must measure exactly the top-K estimated sites, bit-identical
  to exact patching.

## Approximations and limits

- **DLA** freezes the final norm's scale (standard practice). It is not defined for post-norm
  models. With sandwich norm it attributes whole attention layers, since per-head outputs pass
  through a norm.
- **Weight-based tools** fold each norm's weight into the adjacent matrices but ignore its scale
  and centering. Q/K composition ignores RoPE's position-dependent rotation (the tool prints a
  note). Compare composition scores with the printed random baseline, not against zero.
- **Head scores** use random repeated sequences, as is standard for finding induction heads.
- **Patching** reports each patch's raw effect (`patched − corrupted`) and its normalized value,
  that effect divided by the gap `clean − corrupted`. If the corruption doesn't change the metric
  (gap 0), normalized values are undefined: `PatchResult::normalized()` is false, and `tfi patch`
  says so and shows raw effects instead of a table of NaN. If the gap is under 1 it warns, because
  dividing by a small gap amplifies noise. `.` in a table marks a site that doesn't exist (e.g.
  `mlp_out` on an attention-only layer).
- **Attribution patching** is a first-order estimate of patching (see above). It is reliable for
  ranking sites, not for their exact size. **Mixed patching** is exact at the sites it measures;
  the rest are estimates.
- **Mean ablation** replaces a head's output with its average over *reference* sequences: per
  position when they match the input's length, otherwise one pooled mean. It never averages over
  the input being tested. That would copy later tokens into earlier positions and break causality;
  earlier versions did this and could misjudge head importance by several times. Pass references
  with `--reference-file` (one per line), `--reference-text`, `--reference`, or
  `--reference-random N` (only meaningful for models trained on random tokens). In C++, see
  `head_means` and `ablate_head`. `ablate` measures next-token loss over the whole sequence, so for
  formats with unpredictable parts (like the arithmetic prompts) the per-head numbers include that
  noise.
- **Scope:** one sequence at a time, CPU, float32, no tokenizer (token ids only). Only built-in
  tforge components can be exported.

## Performance

Patching, ablation and reference means run many independent forward passes. Three things make
them cheap without changing any result:

- **Skipping unchanged layers.** A patch or ablation at layer ℓ leaves layers 0…ℓ−1 as in the
  unpatched run, so each pass starts at layer ℓ from the cached residual stream
  (`Model::forward_from`).
- **Skipping unreachable positions.** The patching metric at `--pos` depends only on tokens up to
  it, so later tokens are dropped, and patches after `--pos` aren't run (their effect is exactly 0).
- **Threads.** Passes run in parallel on every core by default. Use `--threads N` or `TFI_THREADS`
  to change that, or `set_num_threads()` in C++. Results are bit-identical for any thread count.

On an M5 Pro (18 cores), for a 38M-parameter model (12 layers, d_model 512) and 64 tokens:

| | before | after | speedup |
|---|---|---|---|
| Residual patching, 12 layers × 64 positions | 12.98 s | 1.24 s | 10.4× |
| Same, metric at position 30 | 12.83 s | 0.75 s | 17.1× |
| Head patching, 96 heads | 1.67 s | 0.22 s | 7.7× |
| Mean ablation, 96 heads | 1.82 s | 0.25 s | 7.3× |

Unit tests check that the optimized versions agree with straightforward implementations, and CI
runs them under ThreadSanitizer.

Attribution patching avoids the per-site passes altogether. Excluding the ~0.06 s it takes to load
the model:

| Same model, residual stream | 64 tokens (768 sites) | 128 tokens (1,536 sites) |
|---|---|---|
| Exact patching | 1.17 s | 4.29 s |
| Attribution patching | 0.05 s | 0.11 s |
| Attribution patching, `--steps 8` | 0.12 s | 0.24 s |
| Mixed patching, `--verify 10 --steps 8` | 0.19 s | 0.36 s |

On this model, mixed patching with `--steps 8` matches full exact patching to within 0.016 at
every site (correlation 0.999).
