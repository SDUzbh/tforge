# IndraNet

[![CI](https://github.com/SDUzbh/IndraNet/actions/workflows/ci.yml/badge.svg)](https://github.com/SDUzbh/IndraNet/actions/workflows/ci.yml)

IndraNet takes trained decoder-only transformers apart. Its core is **tfi**, a dependency-free
C++17 engine that re-implements the forward and backward pass with named hook points and runs the
standard reverse-engineering methods on top: the logit lens, direct logit attribution, head scores,
activation patching (exact, attribution and mixed), ablation, and weight-only circuit analysis. The
`indra` command line puts all of it one command away. On macOS the engine uses Accelerate and a
Metal GPU path; elsewhere it runs on portable C++.

The Python package [`indranet`](python/) supplies the models to study. It builds and trains them in
PyTorch from a single config, with the same hook names as the engine, and exports them in one file.

```
indranet (PyTorch)  --train-->  model  --export_model-->  model.safetensors  -->  indra (C++, tfi engine)
```

| Path | What it is |
|---|---|
| [include/tfi/](include/tfi/), [src/](src/) | The tfi engine: `build/libtfi.a`, namespace `tfi` |
| [tools/indra.cpp](tools/indra.cpp) | The `indra` command line |
| [tests/](tests/) | C++ unit tests and the golden-test runner |
| [python/](python/) | The `indranet` package: model building, training helpers, export, its tests |
| [examples/](examples/) | Two circuits trained in Python and reverse-engineered with `indra` |

## Build

```bash
make            # build/indra (CLI), build/libtfi.a, build/tfi_tests, build/tfi_golden
make test       # C++ unit tests
```

Or with CMake: `cmake -S . -B build-cmake && cmake --build build-cmake`. On macOS, matrix
multiplies use Accelerate and batched passes use the GPU; elsewhere a portable loop is used.

## Get a model to study

```bash
pip install -e "python[dev]"
```

```python
from indranet import DecoderLM, ModelConfig
from indranet.export import export_model

model = DecoderLM.from_preset("attn_only_2l")   # or any config; see python/README.md
...                                             # train it
export_model(model, "model.safetensors")        # FP32 weights + config in one file
```

Or train an example circuit and export it in one go:

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
`build/indra --help` for every option.

## Walkthrough: the 2-layer induction circuit

On the model from `examples/induction_heads.py` (2 layers × 4 heads, attention-only, RoPE). The
transcripts below are from a real run, trimmed.

**Which heads form the circuit?** The textbook scores (attention to t−1, to the token after the
earlier copy) stay near zero. The lag-independent two-hop score finds the circuit anyway:

```
$ indra heads induction.safetensors --seg 20 --samples 16
previous token (t-1):
      L0  0.001  0.010  0.001  0.003
two-hop induction (reader attends to r, writer at r attends to the token to copy; any lag):
  L0H2 -> L1H3   0.920
  L0H0 -> L1H3   0.916
  L0H2 -> L1H1   0.914
```

**Where does the answer appear?** Layer 1 does the induction step:

```
$ indra lens induction.safetensors --repeat-random 20 --seed 1 --pos 30
stage     p(target)  rank
embed     0.0000     114
L0.attn   0.0017     112
L1.attn   0.4076     0
```

**Who writes it?** The layer-1 heads write it directly. The parts sum exactly to the real logit difference:

```
$ indra dla induction.safetensors --repeat-random 20 --seed 1 --pos 31 --target 101 --other 99
actual 7.2142   sum of parts 7.2142   (final norm scale frozen)
L1H3                2.4083  +########################################
L1H0                2.2056  +####################################
L1H1                1.8915  +###############################
```

**How does the information move?** Corrupt the source token (position 12) and patch activations back in:

```
$ indra patch induction.safetensors --clean ... --corrupt ... --pos 31 --target 101 --other 55 --site resid_pre
              ...     12  ...     21  ...
      L0       0  1.000    0       0
      L1       0      0    0   1.000
$ indra patch ... --site heads
             H0     H1     H2     H3
      L0  0.475 -0.009  0.549  0.001
```

The source token's information sits at position 12 before layer 0 and at position 21 = 12 + 9
before layer 1. The layer-0 heads H0 and H2 carry it with a lag of 9 (0.48 + 0.55 ≈ 1).

**Can the weights alone confirm it?** No single head copies tokens, but writer→reader OV paths do:

```
$ indra circuits induction.safetensors
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

`indra patch` uses exact patching by default (`--method exact`): every number is measured. It runs
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
$ indra patch induction.safetensors --clean ... --corrupt ... --pos 31 --target 101 --other 55 \
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
$ indra patch arith.safetensors --clean-text "58291+76834=+5" --corrupt-text "58297+76834=+1" \
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

ActivationCache cache;                                       // every hook point, IndraNet names
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

Link against `build/libtfi.a` with `-Iinclude` (on macOS also `-framework Accelerate -framework Metal
-framework MetalPerformanceShaders -framework Foundation`).

## Correctness

- **Golden tests** (`python/tests/test_golden.py`) export ~120 model configurations and
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
  linear in a site. Mixed patching must measure exactly the top-K estimated sites and agree with
  exact patching (bit for bit when it measures every site). Batched patching must agree with
  naive patching for every batch size (on macOS with the GPU on and off), the GPU linear
  layer with the CPU one, and a fused GPU submission with separate calls (bit for bit). Batched
  ablation and reference means must agree with one plain pass per head and per reference. On
  macOS, each GPU kernel must match its CPU counterpart. Passes that keep the residual stream on
  the GPU must match both the plain forward pass and the one-multiply-at-a-time GPU path. That's
  checked for 14 model types, including post-norm and every activation, with hooks on both sides.
  On the same 14 types, batched GPU gradients must match one CPU backward pass per sequence at
  every hook point they report. Attribution and mixed patching on the GPU must match the CPU for
  any batch size and keep the CPU tests' invariants.

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
  (gap 0), normalized values are undefined: `PatchResult::normalized()` is false, and `indra patch`
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
- **Scope:** one sequence at a time, float32 (fp16 GPU inputs with `--fp16`), CPU plus the GPU for
  batched passes on macOS, no tokenizer (token ids only). Only built-in
  indranet components can be exported.

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

Within each pass, the matrix multiplies go to Accelerate on macOS (plain loops elsewhere). The
elementwise work (GELU and other activations, the attention softmax, norms) uses polynomial `exp`,
`erf` and `tanh` that the compiler vectorizes, within 2.5 ulp of the standard library's
(`src/vmath.h`). Patching unembeds only `--pos`, the one row its metric reads.

### Batched exact patching

On macOS the cores share Accelerate's matrix units: together they top out at about 2.1 TFLOP/s
for multiplies of this size, only 2.5× one core, so running more single passes side by side stops
helping. Exact patching therefore also:

- **Batches passes (macOS).** Passes that start at the same layer are stacked into one set of
  rows (up to 1,024, or 512 for head patching), so each linear layer is one large multiply instead
  of many small ones. The portable matrix multiply gains nothing from that, so other systems run
  one pass per batch.
- **Recomputes only what changes.** A residual-stream patch at position t can't change earlier
  positions, so its pass computes only positions t…`--pos`, and attention reads the earlier keys
  and values from the corrupted run. That halves the work on average.
- **Uses the GPU (macOS).** Linear layers with at least 256 rows run on the GPU (Metal
  Performance Shaders, `src/gpu.mm`), which has several times the matrix units' throughput. Each
  model keeps its own copy of the weights on the GPU. Without a Metal device, and on other
  systems, everything stays on the CPU. To keep it on the CPU anyway, use `--no-gpu` or
  `TFI_NO_GPU=1`, or `set_use_gpu(false)` in C++ (on the 38M model at 64 tokens: 0.40 s instead
  of 0.28 s).
- **Keeps the residual stream on the GPU (macOS).** In a batched pass the GPU runs every part of
  a block except attention: the linear layers, norms, activations and residual adds. Only q, k
  and v come back to the CPU for attention, and z goes back, so each block costs one GPU round
  trip, each of which adds about 0.2 ms of overhead. Before, each linear layer's input and output
  crossed over separately, 4 round trips per block. The norm, activation and add kernels are a
  few dozen lines of Metal (`src/gpu.mm`), using the CPU's polynomials, so results agree with the
  CPU to float rounding. A block with a hook on one of those GPU-side activations runs the old
  way. Patching and ablation hook only z or the block a pass starts at, so at most a pass's first
  block runs the old way.
  See [GPU-resident blocks](#gpu-resident-blocks) for timings.
- **fp16, optional (macOS).** `--fp16`, or `set_gpu_fp16(true)` in C++, gives the GPU's multiplies
  fp16 inputs (accumulating in fp32). The multiplies run about 4× faster, which makes patching
  1.3–1.6× faster and ablation 1.6–1.8× faster than in fp32 (38M model). The metric picks
  up errors of up to about 1e-3, so normalized values are off by up to about 1e-3 / gap. On the
  example models (gaps 13–35) the tables don't change at three decimals. On the 38M model with a
  gap of 0.008 they're off by up to 0.14, enough to reorder the top sites, and `indra patch` warns
  when that happens. A block step whose multiply inputs exceed fp16's range (±65,504) is computed
  again in fp32, as is any multiply whose weight does.
- **Skips sites that can't matter.** A site whose clean and corrupted activations are identical,
  or a position before `--pos` after the last layer's attention, has an effect of exactly 0
  without running anything.

Each step on the same model (M5 Pro), all 18 threads, measured in-process with the model loaded:

| | Per pass | + batching | + recompute only changes | + GPU |
|---|---|---|---|---|
| Residual patching, 64 tokens | 0.98 s | 0.61 s | 0.33 s | 0.16 s (6.3×) |
| Residual patching, 128 tokens | 3.61 s | 2.63 s | 1.32 s | 0.60 s (6.0×) |
| Residual patching, 64 tokens, 1 thread | 3.58 s | 1.84 s | 0.88 s | 0.82 s (4.4×) |
| Head patching, 64 tokens | 0.136 s | 0.141 s | | 0.083 s (1.6×) |
| Head patching, 128 tokens | 0.254 s | 0.253 s | | 0.136 s (1.9×) |

On all cores, batching alone doesn't speed up head patching (on one thread it does: 0.46 s →
0.34 s), but it makes the multiplies large enough for the GPU. Batched results match per-pass
patching up to float rounding (about 1e-6 on this model), not bit for bit, because a site's
multiply is shared with the other sites in its batch. They are still identical for any thread
count. Running NEON multiplies on the cores alongside the matrix units was tried and dropped:
it lowered the total throughput. Later changes (below) made the GPU path faster still.

### GPU-resident blocks

Keeping the residual stream on the GPU (above) replaced an earlier version that sent each linear
layer to the GPU separately, 4 round trips per block. In that version the Q, K and V projections
already shared one submission, as did a gated MLP's gate and up projections, which was 0–10%
faster than one submission per weight.

Times on the 38M model (M5 Pro, 18 threads, in-process, best of 5), patching the last position
with one token changed in the middle, mean ablation with 16 reference sequences:

| | fp32 before | fp32 now | fp16 before | fp16 now |
|---|---|---|---|---|
| Residual patching, 64 / 128 tokens | 81 / 211 ms | 62 / 184 ms (1.3× / 1.15×) | 76 / 190 ms | 47 / 136 ms (1.6× / 1.4×) |
| Head patching, 64 / 128 tokens | 78 / 128 ms | 64 / 119 ms (1.2× / 1.07×) | 71 / 108 ms | 41 / 75 ms (1.7× / 1.4×) |
| Mean ablation, 64 / 128 tokens | 90 / 151 ms | 71 / 139 ms (1.3× / 1.09×) | 79 / 122 ms | 44 / 77 ms (1.8× / 1.6×) |
| Zero ablation, 64 / 128 tokens | 61 / 111 ms | 56 / 109 ms (1.08× / 1.02×) | 48 / 90 ms | 32 / 61 ms (1.5× / 1.5×) |

In fp32 the GPU's multiplies already ran at about 85% of its peak and kept it busy for 45–77% of
each command, so there was little left to gain. In fp16 they're about 4× faster, so the round
trips and the CPU's share of the work had become the bottleneck. Attention stays on the CPU,
running on all cores while other batches use the GPU. At up to 128 tokens on this model, CPU
attention takes about as long as the GPU's fp16 multiplies. Longer sequences would make an
attention kernel worthwhile. For bigger models the crossover moves later (roughly 1,000+ tokens
for a model with d_model 2,000–3,600). The batch sizes and the 256-row GPU threshold were
re-tuned and kept: no other setting was faster across the board.

### Batched ablation

`ablate` and the reference means for mean ablation work the same way as head patching: ablation
passes that start at the same layer are stacked (256 rows per batch, which is the GPU threshold and
still leaves several batches to run in parallel), and so are the reference sequences. A head whose
output already equals its replacement at every position has an effect of exactly 0, without a
pass. Results match one pass per head to float rounding (the printed tables are unchanged), and are
identical for any thread count. `--no-gpu` and `--fp16` apply as for patching; fp16 changes each
loss increase by up to about 1e-4.

On the same model, all 18 threads, 16 reference sequences:

| | before | after |
|---|---|---|
| Zero ablation, 32 / 64 / 128 tokens | 0.116 / 0.144 / 0.262 s | 0.041 / 0.062 / 0.114 s (2.3–2.8×) |
| Mean ablation, 32 / 64 / 128 tokens | 0.149 / 0.185 / 0.340 s | 0.068 / 0.091 / 0.153 s (2.0–2.2×) |
| Same with `--fp16` | | 2.2–2.9× (zero), 2.2–2.7× (mean) |

### Weight-only circuits

`circuits` runs on all cores: each head's matrices are computed in parallel, then the head pairs
(composition scores, the random baseline's pairs, single-head and two-head OV paths). An OV path
is evaluated in factored form, (X W_V) (W_V' W_O)ᵀ … (U W_O)ᵀ, the same product as pushing every
sampled token through d_model and the full unembedding, but with rank-`d_head` factors. On the
38M model that's about 4.5× fewer operations. Scores are identical for any thread count. The
random baseline draws each random pair from its own seeded stream, so its value differs slightly
from earlier versions (e.g. `4.6x` → `4.5x` baseline on one arithmetic model).

| `indra circuits`, whole command | before | after |
|---|---|---|
| 38M model (96 heads; two-head paths skipped) | 0.455 s | 0.101 s (4.5×) |
| Arithmetic model, 6 layers × 8 heads (with two-head paths) | 0.132 s | 0.028 s (4.7×) |

Of the 0.101 s, ~0.04 s is loading the model, and the composition scores (0.035 s) already run
the matrix units near their peak.

### Attribution and mixed patching

Attribution patching avoids the per-site passes. It needs one clean run, plus one gradient run (a
forward and a backward pass) per step. Three things make those cheap:

- **All runs at once.** The clean run, the corrupted run and every step's gradient run are
  independent, so they run in parallel instead of in three phases. On the CPU alone they gain
  less than the thread count suggests: they share the CPU's matrix units, so 7 gradient runs at
  once take 2.3× as long as one.
- **Mixed patching reuses attribution's runs.** Its exact patching uses the clean and corrupted
  runs attribution already did, instead of repeating them. They're the same runs, so the exact
  values are identical bit for bit.
- **Gradient runs on the GPU (macOS).** The steps' sequences are stacked into batches (aiming for 8
  batches of at most 256 rows), and each batch's forward and backward passes run on the GPU, one
  submission per block each way. Each block's activations stay on the GPU for the backward pass,
  which has its own kernels for the norms, activations and GLU. Attention's forward and backward
  passes run on the CPU, one sequence at a time. Results agree with the CPU's to float rounding
  and are the same for any thread count. The GPU is used only when steps × positions is at least
  256, since starting it costs about 60 ms in a fresh process. So attribution with one step on a
  short prompt stays on the CPU.

Times on the 38M model (M5 Pro, 18 threads, in-process, best of 5), residual stream:

| | 64 tokens before → now | 128 tokens before → now |
|---|---|---|
| Exact patching (for comparison) | 60 ms | 173 ms |
| Attribution, 1 step | 31 → 24 ms (1.3×) | 56 → 43 ms (1.3×) |
| Attribution, `--steps 8` | 77 → 21 ms (3.7×) | 148 → 35 ms (4.2×) |
| Mixed, `--verify 10`, 1 step | 61 → 37 ms (1.6×) | 100 → 60 ms (1.7×) |
| Mixed, `--verify 10 --steps 8` | 108 → 34 ms (3.2×) | 195 → 50 ms (3.9×) |
| Mixed heads, `--verify 10 --steps 8` | 123 → 43 ms (2.9×) | 210 → 58 ms (3.6×) |

As whole CLI commands, the GPU's startup takes part of the gain: at 128 tokens `--method
attribution --steps 8` goes from 0.20 s to 0.14 s, and `--method mixed --steps 8` from 0.25 s to
0.16 s. At 64 tokens attribution with `--steps 8` is no faster (0.12 s), while mixed goes from
0.16 s to 0.14 s.

On this model, mixed patching with `--steps 8` matches full exact patching to within 0.016 at
every site (correlation 0.999).

## Examples

**Induction heads in a 2-layer model.** `python examples/induction_heads.py` trains the
`attn_only_2l` preset (with RoPE) on sequences containing a repeated segment of random length,
then finds the circuit from the cached patterns. It checks the two-hop "virtual attention"
`P1 @ P0` from each query to the token it should copy, and confirms with head ablations. On CPU it
takes about 40 seconds. In our run the layer-0 heads settled on a lag of 9 rather than the textbook
previous-token lag of 1. RoPE makes every lag equally easy, so check the two-hop score, not just
attention to position t−1. The walkthrough above takes the same model apart with `indra`.

**A transformer for arithmetic.** [examples/arithmetic/](examples/arithmetic/) trains a
3.2M-parameter model on fixed-width `+`, `−`, `×` problems, reaching 100% on 5-digit addition,
99.2–100% on subtraction and 98% on 2-digit multiplication (a larger 10.7M model reaches 80% on
3-digit multiplication). Its [README](examples/arithmetic/README.md) then reverse-engineers how it
adds with `indra`:
- layer-0 heads fetch the matching digits and the carry digits;
- the model reads its own previous output for chained carries;
- the MLPs compute the sum.

```bash
python examples/arithmetic/arithmetic.py train --out arith.pt     # ~3 min on an Apple-silicon GPU
python examples/arithmetic/arithmetic.py export arith.pt arith.safetensors
build/indra attn arith.safetensors --text "58291+76834=+521531" --layer 0
```

## Tests

```bash
make test       # C++ unit tests
pytest          # Python tests, golden tests against PyTorch, the indra CLI and the examples
```

## CI/CD

GitHub Actions runs on every push to `main` and every pull request
([.github/workflows/ci.yml](.github/workflows/ci.yml)):

- **cpp**: the engine built with make (`-Werror -pedantic`) and CMake, using GCC and clang on Linux
  and clang on macOS.
- **sanitizers**: the C++ unit tests, golden tests and CLI on AddressSanitizer +
  UndefinedBehaviorSanitizer builds, and the unit tests under ThreadSanitizer.
- **python**: the full test suite on Linux and macOS × Python 3.9 and 3.12, plus an end-to-end run
  (train the induction model, export it, analyse it with `indra`).

To release, bump `version` in `python/pyproject.toml`, commit, then push a matching tag:

```bash
git tag v0.1.0 && git push origin v0.1.0
```

[.github/workflows/release.yml](.github/workflows/release.yml) checks that the tag matches the
version and reruns CI. It then publishes a GitHub Release with `indra` binaries (CLI, static
library, headers) for Linux x86-64 and macOS arm64, plus the Python wheel and sdist. Tags with a
hyphen, such as `v0.2.0-rc1`, become pre-releases. Dependabot keeps the action versions up to date.

## Not included (yet)

Loading pretrained Hugging Face weights (so models like Qwen or Gemma can't be studied yet), a
tokenizer in the engine, padding masks for batched prompts of different lengths, and
mixture-of-experts layers.
