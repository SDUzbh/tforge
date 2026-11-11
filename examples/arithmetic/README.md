# A transformer for arithmetic, and how it works inside

This guide trains a small tforge model to do integer arithmetic (`+`, `−`, `×`), then
reverse-engineers how it adds, using the C++ tool `tfi`. Every output below comes from a real
run (seed 0, Apple M-series GPU via MPS), trimmed for length.

## Quick start

```bash
python examples/arithmetic/arithmetic.py train --out arith.pt           # ~3 min on an M-series GPU
python examples/arithmetic/arithmetic.py eval arith.pt
python examples/arithmetic/arithmetic.py calc arith.pt "58291+76834" "10000-1" "999*999"
python examples/arithmetic/arithmetic.py export arith.pt arith.safetensors
cd interp && make && cd ..
interp/build/tfi attn arith.safetensors --text "58291+76834=+521531" --layer 0
```

Everything for this example is in this folder:

| File | What it is |
|---|---|
| [task.py](task.py) | The task: format, tokenizer, random problems, evaluation, calculator parsing |
| [arithmetic.py](arithmetic.py) | Command-line script: `train`, `eval`, `calc`, `export` |
| [test_arithmetic.py](test_arithmetic.py) | Tests, including an end-to-end train → calc → export → tfi run (`python -m pytest examples/arithmetic`) |
| README.md | This walkthrough |

It uses tforge for the model and `interp/` (tfi) for the analysis; nothing else in the repo depends on it.

## The format

Every problem uses one fixed layout:

```
00123+00045=+861000      123 + 45 = +168
00007-00042=-530000      7 − 42 = −35
00012*00034=+804000      12 × 34 = +408
```

- **Zero-padded operands** (5 digits for `+` and `−`; up to 3 significant digits for `×`). Every
  sequence is 19 tokens, so batches need no padding masks, and each digit always sits at the same
  position. Operand a's 10^k digit is at position 4−k, b's at 10−k, and the answer's 10^k digit at
  position 13+k (predicted from position 12+k).
- **The answer is written least-significant digit first**, after a sign. The model then writes
  each digit after the carries it depends on. Below we'll see it read its own previous digits to
  handle chains of carries.
- **Fresh random problems every step**, with each operand's digit count uniform. Short and long
  numbers are therefore equally common, and the model can't memorize the test set. Loss is computed
  only on the answer tokens.
- The vocabulary is 14 characters: `0123456789+-*=`.

## Results

The default model has 4 layers, 8 heads, d_model 256 and learned positions (3.2M parameters),
trained for 6,000 steps of 256 problems (163 seconds). Exact-match accuracy on 500 fresh problems
per cell, by operator and the longer operand's digit count:

```
$ python examples/arithmetic/arithmetic.py eval arith.pt
op        1d      2d      3d      4d      5d
+      1.000   1.000   1.000   1.000   1.000
-      1.000   0.992   1.000   1.000   0.998
*      1.000   0.980   0.516
overall exact-match accuracy 0.9605
```

Addition and subtraction are essentially solved. 3-digit multiplication is the hard part, since
every answer digit depends on many partial products.

```
$ python examples/arithmetic/arithmetic.py calc arith.pt "99999+1" "58291+76834" "10000-1" "999*999" "456*789"
99999 + 1 = 100000   ok
58291 + 76834 = 135125   ok
10000 - 1 = 9999   ok
999 * 999 = 998001   ok
456 * 789 = 361184   WRONG (correct: 359784)
```

### Bigger model

More capacity and a longer schedule take 3-digit multiplication from 52% to 80%:

```
$ python examples/arithmetic/arithmetic.py train --out arith-large.pt \
      --layers 6 --d-model 384 --steps 20000 --warmup 500 --eval-every 2500     # 10.7M params, ~23 min
$ python examples/arithmetic/arithmetic.py eval arith-large.pt
op        1d      2d      3d      4d      5d
+      1.000   1.000   1.000   1.000   1.000
-      1.000   1.000   1.000   1.000   1.000
*      1.000   1.000   0.804
overall exact-match accuracy 0.9849
```

Most of the gain came late. At step 10,000 this model was still at 52% on 3-digit
multiplication, the same as the small one. It reached about 80% only in the final steps, as the
learning rate decayed. It now gets `456*789 = 359784` right, and its mistakes are often a single
digit (e.g. `987*654` → 649498 instead of 645498). Everything else is perfect on these 500-problem
samples.

## How it adds: reverse engineering with tfi

`export` writes the model plus its vocabulary, so `tfi` accepts `--text` and prints tokens as
characters. Head numbers below are from this training run. Retraining with a different seed or on
different hardware finds the same mechanism, but possibly on differently numbered heads.

### 1. Layer 0 attention gathers the digits

`tfi attn` shows where each head looks. At each answer position, a few layer-0 heads lock onto
exactly the operand digits that the next answer digit needs:

```
$ tfi attn arith.safetensors --text "58291+76834=+521531" --layer 0
layer 0 attention: for each query, the top-1 key(s) per head as position:token weight
 12   '+' | H0 10:4 1.00 | H1 5:+ 0.93 | H2 6:7 0.16 | H3 10:4 0.70 | H4 5:+ 0.39 | H5 10:4 0.49 | H6 4:1 0.97 | H7 5:+ 0.76 |
 13   '5' | H0 3:9 1.00 | H1 4:1 0.99 | H2 4:1 0.96 | H3 9:3 0.99 | H4 5:+ 0.94 | H5 9:3 0.96 | H6 9:3 0.87 | H7 5:+ 0.54 |
 14   '2' | H0 2:2 1.00 | H1 9:3 0.82 | H2 3:9 0.76 | H3 8:8 1.00 | H4 5:+ 0.99 | H5 8:8 0.81 | H6 8:8 0.70 | H7 5:+ 0.66 |
 15   '1' | H0 1:8 1.00 | H1 8:8 0.60 | H2 8:8 0.56 | H3 7:6 1.00 | H4 5:+ 0.99 | H5 7:6 1.00 | H6 7:6 0.77 | H7 5:+ 0.70 |
```
(Rows 0–11 trimmed. Each entry is head, key position:token at that position, attention weight.)

When predicting the answer's 10^k digit (query 12+k):
- **H0** reads operand a's 10^k digit (position 4−k), and **H3/H5** read b's (position 10−k).
  For the units digit, H6 reads a's and H0 reads b's.
- **H1/H2** read the digits one place lower (positions 5−k and 11−k): the inputs to the carry.
- **H4/H7** read the operator, which tells the model which operation to perform.

Layers 1–3 attend mostly to `+` and `=`, which serve as resting positions.

### 2. Patching confirms the routing

Change a's units digit without changing any carry (`58291` → `58292`, so the units digit goes
5 → 6). Then patch clean activations back in, one layer and position at a time:

```
$ tfi patch arith.safetensors --clean-text "58291+76834=+" --corrupt-text "58292+76834=+" \
      --pos 12 --target 5 --other 6
activation patching (resid_pre), metric at position 12: logit['5'] - logit['6']
clean 13.2543   corrupted -14.0208   (1 = clean behaviour restored, 0 = no effect)
              0      1      2      3      4  ...     11     12
      L0  0.000  0.000  0.000  0.000  1.000  ...  0.000  0.000
      L1  0.000  0.000  0.000  0.000  0.005  ...  0.001  1.002
      L2  0.000  0.000  0.000  0.000  0.001  ...  0.000  1.001
```

Before layer 0 the information sits at a's units digit (position 4). From layer 1 on it's at the
query position (12). Per-head patching (`--site heads`) names the mover: **L0H6 restores 0.985**.

### 3. The carry comes from one head

Now make the change flip a carry. `58291+76834` has tens digit 2. `58297+76834` carries into the
tens place (tens digit 3). Each prompt includes its own units digit:

```
$ tfi patch arith.safetensors --clean-text "58291+76834=+5" --corrupt-text "58297+76834=+1" \
      --pos 13 --target 2 --other 3
              0  ...      4  ...     12     13
      L0  0.000  ...  0.986  ...  0.000  0.054
      L1  0.000  ...  0.002  ...  0.000  1.000
$ tfi patch ... --site heads
             H0     H1     H2     H3     H4     H5     H6     H7
      L0  0.003  0.989  0.132 -0.000  0.002  0.000  0.004 -0.000
```

The carry comes from operand a's units digit (position 4, 0.986), not from the units digit the
model just wrote (position 13, 0.054). **L0H1, the "one place lower" head, carries 0.989 of it.**

### 4. Chained carries: the model reads its own output

A single carry needs only the digits one place lower. A chain, as in `45 + 55 = 100`, needs more:
the hundreds digit depends on a carry that starts in the units place. Compare `45+54 = 99` with
`45+55 = 100`. The tens digits (4 and 5) are the same in both, so the one-place-lower head can't
tell them apart:

```
$ tfi patch arith.safetensors --clean-text "00045+00054=+99" --corrupt-text "00045+00055=+00" \
      --pos 14 --target 0 --other 1
              0  ...     10  ...     13     14
      L0  0.000  ... -0.011  ...  0.005  0.984
      L1  0.000  ... -0.008  ...  0.006  0.988
```

Here the deciding input is **the tens digit the model itself just wrote** (position 14). If the
tens digits sum to 9 and the model wrote a 0, a carry must have come through. This is what writing
the answer least-significant first buys: every carry the model needs is either in the digits one
place lower or already in its own output.

### 5. The MLPs do the arithmetic

Attention only moves digits. The sum is computed by the MLPs:

```
$ tfi dla arith.safetensors --text "58291+76834=+5" --pos 13 --target 2 --other 3
actual 17.1622   sum of parts 17.1622   (final norm scale frozen)
L3.mlp             13.6160  +########################################
L2.mlp              2.0042  +#####
L0.mlp              1.0495  +###
L1.mlp              0.4013  +#
L3H7                0.1411  +
$ tfi lens arith.safetensors --text "58291+76834=+52" --pos 13      (earlier stages and top-k columns trimmed)
stage     p(target)  rank
L1.mlp    0.0037     7
L2.attn   0.0024     8
L2.mlp    0.8901     0
L3.mlp    1.0000     0
```

The right digit jumps from rank 8 to rank 0 in layer 2's MLP, and layer 3's MLP sharpens it to
certainty. No attention head writes more than 0.15 of the 17.2-logit margin directly.

### The algorithm, in one paragraph

For the answer's 10^k digit, layer-0 attention copies a's and b's 10^k digits and the digit pair
one place lower to the current position. The model's own previous output digit is already there,
as the current input token. The MLPs in layers 2–3 then act as a lookup table: from the digit
pair, the lower pair and the previous output digit they work out the incoming carry, and output
the sum digit mod 10. Five tools agree on this picture: attention patterns, residual patching,
head patching, DLA and the logit lens.

## Limits

- Non-negative operands; up to 5 digits for `+`/`−` and 3 for `×` by default (`--digits`,
  `--mul-digits`). The model is trained for one fixed width and doesn't generalize to longer numbers.
- 3-digit multiplication tops out around 80% even for the larger model (see above). The
  walkthrough analyses addition, which both models have solved.
- MPS training isn't bit-for-bit deterministic, so a rerun gives slightly different accuracies and
  possibly different head numbers.
