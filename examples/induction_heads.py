"""Train a 2-layer attention-only model on sequences with a repeated segment, then
find its induction circuit from cached attention patterns.

Each sequence is random tokens, then a random segment of length L, then that same
segment again. Inside the repeat, the next token is only predictable by copying
"the token after the earlier occurrence of the current token". L and the
segment's position vary per sequence, so no fixed positional offset works.

A 2-layer model solves this by composing heads: a layer-0 head at position r
copies in the token from a fixed lag d back (d=1 is the textbook "previous-token
head"; with RoPE any lag is equally easy, so trained models often pick d>1), and
a layer-1 head at query q attends to the r whose lagged token matches q's token.
We measure the circuit by its two-hop "virtual attention" P1 @ P0, which should
point from q to q-L+1 whatever lag the model chose.

    python examples/induction_heads.py
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tforge import DecoderLM  # noqa: E402
from tforge.training import cosine_schedule, make_optimizer, set_lr  # noqa: E402

VOCAB, SEQ, MIN_L = 128, 48, 8


def batch(n: int):
    """Tokens (n, SEQ), segment lengths L (n,), and a mask over query positions p
    whose next token is a copy and so predictable by induction."""
    x = torch.randint(0, VOCAB, (n, SEQ))
    L = torch.randint(MIN_L, SEQ // 2 + 1, (n,))
    copy_start = torch.randint(0, SEQ + 1, (n,)) % (SEQ - 2 * L + 1) + L
    pos = torch.arange(SEQ)
    is_copy = (pos[None, :] >= copy_start[:, None]) & (pos[None, :] < (copy_start + L)[:, None])
    src = (pos[None, :] - L[:, None]).clamp(min=0)
    x = torch.where(is_copy, x.gather(1, src), x)
    predictable = is_copy[:, :-1] & is_copy[:, 1:]
    return x, L, predictable


def losses(model: DecoderLM, x: torch.Tensor, predictable: torch.Tensor):
    logits = model(x[:, :-1]).logits
    per_tok = F.cross_entropy(logits.transpose(1, 2), x[:, 1:], reduction="none")
    return per_tok, per_tok[predictable].mean(), per_tok[~predictable].mean()


@torch.no_grad()
def analyse(model: DecoderLM, n: int = 256):
    x, L, predictable = batch(n)
    _, cache = model.run_with_cache(x, names=lambda name: name.endswith("hook_pattern"))
    p0 = cache["blocks.0.attn.hook_pattern"]  # (B, H, T, T)
    p1 = cache["blocks.1.attn.hook_pattern"]
    b, q = predictable.nonzero(as_tuple=True)  # query positions inside the repeat
    target = q - L[b] + 1                      # the token to copy

    # Layer 0: the lag each head attends to most, and how much attention it gets.
    lags = []
    for h in range(p0.shape[1]):
        rows = p0[b, h, q]                                           # (N, T) over keys
        by_lag = torch.stack([rows[torch.arange(len(q)), (q - d).clamp(min=0)] for d in range(SEQ // 2)])
        mean = by_lag.mean(dim=1)
        lags.append((int(mean.argmax()), float(mean.max())))

    # Two-hop virtual attention q -> r (layer 1 head) -> source (layer 0 head).
    virtual = torch.einsum("bhqr,bgrs->bhgqs", p1, p0)               # (B, H1, H0, T, T)
    two_hop = virtual[b, :, :, q, target].mean(dim=0)                # (H1, H0)
    direct = p1[b, :, q, target].mean(dim=0)                         # classic lag-1 induction score
    return lags, two_hop, direct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--positional", default="rope", help="rope | learned | alibi | ...")
    args = ap.parse_args()
    torch.manual_seed(0)

    model = DecoderLM.from_preset("attn_only_2l", vocab_size=VOCAB, d_model=128, n_heads=4,
                                  max_seq_len=SEQ, positional=args.positional)
    print(f"{model.num_parameters():,} parameters; layers={model.cfg.n_layers}, heads={model.cfg.n_heads}, "
          f"mlp={model.cfg.mlp}, positional={model.cfg.positional}")
    opt = make_optimizer(model, lr=args.lr, weight_decay=0.01)

    start = time.time()
    for step in range(args.steps):
        set_lr(opt, cosine_schedule(step, args.steps, warmup=50, lr=args.lr))
        x, _, predictable = batch(args.batch_size)
        per_tok, copy_loss, rand_loss = losses(model, x, predictable)
        opt.zero_grad(set_to_none=True)
        per_tok.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0 or step == args.steps - 1:
            print(f"step {step:4d}  loss on copies {copy_loss.item():.3f}  on random tokens {rand_loss.item():.3f}"
                  f"  ({time.time() - start:.0f}s)")

    model.eval()
    lags, two_hop, direct = analyse(model)
    print("\nlayer 0 heads, (most-attended lag, attention on it):", [(d, round(a, 2)) for d, a in lags])
    print("layer 1 direct attention to the token to copy (lag-1 induction):", [round(v, 2) for v in direct.tolist()])
    print("two-hop virtual attention to the token to copy [layer-1 head x layer-0 head]:")
    for h1, row in enumerate(two_hop.tolist()):
        print(f"  L1H{h1}: " + "  ".join(f"{v:.2f}" for v in row))
    h1, h0 = divmod(int(two_hop.argmax()), two_hop.shape[1])
    print(f"=> strongest circuit: L0H{h0} (lag {lags[h0][0]}) -> L1H{h1}")

    def ablate(head):
        def fn(z, name):
            z = z.clone()
            z[:, :, head] = 0
            return z
        return fn

    x, _, predictable = batch(256)
    with torch.no_grad():
        base = losses(model, x, predictable)[1].item()
        for layer, head in [(0, h0), (1, h1)]:
            with model.hooks({f"blocks.{layer}.attn.hook_z": ablate(head)}):
                ablated = losses(model, x, predictable)[1].item()
            print(f"loss on copies {base:.3f} -> {ablated:.3f} with L{layer}H{head} zero-ablated")


if __name__ == "__main__":
    main()
