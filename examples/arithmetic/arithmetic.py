"""A transformer for integer arithmetic (+, -, *), built with tforge.

    python examples/arithmetic/arithmetic.py train --out arith.pt          # MPS/CUDA if available
    python examples/arithmetic/arithmetic.py eval arith.pt                 # accuracy by operator and digits
    python examples/arithmetic/arithmetic.py calc arith.pt "12345+678" "99*99"
    python examples/arithmetic/arithmetic.py export arith.pt arith.safetensors   # for interp/build/tfi
    python examples/arithmetic/arithmetic.py problems arith.pt 200 > refs.txt    # reference data for tfi ablate

The task layout and data live in task.py next to this script; README.md walks through
reverse-engineering the trained model.
"""
import argparse
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]  # task.py, then the repo's tforge
from tforge import DecoderLM, ModelConfig  # noqa: E402
from tforge.export import export_model  # noqa: E402
from task import (  # noqa: E402
    VOCAB, ArithmeticTask, decode, evaluate, model_config, parse_expression, predict, solve)
from tforge.training import cosine_schedule, make_optimizer, set_lr  # noqa: E402


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save(path, model, task, step, result=None):
    torch.save({"model_config": model.cfg.to_dict(), "task": task.to_dict(), "step": step,
                "accuracy": None if result is None else result.accuracy,
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()}}, path)


def load(path, device="cpu"):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = DecoderLM(ModelConfig.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ArithmeticTask.from_dict(ckpt["task"]), ckpt


def cmd_train(args):
    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    task = ArithmeticTask(max_digits=args.digits, max_mul_digits=args.mul_digits, ops=tuple(args.ops),
                          reverse_answer=not args.no_reverse)
    model = DecoderLM(model_config(task, d_model=args.d_model, n_layers=args.layers, n_heads=args.heads)).to(device)
    print(f"task: {args.digits}-digit + and -, up to {args.mul_digits}-digit *, sequences of {task.seq_len} tokens "
          f"(e.g. {task.format(123, '+', 45)})")
    print(f"model: {model.num_parameters():,} parameters, {args.layers} layers x {args.heads} heads, "
          f"d_model {args.d_model}, on {device}")
    opt = make_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    best, start = -1.0, time.time()
    for step in range(1, args.steps + 1):
        set_lr(opt, cosine_schedule(step, args.steps, warmup=args.warmup, lr=args.lr))
        inputs, targets = task.batch(args.batch_size, rng, device)
        loss = model(inputs, targets=targets).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.log_every == 0:
            print(f"step {step:6d}  loss {loss.item():.4f}  ({time.time() - start:.0f}s)", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            result = evaluate(model, task, per_cell=args.eval_per_cell)
            model.train()
            print(result.table(), flush=True)
            if result.accuracy >= best:
                best = result.accuracy
                save(args.out, model, task, step, result)
                print(f"saved {args.out} (accuracy {best:.4f})", flush=True)
    print(f"done in {time.time() - start:.0f}s; best accuracy {best:.4f}")


def cmd_eval(args):
    model, task, ckpt = load(args.checkpoint, pick_device(args.device))
    print(f"checkpoint from step {ckpt['step']}")
    print(evaluate(model, task, per_cell=args.per_cell, seed=args.seed).table())


def cmd_calc(args):
    model, task, _ = load(args.checkpoint, pick_device(args.device))
    for expr in args.expressions:
        try:
            problem = parse_expression(expr)
            task.check(*problem)
        except ValueError as e:
            print(f"{expr}: {e}")
            continue
        got = predict(model, task, [problem])[0]
        truth = solve(*problem)
        a, op, b = problem
        mark = "ok" if got == truth else f"WRONG (correct: {truth})"
        print(f"{a} {op} {b} = {got}   {mark}   [{task.prompt(*problem)}{task.answer(got) if got is not None else '?'}]")


def cmd_export(args):
    model, task, _ = load(args.checkpoint)
    export_model(model, args.out, vocab=list(VOCAB), metadata={"task": "arithmetic", "task_config": str(task.to_dict())})
    pos = task.positions()
    print(f"exported to {args.out}")
    print(f"layout of {decode([0])}-padded sequences ({task.seq_len} tokens): a at {pos['a']}, op at {pos['op']}, "
          f"b at {pos['b']}, '=' at {pos['=']}, sign at {pos['sign']}, answer digit 10^k at {pos['answer']}")
    print(f"try: interp/build/tfi run {args.out} --text {task.prompt(123, '+', 45)}{task.answer(168)}")


def cmd_problems(args):
    """Random problems from the checkpoint's task, one full sequence per line (e.g. for tfi --reference-file)."""
    _, task, _ = load(args.checkpoint)
    rng = random.Random(args.seed)
    for _ in range(args.n):
        print(task.format(*task.sample(rng)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train a model")
    t.add_argument("--out", default="arith.pt")
    t.add_argument("--digits", type=int, default=5, help="max operand digits for + and -")
    t.add_argument("--mul-digits", type=int, default=3, help="max operand digits for *")
    t.add_argument("--ops", default="+-*", help="operators to train on, e.g. '+-'")
    t.add_argument("--no-reverse", action="store_true", help="write answers most-significant digit first")
    t.add_argument("--d-model", type=int, default=256)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--steps", type=int, default=6000)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--log-every", type=int, default=250)
    t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--eval-per-cell", type=int, default=200)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--device", default="auto", help="auto | cpu | mps | cuda")
    t.set_defaults(fn=cmd_train)

    e = sub.add_parser("eval", help="accuracy by operator and operand length")
    e.add_argument("checkpoint")
    e.add_argument("--per-cell", type=int, default=500)
    e.add_argument("--seed", type=int, default=99)
    e.add_argument("--device", default="auto")
    e.set_defaults(fn=cmd_eval)

    c = sub.add_parser("calc", help="ask the model to compute expressions like '123+45'")
    c.add_argument("checkpoint")
    c.add_argument("expressions", nargs="+")
    c.add_argument("--device", default="cpu")
    c.set_defaults(fn=cmd_calc)

    x = sub.add_parser("export", help="write a safetensors file for the C++ tools in interp/")
    x.add_argument("checkpoint")
    x.add_argument("out")
    x.set_defaults(fn=cmd_export)

    p = sub.add_parser("problems", help="print random problems, one per line (reference data for tfi ablate)")
    p.add_argument("checkpoint")
    p.add_argument("n", type=int)
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_problems)

    args = ap.parse_args()
    if getattr(args, "ops", None):
        bad = set(args.ops) - set("+-*")
        if bad:
            ap.error(f"--ops may only contain + - *, got {''.join(sorted(bad))}")
    args.fn(args)


if __name__ == "__main__":
    main()
