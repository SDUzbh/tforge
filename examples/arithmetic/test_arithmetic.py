import random
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(ROOT)]

from tforge import DecoderLM  # noqa: E402
from task import (  # noqa: E402
    VOCAB, ArithmeticTask, decode, encode, evaluate, model_config, parse_expression, predict, solve)


def test_layout_and_examples():
    t = ArithmeticTask()
    assert (t.prompt_len, t.answer_len, t.seq_len) == (12, 7, 19)
    assert t.format(123, "+", 45) == "00123+00045=+861000"
    assert t.format(7, "-", 42) == "00007-00042=-530000"
    assert t.format(12, "*", 34) == "00012*00034=+804000"
    assert t.format(99999, "+", 99999) == "99999+99999=+899991"
    plain = ArithmeticTask(reverse_answer=False)
    assert plain.format(123, "+", 45) == "00123+00045=+000168"
    pos = t.positions()
    assert pos["a"] == [0, 1, 2, 3, 4] and pos["b"] == [6, 7, 8, 9, 10] and pos["sign"] == [12]
    assert pos["answer"][0] == 13 and plain.positions()["answer"][0] == 18


@pytest.mark.parametrize("task", [ArithmeticTask(), ArithmeticTask(reverse_answer=False),
                                  ArithmeticTask(max_digits=3, max_mul_digits=3), ArithmeticTask(ops=("*",)),
                                  ArithmeticTask(max_digits=7, max_mul_digits=2, ops=("+", "-"))], ids=str)
def test_every_sample_round_trips(task):
    rng = random.Random(0)
    for _ in range(5000):
        a, op, b = task.sample(rng)
        text = task.format(a, op, b)
        assert len(text) == task.seq_len
        assert decode(encode(text)) == text
        assert task.parse_answer(text[task.prompt_len:]) == solve(a, op, b)


def test_sampling_is_balanced_by_digits():
    rng = random.Random(1)
    t = ArithmeticTask(ops=("+",))
    lengths = [len(str(t.sample(rng)[0])) for _ in range(20000)]
    for d in range(1, 6):
        assert abs(lengths.count(d) / len(lengths) - 0.2) < 0.02


def test_batch_masks_everything_but_the_answer():
    t = ArithmeticTask()
    inputs, targets = t.batch(8, random.Random(2))
    assert inputs.shape == targets.shape == (8, t.seq_len - 1)
    assert (targets[:, : t.prompt_len - 1] == -100).all()
    assert (targets[:, t.prompt_len - 1:] >= 0).all()
    # The first scored target is the sign, predicted from the "=" position.
    assert all(decode([v]) in "+-" for v in targets[:, t.prompt_len - 1].tolist())
    assert all(decode([v]) == "=" for v in inputs[:, t.prompt_len - 1].tolist())


def test_validation():
    t = ArithmeticTask()
    with pytest.raises(ValueError, match="operands"):
        t.prompt(100000, "+", 1)
    with pytest.raises(ValueError, match="operands"):
        t.prompt(1000, "*", 1)  # beyond max_mul_digits
    with pytest.raises(ValueError, match="not in this task"):
        ArithmeticTask(ops=("+",)).prompt(1, "*", 1)
    with pytest.raises(ValueError):
        ArithmeticTask(max_digits=2, max_mul_digits=3)
    with pytest.raises(ValueError):
        encode("12/3")
    assert t.parse_answer("+12") is None and t.parse_answer("x861000") is None and t.parse_answer("+86a000") is None
    assert parse_expression(" 12 * 34 ") == (12, "*", 34)
    with pytest.raises(ValueError):
        parse_expression("-1+2")
    assert ArithmeticTask.from_dict(t.to_dict()) == t


class Oracle(nn.Module):
    """Stands in for a model: always answers correctly (or garbage, if broken)."""

    def __init__(self, task, broken=False):
        super().__init__()
        self.task, self.broken = task, broken
        self.dummy = nn.Parameter(torch.zeros(1))

    def generate(self, prompts, n, temperature=0):
        rows = []
        for p in prompts.tolist():
            text = decode(p)
            a, op, b = int(text[: self.task.max_digits]), text[self.task.max_digits], int(text[self.task.max_digits + 1: -1])
            ans = "=" * n if self.broken else self.task.answer(solve(a, op, b))
            rows.append(p + encode(ans))
        return torch.tensor(rows)


def test_predict_and_evaluate_with_an_oracle():
    t = ArithmeticTask(max_digits=4, max_mul_digits=2)
    assert predict(Oracle(t), t, [(12, "*", 34), (5, "-", 9000)]) == [408, -8995]
    good = evaluate(Oracle(t), t, per_cell=20)
    assert good.accuracy == 1.0
    assert set(good.by_op_digits) == {"+", "-", "*"}
    assert sorted(good.by_op_digits["+"]) == [1, 2, 3, 4] and sorted(good.by_op_digits["*"]) == [1, 2]
    assert "overall exact-match accuracy 1.0000" in good.table()
    assert evaluate(Oracle(t, broken=True), t, per_cell=5).accuracy == 0.0


def test_real_model_generates_well_formed_lengths():
    t = ArithmeticTask(max_digits=3, max_mul_digits=2)
    torch.manual_seed(0)
    model = DecoderLM(model_config(t, d_model=32, n_layers=1, n_heads=2)).eval()
    preds = predict(model, t, [(1, "+", 2), (99, "*", 99)])
    assert len(preds) == 2  # an untrained model may answer None (malformed) or a wrong number
    assert model.cfg.max_seq_len == t.seq_len and model.cfg.vocab_size == len(VOCAB)


@pytest.mark.skipif(shutil.which("make") is None, reason="needs make for the tfi part")
def test_script_end_to_end(tmp_path):
    """train -> eval -> calc -> export -> tfi, all on a tiny model."""
    script = [sys.executable, str(HERE / "arithmetic.py")]
    ckpt, st = tmp_path / "a.pt", tmp_path / "a.safetensors"
    run = lambda *args: subprocess.run([*script, *args], capture_output=True, text=True, check=True).stdout
    out = run("train", "--out", str(ckpt), "--steps", "20", "--eval-every", "20", "--eval-per-cell", "3",
              "--d-model", "32", "--layers", "1", "--heads", "2", "--digits", "3", "--mul-digits", "2", "--device", "cpu")
    assert "saved" in out and ckpt.exists()
    assert "overall exact-match accuracy" in run("eval", str(ckpt), "--per-cell", "3", "--device", "cpu")
    calc = run("calc", str(ckpt), "12+34", "99*99", "7/2")
    assert "12 + 34 =" in calc and "99 * 99 =" in calc and "7/2: expected" in calc
    assert "exported" in run("export", str(ckpt), str(st))

    subprocess.run(["make", "-s", "-C", str(ROOT / "interp"), "all"], check=True)
    tfi = str(ROOT / "interp" / "build" / "tfi")
    out = subprocess.run([tfi, "run", str(st), "--text", "012+034=+640"], capture_output=True, text=True, check=True).stdout
    assert "'+'" in out and "'='" in out
    out = subprocess.run([tfi, "attn", str(st), "--text", "012+034=", "--layer", "0", "--head", "1", "--pos", "7"],
                         capture_output=True, text=True, check=True).stdout
    assert "layer 0 attention" in out and "'='" in out
    out = subprocess.run([tfi, "dla", str(st), "--text", "012+034=", "--target", "+", "--other", "-"],
                         capture_output=True, text=True, check=True).stdout
    assert "logit['+'] - logit['-']" in out
    out = subprocess.run([tfi, "patch", str(st), "--clean-text", "012+034=", "--corrupt-text", "012-034=",
                          "--target", "+", "--other", "-"], capture_output=True, text=True, check=True).stdout
    assert "logit['+'] - logit['-']" in out
    refs = tmp_path / "refs.txt"
    refs.write_text(run("problems", str(ckpt), "20"))
    assert len(refs.read_text().splitlines()) == 20
    out = subprocess.run([tfi, "ablate", str(st), "--text", "012+034=+6400", "--reference-file", str(refs)],
                         capture_output=True, text=True, check=True).stdout
    assert "means over 20 reference sequences, one per position" in out
    bad = subprocess.run([tfi, "run", str(st), "--text", "12/3"], capture_output=True, text=True)
    assert bad.returncode == 1 and "no token matches" in bad.stderr
