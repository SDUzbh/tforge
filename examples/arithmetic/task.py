"""Integer arithmetic (+, -, *) as a fixed-width character task.

Every problem is written in one fixed layout, so all sequences have the same length
(no padding masks needed) and each digit always sits at the same position:

    00123+00045=+861000      123 + 45 = +168, answer digits least-significant first
    00007-00042=-530000      7 - 42 = -35
    00012*00034=+804000      12 * 34 = +408

Layout: operand a (`max_digits` digits, zero-padded), operator, operand b, "=", then the
answer: a sign followed by `answer_digits` digits. Writing the answer least-significant
digit first (``reverse_answer=True``, the default) lets the model emit each digit after
the carries it depends on, which small transformers learn far more easily.
"""
from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from indranet import ModelConfig

VOCAB: Tuple[str, ...] = tuple("0123456789+-*=")
STOI: Dict[str, int] = {c: i for i, c in enumerate(VOCAB)}
OPS = ("+", "-", "*")


def encode(text: str) -> List[int]:
    try:
        return [STOI[c] for c in text]
    except KeyError as e:
        raise ValueError(f"character {e.args[0]!r} is not in the arithmetic vocabulary {''.join(VOCAB)!r}") from None


def decode(ids: Sequence[int]) -> str:
    return "".join(VOCAB[int(i)] for i in ids)


@dataclass(frozen=True)
class ArithmeticTask:
    max_digits: int = 5          # operand width for + and -
    max_mul_digits: int = 3      # largest operands for *, still written max_digits wide
    ops: Tuple[str, ...] = OPS
    reverse_answer: bool = True

    def __post_init__(self):
        if not 1 <= self.max_mul_digits <= self.max_digits:
            raise ValueError("need 1 <= max_mul_digits <= max_digits")
        bad = set(self.ops) - set(OPS)
        if bad or not self.ops:
            raise ValueError(f"ops must be a non-empty subset of {OPS}, got {self.ops}")

    # --- layout ---------------------------------------------------------------
    @property
    def answer_digits(self) -> int:
        widths = [self.max_digits + 1] if {"+", "-"} & set(self.ops) else []
        if "*" in self.ops:
            widths.append(2 * self.max_mul_digits)
        return max(widths)

    @property
    def prompt_len(self) -> int:
        return 2 * self.max_digits + 2  # a op b =

    @property
    def answer_len(self) -> int:
        return 1 + self.answer_digits  # sign + digits

    @property
    def seq_len(self) -> int:
        return self.prompt_len + self.answer_len

    def positions(self) -> Dict[str, List[int]]:
        """Which sequence positions hold what (handy when reading attention patterns)."""
        n = self.max_digits
        ans = list(range(self.prompt_len + 1, self.seq_len))
        return {
            "a": list(range(n)), "op": [n], "b": list(range(n + 1, 2 * n + 1)), "=": [2 * n + 1],
            "sign": [self.prompt_len],
            # answer digit for 10**k, k = 0, 1, ...
            "answer": ans if self.reverse_answer else ans[::-1],
        }

    # --- formatting -------------------------------------------------------------
    def prompt(self, a: int, op: str, b: int) -> str:
        self.check(a, op, b)
        return f"{a:0{self.max_digits}d}{op}{b:0{self.max_digits}d}="

    def answer(self, value: int) -> str:
        digits = f"{abs(value):0{self.answer_digits}d}"
        if len(digits) > self.answer_digits:
            raise ValueError(f"{value} needs more than {self.answer_digits} digits")
        return ("-" if value < 0 else "+") + (digits[::-1] if self.reverse_answer else digits)

    def format(self, a: int, op: str, b: int) -> str:
        return self.prompt(a, op, b) + self.answer(solve(a, op, b))

    def parse_answer(self, text: str) -> Optional[int]:
        """Integer from an answer string, or None if it is malformed."""
        if len(text) != self.answer_len or text[0] not in "+-" or not text[1:].isdigit():
            return None
        digits = text[1:][::-1] if self.reverse_answer else text[1:]
        value = int(digits)
        return -value if text[0] == "-" else value

    def check(self, a: int, op: str, b: int) -> None:
        if op not in self.ops:
            raise ValueError(f"operator {op!r} is not in this task's ops {self.ops}")
        limit = self.max_mul_digits if op == "*" else self.max_digits
        for x in (a, b):
            if not 0 <= x < 10 ** limit:
                raise ValueError(f"operands for {op!r} must be in [0, {10 ** limit - 1}], got {x}")

    # --- sampling -----------------------------------------------------------------
    def sample(self, rng: random.Random) -> Tuple[int, str, int]:
        """One problem; each operand's digit count is uniform, so short and long numbers are balanced."""
        op = rng.choice(self.ops)
        limit = self.max_mul_digits if op == "*" else self.max_digits
        return _number(rng, rng.randint(1, limit)), op, _number(rng, rng.randint(1, limit))

    def batch(self, size: int, rng: random.Random, device="cpu") -> Tuple[Tensor, Tensor]:
        """(inputs, targets) for next-token training; targets are -100 except where an answer token is predicted."""
        seqs = torch.tensor([encode(self.format(*self.sample(rng))) for _ in range(size)], device=device)
        inputs, targets = seqs[:, :-1], seqs[:, 1:].clone()
        targets[:, : self.prompt_len - 1] = -100
        return inputs, targets

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ArithmeticTask":
        d = dict(d)
        d["ops"] = tuple(d.get("ops", OPS))
        return cls(**d)


def _number(rng: random.Random, digits: int) -> int:
    return rng.randint(0, 9) if digits == 1 else rng.randint(10 ** (digits - 1), 10 ** digits - 1)


def solve(a: int, op: str, b: int) -> int:
    return a + b if op == "+" else a - b if op == "-" else a * b


def model_config(task: ArithmeticTask, **overrides) -> ModelConfig:
    """A small decoder sized for the task; learned positions suit the fixed digit layout."""
    base = dict(vocab_size=len(VOCAB), max_seq_len=task.seq_len, d_model=256, n_layers=4, n_heads=8,
                positional="learned", norm="layernorm", mlp="mlp", activation="gelu", bias=True)
    base.update(overrides)
    return ModelConfig(**base)


@torch.no_grad()
def predict(model, task: ArithmeticTask, problems: Sequence[Tuple[int, str, int]], batch_size: int = 512) -> List[Optional[int]]:
    """Greedy answers for (a, op, b) problems; None where the model's output is malformed."""
    device = next(model.parameters()).device
    out: List[Optional[int]] = []
    for s in range(0, len(problems), batch_size):
        chunk = problems[s:s + batch_size]
        prompts = torch.tensor([encode(task.prompt(*p)) for p in chunk], device=device)
        full = model.generate(prompts, task.answer_len, temperature=0)
        out.extend(task.parse_answer(decode(row[task.prompt_len:].tolist())) for row in full)
    return out


@dataclass
class EvalResult:
    accuracy: float
    by_op_digits: Dict[str, Dict[int, float]] = field(default_factory=dict)  # op -> max operand digits -> accuracy

    def table(self) -> str:
        digits = sorted({d for v in self.by_op_digits.values() for d in v})
        lines = ["op  " + "".join(f"{d:>7}d" for d in digits)]
        for op, row in self.by_op_digits.items():
            lines.append(f"{op:<4}" + "".join(f"{row[d]:8.3f}" if d in row else f"{'':8}" for d in digits))
        return "\n".join(lines) + f"\noverall exact-match accuracy {self.accuracy:.4f}"


def evaluate(model, task: ArithmeticTask, per_cell: int = 200, seed: int = 1234) -> EvalResult:
    """Exact-match accuracy for every operator and operand length (the longer operand's digit count)."""
    rng = random.Random(seed)
    problems, keys = [], []
    for op in task.ops:
        limit = task.max_mul_digits if op == "*" else task.max_digits
        for d in range(1, limit + 1):
            for _ in range(per_cell):
                da, db = (d, rng.randint(1, d)) if rng.random() < 0.5 else (rng.randint(1, d), d)
                problems.append((_number(rng, da), op, _number(rng, db)))
                keys.append((op, d))
    preds = predict(model, task, problems)
    hits: Dict[Tuple[str, int], List[bool]] = {}
    for p, k, pred in zip(problems, keys, preds):
        hits.setdefault(k, []).append(pred == solve(*p))
    by: Dict[str, Dict[int, float]] = {}
    for (op, d), h in hits.items():
        by.setdefault(op, {})[d] = sum(h) / len(h)
    total = [x for h in hits.values() for x in h]
    return EvalResult(sum(total) / len(total), by)


_EXPR = re.compile(r"^\s*(\d+)\s*([+\-*])\s*(\d+)\s*$")


def parse_expression(expr: str) -> Tuple[int, str, int]:
    """'123 + 45' -> (123, '+', 45)."""
    m = _EXPR.match(expr)
    if not m:
        raise ValueError(f"expected 'A op B' with non-negative integers and op in + - *, got {expr!r}")
    return int(m.group(1)), m.group(2), int(m.group(3))
