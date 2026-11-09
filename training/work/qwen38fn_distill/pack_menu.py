"""Setting menu for the distillation corpus: mode × instruction × grid representation × label_shape × effort.

Each sample draws one setting **deterministically** from `sha256(uid:attempt)`. Determinism is a hard requirement:
the prompt rebuilt on resume must be bit-identical to the original, otherwise the resumed trajectory continues a
different prompt (and nothing errors). The drawn setting is also written column by column into the parquet, so even if
the weight tables change later, existing records can still tell which prompt they used.

## Why not just use the 11 packs of prompt_packs

`work/qwen38fn_eval/prompt_packs.py` is the menu of an **ablation**: each of its 11 arms is a frozen
(instruction, representation, parser) combination, because the point of that batch was "change one variable at a time".
The goal here is the opposite -- coverage -- so instruction and representation are split into **orthogonal axes** and
multiplied. Instruction paragraphs, grid renderers, and parsers are all imported from that file, not copied; the only
additions are CoT versions of the paragraphs (all 11 arms are tied to the sandbox and unusable in no-tool mode).

## Representation: every cell takes at least one token, no multi-cell-per-token compression

Measured (30x30, this model's tokenizer): `json` 3.000 / `space` 1.999 / `tight` 1.032 tok/cell.
All three **split cell by cell** -- `tight`'s `012\n` is split into `['0','1','2', ...]`; 1.032 is one token per cell
plus the amortized newline, not several cells merged into one token.

The criterion is "does a cell share a token with another cell", not the absolute tok/cell. DSv4's tokenizer merges 3
digits into 1 token, so `tight` collapses there; this one does not -- the hard precondition from
[`work/grid_format/`](../grid_format/), "a new model must re-verify per-cell splitting", has been checked and passed on
this model (see `work/qwen38fn_eval/README.md` prompt ablation §1).

`tight` also has a practical benefit: the prompt drops from 3.00 to 1.03 tok/cell, leaving about 5.5k more of the
40,960 budget for generation.

## Changing the representation drags two things along; miss either and that arm is broken

1. **The S5 contract hardcodes "in the same JSON format"**; a non-JSON arm copying it makes the prompt contradict
   itself → replaced with `S5_CONTRACT_FMT` (deletes only the word "JSON").
2. **The official `backscan_json_parser` only understands JSON**, so a non-JSON arm gets recorded as 0.00%
   → replaced with `parse_any`. The `tight` arm in `work/qwen38fn_eval/` hit this: 0.6% parsed, 0.0% correct, which
   looked like a "representation disaster" but was really the parser not recognizing it.
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str, name: str):
    """Load by path, the same convention the harness uses to load sandbox / parsing (not a cross-unit import:
    CONVENTIONS forbids `import`ing another work unit; loading a module by path is the existing practice)."""
    spec = importlib.util.spec_from_file_location(name, _ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pp = _load("work/qwen38fn_eval/prompt_packs.py", "prompt_packs")


# --------------------------------------------------------------- CoT paragraphs
#
# All 11 arms of prompt_packs include S2_SANDBOX / S5_CONTRACT, both of which assume tools and a turn budget.
# No-tool mode needs its own versions. RATIONALE and RATIONALE_CODE do not mention the sandbox and are reused as-is.

C_GRID = "A grid is a list of lists of integers 0-9."
C_CONTRACT = (
    "Reply with the final output grid for the test input, in the same JSON format as "
    "the training outputs. Your final reply must contain the answer grid itself."
)
# Same reason and same change as S5_CONTRACT_FMT: delete only the word "JSON".
C_CONTRACT_FMT = C_CONTRACT.replace(
    "in the same JSON format as the training outputs",
    "in the same format as the training outputs",
)
C_CHECK = (
    "Before you commit to an answer, check your rule against every training pair: "
    "apply it to each training input and confirm it reproduces the expected training "
    "output exactly. If any training pair does not match, the hypothesis is wrong -- "
    "keep working instead of answering."
)
# No-tool version of STRATEGY: the last sentence "until the sandbox rules the others out" must change;
# the other five steps are verbatim.
C_STRATEGY = pp.STRATEGY.replace(
    "Keep more than one hypothesis alive until the sandbox rules the others out.",
    "Keep more than one hypothesis alive until the training pairs rule the others out.",
)


# --------------------------------------------------------------- instruction menu
#
# Each instruction is a function (s5) -> tuple of paragraphs: s5 is decided by the grid representation,
# so instruction and representation are truly orthogonal and no combination needs to be hand-written.

def _tir(instr: str, s5: str) -> tuple:
    S1, S2, S3, S4 = pp.S1_INTRO, pp.S2_SANDBOX, pp.S3_LIMITS, pp.S4_COACHING
    return {
        "official": (S1, S2, S3, S4, s5),
        "null": (S1, S2, pp.S3_REORDERED, S4, s5),
        "minimal": (S2, S3, s5),
        "verify": (S1, S2, S3, S4, pp.VERIFY_GATE, s5),
        "coach": (S1, S2, S3, S4, pp.VERIFY_GATE, pp.STRATEGY, s5),
        "rationale": (S1, S2, S3, S4, s5, pp.RATIONALE),
        "rulecode": (S1, S2, S3, S4, s5, pp.RATIONALE_CODE),
    }[instr]


def _cot(instr: str, s5: str) -> tuple:
    S1 = pp.S1_INTRO
    return {
        # No system message at all, only the user prompt -- the same condition as the DSv4 notool batch.
        "cot-official": (),
        "cot-minimal": (C_GRID, s5),
        "cot-check": (S1, C_GRID, C_CHECK, s5),
        "cot-strategy": (S1, C_GRID, C_STRATEGY, s5),
        "cot-rationale": (S1, C_GRID, s5, pp.RATIONALE),
        "cot-rulecode": (S1, C_GRID, s5, pp.RATIONALE_CODE),
    }[instr]


# Weights are relative; they need not sum to 100.
MODE_W = {"tir": 75, "cot": 25}
INSTR_W = {
    "tir": {"official": 25, "rationale": 15, "rulecode": 15, "verify": 12,
            "coach": 12, "minimal": 11, "null": 10},
    "cot": {"cot-official": 30, "cot-rationale": 22, "cot-rulecode": 22,
            "cot-check": 10, "cot-strategy": 8, "cot-minimal": 8},
}
# All three split cell by cell (json 3.000 / space 1.999 / tight 1.032 tok/cell, measured).
# json weighted slightly higher: eval uses json, and the main distribution should match the eval condition.
GRID_W = {"json": 40, "space": 30, "tight": 30}
SHAPE_W = {False: 85, True: 15}
EFFORT_W = {"xhigh": 60, "medium": 30, "low": 10}


class Setting:
    """Full configuration of one sample. `pack` is a prompt_packs.Pack instance (rendering and parsing live on it)."""

    __slots__ = ("mode", "instr", "grid_fmt", "label_shape", "effort", "pack")

    def __init__(self, mode, instr, grid_fmt, label_shape, effort):
        self.mode = mode
        self.instr = instr
        self.grid_fmt = grid_fmt
        self.label_shape = label_shape
        self.effort = effort
        is_json = grid_fmt == "json"
        if mode == "tir":
            s5 = pp.S5_CONTRACT if is_json else pp.S5_CONTRACT_FMT
            paragraphs = _tir(instr, s5)
        else:
            s5 = C_CONTRACT if is_json else C_CONTRACT_FMT
            paragraphs = _cot(instr, s5)
        self.pack = pp.Pack(
            name=f"{instr}|{grid_fmt}{'|shape' if label_shape else ''}",
            note=f"{mode} {instr} grid={grid_fmt} shape={label_shape}",
            paragraphs=paragraphs,
            grid_fmt=grid_fmt,
            parse=pp.parse_json_only if is_json else pp.parse_any,
            label_shape=label_shape,
            merge_user_head=(instr == "null"),
        )

    @property
    def uses_tools(self) -> bool:
        return self.mode == "tir"

    @property
    def wants_code(self) -> bool:
        """The rulecode family is asked to emit `def transform(grid)`; after collection it must actually be run once."""
        return self.instr.endswith("rulecode")

    def system(self, cfg: dict, max_turns: int) -> str:
        """An empty string means "put no system message", not "put an empty one"."""
        if not self.pack.paragraphs:
            return ""
        return self.pack.system(cfg, max_turns)

    def user(self, train, test_input):
        return self.pack.user(train, test_input)

    def parse(self, text):
        return self.pack.parse(text)

    def fingerprint(self, cfg: dict, max_turns: int) -> str:
        return pp.fingerprint(self.pack, cfg, max_turns)

    def as_dict(self) -> dict:
        return {"mode": self.mode, "instr": self.instr, "grid_fmt": self.grid_fmt,
                "label_shape": self.label_shape, "effort": self.effort}

    def __repr__(self):
        return (f"Setting({self.mode}/{self.instr}/{self.grid_fmt}"
                f"{'/shape' if self.label_shape else ''}/{self.effort})")


def _weighted(seed: bytes, axis: str, table: dict):
    """Independent hash per axis for the same (uid, attempt) -- axes must not be correlated, otherwise couplings like
    "the samples that drew rulecode all happen to be space grids" silently leak into the dataset."""
    h = hashlib.sha256(seed + b"|" + axis.encode()).digest()
    r = int.from_bytes(h[:8], "big") / float(1 << 64) * sum(table.values())
    acc = 0.0
    for key, w in table.items():
        acc += w
        if r < acc:
            return key
    raise RuntimeError(f"weight table {axis} does not cover r={r} (sum {sum(table.values())})")


def draw(uid: str, attempt: int) -> Setting:
    """Draw a setting deterministically. The same (uid, attempt) always gets the same setting."""
    seed = f"{uid}:{attempt}".encode()
    mode = _weighted(seed, "mode", MODE_W)
    return Setting(
        mode=mode,
        instr=_weighted(seed, "instr", INSTR_W[mode]),
        grid_fmt=_weighted(seed, "grid", GRID_W),
        label_shape=_weighted(seed, "shape", SHAPE_W),
        effort=_weighted(seed, "effort", EFFORT_W),
    )


def all_settings():
    """All possible combinations (for the fingerprint table and the smoke coverage check)."""
    for mode, instrs in INSTR_W.items():
        for instr in instrs:
            for grid_fmt in GRID_W:
                for label_shape in SHAPE_W:
                    yield Setting(mode, instr, grid_fmt, label_shape, "xhigh")


if __name__ == "__main__":
    import argparse
    import collections
    import json

    ap = argparse.ArgumentParser(description="list the menu, validate the weights, print any prompt")
    ap.add_argument("--config-json", default=str(_ROOT / "src/arcforge/tir/tir_config.json"))
    ap.add_argument("--max-turns", type=int, default=120)
    ap.add_argument("--show", default="", help="mode/instr/grid[/shape], e.g. tir/rulecode/space")
    ap.add_argument("--sample", type=int, default=0, help="draw N uids to look at the marginal distributions")
    a = ap.parse_args()
    cfg = json.load(open(a.config_json))

    if a.show:
        parts = a.show.split("/")
        s = Setting(parts[0], parts[1], parts[2], len(parts) > 3, "xhigh")
        print("=" * 70, f"\nSYSTEM ({s!r})\n", "=" * 70, sep="")
        print(s.system(cfg, a.max_turns) or "(no system message)")
        print("=" * 70, "\nUSER (canonical toy task)\n", "=" * 70, sep="")
        print(s.user(pp.CANON_TRAIN, pp.CANON_TEST))
    elif a.sample:
        c = collections.Counter()
        for i in range(a.sample):
            s = draw(f"task{i:06d}", 0)
            c[s.mode] += 1
            c[f"  {s.instr}"] += 1
            c[f"    grid={s.grid_fmt}"] += 1
            c[f"    shape={s.label_shape}"] += 1
            c[f"    effort={s.effort}"] += 1
        for k, v in c.most_common():
            print(f"{v/a.sample*100:6.2f}%  {k}")
    else:
        print(f"{'setting':44s} {'parser':16s} {'sys chars':>9s}  sha256[:16]")
        for s in all_settings():
            sysp = s.system(cfg, a.max_turns)
            print(f"{s.mode + '/' + s.pack.name:44s} {s.pack.parse.__name__:16s} "
                  f"{len(sysp):9d}  {s.fingerprint(cfg, a.max_turns)[:16]}")
