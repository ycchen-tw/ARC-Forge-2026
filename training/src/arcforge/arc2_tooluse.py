"""Tool-use term for arc2 RL (2026-10-28): how much an episode's reasoning computed by hand, whether the code it wrote used
the toolkit, and their within-group advantage.

- ``episode_stats`` (run_arc2.py, reads the codex rollout when the episode ends):
  - ``reasoning_chars``: character count of the reasoning.
  - ``hand_chars``: characters of the mechanical lines (coordinate mappings, row strings, per-cell ✓ / ✗, a row's cell
    lists) in hand-calculation runs; a run = consecutive hand-calculation lines (at most 3 lines between them), and only
    runs of ≥ ``HAND_MIN_RUN`` characters count. Line classification as in work/arc_code/arc2/probes/handcalc_20261028/scan.py.
  - ``code_cmds``: exec_command commands that carry Python source (heredoc / ``python3 -c`` or writing a .py file); a bare
    ``python3 check.py`` run does not count (neither rewarded nor penalised).
  - ``toolkit_cmds``: how many of those call a toolkit function (a top-level def in the workspace arc.py's ``__all__``,
    excluding show / doc / render).
- ``tool_advantages`` (prepare_rl and the orchestrator's group_live use the same definition):
  ``q_i = −w_hand · min(hand_chars_i / hand_scale, hand_cap) + w_toolkit · u_i``, u = toolkit_cmds / code_cmds (no code
  written = 0), ``A_tool_i = q_i − mean q`` (sums to 0 within the group; added to the task advantage,
  adv_normalization none).

episode_stats uses only the standard library (run_arc2.py loads it by file path and runs it in the agent runtime's
python); numpy is imported only inside the two advantage functions.
"""
from __future__ import annotations

import ast
import collections
import json
import re
import math
from pathlib import Path

C = r"\(\s*-?\d+\s*,\s*-?\d+\s*\)"
KINDS = [
    ("coord", re.compile(C + r"\s*(?:->|→|maps to|becomes)\s*" + C)),
    ("rowstr", re.compile(r"[\"'`]\d{4,}[\"'`]|\bc\d+\s*=\s*\d\s*,\s*c\d+\s*=\s*\d")),
    ("rowset", re.compile(r"\brow\s*\d+\s*:.*\{[\d, ]*\}.*(?:→|->|vs|extra|missing|same)", re.I)),
    ("cells", re.compile(r"^\s*-?\s*(?:row|r)\s*\d+\s*[:=]\s*(?:cols?\s*)?[\d\.\-–, ]+(?:\(|$)|^\s*-?\s*(?:row|r)\s*\d+\s*[:=]\s*"
                         + C, re.I)),
    ("fate", re.compile(r"(?:✓|✗|\bkept\b|\bdeleted\b|\bremoved\b|\bstays\b|\bsurvives\b).*" + "|" + C + r".*(?:✓|✗)")),
    ("stroke", re.compile(r"\b(?:diagonal|horizontal|vertical)\b.*" + C + r"|" + C + r".*\b(?:diagonal|horizontal|vertical)\b",
                          re.I)),
    ("count", re.compile(r"\b(?:count|recount|how many|total)\b.*\d", re.I)),
    ("xy", re.compile(C)),
]
#: the kinds a single line of code could replace (count / stroke / xy also show up often in rule-thinking passages; they
#: only serve to join runs together)
MECH = frozenset({"coord", "rowstr", "rowset", "cells", "fate"})
HAND_MIN_RUN = 1500
HAND_MAX_GAP = 3
#: tools that print things / show docs do not count as "computing with the toolkit"
NOT_COMPUTE = frozenset({"show", "doc", "render"})
STATS_KEYS = ("reasoning_chars", "hand_chars", "code_cmds", "toolkit_cmds", "rollouts")
PARAMS = ("hand_weight", "toolkit_weight", "hand_scale", "hand_cap")

_PY_RUN = re.compile(r"\bpython3?\b[^\n|;&]*(?:<<|\s-c\s)")
_PY_FILE = re.compile(r"(?:cat|tee)\s*>{1,2}\s*['\"]?[^\s'\"]+\.py\b")


def toolkit_names(helper: Path) -> frozenset[str]:
    """Toolkit function names: the ``__all__`` entries that are top-level ``def``s in the helper itself (classes,
    constants and re-exports such as np / Counter do not count), minus NOT_COMPUTE. Raises when there is no ``__all__``
    or not a single function."""
    tree = ast.parse(Path(helper).read_text())
    names = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            names = [e.value for e in node.value.elts]
    if names is None:
        raise ValueError(f"{helper}: no __all__")
    defs = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
    out = frozenset(n for n in names if n in defs) - NOT_COMPUTE
    if not out:
        raise ValueError(f"{helper}: __all__ names no top-level function")
    return out


def hand_chars(text: str) -> int:
    """Characters of mechanical lines in the hand-calculation runs of a reasoning text (only runs ≥ HAND_MIN_RUN
    characters count)."""
    total, run, mech, gap = 0, 0, 0, 0
    active = False
    for line in text.splitlines():
        kind = next((name for name, rx in KINDS if rx.search(line)), None)
        if kind is not None:
            if not active:
                active, run, mech = True, 0, 0
            run += len(line) + 1
            mech += len(line) + 1 if kind in MECH else 0
            gap = 0
        elif active:
            gap += 1
            run += len(line) + 1
            if gap > HAND_MAX_GAP:
                total += mech if run >= HAND_MIN_RUN else 0
                active, gap = False, 0
    if active:
        total += mech if run >= HAND_MIN_RUN else 0
    return min(total, len(text))          # the last line has no newline


def command_kind(cmd: str, fn_re: re.Pattern) -> tuple[bool, bool]:
    """(carries Python source, calls a toolkit function)."""
    code = bool(_PY_RUN.search(cmd) or _PY_FILE.search(cmd))
    return code, code and bool(fn_re.search(cmd))


def toolkit_regex(names: frozenset[str]) -> re.Pattern:
    return re.compile(r"(?<![\w.])(" + "|".join(sorted(map(re.escape, names))) + r")\s*\(")


def episode_stats(rollouts: list[Path], names: frozenset[str]) -> dict:
    """codex rollout files (every session of one episode) → STATS_KEYS. Raises when there is no rollout file."""
    if not rollouts:
        raise ValueError("episode_stats: no rollout file")
    fn_re = toolkit_regex(names)
    out = collections.Counter(rollouts=len(rollouts))
    for path in rollouts:
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("type") != "response_item":
                continue
            p = rec["payload"]
            if p.get("type") == "reasoning":
                text = "".join(c.get("text", "") for c in (p.get("content") or []) if isinstance(c, dict))
                out["reasoning_chars"] += len(text)
                out["hand_chars"] += hand_chars(text)
            elif p.get("type") == "function_call" and p.get("name") == "exec_command":
                try:
                    cmd = json.loads(p["arguments"]).get("cmd", "")
                except json.JSONDecodeError:      # the model wrote invalid JSON arguments: scan the raw text
                    cmd = p["arguments"]
                code, toolkit = command_kind(cmd if isinstance(cmd, str) else json.dumps(cmd), fn_re)
                out["code_cmds"] += code
                out["toolkit_cmds"] += toolkit
    return {k: int(out[k]) for k in STATS_KEYS}


def check_params(*, hand_weight: float, toolkit_weight: float, hand_scale: float, hand_cap: float) -> None:
    vals = {"hand_weight": hand_weight, "toolkit_weight": toolkit_weight, "hand_scale": hand_scale, "hand_cap": hand_cap}
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in vals.values()):
        raise ValueError(f"tool reward: parameters must be finite numbers: {vals}")
    if not (hand_weight >= 0 and toolkit_weight >= 0 and hand_weight + toolkit_weight > 0 and hand_scale > 0
            and hand_cap > 0):
        raise ValueError(f"tool reward: weights >= 0 and not both 0, hand_scale > 0, hand_cap > 0: {vals}")


def check_stats(stats: dict) -> None:
    if not isinstance(stats, dict) or set(stats) != set(STATS_KEYS):
        raise ValueError(f"tool_use stats must have exactly {STATS_KEYS}: {stats!r}")
    if not all(isinstance(stats[k], int) and not isinstance(stats[k], bool) and stats[k] >= 0 for k in STATS_KEYS):
        raise ValueError(f"tool_use stats must be non-negative ints: {stats!r}")
    if stats["toolkit_cmds"] > stats["code_cmds"] or stats["hand_chars"] > stats["reasoning_chars"]:
        raise ValueError(f"tool_use stats: toolkit_cmds <= code_cmds and hand_chars <= reasoning_chars: {stats!r}")


def episode_score(stats: dict, *, hand_weight: float, toolkit_weight: float, hand_scale: float,
                  hand_cap: float) -> tuple[float, float, float]:
    """(q, hand-calculation term p, toolkit fraction u)."""
    check_stats(stats)
    p = min(stats["hand_chars"] / hand_scale, hand_cap)
    u = stats["toolkit_cmds"] / stats["code_cmds"] if stats["code_cmds"] else 0.0
    return -hand_weight * p + toolkit_weight * u, p, u


def tool_advantages(stats: list[dict], **params):
    """tool_use stats of one group of kept episodes → (A_tool ndarray, info). A_tool = q − mean q; info: means of q / p / u."""
    import numpy as np
    check_params(**params)
    if len(stats) < 2:
        raise ValueError(f"tool reward: a group needs at least 2 episodes ({len(stats)})")
    rows = np.array([episode_score(s, **params) for s in stats], dtype=np.float64)
    q = rows[:, 0]
    return q - q.mean(), {"q_mean": float(q.mean()), "p_mean": float(rows[:, 1].mean()),
                          "u_mean": float(rows[:, 2].mean())}
