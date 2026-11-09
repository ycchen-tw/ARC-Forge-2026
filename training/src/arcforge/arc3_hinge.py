"""ARC3 single-level RL reward with start-relative token / action thresholds (prepare_rl ``--arc3-reward hinge``; the
orchestrator's group_live and wandb_sync's train reward share this one definition). RL v7 (10-28, user via the coordinator;
docs/plans/arc3_rl_v7_hard_300k_20261028.md).

Per group of episodes of one level::

    D_i = s_i (hinge_tokens h_t,i + hinge_actions h_a,i)          cleared episodes; 0 for uncleared
    R_i = s_i + f (mean_cleared D - D_i)                          cleared; R_i = 0 uncleared

- ``s_i`` is the episode's outcome score, g2seed ``1[cleared] min(1, (kappa b / a)^2)`` (kappa = sqrt(100/115) is the
  official per-level cap line). A cleared episode has ``s > 0``.
- ``h_t = clip((t - T) / (2 T), 0, 1)`` with t = ``tokens_at_clear``: nothing up to the level's threshold T, the full cost
  at 3 T. ``h_a = clip((a - A) / A, 0, 1)`` with a = counted actions (RESET included): nothing up to A, the full cost at
  2 A; a level without an action threshold (``actions`` null in its frozen record) has ``h_a = 0``.
- The cost is centred among the cleared episodes, so it ranks clears against each other and never moves the cleared mean:
  the group's mean R equals its mean s. A slow clear can get a negative advantage; it keeps ``R >= s (1 - f (ht + ha)) > 0``.
- ``f`` scales the cost for the whole group: 0 while the level is not frozen for this group (see below), else ``rho`` =
  the cleared fraction of the episodes given (prepare: the ones it kept), so a hard group mostly sees clear-vs-fail.

**Frozen thresholds.** T and A come from the level's warm-up groups (dispatched at a policy version below
``hinge_warmup_versions``), frozen per level once all of them are done (work/qwen38fn_rl/hinge_refs.py writes the refs
file; the orchestrator never edits it). A level's record carries ``applies_from_gid``: a group uses the thresholds iff its
gid is >= that (it was dispatched after the freeze); every earlier group -- warm-up or dispatched at version >= W before its
level froze -- has ``f = 0``. The decision depends only on the group's gid and the refs file, so prepare, group_live and
wandb agree whenever the file exists by the time they score the group (frozen records never change: the writer keeps
them byte-identical, readers check ``REFS_SCHEMA`` and every field).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

#: prepare_rl ``--arc3-<name>`` / orchestrator config ``arc3_<name>``: all required, no defaults (``hinge_refs`` is a path)
PARAMS = ("hinge_tokens", "hinge_actions", "hinge_warmup_versions", "hinge_refs")
NUMERIC_PARAMS = ("hinge_tokens", "hinge_actions", "hinge_warmup_versions")
REFS_SCHEMA = "arc3-hinge-refs-v1"
#: a level record in the refs file
LEVEL_KEYS = {"tokens", "actions", "clears", "episodes", "groups", "applies_from_gid", "frozen_utc"}
#: task ids of the g2seed stubs: <game>_L<NN>_<seed label>
TASK_LEVEL = re.compile(r"^(?P<game>[a-z0-9]+)_L(?P<level>[0-9]{2})_")
#: the official per-level cap: 100 (b/a)^2 reaches 115 at a = sqrt(100/115) b
KAPPA_OFFICIAL = math.sqrt(100 / 115)


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _plain_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def check_params(*, hinge_tokens: float, hinge_actions: float, hinge_warmup_versions: int, hinge_refs: str) -> None:
    """``hinge_tokens >= 0``, ``hinge_actions >= 0``, sum < 1 (finite), ``hinge_warmup_versions`` an int >= 1 (the
    thresholds come from the warm-up), ``hinge_refs`` an absolute path string; raise otherwise. The file itself may not
    exist yet (no level frozen)."""
    if not (_number(hinge_tokens) and _number(hinge_actions) and hinge_tokens >= 0 and hinge_actions >= 0
            and hinge_tokens + hinge_actions < 1):
        raise ValueError(f"arc3 hinge: want hinge_tokens >= 0, hinge_actions >= 0, sum < 1 (finite): "
                         f"{hinge_tokens!r}, {hinge_actions!r}")
    if not (_plain_int(hinge_warmup_versions) and hinge_warmup_versions >= 1):
        raise ValueError(f"arc3 hinge: hinge_warmup_versions must be an int >= 1, not {hinge_warmup_versions!r}")
    if not (isinstance(hinge_refs, str) and Path(hinge_refs).is_absolute()):
        raise ValueError(f"arc3 hinge: hinge_refs must be an absolute path, not {hinge_refs!r}")


def level_key(task: str) -> str:
    """``g50t_L04_n003-t07`` -> ``g50t_L04`` (raise for anything else)."""
    m = TASK_LEVEL.match(task)
    if m is None:
        raise ValueError(f"arc3 hinge: task id {task!r} is not <game>_L<NN>_<seed>")
    return f"{m.group('game')}_L{m.group('level')}"


def check_level_record(key: str, rec: dict) -> None:
    if not isinstance(rec, dict) or set(rec) != LEVEL_KEYS:
        raise ValueError(f"arc3 hinge refs: level {key}: keys {sorted(rec) if isinstance(rec, dict) else rec!r}, "
                         f"want {sorted(LEVEL_KEYS)}")
    if not (_plain_int(rec["tokens"]) and rec["tokens"] > 0):
        raise ValueError(f"arc3 hinge refs: level {key}: tokens {rec['tokens']!r} must be an int > 0")
    if rec["actions"] is not None and not (_plain_int(rec["actions"]) and rec["actions"] > 0):
        raise ValueError(f"arc3 hinge refs: level {key}: actions {rec['actions']!r} must be null or an int > 0")
    for k in ("clears", "episodes", "groups", "applies_from_gid"):
        if not (_plain_int(rec[k]) and rec[k] >= 0):
            raise ValueError(f"arc3 hinge refs: level {key}: {k} {rec[k]!r} must be an int >= 0")
    if rec["clears"] > rec["episodes"]:
        raise ValueError(f"arc3 hinge refs: level {key}: {rec['clears']} clears > {rec['episodes']} episodes")


def load_refs(path: str | Path) -> dict[str, dict]:
    """The frozen levels of the refs file ({level key: record}); a missing file = no level frozen yet ({})."""
    p = Path(path)
    if not p.exists():
        return {}
    doc = json.loads(p.read_text())
    if not isinstance(doc, dict) or doc.get("schema") != REFS_SCHEMA or not isinstance(doc.get("levels"), dict):
        raise ValueError(f"{p}: not an {REFS_SCHEMA} file")
    for key, rec in doc["levels"].items():
        if level_key(key + "_x") != key:   # the key itself is <game>_L<NN>
            raise ValueError(f"{p}: level key {key!r} is not <game>_L<NN>")
        check_level_record(key, rec)
    return doc["levels"]


def group_factor_level(refs: dict[str, dict], task: str, gid: int, launch_version: int, hinge_warmup_versions: int
                       ) -> dict | None:
    """The level record whose thresholds this group uses, or None (f = 0): the level is frozen and the group's gid is at or
    after the record's ``applies_from_gid``. A group past the boundary must have been dispatched at a version >=
    ``hinge_warmup_versions`` (the writer only freezes once the server is there): raise otherwise."""
    if not (_plain_int(gid) and gid >= 0 and _plain_int(launch_version) and launch_version >= 0):
        raise ValueError(f"arc3 hinge: group gid {gid!r} / launch version {launch_version!r} must be ints >= 0")
    rec = refs.get(level_key(task))
    if rec is None or gid < rec["applies_from_gid"]:
        return None
    if launch_version < hinge_warmup_versions:
        raise ValueError(f"arc3 hinge: group {gid} ({task}) is past its level's freeze (gid {rec['applies_from_gid']}) but "
                         f"was dispatched at version {launch_version} < {hinge_warmup_versions}")
    return rec


def hinge(x: float, threshold: float, span: float) -> float:
    return min(1.0, max(0.0, (x - threshold) / span))


def group_rewards(scores, correct, actions, tokens_at_clear, *, hinge_tokens: float, hinge_actions: float,
                  hinge_warmup_versions: int, hinge_refs: str, task: str, gid: int, launch_version: int,
                  refs: dict[str, dict] | None = None) -> tuple[list[float], list[float], float, dict | None]:
    """One group's episodes -> ``(R, D, f, level record or None)``, lists in input order (D = 0 for uncleared episodes).
    ``refs``: the loaded refs (callers scoring many groups load once); None = read ``hinge_refs`` here.
    Checks: equal lengths and >= 1 episode; score in [0, 1]; correct a bool with ``correct == (score > 0)``; a cleared
    episode has actions >= 1 and tokens_at_clear an int > 0, an uncleared one tokens_at_clear None."""
    check_params(hinge_tokens=hinge_tokens, hinge_actions=hinge_actions, hinge_warmup_versions=hinge_warmup_versions,
                 hinge_refs=hinge_refs)
    n = len(scores)
    if n < 1 or not len(correct) == len(actions) == len(tokens_at_clear) == n:
        raise ValueError(f"arc3 hinge: scores / correct / actions / tokens_at_clear want equal lengths >= 1 "
                         f"({len(scores)}, {len(correct)}, {len(actions)}, {len(tokens_at_clear)})")
    for i, (s, c, a, t) in enumerate(zip(scores, correct, actions, tokens_at_clear)):
        if not (_number(s) and 0 <= s <= 1) or not isinstance(c, bool) or c != (s > 0):
            raise ValueError(f"arc3 hinge: episode {i} score={s!r} correct={c!r}: score must be in [0, 1] and correct a "
                             "bool equal to score > 0")
        if c and not (_plain_int(a) and a >= 1 and _plain_int(t) and t > 0):
            raise ValueError(f"arc3 hinge: cleared episode {i} has actions={a!r} tokens_at_clear={t!r} (want ints >= 1)")
        if not c and t is not None:
            raise ValueError(f"arc3 hinge: uncleared episode {i} has tokens_at_clear={t!r} (want None)")
    rec = group_factor_level(load_refs(hinge_refs) if refs is None else refs, task, gid, launch_version,
                             hinge_warmup_versions)
    cleared = [i for i in range(n) if correct[i]]
    cost = [0.0] * n
    if rec is not None:
        for i in cleared:
            h_t = hinge(tokens_at_clear[i], rec["tokens"], 2 * rec["tokens"])
            h_a = 0.0 if rec["actions"] is None else hinge(actions[i], rec["actions"], rec["actions"])
            cost[i] = float(scores[i]) * (hinge_tokens * h_t + hinge_actions * h_a)
    f = len(cleared) / n if rec is not None else 0.0
    mean_cost = sum(cost[i] for i in cleared) / len(cleared) if cleared else 0.0
    rewards = [float(scores[i]) + f * (mean_cost - cost[i]) if correct[i] else 0.0 for i in range(n)]
    return rewards, cost, f, rec
