"""ARC3 single-level RL reward with in-group efficiency ranks (prepare_rl ``--arc3-reward rank``, the orchestrator's
group_live and wandb_sync's train reward share this one definition).

Per group of episodes of one level::

    R_i = s_i (1 - f (rank_actions * r_act_i + rank_tokens * r_tok_i))

- ``s_i`` is the episode's outcome score: g2seed ``1[cleared] min(1, (kappa b / a)^2)`` (run_arc3_level), 0/1 for the
  older arc3 level episodes. A cleared episode has ``s > 0``, an uncleared one ``s = 0``.
- ``r_act`` / ``r_tok`` rank the group's cleared episodes by counted actions (RESET counts) and by ``tokens_at_clear``
  (the proxy's output tokens at the clear): average rank for ties, ``(rank - 1) / (m - 1)`` in [0, 1], 0 = fewest. Over
  the m cleared episodes each mean is exactly 1/2. Fewer than 2 cleared: r = 0 (nothing to rank against).
- ``f`` scales the efficiency term for the whole group:
  - ``rank_warmup_versions`` W: a group launched at policy version ``< W`` (the version the orchestrator dispatched it at,
    not the trainer's step when it is trained) has ``f = 0``, i.e. ``R = s`` (Kimi k1.5: a length penalty early in
    training slows learning).
  - otherwise ``f = rho`` with ``rank_pass_scale`` 1, ``rho`` = the cleared fraction of the episodes given (prepare: the
    ones it kept), so a hard group (2/24 cleared) has almost no efficiency term and a reliably cleared one the full term
    (DDCA, arXiv 2602.02099; GRPO-lambda); ``f = 1`` with ``rank_pass_scale`` 0.
- ``rank_actions + rank_tokens < 1`` and ``f <= 1``: a cleared episode keeps ``R >= (1 - rank_actions - rank_tokens) s > 0``;
  an uncleared one is 0.

v6 (10-27, user): rank_actions 0.3, rank_tokens 0.2, pass scale 0, warmup 0. Offline check on the 70 v5 / v5k groups still
on disk (s recomputed with kappa 0.7): of 713 cleared pairs where the one with more RESETs also took more actions, it wins
on R in 24 (3.4%, by >= 1.02x fewer tokens at the clear); the score alone never lets it win, and the within-group Spearman
of RESETs vs R over the cleared stays at -0.47.
v6b (10-27, user): the same weights with pass scale 1 and warmup 5.
"""

from __future__ import annotations

import math

#: prepare_rl ``--arc3-<name>`` / orchestrator config ``arc3_<name>``: all are required, no defaults
PARAMS = ("rank_actions", "rank_tokens", "rank_pass_scale", "rank_warmup_versions")


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _plain_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def check_params(*, rank_actions: float, rank_tokens: float, rank_pass_scale: int, rank_warmup_versions: int) -> None:
    """``rank_actions >= 0``, ``rank_tokens >= 0``, ``rank_actions + rank_tokens < 1`` (finite numbers),
    ``rank_pass_scale`` the int 0 or 1, ``rank_warmup_versions`` an int >= 0; raise otherwise."""
    vals = {"rank_actions": rank_actions, "rank_tokens": rank_tokens}
    if not all(_number(v) for v in vals.values()):
        raise ValueError(f"arc3 rank: parameters must be finite numbers: {vals}")
    if not (rank_actions >= 0 and rank_tokens >= 0 and rank_actions + rank_tokens < 1):
        raise ValueError(f"arc3 rank: want rank_actions >= 0, rank_tokens >= 0 and rank_actions + rank_tokens < 1: {vals}")
    if not (_plain_int(rank_pass_scale) and rank_pass_scale in (0, 1)):
        raise ValueError(f"arc3 rank: rank_pass_scale must be the int 0 or 1, not {rank_pass_scale!r}")
    if not (_plain_int(rank_warmup_versions) and rank_warmup_versions >= 0):
        raise ValueError(f"arc3 rank: rank_warmup_versions must be an int >= 0, not {rank_warmup_versions!r}")


def ranks01(values: list) -> list[float]:
    """Average ranks (ties share the mean of their positions) scaled to [0, 1], 0 = smallest; needs >= 2 values."""
    m = len(values)
    if m < 2:
        raise ValueError(f"arc3 rank: ranking needs >= 2 values, got {m}")
    order = sorted(range(m), key=lambda i: values[i])
    out = [0.0] * m
    i = 0
    while i < m:
        j = i
        while j + 1 < m and values[order[j + 1]] == values[order[i]]:
            j += 1
        for k in range(i, j + 1):
            out[order[k]] = (i + j) / 2 / (m - 1)
        i = j + 1
    return out


def group_rewards(scores, correct, actions, tokens_at_clear, *, rank_actions: float, rank_tokens: float,
                  rank_pass_scale: int, rank_warmup_versions: int, launch_version: int | None
                  ) -> tuple[list[float], list[float], list[float], float]:
    """One group's episodes -> ``(R, r_act, r_tok, f)``, lists in input order (r = 0 for uncleared episodes and when fewer
    than 2 cleared; ``f`` the efficiency factor of the module docstring). ``launch_version`` is the policy version the
    group was dispatched at: an int >= 0 whenever ``rank_warmup_versions > 0`` (None is refused then; with no warmup it
    is not read). Checks: equal lengths and >= 1 episode; score in [0, 1]; correct a bool with ``correct == (score > 0)``;
    a cleared episode has actions >= 1 and tokens_at_clear an int > 0, an uncleared one has tokens_at_clear None (its
    actions are not ranked)."""
    check_params(rank_actions=rank_actions, rank_tokens=rank_tokens, rank_pass_scale=rank_pass_scale,
                 rank_warmup_versions=rank_warmup_versions)
    if rank_warmup_versions > 0 and not (_plain_int(launch_version) and launch_version >= 0):
        raise ValueError(f"arc3 rank: rank_warmup_versions {rank_warmup_versions} needs the group's launch version (an "
                         f"int >= 0), got {launch_version!r}")
    n = len(scores)
    if n < 1 or not len(correct) == len(actions) == len(tokens_at_clear) == n:
        raise ValueError(f"arc3 rank: scores / correct / actions / tokens_at_clear want equal lengths >= 1 "
                         f"({len(scores)}, {len(correct)}, {len(actions)}, {len(tokens_at_clear)})")
    for i, (s, c, a, t) in enumerate(zip(scores, correct, actions, tokens_at_clear)):
        if not (_number(s) and 0 <= s <= 1) or not isinstance(c, bool) or c != (s > 0):
            raise ValueError(f"arc3 rank: episode {i} score={s!r} correct={c!r}: score must be in [0, 1] and correct a "
                             "bool equal to score > 0")
        if c and not (_plain_int(a) and a >= 1 and _plain_int(t) and t > 0):
            raise ValueError(f"arc3 rank: cleared episode {i} has actions={a!r} tokens_at_clear={t!r} (want ints >= 1)")
        if not c and t is not None:
            raise ValueError(f"arc3 rank: uncleared episode {i} has tokens_at_clear={t!r} (want None)")
    cleared = [i for i in range(n) if correct[i]]
    r_act, r_tok = [0.0] * n, [0.0] * n
    if len(cleared) >= 2:
        for out, vals in ((r_act, [actions[i] for i in cleared]), (r_tok, [tokens_at_clear[i] for i in cleared])):
            for i, r in zip(cleared, ranks01(vals)):
                out[i] = r
    if rank_warmup_versions > 0 and launch_version < rank_warmup_versions:
        f = 0.0
    else:
        f = len(cleared) / n if rank_pass_scale else 1.0
    rewards = [float(s) * (1.0 - f * (rank_actions * ra + rank_tokens * rt)) for s, ra, rt in zip(scores, r_act, r_tok)]
    return rewards, r_act, r_tok, f
