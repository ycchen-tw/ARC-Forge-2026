"""Pack whole TITO segments into per-rank bins and group bins into balanced steps.

Every rank must run a forward each step (FSDP collectives), so a step is exactly
``world`` bins. Bins are best-fit-decreasing packed to ``cap`` tokens (segments
are never cut), sorted by size and grouped consecutively so the four bins of a
step have similar token counts; step order is then shuffled. The at most
``world - 1`` leftover bins of an epoch are dropped and reported (evaluation keeps
them as a padded last step, plan_eval_steps).
"""

import random

from arcforge.train.packed_text import plan_bins


def plan_steps(lengths, cap, world, seed):
    if world < 1 or cap < 1:
        raise ValueError('world and cap must be positive')
    bins = plan_bins([int(n) for n in lengths], cap)
    sizes = [sum(int(lengths[i]) for i in b) for b in bins]
    order = sorted(range(len(bins)), key=lambda b: (-sizes[b], min(bins[b])))
    usable = len(order) - len(order) % world
    steps = [[sorted(bins[b]) for b in order[s:s + world]] for s in range(0, usable, world)]
    random.Random(seed).shuffle(steps)
    dropped = [sorted(bins[b]) for b in order[usable:]]
    return steps, dropped


def cyclic_pad(items, world):
    """``items`` repeated cyclically up to a multiple of ``world`` (every rank must forward each step), and per
    position whether it is the item's own place (True) or padding that must not be scored or written."""
    if not items:
        raise ValueError('nothing to distribute over the ranks')
    total = -(-len(items) // world) * world
    return [items[i % len(items)] for i in range(total)], [i < len(items) for i in range(total)]


def plan_eval_steps(lengths, cap, world, seed):
    """plan_steps for evaluation: the < ``world`` leftover bins are kept as a last step padded by repeating
    them. Each step comes with per-rank flags; padded ranks forward but their result must be left out.
    An empty split gives an empty plan (the caller decides whether evaluation was required)."""
    if len(lengths) == 0:
        return []
    steps, tail = plan_steps(lengths, cap, world, seed)
    plan = [(step, [True] * world) for step in steps]
    if tail:
        plan.append(cyclic_pad(tail, world))
    return plan


def step_balance(step, lengths):
    tokens = [sum(int(lengths[i]) for i in b) for b in step]
    return max(tokens) / (sum(tokens) / len(tokens))
