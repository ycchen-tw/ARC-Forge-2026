"""Prompt sets for the MTP dense10240 experiment, in accept_probe.py's sequences format ({name: {tokens, prompt_len}}).

    python make_prompts.py <out dir> [--fit 128] [--eval 64] [--seed 20261011]

Source: qwen38fn_distill/out (FN's own NVARC CoT/TIR traces, raw token ids, text only). Each prompt is a trace cut at a
random token inside one of its model spans (span_role 1), so the server continues the model's own reasoning or tool
use from there. Prefix length is drawn uniform in [4096, 28672] (the cut must leave >= 1 model token after it).
fit (collection of MTP MLP inputs for the alpha fit) and eval (accept length) come from disjoint parquet parts and
disjoint task ids (uid before '_'), so the fitted dense FFN is never scored on the prompts it was fitted on.
"""
import argparse
import glob
import hashlib
import json
import os
import random

import pyarrow.parquet as pq

SRC = os.path.join(os.environ["ARC_RUNS"], "qwen38fn_distill", "out")
MIN_PREFIX, MAX_PREFIX = 4096, 28672


def candidates(parts, rng):
    for path in parts:
        table = pq.read_table(path, columns=["uid", "mode", "ids", "span_start", "span_end", "span_role"])
        rows = table.to_pylist()
        rng.shuffle(rows)
        for r in rows:
            yield path, r


def cut(r, rng, margin=1):
    """A cut inside a model span with prefix length in [MIN_PREFIX, MAX_PREFIX] and >= margin model tokens after it."""
    spans = [(s, e) for s, e, role in zip(r["span_start"], r["span_end"], r["span_role"]) if role == 1]
    lo = [(max(s + 1, MIN_PREFIX), min(e - margin, MAX_PREFIX)) for s, e in spans]
    lo = [(a, b) for a, b in lo if a <= b]
    if not lo:
        return None
    a, b = rng.choice(lo)
    return rng.randint(a, b)


def build(parts, n, rng, exclude_tasks, margin=1):
    out, tasks = {}, set()
    for path, r in candidates(parts, rng):
        task = r["uid"].split("_")[0]
        if task in exclude_tasks or task in tasks:
            continue
        c = cut(r, rng, margin)
        if c is None:
            continue
        ids = r["ids"][:c]
        name = f"{r['uid']}-{r['mode']}-c{c}"
        out[name] = {"tokens": ids, "prompt_len": c, "source": os.path.basename(path), "uid": r["uid"],
                     "mode": r["mode"], "sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest()}
        tasks.add(task)
        if len(out) == n:
            return out, tasks
    raise RuntimeError(f"only {len(out)} of {n} prompts found")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--fit", type=int, default=128)
    ap.add_argument("--eval", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261011)
    # 2026-10-11: eval v1 (margin 1) had a cut right before a turn end, so a request stopped at its first (prefill)
    # token with no verify step; eval v2 keeps >= 64 model tokens after the cut. fit is unchanged (margin 1).
    ap.add_argument("--eval-margin", type=int, default=64)
    a = ap.parse_args()
    parts = sorted(glob.glob(f"{SRC}/part-*.parquet"))
    if len(parts) < 400:
        raise RuntimeError(f"expected the full distill corpus, found {len(parts)} parts")
    rng = random.Random(a.seed)
    fit, fit_tasks = build(parts[: len(parts) // 2], a.fit, rng, set())
    ev, ev_tasks = build(parts[len(parts) // 2:], a.eval, rng, fit_tasks, a.eval_margin)
    if fit_tasks & ev_tasks:
        raise RuntimeError("fit and eval share a task")
    os.makedirs(a.out, exist_ok=True)
    for tag, seqs in (("fit", fit), ("eval", ev)):
        json.dump(seqs, open(f"{a.out}/{tag}.json", "w"))
        lens = sorted(s["prompt_len"] for s in seqs.values())
        modes = {m: sum(s["mode"] == m for s in seqs.values()) for m in {s["mode"] for s in seqs.values()}}
        print(tag, len(seqs), "prompt_len min/median/max", lens[0], lens[len(lens) // 2], lens[-1], "modes", modes)


if __name__ == "__main__":
    main()
