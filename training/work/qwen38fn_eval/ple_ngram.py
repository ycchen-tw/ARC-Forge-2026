#!/usr/bin/env python3
"""Occupancy statistics and hot-table construction for the Qwen4 PLE n-gram table.

    python ple_ngram.py tokenize  fn-fp8-xhigh fn-fp8-low fn-reap256-medium
    python ple_ngram.py stats     --arm fn-fp8-xhigh
    python ple_ngram.py build-hot --frac2 0.05 --frac3 0.20 --out hot.npz

**No GPU, no server needed.** An n-gram id is a pure function of token ids,
so the whole hash can be recomputed on CPU with the checkpoint's parameters.

## Why this exists

The 51.2B PLE table takes 51.2 GB of host RAM, which fits neither on Kaggle L4×4 nor on a single RTX 6000 Pro.
The table is an ideal random hash (measured collision count within 0.01% of the random expectation), so low-rank / PQ /
magnitude are all ineffective; **the only dimension left is usage frequency**: ARC touches only 9.3% of the table.

This measures "how many lookup hits a fixed table size keeps"; that curve is what the deployment decision rests on.
Earlier only "how much space keeping the observed buckets takes" was computed, which is a different question.

## Hash contract (read from sglang/srt/models/qwen4_exp.py)

    mix2 = cur * m0 ^ prev1 * m1
    mix3 = mix2 ^ prev2 * m2
    id_h = mix % p_h + offset_h      # h = 0..7 uses mix2, 8..15 uses mix3

m comes from splitmix64(seed=1234), p_h is the (h+1)-th prime after 19999999.
**The 8 heads of one order share the same mix and differ only in the modulus** -- so an unseen n-gram misses on all
8 heads at once: a structural loss of the whole 1280-dim slice, not sparse dropout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from arcforge import paths  # noqa: E402

RUNS = str(paths.runs() / "qwen38fn_eval")
CACHE = str(paths.runs() / "ple_prune")
MODEL = str(paths.models() / "Qwen3.8-Flash-Next-FP8")

VOCAB = 248320  # text_config.vocab_size
NGRAM_SIZE = 3
HEADS_PER_NGRAM = 8
NGRAM_HEADS = (NGRAM_SIZE - 1) * HEADS_PER_NGRAM
NGRAM_VOCAB_BASE = 20000000
HEAD_DIM = 160  # ple_embed_dim 2560 / 16 head
ROW_BYTES = HEAD_DIM  # fp8 e4m3, one byte per dim per row

_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_SEED = 1234  # config.json has no seed field → qwen4_exp.py's default


def _splitmix64(x: int) -> int:
    x = (x + _GAMMA) & _MASK64
    x = ((x ^ (x >> 30)) * _M1) & _MASK64
    x = ((x ^ (x >> 27)) * _M2) & _MASK64
    return (x ^ (x >> 31)) & _MASK64


def multipliers() -> list[int]:
    """Replica of Qwen4ExpNGramEmbedding._build_layer_multipliers (ple_layer_index=0)."""
    half = max(1, (((1 << 63) - 1) // VOCAB) // 2)
    out = [
        2 * (_splitmix64((_SEED + _GAMMA * (i + 1)) & _MASK64) % half) + 1
        for i in range(NGRAM_SIZE)
    ]
    # The multiplication must not overflow int64, otherwise the hash disagrees with the kernel.
    assert all(m * (VOCAB - 1) < (1 << 63) for m in out)
    return out


def primes() -> list[int]:
    import sympy

    p, out = NGRAM_VOCAB_BASE - 1, []
    for _ in range(NGRAM_HEADS):
        p = int(sympy.nextprime(p))
        out.append(p)
    return out


MULT = multipliers()


# ── tokenize ────────────────────────────────────────────────────────────────


def _doc_of(d: dict) -> str:
    """Rebuild a transcript into an approximate server-side token stream.

    No chat template: where the template puts assistant reasoning_content / tool_calls changes with the transformers
    version, and what is wanted here is a measure **stable across reruns**. Hand-assembling
    <|im_start|>role\\n…<|im_end|>\\n differs only in a few control tokens.
    **This is an approximation, not a bit-exact reproduction of the server's token stream** -- every hit rate computed
    from it carries this error.
    """
    parts = []
    for m in d["messages"]:
        body = []
        if m.get("reasoning_content"):
            body.append(m["reasoning_content"])
        if m.get("content"):
            body.append(m["content"])
        for tc in m.get("tool_calls") or []:
            body.append(json.dumps(tc, ensure_ascii=False))
        parts.append(
            "<|im_start|>%s\n%s<|im_end|>\n" % (m.get("role", "?"), "\n".join(body))
        )
    return "".join(parts)


def cmd_tokenize(args) -> None:
    import glob

    from transformers import AutoTokenizer

    os.makedirs(CACHE, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    for arm in args.arms:
        files = sorted(glob.glob(f"{RUNS}/{arm}/tx/*.json"))
        if not files:
            raise SystemExit(f"✗ no transcripts: {RUNS}/{arm}/tx")
        chunks, bounds, off = [], [], 0
        for i, f in enumerate(files):
            ids = tok(_doc_of(json.load(open(f))), add_special_tokens=False)[
                "input_ids"
            ]
            chunks.append(np.asarray(ids, dtype=np.int32))
            bounds.append((os.path.basename(f).split("_")[0], off, off + len(ids)))
            off += len(ids)
            if (i + 1) % 100 == 0:
                print(f"  {arm}: {i+1}/{len(files)}  {off/1e6:.1f}M tok", flush=True)
        np.save(f"{CACHE}/{arm}.ids.npy", np.concatenate(chunks))
        json.dump(bounds, open(f"{CACHE}/{arm}.bounds.json", "w"))
        print(f"✓ {arm}: {len(files)} tx, {off/1e6:.2f}M token")


# ── shared: computing mix ───────────────────────────────────────────────────


def mixes_by_task(arm: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Returns {task: (mix2, mix3)}. n-grams are taken only within one transcript."""
    ids = np.load(f"{CACHE}/{arm}.ids.npy")
    bounds = json.load(open(f"{CACHE}/{arm}.bounds.json"))
    m0, m1, m2 = MULT
    a = ids.astype(np.int64)
    x0 = a * m0
    acc: dict[str, list] = {}
    for t, s, e in bounds:
        if e - s < NGRAM_SIZE:
            continue
        v2 = np.bitwise_xor(x0[s + 2 : e], a[s + 1 : e - 1] * m1)
        q = acc.setdefault(t, ([], []))
        q[0].append(v2)
        q[1].append(np.bitwise_xor(v2, a[s : e - 2] * m2))
    return {t: (np.concatenate(v[0]), np.concatenate(v[1])) for t, v in acc.items()}


def split_tasks(tasks: list[str], holdout: int, seed: int) -> tuple[list, list]:
    """Split by **task**, not by rollout -- two attempts at the same task share many n-grams, and a split by rollout
    inflates the hit rate to near 100%."""
    perm = np.random.default_rng(seed).permutation(len(tasks))
    return [tasks[i] for i in perm[holdout:]], [tasks[i] for i in perm[:holdout]]


# ── stats ───────────────────────────────────────────────────────────────────


def cmd_stats(args) -> None:
    P = primes()
    print(f"multipliers = {MULT}")
    print(f"primes[0]={P[0]} primes[8]={P[8]} primes[15]={P[15]} Σ={sum(P):,}")

    X = mixes_by_task(args.arm)
    tasks = sorted(X)
    build, held = split_tasks(tasks, args.holdout, args.seed)
    print(f"\n=== {args.arm}: {len(tasks)} tasks (build {len(build)} / held-out {len(held)})")

    for k, name, h0 in ((0, "2-gram", 0), (1, "3-gram", 8)):
        allm = np.concatenate([X[t][k] for t in tasks])
        d = len(np.unique(allm))
        b = np.unique(allm % P[h0])
        exp = P[h0] * (1 - np.exp(-d / P[h0]))
        print(
            f"{name}: {len(allm)/1e6:.2f}M lookup, distinct mix {d:,}"
            f" → head{h0} buckets {len(b):,} = {len(b)/P[h0]:.2%} of the table"
            f" (ideal random hash expects {exp:,.0f})"
        )

    print("\n== pruning buckets by lookup frequency: table size vs held-out hits")
    print("  order | table% |    GB |  build hits | held-out hits")
    for k, name, h0 in ((0, "2-gram", 0), (1, "3-gram", 8)):
        bm = np.concatenate([X[t][k] for t in build]) % P[h0]
        tm = np.concatenate([X[t][k] for t in held]) % P[h0]
        u, c = np.unique(bm, return_counts=True)
        order = u[np.argsort(c)[::-1]]
        for frac in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
            keep = order[: int(len(order) * frac)]
            gb = len(keep) * HEADS_PER_NGRAM * ROW_BYTES / 1e9
            print(
                f" {name} | {len(keep)/P[h0]:>6.3%} | {gb:>5.2f} |"
                f" {np.isin(bm, keep).mean():>11.2%} | {np.isin(tm, keep).mean():>13.2%}"
            )

    print("\n== closed-vocabulary construction: keep only n-grams made of top-N tokens (coverage is guaranteed, not extrapolated)")
    ids = np.load(f"{CACHE}/{args.arm}.ids.npy")
    bounds = json.load(open(f"{CACHE}/{args.arm}.bounds.json"))
    order = np.argsort(np.bincount(ids, minlength=VOCAB))[::-1]
    segs = [(s, e) for _, s, e in bounds if e - s >= NGRAM_SIZE]
    cur = np.concatenate([ids[s + 2 : e] for s, e in segs])
    pv1 = np.concatenate([ids[s + 1 : e - 1] for s, e in segs])
    pv2 = np.concatenate([ids[s : e - 2] for s, e in segs])
    for N in (512, 1024, 2048, 4096):
        hot = np.zeros(VOCAB, bool)
        hot[order[:N]] = True
        r2 = 1 - np.exp(-(N**2) / P[0])
        r3 = 1 - np.exp(-(N**3) / P[8])
        print(
            f" top-{N:>5}: 2-gram reachable buckets {r2:>7.2%} of table / lookup coverage"
            f" {(hot[cur] & hot[pv1]).mean():.2%}"
            f" | 3-gram reachable buckets {r3:>7.2%} of table / lookup coverage"
            f" {(hot[cur] & hot[pv1] & hot[pv2]).mean():.2%}"
        )

    print("\n== shape of the misses: how many heads a held-out lookup hits (the 8 heads share one mix, differing only in the modulus)")
    for k, name, h0 in ((0, "2-gram", 0), (1, "3-gram", 8)):
        bm = np.concatenate([X[t][k] for t in build])
        tm = np.concatenate([X[t][k] for t in held])
        n = np.zeros(len(tm), np.int8)
        for h in range(h0, h0 + HEADS_PER_NGRAM):
            n += np.isin(tm % P[h], np.unique(bm % P[h]))
        c = np.bincount(n, minlength=HEADS_PER_NGRAM + 1)
        mid = c[1:HEADS_PER_NGRAM].sum()
        print(
            f" {name}: 8/8 hit {c[8]/len(tm):.2%} | 0/8 {c[0]/len(tm):.2%}"
            f" | in between (pure collisions, getting another n-gram's row) {mid/len(tm):.2%}"
        )


# ── build-hot ───────────────────────────────────────────────────────────────


def cmd_build_hot(args) -> None:
    P = primes()
    per_arm = {a: mixes_by_task(a) for a in args.arms}
    tasks = sorted(per_arm[args.arms[0]])
    for a in args.arms[1:]:
        if sorted(per_arm[a]) != tasks:
            raise SystemExit(f"✗ {a} task set differs from {args.arms[0]}'s")
    build, held = split_tasks(tasks, args.holdout, args.seed)

    out: dict[str, np.ndarray] = {"n_heads": np.int64(NGRAM_HEADS)}
    rows = 0
    for k, name, h0, frac in (
        (0, "2-gram", 0, args.frac2),
        (1, "3-gram", 8, args.frac3),
    ):
        mix = np.concatenate([per_arm[a][t][k] for a in args.arms for t in build])
        tm = np.concatenate([per_arm[args.arms[0]][t][k] for t in held])
        for h in range(h0, h0 + HEADS_PER_NGRAM):
            p = P[h]
            keep = int(p * frac)
            cnt = np.bincount(mix % p, minlength=p)
            nz = int((cnt > 0).sum())
            if keep > nz:
                raise SystemExit(
                    f"✗ head{h}: want to keep {keep:,} buckets but the build corpus touches only {nz:,}. "
                    f"Lower --frac or add corpus -- padding the table with cold buckets is pure waste of space."
                )
            hot = np.sort(np.argpartition(cnt, -keep)[-keep:]).astype(np.int32)
            out[f"head{h}"] = hot
            rows += keep
            if h == h0:  # report one head's hits per order; all 16 heads one by one is too slow and nearly identical
                print(
                    f" {name} head{h}: keep {keep:,}/{p:,} buckets ({keep/p:.3%}, "
                    f"build non-empty {nz/p:.2%}) → held-out lookup hits"
                    f" {np.isin(tm % p, hot).mean():.2%}"
                )
    # The moduli must be stored too: checking only the head count, a table from another layer would silently treat the
    # buckets at the upper edge as cold.
    out["vocab_sizes"] = np.array(P, dtype=np.int64)
    out["build_tasks"] = np.array(build)
    out["holdout_tasks"] = np.array(held)
    out["arms"] = np.array(args.arms)
    np.savez(args.out, **out)
    print(
        f"✓ {args.out}: {rows:,} lookup rows = {rows*ROW_BYTES/1e9:.2f} GB"
        f" (full table {sum(P)*ROW_BYTES/1e9:.2f} GB, {sum(P)/rows:.1f}× compression)"
    )
    if args.holdout_list:
        with open(args.holdout_list, "w") as f:
            f.write("\n".join(held) + "\n")
        print(f"✓ held-out task list → {args.holdout_list} (acceptance runs may only use these {len(held)} tasks)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tokenize")
    t.add_argument("arms", nargs="+")
    t.set_defaults(func=cmd_tokenize)

    s = sub.add_parser("stats")
    s.add_argument("--arm", default="fn-fp8-xhigh")
    s.add_argument("--holdout", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_stats)

    b = sub.add_parser("build-hot")
    b.add_argument(
        "--arms",
        nargs="+",
        default=["fn-fp8-xhigh", "fn-fp8-low", "fn-reap256-medium"],
    )
    b.add_argument("--frac2", type=float, default=0.05)
    b.add_argument("--frac3", type=float, default=0.20)
    b.add_argument("--holdout", type=int, default=20)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    b.add_argument("--holdout-list")
    b.set_defaults(func=cmd_build_hot)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
