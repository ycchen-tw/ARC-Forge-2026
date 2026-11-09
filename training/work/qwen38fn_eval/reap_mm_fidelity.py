"""How far each router-masked (pruned) model is from the full model on the same ledger segments, by token category.

    python reap_mm_fidelity.py --base <jobs dir, no mask> --cand text=<jobs dir> --cand mm=<jobs dir> ... \
        [--boot 2000] --out <report.json>

Inputs are reap_capture.py ``--logprobs`` job directories of one split: the base run without a router mask and one
run per candidate mask, same jobs. Per job, logprobs.npy holds log P(token p | S[:p]) under the raw distribution
(temperature 1) for p = 1..n-1 and labels.npy the category of each token (reap_runtime.CATEGORIES).

The headline is the ``gen`` tokens: they were sampled by the full model itself, so the mean of
log p_full(x) - log p_pruned(x) over them estimates how much worse the pruned model predicts the full model's own
continuations (a KL-like divergence under the recording's sampling distribution; it cannot be negative in
expectation, and a degenerate model cannot game it the way plain NLL on ARC grids is gamed). Also reported: the same
over gen tokens of segments with media, and over prompt text (read, not sampled: indicative only). Media pad
positions carry no prediction and are skipped.

Candidate vs candidate: per job the gen-token-weighted mean difference, bootstrapped over task groups (95% CI).
"""

import argparse
import json
from pathlib import Path

import numpy as np

CATEGORIES = ("prompt", "gen", "image", "video")
GEN, PROMPT = CATEGORIES.index("gen"), CATEGORIES.index("prompt")


def load(directory, only_groups=None):
    out = {}
    for j in sorted(Path(directory).iterdir()):
        if not (j / "DONE").exists():
            continue
        rec = json.loads((j / "record.json").read_text())
        if only_groups is not None and rec["group"] not in only_groups:
            continue
        out[j.name] = {"lp": np.load(j / "logprobs.npy"), "labels": np.load(j / "labels.npy"), "rec": rec}
    if not out:
        raise RuntimeError(f"{directory}: no finished jobs with logprobs")
    return out


def per_job(base, cand):
    rows = []
    for jid, b in base.items():
        if jid not in cand:
            raise RuntimeError(f"candidate lacks job {jid}")
        c = cand[jid]
        if not np.array_equal(b["labels"], c["labels"]) or b["lp"].shape != c["lp"].shape:
            raise RuntimeError(f"{jid}: base and candidate disagree on the job")
        lab = b["labels"][1:]
        d = b["lp"].astype(np.float64) - c["lp"].astype(np.float64)
        has_media = b["rec"]["n_images"] + b["rec"]["n_videos"] > 0
        row = {"job": jid, "group": b["rec"]["group"], "media": has_media}
        for name, idx in (("gen", GEN), ("prompt", PROMPT)):
            sel = lab == idx
            if not np.isfinite(d[sel]).all():
                raise RuntimeError(f"{jid}: nonfinite logprob difference on {name} tokens")
            row[f"{name}_n"] = int(sel.sum())
            row[f"{name}_sum"] = float(d[sel].sum())
        rows.append(row)
    return rows


def summary(rows, name, where=lambda r: True):
    n = sum(r[f"{name}_n"] for r in rows if where(r))
    s = sum(r[f"{name}_sum"] for r in rows if where(r))
    return {"tokens": n, "mean_nats": s / n if n else None}


def paired(rows_a, rows_b, boot, seed=0):
    by = {}
    for a, b in zip(rows_a, rows_b):
        if a["job"] != b["job"]:
            raise RuntimeError("job order differs")
        g = by.setdefault(a["group"], [0.0, 0])
        g[0] += a["gen_sum"] - b["gen_sum"]
        g[1] += a["gen_n"]
    groups = list(by.values())
    sums, ns = np.array([g[0] for g in groups]), np.array([g[1] for g in groups])
    point = sums.sum() / ns.sum()
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(groups), size=(boot, len(groups)))
    est = sums[idx].sum(1) / ns[idx].sum(1)
    lo, hi = np.quantile(est, [0.025, 0.975])
    return {"diff_mean_nats": float(point), "ci95": [float(lo), float(hi)], "groups": len(groups)}


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--base", required=True)
    p.add_argument("--cand", action="append", required=True, help="name=<jobs dir>")
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--only-groups", type=Path, help="a file of task groups (one a line): score only their jobs")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    only = None if args.only_groups is None else set(args.only_groups.read_text().split())
    base = load(args.base, only)
    cands = {}
    for item in args.cand:
        name, _, d = item.partition("=")
        cands[name] = per_job(base, load(d, only))
    report = {"jobs": len(base), "only_groups": None if only is None else sorted(only), "candidates": {}, "paired": {}}
    for name, rows in cands.items():
        report["candidates"][name] = {"gen": summary(rows, "gen"), "gen_media_segments": summary(rows, "gen", lambda r: r["media"]),
                                      "prompt": summary(rows, "prompt")}
    names = list(cands)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            report["paired"][f"{a}-{b}"] = paired(cands[a], cands[b], args.boot)
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
