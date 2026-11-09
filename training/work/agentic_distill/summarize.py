#!/usr/bin/env python
"""agentic_distill collection progress and quality: solve rate (overall / stratified by naive-corpus difficulty / per
hyperparameter axis), tokens, segments, pairing with naive on the same task.

    python work/agentic_distill/summarize.py $ARC_RUNS/agentic_distill/pilot-a $ARC_RUNS/agentic_distill/pilot-b \
        [--naive $ARC_RUNS/agentic_distill/lists/naive_qwen38fn.json] [--list-total 47337]
"""
import argparse
import collections
import glob
import json
import statistics as st
from pathlib import Path


def load(run):
    recs = {}
    for f in glob.glob(str(Path(run) / "tasks" / "*.json")):
        d = json.loads(Path(f).read_text())
        recs[d["task_id"]] = d
    return recs


def pct(a, n):
    return f"{100 * a / n:5.1f}% ({a}/{n})" if n else "   -"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--naive", default=str(Path.home() / "work/runs/agentic_distill/lists/naive_qwen38fn.json"))
    ap.add_argument("--list-total", type=int, default=0, help="number of tasks this batch is expected to run (for progress)")
    args = ap.parse_args()
    recs = {}
    for r in args.runs:
        recs.update(load(r))
    if not recs:
        print("no tasks yet")
        return
    naive = json.loads(Path(args.naive).read_text()) if Path(args.naive).exists() else {}
    n = len(recs)
    ok = {t: all(r["correct"] for r in d["results"]) for t, d in recs.items()}
    gen = [d["tok_completion"] for d in recs.values()]
    wall = [d["wall_s"] for d in recs.values()]
    eps = [e for d in recs.values() for e in d["explorers"] + d["refines"]]
    print(f"== {n} tasks" + (f" / {args.list_total} ({100 * n / args.list_total:.1f}%)" if args.list_total else ""))
    print(f"solved (pass@2, all tests): {pct(sum(ok.values()), n)}   "
          f"attempt-1 only: {pct(sum(all(r['correct_attempt1'] for r in d['results']) for d in recs.values()), n)}")
    print(f"gen tok/task: mean {st.mean(gen) / 1e3:.0f}k  p50 {st.median(gen) / 1e3:.0f}k  p90 {sorted(gen)[int(.9 * n)] / 1e3:.0f}k  "
          f"| wall/task mean {st.mean(wall):.0f}s  | peak ctx max {max(d['peak_total'] for d in recs.values()) / 1e3:.1f}k")
    if ok:
        g_ok = [d["tok_completion"] for t, d in recs.items() if ok[t]]
        g_no = [d["tok_completion"] for t, d in recs.items() if not ok[t]]
        print(f"gen tok: solved mean {st.mean(g_ok) / 1e3:.0f}k" + (f", unsolved mean {st.mean(g_no) / 1e3:.0f}k" if g_no else ""))
    stops = collections.Counter(e["stop_reason"] for e in eps)
    print(f"episodes {len(eps)}: stops {dict(stops)}   compactions {sum(e['n_compactions'] for e in eps)} "
          f"(episodes compacted {sum(1 for e in eps if e['n_compactions'])})   "
          f"explorer code passes all train: {pct(sum(1 for d in recs.values() for c in d['checks'][0] if c['train'] and all(x == 'pass' for x in c['train'])), sum(len(d['checks'][0]) for d in recs.values()))}")
    # stratified by naive
    if naive:
        for name, cond in (("naive-failed", lambda t: t in naive and not naive[t]["correct"]),
                           ("naive-solved", lambda t: t in naive and naive[t]["correct"])):
            ids = [t for t in recs if cond(t)]
            if ids:
                print(f"  {name:13s}: {pct(sum(ok[t] for t in ids), len(ids))}   gen tok mean {st.mean(recs[t]['tok_completion'] for t in ids) / 1e3:.0f}k"
                      f"   naive n_completion mean {st.mean(naive[t]['n_completion'] for t in ids) / 1e3:.0f}k")
    # source split (the tasks json's source; old data without it falls back to lists/source_uids.json)
    src_map = {}
    sp = Path(args.naive).parent / "source_uids.json"
    if sp.exists():
        d = json.loads(sp.read_text())
        src_map = {u: "nvarc_training" for u in d["training"]} | {u: "nvarc_full" for u in d["full"]}
    by_src = collections.defaultdict(list)
    for t, d in recs.items():
        by_src[d.get("source") or src_map.get(t, "?")].append(t)
    for name, ids in sorted(by_src.items()):
        print(f"  source {name:15s}: {pct(sum(ok[t] for t in ids), len(ids))}   gen tok mean {st.mean(recs[t]['tok_completion'] for t in ids) / 1e3:.0f}k")
    # per hyperparameter axis
    axes = collections.defaultdict(lambda: collections.defaultdict(list))
    for t, d in recs.items():
        for k, v in (d.get("hparams") or {}).items():
            axes[k][v].append(t)
    for k, table in axes.items():
        parts = []
        for v, ids in sorted(table.items(), key=lambda kv: str(kv[0])):
            parts.append(f"{v}: {pct(sum(ok[t] for t in ids), len(ids))} {st.mean(recs[t]['tok_completion'] for t in ids) / 1e3:.0f}k")
        print(f"  {k:20s} " + " | ".join(parts))
    # segments
    n_seg = 0
    seg_bytes = 0
    for r in args.runs:
        for f in glob.glob(str(Path(r) / "segments" / "*.jsonl.gz")):
            seg_bytes += Path(f).stat().st_size
            n_seg += 1
    if n_seg:
        print(f"segments files {n_seg}, {seg_bytes / 1e6:.0f} MB ({seg_bytes / n_seg / 1e3:.0f} KB/task)")
    fails = 0
    for r in args.runs:
        p = Path(r) / "failures.jsonl"
        if p.exists():
            fails += sum(1 for _ in p.open())
    if fails:
        print(f"!! failures.jsonl: {fails}")


if __name__ == "__main__":
    main()
