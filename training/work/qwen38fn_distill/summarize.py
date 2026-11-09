"""Collect shards into per-axis yield and token statistics. Read-only, can be viewed mid-run.

Three numbers to read together; any one alone misleads:

- **fit** (`answered*`): fraction that finished and gave an answer. Under the 40,960 hard budget this is the main loss.
- **acc**: accuracy. **The denominator is everything**, not just the finished ones -- truncated equals wrong and cannot
  be removed from the denominator.
- **gen**: generated tokens. Wrong chains are generally longer than right ones (2.29× on DSv4), so two arms with similar
  acc but very different gen differ a lot in cost.

Mid-run statistics have **selection bias**: short ones finish first. For unbiased numbers wait for the sweep to finish,
or at least until the in-flight share is small enough (the first 248 records of the DSv4 batch showed 83.9%,
converging to 65.1%).
"""

from __future__ import annotations

import argparse
import collections
import glob
import statistics as st

import pyarrow.parquet as pq

AXES = ("mode", "instr", "grid_fmt", "effort", "label_shape")


def pct(vals, p):
    v = sorted(vals)
    return v[min(len(v) - 1, int(len(v) * p))] if v else 0


def block(title, groups, total_n):
    print(f"\n=== {title} ===")
    print(f"{'':22s} {'n':>6s} {'fit%':>6s} {'acc%':>6s} {'strict%':>7s} "
          f"{'trunc%':>7s} {'gen p50':>8s} {'gen p90':>8s} {'ids p50':>8s} {'tok/ok':>9s}")
    for key, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        n = len(rs)
        fit = sum(r["fit"] for r in rs)
        ok = sum(r["correct"] for r in rs)
        gen = [r["n_completion"] for r in rs]
        print(f"{str(key):22s} {n:6d} {fit/n*100:6.1f} {ok/n*100:6.1f} "
              f"{sum(r['strict'] for r in rs)/n*100:7.1f} "
              f"{sum(r['trunc'] for r in rs)/n*100:7.1f} "
              f"{pct(gen,.5):8,d} {pct(gen,.9):8,d} "
              f"{pct([r['n_ids'] for r in rs],.5):8,d} "
              f"{sum(gen)//max(ok,1):9,d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="directory holding part-*.parquet")
    ap.add_argument("--by", default="mode,instr,grid_fmt,effort",
                    help=f"comma-separated, available: {','.join(AXES)}")
    a = ap.parse_args()

    files = sorted(glob.glob(f"{a.out}/part-*.parquet"))
    if not files:
        raise SystemExit(f"{a.out} has no part-*.parquet")
    cols = ["uid", "n_completion", "n_prompt", "stop_reason", "correct", "correct_strict",
            "answer_src", "parsed", "n_turns", "code_train_ok", "code_train_n",
            "code_test_match", "confidence", *AXES]
    rows = []
    n_ids_total = 0
    for f in files:
        t = pq.read_table(f, columns=cols + ["span_end"])
        d = t.to_pydict()
        for i in range(len(d["uid"])):
            n_ids = d["span_end"][i][-1]
            n_ids_total += n_ids
            rows.append({k: d[k][i] for k in cols} | {
                "n_ids": n_ids,
                "fit": d["stop_reason"][i].startswith("answered"),
                "trunc": d["stop_reason"][i] in ("context_cap", "prompt_too_long"),
                "strict": d["correct_strict"][i],
            })

    n = len(rows)
    print(f"{n:,} records / {len(files)} shards   ids total {n_ids_total:,} tokens "
          f"(supervised {sum(r['n_completion'] for r in rows):,})")
    print(f"stop_reason: ", dict(collections.Counter(r["stop_reason"] for r in rows)))
    print(f"answer_src : ", dict(collections.Counter(r["answer_src"] for r in rows)))

    for axis in a.by.split(","):
        g = collections.defaultdict(list)
        for r in rows:
            g[r[axis]].append(r)
        block(axis, g, n)

    cd = [r for r in rows if r["code_train_n"] and r["code_train_n"] > 0]
    if cd:
        full = [r for r in cd if r["code_train_ok"] == r["code_train_n"]]
        print(f"\n=== rulecode ===\n{len(cd):,} records ran transform(): "
              f"train all correct {len(full)/len(cd)*100:.1f}%, "
              f"test match {sum(1 for r in cd if r['code_test_match'] == 1)/len(cd)*100:.1f}%")
        # Which is more accurate, "execution result" or "reported answer" -- the reason the rulecode arm exists.
        agree = [r for r in cd if r["code_test_match"] >= 0]
        if agree:
            code_ok = sum(1 for r in agree if r["code_test_match"] == 1)
            said_ok = sum(1 for r in agree if r["correct"])
            print(f"same {len(agree):,} records: trusting transform() correct {code_ok/len(agree)*100:.1f}%, "
                  f"trusting the reported answer correct {said_ok/len(agree)*100:.1f}%")
        conf = [r["confidence"] for r in cd if r["confidence"] >= 0]
        if conf:
            print(f"CONFIDENCE parsed for {len(conf):,} records, median {st.median(conf):.0f}")


if __name__ == "__main__":
    main()
