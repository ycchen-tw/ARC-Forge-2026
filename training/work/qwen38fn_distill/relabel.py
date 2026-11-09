"""Bring `stop_reason` of early shards back to the consistent definition. One-off; should never be needed again.

## What it fixes

The first ~1,500 records after the 2026-08-29 launch used the old definition: when generation hit `max_new_tokens`,
the record was marked `answered_in_reasoning` as long as a grid could be backscanned from the reasoning. That
overestimates -- a trajectory cut by the budget is not "finished answering"; that grid is just something the model
wrote halfway through thinking. Measured: 106 of 1,400 records (7.6%) affected. The runner now does "truncation takes
priority"; this script fixes what was written before.

## Why it can be fixed offline (no rerun)

**The truth is in the data itself**: in a cut-off trajectory the last assistant span does not end with `<|im_end|>`.
`ids` / `span_*` / `parsed` / `correct` are all intact; only one string column is wrong. So this is pure relabelling,
not regeneration.

The criterion is EOS rather than `len(ids) == cap`: the two are equivalent in practice (all 316 records gap=0), but
EOS is the semantic criterion -- the turn did not end -- while cap is an implementation detail that breaks if the cap changes.

Written the same way as the runner: write `.tmp` first, then `os.replace()`, so a hard kill leaves no half-written file.
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
from pathlib import Path

import pyarrow as pq_pa
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--zstd-level", type=int, default=12)
    ap.add_argument("--apply", action="store_true", help="without it, only report and do not modify files")
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    im_end = tok.convert_tokens_to_ids("<|im_end|>")

    files = sorted(glob.glob(f"{a.out}/part-*.parquet"))
    if not files:
        raise SystemExit(f"{a.out} has no part-*.parquet")

    moved = collections.Counter()
    touched = 0
    for f in files:
        table = pq.read_table(f)
        d = table.to_pydict()
        stops = list(d["stop_reason"])
        changed = False
        for i in range(len(stops)):
            sr, se, ids = d["span_role"][i], d["span_end"][i], d["ids"][i]
            asst = [k for k in range(len(sr)) if sr[k] == 1]
            if not asst:
                continue
            truncated = ids[se[asst[-1]] - 1] != im_end
            # Change in one direction only: pull "actually cut off but recorded as answered" back to context_cap.
            # The reverse is left alone -- those were context_cap to begin with and cannot be mislabelled.
            if truncated and stops[i].startswith("answered"):
                moved[stops[i]] += 1
                stops[i] = "context_cap"
                changed = True
        if not changed:
            continue
        touched += 1
        if not a.apply:
            continue
        d["stop_reason"] = stops
        out = pq_pa.table({name: d[name] for name in table.schema.names},
                          schema=table.schema)
        tmp = Path(f).with_suffix(".parquet.tmp")
        pq.write_table(out, tmp, compression="zstd", compression_level=a.zstd_level,
                       use_dictionary=False)
        os.replace(tmp, f)

    total = sum(moved.values())
    print(f"{len(files)} shards, {touched} need changes, {total} records in total")
    for k, v in moved.most_common():
        print(f"  {k} → context_cap : {v}")
    print("(--apply not given; report only, files not modified)" if not a.apply else "✓ rewritten")


if __name__ == "__main__":
    main()
