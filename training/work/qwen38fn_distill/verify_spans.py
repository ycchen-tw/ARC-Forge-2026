"""Verify span and loss-mask boundaries of shards already written. Runs on a sample, any time.

`loss mask = span_role == 1`. If this is wrong there are **no symptoms at all** -- trajectory lengths are right, the
decoded text reads normally, training converges -- the student is just taught things it cannot see at inference, or
conversely, tokens it should learn are left out of the loss. So verify record by record; do not trust it from reading
the code.

Seven checks, each matching one "invisible" way of breaking:

1. **Spans tile the whole sequence without gaps**: `span_start[0]==0`, `span_end[-1]==len(ids)`, each span's
   end equals the next span's start. A gap means some tokens are neither prompt nor assistant, and those tokens are
   silently excluded from the mask.
2. **Valid role sequence**: `0 (1 2)* 1?` -- prompt first, then assistant and tool alternate.
   CoT only has `0 1`.
3. **Assistant spans end with `<|im_end|>`**: the turn boundary. Without it the student never learns to stop.
4. **Assistant spans do not start with `<think>`**: `<think>\\n` is **a cue that is fed in** and belongs to the prompt
   or tool segment. Putting it in the loss teaches the model to generate a token it will always be fed.
5. **Prompt / tool spans end with the cue**: `<|im_start|>assistant\\n<think>\\n`.
   An assistant span must start at the first token after the cue, with no overlap and no gap.
6. **Tool spans start with `\\n<|im_start|>user`**: tool feedback is not model-generated and must be masked out.
7. **n_completion equals the total length of assistant spans**: the parquet statistics column agrees with the trajectory.
"""

from __future__ import annotations

import argparse
import glob
import random
import sys

import pyarrow.parquet as pq

CUE = "<|im_start|>assistant\n<think>\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    IM_END = tok.convert_tokens_to_ids("<|im_end|>")
    THINK = tok.convert_tokens_to_ids("<think>")
    END_OF_TEXT = tok.convert_tokens_to_ids("<|endoftext|>")
    cue_ids = tok(CUE, add_special_tokens=False)["input_ids"]

    files = sorted(glob.glob(f"{a.out}/part-*.parquet"))
    if not files:
        raise SystemExit(f"{a.out} has no part-*.parquet")
    rng = random.Random(a.seed)
    picked = files if len(files) <= 8 else rng.sample(files, 8)

    rows = []
    for f in picked:
        d = pq.read_table(f, columns=["uid", "mode", "stop_reason", "ids", "span_start",
                                      "span_end", "span_role", "n_completion"]).to_pydict()
        for i in range(len(d["uid"])):
            rows.append({k: d[k][i] for k in d})
    if len(rows) > a.sample:
        rows = rng.sample(rows, a.sample)

    fails = {k: [] for k in
             ("tile", "roles", "asst_eos", "asst_not_think", "cue", "tool_head", "n_comp")}
    for r in rows:
        ids, ss, se, sr = r["ids"], r["span_start"], r["span_end"], r["span_role"]
        uid = f"{r['uid']}({r['stop_reason']})"

        if ss[0] != 0 or se[-1] != len(ids) or any(se[k] != ss[k + 1] for k in range(len(ss) - 1)):
            fails["tile"].append(uid)
            continue
        seq = "".join(str(x) for x in sr)
        ok_roles = seq[0] == "0" and all(
            (sr[k] == 1 and sr[k - 1] in (0, 2)) or (sr[k] == 2 and sr[k - 1] == 1)
            for k in range(1, len(sr)))
        if not ok_roles:
            fails["roles"].append(f"{uid} seq={seq}")
        if sum(se[k] - ss[k] for k in range(len(sr)) if sr[k] == 1) != r["n_completion"]:
            fails["n_comp"].append(uid)

        for k in range(len(sr)):
            seg = ids[ss[k]:se[k]]
            if not seg:
                fails["tile"].append(f"{uid} span{k} is empty")
                continue
            if sr[k] == 1:
                # The last span may end without <|im_end|>, in two legitimate cases:
                #   1. truncated by the 40,960 budget (context_cap / turn_cap / prompt_too_long)
                #   2. **the model itself emitted `<|endoftext|>` instead of `<|im_end|>`** -- degenerate generation
                #      that stops far from the cap; stop_reason will be malformed. Measured: 1 of 44,300 records
                #      (`1c0d0a4b_7fe24cdd`, 7,025 tokens from the cap).
                #      This one was found by a downstream session's full scan; my sample missed it.
                # A **middle** assistant span must always end with <|im_end|>, no exceptions --
                # it marks the turn boundary, and its absence means the splice is broken.
                is_last = k == len(sr) - 1
                ok_end = seg[-1] == IM_END or (
                    is_last and (r["stop_reason"] in ("context_cap", "turn_cap",
                                                     "prompt_too_long")
                                 or seg[-1] == END_OF_TEXT))
                if not ok_end:
                    fails["asst_eos"].append(
                        f"{uid} span{k} end={seg[-1]}"
                        f" ({tok.decode([seg[-1]], skip_special_tokens=False)!r})")
                if seg[0] == THINK:
                    fails["asst_not_think"].append(f"{uid} span{k} starts with <think> (cue leaked into the loss)")
            else:
                if seg[-len(cue_ids):] != cue_ids:
                    fails["cue"].append(
                        f"{uid} span{k}(role={sr[k]}) does not end with the cue: {tok.decode(seg[-12:])!r}")
                if sr[k] == 2 and tok.decode(seg[:3], skip_special_tokens=False) \
                        != "\n<|im_start|>user":
                    fails["tool_head"].append(
                        f"{uid} span{k} head={tok.decode(seg[:3], skip_special_tokens=False)!r}")

    names = {
        "tile": "1. spans tile the whole sequence without gaps",
        "roles": "2. valid role sequence 0 (1 2)* 1?",
        "asst_eos": "3. assistant spans end with <|im_end|>",
        "asst_not_think": "4. assistant spans contain no fed-in <think> cue",
        "cue": "5. prompt/tool spans end with the assistant cue",
        "tool_head": "6. tool spans start with \\n<|im_start|>user",
        "n_comp": "7. n_completion == total assistant span length",
    }
    print(f"sampled {len(rows)} records / {len(picked)} shards")
    bad = 0
    for key, label in names.items():
        v = fails[key]
        print(f"{'✓' if not v else '✗'} {label}"
              + ("" if not v else f"  {len(v)} violations, e.g.: {v[0]}"))
        bad += bool(v)
    n_tir = sum(1 for r in rows if r["mode"] == "tir")
    print(f"\n({n_tir} tir / {len(rows) - n_tir} cot; "
          f"supervised tokens {sum(r['n_completion'] for r in rows) / sum(len(r['ids']) for r in rows) * 100:.1f}%)")
    if bad:
        sys.exit(1)
    print("✓ all pass")


if __name__ == "__main__":
    main()
