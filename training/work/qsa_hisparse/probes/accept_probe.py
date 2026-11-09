"""Speculative accept length of a server on real arc-code prompts: the draft side of a numerics change.

  accept_probe.py <port> <server pid> <sequences.json> <out.json>     (a run_g4_arm.sh client, SPEC=nextn)

For each sequence of tf_sequences.py, its own prompt (tokens[:prompt_len]) is sent SAMPLES times (default 3,
sampling_seed 0..SAMPLES-1, the same in every arm), MAX_NEW (512) new tokens at temperature 1 / top_p 1 (the
deployed defaults), stop at EOS, all at once (CONC, default 12). Per request: completion tokens, verify steps,
correct drafts. ACCEPT_TOP_P sets top_p (default 1.0). Reports accept length = sum completion / sum verify steps (the server log's definition), the draft
accept rate = correct / proposed, and a per-sequence table. Temperature 1 runs diverge between arms; with the same
seeds the per-sequence pairs still pair the prompt.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from smoke_client import call, wait_health

SAMPLES = int(os.environ.get("ACCEPT_SAMPLES", "3"))
TOP_P = float(os.environ.get("ACCEPT_TOP_P", "1.0"))
MAX_NEW = int(os.environ.get("ACCEPT_MAX_NEW", "512"))
CONC = int(os.environ.get("ACCEPT_CONC", "12"))


def one(port: int, name: str, ids: list[int], seed: int) -> dict:
    t0 = time.time()
    r = call(port, "/generate", {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": MAX_NEW, "temperature": 1.0, "top_p": TOP_P, "sampling_seed": seed},
    })
    m = r["meta_info"]
    if "spec_verify_ct" not in m:
        raise SystemExit(f"{name}/{seed}: no spec_verify_ct in meta_info (is this a spec server?): {sorted(m)}")
    return {"seq": name, "seed": seed, "completion": m["completion_tokens"], "verify": m["spec_verify_ct"],
            "correct": m["spec_num_correct_drafts"], "proposed": m["spec_num_proposed_drafts"],
            "hist": m.get("spec_correct_drafts_histogram"),  # steps with 0, 1, 2, ... correct drafts
            "finish": (m.get("finish_reason") or {}).get("type"), "seconds": round(time.time() - t0, 2)}


def run(port: int, server_pid: int, seq_path: str, out: str) -> dict:
    wait_health(port, server_pid)
    seqs = json.load(open(seq_path))
    jobs = [(name, s["tokens"][:s["prompt_len"]], seed) for name, s in seqs.items() for seed in range(SAMPLES)]
    with ThreadPoolExecutor(CONC) as pool:
        rows = list(pool.map(lambda j: one(port, *j), jobs))
    comp = sum(r["completion"] for r in rows)
    ver = sum(r["verify"] for r in rows)
    res = {"requests": len(rows), "completion_tokens": comp, "verify_steps": ver,
           "accept_length": comp / ver, "accept_rate": sum(r["correct"] for r in rows) / sum(r["proposed"] for r in rows),
           "per_request_accept_median": statistics.median(r["completion"] / r["verify"] for r in rows),
           "per_seq": {n: sum(r["completion"] for r in rows if r["seq"] == n) / sum(r["verify"] for r in rows
                                                                                  if r["seq"] == n) for n in seqs},
           "rows": rows, "samples": SAMPLES, "max_new": MAX_NEW, "top_p": TOP_P}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows", "per_seq")}), flush=True)
    return res


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("accept_probe.py <port> <server pid> <sequences.json> <out.json>")
    run(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])
