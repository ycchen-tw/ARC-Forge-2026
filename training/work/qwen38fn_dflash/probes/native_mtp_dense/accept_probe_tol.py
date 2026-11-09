"""accept_probe.py (work/qsa_hisparse/probes) with one tolerance: a request that stops at its first token (sampled in
prefill, no verify step) carries no spec_* fields in meta_info; it is recorded as verify 0 instead of aborting the run.
Any other request without spec fields still aborts (a non-spec server). Same arguments and output as accept_probe.py."""
import os
import sys
import time

PROBES = os.path.join(os.environ["ARC_ROOT"], "work", "qsa_hisparse", "probes")
sys.path.insert(0, PROBES)
import accept_probe as ap  # noqa: E402


def one(port, name, ids, seed):
    t0 = time.time()
    r = ap.call(port, "/generate", {
        "input_ids": ids,
        "sampling_params": {"max_new_tokens": ap.MAX_NEW, "temperature": 1.0, "top_p": ap.TOP_P, "sampling_seed": seed},
    })
    m = r["meta_info"]
    finish = (m.get("finish_reason") or {}).get("type")
    if "spec_verify_ct" not in m:
        if m["completion_tokens"] > 1:
            raise SystemExit(f"{name}/{seed}: {m['completion_tokens']} tokens but no spec_verify_ct: {sorted(m)}")
        return {"seq": name, "seed": seed, "completion": m["completion_tokens"], "verify": 0, "correct": 0,
                "proposed": 0, "hist": None, "finish": finish, "seconds": round(time.time() - t0, 2), "no_verify": True}
    return {"seq": name, "seed": seed, "completion": m["completion_tokens"], "verify": m["spec_verify_ct"],
            "correct": m["spec_num_correct_drafts"], "proposed": m["spec_num_proposed_drafts"],
            "hist": m.get("spec_correct_drafts_histogram"), "finish": finish, "seconds": round(time.time() - t0, 2)}


ap.one = one
if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("accept_probe_tol.py <port> <server pid> <sequences.json> <out.json>")
    ap.run(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4])
