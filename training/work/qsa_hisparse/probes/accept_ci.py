"""Accept length per arm with bootstrap CIs, and differences against a reference arm (accept_probe.py outputs).

  python accept_ci.py --ref <tag> <tag>=<accept.json> ... [--json-out J] [--boot 4000]

Accept length = sum completion / sum verify steps over an arm's requests (the server log's definition). The CI
resamples requests within each prompt (every arm sends the same prompts and seeds), 95% percentile interval. A
difference resamples both arms independently the same way. Per-prompt accept lengths are listed too.
With the per-request correct-drafts histogram (steps with 0..K-1 correct drafts), the acceptance of each draft
position given the previous one: d1 = P(>= 1 correct), d2 = P(>= 2 | >= 1), d3 = P(>= 3 | >= 2), with CIs. With MTP
NEXTN 3, d1 comes from the draft extend and d2, d3 from the two draft decode steps.
"""

from __future__ import annotations

import argparse
import json
import random


def load(path: str) -> tuple[dict, dict]:
    rows = json.load(open(path))["rows"]
    by, hists = {}, {}
    # The server drops trailing empty bins: a request that never had all drafts correct reports K-1 bins
    # (2026-10-11, 1 of 128 rows in two dense arms). Pad to the arm's longest histogram; every histogram must
    # still count exactly the request's verify steps.
    width = max((len(r["hist"]) for r in rows if r.get("hist") is not None), default=0)
    for r in rows:
        by.setdefault(r["seq"], []).append((r["completion"], r["verify"], r["correct"], r["proposed"]))
        if r.get("hist") is not None:
            if sum(r["hist"]) != r["verify"]:
                raise SystemExit(f"{path}: {r['seq']} histogram {r['hist']} does not sum to verify {r['verify']}")
            hists.setdefault(r["seq"], []).append(r["hist"] + [0] * (width - len(r["hist"])))
    return by, hists


def positions(hists: dict, rng: random.Random = None) -> list[float]:
    """[d1, d2, d3, ...] conditional acceptances from summed histograms (resampled within prompts with rng)."""
    picked = [h for v in hists.values() for h in ([v[rng.randrange(len(v))] for _ in v] if rng else v)]
    tot = [sum(col) for col in zip(*picked)]
    at_least = [sum(tot[k:]) for k in range(len(tot))]
    return [at_least[k] / at_least[k - 1] for k in range(1, len(tot))]


def accept(by: dict) -> float:
    return sum(c for v in by.values() for c, _, _, _ in v) / sum(s for v in by.values() for _, s, _, _ in v)


def resample(by: dict, rng: random.Random) -> float:
    comp = ver = 0
    for v in by.values():
        for _ in range(len(v)):
            c, s, _, _ = v[rng.randrange(len(v))]
            comp += c
            ver += s
    return comp / ver


def ci(samples: list[float]) -> list[float]:
    s = sorted(samples)
    return [round(s[int(0.025 * len(s))], 4), round(s[int(0.975 * len(s)) - 1], 4)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--boot", type=int, default=4000)
    ap.add_argument("--json-out")
    a = ap.parse_args()
    arms, hists = {}, {}
    for spec in a.arms:
        tag, path = spec.split("=", 1)
        arms[tag], hists[tag] = load(path)
    seqs = sorted(arms[a.ref])
    for tag, by in arms.items():
        if sorted(by) != seqs:
            raise SystemExit(f"{tag} has prompts {sorted(by)}, {a.ref} has {seqs}")
    rng = random.Random(0)
    boots = {tag: [resample(by, rng) for _ in range(a.boot)] for tag, by in arms.items()}
    out = {"ref": a.ref, "arms": {}}
    for tag, by in arms.items():
        n = sum(len(v) for v in by.values())
        rate = (sum(k for v in by.values() for _, _, k, _ in v) / sum(p for v in by.values() for _, _, _, p in v))
        entry = {"requests": n, "tokens": sum(c for v in by.values() for c, _, _, _ in v),
                 "verify_steps": sum(s for v in by.values() for _, s, _, _ in v),
                 "accept": round(accept(by), 4), "ci95": ci(boots[tag]), "draft_accept_rate": round(rate, 4),
                 "per_prompt": {q: round(sum(c for c, _, _, _ in by[q]) / sum(s for _, s, _, _ in by[q]), 3) for q in seqs}}
        if hists[tag]:
            pos_boot = [positions(hists[tag], rng) for _ in range(min(a.boot, 1000))]
            entry["draft_position_accept"] = [round(x, 4) for x in positions(hists[tag])]
            entry["draft_position_ci95"] = [ci([p[k] for p in pos_boot]) for k in range(len(pos_boot[0]))]
        if tag != a.ref:
            d = [x - y for x, y in zip(boots[tag], boots[a.ref])]
            entry["diff_vs_ref"] = round(accept(by) - accept(arms[a.ref]), 4)
            entry["diff_ci95"] = ci(d)
            entry["prompts_lower_than_ref"] = sum(entry["per_prompt"][q] < round(
                sum(c for c, _, _, _ in arms[a.ref][q]) / sum(s for _, s, _, _ in arms[a.ref][q]), 3) for q in seqs)
        out["arms"][tag] = entry
        print(tag, json.dumps({k: v for k, v in entry.items() if k != "per_prompt"}))
    if a.json_out:
        json.dump(out, open(a.json_out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
