"""Probe (not maintained): a keep-set from text REAP of two domains -- ARC-AGI-2 calib jobs plus half of the ARC-AGI-3
games of the diag jobs -- each normalized per layer and summed (equal domain weight).

    python reap_mix_mask.py --calib <jobs dir> --diag <jobs dir> --keep 320 --out <new dir>

The diag games are split by a fixed hash into A (used here) and B (written to ``groups_B.txt``, for scoring with
reap_mm_fidelity.py --only-groups). **diag is arc3 public-25 = bench C: the mask is tainted**; this measures whether a
two-domain calibration beats either single domain on the other domain's held-out games, before collecting clean
ARC-AGI-3 data for it.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from reap_mm_select import load_split, mass, normalized, reap_scores, keep_top

SALT = "reap-mix-v1"


def games_of(directory):
    return sorted({json.loads((j / "record.json").read_text())["group"] for j in Path(directory).iterdir()
                   if (j / "DONE").exists()})


def subset_dir(directory, groups, out):
    """A directory of symlinks to the jobs of these groups (load_split reads a directory)."""
    out.mkdir(parents=True)
    for j in Path(directory).iterdir():
        if (j / "DONE").exists() and json.loads((j / "record.json").read_text())["group"] in groups:
            (out / j.name).symlink_to(j.resolve())
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--calib", required=True)
    p.add_argument("--diag", required=True)
    p.add_argument("--keep", type=int, required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True)
    games = games_of(args.diag)
    order = sorted(games, key=lambda g: hashlib.sha256(f"{SALT}:{g}".encode()).hexdigest())
    a, b = set(order[: (len(order) + 1) // 2]), set(order[(len(order) + 1) // 2:])
    (out / "groups_A.txt").write_text("\n".join(sorted(a)) + "\n")
    (out / "groups_B.txt").write_text("\n".join(sorted(b)) + "\n")
    calib = load_split(args.calib, "token")["acc"]
    diag_a = load_split(subset_dir(args.diag, a, out / "diag_A_jobs"), "token")["acc"]
    diag_b = load_split(subset_dir(args.diag, b, out / "diag_B_jobs"), "token")["acc"]
    score = normalized(reap_scores(calib, "text")) + normalized(reap_scores(diag_a, "text"))
    drop = keep_top(score, args.keep)
    torch.save({"drop": drop, "meta": {"recipe": "norm REAP(text) arc2 calib + norm REAP(text) arc3 games A",
                                       "keep": args.keep, "games_A": sorted(a), "tainted": "arc3 public-25 = bench C"}},
               out / f"reap{args.keep}_mix.pt")
    report = {"games_A": sorted(a), "games_B": sorted(b)}
    for name, acc in (("arc2_calib", calib), ("arc3_A", diag_a), ("arc3_B", diag_b)):
        for view in ("text", "image", "video"):
            sc = reap_scores(acc, view)
            if float(sc.sum()) > 0:
                report[f"{name}/{view}"] = mass(sc, drop)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
