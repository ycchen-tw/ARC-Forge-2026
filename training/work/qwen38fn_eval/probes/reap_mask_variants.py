"""Probe (not maintained): variants around the two-domain mix mask (reap_mix_mask.py), each a keep-set to score with
reap_capture.py --router-mask and reap_mm_fidelity.py.

    python reap_mask_variants.py --calib <jobs dir> --diag <jobs dir> --mix <reap_mix_mask.py out dir> --out <new dir>

Same split of the diag games as reap_mix_mask.py (its diag_A_jobs / diag_B_jobs). **diag is arc3 public-25 = bench C:
every mask that uses it is tainted.** First rebuilds mix from the same scores and requires it equal to the mix dir's
mask. Variants (keep 320 unless named):

- mixB: arc2 + games B, the mirror fold (scored on A it is the other half of a 2-fold cross-validation)
- w25 / w75: arc2 weight 0.25 / 0.75 against games A (mix is 0.5)
- media: mix + 0.25 x normalized REAP of the media tokens (images of arc2 calib and A, video of A)
- group: mix with every ARC task / game weighted equally within its domain (reap_mm_select --weight group)
- freq: mix with routing frequency (counts) in place of REAP
- all23: arc2 + every diag game (in-sample on all of them; what one would deploy if bench C did not matter)
- mix256 / mix384: the mix scores at keep 256 / 384
- random: a uniformly random keep-set per layer (seed 0), for the scale of the metric

Criteria beyond REAP (after freq beat it in the first round), same two domains unless named:

- sumg: soft frequency, sum of the routing weights g
- ean: total contribution, sum of g * ||f|| (REAP times the routing count)
- freq_all: freq over every token category (media included)
- freq_arc2: freq of arc2 calib alone (no bench C)
- freq256: freq at keep 256
- ean_all / sumg_all: ean / sumg over every token category (media included)
- ean_all_arc2: ean_all of arc2 calib alone (no bench C)
- ean_all256: ean_all at keep 256
- ean_b25 / ean_b50: sum of g * ||f|| / N^b, between EAN (b=0) and REAP (b=1)
- sumf: sum of ||f||, the output norm without the gate -- the EAN of Jaiswal et al. / the REAP paper; what this
  module calls ean is the gate-weighted EAN, Liu et al.'s (0,1,1)
- ean_b75: sum of g * ||f|| / N^0.75; man / msan / e002: Liu et al.'s (1,0,1) / (1,0,2) / (0,0,2)
- ean_gen / ean_b50_gen: ean_all / ean_b50 over the gen tokens only (the fidelity metric's tokens; gen and prompt
  keep-sets overlap only 73%); ean_b50_256: ean_b50 at keep 256

``--only name,...`` writes just those masks; ``--sweep mix,ean_all --keeps 128,192,...`` writes each family cut at
every keep count (``<family>_k<keep>``) and nothing else.
"""

import argparse
import json
from pathlib import Path

import torch
from reap_mm_select import (
    CATEGORIES,
    COUNTS,
    FIELDS,
    SUM_GF,
    VIEWS,
    keep_top,
    load_split,
    mass,
    normalized,
    reap_scores,
)


def field_sum(acc, field, view):
    return acc[:, field, [CATEGORIES.index(c) for c in VIEWS[view]]].sum(1)


def counts(acc, view):
    return field_sum(acc, COUNTS, view)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--calib", required=True)
    p.add_argument("--diag", required=True)
    p.add_argument("--mix", required=True)
    p.add_argument("--keep", type=int, default=320)
    p.add_argument("--only", help="comma-separated mask names")
    p.add_argument("--sweep", help="comma-separated score families (mix, ean_all) to cut at every --keeps count")
    p.add_argument("--keeps", help="comma-separated keep counts for --sweep")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out, mix_dir = Path(args.out), Path(args.mix)
    out.mkdir(parents=True)
    (out / "masks").mkdir()

    tok = {"arc2": load_split(args.calib, "token")["acc"], "A": load_split(mix_dir / "diag_A_jobs", "token")["acc"],
           "B": load_split(mix_dir / "diag_B_jobs", "token")["acc"], "all23": load_split(args.diag, "token")["acc"]}
    grp = {"arc2": load_split(args.calib, "group")["acc"], "A": load_split(mix_dir / "diag_A_jobs", "group")["acc"]}
    text = {k: normalized(reap_scores(v, "text")) for k, v in tok.items()}

    mix = 0.5 * text["arc2"] + 0.5 * text["A"]
    ref = torch.load(mix_dir / f"reap{args.keep}_mix.pt")["drop"]
    if not torch.equal(keep_top(mix, args.keep), ref):
        raise RuntimeError("rebuilt mix differs from the mix dir's mask")
    media = 0.5 * normalized(reap_scores(tok["arc2"] + tok["A"], "image")) + 0.5 * normalized(reap_scores(tok["A"], "video"))
    gen = torch.Generator().manual_seed(0)
    n_layers, n_experts = mix.shape
    rand = torch.stack([torch.randperm(n_experts, generator=gen) for _ in range(n_layers)]).double()
    sum_g = FIELDS.index("sum_g")

    def two(fn):  # equal-weight sum of the two domains' per-layer normalized scores
        return 0.5 * normalized(fn(tok["arc2"])) + 0.5 * normalized(fn(tok["A"]))

    freq = two(lambda a: counts(a, "text"))
    ean_all = two(lambda a: field_sum(a, SUM_GF, "all"))
    recipes = {  # name: (scores, keep, recipe)
        "mixB": (0.5 * text["arc2"] + 0.5 * text["B"], args.keep, "norm REAP(text) arc2 + games B"),
        "w25": (0.25 * text["arc2"] + 0.75 * text["A"], args.keep, "0.25 arc2 + 0.75 games A"),
        "w75": (0.75 * text["arc2"] + 0.25 * text["A"], args.keep, "0.75 arc2 + 0.25 games A"),
        "media": (mix + 0.25 * media, args.keep, "mix + 0.25 norm REAP(image arc2+A, video A)"),
        "group": (0.5 * normalized(reap_scores(grp["arc2"], "text")) + 0.5 * normalized(reap_scores(grp["A"], "text")),
                  args.keep, "mix, every task / game weighted equally"),
        "freq": (freq, args.keep, "mix with routing counts instead of REAP"),
        "all23": (0.5 * text["arc2"] + 0.5 * text["all23"], args.keep, "arc2 + all 23 diag games"),
        "mix256": (mix, 256, "mix scores, keep 256"),
        "mix384": (mix, 384, "mix scores, keep 384"),
        "random": (rand, args.keep, "uniform random per layer, seed 0"),
        "sumg": (two(lambda a: field_sum(a, sum_g, "text")), args.keep, "two domains, sum of routing weights g"),
        "ean": (two(lambda a: field_sum(a, SUM_GF, "text")), args.keep, "two domains, sum of g * ||f||"),
        "freq_all": (two(lambda a: counts(a, "all")), args.keep, "two domains, routing counts of every category"),
        "freq_arc2": (normalized(counts(tok["arc2"], "text")), args.keep, "arc2 calib alone, routing counts"),
        "freq256": (freq, 256, "freq scores, keep 256"),
        "ean_all": (ean_all, args.keep, "two domains, sum of g * ||f|| over every category"),
        "sumg_all": (two(lambda a: field_sum(a, sum_g, "all")), args.keep, "two domains, sum of g over every category"),
        "ean_all_arc2": (normalized(field_sum(tok["arc2"], SUM_GF, "all")), args.keep,
                         "arc2 calib alone, sum of g * ||f|| over every category"),
        "ean_all256": (ean_all, 256, "ean_all scores, keep 256"),
        "ean_b25": (two(lambda a: field_sum(a, SUM_GF, "all") / counts(a, "all").clamp_min(1) ** 0.25), args.keep,
                    "two domains, sum of g * ||f|| / N^0.25 (between EAN b=0 and REAP b=1)"),
        "ean_b50": (two(lambda a: field_sum(a, SUM_GF, "all") / counts(a, "all").clamp_min(1) ** 0.5), args.keep,
                    "two domains, sum of g * ||f|| / N^0.5"),
        "sumf": (two(lambda a: field_sum(a, FIELDS.index("sum_f"), "all")), args.keep,
                 "two domains, sum of ||f|| (EAN of Jaiswal et al.: no gate), Liu (0,0,1)"),
        "ean_b75": (two(lambda a: field_sum(a, SUM_GF, "all") / counts(a, "all").clamp_min(1) ** 0.75), args.keep,
                    "two domains, sum of g * ||f|| / N^0.75"),
        "man": (two(lambda a: field_sum(a, FIELDS.index("sum_f"), "all") / counts(a, "all").clamp_min(1)), args.keep,
                "two domains, MAN = sum ||f|| / N, Liu (1,0,1)"),
        "msan": (two(lambda a: field_sum(a, FIELDS.index("sum_f2"), "all") / counts(a, "all").clamp_min(1)), args.keep,
                 "two domains, MSAN = sum ||f||^2 / N, Liu (1,0,2)"),
        "e002": (two(lambda a: field_sum(a, FIELDS.index("sum_f2"), "all")), args.keep,
                 "two domains, sum ||f||^2, Liu (0,0,2)"),
        "ean_gen": (two(lambda a: field_sum(a, SUM_GF, "gen")), args.keep,
                    "two domains, sum of g * ||f|| over the sampled (gen) tokens only"),
        "ean_b50_gen": (two(lambda a: field_sum(a, SUM_GF, "gen") / counts(a, "gen").clamp_min(1) ** 0.5), args.keep,
                        "two domains, sum of g * ||f|| / N^0.5 over the gen tokens only"),
        "ean_b50_256": (two(lambda a: field_sum(a, SUM_GF, "all") / counts(a, "all").clamp_min(1) ** 0.5), 256,
                        "ean_b50 scores, keep 256"),
    }
    if args.sweep:  # only the sweep's masks, named <family>_k<keep>
        families = {"mix": mix, "ean_all": ean_all}
        recipes = {f"{fam}_k{k}": (families[fam], k, f"{fam} scores, keep {k}")
                   for fam in args.sweep.split(",") for k in map(int, args.keeps.split(","))}
    if args.only:
        names = args.only.split(",")
        unknown = set(names) - set(recipes)
        if unknown:
            raise RuntimeError(f"unknown masks {sorted(unknown)}")
        recipes = {n: recipes[n] for n in names}
    report = {"mix_rebuilt_equal": True, "masks": {}}
    mix_drop = ref
    for name, (scores, keep, recipe) in recipes.items():
        drop = keep_top(scores, keep)
        if not ((~drop).sum(1) == keep).all():
            raise RuntimeError(f"{name} does not keep {keep} in every layer")
        tainted = None if name in ("freq_arc2", "ean_all_arc2", "random") else "arc3 public-25 = bench C"
        torch.save({"drop": drop, "meta": {"recipe": recipe, "keep": keep, "tainted": tainted}},
                   out / "masks" / f"{name}.pt")
        row = {"recipe": recipe, "keep": keep,
               "overlap_with_mix_pct": float(((~drop) & (~mix_drop)).sum()) / float((~mix_drop).sum()) * 100}
        for split, view in (("arc2", "text"), ("arc2", "image"), ("A", "text"), ("B", "text"), ("B", "image"),
                            ("B", "video")):
            row[f"{split}/{view}"] = mass(reap_scores(tok[split], view), drop)
        report["masks"][name] = row
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    cols = ("arc2/text", "arc2/image", "A/text", "B/text", "B/image", "B/video")
    print(f"{'mask':8}{'keep':>5}{'ovl':>7}" + "".join(f"{c:>16}" for c in cols))
    for name, r in report["masks"].items():
        print(f"{name:8}{r['keep']:>5}{r['overlap_with_mix_pct']:>7.1f}"
              + "".join(f"{r[c]['mean']:>8.2f} ({r[c]['min']:4.1f})" for c in cols))


if __name__ == "__main__":
    main()
