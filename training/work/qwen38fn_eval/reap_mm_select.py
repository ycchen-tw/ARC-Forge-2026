"""Per-modality REAP: aggregate reap_capture.py jobs, build keep-set masks, and report what each mask keeps per token
category -- for the new masks and for the incumbent (e.g. REAP320's prune_manifest.json).

    python reap_mm_select.py --calib <jobs dir> [--heldout <jobs dir>] [--diag <jobs dir>] --keep 320 \
        --incumbent <checkpoint>/prune_manifest.json --out <new dir>

Inputs are reap_capture.py job directories (``<split>-<n>/reap.safetensors``, ``acc`` [layers, fields, categories,
experts]: CATEGORIES prompt / gen / image / video, FIELDS counts / sum_g / sum_f / sum_f2 / sum_gf). Jobs are summed
as they are (token-weighted, the usual REAP), or with ``--weight group`` each ARC task group contributes equally.

REAP of a category set K, per layer and expert: S_j = sum over K of sum_gf / sum over K of counts (the conditional
mean of g * ||f|| over the tokens routed to j), 0 for an expert those tokens never chose. Retained mass of a mask =
sum of S over kept experts / sum over all, per layer; reported as mean and min over layers -- the number that
predicted capability in qwen35_expert_pruning.md (~55-60% is the cliff), here per category.

Masks (``drop`` bool [layers, experts], prune_experts.py's format, plus the recipe in ``meta``):

- ``text``: REAP over prompt + gen of calib (what a text-only calibration gives).
- ``all``: REAP over every category of calib, pooled. Media are ~2% of the tokens, so this is close to ``text``.
- ``mm``: each category's REAP normalized to sum 1 per layer, combined by max over {text, image} -- an expert vital to
  either modality is kept, however rare that modality is in the tokens.
- ``mm_diag`` (only with --diag; **tainted: diag is arc3 public-25 = bench C**): like ``mm`` with diag's image and
  video REAP added to the max.

Writes ``saliency.pt`` (summed cumulants per split and category), ``masks/reap<K>_<name>.pt``, ``report.json`` and
prints the report. Every mask keeps exactly --keep experts in every layer (sglang needs one count).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file

CATEGORIES = ("prompt", "gen", "image", "video")  # work/qwen38fn_dflash/reap_runtime.py
FIELDS = ("counts", "sum_g", "sum_f", "sum_f2", "sum_gf")
COUNTS, SUM_GF = FIELDS.index("counts"), FIELDS.index("sum_gf")
VIEWS = {"text": ("prompt", "gen"), "prompt": ("prompt",), "gen": ("gen",), "image": ("image",),
         "video": ("video",), "all": CATEGORIES}


def load_split(directory, weight):
    jobs = sorted(p for p in Path(directory).iterdir() if p.is_dir() and (p / "DONE").exists())
    if not jobs:
        raise RuntimeError(f"{directory}: no finished jobs")
    records = [json.loads((j / "record.json").read_text()) for j in jobs]
    group_tokens = Counter()
    for r in records:
        group_tokens[r["group"]] += r["num_tokens"]
    acc, tokens = None, torch.zeros(len(CATEGORIES), dtype=torch.float64)
    for j, r in zip(jobs, records):
        blob = load_file(str(j / "reap.safetensors"))
        w = 1.0 if weight == "token" else 1.0 / group_tokens[r["group"]]
        a = blob["acc"].double() * w
        acc = a if acc is None else acc + a
        tokens += blob["tokens"].double()
    return {"acc": acc, "tokens": tokens, "jobs": len(jobs), "groups": len(group_tokens),
            "num_tokens": sum(r["num_tokens"] for r in records)}


def reap_scores(acc, view):
    idx = [CATEGORIES.index(c) for c in VIEWS[view]]
    counts = acc[:, COUNTS, idx].sum(1)
    gf = acc[:, SUM_GF, idx].sum(1)
    return torch.where(counts > 0, gf / counts.clamp_min(1e-30), torch.zeros_like(gf))


def keep_top(scores, keep):
    drop = torch.ones_like(scores, dtype=torch.bool)
    order = torch.argsort(scores, dim=1, descending=True)
    for layer in range(scores.shape[0]):
        drop[layer, order[layer, :keep]] = False
    return drop


def normalized(scores):
    return scores / scores.sum(1, keepdim=True).clamp_min(1e-30)


def mass(scores, drop):
    tot = scores.sum(1)
    kept = torch.where(drop, torch.zeros_like(scores), scores).sum(1)
    m = torch.where(tot > 0, kept / tot.clamp_min(1e-30), torch.full_like(tot, float("nan")))
    return {"mean": float(m.nanmean()) * 100, "min": float(m[~m.isnan()].min()) * 100 if (~m.isnan()).any() else None}


def routing_mass(acc, view, drop):
    idx = [CATEGORIES.index(c) for c in VIEWS[view]]
    counts = acc[:, COUNTS, idx].sum(1)
    return mass(counts, drop)


def incumbent_drop(manifest_path, n_layers, n_experts, keep):
    m = json.loads(Path(manifest_path).read_text())
    if m["n_experts_before"] != n_experts:
        raise RuntimeError(f"{manifest_path}: {m['n_experts_before']} experts before, the saliency has {n_experts}")
    drop = torch.ones(n_layers, n_experts, dtype=torch.bool)
    for layer in range(n_layers):
        k = m["keep_per_layer"][str(layer)]
        if len(k) != m["n_experts_after"]:
            raise RuntimeError(f"{manifest_path}: layer {layer} keeps {len(k)}")
        drop[layer, k] = False
    return drop, m["n_experts_after"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--calib", required=True)
    p.add_argument("--heldout")
    p.add_argument("--diag", help="arc3 public-25 jobs: bench C, tainted")
    p.add_argument("--keep", type=int, required=True)
    p.add_argument("--weight", choices=("token", "group"), default="token")
    p.add_argument("--incumbent", action="append", default=[], help="prune_manifest.json of a deployed mask; repeat")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True)  # never into an existing directory
    (out / "masks").mkdir()

    splits = {"calib": load_split(args.calib, args.weight)}
    if args.heldout:
        splits["heldout"] = load_split(args.heldout, args.weight)
    if args.diag:
        splits["diag"] = load_split(args.diag, args.weight)
    calib = splits["calib"]["acc"]
    n_layers, _, _, n_experts = calib.shape

    masks = {"text": keep_top(reap_scores(calib, "text"), args.keep),
             "all": keep_top(reap_scores(calib, "all"), args.keep)}
    mm = [normalized(reap_scores(calib, v)) for v in ("text", "image")]
    masks["mm"] = keep_top(torch.stack(mm).amax(0), args.keep)
    if "diag" in splits:
        d = splits["diag"]["acc"]
        mmd = mm + [normalized(reap_scores(d, v)) for v in ("image", "video") if reap_scores(d, v).sum() > 0]
        masks["mm_diag"] = keep_top(torch.stack(mmd).amax(0), args.keep)
    for path in args.incumbent:
        drop, k = incumbent_drop(path, n_layers, n_experts, args.keep)
        masks[f"incumbent:{Path(path).parent.name}"] = drop
        if k != args.keep:
            print(f"· incumbent {path} keeps {k}, not {args.keep}")

    recipes = {"text": "REAP(prompt+gen), calib", "all": "REAP(all categories), calib",
               "mm": "max(norm REAP text, norm REAP image), calib",
               "mm_diag": "max(norm REAP text/image calib, image/video diag) -- diag = bench C, tainted"}
    for name, drop in masks.items():
        if name.startswith("incumbent:"):
            continue
        kept = (~drop).sum(1)
        if not (kept == args.keep).all():
            raise RuntimeError(f"mask {name} keeps {kept.unique().tolist()}")
        torch.save({"drop": drop, "meta": {"recipe": recipes[name], "keep": args.keep, "weight": args.weight,
                                           "calib": args.calib, "diag": args.diag if name == "mm_diag" else None}},
                   out / "masks" / f"reap{args.keep}_{name}.pt")

    report = {"keep": args.keep, "weight": args.weight,
              "splits": {s: {"jobs": v["jobs"], "groups": v["groups"], "num_tokens": v["num_tokens"],
                             "tokens_by_category": dict(zip(CATEGORIES, v["tokens"].tolist()))}
                         for s, v in splits.items()},
              "retained_reap_mass": {}, "retained_routing_mass": {}, "overlap_with": {}}
    for name, drop in masks.items():
        report["retained_reap_mass"][name] = {}
        report["retained_routing_mass"][name] = {}
        for s, v in splits.items():
            for view in ("text", "gen", "image", "video", "all"):
                sc = reap_scores(v["acc"], view)
                if float(sc.sum()) == 0:
                    continue
                report["retained_reap_mass"][name][f"{s}/{view}"] = mass(sc, drop)
                report["retained_routing_mass"][name][f"{s}/{view}"] = routing_mass(v["acc"], view, drop)
        report["overlap_with"][name] = {o: float(((~drop) & (~od)).sum()) / float((~od).sum()) * 100
                                        for o, od in masks.items()}
    # how differently the modalities rank the experts (calib)
    t, i = reap_scores(calib, "text"), reap_scores(calib, "image")
    rt = torch.argsort(torch.argsort(t, dim=1), dim=1).double()
    ri = torch.argsort(torch.argsort(i, dim=1), dim=1).double()
    rt, ri = rt - rt.mean(1, keepdim=True), ri - ri.mean(1, keepdim=True)
    rho = (rt * ri).sum(1) / (rt.norm(dim=1) * ri.norm(dim=1))
    report["text_vs_image_rank_corr"] = {"mean": float(rho.mean()), "min": float(rho.min())}
    report["experts_never_routed"] = {s: {view: int((v["acc"][:, COUNTS, [CATEGORIES.index(c) for c in VIEWS[view]]]
                                                     .sum(1) == 0).sum())
                                          for view in ("text", "image", "video")}
                                      for s, v in splits.items()}
    torch.save({"categories": CATEGORIES, "fields": FIELDS, "weight": args.weight,
                **{s: {"acc": v["acc"], "tokens": v["tokens"]} for s, v in splits.items()}}, out / "saliency.pt")
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n")

    views = [k for k in next(iter(report["retained_reap_mass"].values()))]
    print(f"retained REAP mass, keep {args.keep}, mean% (min layer%):")
    print(f"{'mask':36}" + "".join(f"{v:>18}" for v in views))
    for name, row in report["retained_reap_mass"].items():
        print(f"{name:36}" + "".join(f"{row[v]['mean']:>9.2f} ({row[v]['min']:5.1f})" if v in row else f"{'-':>18}"
                                      for v in views))
    print(json.dumps({k: report[k] for k in ("splits", "text_vs_image_rank_corr", "experts_never_routed")}, indent=1))


if __name__ == "__main__":
    main()
