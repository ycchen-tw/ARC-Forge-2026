"""Fit the dense10240 MTP FFN on rows sampled from the served MTP MoE input (patch/mtpd_runtime.py collection).

    python fit_dense.py --rows <collect/rows> --out <fit.json> [--layer 0 --source M|T<L>] [--device cuda] [--experts 16]

--layer i --source S: the rows of draft layer i (mtp.layers.<i>, the collector's layer tag; 10-11 dense-run rows carry
no tag and are layer 0), whose MoE is S: M = the MTP layer (mtp.layers.0.mlp.* of the checkpoint, 320 kept by
inputs/mtp320.pt), T<L> = the target's layer L (model.language_model.layers.<L>.mlp.*, 320 kept by the INT3 deploy
manifest), as build_drafts.py --spec places them. Everything below is the same for either source.

Routing is recomputed offline as the server does it (RenormalizeNaive: softmax over the router logits, top-10,
renormalized), from the official FP8 checkpoint's MTP router and FP8 experts (dequantized, FP32 math). The target of
the fit is the ORIGINAL 512-expert MTP's routed output y (the drafter the official checkpoint ships); the served
baseline is the 320-expert pruning of it, reported alongside.

Gate first: the served MoE block's own outputs (check-*.pt, routed 320 + gated shared expert, from the live server)
must be reproduced by the offline 320-expert reimplementation; a wrong router/expert/shared reading stops here.
Then, as dense_init.fit_layer (work/qwen38fn_dflash): S = top-16 experts by router mass on the fit rows, alpha_LS =
argmin sum ||y - sum_{e in S} alpha_e f_e(x)||^2 with FP64 normal equations; validation rel errors for uniform 1/16,
alpha_LS and the routed sum restricted to S, per forward mode, plus the error of the whole MLP output (+ shared).
Rows of forward mode EXTEND (prompt prefill; their MLP output only matters at the last position) are excluded from
the fit and reported separately.
"""
import argparse
import glob
import json
import os
import re
import time

import torch
from safetensors import safe_open
from torch.nn import functional as F

from build_drafts import DEPLOY_MANIFEST, DEPLOY_MANIFEST_SHA, MASK_SHA, sha256

FP8_SRC = os.path.join(os.environ["ARC_MODELS"], "Qwen3.8-Flash-Next-FP8")
MASK = os.path.join(os.environ["MTPD_RUN"], "inputs", "mtp320.pt")
E, TOPK, I, H, BLOCK = 512, 10, 640, 2560, 128


def source_prefix(source):
    if source == "M":
        return "mtp.layers.0.mlp."
    if re.fullmatch(r"T\d+", source):
        return f"model.language_model.layers.{int(source[1:])}.mlp."
    raise ValueError(f"source {source!r} is not M or T<L>")


def kept320(source):
    """The 320 original expert ids the served draft layer keeps (router rows in the same order)."""
    if source == "M":
        if sha256(MASK) != MASK_SHA:
            raise RuntimeError("mtp320 mask hash differs")
        mask = torch.load(MASK, map_location="cpu", weights_only=False)
        keep = (~mask["drop"][0]).nonzero().flatten().tolist()
    else:
        if sha256(DEPLOY_MANIFEST) != DEPLOY_MANIFEST_SHA:
            raise RuntimeError("INT3 deploy manifest copy differs from the package")
        keep = json.loads(DEPLOY_MANIFEST.read_text())["keep_per_layer"][source[1:]]
    if len(keep) != 320 or len(set(keep)) != 320:
        raise RuntimeError(f"{source}: keep list is not 320 distinct experts")
    return keep


def load_mlp(P, device):
    index = json.load(open(f"{FP8_SRC}/model.safetensors.index.json"))["weight_map"]
    names = [k for k in index if k.startswith(P)]
    t = {}
    for fn in sorted({index[k] for k in names}):
        with safe_open(f"{FP8_SRC}/{fn}", "pt") as f:
            for k in names:
                if index[k] == fn:
                    t[k] = f.get_tensor(k)

    def dq(name):
        w, s = t[name + ".weight"], t[name + ".weight_scale_inv"].float()
        return (w.float() * s.repeat_interleave(BLOCK, 0).repeat_interleave(BLOCK, 1)).to(device)

    gate_up = torch.stack([torch.cat([dq(f"{P}experts.{e}.gate_proj"), dq(f"{P}experts.{e}.up_proj")]) for e in range(E)])
    down = torch.stack([dq(f"{P}experts.{e}.down_proj") for e in range(E)])
    router = t[P + "gate.weight"].to(device)  # BF16, as served (ReplicatedLinear, unquantized)
    if router.shape != (E, H):
        raise RuntimeError(f"{P}gate.weight {tuple(router.shape)}")
    shared = {k: t[P + "shared_expert." + k + ".weight"].to(device).float() for k in ("gate_proj", "up_proj", "down_proj")}
    shared_gate = t[P + "shared_expert_gate.weight"].to(device).float()
    return gate_up, down, router, shared, shared_gate


def route(x, router, allowed=None):
    """top-10 experts and renormalized softmax weights; `allowed` (bool [E]) restricts the router to those experts."""
    logits = F.linear(x.to(router.dtype), router).float()
    if allowed is not None:
        logits = logits.masked_fill(~allowed, float("-inf"))
    probs = logits.softmax(-1)
    w, idx = probs.topk(TOPK, -1)
    return idx, w / w.sum(-1, keepdim=True)


def expert_outputs(gate_up, down, experts, x):
    out = []
    for e in experts:
        g, u = F.linear(x, gate_up[e]).chunk(2, -1)
        out.append(F.linear(F.silu(g) * u, down[e]))
    return torch.stack(out, 1)


def routed_output(gate_up, down, x, idx, w, within=None):
    y = torch.zeros_like(x)
    for e in idx.unique().tolist():
        if within is not None and not within[e]:
            continue
        rows, slot = (idx == e).nonzero(as_tuple=True)
        g, u = F.linear(x[rows], gate_up[e]).chunk(2, -1)
        y.index_add_(0, rows, w[rows, slot, None] * F.linear(F.silu(g) * u, down[e]))
    return y


def shared_output(shared, shared_gate, x):
    s = F.linear(F.silu(F.linear(x, shared["gate_proj"])) * F.linear(x, shared["up_proj"]), shared["down_proj"])
    return torch.sigmoid(F.linear(x, shared_gate)) * s


def rel(a, b):
    return ((a - b).norm() / b.norm()).item()


def check_files(rows_dir, layer):
    """check-<pid>-L<layer>-<mode>.pt; 10-11 dense-run collections (single layer) wrote check-<pid>-<mode>.pt."""
    files = sorted(glob.glob(f"{rows_dir}/check-*.pt"))
    tagged = [f for f in files if re.fullmatch(r"check-\d+-L\d+-\w+\.pt", os.path.basename(f))]
    if tagged:
        if len(tagged) != len(files):
            raise RuntimeError(f"mixed check file names in {rows_dir}")
        files = [f for f in tagged if os.path.basename(f).split("-")[2] == f"L{layer}"]
    elif layer != 0:
        raise RuntimeError(f"{rows_dir} has untagged (single-layer) check files; --layer {layer} needs a tagged collection")
    if not files:
        raise RuntimeError(f"no check files for layer {layer} in {rows_dir}")
    return files


def check_served(rows_dir, layer, gate_up, down, router, shared, shared_gate, keep320, device):
    """The served block's outputs vs the offline reimplementation (320-expert router, as served)."""
    report = {}
    allowed = torch.zeros(E, dtype=torch.bool, device=device)
    allowed[keep320] = True
    for fn in check_files(rows_dir, layer):
        calls = torch.load(fn)
        if any(c.get("layer", 0) != layer for c in calls):
            raise RuntimeError(f"{fn}: calls from another layer")
        x = torch.cat([c["x"] for c in calls]).to(device).float()
        out = torch.cat([c["out"] for c in calls]).to(device).float()
        sh = shared_output(shared, shared_gate, x)
        idx, w = route(x, router, allowed)
        y320 = routed_output(gate_up, down, x, idx, w)
        idx5, w5 = route(x, router)
        y512 = routed_output(gate_up, down, x, idx5, w5)
        g, u = gate_up.chunk(2, 1)
        swapped = routed_output(torch.cat([u, g], 1), down, x, idx, w)
        report[calls[0]["mode"]] = {
            "rows": x.shape[0],
            "rel_err_offline320": rel(y320 + sh, out),
            "rel_err_offline512": rel(y512 + sh, out),
            "rel_err_no_shared": rel(y320, out),
            "rel_err_gate_up_swapped": rel(swapped + sh, out),
        }
    print(json.dumps({"served_check": report}, indent=1), flush=True)
    for mode, r in report.items():
        # 0.08, not 0.05 (2026-10-11 dry run): per-row median 3.9%, the tail is router near-ties (374 of 1236
        # draft-extend rows have a <1% gap between the 10th and 11th expert), FP8 activation quantization explains
        # none of it; the wrong readings sit at 0.30 (512 router), 0.43 (no shared), 0.91 (gate/up swapped).
        if not (r["rel_err_offline320"] < 0.08 and r["rel_err_gate_up_swapped"] > 0.2 and r["rel_err_no_shared"] > 0.05):
            raise RuntimeError(f"offline MoE does not reproduce the served block ({mode}): {r}")
    return report


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--source", default="M")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--validation-every", type=int, default=6)
    ap.add_argument("--chunk", type=int, default=8192)
    a = ap.parse_args()
    t0 = time.time()
    dev = torch.device(a.device)
    P = source_prefix(a.source)
    keep320 = torch.tensor(kept320(a.source), device=dev)
    gate_up, down, router, shared, shared_gate = load_mlp(P, dev)
    served = check_served(a.rows, a.layer, gate_up, down, router, shared, shared_gate, keep320, dev)

    files = sorted(glob.glob(f"{a.rows}/rows-*.pt"))
    xs, modes = [], []
    names = None
    for i, fn in enumerate(files):
        d = torch.load(fn)
        if names is None:
            names = d["modes"]
        elif any(names.get(k, v) != v for k, v in d["modes"].items()):
            raise RuntimeError("mode codes differ between files (more than one server process?)")
        names.update(d["modes"])
        if "layer" in d:
            sel = d["layer"].long() == a.layer
        elif a.layer == 0:
            sel = torch.ones(d["x"].shape[0], dtype=torch.bool)
        else:
            raise RuntimeError(f"{fn} has no layer tag (single-layer collection); --layer {a.layer} needs a tagged one")
        xs.append(d["x"][sel])
        modes.append(d["mode"][sel])
    x_all, mode_all = torch.cat(xs), torch.cat(modes).long()
    # validation: every --validation-every-th block of 1000 consecutive rows (rows are in serving order, many requests
    # interleaved; a block is ~1 s of decoding)
    val_all = (torch.arange(x_all.shape[0]) // 1000) % a.validation_every == a.validation_every - 1
    code = {v: k for k, v in names.items()}
    counts = {code[c]: int((mode_all == c).sum()) for c in mode_all.unique().tolist()}
    print("rows", x_all.shape[0], "files", len(files), "by mode", counts, flush=True)
    extend = names.get("EXTEND")
    use = mode_all != extend if extend is not None else torch.ones_like(val_all)
    fit_rows = (use & ~val_all).nonzero().flatten()
    val_rows = (use & val_all).nonzero().flatten()

    def chunks(rows):
        for part in rows.split(a.chunk):
            x = x_all[part].to(dev).float()
            idx, w = route(x, router)
            yield part, x, idx, w

    mass = torch.zeros(E, dtype=torch.float64, device=dev)
    for _, x, idx, w in chunks(fit_rows):
        mass.index_add_(0, idx.flatten(), w.double().flatten())
    keep = mass.argsort(descending=True, stable=True)[: a.experts]
    within = torch.zeros(E, dtype=torch.bool, device=dev)
    within[keep] = True
    m = a.experts
    gram = torch.zeros(m, m, dtype=torch.float64, device=dev)
    cross = torch.zeros(m, dtype=torch.float64, device=dev)
    for _, x, idx, w in chunks(fit_rows):
        y = routed_output(gate_up, down, x, idx, w)
        fs = expert_outputs(gate_up, down, keep.tolist(), x).double()
        gram += torch.einsum("rih,rjh->ij", fs, fs)
        cross += torch.einsum("rih,rh->i", fs, y.double())
    alpha = torch.linalg.solve(gram, cross)
    alphas = {"uniform": torch.full((m,), 1.0 / m, dtype=torch.float64, device=dev), "least_squares": alpha}
    keys = ("uniform", "least_squares", "routed_within_S", "y", "mlp_uniform", "mlp_least_squares", "mlp")
    sums = {k: {} for k in keys}

    def add(k, mode, v):
        sums[k][mode] = sums[k].get(mode, 0.0) + v

    for part, x, idx, w in chunks(val_rows):
        y = routed_output(gate_up, down, x, idx, w)
        sh = shared_output(shared, shared_gate, x)
        fs = expert_outputs(gate_up, down, keep.tolist(), x)
        md = mode_all[part.to(mode_all.device)].to(dev)
        for c in md.unique().tolist():
            r = md == c
            mode = code[c]
            add("y", mode, y[r].double().square().sum().item())
            add("mlp", mode, (y[r] + sh[r]).double().square().sum().item())
            for name, al in alphas.items():
                yhat = torch.einsum("rih,i->rh", fs[r], al.float())
                add(name, mode, (y[r] - yhat).double().square().sum().item())
                add("mlp_" + name, mode, (y[r] - yhat).double().square().sum().item())
            add("routed_within_S", mode, (y[r] - routed_output(gate_up, down, x[r], idx[r], w[r], within)).double()
                .square().sum().item())

    def ratio(k, denom):
        tot = sum(sums[k].values()) / sum(sums[denom].values())
        return {"all": tot ** 0.5, **{md: (sums[k][md] / sums[denom][md]) ** 0.5 for md in sums[k]}}

    keep_list = keep.tolist()
    kept_set = set(keep320.tolist())
    res = {
        "source": a.source, "layer": a.layer, "prefix": P,
        "rows_dir": os.path.abspath(a.rows), "row_counts_by_mode": counts,
        "fit_rows": len(fit_rows), "validation_rows": len(val_rows), "excluded_modes": ["EXTEND"],
        "experts": keep_list, "alpha_least_squares": alpha.tolist(),
        "alpha_sum": alpha.sum().item(),
        "router_mass_kept_fit": (mass[keep].sum() / mass.sum()).item(),
        "selected_in_kept320": sum(e in kept_set for e in keep_list),
        "gram_condition": (lambda ev: (ev[-1] / ev[0]).item())(torch.linalg.eigvalsh(gram)),
        "rel_err_routed_validation": {k: ratio(k, "y") for k in ("uniform", "least_squares", "routed_within_S")},
        "rel_err_mlp_validation": {k: ratio("mlp_" + k, "mlp") for k in ("uniform", "least_squares")},
        "served_check": served,
        "target": f"routed output of the original 512-expert router of {P} (RenormalizeNaive top-10), FP8 experts dequantized",
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k not in ("served_check",)}, indent=1))


if __name__ == "__main__":
    main()
