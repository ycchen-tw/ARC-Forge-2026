"""Prune Qwen3.8-Flash-Next-FP8's MoE experts down to N and write a checkpoint that can be served directly.

## Why not expert_prune/prune_checkpoint.py

That one assumes a **packed-3D** layout (`mlp.experts.gate_up_proj [E, 2*inter, hidden]`),
and **does not handle fp8 scale tensors at all** -- its guard compares only the key set, not shapes,
so scales with an expert dimension are copied as is, no longer match the pruned weights, and become a silent wrong
dequantization.

This model has a **per-expert 2D** layout:

    model.language_model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight
    model.language_model.layers.{L}.mlp.experts.{E}.{gate,up,down}_proj.weight_scale_inv
    model.language_model.layers.{L}.mlp.gate.weight                       [512, 2560]

Pruning therefore becomes "drop whole groups of expert-numbered keys + renumber"; **the scales are named per expert
too, so they follow naturally**, and that hole does not exist with this layout. The router's `gate.weight` takes rows by
the same keep index.

## MTP

`mtp.layers.0.mlp.experts.*` also has its own 512 experts. sglang builds only one global expert-location map and the
draft worker reuses the target's directly; **if the draft's num_experts differs from the target's, the result is an
`IndexError` buried deep in weight loading** (docs/research/sglang_mtp_pruned_moe.md).
So MTP is pruned along with it, and its keep count must equal the main model's. MTP's keep-set comes from its own REAP
mask (saliency key `(mtp, layer_id)`, actual draft-rollout extend / decode);
**do not use the down_proj weight L2 as a proxy** -- that was the fallback of the previous round, when MTP was not
measured.

    python work/qwen38fn_eval/prune_experts.py \
        --src $ARC_MODELS/Qwen3.8-Flash-Next-FP8 \
        --mask masks/reap256_agi3.pt --mtp-mask masks/reap256_agi3_mtp.pt \
        --out $ARC_MODELS/Qwen3.8-Flash-Next-FP8-reap256
    python work/qwen38fn_eval/prune_experts.py --src ... --keep-all --dry-run   # report sizes only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT = re.compile(
    r"^(?P<pre>(?:model\.language_model|mtp)\.layers\.(?P<layer>\d+)\.mlp\.experts)\."
    r"(?P<eid>\d+)\.(?P<rest>.+)$"
)
ROUTER = re.compile(r"^(?:model\.language_model|mtp)\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$")
IS_MTP = re.compile(r"^mtp\.")
# RadixArk-style fused 3D layout: mlp.experts.{gate_up_proj,down_proj} are single tensors with the expert dimension
# inside the tensor, not a number in the key. EXPERT does not match such keys (no `.<eid>.`), so they would be copied
# as ordinary tensors -- a silent wrong dequantization; it must be stopped.
FUSED_MTP = re.compile(r"^mtp\.layers\.\d+\.mlp\.experts\.(down_proj|gate_up_proj)$")


def source_map(src: str) -> dict[str, str]:
    idx = json.load(open(os.path.join(src, "model.safetensors.index.json")))
    return idx["weight_map"]


def scan(src: str):
    """Returns (weight_map, the expert id set of each layer, number of main-model layers)."""
    wm = source_map(src)
    for k in wm:
        if FUSED_MTP.match(k):
            raise RuntimeError(
                f"{k}: this is a fused 3D MTP layout (expert dimension inside the tensor, not a number in the key); "
                "this script only understands the per-expert 2D layout and cannot prune this layout directly")
    per_layer: dict[tuple[bool, int], set[int]] = {}
    for k in wm:
        m = EXPERT.match(k)
        if m:
            key = (bool(IS_MTP.match(k)), int(m.group("layer")))
            per_layer.setdefault(key, set()).add(int(m.group("eid")))
    return wm, per_layer


def check_suffix_consistency(wm: dict, main_layers: list[int]) -> None:
    """The 12 tensors every expert should have (3 proj × 4 scale/weight variants) must be consistent.
    One missing means some scale tensor of some expert was not found; after pruning it fails to load or worse --
    silently uses the wrong scale.
    """
    suffixes: dict[tuple[int, int], set[str]] = {}
    for k in wm:
        m = EXPERT.match(k)
        if not m or IS_MTP.match(k):
            continue
        suffixes.setdefault((int(m.group("layer")), int(m.group("eid"))), set()).add(m.group("rest"))
    ref_key = (main_layers[0], 0)
    ref = suffixes[ref_key]
    print(f"expert suffix set (reference layer {ref_key[0]} expert {ref_key[1]}, {len(ref)} entries): {sorted(ref)}")
    for key, s in suffixes.items():
        if s != ref:
            raise RuntimeError(
                f"layer {key[0]} expert {key[1]} suffix set {sorted(s)} ≠ reference {sorted(ref)}")


def mtp_keep_by_weight_norm(src: str, wm: dict, layer: int, n_keep: int) -> list[int]:
    """MTP keep-set: L2 norm of the down_proj fp8 weight as a proxy criterion."""
    scores: dict[int, float] = {}
    by_file: dict[str, list[tuple[int, str]]] = {}
    for k, f in wm.items():
        m = EXPERT.match(k)
        if m and IS_MTP.match(k) and int(m.group("layer")) == layer \
                and m.group("rest") == "down_proj.weight":
            by_file.setdefault(f, []).append((int(m.group("eid")), k))
    ok_dtypes = (torch.float8_e4m3fn, torch.bfloat16, torch.float16, torch.float32)
    for f, items in by_file.items():
        with safe_open(os.path.join(src, f), framework="pt") as fh:
            for eid, k in items:
                t = fh.get_tensor(k)
                if t.dtype == torch.uint8:
                    raise RuntimeError(
                        f"{k} dtype uint8 (packed FP4 weight) -- the L2 norm is meaningless in a packed format; "
                        "use an unpacked tensor (e.g. weight_scale) as the proxy criterion")
                if t.dtype not in ok_dtypes:
                    raise RuntimeError(f"{k} dtype {t.dtype}: not supported (supported: {ok_dtypes})")
                scores[eid] = float(t.float().norm())
    return sorted(sorted(scores, key=lambda e: -scores[e])[:n_keep])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--mask", default="", help="main-model select_experts.py mask (drop=True)")
    ap.add_argument("--mtp-mask", default="", help="MTP's own REAP mask, shape [n_mtp, E]")
    ap.add_argument(
        "--mtp-weight-norm", action="store_true",
        help="emergency fallback: down_proj L2 for MTP. The normal path is --mtp-mask",
    )
    ap.add_argument("--keep-all", action="store_true", help="identity pruning, to validate the pipeline")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--shard-bytes", type=int, default=40 * 1024**3)
    a = ap.parse_args()

    wm, per_layer = scan(a.src)
    main_layers = sorted(l for is_mtp, l in per_layer if not is_mtp)
    mtp_layers = sorted(l for is_mtp, l in per_layer if is_mtp)
    n_experts = len(per_layer[(False, main_layers[0])])
    print(f"source: {len(wm)} tensors, main model {len(main_layers)} layers × {n_experts} experts, "
          f"MTP {len(mtp_layers)} layers")
    for key, ids in per_layer.items():
        if len(ids) != n_experts or ids != set(range(n_experts)):
            raise RuntimeError(f"{key} expert ids are not the complete 0..{n_experts-1}: {len(ids)} ids")
    check_suffix_consistency(wm, main_layers)

    if a.keep_all:
        keep = {l: list(range(n_experts)) for l in main_layers}
        n_keep = n_experts
    else:
        blob = torch.load(a.mask, map_location="cpu")
        drop = blob["drop"]
        if drop.shape != (len(main_layers), n_experts):
            raise RuntimeError(f"mask shape {tuple(drop.shape)} ≠ ({len(main_layers)}, {n_experts})")
        keep = {l: (~drop[l]).nonzero().flatten().tolist() for l in main_layers}
        sizes = {len(v) for v in keep.values()}
        if len(sizes) != 1:
            raise RuntimeError(f"the keep count must be the same in every layer (sglang limitation), got {sorted(sizes)}")
        n_keep = sizes.pop()
    print(f"keep {n_keep}/{n_experts} experts ({n_keep/n_experts*100:.0f}%)")

    # MTP uses its own REAP mask; its keep count must equal the main model's (sglang has one expert map)
    mtp_keep = {}
    mtp_drop = None
    if not a.dry_run and mtp_layers and a.mtp_mask and not a.keep_all:
        mtp_blob = torch.load(a.mtp_mask, map_location="cpu")
        mtp_drop = mtp_blob["drop"]
        if mtp_drop.shape != (len(mtp_layers), n_experts):
            raise RuntimeError(
                f"mtp mask shape {tuple(mtp_drop.shape)} ≠ "
                f"({len(mtp_layers)}, {n_experts})"
            )
    if not a.dry_run:
        for i, l in enumerate(mtp_layers):
            if a.keep_all:
                mtp_keep[l] = list(range(n_experts))
            elif mtp_drop is not None:
                mtp_keep[l] = (~mtp_drop[i]).nonzero().flatten().tolist()
            elif a.mtp_weight_norm:
                mtp_keep[l] = mtp_keep_by_weight_norm(a.src, wm, l, n_keep)
                print(f"MTP layer {l} keep-set (weight-norm fallback): {len(mtp_keep[l])} experts")
            else:
                raise SystemExit(
                    "MTP needs --mtp-mask (REAP with the same keep count as the main model), "
                    "or --keep-all; no more silently using the weight L2"
                )
            if len(mtp_keep[l]) != n_keep:
                raise RuntimeError(
                    f"MTP layer {l} keep {len(mtp_keep[l])} ≠ main model {n_keep}"
                    " -- sglang's expert-location map is one global map"
                )
            print(f"MTP layer {l} keep-set: {len(mtp_keep[l])} experts")

    # --- plan the output: old key -> new key (None = dropped) ---
    remap: dict[str, str | None] = {}
    for k in wm:
        m = EXPERT.match(k)
        if not m:
            r = ROUTER.match(k)
            remap[k] = k          # the router's rows are sliced later; the key stays
            continue
        is_mtp = bool(IS_MTP.match(k))
        layer, eid = int(m.group("layer")), int(m.group("eid"))
        ks = (mtp_keep.get(layer, list(range(n_experts))) if is_mtp else keep[layer])
        if eid not in ks:
            remap[k] = None
        else:
            remap[k] = f"{m.group('pre')}.{ks.index(eid)}.{m.group('rest')}"
    dropped = sum(1 for v in remap.values() if v is None)
    print(f"drop {dropped} tensors, keep {len(remap)-dropped}")

    if a.dry_run:
        tot = kept = 0
        for f in set(wm.values()):
            tot += os.path.getsize(os.path.join(a.src, f))
        print(f"source total size {tot/1e9:.1f} GB (the exact pruned size is known only after a real run)")
        return

    if not a.out:
        raise SystemExit("a non-dry-run needs --out")
    os.makedirs(a.out, exist_ok=True)

    # --- read shard by shard, write shard by shard ---
    by_file: dict[str, list[str]] = {}
    for k, f in wm.items():
        by_file.setdefault(f, []).append(k)
    out_index: dict[str, str] = {}
    buf: dict[str, torch.Tensor] = {}
    buf_bytes = 0
    shard_i = 0
    total_bytes = 0

    def flush():
        nonlocal buf, buf_bytes, shard_i, total_bytes
        if not buf:
            return
        shard_i += 1
        name = f"model-{shard_i:05d}.safetensors"
        save_file(buf, os.path.join(a.out, name), metadata={"format": "pt"})
        for k in buf:
            out_index[k] = name
        total_bytes += buf_bytes
        print(f"  wrote {name} ({len(buf)} tensors, {buf_bytes/1e9:.1f} GB)", flush=True)
        buf, buf_bytes = {}, 0

    for f in sorted(set(wm.values())):
        with safe_open(os.path.join(a.src, f), framework="pt") as fh:
            for k in sorted(by_file[f]):
                nk = remap[k]
                if nk is None:
                    continue
                t = fh.get_tensor(k)
                r = ROUTER.match(k)
                if r:
                    layer = int(r.group("layer"))
                    ks = mtp_keep[layer] if IS_MTP.match(k) else keep[layer]
                    if t.shape[0] != n_experts:
                        raise RuntimeError(f"{k} dim 0 {t.shape[0]} ≠ {n_experts}")
                    t = t[torch.tensor(ks)].contiguous()
                buf[nk] = t
                buf_bytes += t.numel() * t.element_size()
                if buf_bytes >= a.shard_bytes:
                    flush()
    flush()

    json.dump({"metadata": {"total_size": total_bytes}, "weight_map": out_index},
              open(os.path.join(a.out, "model.safetensors.index.json"), "w"), indent=1)

    # --- config and remaining files ---
    cfg = json.load(open(os.path.join(a.src, "config.json")))
    cfg["text_config"]["num_experts"] = n_keep
    json.dump(cfg, open(os.path.join(a.out, "config.json"), "w"), indent=4)
    for fn in ("generation_config.json", "tokenizer.json", "tokenizer_config.json",
               "vocab.json", "merges.txt", "chat_template.jinja",
               "preprocessor_config.json", "video_preprocessor_config.json",
               "hf_quant_config.json"):
        src_f = os.path.join(a.src, fn)
        if os.path.exists(src_f):
            shutil.copy2(src_f, os.path.join(a.out, fn))
    if cfg.get("quantization_config", {}).get("quant_method") == "modelopt":
        if not os.path.exists(os.path.join(a.out, "hf_quant_config.json")):
            raise RuntimeError(
                "config.json says quant_method=modelopt, but the source has no hf_quant_config.json to copy"
                " -- the serving side would read the mixed-precision checkpoint with default quantization settings, silently using the wrong format")
    json.dump({"source": a.src, "mask": a.mask, "mtp_mask": a.mtp_mask,
               "n_experts_before": n_experts,
               "n_experts_after": n_keep, "keep_per_layer": {str(k): v for k, v in keep.items()},
               "mtp_keep": {str(k): v for k, v in mtp_keep.items()}},
              open(os.path.join(a.out, "prune_manifest.json"), "w"))
    verify(a.src, a.out, n_experts, n_keep)
    print(f"✓ {a.out} ({total_bytes/1e9:.1f} GB, {shard_i} shards)")


def verify(src: str, out: str, n_experts: int, n_keep: int) -> None:
    """Read back the written index and check it item by item. None of these errors would show up at load time.

    Most important: **the shared expert must be untouched**. It is called `mlp.shared_expert.*`, a different pattern
    from the routed `mlp.experts.{E}.*`, so "it naturally does not match, so it is fine" -- but that relies on the
    patterns happening to line up, not on a check. With a model that also numbers the shared expert as `experts.N`
    (which is how sglang fuses it at runtime), it would be pruned or renumbered like an ordinary routed expert, while the
    router's gate.weight does not include it, and the symptom would appear far away. So it is checked explicitly here.
    """
    src_wm = source_map(src)
    out_wm = json.load(open(os.path.join(out, "model.safetensors.index.json")))["weight_map"]
    n_shared_src = sum(1 for k in src_wm if "shared_expert" in k)
    n_shared_out = sum(1 for k in out_wm if "shared_expert" in k)
    if n_shared_src != n_shared_out:
        raise RuntimeError(f"shared expert tensor count changed: {n_shared_src} → {n_shared_out}"
                           " -- it must not be pruned; check whether the EXPERT regex mismatches")
    n_routed_src = sum(1 for k in src_wm if EXPERT.match(k))
    n_routed_out = sum(1 for k in out_wm if EXPERT.match(k))
    want = round(n_routed_src * n_keep / n_experts)
    if n_routed_out != want:
        raise RuntimeError(f"routed expert tensor count {n_routed_out} ≠ expected {want}")
    # each layer's expert numbers must be the contiguous 0..n_keep-1, no holes
    per_layer: dict[tuple[bool, int], set[int]] = {}
    for k in out_wm:
        m = EXPERT.match(k)
        if m:
            per_layer.setdefault((bool(IS_MTP.match(k)), int(m.group("layer"))), set()).add(
                int(m.group("eid")))
    for key, ids in per_layer.items():
        if ids != set(range(n_keep)):
            raise RuntimeError(f"{key} expert numbers are not the contiguous 0..{n_keep-1}: "
                               f"{len(ids)} numbers, missing {sorted(set(range(n_keep)) - ids)[:5]}")
    cfg = json.load(open(os.path.join(out, "config.json")))
    if cfg["text_config"]["num_experts"] != n_keep:
        raise RuntimeError("config num_experts was not updated")
    print(f"  verify passed: shared {n_shared_out} untouched, routed {n_routed_out}, "
          f"{len(per_layer)} layers numbered contiguously, config num_experts={n_keep}")


if __name__ == "__main__":
    main()
