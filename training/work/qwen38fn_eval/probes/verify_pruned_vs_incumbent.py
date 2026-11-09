"""Read-only check of a pruned checkpoint against its 512E source and a deployed pruned checkpoint of the same keep.

    python verify_pruned_vs_incumbent.py --src <512E dir> --out <new pruned dir> --incumbent <deployed pruned dir> \
        [--samples 64] [--seed 0]

Independent of prune_experts.py's own verify(). Raises on the first failure:

1. Drop-in: the new checkpoint has exactly the incumbent's tensor names, shapes and dtypes (safetensors headers),
   the same config.json / hf_quant_config.json / tokenizer and processor files -- so every deployment config tuned
   for the incumbent (memory fractions, Kaggle notebooks) applies unchanged. Only the weights and the keep-sets differ.
2. Provenance: for --samples random (layer, new expert id) pairs of the main model and the MTP layer, every tensor of
   that expert is byte-identical to the source expert prune_manifest.json's keep list maps it to; every router row
   (``mlp.gate.weight``) equals the source row of the kept expert, in order.
3. Everything that is not a routed expert or a router is byte-identical to the source.
4. Cross-check against the incumbent (it was cut from the original source, so this also vouches for a local copy of
   the source): sampled experts both keep-sets keep, and the non-expert tensors under --max-cross-mb, byte-identical
   between the new and the incumbent checkpoint.
"""

import argparse
import json
import random
import re
import struct
from pathlib import Path

import torch
from safetensors import safe_open

EXPERT = re.compile(r"^(?P<pre>(?:model\.language_model|mtp)\.layers\.(?P<layer>\d+)\.mlp\.experts)\."
                    r"(?P<eid>\d+)\.(?P<rest>.+)$")
ROUTER = re.compile(r"^(?:model\.language_model|mtp)\.layers\.(?P<layer>\d+)\.mlp\.gate\.weight$")
SAME_FILES = ("config.json", "hf_quant_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
              "preprocessor_config.json", "video_preprocessor_config.json", "generation_config.json")


def weight_map(d):
    return json.loads((Path(d) / "model.safetensors.index.json").read_text())["weight_map"]


def headers(d, wm):
    out = {}
    for f in sorted(set(wm.values())):
        with open(Path(d) / f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            h = json.loads(fh.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                out[k] = (v["dtype"], tuple(v["shape"]))
    return out


class Reader:
    def __init__(self, d):
        self.d, self.wm, self.open = Path(d), weight_map(d), {}

    def get(self, key):
        f = self.wm[key]
        if f not in self.open:
            self.open[f] = safe_open(str(self.d / f), framework="pt")
        return self.open[f].get_tensor(key)


def same(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    return torch.equal(a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--incumbent", required=True)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-cross-mb", type=float, default=256, help="non-expert tensors up to this size are also "
                                                                    "compared with the incumbent's")
    args = p.parse_args()
    out, inc = Path(args.out), Path(args.incumbent)

    for name in SAME_FILES:
        if (out / name).read_bytes() != (inc / name).read_bytes():
            raise RuntimeError(f"{name} differs from the incumbent's")
    out_wm, inc_wm = weight_map(out), weight_map(inc)
    if set(out_wm) != set(inc_wm):
        diff = sorted(set(out_wm) ^ set(inc_wm))
        raise RuntimeError(f"tensor names differ from the incumbent's ({len(diff)}), e.g. {diff[:3]}")
    ho, hi = headers(out, out_wm), headers(inc, inc_wm)
    bad = [k for k in ho if ho[k] != hi[k]]
    if bad:
        raise RuntimeError(f"{len(bad)} tensors differ in dtype/shape from the incumbent's, e.g. {bad[0]}: "
                           f"{ho[bad[0]]} vs {hi[bad[0]]}")
    print(f"✓ drop-in: {len(ho)} tensors, names/dtypes/shapes and {len(SAME_FILES)} files identical to {inc.name}")

    manifest = json.loads((out / "prune_manifest.json").read_text())
    keep = {("main", int(k)): v for k, v in manifest["keep_per_layer"].items()}
    keep.update({("mtp", int(k)): v for k, v in manifest["mtp_keep"].items()})
    src, new = Reader(args.src), Reader(args.out)
    experts = {}
    for k in out_wm:
        m = EXPERT.match(k)
        if m:
            kind = "mtp" if k.startswith("mtp.") else "main"
            experts.setdefault((kind, int(m["layer"]), int(m["eid"])), []).append((m["pre"], m["rest"]))
    rng = random.Random(args.seed)
    picks = rng.sample(sorted(k for k in experts if k[0] == "main"), args.samples)
    picks += [k for k in sorted(experts) if k[0] == "mtp"][:: max(1, len([k for k in experts if k[0] == "mtp"]) // 8)]
    n_t = 0
    for kind, layer, eid in picks:
        src_eid = keep[(kind, layer)][eid]
        for pre, rest in experts[(kind, layer, eid)]:
            if not same(new.get(f"{pre}.{eid}.{rest}"), src.get(f"{pre}.{src_eid}.{rest}")):
                raise RuntimeError(f"{pre}.{eid}.{rest} is not source expert {src_eid}")
            n_t += 1
    print(f"✓ provenance: {len(picks)} experts ({n_t} tensors) byte-identical to their source experts")
    n_r = 0
    for k in out_wm:
        m = ROUTER.match(k)
        if m:
            kind = "mtp" if k.startswith("mtp.") else "main"
            rows = src.get(k)[torch.tensor(keep[(kind, int(m["layer"]))])]
            if not same(new.get(k), rows):
                raise RuntimeError(f"{k} rows are not the source rows of the kept experts")
            n_r += 1
    print(f"✓ routers: {n_r} gate.weight tensors are the source rows of the kept experts")
    n_o = 0
    for k in out_wm:
        if EXPERT.match(k) or ROUTER.match(k):
            continue
        if not same(new.get(k), src.get(k)):
            raise RuntimeError(f"{k} differs from the source")
        n_o += 1
    print(f"✓ the other {n_o} tensors are byte-identical to the source")

    inc_manifest = json.loads((inc / "prune_manifest.json").read_text())
    inc_keep = {("main", int(k)): v for k, v in inc_manifest["keep_per_layer"].items()}
    inc_keep.update({("mtp", int(k)): v for k, v in inc_manifest["mtp_keep"].items()})
    incr = Reader(args.incumbent)
    shared = []
    for (kind, layer), ks in keep.items():
        pos = {s: i for i, s in enumerate(inc_keep[(kind, layer)])}
        shared += [(kind, layer, e, pos[s]) for e, s in enumerate(ks) if s in pos]
    n_c = 0
    for kind, layer, e, e_inc in rng.sample(shared, min(args.samples, len(shared))):
        for pre, rest in experts[(kind, layer, e)]:
            if not same(new.get(f"{pre}.{e}.{rest}"), incr.get(f"{pre}.{e_inc}.{rest}")):
                raise RuntimeError(f"{pre}.{e}.{rest} differs from the incumbent's copy of the same source expert")
            n_c += 1
    small = 0
    for k in out_wm:
        if EXPERT.match(k) or ROUTER.match(k):
            continue
        dtype, shape = ho[k]
        if torch.Size(shape).numel() * torch.empty((), dtype=getattr(torch, {"F8_E4M3": "float8_e4m3fn", "BF16":
                "bfloat16", "F32": "float32", "U8": "uint8", "I8": "int8", "F16": "float16", "I64": "int64",
                "I32": "int32"}[dtype])).element_size() > args.max_cross_mb * 2**20:
            continue
        if not same(new.get(k), incr.get(k)):
            raise RuntimeError(f"{k} differs from the incumbent's")
        small += 1
    print(f"✓ incumbent cross-check: {n_c} tensors of shared experts and {small} non-expert tensors "
          f"(<= {args.max_cross_mb} MB) byte-identical")


if __name__ == "__main__":
    main()
