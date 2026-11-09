"""BF16-carrier "fake-quant" REAP checkpoints for NLL comparison of routed-expert formats on B200.

Template = the pruned FP8 model (prune_experts.py output). Its tensors other than routed
experts and DENSE go into a shared base once (hard links per arm); each arm writes routed
experts as bf16 values of
its quantized weights (``modules_to_not_convert`` gets the expert prefixes, so SGLang uses
the unquantized MoE path) and the dense projections as bf16 or block-FP8.

Expert arms:
  int3-parts:<dir>  GPTQ output of quantize_int3_gptq.py, dequant ``(code - 4) * scale``
  rtn-bf16          Humming's own INT3 g128 RTN of the bf16 weights
  rtn-nvfp4:<dir>   the L4 deployment path: NVFP4 checkpoint -> fp32 -> bf16 -> Humming RTN
Dense: ``--dense fp8`` block-quantizes DENSE (the four groups the L4 stack moves off BF16:
GDN in/out, attention qkv/o) with quantize_nonmoe_fp8.quantize; ``--dense w4`` is the L4
hook's ARC_HC_HUMMING_DENSE=int4:128 (Humming RTN, bf16 carrier); ``--dense-groups`` limits
either to some of gdn_in, gdn_out, qsa_qkv, qsa_o. MTP stays as in the template.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

EXPERT = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate|up|down)_proj\.")
DENSE = re.compile(r"^model\.language_model\.layers\.\d+\.(?:linear_attn\.(?:in_proj_qkv|in_proj_z|out_proj)"
                   r"|self_attn\.(?:q_proj|k_proj|v_proj|o_proj))\.weight$")
GROUPS = {"gdn_in": re.compile(r"\.linear_attn\.(?:in_proj_qkv|in_proj_z)\.weight$"),
          "gdn_out": re.compile(r"\.linear_attn\.out_proj\.weight$"),
          "qsa_qkv": re.compile(r"\.self_attn\.(?:q_proj|k_proj|v_proj)\.weight$"),
          "qsa_o": re.compile(r"\.self_attn\.o_proj\.weight$")}
SHARD = 8 * 1024 ** 3


def write_shards(tensors, out: Path, stem: str, index: dict) -> None:
    buf, size, n = {}, 0, 0

    def flush():
        nonlocal buf, size, n
        if buf:
            n += 1
            name = f"{stem}-{n:03d}.safetensors"
            save_file(buf, str(out / name), metadata={"format": "pt"})
            index.update({k: name for k in buf})
            buf, size = {}, 0
    for k, t in tensors:
        if size + t.numel() * t.element_size() > SHARD:
            flush()
        buf[k] = t.contiguous()
        size += t.numel() * t.element_size()
    flush()


def build_base(template: Path, base: Path) -> None:
    idx = json.loads((template / "model.safetensors.index.json").read_text())["weight_map"]
    base.mkdir(parents=True)
    index: dict[str, str] = {}

    def gen():
        for f in sorted(set(idx.values())):
            with safe_open(template / f, framework="pt") as sf:
                for k in sf.keys():
                    if not EXPERT.match(k) and not DENSE.match(k):
                        yield k, sf.get_tensor(k)
    write_shards(gen(), base, "base", index)
    (base / "index.json").write_text(json.dumps(index, indent=1))


def humming_rtn(w: torch.Tensor, bits: int = 3) -> torch.Tensor:
    """bf16 -> bf16 values of Humming uint<bits> g128 RTN (the load_from_unquantized path)."""
    from humming import dtypes
    from humming.utils.weight import dequantize_weight, quantize_weight
    dtype = {3: dtypes.uint3, 4: dtypes.uint4}[bits]
    q, s, z, g = quantize_weight(w.cuda(), dtype, scale_dtype=dtypes.bfloat16,
                                 group_size=128, pack=False)
    if z is not None or g is not None:
        raise RuntimeError("expected Humming uint without zero point / global scale")
    return dequantize_weight(q, s, None, None, dtype, packed=False).to(torch.bfloat16).cpu()


def dequant_nvfp4(w_u8, scale_e4m3, scale_2):
    """Same as the L4 hook's _dequant_nvfp4."""
    lut = torch.tensor((0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0), device=w_u8.device)
    vals = torch.stack((lut[(w_u8 & 15).long()], lut[(w_u8 >> 4).long()]), -1).reshape(w_u8.shape[0], -1)
    return vals * scale_e4m3.float().repeat_interleave(16, dim=1) * scale_2.float()


_HANDLES: dict[str, dict] = {}


def expert_banks(args, layer: int, keep: torch.Tensor):
    """(gate_up [E, 2I, H], down [E, H, I]) bf16 for the kept experts, in pruned order."""
    kind, _, arg = args.experts.partition(":")
    if kind == "int3-parts":
        from quantize_int3_gptq import dequant_int3
        with safe_open(Path(arg) / f"int3-L{layer:02d}.safetensors", framework="pt") as f:
            if not torch.equal(f.get_tensor("orig_expert_ids").long(), keep):
                raise RuntimeError(f"layer {layer}: parts expert ids differ from the mask")
            return [dequant_int3(f.get_tensor(f"{bank}.codes"), f.get_tensor(f"{bank}.scale"))
                    for bank in ("gate_up", "down")]
    if kind == "rtn-bf16":
        idx = json.loads((args.bf16 / "model.safetensors.index.json").read_text())["weight_map"]
        out = []
        for name in ("gate_up_proj", "down_proj"):
            key = f"model.language_model.layers.{layer}.mlp.experts.{name}"
            with safe_open(args.bf16 / idx[key], framework="pt") as f:
                out.append(humming_rtn(f.get_tensor(key)[keep]))
        return out
    if kind == "rtn-nvfp4":
        src = Path(arg)
        idx = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
        man = json.loads((src / "prune_manifest.json").read_text())
        if man["keep_per_layer"][str(layer)] != keep.tolist():
            raise RuntimeError(f"layer {layer}: NVFP4 checkpoint keeps different experts")

        handles = _HANDLES.setdefault(str(src), {})

        def deq(e, proj):
            base = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}"
            parts = []
            for suffix in (".weight", ".weight_scale", ".weight_scale_2"):
                name = idx[base + suffix]
                if name not in handles:
                    handles[name] = safe_open(src / name, framework="pt")
                parts.append(handles[name].get_tensor(base + suffix).cuda())
            return dequant_nvfp4(*parts)
        n = len(keep)
        w13 = torch.stack([torch.cat([deq(e, "gate_proj"), deq(e, "up_proj")]) for e in range(n)])
        w2 = torch.stack([deq(e, "down_proj") for e in range(n)])
        return humming_rtn(w13.to(torch.bfloat16)), humming_rtn(w2.to(torch.bfloat16))
    raise ValueError(f"unknown --experts {args.experts}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True, help="shared non-expert shards (built once)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mask", type=Path, required=True)
    ap.add_argument("--experts", required=True)
    ap.add_argument("--dense", choices=["bf16", "fp8", "w4"], required=True)
    ap.add_argument("--dense-groups", default="gdn_in,gdn_out,qsa_qkv,qsa_o")
    ap.add_argument("--bf16", type=Path, help="bf16 checkpoint (rtn-bf16)")
    args = ap.parse_args()
    if not args.base.exists():
        build_base(args.template, args.base)
    base_index = json.loads((args.base / "index.json").read_text())
    args.out.mkdir(parents=True)
    if os.stat(args.base).st_dev != os.stat(args.out).st_dev:
        raise RuntimeError("--base and --out must share a filesystem (hard links)")
    index = dict(base_index)
    for f in sorted(set(base_index.values())):
        os.link(args.base / f, args.out / f)
    drop = torch.load(args.mask, map_location="cpu", weights_only=False)["drop"]
    cfg = json.loads((args.template / "config.json").read_text())
    n_layers = cfg["text_config"]["num_hidden_layers"]
    inter = cfg["text_config"]["moe_intermediate_size"]

    def experts():
        for layer in range(n_layers):
            keep = (~drop[layer]).nonzero().flatten()
            gate_up, down = expert_banks(args, layer, keep)
            prefix = f"model.language_model.layers.{layer}.mlp.experts"
            for e in range(len(keep)):
                yield f"{prefix}.{e}.gate_proj.weight", gate_up[e, :inter].clone()
                yield f"{prefix}.{e}.up_proj.weight", gate_up[e, inter:].clone()
                yield f"{prefix}.{e}.down_proj.weight", down[e].clone()
            print(f"experts L{layer:02d}", flush=True)
    write_shards(experts(), args.out, "experts", index)

    t_idx = json.loads((args.template / "model.safetensors.index.json").read_text())["weight_map"]
    dense_names = sorted(k for k in t_idx if DENSE.match(k))

    groups = args.dense_groups.split(",")
    if not set(groups) <= set(GROUPS):
        raise ValueError(f"unknown dense groups {groups}")
    chosen = {k for k in dense_names if any(GROUPS[g].search(k) for g in groups)}
    if args.dense == "bf16":
        chosen = set()

    def dense():
        from quantize_nonmoe_fp8 import quantize
        for k in dense_names:
            with safe_open(args.template / t_idx[k], framework="pt") as f:
                w = f.get_tensor(k)
            if w.dtype != torch.bfloat16:
                raise RuntimeError(f"{k}: template dense is {w.dtype}")
            if k not in chosen:
                yield k, w
            elif args.dense == "w4":
                yield k, humming_rtn(w, bits=4)
            else:
                qw, scale = quantize(w, "cuda")
                yield k, qw
                yield k[: -len("weight")] + "weight_scale_inv", scale
    write_shards(dense(), args.out, f"dense-{args.dense}", index)
    q = cfg["quantization_config"]
    skip = list(q["modules_to_not_convert"])
    if args.dense == "fp8":
        quantized = {k[: -len(".weight")] for k in chosen}
        if not quantized <= set(skip):
            raise RuntimeError("template already quantizes some DENSE module")
        skip = [m for m in skip if m not in quantized]
    skip += [f"model.language_model.layers.{layer}.mlp.experts" for layer in range(n_layers)]
    q["modules_to_not_convert"] = skip
    q["arc_eval_arm"] = {"experts": args.experts, "dense": args.dense,
                         "dense_groups": groups if chosen else []}
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2))
    (args.out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": index}, indent=1))
    for p in args.template.iterdir():
        if p.is_file() and p.suffix in (".json", ".jinja", ".txt") and p.name not in (
                "config.json", "model.safetensors.index.json"):
            shutil.copy2(p, args.out / p.name)
    print(f"built {args.out}", flush=True)


if __name__ == "__main__":
    main()
