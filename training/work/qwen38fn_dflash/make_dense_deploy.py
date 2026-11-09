"""A deployment copy of a dense drafter (convert_dense.py's -bf16 output, convert_plain.py's conversion, or
export_trained.py's export of a dense run): chosen linears in block FP8 (none with an empty --fp8: an all-BF16 copy),
no embedding/head. The default groups are the FFN's (routed for ffn_type dense_shared, mlp for dense) and attention.

The serving drafter takes embed_tokens and lm_head from the target, so they are left out. The FP8 layout is
convert_dense.quantize_fp8 (= patch 0032): e4m3 weight + fp32 `weight_scale_inv` per 128x128 block. Every other tensor
is copied bit for bit. Loads with sglang-arc-2's arc_dflash.worker.build_dense_drafter.

    python work/qwen38fn_dflash/make_dense_deploy.py --source <...-dense6400-bf16> --out <...-dense6400-fp8attn-deploy>
"""

import argparse
import fnmatch
import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_dense import dequant_fp8, publish, quantize_fp8, rel, sha256  # noqa: E402

SHARED_WITH_TARGET = ("embed_tokens.weight", "lm_head.weight")
GROUPS = {
    "routed": [f"layers.*.mlp.routed.{p}" for p in ("gate_proj", "up_proj", "down_proj")],
    "mlp": [f"layers.*.mlp.{p}" for p in ("gate_proj", "up_proj", "down_proj")],  # ffn_type dense (convert_plain.py)
    "attention": [f"layers.*.self_attn.{p}" for p in ("q_proj", "k_proj", "v_proj", "o_proj")],
}


def bits(t):
    return t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--fp8", nargs="*", choices=sorted(GROUPS),
                   help="linear groups in block FP8 (default: the FFN's and attention); none: an all-BF16 deployment "
                   "(e.g. an OPD rollout drafter whose q should match the trainer's numerics)")
    args = p.parse_args()
    src, out = args.source, args.out
    partial = out.with_name(out.name + ".partial")
    if out.exists() or partial.exists():
        raise FileExistsError(out)
    sums = dict(reversed(line.split("  ")) for line in (src / "SHA256SUMS").read_text().splitlines())
    for name, digest in sorted(sums.items()):
        if sha256(src / name) != digest:
            raise RuntimeError(f"source checksum mismatch: {name}")
    config = json.loads((src / "config.json").read_text())
    if (src / "conversion_manifest.json").is_file():
        kind = json.loads((src / "conversion_manifest.json").read_text())["kind"]  # convert_dense.py
    else:
        kind = json.loads((src / "initialization_manifest.json").read_text())["kind"]  # export_trained.py
    if kind not in ("dense-routed-bf16", "trained-bf16-export", "plain-conversion") \
            or config["ffn_type"] not in ("dense_shared", "dense") or "quantization" in config:
        raise RuntimeError(f"{src} ({kind}) is not a BF16 dense export (convert_dense.py, convert_plain.py or "
                           "export_trained.py)")
    if args.fp8 is None:
        args.fp8 = ["routed" if config["ffn_type"] == "dense_shared" else "mlp", "attention"]
    patterns = [pat for g in args.fp8 for pat in GROUPS[g]]
    config["quantization"] = {
        "method": "block_fp8", "fmt": "e4m3", "weight_block_size": [128, 128],
        "scale": "weight_scale_inv fp32 [N/128, K/128], w = q * s", "activation_scheme": "dynamic",
        "modules": patterns, "others": "bf16 (unchanged)",
    }

    partial.mkdir(parents=True)
    (partial / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    files, errors, matched = [], {}, {pat: 0 for pat in patterns}
    for name in ["interface.safetensors"] + [f"layer-{i}.safetensors" for i in range(config["num_layers"])]:
        state = load_file(str(src / name))
        written = {}
        for key, value in state.items():
            if key in SHARED_WITH_TARGET:
                continue
            module = key[: -len(".weight")] if key.endswith(".weight") else None
            hits = [pat for pat in patterns if module is not None and fnmatch.fnmatch(module, pat)]
            if not hits:
                written[key] = value
                continue
            if len(hits) != 1 or value.dtype != torch.bfloat16 or value.dim() != 2:
                raise RuntimeError(f"{key}: patterns {hits}, {value.dtype} {tuple(value.shape)}")
            matched[hits[0]] += 1
            q, s = quantize_fp8(value)
            written[key], written[module + ".weight_scale_inv"] = q, s
            errors[module] = rel(dequant_fp8(q, s), value.float())
        save_file(written, str(partial / name))
        files.append(name)
        # reread: exactly the tensors written (untouched ones are the source's own objects), bit for bit
        with safe_open(str(partial / name), "pt") as f:
            if set(f.keys()) != set(written):
                raise RuntimeError(f"{name}: written keys differ on reread")
            for key, value in written.items():
                if not torch.equal(bits(f.get_tensor(key)), bits(value)):
                    raise RuntimeError(f"{name}: {key} does not round-trip")
        del state
    unmatched = [pat for pat, n in matched.items() if n != config["num_layers"]]
    if unmatched:
        raise RuntimeError(f"patterns not matching once per layer: { {p: matched[p] for p in unmatched} }")
    (partial / "initialization_manifest.json").write_text(json.dumps({
        "complete": True, "files": files, "kind": "dense-routed-deploy",
        "dtype": "bfloat16 + block fp8" if patterns else "bfloat16",
        "shared_with_target": list(SHARED_WITH_TARGET),
    }, indent=2) + "\n")
    print(f"FP8 linears {len(errors)}" + (f", weight rel err max {max(errors.values()):.4f}" if errors else ""))
    publish(partial, out, {
        "complete": True, "kind": "dense-routed-deploy", "source": str(src.resolve()), "source_kind": kind,
        "source_sha256": sums,
        "fp8_groups": args.fp8, "fp8_weight_rel_err": errors, "dropped": list(SHARED_WITH_TARGET),
    })
    print("wrote", out)


if __name__ == "__main__":
    main()
