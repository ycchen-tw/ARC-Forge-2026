"""INT3 GPTQ parts -> Humming-ready sidecar next to the deployed REAP checkpoint.

Writes ``<out>/experts-LXX.safetensors`` per layer, in the pruned expert order of the
deployed checkpoint (``prune_manifest.json`` keep lists):

  model.language_model.layers.L.mlp.experts.w13_qweight  int32 [E, 2I, H*3/32]  rows = gate then up
  model.language_model.layers.L.mlp.experts.w13_scales   bf16  [E, 2I, H/128]
  model.language_model.layers.L.mlp.experts.w2_qweight   int32 [E, H, I*3/32]
  model.language_model.layers.L.mlp.experts.w2_scales    bf16  [E, H, I/128]

qweight is humming.ops.pack_weight(codes, 3) (little-endian 3-bit stream along K; checked
against a reference packer); the Humming layer config is b_dtype uint3, group 128, no zero
point, so ``HummingLayer.load_from_tensors({"weight": qweight, "weight_scale": scales})``
reproduces dequant ``(code - 4) * scale``. ``dense_fp8.safetensors`` holds the block-FP8
dense projections (bit-identical to the L4 hook's ARC_HC_ONLINE_FP8 of the BF16 weights).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from build_int3_eval_model import DENSE
from quantize_int3_gptq import unpack_nibbles
from quantize_nonmoe_fp8 import quantize as quantize_fp8
from safetensors import safe_open
from safetensors.torch import save_file


def ref_pack3(codes: torch.Tensor) -> torch.Tensor:
    """Reference little-endian 3-bit stream packer (small tensors only)."""
    bits = torch.zeros(*codes.shape[:-1], codes.shape[-1] * 3, dtype=torch.int64, device=codes.device)
    for b in range(3):
        bits[..., b::3] = (codes.long() >> b) & 1
    words = (bits.view(*codes.shape[:-1], -1, 32) << torch.arange(32, device=codes.device)).sum(-1)
    return torch.where(words < 2 ** 31, words, words - 2 ** 32).to(torch.int32)


def pack(codes_u8: torch.Tensor) -> torch.Tensor:
    from humming import ops
    codes = codes_u8.cuda().to(torch.int32).contiguous()
    packed = ops.pack_weight(codes, 3)
    if not torch.equal(ops.unpack_weight(packed, 3), codes):
        raise RuntimeError("humming pack/unpack round trip failed")
    if not torch.equal(ref_pack3(codes[:2, :8]), packed[:2, :8]):
        raise RuntimeError("humming pack_weight is not the little-endian 3-bit stream")
    return packed.cpu()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


HESSIAN = {
    "one-shot": "Hessian from the REAP-pruned FP8 forward",
    "sequential": "layer-sequential Hessian: gate/up from the quantized chain, down after GPTQ'd gate/up, every routed token",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", type=Path, required=True)
    ap.add_argument("--deployed", type=Path, required=True, help="REAP checkpoint the sidecar belongs to")
    ap.add_argument("--bf16", type=Path, required=True, help="source of the dense projections")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--calib-sha256", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--dense-source", type=Path,
                    help="copy this existing dense_fp8.safetensors instead of quantizing dense weights again")
    ap.add_argument("--calibration-manifest", type=Path,
                    help="actual multimodal calibration receipt to embed in deployment provenance")
    ap.add_argument("--pruning-manifest", type=Path,
                    help="pruning method and calibration provenance JSON")
    ap.add_argument("--hessian", choices=sorted(HESSIAN), required=True,
                    help="how the parts' Hessians were collected (run_int3_gptq.sh / run_seqint3.py)")
    args = ap.parse_args()
    keep = json.loads((args.deployed / "prune_manifest.json").read_text())["keep_per_layer"]
    n_layers = len(keep)
    args.out.mkdir(parents=True)
    files, reports = {}, []
    for layer in range(n_layers):
        report = json.loads((args.parts / f"layer-{layer:02d}.json").read_text())
        if report["run_id"] != args.run_id or report["format"] != "humming-uint3-g128-nozp":
            raise RuntimeError(f"layer {layer}: unexpected part report {report['run_id']}")
        reports.append({k: v for k, v in report.items() if k != "metrics"})
        prefix = f"model.language_model.layers.{layer}.mlp.experts"
        out = {}
        with safe_open(args.parts / f"int3-L{layer:02d}.safetensors", framework="pt") as f:
            if f.get_tensor("orig_expert_ids").tolist() != keep[str(layer)]:
                raise RuntimeError(f"layer {layer}: expert order differs from the deployed checkpoint")
            for bank, name in (("gate_up", "w13"), ("down", "w2")):
                codes = unpack_nibbles(f.get_tensor(f"{bank}.codes"))
                scale = f.get_tensor(f"{bank}.scale")
                if codes.max() > 7 or scale.dtype != torch.bfloat16:
                    raise RuntimeError(f"layer {layer} {bank}: not uint3 codes with bf16 scales")
                out[f"{prefix}.{name}_qweight"] = pack(codes)
                out[f"{prefix}.{name}_scales"] = scale.contiguous()
        path = args.out / f"experts-L{layer:02d}.safetensors"
        save_file(out, str(path), metadata={"format": "pt", "run_id": args.run_id})
        files[path.name] = sha256(path)
        print(f"L{layer:02d} {path.stat().st_size / 1e9:.2f} GB", flush=True)
    if args.dense_source is not None:
        import shutil
        shutil.copyfile(args.dense_source, args.out / "dense_fp8.safetensors")
    else:
        idx = json.loads((args.bf16 / "model.safetensors.index.json").read_text())["weight_map"]
        dense = {}
        for k in sorted(k for k in idx if DENSE.match(k)):
            with safe_open(args.bf16 / idx[k], framework="pt") as f:
                w = f.get_tensor(k)
            q, s = quantize_fp8(w, "cuda")
            dense[k] = q
            dense[k[: -len("weight")] + "weight_scale_inv"] = s
        save_file(dense, str(args.out / "dense_fp8.safetensors"), metadata={"format": "pt"})
    files["dense_fp8.safetensors"] = sha256(args.out / "dense_fp8.safetensors")
    manifest = {
        "format": {"experts": "humming uint3 g128, no zero point, dequant (code-4)*scale, "
                              "signed bf16 scales, qweight=humming.ops.pack_weight(codes,3)",
                   "w13_rows": "gate then up", "expert_order": "deployed prune_manifest keep_per_layer",
                   "dense": "block-FP8 128x128 e4m3 + fp32 weight_scale_inv (amax/448), "
                            "GDN in_proj_qkv/in_proj_z/out_proj + attention q/k/v/o, main model only"},
        "method": f"GPTQ ({HESSIAN[args.hessian]}, weights from BF16), act-order off, "
                  "damping 0.01, MSE clip search over both scale signs, 256-token pooled prior",
        "run_id": args.run_id, "calib_ids_sha256": args.calib_sha256,
        "keep_per_layer": keep,
        "source_prune_manifest_sha256": sha256(args.deployed / "prune_manifest.json"),
        "deployed": str(args.deployed), "bf16": str(args.bf16),
        "files": files, "layers": reports,
    }
    if args.calibration_manifest is not None:
        manifest["gptq_calibration"] = json.loads(args.calibration_manifest.read_text())
    if args.pruning_manifest is not None:
        manifest["pruning_calibration"] = json.loads(args.pruning_manifest.read_text())
    (args.out / "int3_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"sidecar {args.out}: {sum((args.out / f).stat().st_size for f in files) / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
