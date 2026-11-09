"""The RL trainer base (BF16-dense variant) of a QAD package: bf16dense + a QAD v2 export (train_qad.export_delta).

RL on the int3 model kind serves the deployment package (dense FP8) and trains on its BF16-dense variant
(work/qwen38fn_eval/build_int3_bf16dense.py): the trainer quantizes its BF16 dense masters with
fp8_dense_qat.quantize_block_fp8 and checks that against the serving package bit for bit (--dense-qat-reference).
A QAD export keeps only the deployed format: the 144 dense FP8 projections as e4m3 + weight_scale_inv, no BF16 master,
and no checkpoint was kept. This script writes the BF16 tensors whose quantization is exactly the QAD package's FP8:

  every element  v = bf16(q * scale)                 (|q| <= 448: q * (1 +- 2^-9) rounds back to q in e4m3)
  saturated |q| = 448  v = +-a, a the BF16 value with  a / 448 == scale  on the quantizer's device
                       (export_delta quantized BF16 masters, so each block's amax is a BF16 value and a exists)

`a / 448` is computed with the quantizer's own expression on --device: CUDA divides by a Python scalar through a
reciprocal multiply, so a CPU search would pick the wrong neighbour for ~0.6% of the blocks. Every module is then
re-quantized with fp8_dense_qat.verify_against_package on --device against the QAD package (raises unless bit-exact).

The other export tensors (INT3 group scales, routers, shared experts, the 84 BF16 dense projections, all BF16) replace
the bf16dense tensors of the same name, dtype and shape. Shards without a replaced tensor are hardlinked; the others
are rewritten tensor by tensor (build_int3_bf16dense.rewrite_shard). Checks after writing: same tensor set as
bf16dense; every replaced BF16 tensor's sha256 equals the delta manifest; every other tensor of a rewritten shard
equals bf16dense byte for byte; every non-dense replaced tensor equals the QAD package byte for byte.

    python build_qad_bf16dense.py --bf16dense <gEAN320 bf16dense> --package <materialized QAD package> \
        --delta <export/vNNNNNN> --out <new dir> [--device cuda]

Writes <out>.partial, renames to <out>, writes <out>.sha256.json, prints BUILD-OK. Any mismatch raises.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qwen38fn_eval"))
sys.path.insert(0, str(HERE.parent / "qwen38fn_lora"))
from build_int3_bf16dense import Checkpoint, file_sha256, json_file, rewrite_shard  # noqa: E402
from fp8_dense_qat import BLOCK, FP8_MAX, quantize_block_fp8, verify_against_package  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def invert_block_fp8(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """BF16 [N, K] whose quantize_block_fp8 on q's device is exactly (q, scale); raises when a block has no such amax."""
    if q.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32 or q.dim() != 2 \
            or q.shape[0] % BLOCK or q.shape[1] % BLOCK or tuple(scale.shape) != (q.shape[0] // BLOCK, q.shape[1] // BLOCK):
        raise ValueError(f"block FP8 pair expected, got {q.dtype} {tuple(q.shape)} / {scale.dtype} {tuple(scale.shape)}")
    if scale.device != q.device:
        raise ValueError(f"q on {q.device}, scale on {scale.device}")
    n, k = q.shape
    qf = q.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    v = (qf * scale[:, None, :, None]).to(torch.bfloat16)
    zero = (qf == 0).all(dim=3).all(dim=1)
    if bool((zero & (scale != 1.0)).any()):
        raise ValueError("an all-zero block with a scale other than 1")
    # the block amax: a BF16 value a with a / FP8_MAX == scale under the quantizer's arithmetic (same device, same op)
    a0 = (scale * FP8_MAX).to(torch.bfloat16)
    up = torch.nextafter(a0, torch.full_like(a0, float("inf")))
    down = torch.nextafter(a0, torch.zeros_like(a0))
    cands = torch.stack([a0, up, down])
    ok = (cands.float() / FP8_MAX) == scale
    found = ok.any(dim=0) | zero
    if not bool(found.all()):
        raise ValueError(f"{int((~found).sum())} blocks have no BF16 amax that quantizes to their scale "
                         "(not an export of BF16 masters, or a different quantizer)")
    a = cands.gather(0, ok.float().argmax(dim=0)[None]).squeeze(0)
    sat = qf.abs() == FP8_MAX
    v = torch.where(sat, (torch.sign(qf) * a.float()[:, None, :, None]).to(torch.bfloat16), v)
    has_sat = sat.any(dim=3).any(dim=1)
    if not bool((has_sat | zero).all()):
        raise ValueError(f"{int((~(has_sat | zero)).sum())} non-zero blocks without a saturated element")
    return v.view(n, k)


def build(bf16dense: Path, package: Path, delta: Path, out: Path, device: str, workers: int) -> None:
    partial = out.with_name(out.name + ".partial")
    sha_file = out.with_name(out.name + ".sha256.json")
    for path in (out, partial, sha_file):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"{path} exists; remove it by hand before building")
    if bf16dense.resolve().parent != out.resolve().parent:
        raise ValueError("bf16dense and out must sit in the same directory (hardlinks)")
    if not (delta / "DONE").exists() or not (package / "QAD_DONE").exists():
        raise RuntimeError(f"{delta}: no DONE, or {package}: no QAD_DONE")
    dman = json.loads((delta / "delta_manifest.json").read_text())
    base, pkg = Checkpoint(bf16dense), Checkpoint(package)
    pkg_meta = json.loads((package / "deploy_manifest.json").read_text())
    if "qad" not in pkg_meta:
        raise RuntimeError(f"{package}: deploy_manifest.json has no qad block (not a materialized QAD package)")

    # 1. the export's tensors: 144 FP8 pairs + BF16 replacements, all present in the QAD package with the manifest sha
    tensors = dman["tensors"]
    fp8 = sorted(k.removesuffix(".weight_scale_inv") for k in tensors if k.endswith(".weight_scale_inv"))
    fp8_names = {m + ".weight" for m in fp8} | {m + ".weight_scale_inv" for m in fp8}
    others = sorted(set(tensors) - fp8_names)
    if len(fp8) != 144 or not fp8_names <= set(tensors):
        raise ValueError(f"expected 144 FP8 pairs in the export, got {len(fp8)}")
    for name in sorted(tensors):
        _, shard, entry = pkg.locate(name)
        if shard.sha256(name) != tensors[name]["sha256"]:
            raise ValueError(f"{name}: QAD package bytes differ from the delta manifest")
    log(f"export: {len(fp8)} FP8 dense modules, {len(others)} BF16 tensors; QAD package matches the delta manifest")
    for name in others:
        _, _, b = base.locate(name)
        _, _, p = pkg.locate(name)
        if b["dtype"] != "BF16" or p["dtype"] != "BF16" or b["shape"] != p["shape"]:
            raise ValueError(f"{name}: bf16dense {b['dtype']} {b['shape']} vs package {p['dtype']} {p['shape']}")
    for m in fp8:
        _, _, b = base.locate(m + ".weight")
        _, _, p = pkg.locate(m + ".weight")
        if m + ".weight_scale_inv" in base.weight_map or b["dtype"] != "BF16" or b["shape"] != p["shape"]:
            raise ValueError(f"{m}: bf16dense {b['dtype']} {b['shape']} is not the BF16 form of the package's {p['shape']}")

    # 2. FP8 -> BF16 on the quantizer's device, then the trainer's bit-exact check against the QAD package
    replace_raw, dense = {}, {}
    weights = {}
    for m in fp8:
        _, qs, _ = pkg.locate(m + ".weight")
        _, ss, _ = pkg.locate(m + ".weight_scale_inv")
        q = torch.frombuffer(bytearray(qs.raw(m + ".weight")), dtype=torch.float8_e4m3fn).reshape(qs.tensors[m + ".weight"]["shape"])
        s = torch.frombuffer(bytearray(ss.raw(m + ".weight_scale_inv")), dtype=torch.float32).reshape(
            ss.tensors[m + ".weight_scale_inv"]["shape"])
        v = invert_block_fp8(q.to(device), s.to(device)).cpu().contiguous()
        weights[m] = v
        raw = v.view(torch.uint8).numpy().tobytes()
        replace_raw[m + ".weight"] = raw
        dense[m] = {"shape": list(v.shape), "bf16_sha256": hashlib.sha256(raw).hexdigest()}
    checked = verify_against_package(weights, package, device)
    log(f"bit-exact on {device}: quantize_block_fp8 of the {checked['modules']} BF16 tensors == the QAD package "
        f"({checked['weight_elements']} e4m3, {checked['scale_elements']} scales)")
    del weights
    for name in others:
        _, shard, _ = pkg.locate(name)
        replace_raw[name] = shard.raw(name)

    # 3. O.partial: bf16dense with the replacements
    partial.mkdir()
    shard_files = sorted(set(base.weight_map.values()))
    rewritten, hardlinked = [], []
    for filename in shard_files:
        shard = base.shard(filename)
        replace = {n: replace_raw[n] for n in shard.tensors if n in replace_raw}
        if not replace:
            os.link(bf16dense / filename, partial / filename)
            hardlinked.append(filename)
            continue
        log(f"rewrite {filename}: {len(replace)} tensors")
        rewrite_shard(shard, partial / filename, replace, set())
        rewritten.append(filename)
    log(f"shards: {len(hardlinked)} hardlinked, {len(rewritten)} rewritten")

    # 4. verify the output
    (partial / "model.safetensors.index.json").write_bytes(base.index_bytes)
    new = Checkpoint(partial)
    if new.weight_map != base.weight_map:
        raise RuntimeError("output index differs from bf16dense")
    same, replaced = 0, 0
    for filename in rewritten:
        out_shard, src_shard = new.shard(filename), base.shard(filename)
        if set(out_shard.tensors) != set(src_shard.tensors):
            raise RuntimeError(f"{filename}: tensor set differs from bf16dense")
        for name, entry in out_shard.tensors.items():
            src = src_shard.tensors[name]
            if entry["dtype"] != src["dtype"] or entry["shape"] != src["shape"]:
                raise RuntimeError(f"{name}: {entry['dtype']} {entry['shape']} vs bf16dense {src['dtype']} {src['shape']}")
            digest = out_shard.sha256(name)
            if name in replace_raw:
                module = name.removesuffix(".weight")
                want = dense[module]["bf16_sha256"] if module in dense else tensors[name]["sha256"]
                if digest != want:
                    raise RuntimeError(f"{name}: written bytes differ from the replacement")
                replaced += 1
            elif digest != src_shard.sha256(name):
                raise RuntimeError(f"{name}: differs from bf16dense")
            else:
                same += 1
    for filename in hardlinked:
        if not os.path.samestat(os.stat(bf16dense / filename), os.stat(partial / filename)):
            raise RuntimeError(f"{filename}: not the bf16dense inode")
    if replaced != len(replace_raw):
        raise RuntimeError(f"replaced {replaced} tensors, expected {len(replace_raw)}")
    log(f"verified: {replaced} replaced tensors, {same} unchanged tensors byte-compared in rewritten shards")

    # 5. the non-weight files (config unchanged), manifests
    copied = []
    for path in sorted(bf16dense.iterdir()):
        if path.suffix == ".safetensors" or path.name in ("model.safetensors.index.json", "deploy_manifest.json",
                                                           "bf16dense_manifest.json", ".cache"):
            continue
        if not path.is_file():
            raise ValueError(f"unexpected non-file in bf16dense: {path}")
        shutil.copyfile(path, partial / path.name)
        copied.append(path.name)
    qad = {"schema": "arcforge.qwen38fn.qad_bf16dense.v1",
           "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "bf16dense": {"path": str(bf16dense), "deploy_manifest_sha256": file_sha256(bf16dense / "deploy_manifest.json")},
           "package": {"path": str(package), "deploy_manifest_sha256": file_sha256(package / "deploy_manifest.json"),
                       "qad": pkg_meta["qad"]},
           "delta": {"path": str(delta), "manifest_sha256": file_sha256(delta / "delta_manifest.json"),
                     **{k: v for k, v in dman.items() if k not in ("files", "tensors")}},
           "dense_fp8_inverted": len(fp8), "bf16_replaced": len(others), "bit_exact_device": device,
           "shards_hardlinked": hardlinked, "shards_rewritten": rewritten, "dense": dense}
    bman = json.loads((bf16dense / "bf16dense_manifest.json").read_text())
    bman["qad"] = qad
    json_file(partial / "bf16dense_manifest.json", bman)
    manifest = copy.deepcopy(json.loads((bf16dense / "deploy_manifest.json").read_text()))
    manifest["derived_from"] = {**manifest["derived_from"], "qad": {"package_deploy_manifest_sha256": qad["package"]["deploy_manifest_sha256"],
                                                                     "details": "bf16dense_manifest.json qad"}}
    for filename in ["bf16dense_manifest.json", *rewritten]:
        manifest["files"][filename] = {"sha256": file_sha256(partial / filename), "bytes": (partial / filename).stat().st_size}
    json_file(partial / "deploy_manifest.json", manifest)

    # 6. publish and hash every file
    partial.rename(out)
    paths = sorted(out.iterdir())
    with ThreadPoolExecutor(workers) as pool:
        digests = list(pool.map(file_sha256, paths))
    files = {p.name: {"size": p.stat().st_size, "sha256": d} for p, d in zip(paths, digests)}
    json_file(sha_file, {"source": {"bf16dense": str(bf16dense), "package": str(package), "delta": str(delta),
                                    "variant": "qad-bf16dense"}, "files": files})
    log(f"output {out}: {len(files)} files, {sum(v['size'] for v in files.values()) / 1e9:.2f} GB; hashes in {sha_file}")
    print("BUILD-OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bf16dense", type=Path, required=True, help="the gEAN320 INT3 bf16dense trainer base")
    parser.add_argument("--package", type=Path, required=True, help="the materialized QAD package (materialize_package.py)")
    parser.add_argument("--delta", type=Path, required=True, help="the QAD export the package was built from")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda", help="the trainer's quantizer device (default cuda: bitwise as the trainer)")
    parser.add_argument("--workers", type=int, default=4, help="threads hashing the output files")
    args = parser.parse_args()
    torch.set_num_threads(8)
    build(args.bf16dense, args.package, args.delta, args.out, args.device, args.workers)


if __name__ == "__main__":
    main()
