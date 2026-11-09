"""Derive the all-BF16-dense variant of the gEAN320 INT3 deployment package (RL stage 1).

The package keeps 144 dense modules (attention q/k/v, GDN in_proj_qkv/in_proj_z/out_proj) as block-FP8
(`<m>.weight` F8_E4M3 + `<m>.weight_scale_inv` F32). The rollout server of the first RL stage runs the routed
experts as INT3 but every dense projection in BF16, so the trainer's BF16 LoRA-merged weights sync as they are.
This script replaces those 144 FP8 weights with the official BF16 tensors and drops their scales.

Before anything is written it proves that the official BF16 tensors are the originals the package's FP8 came from
(dequantized FP8 vs BF16, relative RMS below --max-rel-rms) and that every other BF16 tensor the two checkpoints
share is bit-identical (the BF16 LoRA targets o_proj / in_proj_a / in_proj_b among them). Shards without any of
the 144 tensors are hardlinked (same tmpfs, read-only use); the others are rewritten tensor by tensor with the
unchanged tensors' bytes copied verbatim. Official routers and experts are never copied: the router is only
compared (package == official[kept experts]).

Run with a Python that has torch (CPU only; nothing starts CUDA):
  python build_int3_bf16dense.py --package P --bf16-source S --out O
Reads S (NFS) tensor by tensor and never copies a whole file from it. Writes O.partial, renames to O after the
checks pass, then writes O.sha256.json and prints BUILD-OK. Any mismatch raises.
"""
from __future__ import annotations

import argparse
import copy
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from int3_deploy_runtime import audit_model  # noqa: E402  (stdlib-only at import time)

CHUNK = 64 * 1024**2
LAYER = r"model\.language_model\.layers\.(\d+)\."
EXPERT = re.compile(LAYER + r"mlp\.experts\.(w13_qweight|w13_scales|w2_qweight|w2_scales)$")
ROUTER = re.compile(LAYER + r"mlp\.gate\.weight$")
# the 228 projections the RL trainer puts LoRA on; 144 of them are FP8 in the package
LORA = re.compile(LAYER + r"(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|in_proj_a|in_proj_b|out_proj))\.weight$")
# BF16 in the package already: must match the official bytes exactly
LORA_BF16 = re.compile(LAYER + r"(self_attn\.o_proj|linear_attn\.in_proj_a|linear_attn\.in_proj_b)\.weight$")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def json_file(path: Path, value: dict) -> None:
    temp = path.with_name(path.name + ".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


class Shard:
    """One safetensors file's header; tensor bytes are pread on demand, never mapped or loaded whole."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                raise ValueError(f"truncated safetensors header: {path}")
            size = struct.unpack("<Q", raw)[0]
            if size > 256 * 1024**2:
                raise ValueError(f"oversized safetensors header: {path}")
            header = json.loads(f.read(size))
        self.metadata = header.pop("__metadata__", None)
        self.tensors = header
        self.base = 8 + size
        payload = path.stat().st_size - self.base
        for name, entry in header.items():
            start, end = entry["data_offsets"]
            if not 0 <= start <= end <= payload:
                raise ValueError(f"truncated tensor payload {path.name}:{name}")

    def size(self, name: str) -> int:
        start, end = self.tensors[name]["data_offsets"]
        return end - start

    def chunks(self, name: str):
        start, end = self.tensors[name]["data_offsets"]
        fd = os.open(self.path, os.O_RDONLY)
        try:
            pos, stop = self.base + start, self.base + end
            while pos < stop:
                data = os.pread(fd, min(CHUNK, stop - pos), pos)
                if not data:
                    raise EOFError(f"short read: {self.path.name}:{name}")
                yield data
                pos += len(data)
        finally:
            os.close(fd)

    def raw(self, name: str) -> bytes:
        return b"".join(self.chunks(name))

    def sha256(self, name: str) -> str:
        h = hashlib.sha256()
        for block in self.chunks(name):
            h.update(block)
        return h.hexdigest()


class Checkpoint:
    """A model directory with a safetensors index; headers are read lazily, one shard at a time."""

    def __init__(self, root: Path):
        self.root = root
        self.index_bytes = (root / "model.safetensors.index.json").read_bytes()
        self.index = json.loads(self.index_bytes)
        self.weight_map = self.index["weight_map"]
        self._shards: dict[str, Shard] = {}

    def shard(self, filename: str) -> Shard:
        if Path(filename).name != filename:
            raise ValueError(f"unsafe shard filename {filename!r}")
        if filename not in self._shards:
            self._shards[filename] = Shard(self.root / filename)
        return self._shards[filename]

    def locate(self, name: str) -> tuple[str, Shard, dict]:
        filename = self.weight_map[name]
        shard = self.shard(filename)
        if name not in shard.tensors:
            raise ValueError(f"index references absent tensor {filename}:{name}")
        return filename, shard, shard.tensors[name]


def dequant_rel_rms(fp8_raw: bytes, scale_raw: bytes, ref_raw: bytes, shape: list[int], name: str) -> float:
    """||fp8 * scale_inv(block-expanded) - bf16|| / ||bf16||; scales multiply, 128x128 blocks, edge blocks ceil."""
    n, k = shape
    w = torch.frombuffer(bytearray(fp8_raw), dtype=torch.float8_e4m3fn).reshape(n, k).float()
    scale = torch.frombuffer(bytearray(scale_raw), dtype=torch.float32).reshape((n + 127) // 128, (k + 127) // 128)
    if not (torch.isfinite(w).all() and torch.isfinite(scale).all() and (scale > 0).all()):
        raise ValueError(f"{name}: non-finite FP8 weight or non-positive scale")
    scale = scale.repeat_interleave(128, 0)[:n].repeat_interleave(128, 1)[:, :k]
    ref = torch.frombuffer(bytearray(ref_raw), dtype=torch.bfloat16).reshape(n, k).float()
    denominator = torch.sum(ref * ref, dtype=torch.float64).item()
    if denominator == 0:
        raise ValueError(f"{name}: all-zero BF16 reference")
    diff = w * scale - ref
    return math.sqrt(torch.sum(diff * diff, dtype=torch.float64).item() / denominator)


def json_paths(value, needles: set[str], path: str = "") -> list[str]:
    """Every JSON path whose key or string value is one of needles."""
    found = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in needles:
                found.append(f"{path}/{key}")
            found += json_paths(item, needles, f"{path}/{key}")
    elif isinstance(value, list):
        for i, item in enumerate(value):
            found += json_paths(item, needles, f"{path}[{i}]")
    elif isinstance(value, str) and value in needles:
        found.append(path)
    return found


def collapse(paths: list[str]) -> list[str]:
    return sorted({re.sub(r"\[\d+\]", "[]", p) for p in paths})


def diff_paths(a, b, path: str = "") -> list[str]:
    """JSON paths at which two documents differ (list values compared whole)."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for key in sorted(set(a) | set(b))
                for p in (diff_paths(a[key], b[key], f"{path}/{key}") if key in a and key in b else [f"{path}/{key}"])]
    return [] if a == b else [path]


def rewrite_shard(src: Shard, dst: Path, replace: dict[str, bytes], drop: set[str]) -> None:
    """Copy src to dst with `replace` tensors swapped for BF16 bytes and `drop` tensors removed; the rest verbatim."""
    if not (set(replace) | drop) <= set(src.tensors):
        raise ValueError(f"{src.path.name}: replace/drop names not all in this shard")
    entries, plan, pos = {}, [], 0
    for name in sorted(src.tensors, key=lambda n: src.tensors[n]["data_offsets"][0]):
        if name in drop:
            continue
        entry = src.tensors[name]
        if name in replace:
            size = len(replace[name])
            if size != math.prod(entry["shape"]) * 2:
                raise ValueError(f"{name}: BF16 payload {size} bytes disagrees with shape {entry['shape']}")
            entry = {"dtype": "BF16", "shape": entry["shape"]}
        else:
            size = src.size(name)
            entry = dict(entry)
        entry["data_offsets"] = [pos, pos + size]
        entries[name] = entry
        plan.append(name)
        pos += size
    header = {} if src.metadata is None else {"__metadata__": src.metadata}
    header.update(entries)
    blob = json.dumps(header, separators=(",", ":")).encode()
    blob += b" " * (-len(blob) % 8)
    with dst.open("wb") as out:
        out.write(struct.pack("<Q", len(blob)) + blob)
        for name in plan:
            if name in replace:
                out.write(replace[name])
            else:
                for block in src.chunks(name):
                    out.write(block)
        if out.tell() != 8 + len(blob) + pos:
            raise RuntimeError(f"{dst.name}: wrote {out.tell()} bytes, expected {8 + len(blob) + pos}")


def build(package: Path, bf16_source: Path, out: Path, max_rel_rms: float, workers: int) -> None:
    partial = out.with_name(out.name + ".partial")
    sha_file = out.with_name(out.name + ".sha256.json")
    for path in (out, partial, sha_file):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"{path} exists; remove it by hand before building")
    if package.resolve().parent != out.resolve().parent:
        raise ValueError("package and out must sit in the same tmpfs directory (hardlinks)")
    package_table = json.loads(Path(str(package) + ".sha256.json").read_text())
    config = json.loads((package / "config.json").read_text())
    meta = config["quantization_config"]["arc_int3"]
    fp8_modules, bf16_modules = list(meta["dense_fp8_modules"]), list(meta["dense_bf16_modules"])
    if len(fp8_modules) != 144 or len(set(fp8_modules)) != 144:
        raise ValueError(f"expected 144 distinct dense_fp8_modules, got {len(fp8_modules)}")
    if set(fp8_modules) & set(bf16_modules):
        raise ValueError("a module is listed as both FP8 and BF16")
    deploy_manifest = json.loads((package / "deploy_manifest.json").read_text())
    pkg, off = Checkpoint(package), Checkpoint(bf16_source)
    log(f"package {package} ({package_table['repo']}@{package_table['sha']}): {len(pkg.weight_map)} tensors")
    log(f"bf16 source {bf16_source}: {len(off.weight_map)} tensors")

    # 1. every config / manifest place that names the 144 modules, before touching anything
    needles = set(fp8_modules)
    json_docs = {name: json.loads((package / name).read_text())
                 for name in ("config.json", "deploy_manifest.json", "hf_quant_config.json") if (package / name).exists()}
    places = {name: collapse(json_paths(doc, needles)) for name, doc in json_docs.items()}
    known = {"config.json": ["/quantization_config/arc_int3/dense_fp8_modules[]"],
             "deploy_manifest.json": ["/policy/dense_fp8[]"]}
    for name, found in places.items():
        if found and found != known.get(name):
            raise ValueError(f"{name} names the FP8 dense modules at an unexpected place: {found}")
    log(f"FP8 module names live in: {json.dumps({k: v for k, v in places.items() if v})}")
    ignore = config["quantization_config"]["ignore"]
    not_ignored = [m for m in fp8_modules if not any(fnmatch.fnmatchcase(m, pattern) for pattern in ignore)]
    log(f"carrier `ignore` already covers {len(fp8_modules) - len(not_ignored)} of the 144 modules as unquantized")

    # 2. package FP8 pairs vs official BF16: shape, then dequantized relative RMS
    order = []
    for module in fp8_modules:
        wname, sname = module + ".weight", module + ".weight_scale_inv"
        _, _, pw = pkg.locate(wname)
        _, _, ps = pkg.locate(sname)
        ofile, _, ow = off.locate(wname)
        if pw["dtype"] != "F8_E4M3" or ps["dtype"] != "F32" or ow["dtype"] != "BF16":
            raise ValueError(f"{module}: dtypes package {pw['dtype']}/{ps['dtype']}, official {ow['dtype']}")
        if len(pw["shape"]) != 2 or pw["shape"] != ow["shape"]:
            raise ValueError(f"{module}: shape package {pw['shape']} vs official {ow['shape']}")
        if ps["shape"] != [(n + 127) // 128 for n in pw["shape"]]:
            raise ValueError(f"{module}: scale shape {ps['shape']} for weight {pw['shape']}")
        order.append((ofile, ow["data_offsets"][0], module))
    order.sort()  # official shards in file order, tensors in offset order: sequential NFS reads
    stats, replace_raw = {}, {}
    log(f"dequant check: {len(order)} modules, limit {max_rel_rms}")
    for ofile, _, module in order:
        wname, sname = module + ".weight", module + ".weight_scale_inv"
        pfile, pshard, pw = pkg.locate(wname)
        _, sshard, _ = pkg.locate(sname)
        _, oshard, _ = off.locate(wname)
        bf16_raw, fp8_raw, scale_raw = oshard.raw(wname), pshard.raw(wname), sshard.raw(sname)
        value = dequant_rel_rms(fp8_raw, scale_raw, bf16_raw, pw["shape"], module)
        if not value < max_rel_rms:
            raise ValueError(f"{module}: dequantized FP8 vs official BF16 relative RMS {value:.4f} >= {max_rel_rms}")
        replace_raw[wname] = bf16_raw
        stats[module] = {"shape": pw["shape"], "rel_rms": value, "official_shard": ofile, "package_shard": pfile,
                         "bf16_sha256": hashlib.sha256(bf16_raw).hexdigest(),
                         "package_fp8_sha256": hashlib.sha256(fp8_raw).hexdigest(),
                         "package_scale_inv_sha256": hashlib.sha256(scale_raw).hexdigest()}
    by_kind: dict[str, list[float]] = {}
    for module, item in stats.items():
        by_kind.setdefault(".".join(module.split(".")[-2:]), []).append(item["rel_rms"])
    ranked = sorted(item["rel_rms"] for item in stats.values())
    summary = {"count": len(ranked), "min": ranked[0], "median": ranked[len(ranked) // 2],
               "p90": ranked[int(len(ranked) * 0.9)], "max": ranked[-1],
               "by_kind": {k: {"n": len(v), "min": min(v), "max": max(v)} for k, v in sorted(by_kind.items())}}
    log(f"dequant rel RMS: min {summary['min']:.4f} median {summary['median']:.4f} p90 {summary['p90']:.4f} max {summary['max']:.4f}")
    for kind, item in summary["by_kind"].items():
        log(f"  {kind:24s} n={item['n']:3d} min {item['min']:.4f} max {item['max']:.4f}")
    log(f"BF16 payload of the 144 replacements: {sum(len(v) for v in replace_raw.values()) / 1e9:.2f} GB")

    # 3. every other BF16 tensor the two checkpoints share: bit-identical (router: package == official[kept])
    fp8_names = {m + ".weight" for m in fp8_modules} | {m + ".weight_scale_inv" for m in fp8_modules}
    shared, families = [], {"expert": 0, "router": 0, "other_not_compared": {}}
    for name in pkg.weight_map:
        if name in fp8_names:
            continue
        _, _, entry = pkg.locate(name)
        if EXPERT.match(name):
            families["expert"] += 1
        elif ROUTER.match(name):
            families["router"] += 1
        elif entry["dtype"] == "BF16":
            if name not in off.weight_map:
                raise ValueError(f"package BF16 tensor absent from the BF16 source: {name}")
            shared.append(name)
        else:
            families["other_not_compared"][entry["dtype"]] = families["other_not_compared"].get(entry["dtype"], 0) + 1
    shared.sort(key=lambda n: (off.weight_map[n], off.locate(n)[2]["data_offsets"][0]))
    identical = {}
    for name in shared:
        _, pshard, pe = pkg.locate(name)
        _, oshard, oe = off.locate(name)
        if oe["dtype"] != "BF16" or oe["shape"] != pe["shape"]:
            raise ValueError(f"{name}: package {pe['dtype']} {pe['shape']} vs official {oe['dtype']} {oe['shape']}")
        a, b = pshard.sha256(name), oshard.sha256(name)
        if a != b:
            raise ValueError(f"{name}: BF16 differs between package and official")
        identical[name] = a
    lora_bf16 = [n for n in identical if LORA_BF16.match(n)]
    if len(lora_bf16) != 84:
        raise ValueError(f"expected 84 BF16 LoRA projections (o_proj, in_proj_a/b), compared {len(lora_bf16)}")
    log(f"bit-identical to the BF16 source: {len(identical)} BF16 tensors ({len(lora_bf16)} BF16 LoRA projections); "
        f"not compared: {families['expert']} packed expert, {families['router']} router, "
        f"others by dtype {families['other_not_compared']}")
    keep = deploy_manifest["keep_per_layer"]
    for name in pkg.weight_map:
        match = ROUTER.match(name)
        if not match:
            continue
        _, pshard, pe = pkg.locate(name)
        _, oshard, oe = off.locate(name)
        ids = keep[match[1]]
        packed = torch.frombuffer(bytearray(pshard.raw(name)), dtype=torch.int16).reshape(pe["shape"])
        full = torch.frombuffer(bytearray(oshard.raw(name)), dtype=torch.int16).reshape(oe["shape"])
        if pe["dtype"] != "BF16" or oe["dtype"] != "BF16" or oe["shape"][0] != 512 or pe["shape"][0] != len(ids) \
                or not torch.equal(packed, full[torch.tensor(ids)]):
            raise ValueError(f"{name}: package router is not the official router restricted to the kept experts")
    log(f"routers: {families['router']} package routers equal the official routers restricted to keep_per_layer")

    # 4. build O.partial: copy / hardlink / rewrite
    partial.mkdir()
    index_new = {k: v for k, v in pkg.weight_map.items() if k not in {m + ".weight_scale_inv" for m in fp8_modules}}
    shard_files = sorted(set(pkg.weight_map.values()))
    on_disk = {p.name for p in package.glob("*.safetensors")}
    if on_disk != set(shard_files):
        raise ValueError(f"package shards on disk differ from its index: {sorted(on_disk ^ set(shard_files))}")
    rewritten, hardlinked = [], []
    for filename in shard_files:
        shard = pkg.shard(filename)
        drop = {n for n in shard.tensors if n in fp8_names and n.endswith(".weight_scale_inv")}
        replace = {n: replace_raw[n] for n in shard.tensors if n in replace_raw}
        if not drop and not replace:
            os.link(package / filename, partial / filename)
            hardlinked.append(filename)
            continue
        if len(drop) != len(replace):
            raise ValueError(f"{filename}: FP8 weights and scales are split across shards")
        log(f"rewrite {filename}: {len(replace)} FP8 -> BF16, {len(drop)} scales dropped")
        rewrite_shard(shard, partial / filename, replace, drop)
        rewritten.append(filename)
    log(f"shards: {len(hardlinked)} hardlinked, {len(rewritten)} rewritten")

    copied = []
    for path in sorted(package.iterdir()):
        if path.name == ".cache" or path.suffix == ".safetensors" or path.name in ("config.json", "model.safetensors.index.json", "deploy_manifest.json"):
            continue
        if not path.is_file():
            raise ValueError(f"unexpected non-file in the package: {path}")
        shutil.copyfile(path, partial / path.name)
        copied.append(path.name)

    # config.json: the 144 move from dense_fp8_modules to dense_bf16_modules
    new_config = copy.deepcopy(config)
    new_meta = new_config["quantization_config"]["arc_int3"]
    new_meta["dense_fp8_modules"] = []
    new_meta["dense_bf16_modules"] = sorted(bf16_modules + fp8_modules)
    changed_fields = diff_paths(config, new_config)
    if changed_fields != ["/quantization_config/arc_int3/dense_bf16_modules", "/quantization_config/arc_int3/dense_fp8_modules"]:
        raise RuntimeError(f"unexpected config changes: {changed_fields}")
    json_file(partial / "config.json", new_config)
    old_payload = sum(pkg.shard(f).size(n) for n, f in pkg.weight_map.items())
    payload = sum(Shard(partial / f).size(n) for n, f in index_new.items())
    if old_payload != deploy_manifest["payload_bytes"]:
        raise RuntimeError(f"package payload {old_payload} != its manifest {deploy_manifest['payload_bytes']}")
    index = {k: v for k, v in pkg.index.items() if k != "weight_map"}
    if "metadata" in index:  # total_size follows the payload, whatever convention the package used
        index["metadata"] = {**index["metadata"], "total_size": index["metadata"]["total_size"] + payload - old_payload}
    index["weight_map"] = index_new
    json_file(partial / "model.safetensors.index.json", index)

    # 5. verify the partial dir against the package and the BF16 source
    new = Checkpoint(partial)
    if set(new.weight_map) != set(index_new) or len(new.weight_map) != len(pkg.weight_map) - 144:
        raise RuntimeError("output index differs from the expected tensor set")
    for filename in shard_files:
        if set(new.shard(filename).tensors) != {n for n, f in index_new.items() if f == filename}:
            raise RuntimeError(f"{filename}: tensors in the file differ from the index")
    audit = audit_model(partial)
    if (audit["dense_fp8_modules"], audit["dense_bf16_modules"]) != (0, len(bf16_modules) + 144):
        raise RuntimeError(f"plugin audit disagrees: {audit}")
    log(f"plugin CPU audit: {json.dumps(audit)}")
    checked = {"inode_identical": 0, "bytes_identical": 0,
               "by_family": {"expert": 0, "router": 0, "replaced_dense": 0, "bf16_shared": 0, "ple_and_other": 0}}
    for filename in shard_files:
        if filename in hardlinked:
            if not os.path.samestat(os.stat(package / filename), os.stat(partial / filename)):
                raise RuntimeError(f"{filename}: not the package's inode")
            checked["inode_identical"] += len(pkg.shard(filename).tensors)
            continue
        for name in new.shard(filename).tensors:
            if name in replace_raw:
                continue
            _, ps, _ = pkg.locate(name)
            if ps.sha256(name) != new.shard(filename).sha256(name) or ps.tensors[name]["dtype"] != new.shard(filename).tensors[name]["dtype"] \
                    or ps.tensors[name]["shape"] != new.shard(filename).tensors[name]["shape"]:
                raise RuntimeError(f"{name}: differs between package and output")
            checked["bytes_identical"] += 1
    for name in new.weight_map:
        family = "expert" if EXPERT.match(name) else "router" if ROUTER.match(name) else \
            "replaced_dense" if name in replace_raw else "bf16_shared" if name in identical else "ple_and_other"
        checked["by_family"][family] += 1
    # the 228 LoRA projections: 2-D BF16 and bit-identical to the BF16 source
    lora = sorted(n for n in new.weight_map if LORA.match(n))
    if len(lora) != 228:
        raise RuntimeError(f"expected 228 LoRA projections, found {len(lora)}")
    for name in lora:
        _, shard, entry = new.locate(name)
        if entry["dtype"] != "BF16" or len(entry["shape"]) != 2 or name[:-len("weight")] + "weight_scale_inv" in new.weight_map:
            raise RuntimeError(f"{name}: not a plain 2-D BF16 weight in the output")
        _, oshard, oentry = off.locate(name)
        source_sha = stats[name[:-len(".weight")]]["bf16_sha256"] if name in replace_raw else oshard.sha256(name)
        if oentry["shape"] != entry["shape"] or shard.sha256(name) != source_sha:
            raise RuntimeError(f"{name}: differs from the BF16 source")
    log(f"228 LoRA projections: all 2-D BF16, bit-identical to the BF16 source "
        f"(144 written from it, {len(lora) - 144} re-read from it and compared)")
    log(f"unchanged tensors: {checked['inode_identical']} via shared inode, {checked['bytes_identical']} byte-compared in rewritten shards; "
        f"output by family {checked['by_family']}")

    # 6. manifests
    package_manifest_sha = file_sha256(package / "deploy_manifest.json")
    bf16dense = {
        "schema": "arcforge.qwen38fn.int3_bf16dense.v1",
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "package": {"path": str(package), "hf_repo": package_table["repo"], "hf_commit": package_table["sha"],
                    "config_sha256": file_sha256(package / "config.json"),
                    "index_sha256": hashlib.sha256(pkg.index_bytes).hexdigest(),
                    "deploy_manifest_sha256": package_manifest_sha},
        "bf16_source": {"path": str(bf16_source),
                        "config_sha256": file_sha256(bf16_source / "config.json"),
                        "index_sha256": hashlib.sha256(off.index_bytes).hexdigest(),
                        "note": f"tensor-level sha256 only; its {len(set(off.weight_map.values()))} shards were not hashed whole"},
        "change": "the 144 dense_fp8_modules were rewritten as BF16 from bf16_source and their weight_scale_inv dropped",
        "dense_fp8_modules_replaced": 144,
        "dequant_check": {"max_rel_rms_allowed": max_rel_rms, **summary},
        "bf16_identity_check": {"compared_bf16_tensors": len(identical), "bf16_lora_projections": len(lora_bf16),
                                "routers_equal_official_restricted_to_keep": families["router"]},
        "config_fields_changed": changed_fields,
        "deploy_manifest_fields_changed": ["/policy/dense_fp8", "/policy/dense_bf16", "/tensor_count",
                                           "/payload_bytes", "/files", "/derived_from"],
        "shards_hardlinked": hardlinked, "shards_rewritten": rewritten,
        "replaced_tensors": stats,
    }
    json_file(partial / "bf16dense_manifest.json", bf16dense)
    new_manifest = copy.deepcopy(deploy_manifest)
    new_manifest["policy"]["dense_fp8"] = []
    new_manifest["policy"]["dense_bf16"] = sorted(deploy_manifest["policy"]["dense_bf16"] + fp8_modules)
    new_manifest["derived_from"] = {"variant": "bf16dense", "package_deploy_manifest_sha256": package_manifest_sha,
                                    "details": "bf16dense_manifest.json"}
    new_manifest["tensor_count"] = deploy_manifest["tensor_count"] - 144
    if new_manifest["tensor_count"] != len(new.weight_map):
        raise RuntimeError(f"deploy manifest tensor_count {new_manifest['tensor_count']} != index {len(new.weight_map)}")
    new_manifest["payload_bytes"] = payload
    for filename in ["config.json", "model.safetensors.index.json", "bf16dense_manifest.json", *rewritten]:
        new_manifest["files"][filename] = {"sha256": file_sha256(partial / filename), "bytes": (partial / filename).stat().st_size}
    # files the package itself did not list (the Hub adds a dotfile) may stay unlisted, nothing new may
    package_unlisted = {p.name for p in package.iterdir()} - set(deploy_manifest["files"]) - {"deploy_manifest.json", ".cache"}
    unlisted = {p.name for p in partial.iterdir()} - set(new_manifest["files"]) - {"deploy_manifest.json"} - package_unlisted
    if unlisted:
        raise RuntimeError(f"deploy manifest does not list: {sorted(unlisted)}")
    json_file(partial / "deploy_manifest.json", new_manifest)
    for name in ("config.json", "deploy_manifest.json"):
        left = collapse(json_paths(json.loads((partial / name).read_text()), needles))
        if left != {"config.json": ["/quantization_config/arc_int3/dense_bf16_modules[]"],
                    "deploy_manifest.json": ["/policy/dense_bf16[]"]}[name]:
            raise RuntimeError(f"{name}: FP8 module names remain at unexpected places: {left}")

    # 7. publish and hash every file
    partial.rename(out)
    paths = sorted(p for p in out.iterdir())
    if any(not p.is_file() for p in paths):
        raise RuntimeError("non-file entries in the output")
    with ThreadPoolExecutor(workers) as pool:
        digests = list(pool.map(file_sha256, paths))
    files = {p.name: {"size": p.stat().st_size, "sha256": d} for p, d in zip(paths, digests)}
    for filename in hardlinked + copied:
        want = package_table["files"].get(filename)
        if want is not None and (want["size"], want["sha256"]) != (files[filename]["size"], files[filename]["sha256"]):
            raise RuntimeError(f"{filename}: differs from the package's verified sha256 table")
    json_file(sha_file, {"source": {"package": str(package), "hf_repo": package_table["repo"],
                                    "hf_commit": package_table["sha"], "variant": "bf16dense"}, "files": files})
    total = sum(v["size"] for v in files.values())
    log(f"output {out}: {len(files)} files, {total / 1e9:.2f} GB; hashes in {sha_file}")
    print("BUILD-OK", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", type=Path, required=True, help="INT3 package with dense FP8 (local tmpfs)")
    parser.add_argument("--bf16-source", type=Path, required=True, help="official checkpoint holding the BF16 dense weights")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-rel-rms", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4, help="threads hashing the output files")
    args = parser.parse_args()
    torch.set_num_threads(8)
    build(args.package, args.bf16_source, args.out, args.max_rel_rms, args.workers)


if __name__ == "__main__":
    main()
