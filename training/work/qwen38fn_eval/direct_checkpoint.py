"""CPU-only checkpoint staging with guarded read-only NFS access.

NFS defaults to exact-file O_RDONLY|O_DIRECT + preadv into aligned buffers.
Opt-in resident-cache-only checks every pread range with an untouched read-only
mmap/mincore probe, and rejects cold pages. Cache eviction between probe and
pread remains possible; this mode cannot guarantee zero NFS RPCs.
The watchdog reports a stalled syscall and rejects its result if it returns late;
it cannot guarantee cancellation of an uninterruptible NFS hard-mount syscall.
Outputs must be local. No retry, cache-drop, source write, or GPU operation occurs.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import mmap
import os
import re
import shutil
import struct
import sys
import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch
from prune_experts import EXPERT, FUSED_MTP, IS_MTP, ROUTER, check_suffix_consistency

GIB = 1024**3
ALIGNMENT = 4096
DTYPES = {
    "BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8,
    "I16": torch.int16, "I32": torch.int32, "I64": torch.int64,
    "U16": torch.uint16, "U32": torch.uint32, "U64": torch.uint64,
    "F16": torch.float16, "BF16": torch.bfloat16, "F32": torch.float32,
    "F64": torch.float64, "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
    "F8_E4M3FNUZ": torch.float8_e4m3fnuz, "F8_E5M2FNUZ": torch.float8_e5m2fnuz,
}


def event(kind, **fields):
    print(json.dumps({"event": kind, "time": time.time(), **fields}), file=sys.stderr, flush=True)


def filesystem_type(path: Path) -> str:
    """Look up the mount lexically; do not resolve symlinks/stat an NFS tree."""
    absolute = os.path.abspath(path)
    best, fs = -1, None
    for line in Path("/proc/self/mountinfo").read_text().splitlines():
        before, after = line.split(" - ", 1)
        point = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), before.split()[4])
        if absolute == point or absolute.startswith(point.rstrip("/") + "/"):
            if len(point) > best:
                best, fs = len(point), after.split()[0]
    if fs is None:
        raise RuntimeError(f"Cannot identify mount for {path}")
    return fs


def local_path(path: Path):
    # Output/mask paths may resolve local symlinks, but must not traverse NFS.
    if filesystem_type(path) in ("nfs", "nfs4"):
        raise RuntimeError(f"Local filesystem required: {path}")
    if filesystem_type(path.resolve()) in ("nfs", "nfs4"):
        raise RuntimeError(f"Path resolves onto NFS: {path}")


LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
LIBC.mmap.restype = ctypes.c_void_p
LIBC.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
LIBC.mincore.restype = ctypes.c_int
LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
LIBC.munmap.restype = ctypes.c_int
PAGE_BYTES = os.sysconf("SC_PAGESIZE")


def resident_pages(fd, offset, count):
    """Probe without touching mapped bytes or populating the file's page cache."""
    if count == 0:
        return []
    start = offset // PAGE_BYTES * PAGE_BYTES
    length = offset - start + count
    address = LIBC.mmap(None, length, mmap.PROT_READ, mmap.MAP_SHARED, fd, start)
    if address == ctypes.c_void_p(-1).value:
        number = ctypes.get_errno()
        raise OSError(number, os.strerror(number))
    try:
        pages = (ctypes.c_ubyte * ((length + PAGE_BYTES - 1) // PAGE_BYTES))()
        if LIBC.mincore(address, length, pages) != 0:
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))
        return [bool(value & 1) for value in pages]
    finally:
        if LIBC.munmap(address, length) != 0:
            number = ctypes.get_errno()
            raise OSError(number, os.strerror(number))


class ColdCacheError(RuntimeError):
    pass


class IOTimeout(TimeoutError):
    pass


class DirectFile:
    """Bounded reads. direct=True can be forced for local CPU verification."""
    def __init__(self, path, *, direct=None, chunk_bytes=8 * 1024**2,
                 stall_seconds=120, progress_seconds=10, resident_cache_only=False):
        self.path = Path(path)
        self.resident_cache_only = resident_cache_only
        if resident_cache_only and direct is True:
            raise ValueError("resident-cache-only and O_DIRECT are mutually exclusive")
        self.direct = False if resident_cache_only else (filesystem_type(self.path) in ("nfs", "nfs4") if direct is None else direct)
        if not self.direct and not resident_cache_only and filesystem_type(self.path) in ("nfs", "nfs4"):
            raise RuntimeError("Buffered NFS reads are forbidden")
        if chunk_bytes < ALIGNMENT or chunk_bytes % ALIGNMENT or stall_seconds <= 0 or progress_seconds <= 0:
            raise ValueError("chunk_bytes must be 4096-aligned; stall_seconds must be positive")
        self.chunk_bytes = chunk_bytes
        self.stall_seconds = stall_seconds
        self.progress_seconds = progress_seconds
        self.bytes_read = 0
        self.io_bytes_read = 0
        self.io_calls = 0
        self.cache_hits = 0
        self.cache = OrderedDict()
        self.active = None
        self.timed_out = threading.Event()
        self.stop = threading.Event()
        self.fd = None
        self.buffer = None
        self.watchdog = threading.Thread(target=self._watch, daemon=True)
        self.watchdog.start()
        try:
            flags = os.O_RDONLY | (os.O_DIRECT if self.direct else 0)
            # O_NOFOLLOW also closes the race after the lexical symlink check.
            self.fd = self._call("open", lambda: os.open(self.path, flags | os.O_NOFOLLOW))
            actual_path = Path(os.readlink(f"/proc/self/fd/{self.fd}"))
            if not self.direct and not resident_cache_only and filesystem_type(actual_path) in ("nfs", "nfs4"):
                raise RuntimeError("Source resolves onto NFS; give its exact mounted path for O_DIRECT")
            self.size = self._call("fstat", lambda: os.fstat(self.fd).st_size)
            if self.resident_cache_only:
                # Avoid buffered-read readahead into unprobed neighboring pages.
                self._call("fadvise_random", lambda: os.posix_fadvise(self.fd, 0, 0, os.POSIX_FADV_RANDOM))
            if self.direct:
                self.buffer = mmap.mmap(-1, self.chunk_bytes + ALIGNMENT)
            event("source_open", path=str(self.path), direct=self.direct, bytes=self.size,
                  resident_cache_only=self.resident_cache_only)
        except BaseException:
            self.close()
            raise

    def _watch(self):
        last = time.monotonic()
        while not self.stop.wait(min(1, self.progress_seconds)):
            now = time.monotonic()
            active = self.active
            if active and now - active[1] > self.stall_seconds and not self.timed_out.is_set():
                self.timed_out.set()
                event("io_stall", path=str(self.path), operation=active[0],
                      seconds=now - active[1], bytes_read=self.bytes_read, io_bytes_read=self.io_bytes_read, io_calls=self.io_calls,
                      note="syscall may remain in D state; watchdog cannot guarantee cancellation")
            if now - last >= self.progress_seconds:
                event("io_progress", path=str(self.path), bytes_read=self.bytes_read,
                      io_bytes_read=self.io_bytes_read, io_calls=self.io_calls, cache_hits=self.cache_hits,
                      operation=None if active is None else active[0], stalled=self.timed_out.is_set())
                last = now

    def _call(self, operation, function):
        if self.timed_out.is_set():
            raise IOTimeout(f"Previous syscall stalled: {self.path}")
        self.active = (operation, time.monotonic())
        try:
            result = function()
        except OSError as exc:
            event("io_error", path=str(self.path), operation=operation, errno=exc.errno, error=str(exc))
            raise
        finally:
            active = self.active
            self.active = None
        if self.timed_out.is_set() or time.monotonic() - active[1] > self.stall_seconds:
            # open() may have returned a live fd; ensure the caller can close it.
            if operation == "open":
                self.fd = result
            raise IOTimeout(f"{operation} exceeded {self.stall_seconds}s: {self.path}")
        return result

    def read(self, offset: int, count: int) -> bytes:
        if offset < 0 or count < 0 or offset + count > self.size:
            raise ValueError(f"Read outside file: offset={offset}, count={count}, size={self.size}")
        parts = []
        while count:
            amount = min(count, self.chunk_bytes)
            if self.direct:
                # Two fixed-size aligned blocks cover crossing reads and many tiny
                # scales without one NFS RPC per tensor. No file-backed mmap/cache.
                block = offset // self.chunk_bytes * self.chunk_bytes
                skip = offset - block
                amount = min(amount, self.chunk_bytes - skip)
                if block in self.cache:
                    self.cache_hits += 1
                    self.cache.move_to_end(block)
                else:
                    needed = min(self.chunk_bytes, self.size - block)
                    span = (needed + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
                    view = memoryview(self.buffer)[:span]
                    try:
                        n = self._call("preadv", lambda: os.preadv(self.fd, [view], block))
                        if n < needed:
                            raise EOFError(f"Short direct read: {self.path}: {n} < {needed}")
                        self.io_calls += 1
                        self.io_bytes_read += n
                        self.cache[block] = bytes(view[:needed])
                    finally:
                        view.release()
                    while len(self.cache) > 2:
                        self.cache.popitem(last=False)
                part = self.cache[block][skip:skip + amount]
            else:
                if self.resident_cache_only:
                    pages = self._call("mincore", lambda: resident_pages(self.fd, offset, amount))
                    if not all(pages):
                        event("cold_cache_refused", path=str(self.path), offset=offset, bytes=amount,
                              cold_pages=sum(not resident for resident in pages), total_pages=len(pages))
                        raise ColdCacheError(f"Nonresident source pages refused before pread: {self.path}: offset={offset}, bytes={amount}")
                part = self._call("pread", lambda: os.pread(self.fd, amount, offset))
                self.io_calls += 1
                self.io_bytes_read += len(part)
                if len(part) != amount:
                    raise EOFError(f"Short local read: {self.path}")
            self.bytes_read += len(part)
            parts.append(part)
            offset += amount
            count -= amount
        return b"".join(parts)

    def close(self):
        self.stop.set()
        self.cache.clear()
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if self.buffer is not None:
            self.buffer.close()
            self.buffer = None
        self.watchdog.join(timeout=2)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read_small(path, *, cap=64 * 1024**2, **options):
    with DirectFile(path, **options) as f:
        if f.size > cap:
            raise ValueError(f"Metadata file exceeds {cap} bytes: {path}")
        return f.read(0, f.size)


class Reader:
    def __init__(self, path, *, header_cap=64 * 1024**2, tensor_cap=64 * GIB, **options):
        self.file = DirectFile(path, **options)
        self.tensor_cap = tensor_cap
        try:
            if self.file.size < 8:
                raise ValueError("Missing safetensors header")
            length, = struct.unpack("<Q", self.file.read(0, 8))
            if length > header_cap or length > self.file.size - 8:
                raise ValueError(f"Invalid/oversized safetensors header: {length}")
            self.header = json.loads(self.file.read(8, length))
            self.start = 8 + length
            intervals = []
            for key, item in self.header.items():
                if key == "__metadata__":
                    continue
                dtype = DTYPES[item["dtype"]]
                shape, offsets = item["shape"], item["data_offsets"]
                if any(type(n) is not int or n < 0 for n in shape):
                    raise ValueError(f"Invalid shape: {key}")
                size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                if len(offsets) != 2 or any(type(n) is not int for n in offsets):
                    raise ValueError(f"Invalid offsets: {key}")
                lo, hi = offsets
                if lo < 0 or hi < lo or hi - lo != size or self.start + hi > self.file.size:
                    raise ValueError(f"Invalid tensor bounds: {key}")
                intervals.append((lo, hi, key))
            end = 0
            for lo, hi, key in sorted(intervals):
                if lo != end:
                    raise ValueError(f"Noncontiguous/overlapping tensor payload: {key}")
                end = hi
            if self.start + end != self.file.size:
                raise ValueError("Safetensors contains trailing/unindexed payload")
        except BaseException:
            self.file.close()
            raise

    def _allocate(self, item, shape):
        dtype = DTYPES[item["dtype"]]
        nbytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        if nbytes > self.tensor_cap:
            raise ValueError(f"Requested tensor exceeds cap: {nbytes} > {self.tensor_cap}")
        return torch.empty(shape, dtype=dtype)

    def _fill(self, tensor, offset):
        flat = tensor.reshape(-1).view(torch.uint8)
        for begin in range(0, flat.numel(), self.file.chunk_bytes):
            n = min(self.file.chunk_bytes, flat.numel() - begin)
            raw = bytearray(self.file.read(offset + begin, n))
            flat[begin:begin + n].copy_(torch.frombuffer(raw, dtype=torch.uint8))
        return tensor

    def get_tensor(self, key):
        item = self.header[key]
        return self._fill(self._allocate(item, item["shape"]), self.start + item["data_offsets"][0])

    def get_experts(self, key, indices):
        """Read only requested slices on the first dimension, retaining raw dtype bits."""
        item = self.header[key]
        shape = item["shape"]
        ids = list(indices)
        if not shape or any(type(i) is not int or i < 0 or i >= shape[0] for i in ids):
            raise ValueError("Invalid expert/row indices")
        out = self._allocate(item, [len(ids), *shape[1:]])
        stride = math.prod(shape[1:]) * out.element_size()
        begin = 0
        while begin < len(ids):
            end = begin + 1
            while end < len(ids) and ids[end] == ids[end - 1] + 1:
                end += 1
            self._fill(out[begin:end], self.start + item["data_offsets"][0] + ids[begin] * stride)
            begin = end
        return out

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.file.close()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def space_guard(out, needed, floor):
    free = shutil.disk_usage(out).free
    if free < needed + floor:
        raise RuntimeError(f"Insufficient local space: free={free}, write={needed}, floor={floor}")


def output_dir(path):
    local_path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise RuntimeError(f"Output must be empty: {path}")


def mask_keep(path, layers, n_experts):
    local_path(path)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    drop = blob["drop"]
    if drop.dtype != torch.bool or tuple(drop.shape) != (len(layers), n_experts):
        raise ValueError(f"Invalid boolean drop mask: {path}: {tuple(drop.shape)}")
    keep = {layer: (~drop[row]).nonzero().flatten().tolist() for row, layer in enumerate(layers)}
    sizes = {len(ids) for ids in keep.values()}
    if len(sizes) != 1 or 0 in sizes:
        raise ValueError("Each layer must keep the same positive expert count")
    return keep, hashlib.sha256(path.read_bytes()).hexdigest()


def source_index(src, options):
    wm = json.loads(read_small(src / "model.safetensors.index.json", **options))["weight_map"]
    if not wm or any(Path(f).name != f or f in (".", "..") for f in wm.values()):
        raise ValueError("Index must name exact shard files directly in source directory")
    return wm


AUXILIARY = ("generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
             "merges.txt", "chat_template.jinja", "preprocessor_config.json",
             "video_preprocessor_config.json", "hf_quant_config.json")


def optional_small(path, cap, options):
    try:
        return read_small(path, cap=cap, **options)
    except FileNotFoundError:
        return None


def prepare_reuse(args, wm, cfg, per_layer, n_experts, options):
    """Validate a local pruned checkpoint and map original keys to its local keys."""
    root = args.reuse_pruned
    local_path(root)
    manifest_bytes = read_small(root / "prune_manifest.json", **options)
    manifest = json.loads(manifest_bytes)
    if "complete" in manifest and not manifest["complete"]:
        raise ValueError("Reuse checkpoint is marked incomplete")
    old_source = Path(manifest["source"])
    expected_source = os.path.abspath(args.src)
    if os.path.abspath(old_source) != expected_source:
        # The provenance path may be an explicitly recorded local directory symlink.
        # readlink examines that alias only; no source payload is opened through it.
        try:
            link = os.readlink(old_source)
        except OSError as exc:
            raise ValueError("Reuse provenance source must equal full source or be its exact symlink") from exc
        target = os.path.abspath(old_source.parent / link) if not os.path.isabs(link) else os.path.abspath(link)
        if target != expected_source:
            raise ValueError("Reuse provenance symlink points to a different full source")
    old_count = manifest["n_experts_after"]
    if manifest["n_experts_before"] != n_experts or type(old_count) is not int or not 0 < old_count <= n_experts:
        raise ValueError("Reuse provenance expert counts mismatch")
    old_cfg = json.loads(read_small(root / "config.json", **options))
    if old_cfg["text_config"]["num_experts"] != old_count:
        raise ValueError("Reuse config expert count disagrees with provenance")
    old_cfg["text_config"]["num_experts"] = n_experts
    if old_cfg != cfg:
        raise ValueError("Reuse config differs from full source beyond num_experts")
    groups = {}
    for is_mtp, field in ((False, "keep_per_layer"), (True, "mtp_keep")):
        expected_layers = {str(layer) for mtp, layer in per_layer if mtp == is_mtp}
        stored = manifest[field]
        if set(stored) != expected_layers:
            raise ValueError(f"Reuse {field} layer inventory mismatch")
        for layer, ids in stored.items():
            if (len(ids) != old_count or any(type(e) is not int or not 0 <= e < n_experts for e in ids)
                    or len(set(ids)) != len(ids) or ids != sorted(ids)):
                raise ValueError(f"Reuse {field} has invalid original expert IDs")
            groups[(is_mtp, int(layer))] = {old: new for new, old in enumerate(ids)}
    hashes = {}
    for name in AUXILIARY:
        source = optional_small(args.src / name, args.metadata_cap, options)
        reused = optional_small(root / name, args.metadata_cap, options)
        if source != reused:
            raise ValueError(f"Reuse auxiliary identity mismatch: {name}")
        if name == "tokenizer.json" and source is None:
            raise ValueError("Reuse requires tokenizer.json identity")
        if source is not None:
            hashes[name] = hashlib.sha256(source).hexdigest()
    old_wm = source_index(root, options)
    original_to_old, required_experts = {}, set()
    for key in wm:
        match = EXPERT.match(key)
        if match:
            group = (bool(IS_MTP.match(key)), int(match["layer"]))
            old_id = int(match["eid"])
            if old_id in groups[group]:
                old_key = f"{match['pre']}.{groups[group][old_id]}.{match['rest']}"
                required_experts.add(old_key)
                original_to_old[key] = old_key
        elif key in old_wm and not ROUTER.match(key):
            original_to_old[key] = key
    if not required_experts <= set(old_wm):
        missing = sorted(required_experts - set(old_wm))
        raise ValueError(f"Reuse claims retained expert tensors that are missing: {missing[:5]}")
    possible = required_experts | {key for key in wm if not EXPERT.match(key)}
    if set(old_wm) - possible:
        raise ValueError("Reuse index contains unexpected tensor keys")
    metadata = {}
    by_file = {}
    for key, filename in old_wm.items():
        by_file.setdefault(filename, []).append(key)
    local_options = {**options, "chunk_bytes": 8 * 1024**2}
    for filename, keys in sorted(by_file.items()):
        with Reader(root / filename, tensor_cap=args.tensor_cap, **local_options) as reader:
            actual_keys = set(reader.header) - {"__metadata__"}
            if actual_keys != set(keys):
                raise ValueError(f"Reuse shard/header index inventory mismatch: {filename}")
            for key in keys:
                metadata[key] = reader.header[key]
    return {"root": root, "weight_map": old_wm, "metadata": metadata,
            "mapping": original_to_old, "old_count": old_count,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "aux_sha256": hashes}


def prune_fp8(args):
    options = {"stall_seconds": args.stall_seconds, "progress_seconds": args.progress_seconds,
               "resident_cache_only": args.resident_cache_only}
    if args.reuse_pruned is not None and not args.resident_cache_only:
        options["chunk_bytes"] = 512 * 1024
    wm = source_index(args.src, options)
    per_layer = {}
    for key in wm:
        if FUSED_MTP.match(key):
            raise ValueError("FP8 pruning requires per-expert 2D layout, including MTP")
        match = EXPERT.match(key)
        if match:
            group = (bool(IS_MTP.match(key)), int(match["layer"]))
            per_layer.setdefault(group, set()).add(int(match["eid"]))
    layers = sorted(layer for mtp, layer in per_layer if not mtp)
    mtp_layers = sorted(layer for mtp, layer in per_layer if mtp)
    if not layers:
        raise ValueError("No per-expert main layers found")
    n_experts = len(per_layer[(False, layers[0])])
    if any(ids != set(range(n_experts)) for ids in per_layer.values()):
        raise ValueError("Source must contain all original expert IDs in every layer")
    check_suffix_consistency(wm, layers)
    keep, mask_sha = mask_keep(args.mask, layers, n_experts)
    n_keep = len(keep[layers[0]])
    mtp_keep, mtp_sha = ({}, None)
    if mtp_layers:
        if args.mtp_mask is None:
            raise ValueError("--mtp-mask is required when source contains MTP")
        mtp_keep, mtp_sha = mask_keep(args.mtp_mask, mtp_layers, n_experts)
        if any(len(ids) != n_keep for ids in mtp_keep.values()):
            raise ValueError("MTP and main model must keep identical counts")
    maps = {(False, layer): {old: new for new, old in enumerate(ids)} for layer, ids in keep.items()}
    maps.update({(True, layer): {old: new for new, old in enumerate(ids)} for layer, ids in mtp_keep.items()})
    remap = {}
    for key in wm:
        match = EXPERT.match(key)
        if match:
            ids = maps[(bool(IS_MTP.match(key)), int(match["layer"]))]
            old = int(match["eid"])
            if old in ids:
                remap[key] = f"{match['pre']}.{ids[old]}.{match['rest']}"
        else:
            remap[key] = key
    cfg = json.loads(read_small(args.src / "config.json", **options))
    if cfg["text_config"]["num_experts_per_tok"] != args.expected_topk:
        raise ValueError(f"Source top-k must remain {args.expected_topk}")
    reuse = None if args.reuse_pruned is None else prepare_reuse(args, wm, cfg, per_layer, n_experts, options)
    # Plan metadata first, then stream raw tensor bytes. In particular a giant PLE
    # table is never materialized in host RAM or copied by safetensors serialization.
    by_file = {}
    for key in wm:
        by_file.setdefault(wm[key], []).append(key)
    plans = []
    planned_bytes = 0
    reused_bytes = reused_tensors = reused_expert_tensors = 0
    validated_reuse = set()
    for shard_name, keys in sorted(by_file.items()):
        with Reader(args.src / shard_name, tensor_cap=args.tensor_cap, **options) as reader:
            for key in sorted(keys, key=lambda k: reader.header[k]["data_offsets"][0]):
                item = reader.header[key]
                old_key = None
                if reuse is not None:
                    if key in reuse["mapping"]:
                        old_key = reuse["mapping"][key]
                    elif ROUTER.match(key) and key in reuse["weight_map"]:
                        old_key = key
                    if old_key is not None:
                        old_item = reuse["metadata"][old_key]
                        expected_shape = list(item["shape"])
                        if ROUTER.match(key):
                            expected_shape[0] = reuse["old_count"]
                        if old_item["dtype"] != item["dtype"] or old_item["shape"] != expected_shape:
                            raise ValueError(f"Reuse dtype/shape mismatch: {old_key}")
                        validated_reuse.add(old_key)
                if key not in remap:
                    continue
                shape = list(item["shape"])
                ids = None
                router = ROUTER.match(key)
                if router:
                    layer = int(router["layer"])
                    ids = mtp_keep[layer] if IS_MTP.match(key) else keep[layer]
                    if shape[0] != n_experts:
                        raise ValueError(f"Router dimension mismatch: {key}")
                    shape[0] = len(ids)
                size = math.prod(shape) * torch.empty((), dtype=DTYPES[item["dtype"]]).element_size()
                read_root, read_key, read_file = args.src, key, shard_name
                read_offset = item["data_offsets"][0]
                if reuse is not None and old_key is not None and not ROUTER.match(key):
                    read_root, read_key = reuse["root"], old_key
                    read_file = reuse["weight_map"][old_key]
                    read_offset = reuse["metadata"][old_key]["data_offsets"][0]
                    reused_bytes += size
                    reused_tensors += 1
                    reused_expert_tensors += int(EXPERT.match(key) is not None)
                plans.append({"key": read_key, "name": remap[key], "source": read_file,
                              "read_root": str(read_root), "read_offset": read_offset,
                              "dtype": item["dtype"], "shape": shape, "bytes": size, "ids": ids})
                planned_bytes += size
    if reuse is not None:
        if validated_reuse != set(reuse["weight_map"]):
            raise ValueError("Reuse validation did not cover every indexed tensor")
        plans.sort(key=lambda p: (p["read_root"], p["source"], p["read_offset"]))
        event("reuse_plan", local_tensors=reused_tensors, local_expert_tensors=reused_expert_tensors,
              local_bytes=reused_bytes, full_source_bytes=planned_bytes - reused_bytes)
    if len({p["name"] for p in plans}) != len(plans):
        raise ValueError("Pruning produced duplicate output keys")
    output_dir(args.out)
    space_guard(args.out, planned_bytes + 64 * 1024**2, args.free_floor_bytes)
    current, buffered, shard, total, out_index = [], 0, 0, 0, {}

    def flush():
        nonlocal current, buffered, shard, total
        if not current:
            return
        shard += 1
        name = f"model-{shard:05d}.safetensors"
        header, offset = {"__metadata__": {"format": "pt"}}, 0
        for plan in current:
            header[plan["name"]] = {"dtype": plan["dtype"], "shape": plan["shape"],
                                    "data_offsets": [offset, offset + plan["bytes"]]}
            offset += plan["bytes"]
        encoded = json.dumps(header, separators=(",", ":")).encode()
        encoded += b" " * (-len(encoded) % 8)
        space_guard(args.out, buffered + len(encoded) + 8, args.free_floor_bytes)
        temporary = args.out / (name + ".tmp")
        reader, reader_name = None, None
        try:
            with temporary.open("wb") as output:
                output.write(struct.pack("<Q", len(encoded)))
                output.write(encoded)
                for plan in current:
                    identity = (plan["read_root"], plan["source"])
                    if identity != reader_name:
                        if reader is not None:
                            reader.file.close()
                        read_options = options
                        if reuse is not None and plan["read_root"] == str(reuse["root"]):
                            read_options = {**options, "chunk_bytes": 8 * 1024**2}
                        reader = Reader(Path(plan["read_root"]) / plan["source"], tensor_cap=args.tensor_cap, **read_options)
                        reader_name = identity
                    if plan["ids"] is not None:
                        tensor = reader.get_experts(plan["key"], plan["ids"])
                        output.write(tensor.reshape(-1).view(torch.uint8).numpy())
                    else:
                        absolute = reader.start + reader.header[plan["key"]]["data_offsets"][0]
                        for begin in range(0, plan["bytes"], reader.file.chunk_bytes):
                            size = min(reader.file.chunk_bytes, plan["bytes"] - begin)
                            output.write(reader.file.read(absolute + begin, size))
        finally:
            if reader is not None:
                reader.file.close()
        if temporary.stat().st_size != buffered + len(encoded) + 8:
            raise RuntimeError("Staged safetensors size mismatch")
        temporary.replace(args.out / name)
        out_index.update({p["name"]: name for p in current})
        total += buffered
        event("shard_written", file=name, bytes=buffered, tensors=len(current))
        current, buffered = [], 0

    for plan in plans:
        if current and buffered + plan["bytes"] > args.shard_bytes:
            flush()
        if plan["bytes"] > args.shard_bytes:
            event("oversize_tensor_shard", key=plan["key"], bytes=plan["bytes"],
                  note="single tensor cannot be split across safetensors shards; streaming without tensor allocation")
        current.append(plan)
        buffered += plan["bytes"]
    flush()
    cfg = json.loads(read_small(args.src / "config.json", **options))
    if cfg["text_config"]["num_experts_per_tok"] != args.expected_topk:
        raise ValueError(f"Source top-k must remain {args.expected_topk}")
    cfg["text_config"]["num_experts"] = n_keep
    atomic_json(args.out / "config.json", cfg)
    atomic_json(args.out / "model.safetensors.index.json", {"metadata": {"total_size": total}, "weight_map": out_index})
    for name in AUXILIARY:
        try:
            data = read_small(args.src / name, cap=args.metadata_cap, **options)
        except FileNotFoundError:
            continue
        space_guard(args.out, len(data), args.free_floor_bytes)
        (args.out / name).write_bytes(data)
    if cfg.get("quantization_config", {}).get("quant_method") == "modelopt" and not (args.out / "hf_quant_config.json").is_file():
        raise ValueError("Modelopt source requires hf_quant_config.json")
    if set(out_index) != set(remap.values()):
        raise RuntimeError("Output index inventory mismatch")
    atomic_json(args.out / "prune_manifest.json", {"source": str(args.src), "mask": str(args.mask),
        "mask_sha256": mask_sha, "mtp_mask": None if args.mtp_mask is None else str(args.mtp_mask),
        "mtp_mask_sha256": mtp_sha, "n_experts_before": n_experts, "n_experts_after": n_keep,
        "keep_per_layer": {str(k): v for k, v in keep.items()}, "mtp_keep": {str(k): v for k, v in mtp_keep.items()},
        "complete": True, "read_mode": "mincore-guarded resident pread" if args.resident_cache_only else "O_DIRECT on NFS; pread on local", "topk": args.expected_topk,
        "reuse": None if reuse is None else {"source": str(reuse["root"]), "manifest_sha256": reuse["manifest_sha256"],
            "aux_sha256": reuse["aux_sha256"], "local_tensors": reused_tensors,
            "local_expert_tensors": reused_expert_tensors, "local_bytes": reused_bytes,
            "full_source_bytes": planned_bytes - reused_bytes,
            "full_source_chunk_bytes": 8 * 1024**2 if args.resident_cache_only else 512 * 1024}})
    event("prune_complete", output=str(args.out), bytes=total, shards=shard)


def stage_bf16(args):
    options = {"stall_seconds": args.stall_seconds, "progress_seconds": args.progress_seconds,
               "resident_cache_only": args.resident_cache_only}
    wm = source_index(args.src, options)
    cfg = json.loads(read_small(args.src / "config.json", **options))["text_config"]
    layers = list(range(cfg["num_hidden_layers"]))
    keep, mask_sha = mask_keep(args.mask, layers, cfg["num_experts"])
    lo, hi = map(int, args.layers.split(":"))
    if not 0 <= lo < hi <= len(layers):
        raise ValueError("--layers must be an exclusive lo:hi interval inside model")
    output_dir(args.out)
    records = []
    atomic_json(args.out / "manifest.json", {"complete": False, "parts": records, "mask_sha256": mask_sha})
    for layer in range(lo, hi):
        ids = keep[layer]
        tensors = {"orig_expert_ids": torch.tensor(ids, dtype=torch.int32)}
        prefix = f"model.language_model.layers.{layer}.mlp.experts."
        for short, bank in (("gate_up", "gate_up_proj"), ("down", "down_proj")):
            key = prefix + bank
            if key not in wm and key + ".weight" in wm:
                key += ".weight"
            if key not in wm:
                raise ValueError(f"Missing fused BF16 expert bank: {key}")
            with Reader(args.src / wm[key], tensor_cap=args.tensor_cap, **options) as reader:
                if reader.header[key]["dtype"] != "BF16" or reader.header[key]["shape"][0] != cfg["num_experts"]:
                    raise ValueError(f"Expected fused BF16 bank with original expert dimension: {key}")
                shape = reader.header[key]["shape"]
                needed = len(ids) * math.prod(shape[1:]) * 2
                space_guard(args.out, needed + sum(t.numel() * t.element_size() for t in tensors.values()), args.free_floor_bytes)
                tensors[short] = reader.get_experts(key, ids)
        path = args.out / f"layer_{layer:02d}.pt"
        size = sum(t.numel() * t.element_size() for t in tensors.values())
        space_guard(args.out, size + 1024**2, args.free_floor_bytes)
        torch.save({"layer": layer, **tensors}, str(path) + ".tmp")
        Path(str(path) + ".tmp").replace(path)
        records.append({"layer": layer, "file": path.name, "bytes": path.stat().st_size,
                        "orig_expert_ids": ids, "gate_up_shape": list(tensors["gate_up"].shape),
                        "down_shape": list(tensors["down"].shape)})
        atomic_json(args.out / "manifest.json", {"source": str(args.src), "mask_sha256": mask_sha,
            "layers": [lo, hi], "complete": layer + 1 == hi, "parts": records})
        event("layer_staged", layer=layer, file=str(path), bytes=size, experts=len(ids))
        del tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prune-fp8", "stage-bf16"):
        p = commands.add_parser(name)
        p.add_argument("--src", required=True, type=Path)
        p.add_argument("--out", required=True, type=Path)
        p.add_argument("--mask", required=True, type=Path)
        p.add_argument("--stall-seconds", type=float, default=120)
        p.add_argument("--progress-seconds", type=float, default=10)
        p.add_argument("--resident-cache-only", action="store_true", help="Require every pread range resident via mincore; refuse cold reads")
        p.add_argument("--tensor-cap", type=int, default=64 * GIB)
        p.add_argument("--free-floor-bytes", type=int, default=20 * GIB)
        if name == "prune-fp8":
            p.add_argument("--mtp-mask", type=Path)
            p.add_argument("--reuse-pruned", type=Path, help="Validated local same-source pruned FP8 checkpoint")
            p.add_argument("--shard-bytes", type=int, default=4 * GIB)
            p.add_argument("--metadata-cap", type=int, default=256 * 1024**2)
            p.add_argument("--expected-topk", type=int, default=10)
        else:
            p.add_argument("--layers", required=True)
    args = parser.parse_args()
    if args.tensor_cap <= 0 or args.free_floor_bytes < 0 or args.progress_seconds <= 0:
        parser.error("caps and progress interval must be positive; free-space floor must be nonnegative")
    if args.command == "prune-fp8":
        if args.shard_bytes <= 0:
            parser.error("--shard-bytes must be positive")
        prune_fp8(args)
    else:
        stage_bf16(args)


if __name__ == "__main__":
    main()
