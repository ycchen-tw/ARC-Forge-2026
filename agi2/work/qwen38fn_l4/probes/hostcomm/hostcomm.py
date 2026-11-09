"""Python side of host-mapped collectives (see hostcomm.cu).

One HostComm per process group. Rank 0 of the group creates a POSIX shared memory
region; every rank pins + maps it (cudaHostRegisterMapped) and launches kernels that
exchange data through it. Requires the group to call collectives in lockstep, like NCCL.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import time
import subprocess
from multiprocessing import shared_memory
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
MAX_RANKS = 4
TRACE_CALLS, TRACE_BLOCKS = 512, 384  # hostcomm.cu HC_TRACE buffer: [call % TRACE_CALLS][block][8] globaltimer ns
KINDS = {"ar1_bf16": 0, "ar2_bf16": 1, "ag_bf16": 2, "ar1_fp8": 3, "ag_fp8": 4, "ar1_fp8v": 5, "ag_fp8v": 6}


RECEIPT = "libhostcomm.so.receipt.json"


def build(cuda_home: str, out_dir: Path, src: Path | None = None) -> Path:
    """Compile hostcomm.cu (or `src`, e.g. an older revision for an A/B) into out_dir/libhostcomm.so, or reuse it when
    its receipt matches: the receipt records the source sha256, the compiler (path + `nvcc --version`), the macros
    and the command, and the cache key covers all of them, so a different toolchain or knob rebuilds instead of
    silently reusing (ARC_HC_TIMEOUT_S, ARC_HC_POLL_NS, ARC_HC_TRACE=1 -> -DHC_TRACE)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = out_dir / "libhostcomm.so"
    src = HERE / "hostcomm.cu" if src is None else Path(src)
    nvcc = f"{cuda_home}/bin/nvcc"
    macros = []
    timeout_s, poll_ns = os.environ.get("ARC_HC_TIMEOUT_S"), os.environ.get("ARC_HC_POLL_NS")
    if timeout_s is not None:
        # clock64 cycles at ~2 GHz (L4 boost); lower clocks only make the wait longer.
        macros.append(f"-DHC_TIMEOUT_CYCLES={int(float(timeout_s) * 2e9)}LL")
    if poll_ns is not None:
        macros.append(f"-DHC_POLL_NS={int(poll_ns)}")
    trace = os.environ.get("ARC_HC_TRACE", "0")
    if trace not in ("0", "1"):
        raise ValueError(f"ARC_HC_TRACE must be 0 or 1, got {trace!r}")
    if trace == "1":
        macros.append("-DHC_TRACE")
    flags = ["-O3", "-std=c++17", "-arch=sm_89", "-shared", "-Xcompiler", "-fPIC", "--cudart", "shared"]
    version = subprocess.run([nvcc, "--version"], check=True, capture_output=True, text=True).stdout.strip()
    receipt = dict(src_sha256=hashlib.sha256(src.read_bytes()).hexdigest(), nvcc=nvcc, nvcc_version=version,
                   macros=macros, flags=flags)
    key = hashlib.sha256(json.dumps(receipt, sort_keys=True).encode()).hexdigest()
    rpath = out_dir / RECEIPT
    if lib.exists() and rpath.exists():
        old = json.loads(rpath.read_text())
        if old.get("key") == key and old.get("lib_sha256") == hashlib.sha256(lib.read_bytes()).hexdigest():
            return lib
    cmd = [nvcc, *macros, *flags, str(src), "-o", str(lib)]
    subprocess.run(cmd, check=True)
    receipt.update(key=key, cmd=cmd, lib_sha256=hashlib.sha256(lib.read_bytes()).hexdigest())
    rpath.write_text(json.dumps(receipt, indent=1))
    return lib


def check_receipt(lib: Path, src: Path | None = None) -> dict:
    """The build receipt next to `lib`, after checking that it describes this binary and (src given) was compiled from
    that source. Raises otherwise: a lib without a matching receipt is not a valid A/B arm."""
    rpath = Path(lib).parent / RECEIPT
    if not rpath.exists():
        raise RuntimeError(f"{lib}: no build receipt {rpath.name} (build it with hostcomm.build)")
    r = json.loads(rpath.read_text())
    sha = hashlib.sha256(Path(lib).read_bytes()).hexdigest()
    if r["lib_sha256"] != sha:
        raise RuntimeError(f"{lib}: sha256 {sha[:12]} is not the receipt's {r['lib_sha256'][:12]}")
    if src is not None and r["src_sha256"] != hashlib.sha256(Path(src).read_bytes()).hexdigest():
        raise RuntimeError(f"{lib} was built from another hostcomm.cu (receipt src {r['src_sha256'][:12]})")
    return r


def _node_cpus(node: int) -> set[int]:
    cpus = set()
    for part in Path(f"/sys/devices/system/node/node{node}/cpulist").read_text().strip().split(","):
        a, _, b = part.partition("-")
        cpus.update(range(int(a), int(b or a) + 1))
    return cpus


def _gpu_numa_node() -> int:
    """NUMA node of this process's CUDA device as sysfs reports it (-1 when the host has one node)."""
    p = torch.cuda.get_device_properties(torch.cuda.current_device())
    return int(Path(f"/sys/bus/pci/devices/{p.pci_domain_id:04x}:{p.pci_bus_id:02x}:{p.pci_device_id:02x}.0/"
                    "numa_node").read_text())


_libc = ctypes.CDLL(None, use_errno=True)
_libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_MADV_REMOVE = 9  # shmem: free the backing pages of the range (the next touch faults fresh ones)


def _node_free_kb(node: int) -> int:
    for line in Path(f"/sys/devices/system/node/node{node}/meminfo").read_text().splitlines():
        if "MemFree:" in line:
            return int(line.split()[-2])
    raise RuntimeError(f"no MemFree in node{node}/meminfo")


def _mapped_pages_per_node(start: int) -> dict[str, int]:
    """Pages this process has mapped in the VMA starting at `start`, per NUMA node (/proc/self/numa_maps "N<k>=").
    move_pages would answer per address but container seccomp refuses it (Runpod 10-08: EPERM)."""
    for line in Path("/proc/self/numa_maps").read_text().splitlines():
        fields = line.split()
        if int(fields[0], 16) == start:
            return {f.split("=")[0]: int(f.split("=")[1]) for f in fields if f[0] == "N" and "=" in f}
    raise RuntimeError(f"no /proc/self/numa_maps entry at {start:#x}")


class HostComm:
    def __init__(self, lib_path: Path, group, ranks: list[int], name: str, slot_bytes: int, max_blocks: int = 128,
                 pipe_stages: int = 0, fold_epoch: bool = False, trace: bool = False):
        """pipe_stages 1..8: ar1_fp8v, ag_fp8q and rs_owner fp8 run the pipelined kernels (writer / reader blocks,
        `blocks` = chunks, reads of a step overlap writes of the next; bitwise identical outputs); 0 = the originals.
        fold_epoch: the FP8 collectives (ar1_fp8v, ag_fp8v, ag_fp8q, ag_dp_fp8q, rs_owner fp8, pipelined or not)
        advance the epoch in their last block instead of a separate one-thread bump kernel after every call (one
        graph node less per call); the other kinds still bump. trace: record per-call, per-block phase timestamps
        (needs a library built with ARC_HC_TRACE=1; read with trace_dump())."""
        assert len(ranks) <= MAX_RANKS
        if not 0 <= pipe_stages <= 8:
            raise ValueError(f"pipe_stages must be 0..8, got {pipe_stages}")
        self.pipe_stages = pipe_stages
        self.lib = ctypes.CDLL(str(lib_path))
        self.lib.hc_launch.restype = ctypes.c_int
        self.lib.hc_launch.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                                       ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_void_p]
        self.lib.hc_register.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.lib.hc_device_pointer.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        self.lib.hc_ag_dp.restype = ctypes.c_int
        self.lib.hc_ag_dp.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                                      ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.hc_rs_owner.restype = ctypes.c_int
        self.lib.hc_rs_owner.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
                                         ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                         ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                         ctypes.c_int, ctypes.c_void_p]
        self.lib.hc_rs_owner_fp8.restype = ctypes.c_int
        self.lib.hc_rs_owner_fp8.argtypes = self.lib.hc_rs_owner.argtypes[:-1] + [ctypes.c_int, ctypes.c_void_p]
        self.lib.hc_ag_fp8q.restype = ctypes.c_int
        self.lib.hc_ag_fp8q.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                        ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int,
                                        ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                        ctypes.c_void_p]
        self.lib.hc_launch_pipe.restype = ctypes.c_int
        self.lib.hc_launch_pipe.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_void_p]
        self.lib.hc_ag_fp8q_pipe.restype = ctypes.c_int
        self.lib.hc_ag_fp8q_pipe.argtypes = self.lib.hc_ag_fp8q.argtypes[:-1] + [ctypes.c_int, ctypes.c_void_p]
        self.lib.hc_ag_dp_fp8q.restype = ctypes.c_int
        self.lib.hc_ag_dp_fp8q.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                                           ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                           ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                           ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                           ctypes.c_void_p, ctypes.c_void_p]
        self.lib.hc_rs_owner_fp8_pipe.restype = ctypes.c_int
        self.lib.hc_rs_owner_fp8_pipe.argtypes = self.lib.hc_rs_owner.argtypes[:-1] + [ctypes.c_int, ctypes.c_int,
                                                                                       ctypes.c_void_p]
        self.lib.hc_region_opts.restype = ctypes.c_int
        self.lib.hc_region_opts.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.hc_trace_build.restype = ctypes.c_int
        self.lib.hc_pair_check_i64.restype = ctypes.c_int
        self.lib.hc_pair_check_i64.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_void_p]
        self.lib.hc_ag_sum.restype = ctypes.c_int
        self.lib.hc_ag_sum.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                       ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                       ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                       ctypes.c_int, ctypes.c_void_p]
        self.lib.hc_rs_sum_owner.restype = ctypes.c_int
        self.lib.hc_rs_sum_owner.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                             ctypes.c_int64, ctypes.c_int64, ctypes.c_int, ctypes.c_void_p,
                                             ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64,
                                             ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_void_p]
        self.group = group
        self.ranks = ranks
        self.nranks = len(ranks)
        self.rank = ranks.index(dist.get_rank())
        self.slot_bytes = (slot_bytes + 4095) // 4096 * 4096
        self.max_blocks = max_blocks
        flag_bytes = 2 * MAX_RANKS * max_blocks * 32
        data_bytes = 2 * MAX_RANKS * self.slot_bytes
        self.total = 2 * flag_bytes + 2 * data_bytes
        shm_name = f"arc_hc_{name}_{os.getpid() if self.rank == 0 else 0}"
        names = [shm_name if self.rank == 0 else None]
        # ARC_HC_LOCAL_SLOTS=1: every rank first-touches the pages it writes (its flag pages and its data / data2
        # slots; peers only read them) from a CPU on its GPU's NUMA node. Default: rank 0 touches the whole region,
        # so on a two-socket host the ranks on the other socket write across it (Runpod 10-08 hc_buffer_probe:
        # GPU1-3 wrote their slots at 1.2-2.9 GB/s while reads ran at 26 GB/s).
        local = os.environ.get("ARC_HC_LOCAL_SLOTS", "0")
        if local not in ("0", "1"):
            raise ValueError(f"ARC_HC_LOCAL_SLOTS must be 0 or 1, got {local!r}")
        local = local == "1"
        if local and os.environ.get("ARC_HC_NUMA_NODE") is not None:
            raise ValueError("ARC_HC_LOCAL_SLOTS and ARC_HC_NUMA_NODE both decide page placement; set one")
        self.placement = None
        if self.rank == 0:
            self.shm = shared_memory.SharedMemory(name=shm_name, create=True, size=self.total)
            node = os.environ.get("ARC_HC_NUMA_NODE")
            if local:
                pass  # every rank touches its own pages below
            elif node is None:
                ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf)), 0, self.total)
            else:
                # first touch decides the page's NUMA node: zero the region from that node's CPUs
                prev = os.sched_getaffinity(0)
                os.sched_setaffinity(0, _node_cpus(int(node)))
                try:
                    ctypes.memset(ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf)), 0, self.total)
                finally:
                    os.sched_setaffinity(0, prev)
        dist.broadcast_object_list(names, src=ranks[0], group=group)
        if self.rank != 0:
            self.shm = shared_memory.SharedMemory(name=names[0])
        if local:
            self.placement = self._touch_own_pages(flag_bytes, data_bytes)
        dist.barrier(group=group)
        base = ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf))
        err = self.lib.hc_register(base, self.total)
        if err != 0:
            raise RuntimeError(f"cudaHostRegister failed: {err}")
        dev = ctypes.c_void_p()
        err = self.lib.hc_device_pointer(base, ctypes.byref(dev))
        if err != 0:
            raise RuntimeError(f"cudaHostGetDevicePointer failed: {err}")
        d = dev.value
        self.flags = d
        self.flags2 = d + flag_bytes
        self.data = d + 2 * flag_bytes
        self.data2 = self.data + data_bytes
        self.epoch = torch.zeros(1, dtype=torch.int32, device="cuda")
        self.fold_epoch = fold_epoch
        self.done = torch.zeros(1, dtype=torch.int32, device="cuda") if fold_epoch else None
        self.trace = None
        if trace:
            if self.lib.hc_trace_build() != 1:
                raise RuntimeError(f"{lib_path} was not built with HC_TRACE (ARC_HC_TRACE=1)")
            self.trace = torch.zeros(TRACE_CALLS * TRACE_BLOCKS * 8, dtype=torch.int64, device="cuda")
        err = self.lib.hc_region_opts(self.flags, None if self.done is None else self.done.data_ptr(),
                                      None if self.trace is None else self.trace.data_ptr())
        if err != 0:
            raise RuntimeError(f"hc_region_opts error {err} (more than 16 regions in this process?)")
        dist.barrier(group=group)
        if self.rank == 0:
            self.shm.unlink()  # name no longer needed; mappings stay valid

    def _own_chunks(self, flag_bytes: int, data_bytes: int) -> list[tuple[int, int]]:
        """(offset, bytes) of the regions this rank writes: hostcomm.cu flag_at / slot_at for [parity][rank]."""
        per_flag = flag_bytes // (2 * MAX_RANKS)
        if per_flag % 4096 or self.slot_bytes % 4096:
            raise RuntimeError(f"flag chunk {per_flag} / slot {self.slot_bytes} not page-aligned: pages would be shared")
        chunks = []
        for region, size in ((0, per_flag), (flag_bytes, per_flag), (2 * flag_bytes, self.slot_bytes),
                             (2 * flag_bytes + data_bytes, self.slot_bytes)):
            for parity in (0, 1):
                chunks.append((region + (parity * MAX_RANKS + self.rank) * size, size))
        return chunks

    def _touch_own_pages(self, flag_bytes: int, data_bytes: int, attempts: int = 60) -> dict:
        node = _gpu_numa_node()
        base = ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf))
        chunks = self._own_chunks(flag_bytes, data_bytes)
        own = sum(size for _, size in chunks) // 4096
        prev = os.sched_getaffinity(0)
        tries = []
        for attempt in range(1, attempts + 1):
            if node >= 0:
                os.sched_setaffinity(0, _node_cpus(node))
            try:
                for off, size in chunks:
                    ctypes.memset(base + off, 0, size)
            finally:
                os.sched_setaffinity(0, prev)
            # This process has faulted in only its own pages so far (peers touch theirs in their own processes), so
            # the mapping's per-node counts are exactly where its pages landed.
            counts = _mapped_pages_per_node(base)
            for _ in range(20):
                # a page kcompactd / NUMA balancing is migrating at this instant is briefly not mapped (Runpod 10-11
                # b2: 32770 of 32772 at startup); re-read for up to 1 s, a lasting mismatch still raises below
                if sum(counts.values()) == own:
                    break
                time.sleep(0.05)
                counts = _mapped_pages_per_node(base)
            if sum(counts.values()) != own:
                raise RuntimeError(f"ARC_HC_LOCAL_SLOTS: rank {self.rank} touched {own} pages but the mapping holds "
                                   f"{counts}")
            if node < 0 or counts.get(f"N{node}", 0) == own:
                return {"rank": self.rank, "gpu_node": node, "own_pages": own, "pages_per_node": counts,
                        "attempts": attempt, "remote_tries": tries}
            # The node sat under its low watermark (zone_reclaim_mode 0 falls back to the other node instead of
            # reclaiming page cache; Runpod 10-10: node 1 at ~7 GB free of 396 GB, 364 GB page cache, while the
            # notebook unpacked the runtime). mbind / move_pages are refused by the container's seccomp, so free
            # these pages (shmem hole punch), let kswapd catch up, and touch again.
            tries.append({"pages_per_node": counts, "node_free_kb": _node_free_kb(node)})
            for off, size in chunks:
                if _libc.madvise(ctypes.c_void_p(base + off), ctypes.c_size_t(size), _MADV_REMOVE) != 0:
                    raise OSError(ctypes.get_errno(), "madvise(MADV_REMOVE) on a hostcomm slot")
            left = _mapped_pages_per_node(base)
            if sum(left.values()) != 0:
                raise RuntimeError(f"ARC_HC_LOCAL_SLOTS: rank {self.rank} MADV_REMOVE left {left} mapped")
            time.sleep(1.0)
        raise RuntimeError(f"ARC_HC_LOCAL_SLOTS: rank {self.rank} touched {own} pages on node {node}'s CPUs "
                           f"{attempts} times and they never all landed there (memory policy or node full?): "
                           f"{tries[-3:]}")

    def launch(self, kind: str, x: torch.Tensor, out: torch.Tensor, blocks: int, threads: int = 512):
        assert x.is_contiguous() and out.is_contiguous() and x.dtype == torch.bfloat16
        n = x.numel()
        need = n * 2 if kind.endswith("bf16") else n + (n // 128) * 4
        if need > self.slot_bytes:
            raise ValueError(f"slot too small: need {need} have {self.slot_bytes}")
        if n % (128 if kind.endswith("fp8") else 8):
            raise ValueError("bad alignment")
        stream = torch.cuda.current_stream().cuda_stream
        if self.pipe_stages and kind == "ar1_fp8v":
            err = self.lib.hc_launch_pipe(KINDS[kind], x.data_ptr(), out.data_ptr(), n, self.flags, self.data,
                                          self.slot_bytes, self.max_blocks, self.rank, self.nranks,
                                          self.epoch.data_ptr(), blocks, threads, self.pipe_stages, stream)
            if err != 0:
                raise RuntimeError(f"hc_launch_pipe error {err}")
            return out
        err = self.lib.hc_launch(KINDS[kind], x.data_ptr(), out.data_ptr(), n, self.flags, self.data, self.data2,
                                 self.flags2, self.slot_bytes, self.max_blocks, self.rank, self.nranks,
                                 self.epoch.data_ptr(), blocks, threads, stream)
        if err != 0:
            raise RuntimeError(f"hc_launch error {err}")
        return out

    def rs_rows(self, x: torch.Tensor, out: torch.Tensor, start: torch.Tensor, count: torch.Tensor,
                blocks: int = 8, threads: int = 512):
        """out[r] = sum over ranks of x[start + r] for r < count, else 0 (start/count are device scalars)."""
        assert x.dtype == torch.bfloat16 and out.dtype == torch.bfloat16 and x.is_contiguous() and out.is_contiguous()
        assert x.dim() == 2 and out.dim() == 2 and x.shape[1] == out.shape[1]
        assert start.dtype == count.dtype and start.dtype in (torch.int32, torch.int64)
        if x.numel() * 2 > self.slot_bytes:
            raise ValueError("slot too small for rs_rows")
        if not hasattr(self.lib, "_rs_ready"):
            self.lib.hc_rs_rows.restype = ctypes.c_int
            self.lib.hc_rs_rows.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                            ctypes.c_int64, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int,
                                            ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                                            ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            self.lib._rs_ready = True
        err = self.lib.hc_rs_rows(x.data_ptr(), out.data_ptr(), x.shape[0], out.shape[0], x.shape[1],
                                  start.data_ptr(), count.data_ptr(), start.element_size(), self.flags, self.data,
                                  self.slot_bytes, self.max_blocks, self.rank, self.nranks, self.epoch.data_ptr(),
                                  blocks, threads, torch.cuda.current_stream().cuda_stream)
        if err != 0:
            raise RuntimeError(f"hc_rs_rows error {err}")
        return out

    def ag_dp(self, xfull: torch.Tensor, out: torch.Tensor, blocks: int = 16, threads: int = 512,
              mismatch: torch.Tensor | None = None):
        """4 ranks = pairs (0,1), (2,3): out = [x0|x1|x2|x3] where x_r = xfull.tensor_split(2)[r % 2] of rank r.

        xfull (this DP's rows, bitwise replicated within the pair) supplies the own pair's shards; only the other
        pair's shards are read from host memory. mismatch (int64[1]) counts words where the sibling's published
        shard differs from the local copy."""
        assert self.nranks == 4 and xfull.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
        assert xfull.is_contiguous() and out.is_contiguous() and xfull.numel() % 2 == 0
        n = xfull.numel() // 2
        assert out.numel() == 4 * n
        if mismatch is not None:
            assert mismatch.dtype == torch.int64 and mismatch.numel() == 1 and mismatch.is_cuda
        err = self.lib.hc_ag_dp(xfull.data_ptr(), out.data_ptr(), n, self.flags, self.data, self.slot_bytes,
                                self.max_blocks, self.rank, self.nranks, self.epoch.data_ptr(), blocks, threads,
                                None if mismatch is None else mismatch.data_ptr(),
                                torch.cuda.current_stream().cuda_stream)
        if err != 0:
            raise RuntimeError(f"hc_ag_dp error {err}")
        return out

    def trace_dump(self) -> torch.Tensor:
        """The HC_TRACE timestamps, int64 [TRACE_CALLS, TRACE_BLOCKS, 8] on the CPU: row (epoch % TRACE_CALLS) of the
        call, block, phase k (0 start, 1 writes done, 2 published, 3 wait done, 4 stage-2 writes done, 5 stage-2
        published, 6 stage-2 wait done, 7 end; a kernel records the phases it has, 0 = not recorded)."""
        if self.trace is None:
            raise RuntimeError("HostComm(trace=True) not set")
        return self.trace.view(TRACE_CALLS, TRACE_BLOCKS, 8).cpu()

    def rs_owner(self, x: torch.Tensor, out: torch.Tensor, start: torch.Tensor, count: torch.Tensor,
                 blocks: int = 16, threads: int = 512, fp8: bool = False, stage2_fp8: bool = False,
                 sib_wait: bool = False):
        """4 ranks = pairs (0,1), (2,3), MAX_LEN layout (DP d at rows [d*L, (d+1)*L), L = x.shape[0] // 2):
        out[r] = sum over ranks of x[d*L + r] for r < count, else 0; bitwise equal to ar_bf16_twoshot + dp_scatter.
        start/count are device scalars (start must be d*L; the kernel traps otherwise).
        fp8: the partials sent to a quarter's owner travel as e4m3 + one fp32 scale per 128 (owner's own partial
        exact, fp32 rank-order sum); the pair exchange stays bf16, so both ranks of a pair get the same bytes.
        stage2_fp8 (fp8 only, lossy): the pair exchange travels as e4m3 + scales too and the owner keeps the
        dequantized values, so the pair still agrees bitwise. sib_wait (fp8 only): the exchange waits for the
        sibling's flag only (the pipelined kernel always does)."""
        assert self.nranks == 4 and x.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
        assert x.is_contiguous() and out.is_contiguous() and x.dim() == 2 and out.dim() == 2
        assert x.shape[0] % 2 == 0 and out.shape[0] == x.shape[0] // 2 and x.shape[1] == out.shape[1]
        assert start.dtype == count.dtype and start.dtype in (torch.int32, torch.int64)
        if x.numel() * 2 > self.slot_bytes:
            raise ValueError("slot too small for rs_owner")
        if fp8 and x.shape[1] % 128:
            raise ValueError(f"rs_owner fp8: hidden {x.shape[1]} is not a multiple of the 128-value scale group")
        if (stage2_fp8 or sib_wait) and not fp8:
            raise ValueError("rs_owner stage2_fp8 / sib_wait are options of the fp8 kernel")
        args = (x.data_ptr(), out.data_ptr(), x.shape[0], x.shape[1], start.data_ptr(), count.data_ptr(),
                start.element_size(), self.flags, self.data, self.data2, self.flags2, self.slot_bytes, self.max_blocks,
                self.rank, self.nranks, self.epoch.data_ptr(), blocks, threads)
        stream = torch.cuda.current_stream().cuda_stream
        mode = int(stage2_fp8) | 2 * int(sib_wait)
        if fp8 and self.pipe_stages:
            err = self.lib.hc_rs_owner_fp8_pipe(*args, self.pipe_stages, mode, stream)
        elif fp8:
            err = self.lib.hc_rs_owner_fp8(*args, mode, stream)
        else:
            err = self.lib.hc_rs_owner(*args, stream)
        if err != 0:
            raise RuntimeError(f"hc_rs_owner{'_fp8' if fp8 else ''} error {err}")
        return out

    def ag_sum(self, xloc: torch.Tensor, out: torch.Tensor, n0: int, n1: int, blocks: int = 16, threads: int = 512):
        """4 ranks = pairs (0,1), (2,3), SUM_LEN layout (DP 0 at global rows [0, n0), DP 1 at [n0, n0 + n1); host
        ints): out = the global rows, the own DP's from xloc (its first n_d rows, bitwise replicated in the pair),
        the other DP's read from the other pair's slots; rows >= n0 + n1 zero. Bitwise equal to the zero-filled
        buffer's AR4 (dp_attention._dp_gather_via_all_reduce)."""
        assert self.nranks == 4 and xloc.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
        assert xloc.is_contiguous() and out.is_contiguous() and xloc.dim() == 2 and out.dim() == 2
        nd = n1 if self.rank >= 2 else n0
        assert xloc.shape[0] >= nd and out.shape[0] >= n0 + n1 and xloc.shape[1] == out.shape[1]
        if (n0 + n1) * out.shape[1] * 2 > self.slot_bytes:
            raise ValueError(f"slot too small for ag_sum: {n0}+{n1} rows x {out.shape[1]}, slot {self.slot_bytes}")
        err = self.lib.hc_ag_sum(xloc.data_ptr(), out.data_ptr(), n0, n1, out.shape[0], out.shape[1], self.flags,
                                 self.data, self.slot_bytes, self.max_blocks, self.rank, self.nranks,
                                 self.epoch.data_ptr(), blocks, threads, torch.cuda.current_stream().cuda_stream)
        if err != 0:
            raise RuntimeError(f"hc_ag_sum error {err}")
        return out

    def rs_sum_owner(self, x: torch.Tensor, out: torch.Tensor, n0: int, n1: int, fp8: bool = False, blocks: int = 16,
                     threads: int = 512):
        """SUM_LEN MoE combine (layout as ag_sum): out[r] = sum over ranks of x[s_d + r] for r < n_d, else 0; bf16 is
        bitwise equal to ar_bf16_twoshot + dp_scatter. fp8: the partials sent to a quarter's owner travel as e4m3 +
        one fp32 scale per 128 (owner's own partial exact); the pair exchange stays bf16."""
        assert self.nranks == 4 and x.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
        assert x.is_contiguous() and out.is_contiguous() and x.dim() == 2 and out.dim() == 2
        nd = n1 if self.rank >= 2 else n0
        assert x.shape[0] >= n0 + n1 and out.shape[0] >= nd and x.shape[1] == out.shape[1]
        if (n0 + n1) * x.shape[1] * 2 > self.slot_bytes:
            raise ValueError(f"slot too small for rs_sum_owner: {n0}+{n1} rows x {x.shape[1]}, slot {self.slot_bytes}")
        if fp8 and x.shape[1] % 128:
            raise ValueError(f"rs_sum_owner fp8: hidden {x.shape[1]} is not a multiple of the 128-value scale group")
        err = self.lib.hc_rs_sum_owner(x.data_ptr(), out.data_ptr(), n0, n1, out.shape[0], x.shape[1], int(fp8),
                                       self.flags, self.data, self.data2, self.flags2, self.slot_bytes,
                                       self.max_blocks, self.rank, self.nranks, self.epoch.data_ptr(), blocks, threads,
                                       torch.cuda.current_stream().cuda_stream)
        if err != 0:
            raise RuntimeError(f"hc_rs_sum_owner error {err}")
        return out

    def ag_fp8q(self, x: torch.Tensor, out: torch.Tensor, qout: torch.Tensor, sout: torch.Tensor, blocks: int = 16,
                threads: int = 512):
        """ag_fp8v (out = the dequantized gather, rank order, own shard too) that also writes the payload as it
        travelled: qout[:nranks*n] e4m3 bytes, sout[:nranks*n/128] fp32 scales (n = x.numel(), a multiple of 128).
        With x rows of whole 128-groups, qout / sout are the [rows, K] FP8 input and [rows, K/128] group scales of
        the gathered rows."""
        n = x.numel()
        assert x.dtype == torch.bfloat16 and out.dtype == torch.bfloat16 and x.is_contiguous() and out.is_contiguous()
        assert out.numel() == n * self.nranks and n % 128 == 0
        assert qout.dtype == torch.uint8 and sout.dtype == torch.float32 and qout.is_contiguous() and sout.is_contiguous()
        assert qout.numel() >= n * self.nranks and sout.numel() >= n * self.nranks // 128
        if n + (n // 128) * 4 > self.slot_bytes:
            raise ValueError(f"slot too small: need {n + (n // 128) * 4} have {self.slot_bytes}")
        args = (x.data_ptr(), out.data_ptr(), qout.data_ptr(), sout.data_ptr(), n, self.flags, self.data,
                self.slot_bytes, self.max_blocks, self.rank, self.nranks, self.epoch.data_ptr(), blocks, threads)
        stream = torch.cuda.current_stream().cuda_stream
        if self.pipe_stages:
            err = self.lib.hc_ag_fp8q_pipe(*args, self.pipe_stages, stream)
        else:
            err = self.lib.hc_ag_fp8q(*args, stream)
        if err != 0:
            raise RuntimeError(f"hc_ag_fp8q error {err}")
        return out

    def ag_dp_fp8q(self, xfull: torch.Tensor, out: torch.Tensor, qout: torch.Tensor, sout: torch.Tensor,
                   blocks: int = 16, threads: int = 512, mismatch: torch.Tensor | None = None,
                   check_ctr: torch.Tensor | None = None, check_every: int = 1, checked: torch.Tensor | None = None):
        """ag_fp8q of x_r = xfull.tensor_split(2)[r % 2] under DP attention (4 ranks, pairs (0,1), (2,3)), bitwise equal
        to it when xfull is bitwise replicated in the pair: the sibling's shard is quantized locally, only the other
        pair's two shards are read from host memory. Pipelined with pipe_stages (1 step when 0). mismatch (int64[1]):
        add the 32-bit words of the sibling's published payload and scales that differ from the local quantization.
        Same slot bytes, flag words and chunks as ag_fp8q at the same blocks and pipe_stages (pipe_stages 0: the
        original kernel's one flag per block), so the ranks of one gather may mix the two calls.
        check_ctr (int64[1] device call counter, advanced by every call that passes it): check only one call in
        check_every, at a position that moves by one every check_every calls (decided on the device, so it rotates
        under graph replay); checked (int64[1]) += 1 per checked call."""
        assert self.nranks == 4 and xfull.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
        assert xfull.is_contiguous() and out.is_contiguous() and xfull.numel() % 256 == 0
        n = xfull.numel() // 2
        assert out.numel() == 4 * n
        assert qout.dtype == torch.uint8 and sout.dtype == torch.float32 and qout.is_contiguous() and sout.is_contiguous()
        assert qout.numel() >= 4 * n and sout.numel() >= 4 * n // 128
        for t in (mismatch, check_ctr, checked):
            if t is not None:
                assert t.dtype == torch.int64 and t.numel() == 1 and t.is_cuda
        if (check_ctr is not None or checked is not None) and mismatch is None:
            raise ValueError("check_ctr / checked need mismatch")
        if check_every < 1:
            raise ValueError(f"check_every must be >= 1, got {check_every}")
        if n + (n // 128) * 4 > self.slot_bytes:
            raise ValueError(f"slot too small: need {n + (n // 128) * 4} have {self.slot_bytes}")
        err = self.lib.hc_ag_dp_fp8q(xfull.data_ptr(), out.data_ptr(), qout.data_ptr(), sout.data_ptr(), n, self.flags,
                                     self.data, self.slot_bytes, self.max_blocks, self.rank, self.epoch.data_ptr(),
                                     blocks, threads, max(1, self.pipe_stages),
                                     None if mismatch is None else mismatch.data_ptr(),
                                     None if check_ctr is None else check_ctr.data_ptr(), check_every,
                                     None if checked is None else checked.data_ptr(),
                                     torch.cuda.current_stream().cuda_stream)
        if err != 0:
            raise RuntimeError(f"hc_ag_dp_fp8q error {err}")
        return out

    def ag_words(self, x: torch.Tensor, blocks: int = 1) -> torch.Tensor:
        """All-gather a 1-D tensor of 2/4/8-byte elements as raw bytes: [nranks, x.numel()] of x.dtype (rank order).
        Small packets only (one block); pads to the gather's 16-byte vectors."""
        assert x.is_cuda and x.dim() == 1 and x.element_size() in (2, 4, 8)
        pad = (-x.numel()) % (16 // x.element_size())
        src = torch.cat([x, x.new_zeros(pad)]) if pad else x.contiguous()
        out = torch.empty((self.nranks, src.numel()), dtype=x.dtype, device=x.device)
        self.launch("ag_bf16", src.view(torch.bfloat16), out.view(-1).view(torch.bfloat16), blocks)
        return out[:, :x.numel()]

    def pair_check_i64(self, packet: torch.Tensor, tag: int) -> None:
        """All-gather the int64 packet, whose last word is this rank's validity flag (0 = valid), and fail a device
        assert (hc_pair_check_i64) unless every rank's copy equals rank 0's and no flag is set. No host sync.
        The attention-TP pair (DP2) or the whole TP4 group (pure TP4, ARC_HISPARSE_TP4)."""
        assert self.nranks >= 2 and packet.dtype == torch.int64 and packet.is_cuda and packet.dim() == 1
        if packet.numel() % 2:  # 16-byte vectors: pad before the flag word, which must stay last
            packet = torch.cat([packet[:-1], packet.new_zeros(1), packet[-1:]])
        g = self.ag_words(packet)
        err = self.lib.hc_pair_check_i64(g.data_ptr(), packet.numel(), self.nranks, tag,
                                         torch.cuda.current_stream().cuda_stream)
        if err != 0:
            raise RuntimeError(f"hc_pair_check_i64 error {err}")

    def all_reduce(self, x, kind="ar1_bf16", blocks=32, threads=512):
        out = torch.empty_like(x)
        return self.launch(kind, x, out, blocks, threads)

    def all_gather(self, x, kind="ag_bf16", blocks=32, threads=512):
        out = torch.empty((x.shape[0] * self.nranks,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device)
        return self.launch(kind, x, out, blocks, threads)


class PairP2P:
    """One-shot FP8 all-reduce of an attention pair over CUDA peer access (hostcomm.cu fp8v_kernel<0, P2P>): each rank
    writes its quantized rows straight into the sibling's device region (IPC mapping) and reads only its own. Bitwise
    ar1_fp8v. Needs peer access between the pair's GPUs (or both ranks on one GPU, for a code-path test)."""

    def __init__(self, lib_path: Path, group, ranks: list[int], slot_bytes: int, max_blocks: int = 128):
        if len(ranks) != 2:
            raise ValueError(f"PairP2P is for a pair, got ranks {ranks}")
        self.lib = ctypes.CDLL(str(lib_path))
        for name, args in (("hc_p2p_alloc", [ctypes.c_int64, ctypes.POINTER(ctypes.c_void_p)]),
                           ("hc_p2p_free", [ctypes.c_void_p]), ("hc_ipc_handle", [ctypes.c_void_p, ctypes.c_char_p]),
                           ("hc_ipc_open", [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]),
                           ("hc_ipc_close", [ctypes.c_void_p]),
                           ("hc_ar_fp8v_p2p", [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p,
                                               ctypes.c_void_p, ctypes.c_int64, ctypes.c_int, ctypes.c_int,
                                               ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p])):
            fn = getattr(self.lib, name)
            fn.restype = ctypes.c_int
            fn.argtypes = args
        self.lib.hc_p2p_bytes.restype = ctypes.c_int64
        self.lib.hc_p2p_bytes.argtypes = [ctypes.c_int64, ctypes.c_int]
        self.rank = ranks.index(dist.get_rank())
        self.slot_bytes = (slot_bytes + 255) // 256 * 256
        self.max_blocks = max_blocks
        self.bytes = self.lib.hc_p2p_bytes(self.slot_bytes, max_blocks)
        local = ctypes.c_void_p()
        err = self.lib.hc_p2p_alloc(self.bytes, ctypes.byref(local))
        if err != 0:
            raise RuntimeError(f"hc_p2p_alloc error {err}")
        self.local = local.value
        handle = ctypes.create_string_buffer(64)
        err = self.lib.hc_ipc_handle(self.local, handle)
        if err != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle error {err}")
        handles = [None, None]
        dist.all_gather_object(handles, handle.raw, group=group)
        peer = ctypes.c_void_p()
        err = self.lib.hc_ipc_open(handles[1 - self.rank], ctypes.byref(peer))
        if err != 0:
            raise RuntimeError(f"cudaIpcOpenMemHandle error {err} (no peer access between the pair's GPUs?)")
        self.peer = peer.value
        self.epoch = torch.zeros(1, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        dist.barrier(group=group)

    def all_reduce(self, x: torch.Tensor, blocks: int = 32, threads: int = 512) -> torch.Tensor:
        assert x.is_contiguous() and x.dtype == torch.bfloat16
        n = x.numel()
        if n % 128 or n + (n // 128) * 4 > self.slot_bytes:
            raise ValueError(f"PairP2P: {n} elements need a multiple of 128 and {n + n // 32} slot bytes "
                             f"(have {self.slot_bytes})")
        out = torch.empty_like(x)
        err = self.lib.hc_ar_fp8v_p2p(x.data_ptr(), out.data_ptr(), n, self.local, self.peer, self.slot_bytes,
                                      self.max_blocks, self.rank, self.epoch.data_ptr(), blocks, threads,
                                      torch.cuda.current_stream().cuda_stream)
        if err != 0:
            raise RuntimeError(f"hc_ar_fp8v_p2p error {err}")
        return out
