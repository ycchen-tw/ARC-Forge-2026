"""Kaggle v16 (K8 DFlash): marlin MoE never runs with moe_block_size 48 (ARC_V16_MARLIN_BLOCK_REMAP=1); 64 instead.
Also points Humming at the runtime's CUDA 13 (ARC_V16_HUMMING_CUDA_ROOT).

v16 v2 died of a CUDA illegal address in the target verify graph, 5-100 min into full codex load, only with DFlash
K8 C32 (Runpod repros, 10-05; mean time to crash ~62 min). K8 at 16 requests/DP is 128 verify rows/DP, so the EP
MoE sees M=256 across both DPs, and fused_marlin_moe's rule (first of 8,16,32,48,64 with M*topk/E/b < 0.9) picks
block 48 only for M in [230, 346). Nothing else that has run long on L4 lands there: K4 and MTP verify2 are M=128
(block 32), prefill chunks pick 64, NoMTP decode is M <= C; and the 10-02 MTP verify4 C64 run (also 128 rows/DP)
crashed the same way while verify2 passed. G4 runs K8 to completion, but its MoE is native NVFP4, not marlin.
(Dropping the attention layers' alt stream, the previous suspect, still crashed at 52 min.)

The block size is a local in fused_marlin_moe, so the two calls downstream of it are remapped consistently:
the package's moe_align_block_size (fused_marlin_moe imports it at call time) and the module's
moe_wna16_marlin_gemm (both GEMMs pass moe_block_size= by keyword). 64 is what the rule picks with 48 removed;
the GEMM is a JIT marlin kernel taking the block as a runtime argument, so compiled caches stay valid. The
workspace is still sized with 48 (more blocks than 64 needs, capped at 4*SMs either way). The drafter's marlin
MoE goes through the same module and is remapped too. Only align calls made while fused_marlin_moe runs are
remapped: Humming's MoE (ARC_HC_HUMMING_MOE) calls the same package-level align with its kernel's block_shape_m,
which must pass through unchanged.

The remap is NOT a fix: Kaggle v5 (this patch live on all four schedulers) still hit the illegal address after
~97 min of codex. Runpod 10-06 with the target's routed experts on Humming INT3 (no marlin in the target; the
drafter's marlin still remapped) ran 4 h clean, hence the INT3 profile (v16-k8-int3-profile.json).

ARC_V16_HUMMING_CUDA_ROOT=<dir>: Humming's CUDA discovery (humming.utils.cuda.find_all_cuda_paths) keeps only the
install rooted there (the runtime's pip nvidia/cu13). Unpatched, Humming takes /usr/local/cuda, which on the
Runpod image is CUDA 12.4 labelled 13.0 (NVRTC then fails on __nv_fp8_e8m0), and the pip cu13 entry (minor None)
always sorts last. Humming compiles in a child process (nvrtc_compile) that dlopens libnvrtc-builtins by soname,
so the install's lib directory goes on LD_LIBRARY_PATH, which that child inherits.

ARC_HOST_POOL_PRIVATE=1 (l4opt 10-11, port of G4's work/qwen38fn_eval/int3_deploy_runtime.py): the pinned host pools
(QSA HiSparse host KV, the PLE table) on private anonymous pages instead of SGLang's MAP_SHARED /dev/zero mapping. The
Runpod L4 host had ~19 GiB of such shmem per scheduler, kcompactd failing ~7.5M migrations/min with no order >= 9 block
free on node 1, and NUMA balancing scanning the schedulers; a pinned shmem page is unmapped (TLB shootdown on every CPU
running the process) before the pin is found. A pinned private anonymous page fails compaction's pinned-anon check
before it is isolated, and NUMA hinting skips DMA-pinned pages of private mappings. MADV_DONTFORK keeps fork() (Humming's
nvrtc child) from copying the pinned range, the reason SGLang chose MAP_SHARED.

This directory sits after hc_inject on the server's PYTHONPATH: hc_inject's sitecustomize chains to the next
sitecustomize on sys.path, which is this one; this one chains on to anything after it.
"""
from __future__ import annotations

import functools
import importlib.abc
import importlib.machinery
import importlib.util
import math
import os
import sys

PACKAGE = "sglang.srt.layers.moe.fused_moe_triton"
MODULE = "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe"
HUMMING_CUDA = "humming.utils.cuda"
REMAP = {48: 64}
_seen = set()
_in_marlin = [0]  # > 0 while the original fused_marlin_moe runs


def _receipt(text):
    print(f"ARC_V16_MARLIN_BLOCK_REMAP {text} pid={os.getpid()}", file=sys.stderr, flush=True)


def _patch_package(module):
    original = module.moe_align_block_size

    @functools.wraps(original)
    def moe_align_block_size(topk_ids, block_size, *args, **kwargs):
        new = REMAP.get(block_size, block_size) if _in_marlin[0] else block_size
        if new != block_size and topk_ids.shape[0] not in _seen:
            _seen.add(topk_ids.shape[0])
            _receipt(f"align M={topk_ids.shape[0]} {block_size}->{new}")
        return original(topk_ids, new, *args, **kwargs)

    module.moe_align_block_size = moe_align_block_size
    _receipt(f"patched {module.__name__}")


def _patch_module(module):
    original = module.moe_wna16_marlin_gemm

    @functools.wraps(original)
    def moe_wna16_marlin_gemm(*args, **kwargs):
        if "moe_block_size" not in kwargs:
            raise RuntimeError("ARC_V16_MARLIN_BLOCK_REMAP: moe_wna16_marlin_gemm called without moe_block_size=")
        if not _in_marlin[0]:
            raise RuntimeError("ARC_V16_MARLIN_BLOCK_REMAP: moe_wna16_marlin_gemm called outside fused_marlin_moe")
        kwargs["moe_block_size"] = REMAP.get(kwargs["moe_block_size"], kwargs["moe_block_size"])
        return original(*args, **kwargs)

    module.moe_wna16_marlin_gemm = moe_wna16_marlin_gemm
    original_moe = module.fused_marlin_moe

    def fused_marlin_moe(*args, **kwargs):
        _in_marlin[0] += 1
        try:
            return original_moe(*args, **kwargs)
        finally:
            _in_marlin[0] -= 1

    module.fused_marlin_moe = fused_marlin_moe
    _receipt(f"patched {module.__name__}")


def _patch_humming_cuda(module):
    root = os.path.realpath(os.environ["ARC_V16_HUMMING_CUDA_ROOT"])
    lib = os.path.join(root, "lib")
    if not os.path.isdir(lib):
        raise RuntimeError(f"ARC_V16_HUMMING_CUDA_ROOT: {lib} missing")
    original = module.find_all_cuda_paths

    @functools.wraps(original)
    def find_all_cuda_paths():
        found = original()
        kept = [entry for entry in found if os.path.realpath(entry["path"]) == root]
        if not kept:
            raise RuntimeError(f"ARC_V16_HUMMING_CUDA_ROOT: {root} is not among Humming's CUDA installs "
                               f"{[entry['path'] for entry in found]}")
        return kept

    module.find_all_cuda_paths = find_all_cuda_paths
    os.environ["LD_LIBRARY_PATH"] = lib + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")
    print(f"ARC_V16_HUMMING_CUDA root={root} pid={os.getpid()}", file=sys.stderr, flush=True)


SPEC_TRACE = "sglang.srt.speculative.arc_dflash.worker"
_trace = {"calls": 0, "mismatches": 0}


def _patch_spec_trace(module):
    """ARC_V16_SPEC_TRACE=1 (debug): at every DFlash prefill-extend / draft / decode-extend call, both ranks of the
    attention-TP pair exchange checksums of the drafter's inputs (target GR hidden states, anchor / bonus tokens) and
    of its proposals; the first ARC_V16_SPEC_TRACE_LIMIT (default 40) mismatches are printed with element-level detail.
    Kaggle v6 / v8 died in the batch gate on 'attention-TP lifecycle mismatch at spec-verify-input' right after a
    matching spec-accept (same target predictions): this says whether the GR the drafter reads already differs.
    The exchange is NCCL on the pair's device group (not hostcomm, which is itself a suspect), outside CUDA capture.

    Kaggle v9 (this compare mode) passed every gate with no mismatch, so the compare's side effects hide the bug. The
    other modes keep one side effect each at the same call sites, to tell which: ARC_V16_SPEC_TRACE=host only
    synchronizes the current stream with the host (the CPU no longer runs ahead of the GPU); =pair only runs a
    one-element NCCL all-gather in the pair (the two GPUs' timelines meet, the CPU is not held); =hostinit is host
    plus one pair all-gather at the first call (NCCL's communicator comes up, as in pair mode, then never again).
    Kaggle: compare and pair passed, host failed exactly like v6 (DP1 spec-verify-input #52)."""
    import json as _json

    import torch
    import torch.distributed as dist

    limit = int(os.environ.get("ARC_V16_SPEC_TRACE_LIMIT", "40"))
    mode = os.environ["ARC_V16_SPEC_TRACE"]
    modes = ("1", "host", "pair", "hostinit", "pair_prefill", "pair_decode", "dump")
    if mode not in modes:
        raise RuntimeError(f"ARC_V16_SPEC_TRACE must be one of {modes}, got {mode!r}")
    # pair_prefill / pair_decode: the pair barrier at the prefill-extend site only, or at the three decode sites only
    sites = {"pair_prefill": ("prefill",), "pair_decode": ("draft-in", "draft-out", "decode-extend")}.get(mode)
    cls = module.DFlashDraftWorker

    def flat(t):
        x = t.detach().contiguous()
        if x.dtype in (torch.bfloat16, torch.float16):
            return x.view(torch.int16).reshape(-1).to(torch.int64)
        if x.dtype == torch.float32:
            return x.view(torch.int32).reshape(-1).to(torch.int64)
        return x.reshape(-1).to(torch.int64)

    def digest(t):
        v = flat(t)
        w = torch.arange(v.numel(), device=v.device, dtype=torch.int64) % 1000003 + 1
        return torch.stack([torch.full((), v.numel(), dtype=torch.int64, device=v.device), (v * w).sum(), v.sum()])

    def gather(group, t):
        out = [torch.empty_like(t) for _ in range(group.world_size)]
        dist.all_gather(out, t.contiguous(), group=group.device_group)
        return out

    dump_state = {}

    def dump(self, where, named):
        """No pair traffic and no device allocation (host sync alone kept the bug, Kaggle v10): copy each tensor to
        the host as is and write its SHA-256 per call to this rank's file; pairs are compared offline."""
        import hashlib

        if "file" not in dump_state:
            out = os.environ.get("ARC_V16_SPEC_TRACE_DIR", "/kaggle/working/qwen38fn_l4/spec_trace")
            os.makedirs(out, exist_ok=True)
            dump_state["file"] = open(os.path.join(out, f"rank{dist.get_rank()}.jsonl"), "w")
            dump_state["n"] = 0
        dump_state["n"] += 1
        if dump_state["n"] > 20000:
            return
        row = {"n": dump_state["n"], "where": where, "cycle": self.cycle}
        for name, t in named:
            if t is None:
                row[name] = None
                continue
            host = t.detach().cpu().contiguous()
            raw = host.view(torch.int16) if host.dtype == torch.bfloat16 else host
            entry = {"shape": list(t.shape), "dtype": str(t.dtype), "contiguous": t.is_contiguous(),
                     "sha": hashlib.sha256(raw.numpy().tobytes()).hexdigest()[:16]}
            if not t.is_floating_point() and t.numel() <= 256:
                entry["values"] = host.reshape(-1).tolist()
            if t.is_floating_point() and t.dim() >= 2 and t.shape[0] <= 256:
                rows = raw.reshape(t.shape[0], -1).numpy()
                entry["row_sha"] = [hashlib.sha256(r.tobytes()).hexdigest()[:8] for r in rows]
            row[name] = entry
        dump_state["file"].write(_json.dumps(row) + "\n")
        dump_state["file"].flush()

    def compare(self, where, named):
        group = self.group
        if group.world_size < 2 or torch.cuda.is_current_stream_capturing():
            return
        if sites is not None and where not in sites:
            return
        if mode == "dump":
            dump(self, where, named)
            return
        if mode != "1":
            _trace["calls"] += 1
            if mode == "hostinit" and _trace["calls"] == 1:
                # one NCCL collective in the pair, then host syncs only: does the pair mode's NCCL communicator
                # setup (its memory) rather than its per-call barrier hide the bug?
                gather(group, torch.zeros(1, dtype=torch.int64, device=torch.cuda.current_device()))
            if mode in ("host", "hostinit"):
                torch.cuda.current_stream().synchronize()
            else:  # pair, pair_prefill, pair_decode
                gather(group, torch.zeros(1, dtype=torch.int64, device=torch.cuda.current_device()))
            if _trace["calls"] <= 3 or _trace["calls"] % 5000 == 0:
                print(f"ARC_V16_SPEC_TRACE {mode} {where} cycle={self.cycle} calls={_trace['calls']} "
                      f"rank={dist.get_rank()}", file=sys.stderr, flush=True)
            return
        names = [n for n, t in named if t is not None]
        tensors = dict(named)
        local = torch.stack([digest(tensors[n]) for n in names])
        peers = gather(group, local)
        differ = [n for i, n in enumerate(names) if any(not torch.equal(peers[0][i], p[i]) for p in peers[1:])]
        _trace["calls"] += 1
        if not differ:
            if _trace["calls"] <= 3 or _trace["calls"] % 1000 == 0:
                print(f"ARC_V16_SPEC_TRACE ok {where} cycle={self.cycle} calls={_trace['calls']} "
                      f"mismatches={_trace['mismatches']} rank={dist.get_rank()}", file=sys.stderr, flush=True)
            return
        _trace["mismatches"] += 1
        if _trace["mismatches"] > limit:
            return
        detail = {}
        for n in differ:  # same list on every rank (computed from the gathered digests): collectives stay paired
            t = tensors[n].detach()
            both = gather(group, t)
            a, b = both[0], both[1]
            diff = a != b
            entry = {"shape": list(t.shape), "dtype": str(t.dtype), "n_diff": int(diff.sum())}
            if t.is_floating_point():
                entry["max_abs_diff"] = float((a.float() - b.float()).abs().max())
                entry["max_abs"] = float(a.float().abs().max())
            if t.dim() >= 1:
                rows = diff.reshape(t.shape[0], -1).any(1).nonzero().flatten()[:16].tolist()
                entry["rows"] = rows
                if not t.is_floating_point():
                    entry["a"] = [a[r].reshape(-1)[:16].tolist() for r in rows[:4]]
                    entry["b"] = [b[r].reshape(-1)[:16].tolist() for r in rows[:4]]
            detail[n] = entry
        print(f"ARC_V16_SPEC_TRACE MISMATCH {where} cycle={self.cycle} calls={_trace['calls']} "
              f"n={_trace['mismatches']} rank={dist.get_rank()} differ={differ} detail={_json.dumps(detail)}",
              file=sys.stderr, flush=True)

    original_prefill = cls._draft_extend_for_prefill
    original_draft = cls.draft
    original_decode = cls._draft_extend_for_decode

    def _draft_extend_for_prefill(self, batch, target_hidden_states, next_token_ids, *args, **kwargs):
        if not batch.forward_mode.is_idle():
            gr = self._unpad(target_hidden_states, batch.input_ids.numel(), "prefill")  # attention-TP padding rows
            compare(self, "prefill", [("input_ids", batch.input_ids), ("target_gr", gr), ("next_token_ids", next_token_ids)])
        return original_prefill(self, batch, target_hidden_states, next_token_ids, *args, **kwargs)

    def draft(self, batch):
        idle = batch.forward_mode.is_idle() or batch.seq_lens.shape[0] == 0
        if not idle:
            compare(self, "draft-in", [("anchor", batch.spec_info.bonus_tokens), ("seq_lens", batch.seq_lens),
                                       ("temperature", batch.sampling_info.temperatures)])
        verify = original_draft(self, batch)
        if not idle:
            compare(self, "draft-out", [("draft_token", verify.draft_token), ("positions", verify.positions)])
        return verify

    def _draft_extend_for_decode(self, batch, batch_result):
        bs = batch.seq_lens.shape[0]
        if not (batch.forward_mode.is_idle() or bs == 0):
            gr = self._unpad(batch_result.logits_output.hidden_states, bs * self.width, "verify")
            compare(self, "decode-extend", [
                ("verify_gr", gr), ("accept_lens", batch_result.accept_lens),
                ("bonus", batch_result.next_draft_input.bonus_tokens), ("verified_draft", batch.spec_info.draft_token)])
        return original_decode(self, batch, batch_result)

    cls._draft_extend_for_prefill = _draft_extend_for_prefill
    cls.draft = draft
    cls._draft_extend_for_decode = _draft_extend_for_decode
    print(f"ARC_V16_SPEC_TRACE patched {module.__name__}.DFlashDraftWorker mode={mode} limit={limit} pid={os.getpid()}",
          file=sys.stderr, flush=True)


HOSTCOMM = "hostcomm"
MODEL_RUNNER = "sglang.srt.model_executor.model_runner"
HC_HOOK = "arc_hostcomm_hook"


def _patch_hc_streams(module):
    """ARC_V16_HC_STREAMS=1 (debug): log every new (group, CUDA stream, capturing, thread) a HostComm collective is
    launched from, with the Python stack. Every launch reads and bumps the group's one device epoch, which only works
    if all of a group's collectives run in one order on one stream; Kaggle v10 (host sync at the drafter's call sites)
    still diverged where a pair-GPU barrier there (debug-pair v1) did not, so a cross-stream collective is a suspect.
    Python-side only: CUDA graph replays do not pass here (their kernels were recorded at capture, which does)."""
    import threading
    import traceback

    import torch

    seen = {}
    cls = module.HostComm

    def wrap(name):
        original = getattr(cls, name)

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            stream = torch.cuda.current_stream().cuda_stream
            key = (tuple(self.ranks), stream, torch.cuda.is_current_stream_capturing(), threading.current_thread().name)
            if key not in seen:
                seen[key] = 0
                print(f"ARC_V16_HC_STREAMS new ranks={list(self.ranks)} op={name} stream={stream:#x} capturing={key[2]} "
                      f"thread={key[3]} keys={len(seen)} pid={os.getpid()}\n" + "".join(traceback.format_stack(limit=9)[:-1]),
                      file=sys.stderr, flush=True)
            seen[key] += 1
            return original(self, *args, **kwargs)

        setattr(cls, name, wrapped)

    for name in ("launch", "rs_rows"):
        wrap(name)
    # graph replays run captured HostComm kernels without passing here: log the streams / threads they replay on
    replays = set()
    original_replay = torch.cuda.CUDAGraph.replay

    def replay(self):
        key = (torch.cuda.current_stream().cuda_stream, threading.current_thread().name)
        if key not in replays:
            replays.add(key)
            print(f"ARC_V16_HC_STREAMS replay stream={key[0]:#x} thread={key[1]} keys={len(replays)} pid={os.getpid()}\n"
                  + "".join(traceback.format_stack(limit=7)[:-1]), file=sys.stderr, flush=True)
        return original_replay(self)

    torch.cuda.CUDAGraph.replay = replay
    print(f"ARC_V16_HC_STREAMS patched {module.__name__}.HostComm and CUDAGraph.replay pid={os.getpid()}",
          file=sys.stderr, flush=True)


def _patch_zero_freed(module):
    """ARC_V16_ZERO_FREED=1 (debug): zero the target's Marlin routed-expert tensors right after the model loads, before
    the Humming MoE (ARC_HC_HUMMING_MOE) replaces and frees them (~3.2 GB per rank, a different expert slice on every
    rank). Kaggle v6/v8/v10 diverged deterministically (same op index) inside an attention-TP pair: if something
    later reads memory it never wrote, it reads this rank's old expert bytes. Humming reloads from the checkpoint and
    never reads the Marlin copies."""
    import torch

    cls = module.ModelRunner
    original = cls.load_model

    def load_model(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        if getattr(self, "is_draft_worker", False):
            return result
        zeroed = layers = 0
        for name, layer in self.model.named_modules():
            if not (hasattr(layer, "w13_weight") and hasattr(layer, "_num_local_routed")
                    and type(layer.quant_method).__name__ == "ModelOptNvFp4FusedMoEMethod") or "mtp" in name:
                continue
            for attr in ("w13_weight", "w2_weight", "w13_weight_scale", "w2_weight_scale"):
                t = getattr(layer, attr)
                t.data.zero_()
                zeroed += t.numel() * t.element_size()
            layers += 1
        torch.cuda.synchronize()
        if layers == 0:
            raise RuntimeError("ARC_V16_ZERO_FREED: no NVFP4 FusedMoE layer found")
        print(f"ARC_V16_ZERO_FREED layers={layers} bytes={zeroed} pid={os.getpid()}", file=sys.stderr, flush=True)
        return result

    cls.load_model = load_model
    print(f"ARC_V16_ZERO_FREED patched {module.__name__}.ModelRunner.load_model pid={os.getpid()}", file=sys.stderr, flush=True)


def _patch_hc_no_attn(module):
    """ARC_V16_HC_NO_ATTN=1 (debug): no HostComm for the attention-TP groups (their collectives fall back to the
    stock path); the TP4 group keeps it. Tells whether the pair's own host-mapped collectives carry the divergence."""
    original = module._setup_group

    def _setup_group(coord, tag):
        if tag.startswith("attn"):
            print(f"ARC_V16_HC_NO_ATTN skipped {tag} ranks={list(coord.ranks)} pid={os.getpid()}", file=sys.stderr, flush=True)
            return None
        return original(coord, tag)

    module._setup_group = _setup_group
    print(f"ARC_V16_HC_NO_ATTN patched {module.__name__}._setup_group pid={os.getpid()}", file=sys.stderr, flush=True)


POOL_HOST_COMMON = "sglang.srt.mem_cache.pool_host.common"
_MADV_POPULATE_WRITE = 23  # Linux 5.14+; the stdlib mmap module may not name it


def _patch_host_pool_private(module):
    """ARC_HOST_POOL_PRIVATE=1: HostTensorAllocator (the QSA host KV pools and the PLE table both allocate through it)
    gets private anonymous pages, every page faulted in, the range not inherited by fork()."""
    import mmap

    import torch
    from sglang.srt.environ import envs
    from sglang.srt.mem_cache.storage.mmap import mmap_allocator

    if module.alloc_mmap is not mmap_allocator.alloc_mmap:
        raise RuntimeError("ARC_HOST_POOL_PRIVATE: pool_host.common no longer calls mmap_allocator.alloc_mmap")
    if (envs.SGLANG_HUGEPAGE_SIZE.get() or "").strip():
        raise RuntimeError("ARC_HOST_POOL_PRIVATE does not cover SGLANG_HUGEPAGE_SIZE hugepage pools")

    def private_host_buffer(dims, dtype):
        count = math.prod(dims)
        alloc_bytes = math.ceil(count * torch.empty([], dtype=dtype).element_size() / mmap.PAGESIZE) * mmap.PAGESIZE
        mm = mmap.mmap(-1, alloc_bytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, prot=mmap.PROT_READ | mmap.PROT_WRITE)
        mm.madvise(mmap.MADV_DONTFORK)
        mm.madvise(getattr(mmap, "MADV_POPULATE_WRITE", _MADV_POPULATE_WRITE))
        print(f"ARC_HOST_POOL_PRIVATE alloc {alloc_bytes} B pid={os.getpid()}", file=sys.stderr, flush=True)
        return torch.frombuffer(mm, dtype=dtype, count=count).reshape(dims)

    module.alloc_mmap = private_host_buffer
    print(f"ARC_HOST_POOL_PRIVATE patched {module.__name__}.alloc_mmap pid={os.getpid()}", file=sys.stderr, flush=True)


MODEL_LOADER = "sglang.srt.model_loader.loader"
WEIGHT_PATCH_SCHEMA = "arcforge.kaggle.weight_patch.v1"


class _WeightPatch:
    """ARC_L4_WEIGHT_PATCH=<dir> (work/qwen38fn_rl/rl_weight_patch.py; port of G4's int3_deploy_runtime.WeightPatch):
    tensors that replace the checkpoint's own under the same name, so an RL version is served off the unchanged
    mounted package. Every patch tensor must arrive exactly once, with the checkpoint's dtype and shape, and hash to
    the sha256 patch.json records; a missing one, a duplicate or a second load raises."""

    def __init__(self, root):
        import json
        self.root = root
        with open(os.path.join(root, "patch.json")) as stream:
            self.manifest = json.load(stream)
        if self.manifest["schema"] != WEIGHT_PATCH_SCHEMA:
            raise ValueError(f"{root}/patch.json is {self.manifest['schema']!r}, not {WEIGHT_PATCH_SCHEMA}")
        self.pending = dict(self.manifest["tensors"])
        self.done = set()
        self.handles = {}

    def replace(self, name, tensor):
        import hashlib

        import torch
        from safetensors import safe_open
        entry = self.pending.pop(name)
        if entry["file"] not in self.handles:
            self.handles[entry["file"]] = safe_open(os.path.join(self.root, entry["file"]), "pt")
        new = self.handles[entry["file"]].get_tensor(name)
        if new.dtype != tensor.dtype or tuple(new.shape) != tuple(tensor.shape):
            raise ValueError(f"weight patch {name}: {new.dtype} {tuple(new.shape)}, the checkpoint has "
                             f"{tensor.dtype} {tuple(tensor.shape)}")
        digest = hashlib.sha256(new.contiguous().reshape(-1).view(torch.uint8).numpy()).hexdigest()
        if digest != entry["sha256"]:
            raise ValueError(f"weight patch {name} is {digest}, patch.json says {entry['sha256']}")
        self.done.add(name)
        return new.to(tensor.device)

    def substitute(self, weights):
        for name, tensor in weights:
            if name in self.done:
                raise ValueError(f"weight patch: {name} arrived twice")
            yield name, (self.replace(name, tensor) if name in self.pending else tensor)

    def finish(self):
        if self.pending:
            raise ValueError(f"weight patch: {len(self.pending)} tensors never arrived: {sorted(self.pending)[:4]}")
        self.handles.clear()
        return len(self.done)


def _patch_model_loader(module):
    """ARC_L4_WEIGHT_PATCH=<dir> with ARC_L4_WEIGHT_PATCH_MODEL=<the served target directory>: DefaultModelLoader's
    weight iterator for that directory (the normal and the startup-overlap load both go through it; the DFlash drafter
    is another directory and passes untouched) yields the patch's tensors in place of the checkpoint's. Once per
    process: a second load of the target would serve the patch over a later weight sync."""
    root = os.environ["ARC_L4_WEIGHT_PATCH"]
    target = os.path.realpath(os.environ["ARC_L4_WEIGHT_PATCH_MODEL"])
    loader = module.DefaultModelLoader
    original = loader._get_weights_iterator
    used = [False]

    @functools.wraps(original)
    def _get_weights_iterator(self, source, *args, **kwargs):
        weights = original(self, source, *args, **kwargs)
        if os.path.realpath(str(source.model_or_path)) != target:
            return weights
        if used[0]:
            raise RuntimeError("ARC_L4_WEIGHT_PATCH covers the first target load only")
        used[0] = True
        return _patched(_WeightPatch(root), weights)

    def _patched(patch, weights):
        yield from patch.substitute(weights)
        count = patch.finish()
        import torch.distributed as dist
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
        print(f"ARC_L4_WEIGHT_PATCH_LOADED tensors={count} global_rank={rank} "
              f"deploy_manifest={patch.manifest['deploy_manifest_sha256']} pid={os.getpid()}", flush=True)

    loader._get_weights_iterator = _get_weights_iterator
    print(f"ARC_L4_WEIGHT_PATCH patched {module.__name__}.DefaultModelLoader._get_weights_iterator "
          f"target={target} pid={os.getpid()}", file=sys.stderr, flush=True)


THINK_BUDGET = "sglang.srt.speculative.arc_think_budget"
THINK_NEG = -1.0e4  # finite: a shard row of all -inf makes sharded_verify's summary kernels produce NaN


def _patch_think_budget(module):
    """ARC_THINK_BUDGET with ARC_SHARDED_VERIFY=1 (x35 12 h Save & Runs 10-31: the server died at the first armed
    request, then at the first forced close).

    run_eagle_verify hands ThinkBudget.apply the target-verify logits before eagle_sample; under ARC_SHARDED_VERIFY
    those are this rank's vocab shard ([rows, 49,152] compact ARC_TARGET_VOCAB ids on L4), but the fork's apply sizes
    its token-class tables by that width and forces rows by vocabulary id (</think> 248069): a device-side index
    assert. Gathering the shards first (the 10-31 v2 patch) costs 236 MiB of FP32 rows at full load, which L4 did not
    have. Here the state machine runs as in the fork (candidates are vocabulary ids; the cls / banned tables are built
    for the FULL vocabulary) and the forcing is written on the shard itself: the forced token's compact id if this
    rank owns it, THINK_NEG everywhere else (both ranks force the same rows: states are identical on the pair). The
    chain rejection sampler then commits exactly that token (PC 2x4090 unit test, 400 trials x 5 cases); batches that
    fall back to gather_full see 0 / THINK_NEG / -inf rows whose argmax is still the forced token."""
    import torch

    from sglang.srt.distributed.parallel_state import get_attn_tp_group
    from sglang.srt.speculative.arc_dflash import target_vocab

    TB = module.ThinkBudget
    original = TB.apply
    ST_OK, ST_NLOK, ST_END, ST_NN, ST_DONE, ST_WAIT = (module.ST_OK, module.ST_NLOK, module.ST_END, module.ST_NN,
                                                      module.ST_DONE, module.ST_WAIT)
    CLS_NORMAL = module.CLS_NORMAL
    ENV = module.ENV

    def shard_tables(self, Vs, pool_rows, device):
        """once per scheduler: this rank's offset, the fork's tables sized by the full vocabulary, the banned mask
        restricted to the shard, and the vocabulary -> compact map (None without ARC_TARGET_VOCAB)"""
        cache = getattr(self, "_arc_shard", None)
        if cache is not None:
            if cache["Vs"] != Vs:
                raise RuntimeError(f"{ENV}: shard width changed {cache['Vs']} -> {Vs}")
            return cache
        group = get_attn_tp_group()
        n, rank = group.world_size, group.rank_in_group
        lo = rank * Vs
        if target_vocab.active():
            t = target_vocab.tables()
            if t["size"] != Vs * n or t["n"] != n or t["rank"] != rank:
                raise RuntimeError(f"{ENV}: shard {Vs} x {n} rank {rank} vs ARC_TARGET_VOCAB tables size {t['size']} "
                                   f"n {t['n']} rank {t['rank']}")
            full, g2c, c2g = t["vocab"], t["g2c"], t["c2g"]
            missing = [tid for tid in (self.nl_id, self.nn_id, self.end_id) if int(g2c[tid]) < 0]  # one sync, once
            if missing:
                raise RuntimeError(f"{ENV}: forced tokens {missing} are not in ARC_TARGET_VOCAB")
        else:
            full, g2c, c2g = Vs * n, None, None
        if full < len(self.tokenizer):
            raise RuntimeError(f"{ENV}: full vocab {full} < tokenizer {len(self.tokenizer)}")
        self._ensure_tensors(full, pool_rows, device)  # cls / banned / state sized by the FULL vocab
        banned = self.banned[c2g[lo:lo + Vs]] if c2g is not None else self.banned[lo:lo + Vs]
        if banned.shape[0] != Vs:
            raise RuntimeError(f"{ENV}: banned shard {banned.shape[0]} != {Vs}")
        self._arc_shard = {"Vs": Vs, "lo": lo, "n": n, "g2c": g2c, "banned": banned.contiguous()}
        return self._arc_shard

    def apply(self, batch, rows, candidates, logits, num_draft_tokens, pool_rows):
        if logits.shape[-1] >= len(self.tokenizer):  # full rows (no sharding): the fork's apply
            return original(self, batch, rows, candidates, logits, num_draft_tokens, pool_rows)
        bs, k, Vs = len(batch.reqs), num_draft_tokens, logits.shape[-1]
        if logits.shape[0] != bs * k or candidates.numel() != bs * k:
            raise RuntimeError(f"{ENV}: verify logits {tuple(logits.shape)} / candidates {candidates.numel()} "
                               f"for {bs} requests x {k} rows")
        device = logits.device
        sh = shard_tables(self, Vs, pool_rows, device)
        # ---- the fork's apply up to `states` (candidates are vocabulary ids; cls is full-vocabulary) ----
        fresh, fresh_state = [], []
        for i in rows:
            req = batch.reqs[i]
            if req.arc_think_slot != req.kv.req_pool_idx:
                req.arc_think_slot = req.kv.req_pool_idx
                fresh.append(req.kv.req_pool_idx)
                fresh_state.append(self._cpu_state(req))
        row_t = torch.tensor(rows, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
        slots = batch.req_pool_indices.index_select(0, row_t).to(torch.int64)
        cand = candidates.view(bs, k).index_select(0, row_t).to(torch.int64)
        if fresh:
            fresh_t = torch.tensor(fresh, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
            val_t = torch.tensor(fresh_state, dtype=torch.int64, pin_memory=True).to(device, non_blocking=True)
            pos = (slots.unsqueeze(0) == fresh_t.unsqueeze(1)).to(torch.int64).argmax(dim=1)
            c0 = cand[pos, 0]
            from_c0 = torch.where(c0 == self.end_id, ST_END,
                                  torch.where(c0 == self.tool_id, ST_DONE,
                                              torch.where(self.cls[c0] == CLS_NORMAL, ST_OK, ST_WAIT)))
            self.state.index_copy_(0, fresh_t, torch.where(val_t >= 0, val_t, from_c0).to(torch.int8))
        st = self.state[slots].to(torch.int64)
        states = [st]
        for j in range(1, k):
            st = self._advance(st, cand[:, j])
            states.append(st)
        states = torch.stack(states, dim=1)  # [n, k]
        force = (states == ST_OK) | (states == ST_NLOK) | (states == ST_END)
        token = torch.where(states == ST_END, torch.full_like(cand, self.nn_id),
                            torch.where(states == ST_NLOK, torch.full_like(cand, self.end_id),
                                        torch.full_like(cand, self.nl_id)))
        # ---- shard write-back: one-hot at the forced token if this rank owns it, THINK_NEG everywhere else ----
        compact = sh["g2c"][token] if sh["g2c"] is not None else token
        local = compact - sh["lo"]
        own = (local >= 0) & (local < Vs)
        view = logits.view(bs, k, Vs)
        sub = view.index_select(0, row_t)  # [n, k, Vs]
        neg = torch.full_like(sub, THINK_NEG)
        hit = torch.where(own, 0.0, THINK_NEG).to(sub.dtype)
        onehot = neg.scatter(2, local.clamp(0, Vs - 1).unsqueeze(2), hit.unsqueeze(2))
        mask = (states == ST_NN).unsqueeze(2) & sh["banned"].view(1, 1, Vs)
        sub = torch.where(mask, neg, sub)
        view.index_copy_(0, row_t, torch.where(force.unsqueeze(2), onehot, sub))
        return row_t, slots

    TB.apply = apply
    print(f"ARC_THINK_BUDGET_SHARDED patched {module.__name__}.ThinkBudget.apply (shard-native) pid={os.getpid()}",
          file=sys.stderr, flush=True)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self, patches):
        self.patches = patches

    def find_spec(self, name, path, target=None):
        patch = self.patches.get(name)
        if patch is None:
            return None
        spec = importlib.machinery.PathFinder.find_spec(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"v16_patch: {name} not found")
        original_exec = spec.loader.exec_module

        def exec_module(module):
            original_exec(module)
            patch(module)

        spec.loader.exec_module = exec_module
        return spec


def _chain_other_sitecustomize():
    """Run the next sitecustomize after this one; skip hc_inject's (it is what chained here) and this directory."""
    skip = {os.path.dirname(os.path.abspath(__file__))}
    if os.environ.get("ARC_V16_HC_INJECT_DIR"):
        skip.add(os.path.abspath(os.environ["ARC_V16_HC_INJECT_DIR"]))
    rest = [p for p in sys.path if os.path.abspath(p or os.curdir) not in skip]
    spec = importlib.machinery.PathFinder.find_spec("sitecustomize", rest)
    if spec is None:
        return
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def _main():
    patches = {}
    if os.environ.get("ARC_V16_MARLIN_BLOCK_REMAP") == "1":
        patches.update({PACKAGE: _patch_package, MODULE: _patch_module})
    if os.environ.get("ARC_V16_HUMMING_CUDA_ROOT"):
        patches[HUMMING_CUDA] = _patch_humming_cuda
    if os.environ.get("ARC_V16_SPEC_TRACE"):
        patches[SPEC_TRACE] = _patch_spec_trace
    if os.environ.get("ARC_V16_HC_STREAMS") == "1":
        patches[HOSTCOMM] = _patch_hc_streams
    if os.environ.get("ARC_V16_HC_NO_ATTN") == "1":
        patches[HC_HOOK] = _patch_hc_no_attn
    if os.environ.get("ARC_V16_ZERO_FREED") == "1":
        patches[MODEL_RUNNER] = _patch_zero_freed
    private = os.environ.get("ARC_HOST_POOL_PRIVATE", "0")
    if private not in ("0", "1"):
        raise ValueError(f"ARC_HOST_POOL_PRIVATE must be 0 or 1, got {private!r}")
    if private == "1":
        patches[POOL_HOST_COMMON] = _patch_host_pool_private
    if os.environ.get("ARC_L4_WEIGHT_PATCH"):
        if not os.path.isfile(os.path.join(os.environ["ARC_L4_WEIGHT_PATCH"], "patch.json")):
            raise FileNotFoundError(f"ARC_L4_WEIGHT_PATCH={os.environ['ARC_L4_WEIGHT_PATCH']} has no patch.json")
        if not os.path.isdir(os.environ["ARC_L4_WEIGHT_PATCH_MODEL"]):
            raise FileNotFoundError(f"ARC_L4_WEIGHT_PATCH_MODEL={os.environ['ARC_L4_WEIGHT_PATCH_MODEL']} "
                                    "is not a directory")
        patches[MODEL_LOADER] = _patch_model_loader
    if os.environ.get("ARC_THINK_BUDGET") and os.environ.get("ARC_SHARDED_VERIFY") == "1":
        patches[THINK_BUDGET] = _patch_think_budget
    loaded = [name for name in patches if name in sys.modules]
    if loaded:
        raise RuntimeError(f"v16_patch: {loaded} imported before the patch could hook them")
    if patches:
        sys.meta_path.insert(0, _Finder(patches))
    _chain_other_sitecustomize()


try:
    _main()
except BaseException:
    # site.py only prints "Error in sitecustomize" and carries on, which would silently drop the patch.
    import traceback
    traceback.print_exc()
    sys.stderr.flush()
    os._exit(97)
