"""Checkpoint page-cache helpers; the measured prefetch reader is carried verbatim.

No GPU/runtime imports. Reads and DONTNEED advice touch only the exact verified flat
shards (model-00001..model-N as the identity lists them: 20 for REAP320 NVFP4, 18 for
the gEAN320 INT3 Kaggle / HF layout). DONTNEED is advisory; callers must measure actual cgroup headroom.
"""
from __future__ import annotations

import contextlib
import hashlib
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

PREFETCH_PROVENANCE = {
    "source_path": "work/qwen38fn_eval/sm120_runtime.py",
    "source_symbol": "prefetch_checkpoint",
    "source_function_sha256": "480f29169e969566f5e40e9aca054366e43eddf35c8cc43d25dd931d53c6463c",
    "source_file_sha256": "ccde322f08de2c4fc53be3b8ab58a214bf1a535b6979be454ab4cdf708409cc4",
    "measured_receipt": "artifacts/arc_agi3_code/20260924-startup-probe/v2-output/kaggle-arm-state.json",
    "workers": 16,
    "chunk_bytes": 256 << 20,
}


def prefetch_checkpoint(source: Path, workers: int = 16,
                        chunk_bytes: int = 256 << 20) -> dict:
    """Warm input-mount page cache without duplicating 95 GB on the root disk.

    Explicit reads finish before the loader starts. Per-range SHA256 records
    exactly what was prefetched; residency is best-effort under kernel eviction.
    """
    started = time.monotonic()
    shards = sorted(source.glob("*.safetensors"))
    if not shards or workers < 1:
        raise ValueError("Expected checkpoint shards and a positive worker count")
    total = sum(p.stat().st_size for p in shards)
    jobs = [(p, offset, min(chunk_bytes, p.stat().st_size - offset))
            for p in shards for offset in range(0, p.stat().st_size, chunk_bytes)]

    def read_range(job):
        path, offset, length = job
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            stream.seek(offset)
            remaining = length
            while remaining:
                block = stream.read(min(remaining, 8 << 20))
                if not block:
                    raise EOFError(f"Short prefetch read: {path}:{offset}")
                digest.update(block)
                remaining -= len(block)
        return {"file": path.name, "offset": offset, "bytes": length,
                "sha256": digest.hexdigest()}

    ranges, loaded, reported = [], 0, started
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(read_range, job) for job in jobs]):
            row = future.result()
            ranges.append(row)
            loaded += row["bytes"]
            if time.monotonic() - reported >= 15:
                reported = time.monotonic()
                print(f"CHECKPOINT_PREFETCH {loaded / 1e9:.1f}/{total / 1e9:.1f} GB "
                      f"elapsed={reported - started:.1f}s threads={workers}", flush=True)
    if loaded != total:
        raise RuntimeError("Prefetch coverage is incomplete")
    seconds = time.monotonic() - started
    return {"mode": "pagecache", "source": str(source), "workers": workers,
            "bytes": total, "seconds": seconds, "GB_per_second": total / seconds / 1e9,
            "ranges": sorted(ranges, key=lambda r: (r["file"], r["offset"]))}


def verified_checkpoint_shards(model: Path, identity: dict) -> tuple[Path, ...]:
    """Require the exact verified flat shards model-00001..model-N, including every recorded byte count."""
    model = Path(model)
    sizes = identity['target_sizes']
    expected = sorted(name for name in sizes if name.endswith('.safetensors'))
    canonical = [f'model-{index:05d}.safetensors' for index in range(1, len(expected) + 1)]
    if not expected or expected != canonical:
        raise ValueError('checkpoint identity must list the contiguous flat shards model-00001..model-N')
    paths = tuple(sorted(model.glob('*.safetensors')))
    if [path.name for path in paths] != expected:
        raise ValueError('checkpoint shard roster differs from verified identity')
    for path in paths:
        size = sizes[path.name]
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0 or not path.is_file() or path.stat().st_size != size:
            raise ValueError(f'checkpoint shard size differs from verified identity: {path.name}')
    return paths


def _prefetch_parameters(workers: int, chunk_bytes: int) -> None:
    if isinstance(workers, bool) or not isinstance(workers, int) or workers <= 0:
        raise ValueError('prefetch workers must be a positive integer')
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes <= 0:
        raise ValueError('prefetch chunk bytes must be a positive integer')


def _parse_cpulist(text: str) -> set[int]:
    cpus = set()
    for part in filter(None, text.strip().split(',')):
        lo, _, hi = part.partition('-')
        cpus.update(range(int(lo), int(hi or lo) + 1))
    return cpus


def roomiest_node(sysfs: Path = Path('/sys/devices/system/node')) -> dict | None:
    """The NUMA node with the most free memory and this process's CPUs on it; None on a one-node host.

    Page cache lands on the node of the CPU whose read faults it in (zone_reclaim_mode 0 then spills to the other node
    rather than reclaim). Runpod L4 10-11: an 89 GB checkpoint cached 80 GiB on node 1 left it 3 GB free, and the
    ranks on node 1 could not place their HostComm slots or host pools there.
    """
    allowed = os.sched_getaffinity(0)
    nodes = []
    for node_dir in sorted(sysfs.glob('node[0-9]*')):
        cpus = _parse_cpulist((node_dir / 'cpulist').read_text()) & allowed
        meminfo = (node_dir / 'meminfo').read_text()
        free = [int(line.split()[3]) for line in meminfo.splitlines() if line.split()[2:3] == ['MemFree:']]
        if len(free) != 1:
            raise RuntimeError(f'no MemFree line in {node_dir}/meminfo')
        nodes.append((int(node_dir.name[4:]), cpus, free[0]))
    if len(nodes) <= 1:
        return None
    node, cpus, _ = max((row for row in nodes if row[1]), key=lambda row: row[2])
    return {'node': node, 'cpus': sorted(cpus), 'free_kb_by_node': {f'N{n}': kb for n, _, kb in nodes}}


@contextlib.contextmanager
def _pinned_to(choice: dict | None):
    """Threads started inside inherit the affinity (the prefetch pool's readers)."""
    if choice is None:
        yield
        return
    previous = os.sched_getaffinity(0)
    os.sched_setaffinity(0, choice['cpus'])
    try:
        yield
    finally:
        os.sched_setaffinity(0, previous)


def _placement(model: Path, identity: dict) -> dict | None:
    """place_on_roomiest_node: drop this checkpoint's cached pages (a reread never moves a cached page off its node;
    after a clean run h30 already dropped them) and pick the node the reads run on."""
    choice = roomiest_node()
    if choice is not None:
        drop_checkpoint_pages(model, identity)
    return choice


@dataclass
class PrefetchHandle:
    thread: threading.Thread
    outcome: dict
    model: Path
    identity: dict


def start_prefetch(model: Path, identity: dict, *, workers: int = 16,
                   chunk_bytes: int = 256 << 20, place_on_roomiest_node: bool = False) -> PrefetchHandle:
    """Start the measured reader on a daemon thread so runtime staging can overlap it."""
    _prefetch_parameters(workers, chunk_bytes)
    verified_checkpoint_shards(model, identity)
    frozen_identity = {'target_sizes': dict(identity['target_sizes'])}
    outcome = {'result': None, 'error': None}
    choice = _placement(Path(model), identity) if place_on_roomiest_node else None
    def worker():
        try:
            with _pinned_to(choice):
                outcome['result'] = {**prefetch_checkpoint(Path(model), workers, chunk_bytes), 'numa_placement': choice}
            verified_checkpoint_shards(model, frozen_identity)
        except BaseException as error:
            outcome['error'] = error
    thread = threading.Thread(target=worker, name='owned-checkpoint-prefetch', daemon=True)
    handle = PrefetchHandle(thread, outcome, Path(model), frozen_identity)
    thread.start()
    return handle


def join_prefetch(handle: PrefetchHandle, *, timeout: float | None = None) -> dict:
    """Wait for coverage and propagate errors; timeout does not claim the reader stopped."""
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('prefetch join timeout must be finite and positive')
    handle.thread.join(timeout)
    if handle.thread.is_alive():
        raise TimeoutError('owned checkpoint prefetch is still running; no load-ready receipt')
    if handle.outcome['error'] is not None:
        raise RuntimeError('owned checkpoint prefetch failed') from handle.outcome['error']
    verified_checkpoint_shards(handle.model, handle.identity)
    result = handle.outcome['result']
    if result is None:
        raise RuntimeError('prefetch stopped without a coverage receipt')
    return {**result, 'stage': 'prefetch', 'provenance': dict(PREFETCH_PROVENANCE)}


def rewarm(model: Path, identity: dict, *, workers: int = 16,
           chunk_bytes: int = 256 << 20, place_on_roomiest_node: bool = False) -> dict:
    """Read once again immediately before loading, promoting first-pass pages on the LRU.

    place_on_roomiest_node: the reads run on the node with the most free memory; pages the prefetch already cached
    there stay (no drop here), pages elsewhere are not moved."""
    _prefetch_parameters(workers, chunk_bytes)
    verified_checkpoint_shards(model, identity)
    choice = roomiest_node() if place_on_roomiest_node else None
    with _pinned_to(choice):
        result = prefetch_checkpoint(Path(model), workers, chunk_bytes)
    verified_checkpoint_shards(model, identity)
    return {**result, 'stage': 'rewarm', 'cold_read_warning': result['GB_per_second'] < 5,
            'numa_placement': choice, 'provenance': dict(PREFETCH_PROVENANCE)}


def drop_checkpoint_pages(model_root: Path, identity: dict) -> dict:
    """Advise DONTNEED on verified shards after all loader ranks finish copying them.

    The caller owns the four-rank barrier. This neither synchronizes ranks nor
    estimates reclaimed RAM; pinned PLE tensors and model weights are untouched.
    """
    paths = verified_checkpoint_shards(model_root, identity)
    descriptors = []
    started = time.monotonic()
    try:
        # Validate all open descriptors before the first advisory call, preserving
        # exact ownership even if a symlink/size changed after the initial scan.
        for path in paths:
            descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
            descriptors.append((path, descriptor))
            if os.fstat(descriptor).st_size != identity['target_sizes'][path.name]:
                raise ValueError(f'opened checkpoint shard size changed: {path.name}')
        for _, descriptor in descriptors:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        for _, descriptor in descriptors:
            os.close(descriptor)
    return {'operation': 'POSIX_FADV_DONTNEED', 'advisory_only': True,
            'files': [path.name for path in paths], 'files_advised': len(paths),
            'bytes_advised': sum(identity['target_sizes'][path.name] for path in paths),
            'seconds': time.monotonic() - started, 'reclaimed_bytes': None}
