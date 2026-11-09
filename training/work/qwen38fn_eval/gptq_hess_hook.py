"""Accumulate per-expert GPTQ Hessians (XᵀX) on the pruned FP8 MoE.

MXFP4 GPTQ cannot load 176B into transformers. This hook rides the same
LoRA after_gate_up / after_down pipe as REAP saliency. MTP is skipped
(MXFP4 only touches main-model routed experts). CUDA graphs must be off:
the expert loop has data-dependent token counts.

SIGUSR1 freezes collection and requests a dump at the next last-layer forward
boundary. SIGUSR2 arms after warmup. ARC_HESS_DEVICE=cuda retains Grams on GPU.
ARC_HESS_LAYERS=lo:hi accumulates only layers lo..hi-1, so 512-expert Hessians
fit on two GPUs one window per server run.
"""
from __future__ import annotations

import json
import os
import sys
import threading

import torch

# ------- process state

_LOCK = threading.Lock()
_STATE: dict = {}


def enabled() -> bool:
    return bool(os.environ.get("ARC_SALIENCY_OUT"))


def _rank() -> int:
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:  # noqa: BLE001
        pass
    return -os.getpid()


def kind_from_config(config) -> str:
    layer = getattr(config, "layer", None)
    name = getattr(layer, "layer_name", None) or ""
    if not name:
        raise RuntimeError("Hessian hook requires the MoE layer name to exclude MTP")
    parts = name.replace("/", ".").split(".")
    if parts and (parts[0] == "mtp" or "mtp" in parts):
        return "mtp"
    return "target"


def _forward_batch_from_stack():
    frame = sys._getframe()
    while frame is not None:
        fb = frame.f_locals.get("forward_batch")
        if fb is not None and hasattr(fb, "forward_mode"):
            return fb
        frame = frame.f_back
    return None


def _idle(fb) -> bool:
    if fb is None:
        return True
    mode = fb.forward_mode
    is_idle = getattr(mode, "is_idle", None)
    if callable(is_idle) and is_idle():
        return True
    name = str(getattr(mode, "name", mode)).split(".")[-1].upper()
    return name in ("IDLE",)


def arm() -> None:
    torch.set_num_threads(int(os.environ.get("ARC_HESS_CPU_THREADS", "4")))
    torch.backends.cuda.matmul.allow_tf32 = False
    _STATE["armed"] = True
    print("[arc-forge] gptq hessian armed", flush=True)


def _bucket(kind: str, layer_id: int, name: str, n_local: int, dim: int, device):
    st = _STATE.setdefault("H", {})
    key = (kind, layer_id, name)
    if key not in st:
        dest = device if os.environ.get("ARC_HESS_DEVICE", "cpu") == "cuda" else "cpu"
        st[key] = torch.zeros(n_local, dim, dim, dtype=torch.float32, device=dest)
        _STATE.setdefault("nsamp", {})[key] = torch.zeros(
            n_local, dtype=torch.float64
        )
    return st[key]


def _max_nsamp() -> int:
    raw = os.environ.get("ARC_HESS_MAX_NSAMP", "16384")
    value = int(raw)
    if value <= 0:
        raise ValueError("ARC_HESS_MAX_NSAMP must be positive")
    return value


def _layer_window() -> tuple[int, int] | None:
    raw = os.environ.get("ARC_HESS_LAYERS", "").strip()
    if not raw:
        return None
    lo, hi = (int(v) for v in raw.split(":"))
    if not 0 <= lo < hi:
        raise ValueError(f"ARC_HESS_LAYERS must be lo:hi with 0 <= lo < hi, got {raw!r}")
    return lo, hi


def _in_window(layer_id: int) -> bool:
    window = _STATE["window"]
    return window is None or window[0] <= layer_id < window[1]


def _blob_identity() -> dict:
    return {
        "run_id": os.environ.get("ARC_HESS_RUN_ID", ""),
        "src": os.environ.get("ARC_SALIENCY_MODEL", ""),
        "mask_sha256": os.environ.get("ARC_HESS_MASK_SHA256", ""),
        "mtp_mask_sha256": os.environ.get("ARC_HESS_MTP_MASK_SHA256", ""),
    }


def _add_xx_cpu(H: torch.Tensor, nsamp: torch.Tensor, x: torch.Tensor, ids, n_local: int):
    """H[e] += X_eᵀ X_e. Padded batched GEMM + one D2H. Skip saturated experts."""
    x = x.float().detach()
    if ids.dim() == 2 and x.shape[0] == ids.shape[0]:
        per_token = True
        ids_w = ids.detach()
    else:
        per_token = False
        ids_w = ids.reshape(-1).detach()
        if x.shape[0] != ids_w.numel():
            raise RuntimeError(
                f"hess x rows {x.shape[0]} != ids {tuple(ids.shape)}"
            )
    cap = _max_nsamp()
    uniq = torch.unique(ids_w.reshape(-1))
    uniq = uniq[(uniq >= 0) & (uniq < n_local)]
    xs = []
    expert_ids = []
    counts = []
    skipped = 0
    for e in uniq.tolist():
        e = int(e)
        if float(nsamp[e]) >= cap:
            skipped += 1
            continue
        if per_token:
            xe = x[(ids_w == e).any(dim=1)]
        else:
            xe = x[ids_w == e]
        xe = xe[:max(0, cap - int(nsamp[e]))]
        n = xe.shape[0]
        if n == 0:
            continue
        xs.append(xe)
        expert_ids.append(e)
        counts.append(n)
    chunk = 32
    for start in range(0, len(xs), chunk):
        sl = slice(start, start + chunk)
        group = xs[sl]
        max_n = max(t.shape[0] for t in group)
        dim = group[0].shape[1]
        batched = group[0].new_zeros(len(group), max_n, dim)
        for i, xe in enumerate(group):
            batched[i, : xe.shape[0]] = xe
        grams = (batched.transpose(1, 2) @ batched).to(
            device="cpu", dtype=torch.float32,
        )
        for i, e in enumerate(expert_ids[sl]):
            H[e] += grams[i]
            nsamp[e] += counts[start + i]
    st = _STATE.setdefault("coverage", {})
    st["skipped_capped"] = st.get("skipped_capped", 0) + skipped
    st["gram_batches"] = st.get("gram_batches", 0) + 1
    st["gram_experts"] = st.get("gram_experts", 0) + len(expert_ids)


def _add_xx(H: torch.Tensor, nsamp: torch.Tensor, x: torch.Tensor, ids, n_local: int):
    """Sort routes once; form similarly sized batches; retain the Gram on GPU.

    Only O(experts) counts cross to the host. No d-by-d matrices cross PCIe.
    nsamp stays on CPU so cap decisions never synchronize individual experts.
    """
    if H.device.type == "cpu":
        return _add_xx_cpu(H, nsamp, x, ids, n_local)
    cap = _max_nsamp()
    remaining = [max(0, cap - int(v)) for v in nsamp.tolist()]
    if not any(remaining):
        return
    divisor = ids.shape[1] if ids.ndim == 2 and x.shape[0] == ids.shape[0] else 1
    flat = ids.reshape(-1).long()
    if divisor == 1 and x.shape[0] != flat.numel():
        raise RuntimeError("down activation rows do not match routing slots")
    valid = (flat >= 0) & (flat < n_local)
    bucket = torch.where(valid, flat + 1, 0)
    counts = torch.bincount(bucket, minlength=n_local + 1).cpu().tolist()
    order = torch.argsort(bucket, stable=True)
    offsets, offset = [], counts[0]
    for count in counts[1:]:
        offsets.append(offset)
        offset += count
    take = [min(counts[e + 1], remaining[e]) for e in range(n_local)]
    experts = sorted((e for e in range(n_local) if take[e]), key=lambda e: take[e])
    group_size = int(os.environ.get("ARC_HESS_GEMM_BATCH", "8"))
    for begin in range(0, len(experts), group_size):
        es = experts[begin:begin + group_size]
        ei = torch.tensor(es, device=x.device, dtype=torch.long)
        lengths = torch.tensor([take[e] for e in es], device=x.device)
        start = torch.tensor([offsets[e] for e in es], device=x.device)
        col = torch.arange(max(take[e] for e in es), device=x.device)
        positions = (start[:, None] + col[None, :]).clamp_max(flat.numel() - 1)
        rows = order[positions] // divisor
        batch = x[rows].float()
        batch.masked_fill_(col[None, :, None] >= lengths[:, None, None], 0)
        updated = torch.baddbmm(H.index_select(0, ei), batch.transpose(1, 2), batch)
        H.index_copy_(0, ei, updated)
        for e in es:
            nsamp[e] += take[e]
    stats = _STATE.setdefault("coverage", {})
    stats["gram_batches"] = stats.get("gram_batches", 0) + 1
    stats["gram_experts"] = stats.get("gram_experts", 0) + len(experts)


def _request_dump(*_args):
    # Signals must never serialize: repeated signals used to re-enter torch.save.
    _STATE["dump_requested"] = True
    _STATE["armed"] = False


def _maybe_dump(layer_id: int):
    if layer_id != int(os.environ.get("ARC_HESS_N_LAYERS", "48")) - 1:
        return
    if not _STATE.pop("dump_requested", False):
        return
    _STATE["armed"] = False
    dump_dir(os.environ["ARC_SALIENCY_OUT"])


def _dump_root(out: str) -> str:
    """Write Hessians to shm. Never put multi-GB .pt files on NFS."""
    override = os.environ.get("ARC_HESS_SHM", "").strip()
    dest = override or out
    nfs = dest.startswith("/nfs/") or dest.startswith("/mnt/") or "/nfs/" in dest
    if nfs:
        dest = override or f"/dev/shm/qwen38fn-hess-{os.getuid()}"
        print(f"[arc-forge] hessian dump redirected to {dest} (not NFS)", flush=True)
    os.makedirs(dest, exist_ok=True)
    return dest


def dump_dir(out: str, **_kwargs) -> list[str]:
    out = _dump_root(out)
    rank = _rank()
    written = []
    layers = sorted({k[1] for k in _STATE.get("H", {}) if k[0] == "target"})
    n_local = int(_STATE.get("n_local") or 0)
    coverage = {}
    n_zero = 0
    for layer_id in layers:
        ns = _STATE["nsamp"].get(("target", layer_id, "gateup"))
        if ns is None:
            continue
        n_hit = int((ns > 0).sum())
        n_zero += int((ns <= 0).sum())
        coverage[str(layer_id)] = {
            "n_local": n_local,
            "n_hit": n_hit,
            "n_zero": int((ns <= 0).sum()),
            "nsamp_min": float(ns.min()) if ns.numel() else 0,
            "nsamp_max": float(ns.max()) if ns.numel() else 0,
            "nsamp_mean": float(ns.mean()) if ns.numel() else 0,
        }
    ident = _blob_identity()
    meta = {
        "rank": rank,
        "n_local": _STATE.get("n_local"),
        "n_experts": _STATE.get("n_experts"),
        "armed": bool(_STATE.get("armed")),
        "layers": layers,
        "max_nsamp": _max_nsamp(),
        "layer_window": _STATE.get("window"),
        "coverage": coverage,
        "nsamp_zero_local": n_zero,
        "hook_stats": _STATE.get("coverage", {}),
        **ident,
    }
    meta_path = os.path.join(out, f"hess_rank{rank}_meta.json")
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle)
    written.append(meta_path)
    for layer_id in layers:
        payload = {
            "rank": rank,
            "layer": layer_id,
            "n_local": _STATE.get("n_local"),
            "n_experts": _STATE.get("n_experts"),
            "H_gateup": _STATE["H"].get(("target", layer_id, "gateup")),
            "H_down": _STATE["H"].get(("target", layer_id, "down")),
            "nsamp_gateup": _STATE["nsamp"].get(("target", layer_id, "gateup")),
            "nsamp_down": _STATE["nsamp"].get(("target", layer_id, "down")),
            **ident,
        }
        payload = {k: v.cpu() if torch.is_tensor(v) else v for k, v in payload.items()}
        path = os.path.join(out, f"hess_rank{rank}_L{layer_id:02d}.pt")
        tmp = path + ".tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)
        written.append(path)
        print(
            f"[arc-forge] hessian L{layer_id:02d} rank={rank} "
            f"{os.path.getsize(path)/1e9:.2f}G -> {path}",
            flush=True,
        )
    print(
        f"[arc-forge] gptq hessian wrote rank={rank} layers={len(layers)} -> {out}",
        flush=True,
    )
    done = os.path.join(out, f"hess_rank{rank}_done.json")
    with open(done + ".tmp", "w", encoding="utf-8") as handle:
        json.dump({**meta, "files": [os.path.basename(p) for p in written],
                   "pid": os.getpid(), "complete": True}, handle)
    os.replace(done + ".tmp", done)
    return written


def _register_dump_once():
    if _STATE.get("_signals_registered"):
        return
    import signal

    _STATE["_signals_registered"] = True
    # Persistent matrices are saved explicitly, once, at a forward boundary.
    try:
        signal.signal(signal.SIGUSR1, _request_dump)
        signal.signal(signal.SIGUSR2, lambda *_: arm())
    except (ValueError, OSError):
        pass


def make_hooks(config, *, kind_override: str | None = None):
    """kind_override='target' is only for a no-spec, target-only native-runner bridge.

    The bridge must assert exactly one MoE call per target layer per forward.
    Normal serving still requires the model layer name to identify MTP.
    """
    if not enabled():
        return None
    from sglang.srt.lora.lora_moe_runners import LoRAHooks

    layer_id = config.layer_id
    n_experts = config.num_experts
    n_local = config.num_local_experts
    if layer_id is None or n_experts is None or n_local is None:
        raise RuntimeError("gptq hessian hook needs layer_id/num_experts/num_local")
    if n_local == n_experts:
        raise RuntimeError("gptq hessian hook requires EP (local != global)")
    if kind_override is not None and kind_override != "target":
        raise RuntimeError(f"unsupported explicit Hessian kind override {kind_override!r}")
    kind = kind_from_config(config) if kind_override is None else kind_override
    _STATE["n_local"] = n_local
    _STATE["n_experts"] = n_experts
    _STATE["window"] = _layer_window()
    _register_dump_once()

    def after_gate_up(hidden_states, _cache1, _w, topk_ids):
        if kind != "target" or not _STATE.get("armed") or not _in_window(layer_id):
            return
        if _idle(_forward_batch_from_stack()):
            return
        dim = hidden_states.shape[-1]
        H = _bucket(kind, layer_id, "gateup", n_local, dim, hidden_states.device)
        ns = _STATE["nsamp"][(kind, layer_id, "gateup")]
        with _LOCK:
            _add_xx(H, ns, hidden_states, topk_ids, n_local)

    def after_down(cache2, _cache3, _w, topk_ids):
        if kind != "target":
            return
        if not _STATE.get("armed"):
            _maybe_dump(layer_id)
            return
        if not _in_window(layer_id) or _idle(_forward_batch_from_stack()):
            return
        if cache2.dim() == 3:
            x = cache2.reshape(-1, cache2.shape[-1])
        else:
            ntok = topk_ids.shape[0] * topk_ids.shape[1]
            x = cache2[:ntok]
        dim = x.shape[-1]
        H = _bucket(kind, layer_id, "down", n_local, dim, x.device)
        ns = _STATE["nsamp"][(kind, layer_id, "down")]
        with _LOCK:
            _add_xx(H, ns, x, topk_ids.reshape(-1), n_local)
        _maybe_dump(layer_id)

    return LoRAHooks(after_gate_up=after_gate_up, after_down=after_down)


def microbench(device="cpu", n_local=32, dim=256, T=512, topk=8, rounds=8):
    """Single-layer Gram timing. Do not use serving GPUs."""
    import time

    x = torch.randn(T, dim, device=device)
    ids = torch.randint(0, n_local, (T, topk), device=device)
    H = torch.zeros(n_local, dim, dim, dtype=torch.float32)
    nsamp = torch.zeros(n_local, dtype=torch.float64)
    _add_xx(H, nsamp, x, ids, n_local)
    nsamp.zero_()
    H.zero_()
    t0 = time.perf_counter()
    for _ in range(rounds):
        _add_xx(H, nsamp, x, ids, n_local)
    elapsed = time.perf_counter() - t0
    report = {
        "device": str(device),
        "n_local": n_local,
        "dim": dim,
        "T": T,
        "topk": topk,
        "rounds": rounds,
        "seconds": round(elapsed, 4),
        "tok_s": round((T * rounds) / max(elapsed, 1e-9), 1),
        "experts_hit": int((nsamp > 0).sum()),
        "nsamp_mean": round(float(nsamp.mean()), 2),
        "max_nsamp": _max_nsamp(),
        "hook_stats": _STATE.get("coverage", {}),
    }
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--microbench", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if not args.microbench:
        raise SystemExit("pass --microbench")
    microbench(device=args.device)
