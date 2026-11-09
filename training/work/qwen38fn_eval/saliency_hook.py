"""Accumulate per-expert saliency on sglang's MoE runner, for REAP pruning to choose experts.

## Why capture it this way

REAP needs `‖f_j‖` -- the norm of the expert output **before router-weight scaling**. On Qwen3.8-Flash-Next's default
serving path this value does not exist at the Python level at all: `flashinfer_trtllm` defers even the top-k selection
into the CUDA kernel (`TopK.forward_cuda` returns a `BypassedTopKOutput` carrying only `hidden_states` and
`router_logits`), so

**`--enable-return-routed-experts` and `--expert-distribution-recorder-mode` are silently ineffective with this
backend** -- their capture points are buried in `select_experts()`, and that path is never called. The symptom is an
empty or stale buffer, not an error.

With `--moe-runner-backend triton` two things hold at once:

1. topk uses the `STANDARD` format; `topk_ids` / `topk_weights` are available at the Python level.
2. The down-GEMM result lands in `intermediate_cache3`, shape `(tokens, topk, hidden)`, and `mul_routed_weight=True`
   (default), so **`intermediate_cache3[t,j]` is exactly `g_j · f_j(x_t)`**. Hence `‖f_j‖ = ‖g_j·f_j‖ / g_j`, and all
   five accumulators can be recovered.

And sglang already has an **official extension point** that exposes it: `fused_experts_impl`'s
`hooks.after_down(intermediate_cache2, intermediate_cache3, topk_weights, topk_ids)` (used by LoRA). Attaching a hook
has a side effect that is exactly what we want --
`_use_intermediate = not no_combine and (topk != 1 or hooks)`:
**as long as hooks is not None, `intermediate_cache3` is always materialized**.

So no monkeypatching of the kernel wrapper is needed; `MoeRunner._maybe_build_lora_hooks` only has to return the hook
from here when there is no LoRA. That change lives in
`patches/sglang-qwen4-pr36497/0002-saliency-hook.patch`.

## Statistics key = (target|mtp, layer_id)

The MTP draft is another MoE in the same process (`mtp.layers.*`); its `layer_id` starts at 0 and collides with the
main model's layer 0. The accumulators therefore use `(kind, layer_id)`, dumped as two files:
`saliency_rank{N}_target.pt` and `saliency_rank{N}_mtp.pt`.
kind comes from `FusedMoE.layer_name` (the patch puts `layer=self` into MoeRunnerConfig).

MTP must be on, running actual draft rollouts. Passes are further split into `draft_extend` / `decode`
(target: `extend` / `target_verify` / `decode`). CUDA graphs are captured per mode, so the pass is read from
`ForwardBatch.forward_mode` in Python at capture time -- replay uses the scatter_add destinations recorded then.

## warmup / padding

padding: `topk_ids` of -1 or ≥ n_experts, or router weight ≈ 0 -- the `valid` mask is a static-shape tensor op,
correct under graph replay too.

warmup: the Python control flow of CUDA graph capture and of the start-up dummy forward cannot be used to skip
scatter_add (skipped, it is not recorded into the graph, and real decodes later would not accumulate either). Instead
it multiplies by a 0-d GPU gate: 0 at server start-up, filled with 1 after SIGUSR2 `arm()`. The capture-time dummy
multiplies by 0; replays after arm multiply by 1.

## Why EP matters

Under expert parallelism (`--ep 4`) experts are **split by id** and each expert lives whole on one rank, so
`intermediate_cache3` is the full `g·f`. Pure TP is different -- `down_proj` is split along the intermediate dimension,
each GPU holds only a partial contribution to hidden, `‖partial‖ ≠ ‖full‖`, and saliency measured that way is wrong.
**This is correct only under EP**, hence the hard assertion below.

## Output format

Deliberately aligned with the blob schema of `work/expert_prune/accumulate_saliency.py`, so that
`work/expert_prune/select_experts.py` can consume it directly. Two differences, both recorded in the blob:

- The `supervised` slot does not exist (this is a serving-time capture, no labels); slot 1 = slot 0.
- `sum_gfs` (EASY-EP) is unavailable -- it needs the MoE block's output and input to compute `1-cos`,
  and `after_down` happens before combine. Left as zeros, so the `easyep` criterion is unusable.
  This project's criterion is REAP, which is unaffected.

Also carries `cumulants_by_pass`: the accumulators per pass; `cumulants` is their sum, used by REAP.
"""

from __future__ import annotations

import os
import sys
import threading

import torch

_LOCK = threading.Lock()
_STATE: dict = {}
FIELDS = ("counts", "sum_g", "sum_f", "sum_f2", "sum_gf")
KINDS = ("target", "mtp")
# target: extend / target_verify / decode
# mtp:    draft_extend / decode
PASSES = ("extend", "target_verify", "decode", "draft_extend")


def enabled() -> bool:
    return bool(os.environ.get("ARC_SALIENCY_OUT"))


def _rank() -> int:
    """TP rank. Each rank is a separate process with its own accumulators; dumps must be kept apart."""
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:                       # noqa: BLE001 — fall back to the pid; never block the measurement
        pass
    return -os.getpid()


def _register_dump_once():
    """Write the accumulators out when the process ends. The server has no "measurement done" API; this is the only
    way."""
    if _STATE.get("_atexit"):
        return
    import atexit
    _STATE["_atexit"] = True

    def _on_exit():
        if not _STATE.get("layers"):
            return
        out = os.environ["ARC_SALIENCY_OUT"]
        os.makedirs(out, exist_ok=True)
        try:
            written = dump_dir(
                out, model=os.environ.get("ARC_SALIENCY_MODEL", ""), tag="capture"
            )
            print(f"[arc-forge] saliency written {written}", flush=True)
        except Exception as e:                # noqa: BLE001
            print(f"[arc-forge] saliency write failed: {type(e).__name__}: {e}", flush=True)

    atexit.register(_on_exit)
    # atexit runs only on a clean exit, and the server is taken down by a signal -- it cannot be the only path.
    # SIGUSR1 dumps; SIGUSR2 arms (warmup gate → 1). Accumulation cannot be skipped with a Python branch at capture,
    # otherwise graph replay would not accumulate either.
    try:
        import signal
        signal.signal(signal.SIGUSR1, lambda *_: _on_exit())
        signal.signal(signal.SIGUSR2, lambda *_: arm())
    except (ValueError, OSError):
        # signals cannot be registered from a non-main thread; atexit is still there, and this must not kill the
        # measurement
        pass


def kind_from_config(config) -> str:
    """target vs mtp: from FusedMoE.layer_name (`mtp.layers.*` vs the main model)."""
    layer = getattr(config, "layer", None)
    name = getattr(layer, "layer_name", None) or ""
    if not name:
        raise RuntimeError(
            "saliency hook needs MoeRunnerConfig.layer.layer_name to tell MTP "
            f"from the main model (layer_id={getattr(config, 'layer_id', None)}). "
            "Check that patches/sglang-qwen4-pr36497/0002-saliency-hook.patch passes "
            "layer=self into MoeRunnerConfig"
        )
    parts = name.replace("/", ".").split(".")
    if parts[0] == "mtp" or "mtp" in parts:
        return "mtp"
    return "target"


def _forward_batch_from_stack():
    """CUDA graphs are split by mode: ForwardBatch is readable at capture time; replay no longer runs Python."""
    frame = sys._getframe()
    while frame is not None:
        fb = frame.f_locals.get("forward_batch")
        if fb is not None and hasattr(fb, "forward_mode"):
            return fb
        frame = frame.f_back
    return None


def _pass_from_batch(kind: str, fb) -> str | None:
    """None = skip (idle). Otherwise a name from PASSES."""
    mode = fb.forward_mode
    is_idle = getattr(mode, "is_idle", None)
    if callable(is_idle) and is_idle():
        return None
    name = getattr(mode, "name", None) or str(mode)
    name = str(name).split(".")[-1].upper()
    if name in ("IDLE",):
        return None
    if kind == "mtp":
        if name in ("DECODE", "PREBUILT"):
            return "decode"
        if name in ("DRAFT_EXTEND_V2", "DRAFT_EXTEND", "EXTEND", "MIXED",
                    "SPLIT_PREFILL", "DLLM_EXTEND"):
            return "draft_extend"
        raise RuntimeError(
            f"MTP MoE hit unexpected forward_mode={name!r} -- "
            "the draft should only be extend / decode; mixing others in would count warmup or target verify into MTP"
        )
    if name in ("EXTEND", "MIXED", "SPLIT_PREFILL", "DLLM_EXTEND"):
        return "extend"
    if name == "TARGET_VERIFY":
        return "target_verify"
    if name in ("DECODE", "PREBUILT"):
        return "decode"
    raise RuntimeError(f"target MoE unexpected forward_mode={name!r}")


def _gate(device) -> torch.Tensor:
    """0-d float64 on `device`. Capture records reads of this buffer; replays count only after arm()."""
    gates = _STATE.setdefault("gates", {})
    t = gates.get(device)
    if t is None:
        t = torch.zeros((), dtype=torch.float64, device=device)
        if _STATE.get("armed"):
            t.fill_(1.0)
        gates[device] = t
    return t


def arm() -> None:
    """SIGUSR2: start accumulating once start-up warmup / CUDA graph capture is over."""
    _STATE["armed"] = True
    for t in _STATE.get("gates", {}).values():
        t.fill_(1.0)
    print("[arc-forge] saliency armed (only forwards from now on count)", flush=True)


def _accum(kind: str, layer_id: int, pass_name: str, n_experts: int, device) -> dict:
    st = _STATE.setdefault("layers", {})
    key = (kind, layer_id, pass_name)
    if key not in st:
        st[key] = {
            f: torch.zeros(n_experts, dtype=torch.float64, device=device) for f in FIELDS
        }
    return st[key]


def make_hooks(config):
    """Returns a hook object with `after_down`, or None (when off, behavior is not affected at all)."""
    if not enabled():
        return None
    from sglang.srt.lora.lora_moe_runners import LoRAHooks

    layer_id = config.layer_id
    n_experts = config.num_experts
    n_local = config.num_local_experts
    if layer_id is None or n_experts is None:
        raise RuntimeError(
            "saliency hook needs MoeRunnerConfig to carry layer_id and num_experts, "
            f"got layer_id={layer_id} num_experts={n_experts}"
        )
    kind = kind_from_config(config)
    # Under EP each expert lives whole on one GPU; pure TP splits down_proj along the intermediate dimension, so
    # intermediate_cache3 is only a partial sum and the norm is wrong. Stop it loudly here rather than measure garbage.
    _STATE["ep"] = bool(n_local is not None and n_local != n_experts)
    _STATE["n_local_experts"] = n_local
    _STATE["top_k"] = config.top_k
    _register_dump_once()

    def after_down(intermediate_cache2, intermediate_cache3, topk_weights, topk_ids):
        # intermediate_cache3: (tokens, topk, hidden), already multiplied by the router weight
        if intermediate_cache3 is None or intermediate_cache3.dim() != 3:
            raise RuntimeError(
                "saliency hook expects a 3-D intermediate_cache3 (tokens, topk, hidden), "
                f"got {None if intermediate_cache3 is None else tuple(intermediate_cache3.shape)}"
                " -- most likely --enable-fused-moe-sum-all-reduce is on, which makes the down-GEMM write directly "
                "into out_hidden_states, skipping intermediate_cache3"
            )
        fb = _forward_batch_from_stack()
        if fb is None:
            raise RuntimeError(
                "saliency after_down cannot see the ForwardBatch -- "
                "cannot tell draft extend from decode, nor drop idle"
            )
        pass_name = _pass_from_batch(kind, fb)
        if pass_name is None:
            return
        gf = intermediate_cache3.float().norm(dim=-1).double()          # (T,k) = g·‖f‖
        g = topk_weights.float().double()                               # (T,k)
        ids = topk_ids.long().reshape(-1)                               # (T*k,)
        gf_f, g_f = gf.reshape(-1), g.reshape(-1)

        # **Static shapes throughout.** Boolean indexing (`ids[keep]`) has a data-dependent output length and cannot
        # be captured under CUDA graph capture; `int(ids.max())` is a device→host sync and makes capture fail outright.
        # The first version did both; the symptom was "Capture cuda graph failed: device-side assert triggered" --
        # the real cause was padding topk_ids (-1 or >= n_experts) fed into scatter_add_, while the error message only
        # tells you to lower mem-fraction.
        valid = (ids >= 0) & (ids < n_experts) & (g_f > 1e-8)
        safe = torch.where(valid, ids, torch.zeros_like(ids))
        z = torch.zeros_like(gf_f)
        gate = _gate(gf.device)
        w_cnt = torch.where(valid, torch.ones_like(gf_f), z) * gate
        w_g = torch.where(valid, g_f, z) * gate
        w_gf = torch.where(valid, gf_f, z) * gate
        # ‖f‖ = ‖g·f‖ / g. The denominator is clamped, and invalid positions' contributions are already zeroed by where.
        fnorm = torch.where(valid, gf_f / g_f.clamp_min(1e-8), z) * gate
        acc = _accum(kind, layer_id, pass_name, n_experts, gf.device)
        with _LOCK:
            acc["counts"].scatter_add_(0, safe, w_cnt)
            acc["sum_g"].scatter_add_(0, safe, w_g)
            acc["sum_f"].scatter_add_(0, safe, fnorm)
            acc["sum_f2"].scatter_add_(0, safe, fnorm * fnorm)
            acc["sum_gf"].scatter_add_(0, safe, w_gf)
            # token count and id range accumulated as GPU tensors, no host sync (capture compatibility again).
            st = _STATE.setdefault("meta", {})
            dev = gf.device
            tok_key = (kind, pass_name, "tokens")
            hi_key = (kind, "id_hi")
            if tok_key not in st:
                st[tok_key] = torch.zeros((), dtype=torch.float64, device=dev)
            if hi_key not in st:
                st[hi_key] = torch.zeros((), dtype=torch.long, device=dev)
            # in-place: an out-of-place assignment would write into a discarded temporary under CUDA graph replay
            st[tok_key].add_(gate * float(gf.shape[0]))
            st[hi_key].copy_(
                torch.maximum(st[hi_key], torch.where(valid, ids, safe).max())
            )
    return LoRAHooks(after_down=after_down)


def _empty_cum(n_layers: int, n_experts: int) -> dict:
    cum = {
        f: torch.zeros(n_layers, 2, n_experts, dtype=torch.float64) for f in FIELDS
    }
    cum["sum_gfs"] = torch.zeros(n_layers, 2, n_experts, dtype=torch.float64)
    return cum


def _add_layer_into(cum: dict, layer_id: int, acc: dict) -> None:
    for f in FIELDS:
        v = acc[f].cpu()
        cum[f][layer_id, 0] += v
        cum[f][layer_id, 1] += v          # no supervised slot, see the module docstring


def _blob_for_kind(kind: str, model: str, tag: str) -> dict | None:
    layers = _STATE.get("layers", {})
    keys = [k for k in layers if k[0] == kind]
    if not keys:
        return None
    layer_ids = sorted({k[1] for k in keys})
    passes = sorted({k[2] for k in keys})
    n_layers = max(layer_ids) + 1
    n_experts = layers[keys[0]]["counts"].numel()
    by_pass = {p: _empty_cum(n_layers, n_experts) for p in passes}
    combined = _empty_cum(n_layers, n_experts)
    for _, li, p in keys:
        _add_layer_into(by_pass[p], li, layers[(kind, li, p)])
        _add_layer_into(combined, li, layers[(kind, li, p)])
    scores = _scores_from(combined)
    meta = _STATE.get("meta", {})
    n_tokens = 0
    tokens_by_pass = {}
    for p in passes:
        raw = meta.get((kind, p, "tokens"), 0)
        n = int(raw.item() if torch.is_tensor(raw) else raw)
        tokens_by_pass[p] = n
        n_tokens += n
    id_hi = meta.get((kind, "id_hi"), -1)
    blob = {
        "model": model,
        "tag": tag,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "top_k": _STATE.get("top_k", 0),
        "n_tokens": n_tokens,
        "n_sup": 0,
        "args": {
            "capture": "sglang triton after_down hook",
            "kind": kind,
            "ep": _STATE.get("ep"),
            "n_local_experts": _STATE.get("n_local_experts"),
            "expert_id_max": int(id_hi.item() if torch.is_tensor(id_hi) else id_hi),
            "rank": _rank(),
            "armed": bool(_STATE.get("armed")),
            "passes": passes,
            "tokens_by_pass": tokens_by_pass,
        },
        "manifest": [],
        "cumulants": combined,
        "cumulants_by_pass": by_pass,
        "scores": {"all": scores, "supervised": scores},
        "scores_by_pass": {p: _scores_from(by_pass[p]) for p in passes},
        "mean_s_t": [],
        "_unavailable": [
            "sum_gfs / easyep (after_down precedes combine, so the block output is unavailable)",
            "supervised slot (serving-time capture, no labels)",
        ],
        "_layers_seen": layer_ids,
        "_passes_seen": passes,
    }
    return blob


def dump_dir(out_dir: str, model: str = "", tag: str = "") -> list[str]:
    """Writes `saliency_rank{N}_{kind}.pt`, target and mtp separately."""
    if not _STATE.get("layers"):
        raise RuntimeError("no saliency accumulated -- the hook was never called")
    os.makedirs(out_dir, exist_ok=True)
    written = []
    rank = _rank()
    for kind in KINDS:
        blob = _blob_for_kind(kind, model, tag)
        if blob is None:
            continue
        path = os.path.join(out_dir, f"saliency_rank{rank}_{kind}.pt")
        torch.save(blob, path)
        written.append(path)
    if not written:
        raise RuntimeError("layers is non-empty but no kind was written -- internal keys are broken")
    return written


def dump(path: str, model: str = "", tag: str = "") -> dict:
    """Compatibility with old callers: when path is a single file, write only that kind (guessed from the file name);
    otherwise treat it as a directory."""
    if path.endswith(".pt"):
        kind = "mtp" if path.endswith("_mtp.pt") else "target"
        blob = _blob_for_kind(kind, model, tag)
        if blob is None:
            raise RuntimeError(f"no {kind} saliency")
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        torch.save(blob, path)
        return blob
    dump_dir(path, model=model, tag=tag)
    return _blob_for_kind("target", model, tag) or _blob_for_kind("mtp", model, tag)


def _scores_from(cum: dict) -> dict:
    """S_j(b,α,β) = (1/N_j^b)·Σ_t 1[j∈E_t]·g^α·‖f‖^β, the same formula as expert_prune."""
    c = cum["counts"][:, 0]
    safe = c.clamp_min(1.0)
    out = {
        "frequency": c.clone(),
        "seer": cum["sum_g"][:, 0].clone(),
        "ean": cum["sum_f"][:, 0].clone(),
        "reap": cum["sum_gf"][:, 0] / safe,
        "man": cum["sum_f"][:, 0] / safe,
        "msan": cum["sum_f2"][:, 0] / safe,
        # easyep needs the MoE block's output and input to compute 1-cos, and after_down precedes combine, so it is
        # unavailable. Zeros are left so that expert_prune/select_experts.py's --report runs to completion;
        # **the numbers on that row are meaningless**, do not compare them.
        "easyep": torch.zeros_like(cum["sum_gf"][:, 0]),
    }
    dead = c == 0
    for k in out:
        out[k] = out[k].clone()
        out[k][dead] = 0.0
    return out
