"""INT3 GPTQ of the bf16 routed experts, one worker per GPU, one Hessian window at a time.

Weights come from the bf16 checkpoint's fused banks (``mlp.experts.gate_up_proj``
``[E, 2I, H]`` = gate rows then up rows, ``mlp.experts.down_proj`` ``[E, H, I]``).
Hessians come from gptq_hess_hook.py dumps of the pruned calibration model (EP ranks
hold contiguous slices of the renumbered experts); ``--mask`` maps pruned index k to
the original expert ``keep[layer][k]`` (prune_experts.py keeps ascending order).
Format is Humming uint3 without zero point: dequant ``(code - 4) * scale``, signed bf16 scale.
Rare experts get rank-deficient Hessians (a few dozen routed tokens vs 2560 inputs), so
every expert's H gets ``--prior-tokens`` pseudo-tokens of the layer-pooled per-token
second moment (down: rescaled to the expert's own trace); experts with 10k+ tokens are
essentially unchanged.

Per layer the worker writes ``parts/int3-LXX.safetensors`` with, per bank
(``gate_up`` / ``down``): ``codes`` (uint8, two 4-bit codes per byte, low nibble =
even column) and ``scale`` (bf16, ``[E, out, in/128]``), plus
``layer-XX.json`` with error metrics. Deployment packing is a separate step.

    python quantize_int3_gptq.py --src <bf16 dir> --mask <reap mask .pt> \\
        --hess-dir <window dump> --parts <dir> --run-id <id> --layers 0:16 \\
        --worker-id 0 --world-size 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BANKS = (("gate_up", "gate_up_proj", "H_gateup", "nsamp_gateup"),
         ("down", "down_proj", "H_down", "nsamp_down"))


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    if codes.shape[-1] % 2 or codes.max() > 15:
        raise ValueError("nibble packing needs an even width and codes < 16")
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    return torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)


def dequant_int3(packed: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """bf16 ``(code - 4) * scale`` per 128-column group, computed in fp32."""
    codes = unpack_nibbles(packed).float()
    e, n, k = codes.shape
    return ((codes.view(e, n, k // 128, 128) - 4) * scale.float()[..., None]).view(e, n, k).to(torch.bfloat16)


def load_hessians(hess_dir: Path, layer: int, run_id: str, n_experts: int):
    blobs = sorted(hess_dir.glob(f"hess_rank*_L{layer:02d}.pt"))
    if not blobs:
        raise RuntimeError(f"no Hessian dump for layer {layer} in {hess_dir}")
    loaded = [torch.load(p, map_location="cpu", weights_only=False, mmap=True) for p in blobs]
    loaded.sort(key=lambda b: b["rank"])
    if [b["rank"] for b in loaded] != list(range(len(loaded))):
        raise RuntimeError(f"layer {layer}: ranks {[b['rank'] for b in loaded]} are not 0..n-1")
    for b in loaded:
        if b["layer"] != layer or b["run_id"] != run_id or b["n_experts"] != n_experts:
            raise RuntimeError(f"layer {layer}: mixed Hessian snapshot {b['layer']} {b['run_id']}")
        if b["n_local"] * len(loaded) != n_experts:
            raise RuntimeError(f"layer {layer}: {len(loaded)} ranks x {b['n_local']} != {n_experts}")
        for _, _, hk, nk in BANKS:
            if b[hk] is None or b[nk] is None:
                raise RuntimeError(f"layer {layer} rank {b['rank']}: missing {hk}")
    return {key: torch.cat([b[key] for b in loaded]) for _, _, h, n in BANKS for key in (h, n)}


def pooled_prior(h: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    """Layer-pooled per-token second moment ``sum_e H_e / sum_e n_e`` (fp32, on GPU)."""
    total = torch.zeros(h.shape[1:], dtype=torch.float64, device="cuda")
    for begin in range(0, h.shape[0], 32):
        total += h[begin:begin + 32].to("cuda", torch.float64).sum(0)
    return (total / n.sum()).float()


def shrink(h: torch.Tensor, n: torch.Tensor, pooled: torch.Tensor, k: float, own_trace: bool):
    """``H + k * prior`` and ``n + k`` for one GPU batch."""
    prior = pooled.expand_as(h)
    if own_trace:
        own = h.diagonal(dim1=1, dim2=2).sum(-1) / n.to(h.device).clamp_min(1)
        scale = torch.where(n.to(h.device) > 0, own / pooled.trace(), torch.ones_like(own))
        prior = prior * scale[:, None, None]
    return h + k * prior, n + k


def quantize_bank(bank: str, full: torch.Tensor, h: torch.Tensor, n: torch.Tensor,
                  pooled: torch.Tensor, *, batch_experts: int, damping: float, prior_tokens: float):
    """GPTQ one bank ``[E, out, in]`` (bf16, CPU) against fp32 ``h [E, in, in]`` with ``n`` tokens.

    Returns nibble-packed codes, bf16 scales ``[E, out, in/128]`` and per-batch reports.
    """
    from gptq_int3 import gptq_batch
    codes, scales, reports = [], [], []
    for begin in range(0, full.shape[0], batch_experts):
        sl = slice(begin, begin + batch_experts)
        w = full[sl].to("cuda")
        hb, nb = shrink(h[sl].to("cuda"), n[sl], pooled, prior_tokens, own_trace=bank == "down")
        c, s, z, report = gptq_batch(w, hb, nb, bits=3, sym=True, mse=True, damping=damping)
        del hb
        if not torch.all(z == 4) or not torch.equal(s, s.bfloat16().float()):
            raise RuntimeError("symmetric bf16-scale invariant broken")
        codes.append(pack_nibbles(c).cpu())
        scales.append(s.bfloat16().cpu())
        reports.append({"bank": bank, "experts": [begin, sl.stop],
                        "nsamp_min": float(n[sl].min()), **report})
        del w, c, s, z
    return torch.cat(codes), torch.cat(scales), reports


def summarize(reports: list[dict]) -> dict:
    """Per-bank mean error metrics and RTN-fallback counts for ``layer-XX.json``."""
    return {"mean": {bank: {k: sum(r[k] for r in reports if r["bank"] == bank) /
                            sum(1 for r in reports if r["bank"] == bank)
                            for k in ("rel_h_eval", "rtn_h_eval", "rel_w")}
                     for bank, *_ in BANKS},
            "n_rtn_experts": {bank: sum(r["n_rtn_experts"] for r in reports if r["bank"] == bank)
                              for bank, *_ in BANKS},
            "metrics": reports}


def run_worker(args) -> None:
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    drop = torch.load(args.mask, map_location="cpu", weights_only=False)["drop"]
    if args.source_parts is not None:
        idx = None
        n_source = drop.shape[1]
    else:
        idx = json.loads((args.src / "model.safetensors.index.json").read_text())["weight_map"]
        n_source = json.loads((args.src / "config.json").read_text())["text_config"]["num_experts"]
    if drop.shape[1] != n_source:
        raise RuntimeError(f"mask {tuple(drop.shape)} does not match {n_source} source experts")
    keep = [(~drop[layer]).nonzero().flatten() for layer in range(drop.shape[0])]
    n_experts = len(keep[0])
    if any(len(k) != n_experts for k in keep):
        raise RuntimeError("mask keeps a different expert count per layer")
    lo, hi = (int(v) for v in args.layers.split(":"))
    args.parts.mkdir(parents=True, exist_ok=True)
    for layer in range(lo + args.worker_id, hi, args.world_size):
        start = time.perf_counter()
        staged = None
        if args.source_parts is not None:
            staged = torch.load(args.source_parts / f"layer_{layer:02d}.pt", map_location="cpu",
                                weights_only=True, mmap=True)
            if staged["layer"] != layer or not torch.equal(staged["orig_expert_ids"].long(), keep[layer]):
                raise RuntimeError(f"layer {layer}: staged BF16 expert identity differs from mask")
        hess = load_hessians(args.hess_dir, layer, args.run_id, n_experts)
        output, reports = {"orig_expert_ids": keep[layer].to(torch.int32)}, []
        for bank, tensor_name, hkey, nkey in BANKS:
            pooled = pooled_prior(hess[hkey], hess[nkey])
            key = f"model.language_model.layers.{layer}.mlp.experts.{tensor_name}"
            if staged is not None:
                full = staged[bank]
            else:
                with safe_open(args.src / idx[key], framework="pt") as f:
                    full = f.get_tensor(key)
            expected_experts = n_experts if staged is not None else n_source
            if full.shape[0] != expected_experts or full.dtype != torch.bfloat16:
                raise RuntimeError(f"{key}: unexpected {full.dtype} {tuple(full.shape)}")
            if staged is None:
                full = full[keep[layer]]
            output[f"{bank}.codes"], output[f"{bank}.scale"], bank_reports = quantize_bank(
                bank, full, hess[hkey], hess[nkey], pooled, batch_experts=args.batch_experts,
                damping=args.damping, prior_tokens=args.prior_tokens)
            reports += bank_reports
            del full
        path = args.parts / f"int3-L{layer:02d}.safetensors"
        save_file(output, str(path) + ".tmp", metadata={"format": "pt", "run_id": args.run_id})
        Path(str(path) + ".tmp").replace(path)
        report = {"layer": layer, "worker": args.worker_id, "run_id": args.run_id,
                  "format": "humming-uint3-g128-nozp", "damping": args.damping,
                  "prior_tokens": args.prior_tokens, "batch_experts": args.batch_experts,
                  "seconds": time.perf_counter() - start, "bytes": path.stat().st_size,
                  **summarize(reports)}
        atomic_json(args.parts / f"layer-{layer:02d}.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "metrics"}), flush=True)
        del hess, output, staged


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True, help="bf16 checkpoint")
    ap.add_argument("--source-parts", type=Path,
                    help="local BF16 layer_XX.pt files from direct_checkpoint stage-bf16; avoids NFS mmap")
    ap.add_argument("--mask", type=Path, required=True, help="REAP mask .pt (drop=True)")
    ap.add_argument("--hess-dir", type=Path, required=True)
    ap.add_argument("--parts", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--layers", required=True, help="lo:hi")
    ap.add_argument("--worker-id", type=int, required=True)
    ap.add_argument("--world-size", type=int, required=True)
    ap.add_argument("--batch-experts", type=int, default=32)
    ap.add_argument("--damping", type=float, default=0.01)
    ap.add_argument("--prior-tokens", type=float, default=256.0)
    run_worker(ap.parse_args())


if __name__ == "__main__":
    main()
