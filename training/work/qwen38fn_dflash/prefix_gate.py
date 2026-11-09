"""Prefix gate v3 (docs/plans/qwen38fn_dflash_prefix_gate_v3_20261012.md): after a data token the frozen target would
almost never sample, the block's later supervision -- WTV horizons and the coverage KL alike -- is discounted.

Per token x[t] of the data, u_t is the target's full-vocabulary tail mass at the gate temperature (tokens no more
likely than x[t], ties included; robust_objective.tail_mass_from_logits) from H[t-1], and c_t = min(1, u_t / alpha).
For an anchor a and slot j (label y[j] = x[a+j+1]):

    w_j = c_a * prod_{i<j} c_{a+i+1}

so the anchor's trust scales every slot and a label's trust only the slots after it (the last label's never counts).
Trust floor 0, no Huber, no ramp: the gate is fully on from the first step (a fresh run from an init -- no parent
objective to blend from). The weights are detached; the WTV and coverage numerators are sum(valid * w * term), still
over the step's global anchors and supervised slots (never over sum(w)).

Shapes: anchor tails [A], label tails and valid [A, M]. valid is a contiguous prefix per row.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from robust_objective import CORPORA
from torch import Tensor

VERSION = "prefix_gate_v3"
THRESHOLDS = (1e-1, 1e-2, 1e-4)  # diagnostics: the share of supervised slots whose weight is below each


@dataclass(frozen=True)
class PrefixGateConfig:
    tail_alpha: float
    gate_temperature: float

    def __post_init__(self) -> None:
        if not 0.0 < self.tail_alpha <= 1.0:
            raise ValueError(f"tail_alpha {self.tail_alpha} must be in (0, 1]")
        if not math.isfinite(self.gate_temperature) or self.gate_temperature <= 0:
            raise ValueError(f"gate_temperature {self.gate_temperature} must be positive")

    def spec(self) -> dict:
        """What a run records and a resume must match."""
        return {"version": VERSION, "tail_alpha": self.tail_alpha, "gate_temperature": self.gate_temperature,
                "trust_floor": 0.0, "ramp": "none", "anchor_inclusion": True, "kl_gated": True}


@torch.no_grad()
def prefix_weights(anchor_tail: Tensor, label_tail: Tensor, valid: Tensor, cfg: PrefixGateConfig) -> Tensor:
    """w [A, M] in FP32 (FP64 for FP64 tails). Past a row's valid prefix the label tails are clamped copies: their c is
    1 there, and their weights are never read (callers mask by valid). A non-finite tail inside the valid prefix makes
    its later weights non-finite, so the loss and the gradient norm fail loudly (train.sync_clip_replicated)."""
    if valid.dtype != torch.bool or label_tail.shape != valid.shape or anchor_tail.shape != valid.shape[:1]:
        raise ValueError(f"tails {tuple(anchor_tail.shape)} / {tuple(label_tail.shape)} vs valid {tuple(valid.shape)}")
    dtype = torch.float64 if label_tail.dtype == torch.float64 else torch.float32
    c_anchor = (anchor_tail.to(dtype) / cfg.tail_alpha).clamp(0.0, 1.0)
    c_label = (label_tail.to(dtype) / cfg.tail_alpha).clamp(0.0, 1.0)
    c_label = torch.where(valid, c_label, torch.ones_like(c_label))
    before = torch.cat([torch.ones_like(c_label[:, :1]), c_label[:, :-1].cumprod(-1)], dim=-1)
    return c_anchor[:, None] * before


@torch.no_grad()
def diagnostics(anchor_tail: Tensor, label_tail: Tensor, weights: Tensor, valid: Tensor, cfg: PrefixGateConfig,
                corpus: str) -> dict[str, Tensor]:
    """Additive sums (summed over micro-batches and ranks before any division), the same keys on every rank."""
    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}")
    v = valid.float()
    w = torch.where(valid, weights.float(), torch.zeros_like(v))
    anchor_hit = (anchor_tail < cfg.tail_alpha).float()
    terms = {
        "gate_anchor_hits": anchor_hit.sum(),
        "gate_label_hits": ((label_tail < cfg.tail_alpha) & valid).float().sum(0),
        "gate_weight_sum": w.sum(0),
        "gate_zero_anchors": (weights[:, 0] == 0).float().sum(),
        "gate_nonfinite": (~torch.isfinite(anchor_tail)).float().sum()
        + (~torch.isfinite(label_tail) & valid).float().sum(),
    }
    for threshold in THRESHOLDS:
        terms[f"gate_weight_below_{threshold:g}"] = ((weights < threshold) & valid).float().sum()
    for name in CORPORA:
        mine = name == corpus
        terms[f"gate_weight_sum/{name}"] = w.sum() if mine else w.new_zeros(())
        terms[f"gate_positions/{name}"] = v.sum() if mine else v.new_zeros(())
        terms[f"gate_anchor_hits/{name}"] = anchor_hit.sum() if mine else v.new_zeros(())
        terms[f"gate_blocks/{name}"] = v.new_tensor(float(len(valid))) if mine else v.new_zeros(())
    return terms


def zero_terms(slots: int, device) -> dict[str, Tensor]:
    """diagnostics' keys for a rank with nothing to do this step, plus the two optimized numerators."""
    terms = {k: torch.zeros((), device=device)
             for k in ("accept_opt_num", "cov_opt_num", "gate_anchor_hits", "gate_zero_anchors", "gate_nonfinite")}
    terms |= {k: torch.zeros(slots, device=device) for k in ("gate_label_hits", "gate_weight_sum")}
    terms |= {f"gate_weight_below_{t:g}": torch.zeros((), device=device) for t in THRESHOLDS}
    terms |= {f"gate_{kind}/{name}": torch.zeros((), device=device)
              for name in CORPORA for kind in ("weight_sum", "positions", "anchor_hits", "blocks")}
    return terms


def metrics(report: dict[str, Tensor]) -> dict:
    """The reduced terms as metrics.jsonl fields (train.accept_metrics)."""
    blocks = report["blocks"].clamp_min(1)
    positions = report["positions"].clamp_min(1)
    reach = report["valid_position"].clamp_min(1)
    out = {
        "gate_accept_optimization": (report["accept_opt_num"] / blocks).item(),
        "gate_coverage_kl_optimization": (report["cov_opt_num"] / positions).item(),
        "gate_weight_mean": (report["gate_weight_sum"].sum() / positions).item(),
        "gate_weight_position": (report["gate_weight_sum"] / reach).tolist(),
        "gate_label_hit_position": (report["gate_label_hits"] / reach).tolist(),
        "gate_anchor_hit": (report["gate_anchor_hits"] / blocks).item(),
        "gate_zero_anchor_fraction": (report["gate_zero_anchors"] / blocks).item(),
        "gate_nonfinite": int(report["gate_nonfinite"].item()),
    }
    for threshold in THRESHOLDS:
        out[f"gate_weight_below_{threshold:g}"] = (report[f"gate_weight_below_{threshold:g}"] / positions).item()
    for name in CORPORA:
        corpus_positions = report[f"gate_positions/{name}"]
        if corpus_positions > 0:
            out[f"gate_weight_mean/{name}"] = (report[f"gate_weight_sum/{name}"] / corpus_positions).item()
            out[f"gate_anchor_hit/{name}"] = (report[f"gate_anchor_hits/{name}"]
                                              / report[f"gate_blocks/{name}"].clamp_min(1)).item()
    return out
