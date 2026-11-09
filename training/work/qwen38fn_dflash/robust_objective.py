"""One-sided deficit Huber and exclusive-prefix trust (robust objective v2).

Shapes: p_c, q [A,M,K]; tail_mass, valid [A,M].
Teacher masses p_c retain FULL-vocabulary normalization. q sums to one over K.
Production callers must validate finite inputs and prefix-contiguous valid masks
outside compiled hot paths. All teacher quantities and trust weights are frozen.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import Tensor

@dataclass(frozen=True)
class RobustConfig:
    relative_band: float = 0.10
    tail_alpha: float = 0.01
    trust_floor: float = 0.50
    gate_temperature: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.relative_band <= 1.0:
            raise ValueError("relative_band must be in [0,1]")
        if not 0.0 < self.tail_alpha <= 1.0:
            raise ValueError("tail_alpha must be in (0,1]")
        if not 0.0 <= self.trust_floor <= 1.0:
            raise ValueError("trust_floor must be in [0,1]")
        if not math.isfinite(self.gate_temperature) or self.gate_temperature <= 0:
            raise ValueError("gate_temperature must be positive")

def compute_dtype(x: Tensor) -> torch.dtype:
    return torch.float64 if x.dtype == torch.float64 else torch.float32

@torch.no_grad()
def tail_mass_from_logits(logits: Tensor, labels: Tensor,
                          temperature: float = 1.0) -> Tensor:
    """Reference only: fuse these reductions in the EXISTING teacher head chunk.
    No sorting, no top-K approximation, ties are included. NaNs must be fatal.
    """
    z = logits.detach().to(compute_dtype(logits)) / temperature
    zy = z.gather(-1, labels[..., None])
    log_num = z.masked_fill(z > zy, -torch.inf).logsumexp(-1)
    log_den = z.logsumexp(-1)
    return (log_num - log_den).exp().clamp(0.0, 1.0)

@torch.no_grad()
def prefix_trust(tail_mass: Tensor, valid: Tensor, cfg: RobustConfig) -> Tensor:
    c = (tail_mass.detach() / cfg.tail_alpha).clamp(0.0, 1.0)
    c = torch.where(valid, c, torch.ones_like(c))
    before = torch.cat([torch.ones_like(c[:, :1]), c[:, :-1].cumprod(-1)], dim=-1)
    return cfg.trust_floor + (1.0 - cfg.trust_floor) * before

def one_sided_huber_deficit(d: Tensor, teacher_mass: Tensor,
                            relative_band: float) -> Tensor:
    """Huberized positive deficit d=p-q; zero for q>=p.
    Unit slope for d>=beta, quadratic for 0<d<beta. No outer 1/2.
    The kappa=0 branch uses a symmetric positive-part representation so
    exact ties reproduce torch.minimum's half/half subgradient convention.
    """
    if relative_band == 0.0:
        return 0.5 * (d + d.abs())
    beta = relative_band * teacher_mass.detach()
    inside = (beta > 0) & (d > 0) & (d < beta)
    safe_beta = torch.where(inside, beta, torch.ones_like(beta))
    quadratic = 0.5 * d * (d / safe_beta)
    return torch.where(d <= 0, torch.zeros_like(d),
                       torch.where(inside, quadratic, d - 0.5 * beta))

def sparse_smooth_tv(p_c: Tensor, q: Tensor, relative_band: float) -> Tensor:
    """1-p(C)+sum Huberized (p-q)_+. This is a biased LOCAL surrogate.
    Its forward value must never be substituted for real acceptance.
    """
    p = p_c.detach().to(compute_dtype(q))
    q = q.to(compute_dtype(q))
    return 1.0 - p.sum(-1) + one_sided_huber_deficit(
        p - q, p, relative_band).sum(-1)

def robust_wtv_terms(p_c: Tensor, q: Tensor, tail_mass: Tensor,
                     valid: Tensor, mix: Tensor,
                     cfg: RobustConfig = RobustConfig()) -> dict[str, Tensor]:
    """ADDITIVE sums; caller divides by the GLOBAL number of anchors.
    mix is a scalar tensor, fixed across a global optimizer step, with no grad.
    The ST value is exact raw overlap, but its backward uses smooth local TV.
    Keep accept_num as the raw UNWEIGHTED diagnostic; optimize accept_opt_num.
    """
    if p_c.shape != q.shape or q.ndim != 3:
        raise ValueError("p_c and q must have matching [A,M,K] shapes")
    if valid.shape != q.shape[:2] or valid.dtype != torch.bool:
        raise ValueError("valid must be bool [A,M]")
    if tail_mass.shape != valid.shape or mix.numel() != 1:
        raise ValueError("tail_mass must be [A,M]; mix must be scalar")
    dtype = compute_dtype(q)
    p, qf = p_c.detach().to(dtype), q.to(dtype)
    v = valid.to(dtype)
    raw_a = torch.minimum(p, qf).sum(-1)
    raw_a_valid = torch.where(valid, raw_a, torch.ones_like(raw_a))
    raw_survival = raw_a_valid.cumprod(-1)
    raw_by_anchor = ((1.0 - raw_survival) * v).sum(-1) / q.shape[1]
    tv_h = sparse_smooth_tv(p, qf, cfg.relative_band)
    # Parentheses are deliberate: subtract two IDENTICAL values before adding
    # to raw_a, rather than subtracting large nearly equal totals afterward.
    st_a = raw_a.detach() - (tv_h - tv_h.detach())
    st_a_valid = torch.where(valid, st_a, torch.ones_like(st_a))
    trust = prefix_trust(tail_mass.to(dtype), valid, cfg)
    robust_by_anchor = (trust * v * (1.0 - st_a_valid.cumprod(-1))).sum(-1) / q.shape[1]
    r = mix.detach().to(device=q.device, dtype=dtype)
    opt_num = ((1.0 - r) * raw_by_anchor + r * robust_by_anchor).sum()
    return {
        "accept_opt_num": opt_num,
        "accept_num": raw_by_anchor.detach().sum(),
        "accept_trust_num": robust_by_anchor.detach().sum(),
        "blocks": torch.tensor(q.shape[0], dtype=dtype, device=q.device),
        "raw_overlap": raw_a.detach(),
        "raw_survival": raw_survival.detach() * v,
        "horizon_trust": trust.detach(),
        "tv_h": tv_h,
    }

@torch.no_grad()
def wtv_marginal_weights(raw_a: Tensor, trust: Tensor, valid: Tensor) -> Tensor:
    """Independent O(M) analytic gradient reference. Never divides by a_j."""
    a = torch.where(valid, raw_a, torch.ones_like(raw_a))
    horizon = trust * valid.to(trust.dtype)
    before = torch.cat([torch.ones_like(a[:, :1]), a[:, :-1].cumprod(-1)], -1)
    tail = horizon[:, -1]
    terms = [tail]
    for j in range(a.shape[1] - 2, -1, -1):
        tail = horizon[:, j] + a[:, j + 1] * tail
        terms.append(tail)
    tails = torch.stack(terms[::-1], -1)
    return before * tails * valid.to(a.dtype) / a.shape[1]


VERSION = "one_sided_deficit_huber_prefix_trust_v2"
CORPORA = ("agentic", "arc2", "arc3", "node1", "naive", "nemotron", "other")
SCALARS = ("accept_opt_num", "accept_trust_num", "robust_band_mass_num", "robust_candidate_mass_num")
VECTORS = ("robust_tail_hits", "robust_trust_sum", "robust_floor_hits")

def additive_terms(p_c, q, tail_mass, valid, mix, cfg, corpus="other"):
    """Fixed schema on every rank, including synthetic warm-up and empty ranks."""
    if corpus not in CORPORA:
        raise ValueError(f"unknown robust diagnostic corpus {corpus!r}")
    out = robust_wtv_terms(p_c, q, tail_mass, valid, mix, cfg)
    p = p_c.detach().float()
    with torch.no_grad():
        band = (q.detach() < p) & (q.detach() > (1 - cfg.relative_band) * p)
        hit = (tail_mass < cfg.tail_alpha) & valid
        trust = out["horizon_trust"]
        metrics = {
            "robust_band_mass_num": (p * band * valid[..., None]).sum(),
            "robust_candidate_mass_num": (p * valid[..., None]).sum(),
            "robust_tail_hits": hit.sum(0).float(),
            "robust_trust_sum": (trust * valid).sum(0),
            "robust_floor_hits": ((trust <= cfg.trust_floor + 1e-3) & valid).sum(0).float(),
        }
        for name in CORPORA:
            metrics[f"robust_tail_hits/{name}"] = hit.sum().float() if name == corpus else q.new_zeros(())
            metrics[f"robust_positions/{name}"] = valid.sum().float() if name == corpus else q.new_zeros(())
    return {"accept_opt_num": out["accept_opt_num"], "accept_trust_num": out["accept_trust_num"], **metrics}

def zero_terms(slots, device):
    terms = {key: torch.zeros((), device=device) for key in SCALARS}
    terms.update({key: torch.zeros(slots, device=device) for key in VECTORS})
    terms.update({f"robust_{kind}/{name}": torch.zeros((), device=device)
                  for name in CORPORA for kind in ("tail_hits", "positions")})
    return terms
