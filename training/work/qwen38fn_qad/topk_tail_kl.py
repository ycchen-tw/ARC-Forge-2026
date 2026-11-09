"""KL32_tail: forward KL of the teacher's top-K plus one tail bucket, student fully normalized (QAD v2 plan §3).

For a position with teacher top-K ids K (full-vocabulary log-probs ``t_lp``) and tail ``t_tail = log(1 - sum p_K)``:

    L = sum_{i in K} p_i (log p_i - log q_i) + p_R (log p_R - log q_R),     q = softmax(student logits), full vocab
    q_R = sum_{i not in K} q_i                                               (logsumexp over the non-K logits)

i.e. the KL of the 33-class coarsening of both distributions. Not the full-vocabulary KL (it ignores how the tail
mass is split) and never the top-K-renormalized shortcut (that would let the student move mass outside K for free).

Gradient w.r.t. the student logits (per position):  i in K: q_i - p_i;  i not in K: q_i (1 - p_R / q_R).

``direction="reverse"`` is the reverse KL of the same 33-class coarsening (mode seeking, for training on the
student's own trajectories):

    L = sum_{i in K} q_i (log q_i - log p_i) + q_R (log q_R - log p_R)

With g_c = log Q_c - log P_c per class c (Q_c, P_c the 33-class masses) and L = sum_c Q_c g_c, the gradient w.r.t.
logit j in class c(j) is  q_j (g_{c(j)} - L).  The teacher's 33-class log-probs must all be finite (a class the
teacher gives exactly zero mass makes it infinite); trainer-prefill targets are (tail = logsumexp of the non-K
logits, f16 log-probs down to -65504), so this raises instead of clamping.

``fused_topk_tail_kl(hidden, weight, ids, t_lp, t_tail)`` computes the summed loss over rows and the gradient of
the hidden rows in the forward, chunk by chunk ([chunk, V] logits in fp32, never [N, V]); the frozen LM head gets
no gradient. Backward scales the saved dH by the upstream gradient. The same per-chunk function serves the teacher
side (``teacher_topk_tail``) so both use one logit definition: ``(hidden @ weight.T).float()`` over the whole
padded LM-head vocabulary.
"""

import torch

CHUNK_ROWS = 2048
# dH = dz @ W (a [chunk, V] x [V, H] GEMM, reduced over the 248k vocabulary): "fp32" (TF32 off, the reference) or "tf32"
# (~1e-3 relative, the size of the BF16 cast dH gets right after; 8x faster on B200). Set by the trainer.
DH = {"mode": "fp32"}


def _student_logits(h, weight):
    return torch.nn.functional.linear(h, weight).float()


def _tail_logsumexp(z, ids):
    """logsumexp over every column except ids (per row)."""
    return z.scatter(1, ids, float("-inf")).logsumexp(-1)


DIRECTIONS = ("forward", "reverse")


def chunk_loss_and_grad(z, ids, t_lp, t_tail, direction="forward"):
    """z [c, V] fp32 student logits -> (per-row loss [c] fp64-accumulated in fp32, dL/dz [c, V] fp32)."""
    lse = z.logsumexp(-1)
    lq_k = z.gather(1, ids) - lse[:, None]
    lq_r = _tail_logsumexp(z, ids) - lse
    if direction == "reverse":
        return _reverse_loss_and_grad(z, ids, lse, lq_k, lq_r, t_lp, t_tail)
    if direction != "forward":
        raise ValueError(f"direction {direction!r}")
    p_k = t_lp.exp()
    p_r = t_tail.exp()
    # p log p with the p = 0 limit; log q of a token the student gives no mass is -inf only if the logit is -inf
    head = (p_k * (t_lp - lq_k)).sum(-1)
    tail = torch.where(p_r > 0, p_r * (t_tail - lq_r), torch.zeros_like(p_r))
    loss = head + tail
    q = (z - lse[:, None]).exp()
    q_r = lq_r.exp()
    factor = torch.where(q_r > 0, 1 - p_r / q_r, torch.ones_like(q_r))
    dz = q * factor[:, None]
    dz.scatter_(1, ids, lq_k.exp() - p_k)
    return loss, dz


def _reverse_loss_and_grad(z, ids, lse, lq_k, lq_r, t_lp, t_tail):
    g_k = lq_k - t_lp
    g_r = lq_r - t_tail
    q_k = lq_k.exp()
    q_r = lq_r.exp()
    # Q log Q with the Q = 0 limit (a student class underflowing to zero mass contributes 0, gradient q_j * ... = 0)
    head = torch.where(q_k > 0, q_k * g_k, torch.zeros_like(q_k)).sum(-1)
    tail = torch.where(q_r > 0, q_r * g_r, torch.zeros_like(q_r))
    loss = head + tail
    q = (z - lse[:, None]).exp()
    g_r = torch.where(q_r > 0, g_r, torch.zeros_like(g_r))   # -inf - x only where every tail q_j is 0 anyway
    dz = q * (g_r - loss)[:, None]
    dz.scatter_(1, ids, torch.where(q_k > 0, q_k * (g_k - loss[:, None]), torch.zeros_like(q_k)))
    return loss, dz


class FusedTopKTailKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, ids, t_lp, t_tail, chunk_rows, direction):
        if hidden.dim() != 2 or ids.shape[0] != hidden.shape[0] or t_lp.shape != ids.shape or t_tail.shape != (ids.shape[0],):
            raise ValueError(f"shapes: hidden {tuple(hidden.shape)} ids {tuple(ids.shape)} t_lp {tuple(t_lp.shape)} "
                             f"t_tail {tuple(t_tail.shape)}")
        if t_lp.dtype != torch.float32 or t_tail.dtype != torch.float32:
            raise TypeError("teacher log-probs must be fp32")
        if ids.dtype != torch.int64 or bool((ids < 0).any()) or bool((ids >= weight.shape[0]).any()):
            raise ValueError("teacher ids out of the LM-head range")
        if weight.requires_grad:
            raise RuntimeError("the LM head is frozen in QAD v2; this loss gives it no gradient")
        if direction == "reverse" and not bool(torch.isfinite(t_lp).all() & torch.isfinite(t_tail).all()):
            raise ValueError("reverse KL needs finite teacher log-probs for all 33 classes")
        total = torch.zeros((), dtype=torch.float64, device=hidden.device)
        finite = torch.ones((), dtype=torch.bool, device=hidden.device)   # checked once after the loop (one sync)
        dh = torch.empty_like(hidden) if hidden.requires_grad else None
        for lo in range(0, hidden.shape[0], chunk_rows):
            hi = min(lo + chunk_rows, hidden.shape[0])
            z = _student_logits(hidden[lo:hi], weight)
            loss, dz = chunk_loss_and_grad(z, ids[lo:hi], t_lp[lo:hi], t_tail[lo:hi], direction)
            finite &= torch.isfinite(loss).all()
            total += loss.double().sum()
            if dh is not None:
                if DH["mode"] not in ("fp32", "tf32"):
                    raise ValueError(f"dH mode {DH['mode']!r}")
                previous = torch.backends.cuda.matmul.allow_tf32
                torch.backends.cuda.matmul.allow_tf32 = DH["mode"] == "tf32"
                try:
                    dh[lo:hi] = (dz @ weight.float()).to(hidden.dtype)
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = previous
        if not bool(finite):
            raise RuntimeError(f"nonfinite KL32_tail ({direction}) rows")
        ctx.save_for_backward(dh)
        return total.float()

    @staticmethod
    def backward(ctx, grad_output):
        (dh,) = ctx.saved_tensors
        return dh * grad_output.to(dh.dtype), None, None, None, None, None, None


def fused_topk_tail_kl(hidden, weight, ids, t_lp, t_tail, chunk_rows=CHUNK_ROWS, direction="forward"):
    """Summed KL32_tail (``direction`` forward or reverse) over the rows (divide by the global supervised count outside)."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction {direction!r}")
    return FusedTopKTailKL.apply(hidden, weight, ids, t_lp, t_tail, chunk_rows, direction)


@torch.no_grad()
def topk_tail_kl_both(hidden, weight, ids, t_lp, t_tail, chunk_rows=CHUNK_ROWS):
    """Summed (forward, reverse) KL32_tail over the rows, values only (evaluation): fp64 [2]."""
    total = torch.zeros(2, dtype=torch.float64, device=hidden.device)
    for lo in range(0, hidden.shape[0], chunk_rows):
        hi = min(lo + chunk_rows, hidden.shape[0])
        z = _student_logits(hidden[lo:hi], weight)
        lse = z.logsumexp(-1)
        lq_k = z.gather(1, ids[lo:hi]) - lse[:, None]
        lq_r = _tail_logsumexp(z, ids[lo:hi]) - lse
        p_k, p_r = t_lp[lo:hi].exp(), t_tail[lo:hi].exp()
        fwd = (p_k * (t_lp[lo:hi] - lq_k)).sum(-1) + torch.where(p_r > 0, p_r * (t_tail[lo:hi] - lq_r), 0)
        q_k, q_r = lq_k.exp(), lq_r.exp()
        rev = (torch.where(q_k > 0, q_k * (lq_k - t_lp[lo:hi]), 0).sum(-1)
               + torch.where(q_r > 0, q_r * (lq_r - t_tail[lo:hi]), 0))
        total += torch.stack([fwd.double().sum(), rev.double().sum()])
    if not bool(torch.isfinite(total).all()):
        raise RuntimeError("nonfinite KL32_tail rows")
    return total


@torch.no_grad()
def teacher_topk_tail(hidden, weight, k, chunk_rows=CHUNK_ROWS):
    """Teacher targets from its final hidden rows: (ids int64 [N, k], log-probs fp32 [N, k], tail log-prob fp32 [N]),
    full-vocabulary normalized, T = 1. ids are sorted by descending probability."""
    ids_out, lp_out, tail_out = [], [], []
    for lo in range(0, hidden.shape[0], chunk_rows):
        z = _student_logits(hidden[lo:lo + chunk_rows], weight)
        lse = z.logsumexp(-1)
        top, ids = z.topk(k, dim=-1)
        ids_out.append(ids)
        lp_out.append(top - lse[:, None])
        tail_out.append(_tail_logsumexp(z, ids) - lse)
    return torch.cat(ids_out), torch.cat(lp_out), torch.cat(tail_out)


def reference_loss(z, ids, t_lp, t_tail, direction="forward"):
    """float64 torch reference on full logits (tests only)."""
    z = z.double()
    logq = z.log_softmax(-1)
    lq_k = logq.gather(1, ids)
    mask = torch.zeros_like(z, dtype=torch.bool).scatter(1, ids, True)
    lq_r = logq.masked_fill(mask, float("-inf")).logsumexp(-1)
    p_k, p_r = t_lp.double().exp(), t_tail.double().exp()
    if direction == "reverse":
        q_k, q_r = lq_k.exp(), lq_r.exp()
        return (q_k * (lq_k - t_lp.double())).sum() + (q_r * (lq_r - t_tail.double())).sum()
    return (p_k * (t_lp.double() - lq_k)).sum() + torch.where(p_r > 0, p_r * (t_tail.double() - lq_r), 0).sum()
