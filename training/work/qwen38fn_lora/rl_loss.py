"""RL loss of the FN LoRA trainer (work/qwen38fn_rl/SPEC.md §1): chunked selective log-probability, BPO and MIS.

``selective_logprob(hidden, weight, targets, T)`` = ``log_softmax(hidden @ weight.T / T)[targets]`` in fp32 without
materializing the (N, V) logits: rows and vocabulary are walked in chunks with an online log-sum-exp, the backward
recomputes each chunk's logits from the saved hidden rows and log-normalizer. ``with_entropy=True`` also returns each
row's full-vocabulary entropy of softmax(logits / T), the distribution the rollout samples from at T (detached; the same
online pass carries sum exp(z - max) z next to the log-sum-exp). The lm_head is frozen, so the backward
returns a gradient for ``hidden`` only (passing a weight that requires grad raises). Peak extra memory is about
``ROW_CHUNK x VOCAB_CHUNK x 10`` bytes (bf16 logits, fp32 logits, one fp32 temporary): 0.6 GiB at the defaults,
whatever N (<= 120k gen tokens per bin) and V (248,320).

Numerics follow the trainer's score mode (train_tito_mm.py ``--mode score``): logits are the GEMM in the input dtype
(bf16 on GPU, as SGLang's lm_head), cast to fp32, divided by T; the log-probability is ``(z_t - max) - log(sum)``.
The backward forms d(logits) in fp32, casts it to the input dtype for the GEMM with the weight (fp32 accumulation,
like liger's fused CE that the SFT path uses) and accumulates d(hidden) over vocabulary chunks in fp32.

``bpo_loss`` and ``mis_loss`` are the SPEC formulas verbatim (weights detached; the sign of ``coef`` is the sign of
the advantage); ``rl.algo`` of the batch picks one (``algo_loss``, parameters ``ALGOS``). The trainer sums the loss
over every bin of every rank of one optimizer step and divides by nothing. ``summarize_bpo`` / ``summarize_mis``
(``summarize_terms``) turn the per-token terms of a step into the SPEC §1 report; ``summarize_entropy`` adds the
policy entropy (from ``selective_logprob(..., with_entropy=True)``) and the sampled tokens' -log pi.
"""

import math

import numpy as np
import torch

ROW_CHUNK = 4096
VOCAB_CHUNK = 16384
# support_logprob gathers (rows, K, H) lm_head rows per chunk: 1024 x 32 x 4096 bf16 = 256 MiB
SUPPORT_ROW_CHUNK = 1024
# Ratio bands reported (pi/mu outside each): the routing-replay study's two (docs/research/qwen38fn_rl_routing_sync_20261007.md)
# and MIS's default band.
RATIO_BANDS = ((0.8, 1.28), (0.5, 2.0), (0.2, 5.0))
#: rl.algo -> its parameters (manifest.rl keys and prepare_rl flags, in the order algo_loss passes them) with
#: prepare_rl's defaults (SPEC §1)
ALGOS = {'bpo': {'eps': 0.1, 'cap': 3.0, 'eps_low': 0.2, 'eps_high': 0.28},
         'mis': {'mis_low': 0.2, 'mis_high': 5.0}}


def _mm_f32(a, b):
    """``a @ b`` accumulated and returned in fp32 (CUDA bf16 GEMM with fp32 output; CPU computes the same in fp32)."""
    if a.is_cuda and a.dtype != torch.float32:
        return torch.mm(a, b, out_dtype=torch.float32)
    return torch.mm(a.float(), b.float())


def _bmm_f32(a, b):
    """Batched ``a @ b`` accumulated and returned in fp32, as ``_mm_f32``."""
    if a.is_cuda and a.dtype != torch.float32:
        return torch.bmm(a, b, out_dtype=torch.float32)
    return torch.bmm(a.float(), b.float())


def _logits(hidden, weight, temperature, fp32=False):
    """(rows, V) fp32 logits / T. Default: the input-dtype GEMM rounded to it, then fp32 (a bf16 logit near 30 is
    0.125 coarse). ``fp32``: the GEMM's fp32 output itself, which a sampling-support replay needs (its support and
    full-vocabulary passes are separate GEMMs; two bf16 roundings of one logit differ by a whole bf16 step)."""
    if fp32:
        return _mm_f32(hidden, weight.t()).div_(temperature)
    return torch.nn.functional.linear(hidden, weight).float().div_(temperature)


class SelectiveLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, temperature, row_chunk, vocab_chunk, fp32_logits):
        rows, vocab = hidden.shape[0], weight.shape[0]
        logp = torch.empty(rows, dtype=torch.float32, device=hidden.device)
        log_z = torch.empty(rows, dtype=torch.float32, device=hidden.device)
        entropy = torch.empty(rows, dtype=torch.float32, device=hidden.device)
        for r0 in range(0, rows, row_chunk):
            h, t = hidden[r0:r0 + row_chunk], targets[r0:r0 + row_chunk]
            run_max = torch.full((h.shape[0],), float('-inf'), dtype=torch.float32, device=h.device)
            run_sum = torch.zeros(h.shape[0], dtype=torch.float32, device=h.device)
            run_ez = torch.zeros(h.shape[0], dtype=torch.float32, device=h.device)    # sum exp(z - run_max) (z - run_max)
            picked = torch.zeros(h.shape[0], dtype=torch.float32, device=h.device)
            for v0 in range(0, vocab, vocab_chunk):
                z = _logits(h, weight[v0:v0 + vocab_chunk], temperature, fp32_logits)
                width = z.shape[1]
                new_max = torch.maximum(run_max, z.amax(1))
                shift = run_max - new_max                       # <= 0; -inf on the first chunk
                rescale = torch.exp(shift)
                # carried terms are relative to the old max: exp(z - new) (z - new) = rescale (exp(z - old) (z - old)
                # + exp(z - old) shift); on the first chunk rescale = 0 and both sums are 0, so no inf * 0 appears
                carried = torch.where(rescale > 0, run_ez + run_sum * shift, 0.0)
                # one fp32 temporary as before (peak memory unchanged): e = exp(z - new), then e * z in place;
                # sum e (z - new) = sum e z - new sum e
                e = (z - new_max[:, None]).exp_()
                e_sum = e.sum(1)
                run_ez = carried * rescale + (e.mul_(z).sum(1) - new_max * e_sum)
                run_sum = run_sum * rescale + e_sum
                del e
                run_max = new_max
                inside = (t >= v0) & (t < v0 + width)
                local = (t - v0).clamp(0, width - 1)
                picked = torch.where(inside, z.gather(1, local[:, None])[:, 0], picked)
            log_sum = torch.log(run_sum)
            logp[r0:r0 + row_chunk] = (picked - run_max) - log_sum
            log_z[r0:r0 + row_chunk] = run_max + log_sum
            # H = log Z - E_p[z] = log_sum - E_p[z - max]
            entropy[r0:r0 + row_chunk] = log_sum - run_ez / run_sum
        ctx.save_for_backward(hidden, weight, targets, log_z)
        ctx.temperature, ctx.row_chunk, ctx.vocab_chunk = temperature, row_chunk, vocab_chunk
        ctx.fp32_logits = fp32_logits
        ctx.mark_non_differentiable(entropy)
        return logp, entropy

    @staticmethod
    def backward(ctx, grad_logp, grad_entropy):
        hidden, weight, targets, log_z = ctx.saved_tensors
        temperature, row_chunk, vocab_chunk = ctx.temperature, ctx.row_chunk, ctx.vocab_chunk
        fp32_logits = ctx.fp32_logits
        grad_logp = grad_logp.float()
        grad_hidden = torch.empty_like(hidden)
        for r0 in range(0, hidden.shape[0], row_chunk):
            h, t = hidden[r0:r0 + row_chunk], targets[r0:r0 + row_chunk]
            g, lz = grad_logp[r0:r0 + row_chunk], log_z[r0:r0 + row_chunk]
            acc = torch.zeros(h.shape, dtype=torch.float32, device=h.device)
            for v0 in range(0, weight.shape[0], vocab_chunk):
                w = weight[v0:v0 + vocab_chunk]
                z = _logits(h, w, temperature, fp32_logits)
                width = z.shape[1]
                # d logp_t / d z_v = onehot(t)_v - softmax_v ; times d(z)/d(logits) = 1/T
                dz = z.sub_(lz[:, None]).exp_().mul_(-g[:, None])
                inside = (t >= v0) & (t < v0 + width)
                local = (t - v0).clamp(0, width - 1)
                dz.scatter_add_(1, local[:, None], torch.where(inside, g, 0.0)[:, None])
                dz.div_(temperature)
                acc += _mm_f32(dz.to(hidden.dtype), w)
            grad_hidden[r0:r0 + row_chunk] = acc.to(hidden.dtype)
        return grad_hidden, None, None, None, None, None, None


def selective_logprob(hidden, weight, targets, temperature, row_chunk=ROW_CHUNK, vocab_chunk=VOCAB_CHUNK,
                      with_entropy=False, fp32_logits=False):
    """fp32 ``log_softmax(hidden @ weight.T / temperature)[targets]``, (N,); differentiable in ``hidden`` only.
    ``with_entropy``: ``(logp, entropy)``, entropy (N,) fp32 detached = -sum_v p_v log p_v of the same softmax.
    ``fp32_logits``: logits from the GEMM's fp32 output, not rounded to the input dtype (``_logits``)."""
    if hidden.dim() != 2 or weight.dim() != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError(f'hidden {tuple(hidden.shape)} and weight {tuple(weight.shape)} must be (N, H) and (V, H)')
    if hidden.dtype != weight.dtype or hidden.device != weight.device:
        raise ValueError(f'hidden {hidden.dtype}@{hidden.device} vs weight {weight.dtype}@{weight.device}')
    if targets.dtype != torch.int64 or targets.shape != (hidden.shape[0],):
        raise ValueError(f'targets must be int64 ({hidden.shape[0]},), got {targets.dtype}{tuple(targets.shape)}')
    if weight.requires_grad:
        raise ValueError('the lm_head is frozen here: this kernel computes no weight gradient')
    if not isinstance(temperature, float) or not temperature > 0:
        raise ValueError(f'temperature must be a positive float, got {temperature!r}')
    if row_chunk < 1 or vocab_chunk < 1:
        raise ValueError('chunk sizes must be positive')
    if targets.is_cuda:
        # no host sync (two .item() here drained the GPU queue at the end of every RL forward): a device-side assert
        # stops the process at its next CUDA call
        torch._assert_async(((targets >= 0) & (targets < weight.shape[0])).all(),
                            f'selective_logprob: target ids outside [0, {weight.shape[0]})')
    elif targets.numel() and (int(targets.min()) < 0 or int(targets.max()) >= weight.shape[0]):
        raise ValueError(f'target ids outside [0, {weight.shape[0]})')
    logp, entropy = SelectiveLogProb.apply(hidden.contiguous(), weight, targets, temperature, row_chunk, vocab_chunk,
                                           fp32_logits)
    return (logp, entropy) if with_entropy else logp


def _support_logits(hidden, weight, ids, valid, temperature):
    """(rows, K) fp32 logits / T at the support ids (padding -> -inf): the GEMM's fp32 output against the gathered
    lm_head rows, as ``_logits(..., fp32=True)`` (the full-vocabulary report pass, selective_logprob fp32_logits)."""
    w = weight[ids.clamp(min=0)]                                         # (rows, K, H)
    z = _bmm_f32(w, hidden[:, :, None])[:, :, 0].div_(temperature)
    return z.masked_fill_(~valid, float('-inf'))


class SupportLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, targets, support, temperature, row_chunk):
        rows = hidden.shape[0]
        logp = torch.empty(rows, dtype=torch.float32, device=hidden.device)
        log_z = torch.empty(rows, dtype=torch.float32, device=hidden.device)
        entropy = torch.empty(rows, dtype=torch.float32, device=hidden.device)
        for r0 in range(0, rows, row_chunk):
            h, t, ids = hidden[r0:r0 + row_chunk], targets[r0:r0 + row_chunk], support[r0:r0 + row_chunk]
            valid = ids >= 0
            z = _support_logits(h, weight, ids, valid, temperature)
            lz = torch.logsumexp(z, 1)
            lp = z.sub_(lz[:, None])                                     # log p over the support, -inf on padding
            logp[r0:r0 + row_chunk] = torch.where(ids == t[:, None], lp, float('-inf')).amax(1)
            log_z[r0:r0 + row_chunk] = lz
            entropy[r0:r0 + row_chunk] = -torch.where(valid, lp.exp() * lp, 0.0).sum(1)
        ctx.save_for_backward(hidden, weight, targets, support, log_z)
        ctx.temperature, ctx.row_chunk = temperature, row_chunk
        ctx.mark_non_differentiable(entropy)
        return logp, entropy

    @staticmethod
    def backward(ctx, grad_logp, grad_entropy):
        hidden, weight, targets, support, log_z = ctx.saved_tensors
        temperature, row_chunk = ctx.temperature, ctx.row_chunk
        grad_logp = grad_logp.float()
        grad_hidden = torch.empty_like(hidden)
        for r0 in range(0, hidden.shape[0], row_chunk):
            h, t, ids = hidden[r0:r0 + row_chunk], targets[r0:r0 + row_chunk], support[r0:r0 + row_chunk]
            g, lz = grad_logp[r0:r0 + row_chunk], log_z[r0:r0 + row_chunk]
            valid = ids >= 0
            z = _support_logits(h, weight, ids, valid, temperature)
            # d logp_t / d z_v = onehot(t)_v - p_v over the support (p = 0 on padding); times d(z)/d(logits) = 1/T
            dz = z.sub_(lz[:, None]).exp_().mul_(-g[:, None])
            dz += torch.where(ids == t[:, None], g[:, None], 0.0)
            dz.div_(temperature)
            w = weight[ids.clamp(min=0)]                                 # (rows, K, H)
            grad_hidden[r0:r0 + row_chunk] = torch.bmm(dz.to(hidden.dtype)[:, None, :], w)[:, 0].to(hidden.dtype)
        return grad_hidden, None, None, None, None, None


def support_logprob(hidden, weight, targets, support, temperature, row_chunk=SUPPORT_ROW_CHUNK):
    """fp32 ``log_softmax`` of ``hidden @ weight.T / temperature`` restricted to each row's sampling support, at
    ``targets``: the rollout's top-k / top-p kept set replayed (SPEC §1 ``replay: mask``), so pi and mu share one
    action space. ``support`` (N, K) int64: the kept token ids, padded with -1 on the right; every row's target must
    be one of its ids. Returns ``(logp, entropy)``, entropy (N,) fp32 detached = the entropy of the restricted
    softmax. Logits are the GEMM's fp32 output (``_support_logits``): compare with selective_logprob(fp32_logits=True). Differentiable in ``hidden`` only (the lm_head is frozen); tokens outside the support get no gradient."""
    if hidden.dim() != 2 or weight.dim() != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError(f'hidden {tuple(hidden.shape)} and weight {tuple(weight.shape)} must be (N, H) and (V, H)')
    if hidden.dtype != weight.dtype or hidden.device != weight.device:
        raise ValueError(f'hidden {hidden.dtype}@{hidden.device} vs weight {weight.dtype}@{weight.device}')
    if targets.dtype != torch.int64 or targets.shape != (hidden.shape[0],):
        raise ValueError(f'targets must be int64 ({hidden.shape[0]},), got {targets.dtype}{tuple(targets.shape)}')
    if support.dtype != torch.int64 or support.dim() != 2 or support.shape[0] != hidden.shape[0] \
            or support.shape[1] < 1 or support.device != hidden.device:
        raise ValueError(f'support must be int64 ({hidden.shape[0]}, K>=1) on {hidden.device}, '
                         f'got {support.dtype}{tuple(support.shape)}@{support.device}')
    if weight.requires_grad:
        raise ValueError('the lm_head is frozen here: this kernel computes no weight gradient')
    if not isinstance(temperature, float) or not temperature > 0:
        raise ValueError(f'temperature must be a positive float, got {temperature!r}')
    if row_chunk < 1:
        raise ValueError('row_chunk must be positive')
    # every id in [-1, V), the padding only on the right, no id twice in a row, the target inside the row's ids
    ordered = support.sort(1).values
    ok = ((support >= -1) & (support < weight.shape[0])).all() \
        & ((support[:, :-1] < 0) <= (support[:, 1:] < 0)).all() & (support[:, 0] >= 0).all() \
        & ((ordered[:, 1:] != ordered[:, :-1]) | (ordered[:, 1:] < 0)).all() \
        & (support == targets[:, None]).any(1).all()
    why = 'support ids outside [-1, V), padding not on the right, a repeated id, or a target outside its support'
    if targets.is_cuda:
        torch._assert_async(ok, f'support_logprob: {why}')
    elif not bool(ok):
        raise ValueError(why)
    return SupportLogProb.apply(hidden.contiguous(), weight, targets, support.contiguous(), temperature, row_chunk)


def check_bpo_params(eps, cap, eps_low, eps_high):
    if not (eps > 0 and cap > 0 and 0 < eps_low < 1 and eps_high > 0):
        raise ValueError(f'BPO parameters eps={eps} cap={cap} eps_low={eps_low} eps_high={eps_high}')


def bpo_loss(logp, behavior_logprob, coef, eps, cap, eps_low, eps_high):
    """SPEC §1. ``logp`` (N,) fp32 with grad; ``behavior_logprob`` (N,) the ledger's log mu; ``coef`` (N,)
    A / denominator (prepare_rl --loss-reduction).

    Returns ``(loss, terms)``: ``loss = -sum(coef * w * logp)`` with ``w = M * min(omega, cap)`` detached, and the
    detached per-token ``terms`` (delta = logp - behavior, omega, masked = M==0, weight w, coef) for summarize_bpo.
    """
    check_bpo_params(eps, cap, eps_low, eps_high)
    check_token_tensors(logp, behavior_logprob, coef)
    with torch.no_grad():
        mu = behavior_logprob.exp()
        pi = logp.detach().exp()
        omega = (1 + eps - mu) / (1 + eps - pi)
        masked = ((coef > 0) & (omega > 1 + eps_high)) | ((coef < 0) & (omega < 1 - eps_low))
        weight = torch.where(masked, 0.0, torch.clamp(omega, max=cap))
    loss = -(coef * weight * logp).sum()
    terms = {'delta': logp.detach() - behavior_logprob, 'omega': omega, 'masked': masked, 'weight': weight,
             'coef': coef}
    return loss, terms


def check_mis_params(mis_low, mis_high):
    if not (0 < mis_low <= 1 <= mis_high < math.inf):
        raise ValueError(f'MIS parameters mis_low={mis_low} mis_high={mis_high}: need 0 < low <= 1 <= high < inf')


def check_algo_params(algo, params):
    """``params`` = the values of ``ALGOS[algo]``, in its order."""
    if algo not in ALGOS:
        raise ValueError(f'rl.algo {algo!r} not in {sorted(ALGOS)}')
    if len(params) != len(ALGOS[algo]):
        raise ValueError(f'{algo} takes {list(ALGOS[algo])}, got {params!r}')
    (check_bpo_params if algo == 'bpo' else check_mis_params)(*params)


def check_token_tensors(logp, behavior_logprob, coef):
    for name, x in (('logp', logp), ('behavior_logprob', behavior_logprob), ('coef', coef)):
        if x.dtype != torch.float32 or x.dim() != 1 or x.shape != logp.shape:
            raise ValueError(f'{name} must be fp32 {tuple(logp.shape)}, got {x.dtype}{tuple(x.shape)}')


def mis_loss(logp, behavior_logprob, coef, mis_low, mis_high):
    """SPEC §1 MIS (direct token importance sampling, two-sided masking). ``logp`` (N,) fp32 with grad;
    ``behavior_logprob`` (N,) the ledger's log mu; ``coef`` (N,) A / denominator (prepare_rl --loss-reduction).

    ``r = exp(logp - behavior)``, ``M = mis_low <= r <= mis_high`` (decided in log space), ``w = r * M`` detached:
    a token outside the band weighs 0 -- dropped, not clamped to the bound -- whatever the sign of ``coef``. Returns
    ``(loss, terms)``: ``loss = -sum(coef * w * logp)`` and the detached per-token ``terms`` (delta = logp - behavior,
    masked = M==0, weight w, coef) for summarize_mis.
    """
    check_mis_params(mis_low, mis_high)
    check_token_tensors(logp, behavior_logprob, coef)
    low, high = math.log(mis_low), math.log(mis_high)
    with torch.no_grad():
        delta = logp.detach() - behavior_logprob
        kept = (delta >= low) & (delta <= high)
        # the clamp only keeps exp() of a dropped token finite; kept tokens are inside [low, high] already
        weight = torch.where(kept, delta.clamp(low, high).exp(), 0.0)
    loss = -(coef * weight * logp).sum()
    return loss, {'delta': delta, 'masked': ~kept, 'weight': weight, 'coef': coef}


def algo_loss(algo, params, logp, behavior_logprob, coef):
    """The batch's loss: ``rl.algo`` with ``params`` = the values of ``ALGOS[algo]`` in order -> ``(loss, terms)``."""
    check_algo_params(algo, params)
    return (bpo_loss if algo == 'bpo' else mis_loss)(logp, behavior_logprob, coef, *params)


def _shared_report(delta, masked, coef):
    """The part of the SPEC §1 report that BPO and MIS share: mismatch, masks by advantage sign, ratio bands."""
    n = len(delta)
    if n == 0 or not (len(masked) == len(coef) == n):
        raise ValueError('the report needs equally long, non-empty token arrays')
    d = delta.astype(np.float64)
    absd = np.abs(d)
    ratio = np.exp(d)  # pi / mu
    positive, negative = coef > 0, coef < 0
    weight = np.abs(coef.astype(np.float64))
    report = {
        'tokens': int(n),
        'abs_delta_mean': float(absd.mean()), 'abs_delta_p50': float(np.percentile(absd, 50)),
        'abs_delta_p95': float(np.percentile(absd, 95)), 'abs_delta_p99': float(np.percentile(absd, 99)),
        'delta_mean': float(d.mean()), 'delta_p01': float(np.percentile(d, 1)), 'delta_p99': float(np.percentile(d, 99)),
        # k3 estimate of KL(mu || pi) from tokens sampled under mu: E_mu[r - 1 - log r], r = pi/mu
        'kl_k3': float((ratio - 1 - d).mean()),
        'masked_frac': float(masked.mean()),
        'masked_frac_adv_pos': float(masked[positive].mean()) if positive.any() else None,
        'masked_frac_adv_neg': float(masked[negative].mean()) if negative.any() else None,
        'tokens_adv_pos': int(positive.sum()), 'tokens_adv_neg': int(negative.sum()),
        # share of sum |coef| (the advantage-weighted token mass) the mask keeps
        'coef_kept_frac': float(weight[~masked].sum() / weight.sum()) if weight.sum() > 0 else None,
    }
    for low, high in RATIO_BANDS:
        outside = (ratio < low) | (ratio > high)
        report[f'ratio_outside_{low}_{high}'] = float(outside.mean())
        # importance estimate of the trainer's probability mass outside the band, E_mu[r 1(outside)] = sum pi_v there
        report[f'pi_mass_outside_{low}_{high}'] = float((ratio * outside).mean())
    return report


def summarize_bpo(delta, omega, masked, coef, cap):
    """SPEC §1 BPO report from the concatenated per-token terms (numpy) of one optimizer step, all ranks and bins."""
    if len(omega) != len(delta):
        raise ValueError('summarize_bpo needs equally long, non-empty token arrays')
    report = _shared_report(delta, masked, coef)
    report.update({'omega_mean': float(omega.astype(np.float64).mean()), 'omega_p95': float(np.percentile(omega, 95)),
                   'omega_over_cap_frac': float((omega > cap).mean())})
    return report


def summarize_mis(delta, masked, coef):
    """SPEC §1 MIS report from the concatenated per-token terms (numpy) of one optimizer step, all ranks and bins:
    the shared part, which side of the band the masked tokens fall on, and the mean importance weight
    (w = r on kept tokens, 0 on masked ones; recomputed from delta)."""
    report = _shared_report(delta, masked, coef)
    d = delta.astype(np.float64)
    report.update({'masked_below_frac': float((masked & (d < 0)).mean()),
                   'masked_above_frac': float((masked & (d > 0)).mean()),
                   'is_weight_mean': float(np.where(masked, 0.0, np.exp(d)).mean())})
    return report


def summarize_entropy(entropy, nll, coef):
    """The policy-entropy part of the report: the trainer's full-vocabulary entropy of softmax(logits / T) at each
    generated token (``entropy``) and the sampled token's -log pi (``nll``; on-policy E_mu[-log mu] is the rollout
    policy's entropy, so the two agree up to the train/serve mismatch and lag), token-weighted, overall and by the sign
    of the advantage. A negative or nonfinite entropy is a kernel bug: raise."""
    n = len(entropy)
    if n == 0 or not (len(nll) == len(coef) == n):
        raise ValueError('the entropy report needs equally long, non-empty token arrays')
    h = entropy.astype(np.float64)
    if not np.isfinite(h).all() or h.min() < -1e-3:
        raise ValueError(f'entropy must be finite and >= 0, got min {h.min()!r}')
    positive, negative = coef > 0, coef < 0
    p10, p50, p90 = np.percentile(h, (10, 50, 90))        # one partition pass (~14M tokens a step)
    return {'entropy_mean': float(h.mean()), 'entropy_p10': float(p10), 'entropy_p50': float(p50),
            'entropy_p90': float(p90),
            'entropy_mean_adv_pos': float(h[positive].mean()) if positive.any() else None,
            'entropy_mean_adv_neg': float(h[negative].mean()) if negative.any() else None,
            'nll_mean': float(nll.astype(np.float64).mean())}


def summarize_support(support_len, entropy_support, log_pi_kept, log_mu_kept, delta_full):
    """The sampling-replay part of the report (``replay: mask``): the support size (and the share of singletons,
    whose log-probability is 0 and gradient none), the restricted softmax's entropy, the probability mass the trainer
    (``pi_kept_*``) and the sampler (``mu_kept_*``) put on the support, and the full-vocabulary mismatch
    ``logp_full - behavior_full`` (train/serve mismatch without the truncation). A kept mass above 1 or a negative
    entropy is a kernel bug: raise."""
    n = len(support_len)
    if n == 0 or not (len(entropy_support) == len(log_pi_kept) == len(log_mu_kept) == len(delta_full) == n):
        raise ValueError('the support report needs equally long, non-empty token arrays')
    h = entropy_support.astype(np.float64)
    pi, mu = log_pi_kept.astype(np.float64), log_mu_kept.astype(np.float64)
    if not (np.isfinite(h).all() and np.isfinite(pi).all() and np.isfinite(mu).all()) or h.min() < -1e-3 \
            or pi.max() > 1e-3 or mu.max() > 1e-3:
        raise ValueError(f'support entropy >= 0 and kept log-masses <= 0 expected, got entropy min {h.min()!r}, '
                         f'log pi kept max {pi.max()!r}, log mu kept max {mu.max()!r}')
    size = support_len.astype(np.float64)
    d = delta_full.astype(np.float64)
    return {'support_len_mean': float(size.mean()), 'support_len_p50': float(np.percentile(size, 50)),
            'support_len_p90': float(np.percentile(size, 90)), 'support_singleton_frac': float((size == 1).mean()),
            'entropy_support_mean': float(h.mean()),
            'pi_kept_mean': float(np.exp(pi).mean()), 'pi_kept_p10': float(np.exp(np.percentile(pi, 10))),
            'mu_kept_mean': float(np.exp(mu).mean()), 'mu_kept_p10': float(np.exp(np.percentile(mu, 10))),
            'abs_delta_full_mean': float(np.abs(d).mean()), 'delta_full_mean': float(d.mean())}


SUPPORT_TERMS = ('support_len', 'entropy_support', 'log_pi_kept', 'log_mu_kept', 'delta_full')


def summarize_terms(algo, params, terms):
    """One step's concatenated per-token ``terms`` (numpy: the loss's terms without ``weight``, plus the forward's
    ``entropy`` and ``nll``; a ``replay: mask`` batch also ``SUPPORT_TERMS``) -> its SPEC §1 report."""
    check_algo_params(algo, params)
    if algo == 'bpo':
        report = summarize_bpo(terms['delta'], terms['omega'], terms['masked'], terms['coef'], params[1])
    else:
        report = summarize_mis(terms['delta'], terms['masked'], terms['coef'])
    present = [k in terms for k in SUPPORT_TERMS]
    if any(present) and not all(present):
        raise ValueError(f'support terms {[k for k, p in zip(SUPPORT_TERMS, present) if p]} without the rest')
    support = summarize_support(*(terms[k] for k in SUPPORT_TERMS)) if all(present) else {}
    return {**report, **summarize_entropy(terms['entropy'], terms['nll'], terms['coef']), **support}
