"""Hard CE + detached D-PACE, strict-top-K selector CE, early teacher KL."""

import torch
from torch.nn import functional as F


def dpace_weights(probability, valid, alpha=0.5):
    if not 0 <= alpha <= 1 or probability.shape != valid.shape:
        raise ValueError("invalid D-PACE inputs")
    smooth = torch.where(valid, alpha + (1 - alpha) * probability.detach(), 1.0)
    prefix = smooth.cumprod(-1)
    return (prefix * valid).flip(-1).cumsum(-1).flip(-1).detach()


def objective(model, hidden, labels, predecessors, teacher_gr, valid=None):
    """One objective chunk; teacher_gr is H[a], H[a+1] for labels a+1,a+2.

    Return additive numerators/denominators so uneven chunks/ranks can be
    normalized globally. No fixed exponential position decay or LK/TV loss.
    """
    if hidden.shape[:2] != labels.shape or predecessors.shape != labels.shape:
        raise ValueError("label/predecessor alignment mismatch")
    if valid is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    logits = model.lm_head(hidden.reshape(-1, hidden.shape[-1])).view(*labels.shape, -1).float()
    ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").view_as(labels)
    weight = dpace_weights((-ce).exp(), valid) * valid
    unary, candidates = logits.detach().topk(model.candidate_selector.top_k, -1)
    matches = candidates == labels[..., None]
    covered = matches.any(-1) & valid
    selected = model.candidate_selector(candidates, unary, hidden.detach(), predecessors)
    selector_ce = F.cross_entropy(
        selected.float().flatten(0, 1), matches.long().argmax(-1).flatten(), reduction="none"
    ).view_as(ce)
    selector_weight = weight * covered
    with torch.no_grad():
        teacher_hidden = model.target_reducer(teacher_gr)
        teacher_logits = (
            model.lm_head(teacher_hidden.reshape(-1, teacher_hidden.shape[-1]))
            .view(*teacher_hidden.shape[:-1], -1)
            .float()
        )
        teacher_logp = teacher_logits.log_softmax(-1)
    if teacher_logits.shape != logits[:, :2].shape:
        raise ValueError("early KL requires exactly two aligned target residuals")
    kl = (teacher_logp.exp() * (teacher_logp - logits[:, :2].log_softmax(-1))).sum(-1)
    chosen = candidates.gather(-1, selected.argmax(-1, keepdim=True)).squeeze(-1)
    # Greedy serving acceptance: the leading run of correct selector choices is
    # exactly what a greedy verify commits, so this matches cycles.jsonl.
    accept = (chosen == labels) & valid
    run = accept.long().cumprod(-1)
    terms = {
        "ce_num": (ce * weight).sum(),
        "ce_den": weight.sum(),
        "selector_num": (selector_ce * selector_weight).sum(),
        "selector_den": selector_weight.sum(),
        "kl_num": (kl * valid[:, :2]).sum(),
        "kl_den": valid[:, :2].sum(),
        "covered": covered.sum(),
        "positions": valid.sum(),
        "selector_correct": ((chosen == labels) & covered).sum(),
        "unary_correct": ((logits.argmax(-1) == labels) & valid).sum(),
        # Per-position vectors; summed over chunks and ranks, divided by "blocks".
        "accept_prefix": run.sum(0),
        "accept_position": accept.sum(0),
        "covered_position": covered.sum(0),
        "accept_histogram": torch.bincount(run.sum(-1), minlength=labels.shape[1] + 1),
        "blocks": torch.tensor(labels.shape[0], device=labels.device),
    }
    return terms


def normalized_loss(terms):
    return (
        terms["ce_num"] / terms["ce_den"].clamp_min(1e-9)
        + terms["selector_num"] / terms["selector_den"].clamp_min(1e-9)
        + 0.1 * terms["kl_num"] / terms["kl_den"].clamp_min(1)
    )


def acceptance_objective(model, hidden, labels, predecessors, teacher_gr, *, temperature, joint, valid=None,
                         robust=None, robust_mix=None, corpus="other", gate=None, anchor_tokens=None, anchor_gr=None):
    """The temperature-matched acceptance objective for one chunk: the reference, literal and unfused (fast_train has
    the fast one). Design: docs/research/qwen38fn_dflash_acceptance_objective_20261006.md.

    hidden [A, M, D] are the proposal states, labels x[a+1..a+M], predecessors x[a..a+M-1] (teacher forcing),
    teacher_gr GR[a..a+M-1] (GR at t is the target's state for x[t+1]). At the deployment temperature T, per slot:
    the teacher p = softmax(target logits / T) over the vocabulary; the drafter's unary r = softmax(U / T); the
    candidates C = top-K of U; the proposal q = softmax(S / T) over C, S the selector's scores given the data
    predecessor -- exactly what serving samples from (sampling.propose_path) and divides by in rejection sampling.

    - coverage: KL(p || r) over the vocabulary, every slot. Teaches the candidates; the only signal for a token
      outside C.
    - acceptance: a = sum_{v in C} min(p(v), q(v)), the probability that rejection sampling accepts this slot's draft
      at this state; per block 1 - (1/M) sum_k prod_{j<=k} a_j (WTV, Acceptance-Aware Draft Model Training). With
      joint, its gradient reaches the backbone through U on C and the selector's hidden projection; without it only
      the selector learns from it.

    valid [A, M] (data.block_targets): a block that reaches the end of its run of supervised tokens trains its slots
    up to there; past it the slot counts as accepted (serving stops at the turn's end) and carries no gradient.

    Returns additive sums (normalize by the step's global blocks / positions): cov_num, accept_num, plus metrics.

    gate (prefix_gate.PrefixGateConfig, with anchor_tokens x[a] [A] and anchor_gr GR[a-1] [A, branches, hidden]): adds
    the optimized accept_opt_num / cov_opt_num, both weighted by the detached prefix weights, and the gate's
    diagnostics; accept_num / cov_num stay the raw, unweighted sums.
    """
    if hidden.shape[:2] != labels.shape or predecessors.shape != labels.shape:
        raise ValueError("label/predecessor alignment mismatch")
    if teacher_gr.shape[:2] != labels.shape:
        raise ValueError("the acceptance objective needs the teacher GR of every slot")
    count, slots = labels.shape
    if valid is None:
        valid = torch.ones_like(labels, dtype=torch.bool)
    inverse = 1.0 / temperature
    logits = model.lm_head(hidden.reshape(-1, hidden.shape[-1])).view(count, slots, -1).float()
    with torch.no_grad():
        teacher_logits = model.lm_head(model.target_reducer(teacher_gr)).float()
        teacher_logp = (teacher_logits * inverse).log_softmax(-1)
    teacher = teacher_logp.exp()
    coverage = (teacher * (teacher_logp - (logits * inverse).log_softmax(-1))).sum(-1)
    unary, candidates = logits.topk(model.candidate_selector.top_k, -1)
    selector_hidden = hidden
    if not joint:
        unary, selector_hidden = unary.detach(), hidden.detach()
    scores = model.candidate_selector(candidates, unary, selector_hidden, predecessors).float()
    proposal = (scores * inverse).softmax(-1)
    mass = teacher.gather(-1, candidates)
    overlap = torch.where(valid, torch.minimum(mass, proposal).sum(-1), 1.0)
    survival = overlap.cumprod(-1)
    extra = {}
    if robust is not None:
        if robust_mix is None:
            raise ValueError("robust acceptance requires a step-wide mix tensor")
        from robust_objective import additive_terms, tail_mass_from_logits

        tails = tail_mass_from_logits(teacher_logits, labels, robust.gate_temperature)
        extra = additive_terms(mass, proposal, tails, valid, robust_mix, robust, corpus)
    if gate is not None:
        if robust is not None or anchor_tokens is None or anchor_gr is None:
            raise ValueError("the prefix gate needs the anchors' tokens and predecessor GR, and no robust objective")
        from prefix_gate import diagnostics, prefix_weights
        from robust_objective import tail_mass_from_logits

        with torch.no_grad():
            anchor_logits = model.lm_head(model.target_reducer(anchor_gr)).float()
            anchor_tail = tail_mass_from_logits(anchor_logits, anchor_tokens, gate.gate_temperature)
            label_tail = tail_mass_from_logits(teacher_logits, labels, gate.gate_temperature)
            weights = prefix_weights(anchor_tail, label_tail, valid, gate)
            scale = torch.where(valid, weights, torch.zeros_like(weights))
        extra = {"accept_opt_num": ((1 - survival) * scale).sum() / slots, "cov_opt_num": (coverage * scale).sum(),
                 **diagnostics(anchor_tail, label_tail, weights, valid, gate, corpus)}
    elif anchor_tokens is not None or anchor_gr is not None:
        raise ValueError("anchor tokens and GR belong to the prefix gate")
    with torch.no_grad():
        ce = F.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="none").view_as(labels)
        ce_t = F.cross_entropy((logits * inverse).flatten(0, 1), labels.flatten(), reduction="none").view_as(labels)
        matches = candidates == labels[..., None]
        covered = matches.any(-1) & valid
        chosen = candidates.gather(-1, scores.argmax(-1, keepdim=True)).squeeze(-1)
        accept = (chosen == labels) & valid
        run = accept.long().cumprod(-1)
    return {
        "cov_num": (coverage * valid).sum(),
        "accept_num": ((1 - survival) * valid).sum() / slots,
        "blocks": torch.tensor(count, device=labels.device),
        "positions": valid.sum(),
        # Per slot, summed over the blocks that reach it (valid_position): acceptance a_j, teacher mass on the
        # candidates p(C_j), and the teacher-forced survival prod_{i<=j} a_i, whose sum over slots is the expected
        # accepted drafts per cycle at temperature T.
        "valid_position": valid.sum(0),
        "overlap_position": (overlap.detach() * valid).sum(0),
        "mass_position": (mass.detach().sum(-1) * valid).sum(0),
        "survival_position": (survival.detach() * valid).sum(0),
        # Hard CE of the data token at T = 1 and at T: diagnostics, not trained.
        "ce_t1_num": (ce * valid).sum(),
        "ce_t_num": (ce_t * valid).sum(),
        "covered": covered.sum(),
        "selector_correct": (accept & covered).sum(),
        "unary_correct": ((candidates[..., 0] == labels) & valid).sum(),
        # The selector's argmax against the recorded (sampled) continuation: the leading run of matches. Not what a
        # greedy verify commits -- that compares with the target's argmax.
        "accept_prefix": run.sum(0),
        "accept_position": accept.sum(0),
        "covered_position": covered.sum(0),
        "accept_histogram": torch.bincount(run.sum(-1), minlength=slots + 1),
    } | extra


WEIGHT_DRIFT = 1.2  # importance_terms' drift diagnostic: q_j(y^_j) at least 20% above the sampler's


def importance_terms(overlap, q_drafted, p_drafted, sampler_q, candidates, proposal, sampler_candidates):
    """This drafter's expected accepted length at recorded cycles another drafter (the sampler, q^s) proposed. Per slot
    k: a_k times the importance weights of the drafts before it, w_j = min(p_j, q_j)(y^_j) / q^s_j(y^_j) (q_j = 0 for a
    draft outside the current candidates). P(L > k) = sum over y_{<k} of prod_{j<k} min(p_j, q_j)(y_j) times a_k --
    the drafts before slot k proposed and accepted, slot k accepted in closed form (Rao-Blackwell; its teacher is the
    verify row read after y^_{<k}) -- so sum_k a_k prod_{j<k} w_j is unbiased for E[L] of this drafter at the cycle's
    state wherever q^s covers its proposal, and with q^s = q, w_j is the fresh replay's rho_j. sampler_covered_position:
    this drafter's proposal mass on the sampler's candidates; below 1, the estimate misses the rest (reads low).

    is_num, the estimate's sum, carries the gradient of overlap and of q_drafted: on recorded drafts q^s drew (fixed,
    q^s does not move), its gradient is unbiased for that of E[L]. That is the reached TV's term -- a_k on the slots
    the drafts reach, here weighted by the chance this drafter reaches them -- plus what the reached TV leaves out:
    raising q_j(y^_j) where it is below p_j, by the acceptance after y^_j (the later slots it lets the cycle reach)."""
    if sampler_q.shape != q_drafted.shape or sampler_candidates.shape != candidates.shape:
        raise ValueError("the sampler's q and candidates must align with the cycles")
    if (sampler_q <= 0).any():
        raise ValueError("a recorded draft with sampler q <= 0 could not have been drawn")
    weight = torch.minimum(p_drafted, q_drafted) / sampler_q
    before = torch.cat((torch.ones_like(weight[:, :1]), weight[:, :-1].cumprod(-1)), 1)
    estimate = overlap * before
    with torch.no_grad():
        detached = estimate.detach()
        covered = (proposal * (candidates[..., :, None] == sampler_candidates[..., None, :]).any(-1)).sum(-1)
        diagnostics = {
            "is_survival_position": detached.sum(0),
            "is_length_sq": (detached.sum(-1) ** 2).sum(),
            "sampler_covered_position": covered.sum(0),
            # weights above WEIGHT_DRIFT (the drafts that weigh in: slots 0..M-2): this drafter proposes y^_j well
            # beyond the sampler and p_j allows it -- how far the trained q has left the rollouts. Not 1: at a fresh
            # replay w_j = min(rho_j, q_j / q^s_j) sits at 1 within rounding wherever p_j >= q_j.
            "is_weight_drifted": (weight[:, :-1] > WEIGHT_DRIFT).sum(),
        }
    return {"is_num": estimate.sum()} | diagnostics


def opd_objective(model, hidden, drafted, anchor_tokens, teacher_gr, accepted, *, temperature, joint, sampler_q=None,
                  sampler_candidates=None):
    """On-policy (replay) terms for one chunk of recorded verify cycles: the drafter at a cycle's anchor, judged on the
    path it actually proposed there (design doc part 2, sections 15-20). Reference: literal and unfused.

    hidden [A, M, D] are the drafter's proposal states at the anchor -- path-independent, the block runs in parallel;
    drafted [A, M] the tokens it proposed; anchor_tokens [A]; teacher_gr [A, M, 4, D] the target's GR at the verify
    block's positions 0..M-1 (the anchor, then drafts 1..M-1): the target's state for slot j on the proposal prefix
    (s0, y^_{<j}); accepted [A] the drafts the cycle accepted. Per slot, at temperature T: the candidates C_j = top-K(U_j)
    as offline; the proposal q_j = softmax(S_j / T) over C_j with the SAMPLED predecessor (slot 0: the anchor, slot j:
    y^_{j-1}); the teacher p_j = softmax(target logits / T) on the same prefix; a_j = sum_{v in C_j} min(p_j, q_j).

    - reached TV (section 18): slots 0..min(L, M-1) -- every accepted draft and the first rejection -- sum (1 - a_j)
      into reach_num, their count into reach_den.
    - path WTV (section 19): per block 1 - (1/M) sum_k prod_{j<=k} a_j over the whole recorded path, the suffix after
      the first rejection included (the verify forward computed it); a surrogate, not the exact expected length.
    - cov_num: KL(p_j || r_j) on the reached slots, for an on-path coverage term if one is wanted.
    With joint, the acceptance terms reach the backbone as in acceptance_objective.
    sampler_q [A, M], sampler_candidates [A, M, K]: the recording drafter's q of each draft and its candidates (the
    server's record) -- adds importance_terms, this drafter's expected accepted length off another's rollouts, whose
    is_num trains as the OPD loss "importance".
    """
    if (sampler_q is None) != (sampler_candidates is None):
        raise ValueError("sampler_q and sampler_candidates go together")
    if hidden.shape[:2] != drafted.shape or teacher_gr.shape[:2] != drafted.shape:
        raise ValueError("hidden, drafted tokens and teacher GR must align on [cycles, slots]")
    if anchor_tokens.shape != drafted.shape[:1] or accepted.shape != drafted.shape[:1]:
        raise ValueError("one anchor token and one accepted count per cycle")
    count, slots = drafted.shape
    if ((accepted < 0) | (accepted > slots)).any():
        raise ValueError("accepted drafts out of range")
    inverse = 1.0 / temperature
    predecessors = torch.cat((anchor_tokens[:, None], drafted[:, :-1]), 1)
    logits = model.lm_head(hidden.reshape(-1, hidden.shape[-1])).view(count, slots, -1).float()
    with torch.no_grad():
        teacher_logp = (model.lm_head(model.target_reducer(teacher_gr)).float() * inverse).log_softmax(-1)
    teacher = teacher_logp.exp()
    coverage = (teacher * (teacher_logp - (logits * inverse).log_softmax(-1))).sum(-1)
    unary, candidates = logits.topk(model.candidate_selector.top_k, -1)
    selector_hidden = hidden
    if not joint:
        unary, selector_hidden = unary.detach(), hidden.detach()
    scores = model.candidate_selector(candidates, unary, selector_hidden, predecessors).float()
    proposal = (scores * inverse).softmax(-1)
    mass = teacher.gather(-1, candidates)
    overlap = torch.minimum(mass, proposal).sum(-1)
    survival = overlap.cumprod(-1)
    reached = torch.arange(slots, device=drafted.device)[None] <= accepted.clamp(max=slots - 1)[:, None]
    with torch.no_grad():
        picked = candidates == drafted[..., None]
        in_candidates = picked.any(-1)
        p_drafted = teacher.gather(-1, drafted[..., None]).squeeze(-1)
    q_drafted = (proposal * picked).sum(-1)  # with its gradient: importance_terms' weights train through it
    with torch.no_grad():
        # Section 20: given the path, rejection sampling accepts slot j with rho_j = min(1, p_j(y^_j) / q_j(y^_j)), and
        # sum_k prod_{j<=k} rho_j is E[L | s0, y^]; over paths drawn from the current q it averages to E[L | s0]. Only
        # a fresh replay (drafts drawn by these weights) makes it that; a draft outside C_j counts as rejected.
        rho = torch.where(q_drafted > 0, (p_drafted / q_drafted.clamp_min(1e-30)).clamp(max=1), 0.0)
    importance = {} if sampler_q is None else importance_terms(
        overlap, q_drafted, p_drafted, sampler_q, candidates, proposal, sampler_candidates)
    return importance | {
        "reach_num": ((1 - overlap) * reached).sum(),
        "reach_den": reached.sum(),
        "wtv_num": (1 - survival.sum(-1) / slots).sum(),
        "blocks": torch.tensor(count, device=drafted.device),
        "cov_num": (coverage * reached).sum(),
        "overlap_position": overlap.detach().sum(0),
        "mass_position": mass.detach().sum(-1).sum(0),
        "survival_position": survival.detach().sum(0),
        "reached_position": reached.sum(0),
        # The recorded drafts still among the current drafter's candidates (1 when the replay is fresh).
        "drafted_candidate_position": in_candidates.sum(0),
        "rho_survival_position": rho.cumprod(-1).sum(0),
        "accepted_observed": accepted.sum(),
        "accepted_histogram": torch.bincount(accepted, minlength=slots + 1),
    }
