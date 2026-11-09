"""Exact-token records, prompt-group split, and non-leaking anchor alignment."""

import hashlib

import torch


def prompt_key(tokens, n_prompt):
    if not 0 < n_prompt <= len(tokens):
        raise ValueError("invalid prompt length")
    return hashlib.sha256(b"".join(int(t).to_bytes(4, "little", signed=False) for t in tokens[:n_prompt])).hexdigest()


def split_for_key(key, heldout_permyriad=100):
    if not 0 < heldout_permyriad < 10000:
        raise ValueError("invalid held-out fraction")
    return (
        "heldout"
        if int(hashlib.sha256(("mtp3-v1:" + key).encode()).hexdigest()[:16], 16) % 10000 < heldout_permyriad
        else "train"
    )


def validate_row(row, vocab_size):
    ids, mask, prompt = row["input_ids"], row["loss_mask"], row["n_prompt"]
    if len(ids) != len(mask) or not 0 < prompt <= len(ids):
        raise ValueError("unaligned token record")
    if any(t < 0 or t >= vocab_size for t in ids):
        raise ValueError("token outside target vocabulary")
    if any(v not in (0, 1) for v in mask) or any(mask[:prompt]):
        raise ValueError("invalid mask or supervised prompt")
    return prompt_key(ids, prompt)


def eligible_anchors(mask, *, block_size=8):
    """Every position that can be an anchor, ascending: a >= 1 with mask[a] and mask[a + 1]. The anchor is itself a
    sampled token (never the prompt's last token) and so is at least its first label; a block that runs past the end
    of its supervised run trains the supervised labels only (block_targets' valid). Anchor 0 is excluded: it has no
    predecessor GR, and the drafter conditions on anchor-1. block_size does not decide eligibility (a block has at
    least one label)."""
    if mask.ndim != 1 or mask.dtype != torch.bool:
        raise ValueError("expected 1D boolean loss mask")
    if block_size < 2:
        raise ValueError("a block needs at least one label")
    good = (mask[:-1] & mask[1:]).nonzero().flatten()
    return good[good >= 1]


def sample_anchors(mask, count, *, block_size=8, generator=None):
    good = eligible_anchors(mask, block_size=block_size)
    order = torch.randperm(len(good), generator=generator, device=mask.device)
    return good[order[:count]]


def slot_mask(mask, anchors, block_size=8):
    """[anchors, block_size - 1]: slot j's label x[a+j+1] and every label before it are supervised and inside the
    sequence. A block stops at the end of its run of supervised tokens -- serving stops at the turn's last token."""
    indices = anchors[:, None] + torch.arange(1, block_size, device=anchors.device)
    inside = indices < len(mask)
    return (mask[indices.clamp(max=len(mask) - 1)] & inside).long().cumprod(-1).bool()


def block_targets(tokens, gr, anchors, block_size=8, teacher_slots=2, *, mask):
    """Labels x[a+1..a+b-1], predecessors x[a..a+b-2], the teacher GR of the first teacher_slots slots (2: the
    D-PACE objective's early KL; b - 1: the acceptance objective's every slot), and slot_mask's valid [anchors, b - 1].
    Past a block's valid slots the labels and teachers are clamped copies that nothing may read."""
    if not 0 < teacher_slots < block_size:
        raise ValueError("teacher slots must lie within the block's proposals")
    valid = slot_mask(mask, anchors, block_size)
    if not valid[:, 0].all():
        raise ValueError("an anchor whose first label is not supervised")
    indices = (anchors[:, None] + torch.arange(1, block_size, device=anchors.device)).clamp(max=len(tokens) - 1)
    # Teacher next-token logits for x[t] come from H[t-1].
    return tokens[indices], tokens[indices - 1], gr[indices[:, :teacher_slots] - 1], valid
