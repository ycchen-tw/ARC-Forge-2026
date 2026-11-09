"""Sequence packing for text-only SFT of Qwen3.5-35B-A3B (varlen, B=1).

Why packing: unpacked B=1 variable-length training leaves ~36% of the GPU idle
(per-microstep fixed overhead, FSDP straggler across the 4 ranks, kernel shape
churn). Packing many sequences into one fixed-size bin removes all three without
changing the FLOPs: block-diagonal attention over segments costs exactly the sum
of the per-sequence costs, and the MoE/MLP cost is linear in tokens either way.

Segment isolation is NOT provided by an attention mask. With
``attn_implementation="flash_attention_4"`` transformers' ``flash_attention_mask``
ignores the mask function entirely and ``create_causal_mask`` returns ``None``;
isolation comes from the metadata this collator emits, consumed at three places
in ``modeling_qwen3_5_moe``:

- ``cu_seq_lens_q``/``cu_seq_lens_k`` + ``max_length_q``/``max_length_k`` reach
  ``_flash_attention_forward``, which dispatches to
  ``flash_attn.cute.flash_attn_varlen_func`` (block-diagonal) for the 10
  full-attention layers.
- ``cu_seq_lens_q`` reaches ``chunk_gated_delta_rule(cu_seqlens=...)``, which
  resets the recurrent state at segment boundaries in the 30 linear-attention
  layers. A mask cannot do this -- without it, state leaks across segments.
- ``seq_idx`` reaches ``causal_conv1d_fn(seq_idx=...)`` so the short causal conv
  does not smear across a boundary.

``attention_mask`` is deliberately absent from the emitted dict: if it is
present, ``_flash_attention_forward`` takes the ``_upad_input`` branch and
recomputes ``cu_seq_lens`` from the padding mask, silently destroying isolation.

``position_ids`` is (4, 1, T) -- Qwen3.5 m-rope splits axis 0 off as the text
axis (used for mask construction) and feeds axes 1:4 to the rotary embedding.
For pure text all four are the same per-segment ``arange``, which is what the
unpacked path builds too.

Nemotron-H (``rope_axes=1``) consumes the same three metadata channels, with the
consumers relabelled: ``cu_seq_lens_*`` feed the 6 full-attention blocks, and
``seq_idx`` feeds the 23 Mamba2 blocks -- both the SSD chunk scan (recurrent state
reset) and the width-4 causal conv, through ``nemotron_patch.patch_mamba_seq_idx``.
Nemotron-H applies no rotary at all, so its ``position_ids`` is informational only.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence

import numpy as np
import torch


def round_up(n: int, multiple: int) -> int:
    return n if multiple <= 1 else -(-n // multiple) * multiple


def plan_bins(lengths: Sequence[int], target_len: int, segment_multiple: int = 1) -> list[list[int]]:
    """Best-fit-decreasing bin packing of sample indices into token bins.

    ``segment_multiple`` must match the collator's: the planner has to reserve the
    per-segment alignment padding, or a bin that looks like it fits will overflow
    ``target_len`` once the collator rounds each segment up.

    Best-fit rather than the first-fit in ``arcforge.mrope.packing.pack_indices``:
    at this scale (88k samples -> 33k bins) first-fit's linear rescan of every
    open bin is quadratic (~31s), while the sorted-room bisect below is O(n log n)
    (~0.2s) and packs marginally tighter. On the real SFT length distribution it
    lands within 25 bins of the ``ceil(sum/target)`` lower bound (99.9% density).
    """
    if not lengths:
        raise ValueError("lengths must be non-empty")
    eff = [round_up(int(n), segment_multiple) for n in lengths]
    too_long = [i for i, n in enumerate(eff) if n > target_len]
    if too_long:
        raise ValueError(
            f"{len(too_long)} samples exceed target_len={target_len} "
            f"(first: index {too_long[0]}, length {lengths[too_long[0]]}"
            f" -> {eff[too_long[0]]} after alignment padding)"
        )
    order = sorted(range(len(eff)), key=lambda i: -eff[i])
    rooms: list[tuple[int, int]] = []  # (remaining room, bin id), kept sorted
    bins: list[list[int]] = []
    for i in order:
        n = eff[i]
        j = bisect.bisect_left(rooms, (n, -1))
        if j < len(rooms):
            room, b = rooms.pop(j)
            bins[b].append(i)
            bisect.insort(rooms, (room - n, b))
        else:
            b = len(bins)
            bins.append([i])
            bisect.insort(rooms, (target_len - n, b))
    return bins


# Mirrors arcforge.train.topk_kd_data.KD_IGNORE_ID. Duplicated rather than imported
# so packed_text keeps its minimal import surface; the value is asserted equal in
# tests/test_kd_packed_collate.py.
_KD_IGNORE_ID = -1


class PackedTextCollator:
    """Collate one bin of tokenized text samples into a packed (B=1) batch.

    Emits exactly the kwargs ``Qwen3_5MoeForConditionalGeneration.forward``
    forwards down to the attention / linear-attention / conv kernels. Every
    tensor is built on ``device`` directly; nothing here is a hot path relative
    to a 128k forward, but the per-segment ``arange`` is the one thing that would
    be slow in pure Python, so it goes through numpy.
    """

    def __init__(
        self,
        *,
        pad_token_id: int,
        pad_to_len: int | None = None,
        label_pad_token_id: int = -100,
        device: torch.device | str = "cpu",
        rope_axes: int = 4,
        segment_multiple: int = 1,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.pad_to_len = pad_to_len
        self.label_pad_token_id = int(label_pad_token_id)
        self.device = torch.device(device)
        # Mamba2 REQUIRES this. mamba_ssm's seq_idx isolates segments only at chunk
        # granularity: _state_passing_fwd (ssd_state_passing.py:75) samples seq_idx at
        # the LAST TOKEN of each chunk to decide whether to zero the inter-chunk state
        # carry, so a boundary landing mid-chunk leaks the previous segment's state
        # into the tokens of the next one that share that chunk. Measured on the real
        # Nemotron-H geometry (diag_mamba_seqidx.py): unaligned boundaries leak,
        # boundaries on multiples of chunk_size are bit-exact. Setting this to the
        # model's config.chunk_size pads every segment up to a chunk multiple, which
        # puts all boundaries on chunk lines. Cost is < chunk_size tokens per sample.
        # Attention-only models (Qwen3.5) leave this at 1 -- FA4 varlen has no such
        # granularity, and padding would be pure waste.
        if segment_multiple < 1:
            raise ValueError(f"segment_multiple must be >= 1, got {segment_multiple}")
        self.segment_multiple = int(segment_multiple)
        if pad_to_len is not None and pad_to_len % self.segment_multiple:
            raise ValueError(
                f"pad_to_len={pad_to_len} is not a multiple of segment_multiple="
                f"{self.segment_multiple}; the tail pad segment would start off-chunk")
        # 4 -> Qwen3.5 m-rope, position_ids is (4, 1, T).
        # 1 -> conventional (1, T). Nemotron-H is NoPE -- NemotronHAttention applies
        #      no rotary at all -- so its position_ids only ever rides in kwargs. We
        #      still emit the per-segment arange rather than dropping the key, so the
        #      tensor never contradicts cu_seq_lens for any consumer that reads it.
        if rope_axes not in (1, 4):
            raise ValueError(f"rope_axes must be 1 or 4, got {rope_axes}")
        self.rope_axes = int(rope_axes)

    def __call__(
        self,
        segments: Sequence[tuple[np.ndarray, np.ndarray]],
        kd_segments: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
    ) -> dict:
        """Collate one bin. ``kd_segments`` carries teacher top-K per segment.

        When given, it must be parallel to ``segments``: entry ``i`` is
        ``(kd_ids, kd_logprobs)`` of shape ``[len(segments[i][0]), K]``, indexed by
        predicted token exactly like that segment's labels. The KD arrays are
        appended at every point ``lab_parts`` is appended -- real segment, alignment
        pad, tail pad -- so they cannot drift from the labels by construction. Pad
        rows carry ``KD_IGNORE_ID`` and are dropped by the supervised filter, since
        their label is ``label_pad_token_id``.
        """
        if not segments:
            raise ValueError("cannot collate an empty bin")
        if kd_segments is not None and len(kd_segments) != len(segments):
            raise ValueError(
                f"kd_segments has {len(kd_segments)} entries but there are "
                f"{len(segments)} segments; they must be parallel")

        ids_parts: list[np.ndarray] = []
        lab_parts: list[np.ndarray] = []
        kd_id_parts: list[np.ndarray] = []
        kd_lp_parts: list[np.ndarray] = []
        seg_lengths: list[int] = []
        kd_k = None
        if kd_segments is not None:
            kd_k = int(np.asarray(kd_segments[0][0]).shape[-1])
        for si, (ids, lab) in enumerate(segments):
            ids = np.asarray(ids)
            lab = np.asarray(lab)
            if ids.ndim != 1 or ids.size == 0:
                raise ValueError(f"segment input_ids must be 1-D and non-empty, got shape {ids.shape}")
            if lab.shape != ids.shape:
                raise ValueError(f"labels shape {lab.shape} != input_ids shape {ids.shape}")
            ids_parts.append(ids)
            lab_parts.append(lab)
            n = int(ids.size)
            if kd_segments is not None:
                k_ids, k_lp = (np.asarray(x) for x in kd_segments[si])
                if k_ids.shape != (n, kd_k) or k_lp.shape != (n, kd_k):
                    raise ValueError(
                        f"segment {si}: kd arrays must be [{n}, {kd_k}], got "
                        f"{k_ids.shape} and {k_lp.shape}")
                kd_id_parts.append(k_ids)
                kd_lp_parts.append(k_lp)
            # Alignment padding rides INSIDE the segment: same seq_idx, same
            # cu_seqlens entry. It sits after the real tokens, so causal attention
            # never lets a real token see it, its labels are -100 so it contributes
            # no loss and no gradient, and the next segment now starts on a chunk
            # line. Giving it its own segment instead would only move the unaligned
            # boundary one place to the left.
            pad = round_up(n, self.segment_multiple) - n
            if pad:
                ids_parts.append(np.full(pad, self.pad_token_id, dtype=np.int64))
                lab_parts.append(np.full(pad, self.label_pad_token_id, dtype=np.int64))
                if kd_segments is not None:
                    kd_id_parts.append(np.full((pad, kd_k), _KD_IGNORE_ID, dtype=np.int32))
                    kd_lp_parts.append(np.zeros((pad, kd_k), dtype=np.float16))
            seg_lengths.append(n + pad)

        filled = sum(seg_lengths)
        if self.pad_to_len is not None:
            missing = self.pad_to_len - filled
            if missing < 0:
                raise ValueError(f"bin length {filled} exceeds pad_to_len {self.pad_to_len}")
            if missing > 0:
                # Tail padding is its OWN segment, not appended to the last real
                # one: it gets its own position reset and cu_seqlens entry, so the
                # last real sequence cannot attend to it (and vice versa). Labels
                # are all -100, so it contributes nothing to the loss.
                ids_parts.append(np.full(missing, self.pad_token_id, dtype=np.int64))
                lab_parts.append(np.full(missing, self.label_pad_token_id, dtype=np.int64))
                if kd_segments is not None:
                    kd_id_parts.append(np.full((missing, kd_k), _KD_IGNORE_ID, dtype=np.int32))
                    kd_lp_parts.append(np.zeros((missing, kd_k), dtype=np.float16))
                seg_lengths.append(missing)

        total = sum(seg_lengths)
        bad = [n for n in seg_lengths if n % self.segment_multiple]
        if bad:
            raise AssertionError(
                f"segment lengths {bad} are not multiples of {self.segment_multiple} "
                "after alignment padding -- Mamba segment isolation would silently break")
        input_ids = torch.from_numpy(np.concatenate(ids_parts).astype(np.int64, copy=False))
        labels = torch.from_numpy(np.concatenate(lab_parts).astype(np.int64, copy=False))

        # text axis restarts at 0 for every segment; for m-rope it is replicated
        # across all 4 axes (axis 0 -> mask/packing detection, axes 1:4 -> rotary)
        pos = np.concatenate([np.arange(n, dtype=np.int64) for n in seg_lengths])
        pos_t = torch.from_numpy(pos)
        position_ids = (pos_t.view(1, 1, -1).expand(4, 1, -1).contiguous()
                        if self.rope_axes == 4 else pos_t.view(1, -1))

        seg = np.asarray(seg_lengths, dtype=np.int64)
        cu = np.zeros(len(seg_lengths) + 1, dtype=np.int32)
        np.cumsum(seg, out=cu[1:], dtype=np.int32)
        # int32: required by the FA4 cute kernel; the FLA kernels are dtype-generic
        cu_seq_lens = torch.from_numpy(cu)
        seq_idx = torch.from_numpy(
            np.repeat(np.arange(len(seg_lengths), dtype=np.int32), seg)
        ).view(1, total)

        dev = self.device
        batch = {
            "input_ids": input_ids.view(1, total).to(dev, non_blocking=True),
            "labels": labels.view(1, total).to(dev, non_blocking=True),
            "position_ids": position_ids.to(dev, non_blocking=True),
            "cu_seq_lens_q": cu_seq_lens.to(dev, non_blocking=True),
            "cu_seq_lens_k": cu_seq_lens.clone().to(dev, non_blocking=True),
            # ints, not tensors: flash_attn_varlen_func requires python ints
            "max_length_q": int(seg.max()),
            "max_length_k": int(seg.max()),
            "seq_idx": seq_idx.to(dev, non_blocking=True),
        }
        if kd_segments is not None:
            kd_ids = torch.from_numpy(np.concatenate(kd_id_parts).astype(np.int32, copy=False))
            kd_lp = torch.from_numpy(np.concatenate(kd_lp_parts).astype(np.float16, copy=False))
            if kd_ids.shape[0] != total:
                raise AssertionError(
                    f"kd rows {kd_ids.shape[0]} != packed length {total}; the KD arrays "
                    "did not follow the labels through padding")
            batch["kd_ids"] = kd_ids.view(1, total, kd_k).to(dev, non_blocking=True)
            batch["kd_logprobs"] = kd_lp.view(1, total, kd_k).to(dev, non_blocking=True)
        # NOTE: no "attention_mask" key -- see module docstring.
        return batch


def assert_chunk_aligned(batch: dict, chunk_size: int) -> None:
    """Every packed segment boundary must sit on a multiple of ``chunk_size``.

    Called once on the first batch rather than inside the mixer: the check needs a
    host-side decision, so running it per-layer would force 23 device syncs per
    microstep on Nemotron-H. The collator guarantees the property by construction;
    this verifies the guarantee actually reached the tensors.
    """
    cu = batch["cu_seq_lens_q"]
    off = [int(x) for x in cu.tolist()]
    bad = [x for x in off[1:-1] if x % chunk_size]
    if bad:
        raise AssertionError(
            f"packed segment boundaries {bad} are not multiples of chunk_size="
            f"{chunk_size}. mamba_ssm's seq_idx resets the SSM state only at chunk "
            "granularity, so these boundaries leak state into the next segment. "
            "Build the collator with segment_multiple=chunk_size.")
    seq_idx = batch["seq_idx"]
    edges = (seq_idx[0, 1:] != seq_idx[0, :-1]).nonzero().flatten() + 1
    mismatch = sorted(set(int(x) for x in edges.tolist()) ^ set(off[1:-1]))
    if mismatch:
        raise AssertionError(
            f"seq_idx boundaries disagree with cu_seq_lens at {mismatch[:8]} -- the "
            "attention and Mamba views of the packing are not the same")
