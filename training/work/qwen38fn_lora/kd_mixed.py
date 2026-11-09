"""--kd-*: the rows of one origin of a merged tito-mm dataset train on KL32_tail against teacher top-K targets instead
of CE; every other row keeps CE (train_tito_mm.py).

First use, gd-v3-rKL (user 11-01: "train arc-instruct v3 once more, but train the QAD data with reverse-KL"): gd-v3's qad20m rows
(clean-pool QAD-R student trajectories cut to <= 20k tokens) take QAD-R's loss -- reverse KL32_tail
(qwen38fn_qad/topk_tail_kl.py) against the 512E FP8 teacher's top-32 -- with the targets sliced from QAD-R's teacher
capture (work/grid_describe/gdv3_rkl/build_kd_targets.py). Weight 1 per supervised position, the loss summed with the
CE and divided by the global supervised count like the CE alone (user 11-01).

The targets are a qad-topk-v1 store bound to the KD origin's own dataset (``--kd-data``); training rows are matched to
its rows by key, and every pack checks the teacher label column against the row's supervised tokens.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

QAD_DIR = Path(__file__).resolve().parents[1] / 'qwen38fn_qad'
if str(QAD_DIR) not in sys.path:
    sys.path.append(str(QAD_DIR))
import topk_tail_kl  # noqa: E402
import topk_targets  # noqa: E402

MODULES = ('topk_tail_kl.py', 'topk_targets.py')  # copied into a run's code/ beside this file


class KdPack:
    """The KD tensors of one pack (KdTargets.batch) plus the trainer's ``direction`` / ``update``. A plain class on
    purpose: FSDP2's cast_forward_inputs (_apply_to_tensors) walks tensors, dicts, lists, tuples and dataclasses in the
    forward's arguments and casts them to the BF16 param dtype -- the fp32 teacher log-probs included (smoke 11-01:
    "teacher log-probs must be fp32") -- and leaves other objects alone."""
    __slots__ = ('mask', 'ids', 'lp', 'tail', 'labels', 'n_kd', 'n_rest', 'direction', 'update')

    def __init__(self, **fields):
        if set(fields) != set(self.__slots__):
            raise ValueError(f'KdPack fields {sorted(fields)}')
        for k, v in fields.items():
            setattr(self, k, v)


class KdTargets:
    """Teacher targets of the ``origin`` rows of one split of ``data`` (a merged TitoMMDataset), by row key."""

    def __init__(self, targets_dir, kd_data_dir, data, origin):
        from tito_mm_data import TitoMMDataset

        self.origin = origin
        self.kd = TitoMMDataset(kd_data_dir, data.split)
        self.targets = topk_targets.TopkTargets(targets_dir, self.kd)
        self.where = {key: j for j, key in enumerate(self.kd.keys)}
        if len(self.where) != len(self.kd):
            raise RuntimeError(f'{kd_data_dir} {data.split}: duplicate keys')
        mine = {r['key']: r for r in data.index if r['origin'] == origin}
        if set(mine) != set(self.where):
            raise RuntimeError(f'{data.data_dir} {data.split}: {len(mine)} {origin!r} rows, {kd_data_dir} has '
                               f'{len(self.where)}; {len(set(mine) ^ set(self.where))} keys in only one of them')
        for key, r in mine.items():
            k = self.kd.index[self.where[key]]
            if (r['length'], r['n_loss']) != (k['length'], k['n_loss']):
                raise RuntimeError(f'{key}: length / n_loss {r["length"]}/{r["n_loss"]} here, '
                                   f'{k["length"]}/{k["n_loss"]} in {kd_data_dir}')
        self.positions = int(sum(r['n_loss'] for r in mine.values()))

    def batch(self, rows, origins, device):
        """KdPack of one pack, or None when none of its rows is a KD row: ``mask`` over the pack's supervised
        positions in pack order (True = KD), the KD positions' teacher ``ids`` / ``lp`` / ``tail`` and ``labels``,
        and the host counts ``n_kd`` / ``n_rest``; the caller sets ``direction`` and ``update``."""
        if len(rows) != len(origins):
            raise ValueError(f'{len(rows)} rows, {len(origins)} origins')
        flags, parts = [], []
        for row, origin in zip(rows, origins, strict=True):
            loss = row['loss_mask']
            n = int(loss[1:].sum())  # pack_segments never supervises a segment's first token
            is_kd = origin == self.origin
            if is_kd:
                if loss[0]:
                    raise RuntimeError(f'{row["key"]}: a KD row supervises its first token (no target position for it)')
                t = self.targets.row(self.where[row['key']])
                if not np.array_equal(t['labels'], row['input_ids'][loss]):
                    raise RuntimeError(f'{row["key"]}: teacher labels differ from the row\'s supervised tokens')
                if not t['keep'].all():  # the gd-v3 targets have none (build_kd_targets.py: flagged 0)
                    raise RuntimeError(f'{row["key"]}: flagged teacher positions are not supported in the mixed loss')
                parts.append(t)
            flags.append(np.full(n, is_kd))
        if not parts:
            return None
        mask = np.concatenate(flags)

        def cat(key, dtype):
            return torch.from_numpy(np.concatenate([t[key] for t in parts])).to(device=device, dtype=dtype)

        n_kd = int(mask.sum())
        return KdPack(mask=torch.from_numpy(mask).to(device), ids=cat('ids', torch.int64), lp=cat('lp', torch.float32),
                      tail=cat('tail', torch.float32), labels=cat('labels', torch.int64), n_kd=n_kd,
                      n_rest=len(mask) - n_kd, direction=None, update=None)


def mixed_loss(shifted, wanted, weight, kd):
    """Summed loss of one pack and its terms.

    kd.update (train): CE (LigerFusedLinearCrossEntropyLoss, sum) on the non-KD positions + KL32_tail in ``kd.direction``
    (fused_topk_tail_kl, sum; the LM head must be frozen) on the KD positions. Terms: ``ce_sum``, ``kd_sum``.
    eval: the loss is the CE over every position (comparable with a CE-only run's validation loss); terms add the
    KD positions' fp64 forward / reverse KL32_tail sums ``kd_fkl`` / ``kd_rkl``.
    """
    from liger_kernel.transformers.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyLoss

    if not isinstance(kd, KdPack) or kd.update not in (True, False) or kd.direction not in topk_tail_kl.DIRECTIONS:
        raise TypeError('mixed_loss takes a KdPack with update and direction set')
    if kd.lp.dtype != torch.float32 or kd.tail.dtype != torch.float32:
        raise TypeError(f'teacher log-probs must be fp32, got {kd.lp.dtype} / {kd.tail.dtype}')
    ce = LigerFusedLinearCrossEntropyLoss(reduction='sum')
    mask = kd.mask
    if mask.shape != wanted.shape:
        raise RuntimeError(f'KD mask over {mask.shape[0]} positions, the pack has {wanted.shape[0]} supervised')
    if not torch.equal(wanted[mask], kd.labels):
        raise RuntimeError('teacher targets are not aligned with this pack\'s labels')
    hidden = shifted[mask]
    if not kd.update:
        both = topk_tail_kl.topk_tail_kl_both(hidden, weight, kd.ids, kd.lp, kd.tail)
        loss = ce(weight, shifted, wanted)
        return loss, {'ce_sum': loss.detach(), 'kd_fkl': both[0], 'kd_rkl': both[1]}
    kl = topk_tail_kl.fused_topk_tail_kl(hidden, weight, kd.ids, kd.lp, kd.tail, direction=kd.direction)
    if kd.n_rest:
        rest = ~mask
        ce_sum = ce(weight, shifted[rest], wanted[rest])
    else:
        ce_sum = torch.zeros((), dtype=kl.dtype, device=kl.device)
    return ce_sum + kl, {'ce_sum': ce_sum.detach(), 'kd_sum': kl.detach()}
