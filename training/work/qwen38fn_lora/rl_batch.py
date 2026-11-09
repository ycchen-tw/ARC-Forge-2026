"""Reader for an RL batch (``tito-rl-v1``). The producer is ``prepare_rl.py``; the contract is
``work/qwen38fn_rl/SPEC.md`` §2, §7.

Format = the ``tito-mm-v1`` directory of ``tito_mm_data.py`` (single split ``train``, ``routing.i16`` mandatory);
``train/`` additionally has:

| File | Content |
|---|---|
| ``behavior_logprob.f32`` | one per position (little-endian float32): at loss_mask positions = the sampling logprob recorded in the ledger (position p holds the logprob of the sampled token ``input_ids[p]``, after the sampling temperature, before top-p), NaN elsewhere |
| ``coef.f32`` | one per position: at loss_mask positions = ``A_i / denominator_i`` (constant within one sample; for the denominator see ``coef_denominator``), 0 elsewhere |
| ``routing_status.u8`` | per position: 0 = R (recorded), 1 = N (never forwarded, only at the segment end); the kept data has no other values |
| ``weight_version.i32`` | per position: at loss_mask positions = the weight version the token was sampled with, −1 elsewhere; all −1 for an ``unversioned`` batch |

**Truncated sampling (``manifest.rl.replay == "mask"``; top-p < 1 or top-k on, SPEC §1)**: ``behavior_logprob.f32`` is
log mu renormalized over the support (after temperature, top-k / top-p), and there are also:

| File | Content |
|---|---|
| ``support_len.u16`` | per position: at loss_mask position p = the support size when ``input_ids[p]`` was sampled (≥ 1), 0 elsewhere |
| ``support_ids.i32`` | the support ids of every loss position, concatenated in position order; sample i's run starts at the index's ``support_offset`` and is ``support_tokens`` long |
| ``behavior_logprob_full.f32`` | per position: at loss_mask positions = the ledger's full-vocabulary logprob (before truncation; diagnostics only), NaN elsewhere |

A batch without the ``replay`` key (from before prepare_rl gained --top-p / --top-k) = ``none`` (top_p 1, top_k off).

Each ``index.jsonl`` row additionally has ``episode``, ``group``, ``reward``, ``advantage``, ``gen_tokens`` (the
episode's generated token count L_i), ``weight_version`` (same as ``manifest.rl.weight_version``),
``weight_version_tokens`` (the episode's loss-token count per version), ``run``, ``coef`` (this row's coef value),
``minibatch``, ``pad``, ``pad_of``.

**Pad samples (``pad: true``)**: every rank needs one bin per micro-step (FSDP collectives); when a minibatch's samples
do not fill a multiple of ``world``, the gap is filled with copies of that minibatch's shortest real sample:
tokens / media / routing / behavior_logprob identical to the original, ``loss_mask`` unchanged, but **coef all 0**,
``episode`` / ``group`` / ``reward`` null, ``advantage`` = 0, ``gen_tokens`` = 0, ``pad_of`` = the original sample's key.
The loss ``-Σ coef·w·logp`` is identically 0 for it and so is its gradient; the trainer needs no special handling, but
must exclude these via ``RLBatch.pad`` when reporting token counts, |logp − behavior| and similar metrics.

``manifest["rl"]``: ``round``, ``weight_version`` (v1: one version = a string, including ``unversioned``; async: the
allowed set = a list sorted by version number, SPEC §8; streaming: the versions that actually occur, same rule),
``weight_version_tokens``, ``temperature``, ``algo`` (``bpo`` / ``mis``) and its parameters (``rl_loss.ALGOS``:
bpo = ``eps``, ``cap``, ``eps_low``, ``eps_high``; mis = ``mis_low``, ``mis_high``; parameters of another algo must not
appear), ``world``, ``bin_tokens``, ``episodes``, ``groups``, ``dropped`` (by reason), ``current_version`` /
``max_staleness`` (the streaming staleness window V, S; both absent or both null = not streaming), ``minibatches`` =
``[{"steps": steps, "episodes": E, ...}, ...]``, where ``steps[m][rank]`` is the list of sample indices for that rank in
the m-th micro-step (one minibatch = one optimizer step; the episodes of one group are always in the same minibatch;
every bin ≤ ``bin_tokens``).
``loss_reduction``: ``sequence`` = denominator E · L_i (E = the minibatch's episode count); ``prompt`` = denominator
H · T_g (H = the minibatch's group count = ``len(minibatches[m]["groups"])``, T_g =
``minibatches[m]["group_tokens"][group]``, the sum of ``gen_tokens`` over the group's kept episodes). Batches without
this key (from before prepare_rl gained the option) are ``sequence``. ``adv_normalization`` (``std`` / ``none``) is
only a record; the reader does not use it.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rl_loss import ALGOS  # noqa: E402
from tito_mm_data import TitoMMDataset  # noqa: E402

RL_FORMAT = "tito-rl-v1"
SPLIT = "train"
ROUTING_R = 0                      # recorded
ROUTING_N = 1                      # never forwarded (the last position of a segment)
#: extra per-position files of tito-rl-v1 → dtype
RL_FILES = {"behavior_logprob.f32": "<f4", "coef.f32": "<f4", "routing_status.u8": np.uint8, "weight_version.i32": "<i4"}
#: per-position files of a replay: mask batch (truncated sampling) → dtype; support_ids.i32 is per support entry
REPLAY_FILES = {"support_len.u16": "<u2", "behavior_logprob_full.f32": "<f4"}
REPLAYS = ("none", "mask")
UNVERSIONED = "unversioned"
RL_INDEX_KEYS = ("episode", "group", "reward", "advantage", "gen_tokens", "weight_version", "run", "coef",
                 "minibatch", "pad", "pad_of")
RL_MANIFEST_KEYS = ("round", "weight_version", "temperature", "algo", "world", "bin_tokens", "episodes", "groups",
                    "dropped", "minibatches")
#: every batch without a ``loss_reduction`` key (written before prepare_rl gained --loss-reduction) is this
LEGACY_LOSS_REDUCTION = "sequence"
LOSS_REDUCTIONS = ("sequence", "prompt")
#: the streaming staleness window (SPEC §7): batches from before prepare_rl 2ddf2c89 lack both keys; --weight-version
#: batches have them null
WINDOW_KEYS = ("current_version", "max_staleness")
#: behavior logprobs are log-probabilities: anything above this is a broken record, not rounding
LOGPROB_MAX = 1e-5


def staleness_window(rl: dict, where) -> tuple[int, int] | None:
    """``manifest.rl``'s staleness window → ``(V, S)`` (token versions must lie in [max(0, V−S), V], the trainer at V);
    a batch that is not streaming (both keys absent = old batch, or both null = ``--weight-version``) → None. Only one
    of them present, or not a non-negative int → raise."""
    given = [k for k in WINDOW_KEYS if rl.get(k) is not None]      # absent and null are both "no window" (see WINDOW_KEYS)
    if not given:
        return None
    if len(given) != len(WINDOW_KEYS):
        raise RuntimeError(f"{where}: rl has only {given}: {' and '.join(WINDOW_KEYS)} must be given together")
    for k in WINDOW_KEYS:
        v = rl[k]
        if not isinstance(v, int) or isinstance(v, bool) or v < 0:
            raise RuntimeError(f"{where}: rl.{k}={v!r} is not a non-negative integer")
    return rl["current_version"], rl["max_staleness"]


def window_low(window: tuple[int, int]) -> int:
    """Oldest version the staleness window (V, S) allows: max(0, V−S)."""
    return max(0, window[0] - window[1])


class RLBatch(TitoMMDataset):
    """``RLBatch(path)``: one RL round's training data (SPEC §7).

    - ``len(b)``, ``b.keys``, ``b.lengths``, ``b.loss_tokens``, ``b.tokenizer_sha256``, ``b.manifest``: as in
      ``TitoMMDataset``.
    - ``b.rl`` = ``manifest["rl"]``; ``b.minibatches`` = ``b.rl["minibatches"]``; ``b.pad``: np.bool_, whether each row
      is a pad sample.
    - ``b[i]``: ``key``, ``input_ids`` int64, ``loss_mask`` bool, ``mrope_positions`` int32 (3, T), ``media`` (with
      pixels), ``routing`` int16 (T, layers, top_k), ``routing_status`` uint8 (T,), ``behavior_logprob`` float32 (T,),
      ``coef`` float32 (T,), ``weight_version`` int32 (T,) (per-token version; the batch-level value is in
      ``b.rl["weight_version"]``), plus the remaining fields of this index.jsonl row. ``b.tokens(i)`` is the same
      without reading pixels.
    - ``b.algo``: ``rl.algo`` (``bpo`` / ``mis``); ``b.algo_params``: its parameters in ``rl_loss.ALGOS[algo]`` order
      (float).
    - ``b.weight_versions``: the allowed set (``list[str]``, one element for v1).
    - ``b.window``: the streaming staleness window ``(V, S)`` (``staleness_window``; on open, checks that every
      ``weight_versions`` entry is inside it), else None.

    On open, validates the minibatch plan (every micro-step has exactly ``world`` non-empty bins, every sample appears
    exactly once, bin ≤ ``bin_tokens``, no group spans minibatches, each sample's ``minibatch`` field matches the plan);
    every read validates the shapes and rules of status / behavior / coef."""

    FORMAT = RL_FORMAT

    def __init__(self, data_dir: str | Path):
        super().__init__(data_dir, SPLIT, routing=True)
        if set(self.manifest["splits"]) != {SPLIT}:
            raise RuntimeError(f"{self.data_dir}: tito-rl-v1 has only split {SPLIT!r}, manifest has {sorted(self.manifest['splits'])}")
        self.rl = self.manifest["rl"]
        missing = [k for k in RL_MANIFEST_KEYS if k not in self.rl]
        if missing:
            raise RuntimeError(f"{self.data_dir}: manifest.rl is missing {missing}")
        self.algo = self.rl["algo"]
        if self.algo not in ALGOS:
            raise RuntimeError(f"{self.data_dir}: rl.algo={self.algo!r} not in {sorted(ALGOS)}")
        lack = [k for k in ALGOS[self.algo] if k not in self.rl]
        foreign = [k for algo, params in ALGOS.items() if algo != self.algo for k in params if k in self.rl]
        if lack or foreign:
            raise RuntimeError(f"{self.data_dir}: rl.algo={self.algo!r} is missing parameters {lack}, carries another algo's parameters {foreign}")
        #: values in ``ALGOS[algo]`` order (the params of rl_loss.algo_loss)
        self.algo_params = tuple(float(self.rl[k]) for k in ALGOS[self.algo])
        for r in self.index:
            lack = [k for k in RL_INDEX_KEYS if k not in r]
            if lack:
                raise RuntimeError(f"{self.data_dir}: index row {r['key']} is missing {lack}")
        n = int(self.lengths.sum())
        if n == 0:
            raise RuntimeError(f"{self.data_dir}: empty RL batch")
        d = self.data_dir / SPLIT
        self.replay = self.rl.get("replay", "none")
        if self.replay not in REPLAYS:
            raise RuntimeError(f"{self.data_dir}: rl.replay={self.replay!r} not in {REPLAYS}")
        truncated = self.rl.get("top_p", 1.0) < 1 or self.rl.get("top_k") is not None
        if truncated != (self.replay == "mask"):
            raise RuntimeError(f"{self.data_dir}: rl.replay={self.replay!r} does not match top_p={self.rl.get('top_p')!r}, "
                               f"top_k={self.rl.get('top_k')!r} (mask only with truncated sampling)")
        self._rl = {}
        files = dict(RL_FILES, **(REPLAY_FILES if self.replay == "mask" else {}))
        for name, dtype in files.items():
            arr = np.memmap(d / name, dtype=dtype, mode="r")
            if arr.shape[0] != n:
                raise RuntimeError(f"{d}/{name}: {arr.shape[0]} values, index total length {n}")
            self._rl[name] = arr
        if self.replay == "mask":
            ids = np.memmap(d / "support_ids.i32", dtype="<i4", mode="r")
            ends = [r["support_offset"] + r["support_tokens"] for r in self.index]
            if any(r["support_offset"] != (ends[i - 1] if i else 0) for i, r in enumerate(self.index)) \
                    or ids.shape[0] != (ends[-1] if ends else 0):
                raise RuntimeError(f"{d}/support_ids.i32: {ids.shape[0]} ids, not contiguous with / not matching the index's "
                                   "support_offset / support_tokens")
            self._rl["support_ids.i32"] = ids
        self.pad = np.array([bool(r["pad"]) for r in self.index], dtype=np.bool_)
        self.minibatches = self.rl["minibatches"]
        self.loss_reduction = self.rl["loss_reduction"] if "loss_reduction" in self.rl else LEGACY_LOSS_REDUCTION
        if self.loss_reduction not in LOSS_REDUCTIONS:
            raise RuntimeError(f"{self.data_dir}: rl.loss_reduction={self.loss_reduction!r} not in {LOSS_REDUCTIONS}")
        self.weight_versions = self._allowed_versions(self.rl["weight_version"])
        self._version_ints = None if self.weight_versions == [UNVERSIONED] else np.array(
            [int(v) for v in self.weight_versions], np.int32)
        self.window = staleness_window(self.rl, self.data_dir)
        if self.window is not None:
            low, current = window_low(self.window), self.window[0]
            if self._version_ints is None or not ((low <= self._version_ints) & (self._version_ints <= current)).all():
                raise RuntimeError(f"{self.data_dir}: rl.weight_version={self.rl['weight_version']!r} outside the staleness window "
                                   f"[{low}, {current}] (current_version {current}, max_staleness {self.window[1]})")
        self._check_plan()

    def _allowed_versions(self, wv) -> list[str]:
        """``rl.weight_version``: a string (v1, including unversioned) or a list of ≥ 2 distinct decimal version numbers
        sorted by version number."""
        vals = [wv] if isinstance(wv, str) else wv
        if not isinstance(vals, list) or not vals or (not isinstance(wv, str) and len(vals) < 2):
            raise RuntimeError(f"{self.data_dir}: rl.weight_version={wv!r} is neither a string nor a list of ≥ 2 versions")
        if vals == [UNVERSIONED]:
            return vals
        if not all(isinstance(v, str) and v.isdigit() and str(int(v)) == v for v in vals) or \
                vals != sorted(set(vals), key=int):
            raise RuntimeError(f"{self.data_dir}: rl.weight_version={wv!r}: versions must be decimal integer strings, sorted, no duplicates")
        return vals

    def _check_plan(self) -> None:
        world, cap = int(self.rl["world"]), int(self.rl["bin_tokens"])
        seen = np.zeros(len(self), np.int64)
        group_mb: dict[str, int] = {}
        for m, mb in enumerate(self.minibatches):
            if not mb["steps"]:
                raise RuntimeError(f"{self.data_dir}: minibatch {m} has no micro-step")
            episodes = set()
            group_tokens: dict[str, dict[str, int]] = collections.defaultdict(dict)
            for s, step in enumerate(mb["steps"]):
                if len(step) != world:
                    raise RuntimeError(f"{self.data_dir}: minibatch {m} step {s} has {len(step)} bins, world={world}")
                for rank, bin_ in enumerate(step):
                    if not bin_:
                        raise RuntimeError(f"{self.data_dir}: minibatch {m} step {s} rank {rank} is an empty bin")
                    for i in bin_:
                        if not 0 <= i < len(self):
                            raise RuntimeError(f"{self.data_dir}: minibatch {m} points to nonexistent sample {i}")
                        seen[i] += 1
                        r = self.index[i]
                        if r["minibatch"] != m:
                            raise RuntimeError(f"{self.data_dir}: {r['key']} has minibatch={r['minibatch']}, the plan puts it in {m}")
                        if not r["pad"]:
                            episodes.add(r["episode"])
                            group_tokens[r["group"]][r["episode"]] = r["gen_tokens"]
                            if group_mb.setdefault(r["group"], m) != m:
                                raise RuntimeError(f"{self.data_dir}: group {r['group']} spans minibatches "
                                                   f"{group_mb[r['group']]} and {m}")
                    tokens = int(self.lengths[bin_].sum())
                    if tokens > cap:
                        raise RuntimeError(f"{self.data_dir}: minibatch {m} step {s} rank {rank} bin has {tokens} tokens > {cap}")
            if len(episodes) != mb["episodes"]:
                raise RuntimeError(f"{self.data_dir}: minibatch {m} has {len(episodes)} episodes, manifest says {mb['episodes']}")
            if self.loss_reduction == "prompt":
                want = {g: sum(eps.values()) for g, eps in group_tokens.items()}
                if mb["group_tokens"] != want or sorted(mb["groups"]) != sorted(want):
                    raise RuntimeError(f"{self.data_dir}: minibatch {m} group_tokens / groups do not match the samples' gen_tokens sums")
        if (seen != 1).any():
            bad = int(np.flatnonzero(seen != 1)[0])
            raise RuntimeError(f"{self.data_dir}: sample {self.keys[bad]} appears {int(seen[bad])} times in the minibatch plan (must be exactly once)")

    def coef_denominator(self, minibatch: int, row: dict) -> int:
        """coef denominator of real sample ``row`` (an index row): ``coef == advantage / denominator`` (see
        ``loss_reduction`` in the module docstring)."""
        if row["pad"]:
            raise ValueError(f"{row['key']}: a pad sample has no denominator (its coef is always 0)")
        mb = self.minibatches[minibatch]
        if self.loss_reduction == "sequence":
            return mb["episodes"] * row["gen_tokens"]
        return len(mb["groups"]) * mb["group_tokens"][row["group"]]

    def tokens(self, i: int) -> dict:
        item = super().tokens(i)
        r = self.index[i]
        o, n = r["offset"], r["length"]
        status = np.array(self._rl["routing_status.u8"][o:o + n], dtype=np.uint8)
        behavior = np.array(self._rl["behavior_logprob.f32"][o:o + n], dtype=np.float32)
        coef = np.array(self._rl["coef.f32"][o:o + n], dtype=np.float32)
        loss = item["loss_mask"]
        if (status[:n - 1] != ROUTING_R).any() or status[n - 1] != ROUTING_N:
            raise RuntimeError(f"{r['key']}: routing_status is not 'N at the segment end, R elsewhere': {np.unique(status).tolist()}")
        if (~np.isfinite(behavior[loss])).any() or (behavior[loss] > LOGPROB_MAX).any():
            raise RuntimeError(f"{r['key']}: behavior_logprob at loss positions has non-finite values or > 0")
        if not np.isnan(behavior[~loss]).all():
            raise RuntimeError(f"{r['key']}: behavior_logprob at non-loss positions is not NaN")
        if (coef[~loss] != 0).any():
            raise RuntimeError(f"{r['key']}: coef at non-loss positions is not 0")
        want = np.float32(0.0 if r["pad"] else r["coef"])
        if (coef[loss] != want).any():
            raise RuntimeError(f"{r['key']}: coef at loss positions is not the index's {float(want)} (pad={r['pad']})")
        version = np.array(self._rl["weight_version.i32"][o:o + n], dtype=np.int32)
        if self._version_ints is None:
            if (version != -1).any():
                raise RuntimeError(f"{r['key']}: unversioned batch but has per-token versions")
        elif not np.array_equal(version != -1, loss) or not np.isin(version[loss], self._version_ints).all():
            raise RuntimeError(f"{r['key']}: per-token versions are not 'loss positions ∈ {self.weight_versions}, −1 elsewhere': "
                               f"{np.unique(version).tolist()}")
        if self.replay == "mask":
            lens = np.array(self._rl["support_len.u16"][o:o + n], dtype=np.int32)
            so, sn = r["support_offset"], r["support_tokens"]
            ids = np.array(self._rl["support_ids.i32"][so:so + sn], dtype=np.int32)
            full = np.array(self._rl["behavior_logprob_full.f32"][o:o + n], dtype=np.float32)
            if not np.array_equal(lens > 0, loss) or int(lens.sum()) != sn:
                raise RuntimeError(f"{r['key']}: support_len is not 'loss positions ≥ 1, 0 elsewhere' or its sum ≠ support_tokens {sn}")
            if (~np.isfinite(full[loss])).any() or (full[loss] > LOGPROB_MAX).any() or not np.isnan(full[~loss]).all():
                raise RuntimeError(f"{r['key']}: behavior_logprob_full is not 'finite ≤ 0 at loss positions, NaN elsewhere'")
            # every kept token is in its own support, exactly once
            row = np.repeat(np.flatnonzero(loss), lens[loss])
            hits = np.bincount(row[ids == item["input_ids"][row]], minlength=n)
            if (hits[loss] != 1).any():
                raise RuntimeError(f"{r['key']}: a loss position's token is not in its support (or is in it more than once)")
            item.update(support_len=lens, support_ids=ids, behavior_logprob_full=full)
        item["mrope_positions"] = item["mrope_positions"].astype(np.int32)
        item.update(routing_status=status, behavior_logprob=behavior, coef=coef, weight_version=version)
        for k, v in r.items():
            if k not in item and k != "media":
                item[k] = v
        return item
