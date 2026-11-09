"""TITO ledger + one round's rollout ``tito_manifest.json`` → RL batch (``tito-rl-v1``; reader ``rl_batch.RLBatch``).

Contract: ``work/qwen38fn_rl/SPEC.md`` §1 (advantage / coef), §2 (format, cases that must raise), §5 (routing status), §7
(CLI). One sample = one ledger segment, the same as ``prepare_tito_mm.py``: ``input_ids`` = ledger S, ``loss_mask`` =
``Segment.gen_spans`` (rewound tails, template-filled tokens, tool output, prompt and media pads are not loss); media and
M-RoPE go through ``prepare_tito_mm``'s deployed-processor worker (needs sglang: ``prepare_rl.sh``). When the whole batch
has no media, sglang is not needed (M-RoPE = arange).

1. **join**: episode = ``<run>/<item id>`` (run = name of the directory holding the manifest), reward = the outcome's
   ``score`` (arc2: test-pair partial-credit pass@1; arc3 single-level episodes, ``work/arc_code/run_arc3_level.py``:
   cleared within quota = 1). Two kinds of manifest (only one kind per call, ``manifest_groups``):
   - classic: each manifest is one sample run over the same set of tasks (s0…s7); group = task = item id (one task across
     runs = one GRPO group).
   - group mode (streaming async): one manifest = one group, every item id is ``<tid>__s<k>`` (one tid, each k = 0…n−1
     once), group = ``<group run dir name>:<tid>`` (one batch may hold two groups of the same task), the index's
     ``task`` = tid.
   The rid join uses ``arcforge.tito.join.rl_rid_join``: only "the last turn that hit the budget" (for arc3 also the last
   turn killed by a level clear / action cap / wall) and "the turn cut by the wrap-up line" (the outcome's
   ``wrapup_cut_rid``) may be extra. Segments without seg_close (sessions cut off at the budget) are kept; turns truncated
   by max_new_tokens (finish length) are kept too -- both are on-policy samples, trained with the episode's reward.
2. **raise** (SPEC §2 / §5): a manifest that is neither arc2 nor arc3 with score, or one call mixing the two; classic and
   group mode mixed, a group manifest not of one task or with sample numbers other than 0…n−1; an episode whose manifest
   records no compaction has a segment with window ≠ 0; generated tokens without logprobs;
   sampling parameters other than T = ``--temperature``, top_p 1, top_k off, min_p 0, neutral penalties (all seven fields
   must be recorded); a token's weight version outside the allowed set of ``--weight-version`` (comma-separated; one value =
   v1, ``unversioned`` = the ledger recorded no version at all, only for ledgers from before the fork fix; async is
   ``k-1,k``, SPEC §8), or (streaming: ``--current-version V --max-staleness S``) newer than V;
   an append without a version key that is not unversioned; routing X (duplicate ids, out of range, all-zero rows);
   N anywhere but the segment end; episodes dropped for M above ``--max-routing-m-fraction`` (default 0.5%), dropped for
   unrecorded versions above 0.5%, dropped for D_cross above ``--max-d-cross-fraction``
   (default ``arcforge.rl_storage.MAX_D_CROSS_FRACTION`` 2%) (``--defer-fraction-gates``: not judged here; the summary's
   ``quality`` gives the counts and the orchestrator's group_ready judges them over the whole optimizer step); behavior
   logprobs misaligned with the ledger; session resume contract violations (``ledger.resume_violations``, contract v2.1);
   an extension turn with neither resume.pos nor cached_tokens (no way to know which positions were forwarded twice);
   a versioned batch with D_unknown not caused by unrecorded ``weight_versions`` (such episodes are already dropped as
   ``weight_version_unrecorded``); one episode's segments coming from servers with session_exact_resume on and off.
3. **Dropped episodes** (counted by reason): ``harness_failed`` (no reward in the outcome), ``compacted`` (a thread with
   more than one window or a compaction request: RL v1 does not train compaction; ``--allow-compaction`` (whole-game
   episodes, run_arc3_level's arc3_game) keeps them: every window is its own segment, the summary is the generation of
   the last append of window n's segment, and the whole episode shares one advantage), ``context_overflow`` (outcome
   ``context_overflows`` > 0: the harness hit the context length and opened a new thread), ``no_tokens`` (the ledger has
   no generation for it), ``routing_M`` (ledger routing missing: routing_error / null, late branch, sentinel; details
   counted separately), ``weight_version_unrecorded`` (an append's ``weight_versions`` is null), ``routing_D_cross``
   (a position forwarded twice with different weight versions: the ledger routing is the first forward's, later tokens
   used the second forward's KV, ``ledger.segment_reforward``), ``stale`` (streaming: a generated token (rewound tails
   included) has version < V−S), ``too_long`` (a segment > bin_tokens).
   Forwarded twice with the same version (D_same) is kept: routing from the first forward (like every other position, off
   only by serving's non-batch-invariance), written as R in the batch and counted; ``--weight-version unversioned``
   ledgers have no versions, so every D is D_unknown, likewise kept and counted.
   **The forward-twice rules (drop D_cross, its fraction gate, the D_unknown raise) are enforced only for episodes whose
   server had session_exact_resume on** (``Segment.exact_resume`` = ``server.json``'s
   ``server_args.session_exact_resume``): ledgers from the old fork or with the flag off are only counted, every D kept
   and written as R (``D_cross_unenforced_episodes``, one WARN line on stderr) -- the same as prepare before this rule.
   Afterwards a group with fewer than 2 episodes left (``group_lt2``) or with all advantages 0 (``group_std0``: all
   rewards equal and no efficiency term) is dropped whole.
4. **advantage** (``--adv-normalization``, over the episodes left in the group, before the minibatch split): ``std``
   (default) ``A_i = (R_i − mean_g) / std_g`` (population std); ``none`` ``A_i = R_i − mean_g`` (Dr. GRPO: no division
   by std).
   ``--efficiency-beta β > 0`` (only with ``none``) adds an efficiency term (``efficiency_advantages``): only among
   episodes with reward 1 (all tests correct) it compares ``e = 1 − L/B`` (L = the ledger's generated tokens
   ``gen_tokens``, exactly the tokens of this episode that enter the loss; before 10-22 it was the outcome's
   ``output_tokens`` = the proxy count, which includes the turns codex resent after truncation and discarded (10-18 audit
   A1: 449 episodes, 2.47M tokens), and so penalized tokens that never entered the trajectory; B = the harness's output
   budget, negative beyond B), ``A_eff = β_g (e_i − mean e of the all-correct)``; all-correct groups (task term all 0)
   therefore also carry a signal and are not dropped.
   ``--arc3-reward eff`` (only with harness arc3 and ``none``, not with ``--efficiency-beta``; ``arc3_eff_rewards``): the
   reward becomes the within-group relative ``R_i = c_i (1 + w(p̂) E_i)``, c = cleared, p̂ = the group's clear rate,
   E = log differences of action count (×2) and ``tokens_at_clear`` among the clearing episodes (clipped, then re-centred
   among the clearing episodes), ``A_i = R_i − mean R = (c_i − p̂) + w c_i E_i``; all-clear groups therefore also carry a
   signal. The summary and ``rl`` record ``arc3_reward``.
   ``--arc3-reward c0`` (same pairing restrictions; ``arc3_c0_rewards``): E is replaced by a smooth term centred among the
   clearing episodes, ``D = λ_a (q_a − mean q_a) + λ_t (q_t − mean q_t)``, q = σ(κ (mean ln cost of the clearing
   episodes − ln cost)) (action count, ``tokens_at_clear``), not clipped; ``R_i = c_i (1 + s w D_i)``, s keeps every
   clearing R ≥ success_floor. The mean R of the clearing episodes is 1; cleared vs not cleared always differ by 1.
   ``--arc3-reward official`` (same restrictions; ``arc3_official_rewards``): ``R_i = c_i f(S_i)``, S = the official
   per-level score min(100 (baseline / actions)², 115), f piecewise linear through (top_score, 1),
   (knee_score, knee_reward), (zero_score, 0); no token term.
   ``--arc3-reward rank`` (same restrictions; ``arc3_rank_rewards``, ``arcforge.arc3_rank``):
   ``R_i = s_i (1 − α r_act − β r_tok)``, s = the outcome's score (g2seed's κ score, may lie between 0 and 1), r = the
   mean rank of action count / ``tokens_at_clear`` among the clearing episodes, normalized to [0,1] (0 best; r = 0 with
   fewer than 2 clears).
   ``--arc3-reward hinge`` (same restrictions; ``arc3_hinge_rewards``, ``arcforge.arc3_hinge``, RL v7):
   ``R_i = s_i + f (mean_cleared D − D_i)``, D = s·(λ_t h_t + λ_a h_a), h = the fraction by which tokens_at_clear exceeds
   the level's frozen threshold T (full deduction at 3T) and the action count exceeds A (full at 2A); f = the clear
   fraction of the kept episodes when this group's gid is past the level's freeze (applies_from_gid in the refs file),
   else 0.
   ``--arc3-reward note`` (same restrictions, group mode only; ``arc3_note_rewards``, ``arcforge.arc3_note``, RL v8: an
   episode ends at its first window change): a clear gets ``R = clear_reward(actions, baseline)`` (official score,
   exponential penalty beyond baseline); an episode with stop window and window_cause threshold gets R = the reward of its
   note in the group directory's ``judge.json`` (written by the orchestrator's note judge) -- recomputed from grade /
   actions / remaining_actions with ``arc3_note.note_reward`` and required to equal the one in the file; early (called
   new_context itself before the window change point) and other stops get 0. A group with episodes to judge but no
   judge.json, or a judge.json missing an episode or holding an extra one, raises; the summary records each judge.json's
   sha256.
   Every arc3 mode records ``training_reward`` in the index (= R; the advantage is it minus the group mean; score mode =
   reward).
5. **minibatch**: K of them; groups are not split, LPT-balanced by token count (ties shuffled by seed). coef
   (``--loss-reduction``): ``sequence`` (default) ``coef = A_i / (E · L_i)``, E = this minibatch's episode count, L_i =
   the loss tokens over all of episode i's segments (each episode averaged on its own); ``prompt``
   ``coef = A_i / (H · T_g)``, H = this minibatch's group count, T_g = the sum of L over the kept episodes of episode i's
   group (one group's tokens averaged together, every group weighted equally).
6. **bin**: each minibatch's samples go LPT into B bins (B = a multiple of world, counting up from the best-fit bin count,
   each ≤ bin_tokens); after sorting by size, each run of world consecutive bins is one micro-step (bins of one step are of
   similar size). With fewer samples than bins, empty bins get a copy of the minibatch's shortest sample (``pad``, coef
   all 0, see ``rl_batch.py``).
7. **behavior_logprob**: position p (a loss position) = the logprob recorded by the append that sampled ``input_ids[p]``.
   Asserts ``input_ids[span] == append.gen[:len(span)]`` span by span (the sampled id is the token at that position)
   before filling; on read-back it is re-derived independently from the append records.
   **weight_version.i32** uses the same alignment: loss positions = the version given by that append's
   ``weight_versions`` ranges, −1 elsewhere.
   ``manifest.rl.weight_version``: a single value stays a string, a set is a list sorted by version number;
   ``weight_version_tokens`` is recorded too. In streaming it is the versions actually present in the batch (same rule:
   one = string, several = list), plus ``current_version``, ``max_staleness`` and staleness (V − version) histograms:
   ``staleness_loss_tokens`` (loss-token counts), ``staleness_episodes`` (kept episodes, by their oldest loss token),
   ``staleness_episodes_stale`` (episodes dropped as ``stale``, by their oldest generated token),
   ``staleness_stale_tokens`` (generated tokens of episodes dropped as ``stale``, each by its own staleness; unrecorded
   versions under ``<absent>`` / ``<unrecorded>``); all null with ``--weight-version``.
8. After writing, ``RLBatch`` reads back **every row** and compares ids / loss / behavior / coef bit for bit with the
   ledger (routing sampled), and checks each minibatch's Σcoef = Σ_i A_i · L_i / denominator_i (Σ_i A_i / E for
   sequence). The last stdout line is a one-line JSON summary.
9. ``--error-json`` (the orchestrator's group_ready): a ``ManifestError`` that points at manifests (one group's data
   incomplete) and routing over ``--max-gb`` do not raise; they are written as machine-readable JSON and exit with
   ``PREPARE_EXIT_CODES`` (``cli``); everything else still raises (run-level errors still stop).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import json
import multiprocessing as mp_ctx
import os
import random
import re
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_tito_mm as ptm  # noqa: E402
from rl_batch import LOGPROB_MAX, RL_FORMAT, ROUTING_N, ROUTING_R, SPLIT, RLBatch  # noqa: E402
from rl_loss import ALGOS, check_algo_params  # noqa: E402

UNVERSIONED = "unversioned"
#: special keys of segment_versions: the append has no version key (before the fork fix) / weight_versions is null
#: (the fork failed to record it)
ABSENT, UNRECORDED = "<absent>", "<unrecorded>"
#: SPEC §5: cap on the fraction of a round's episodes dropped for routing M (default of ``--max-routing-m-fraction``)
MAX_M_DROP_FRACTION = 0.005
#: SPEC §8: cap on the fraction of a round's episodes dropped because weight_versions was not recorded (null)
MAX_UNRECORDED_FRACTION = 0.005
#: the three forward-twice classes among the routing statuses (ROUTING_* of arcforge.tito.ledger); summary / manifest keys
#: use these names
D_CLASSES = ("D_same", "D_unknown", "D_cross")
#: manifest rl.reforward_rule (SPEC §5). Recorded batch metadata, kept verbatim. In English: positions forwarded twice
#: (recompute window [reuse, prefix_len-1), reuse = resume.pos, else usage.cached_tokens) are classed D_same / D_unknown /
#: D_cross; routing is always the first forward's. Enforcement depends on the segment's server (server.json
#: server_args.session_exact_resume, must agree across one episode's segments): on -- D_same kept and written as R;
#: D_cross episodes dropped (routing_D_cross, fraction gate max_d_cross_fraction); in a versioned batch, D_unknown not
#: caused by unrecorded weight_versions → raise; off (old fork or flag off) -- only counted, every D kept and written as R
#: (D_cross_unenforced_episodes / kept_D_cross_positions), as prepare did before this rule. The index's routing_D_same /
#: routing_D_unknown / routing_D_cross are each row's counts of positions written as R. For ledgers without resume.pos
#: (old fork, flag off) cached_tokens reflects only the first admission (the part recomputed from the alignment point
#: after a retract is invisible), so D is a lower bound.
REFORWARD_RULE = (
    "forward 兩次（重算窗 [reuse, prefix_len-1)，reuse＝resume.pos，沒有就是 usage.cached_tokens）的位置分 D_same／D_unknown／"
    "D_cross，routing 一律是第一次 forward 的值。執行與否看段的 server（server.json server_args.session_exact_resume，"
    "同一個 episode 的段要一致）：開了——D_same 照收、寫成 R；D_cross 的 episode 丟掉（routing_D_cross，比例門檻 "
    "max_d_cross_fraction）；有版本的 batch 出現不是因為 weight_versions 沒記到的 D_unknown → raise；沒開（舊 fork 或旗標關）——"
    "只計數，D 全部照收、寫成 R（D_cross_unenforced_episodes／kept_D_cross_positions），與沒有這條規則前的 prepare 相同。"
    "index 的 routing_D_same／routing_D_unknown／routing_D_cross 是每筆寫成 R 的位置數。沒有 resume.pos 的帳本（舊 fork、旗標關）"
    "的 cached_tokens 只反映第一次 admission（retract 後從對齊點重算的那段看不到），D 是下界")
ARC2 = "arc2"
#: arc3 single-level episodes (work/arc_code/run_arc3_level.py): the outcome has the same shape as arc2 (score / correct /
#: stop / context_overflows / wrapup_cut_rid / output_tokens); the rid join also lets through the last turn the harness
#: killed on a level clear / action cap / wall (join.RL_KILL_STOPS)
ARC3 = "arc3"
RL_HARNESSES = (ARC2, ARC3)
#: ``--arc3-reward``: score = the outcome's score (0/1, the original behavior); eff = within-group relative clear +
#: efficiency (``arc3_eff_rewards``); c0 = a smooth efficiency term centred among the clearing episodes, cleared vs not
#: cleared always differ by 1 (``arc3_c0_rewards``); official = piecewise linear of the official per-level score
#: (``arc3_official_rewards``, ``arcforge.arc3_official``); rank = score times a discount by within-group efficiency rank
#: (``arc3_rank_rewards``, ``arcforge.arc3_rank``)
ARC3_REWARDS = ("score", "eff", "c0", "official", "rank", "hinge", "note")
#: modes with an efficiency term → their parameters (CLI ``--arc3-<name>``, attribute ``arc3_<name>``): all must be given
#: explicitly, no defaults, and only with their own mode
ARC3_MODE_PARAMS = {"eff": ("beta_actions", "beta_tokens", "w_start", "eff_clip"),
                    "c0": ("lambda_actions", "kappa_actions", "lambda_tokens", "kappa_tokens", "w_start", "success_floor"),
                    # = arcforge.arc3_official.PARAMS (test_prepare_rl checks)
                    "official": ("top_score", "knee_score", "knee_reward", "zero_score"),
                    # = arcforge.arc3_rank.PARAMS (test_prepare_rl checks)
                    "rank": ("rank_actions", "rank_tokens", "rank_pass_scale", "rank_warmup_versions"),
                    # = arcforge.arc3_hinge.PARAMS (test_prepare_rl checks)
                    "hinge": ("hinge_tokens", "hinge_actions", "hinge_warmup_versions", "hinge_refs"),
                    # note (RL v8) has no parameters: the formula is in arcforge.arc3_note, the judge's scores in the
                    # group directory's judge.json
                    "note": ()}
#: outcome fields eff / c0 / rank read from the manifest (run_arc3_level.tito_items)
ARC3_EFF_OUTCOME = ("correct", "actions", "tokens_at_clear")
#: fields official reads: action count and this level's human baseline (the stub's baseline_actions)
ARC3_OFFICIAL_OUTCOME = ("correct", "actions", "baseline_actions")
#: fields note reads: cleared, action count, baseline, window-change cause (threshold / early for stop-window episodes,
#: None otherwise), RESET count (checked against judge.json)
ARC3_NOTE_OUTCOME = ("correct", "actions", "baseline_actions", "window_cause", "resets")
ARC3_OUTCOME = {"eff": ARC3_EFF_OUTCOME, "c0": ARC3_EFF_OUTCOME, "official": ARC3_OFFICIAL_OUTCOME,
                "rank": ARC3_EFF_OUTCOME, "hinge": ARC3_EFF_OUTCOME, "note": ARC3_NOTE_OUTCOME}
#: note: the judge results the orchestrator writes into the group directory (output of arc3_note_judge.judge_group)
JUDGE_FILE = "judge.json"
#: run directory name in group mode (the orchestrator's gNNNNNN): hinge uses its gid to tell whether the level's threshold
#: is frozen yet
GROUP_RUN = re.compile(r"^g(?P<gid>[0-9]{6})$")
#: join mode: classic = one sample run per manifest (group = task across runs); group = one group per manifest (streaming)
CLASSIC, GROUP = "classic", "group"
#: ``--wrapup-loss`` (orchestrator prepare.wrapup_loss): train = wrap-up generations are trained as usual; mask = not
#: trained (the user, 10-31, second-generation RL)
WRAPUP_LOSS_MODES = ("train", "mask")
#: group-mode item id: ``<tid>__s<k>``
GROUP_ITEM = re.compile(r"^(?P<tid>.+)__s(?P<k>[0-9]+)$")
#: ``--error-json`` error kind → dedicated exit code (copied in orchestrator.PREPARE_EXIT_CODES, test_orchestrator checks
#: the two match): group_data = the data of some manifests (groups) is incomplete (a ``ManifestError`` that points at
#: manifests); the caller quarantines those groups and redoes the rest; routing_over_max_gb = this packet's routing
#: exceeds ``--max-gb`` (operating cap); the caller splits the packet smaller and redoes it
PREPARE_EXIT_CODES = {"group_data": 65, "routing_over_max_gb": 66}


class RoutingOverMaxGB(RuntimeError):
    """routing output exceeds ``--max-gb``: the data is fine, this packet is too big (``cli``'s routing_over_max_gb)."""

    def __init__(self, routing_bytes: int, max_gb: float):
        super().__init__(f"routing {routing_bytes / 1e9:.1f} GB exceeds --max-gb {max_gb}")
        self.routing_bytes, self.max_gb = routing_bytes, max_gb


# ----------------------------------------------------------------------------- per segment


def _no_init() -> None:
    pass


def _routing_task(task) -> tuple[np.ndarray, dict[str, int]]:
    """Worker (spawn): ``segment_routing_status`` of one segment, returns (status, reasons). When ``out_path`` is not
    None, pwrites the routing (``(n, layers, top_k)`` int16) into the preallocated ``routing.i16`` starting at row
    ``row_offset``: segment ranges do not overlap, so the bytes written equal those of writing in order. The validity
    checks of a whole segment's routing (np.sort for duplicate ids, etc.) are the bulk of prepare's work, hence the
    workers."""
    from arcforge.tito.ledger import segment_routing_status

    directory, seg, layers, top_k, num_experts, out_path, row_offset = task
    st = segment_routing_status(Path(directory), seg, layers=layers, top_k=top_k, num_experts=num_experts)
    if out_path is not None:
        data = memoryview(np.ascontiguousarray(st.routing, dtype="<i2")).cast("B")
        fd = os.open(out_path, os.O_WRONLY)
        try:
            pos, done = row_offset * layers * top_k * 2, 0
            while done < len(data):
                n = os.pwrite(fd, data[done:], pos + done)
                if n <= 0:
                    raise OSError(f"{out_path}: pwrite wrote {n} bytes ({seg.key})")
                done += n
        finally:
            os.close(fd)
    return st.status, st.reasons


def behavior_logprobs(seg, replay: bool = False) -> np.ndarray:
    """``(len(seg),)`` float32: gen-span position p = the logprob recorded by the append that sampled ``seg.ids[p]``,
    NaN elsewhere. ``replay`` (truncated sampling, SPEC §1 ``replay: mask``): takes the append's
    ``sampling_mask.logprobs`` (log mu renormalized over the support), not ``logprobs`` (full vocabulary, before
    truncation).

    Alignment assertions (raise on violation): span start = that append's gen start, the ids on the span = that append's
    sampled gen (id by id), logprobs recorded and long enough, finite, ≤ 0. The result is then compared with
    ``Segment.gen_logprobs()`` / ``gen_sampling_mask()`` (an independent assembly)."""
    n = len(seg)
    ids = seg.ids
    out = np.full(n, np.nan, np.float32)
    by_k = {a.k: a for a in seg.appends}
    for span in seg.gen_spans:
        a = by_k[span.k]
        m = span.end - span.start
        if span.start != a.gen_start or not 0 < m <= a.gen_len:
            raise RuntimeError(f"{seg.key}: gen span [{span.start}, {span.end}) does not match append {a.k}'s gen "
                               f"[{a.gen_start}, {a.gen_start + a.gen_len})")
        if a.logprobs is None:
            raise RuntimeError(f"{seg.key}: append {a.k} (rid {a.rid}) generated tokens without logprobs"
                               f" (logprobs_error={a.logprobs_error!r})")
        if replay and a.sampling_mask is None:
            raise RuntimeError(f"{seg.key}: append {a.k} (rid {a.rid}) generated tokens without a sampling support"
                               f" (sampling_mask_error={a.sampling_mask_error!r})")
        if not np.array_equal(ids[span.start:span.end], a.gen[:m]):
            bad = int(np.flatnonzero(ids[span.start:span.end] != a.gen[:m])[0])
            raise RuntimeError(f"{seg.key}: position {span.start + bad} has token {int(ids[span.start + bad])}, not append "
                               f"{a.k}'s sampled id #{bad} {int(a.gen[bad])} (behavior logprob would be misaligned)")
        lp = (a.sampling_mask.logprobs if replay else a.logprobs)[:m]
        if not np.isfinite(lp).all() or (lp > LOGPROB_MAX).any():
            raise RuntimeError(f"{seg.key}: append {a.k} logprobs have non-finite values or > 0")
        out[span.start:span.end] = lp
    mask = ptm.loss_mask_from_spans(n, [(s.start, s.end) for s in seg.gen_spans]).astype(bool)
    if replay:
        sup = seg.gen_sampling_mask()
        other = None if sup is None else sup.logprobs
    else:
        other = seg.gen_logprobs()
    if other is None or not np.array_equal(out[mask], other) or np.isfinite(out[~mask]).any():
        raise RuntimeError(f"{seg.key}: behavior logprob disagrees with Segment.gen_logprobs() / gen_sampling_mask()")
    return out


def support_arrays(seg) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-position support of truncated sampling (``replay: mask``): ``(lens, ids, full)``. ``lens`` (len(seg),) uint16
    = the support size when loss position p sampled ``seg.ids[p]`` (≥ 1), 0 elsewhere; ``ids`` int32 = the supports of the
    loss positions concatenated in position order (no duplicates within a row, contains ``seg.ids[p]``, checked by the
    ledger reader); ``full`` (len(seg),) float32 = the full-vocabulary logprob at loss positions (ledger ``logprobs``,
    before truncation; diagnostics only), NaN elsewhere. Alignment as in ``behavior_logprobs``; the result is then
    compared with ``Segment.gen_sampling_mask()``."""
    n = len(seg)
    lens = np.zeros(n, np.int64)
    full = np.full(n, np.nan, np.float32)
    by_k = {a.k: a for a in seg.appends}
    parts = []
    for span in seg.gen_spans:
        a = by_k[span.k]
        m = span.end - span.start
        if a.sampling_mask is None or a.logprobs is None:
            raise RuntimeError(f"{seg.key}: append {a.k} has no sampling support or logprobs"
                               f" ({a.sampling_mask_error!r} / {a.logprobs_error!r})")
        sup = a.sampling_mask.head(m)
        if not np.array_equal(seg.ids[span.start:span.end], a.gen[:m]):
            raise RuntimeError(f"{seg.key}: ids of span [{span.start}, {span.end}) are not append {a.k}'s sampled gen")
        lens[span.start:span.end] = sup.lens
        parts.append(sup.ids)
        full[span.start:span.end] = a.logprobs[:m]
    if lens.max(initial=0) > np.iinfo(np.uint16).max:
        raise RuntimeError(f"{seg.key}: support size {int(lens.max())} exceeds uint16")
    ids = np.concatenate(parts).astype(np.int32) if parts else np.empty(0, np.int32)
    other = seg.gen_sampling_mask()
    mask = lens > 0
    if other is None or not np.array_equal(lens[mask], other.lens) or not np.array_equal(ids, other.ids):
        raise RuntimeError(f"{seg.key}: sampling support disagrees with Segment.gen_sampling_mask()")
    if not np.array_equal(mask, ptm.loss_mask_from_spans(n, [(s.start, s.end) for s in seg.gen_spans]).astype(bool)):
        raise RuntimeError(f"{seg.key}: positions with a support are not the loss positions")
    return lens.astype(np.uint16), ids, full


def sampling_replay(top_p: float, top_k: int | None) -> bool:
    """Truncated sampling (top-p < 1 or top-k on): the trainer must replay the ledger's support (SPEC §1
    ``replay: mask``)."""
    return top_p < 1 or top_k is not None


def segment_problems(seg, temperature: float, top_p: float = 1.0, top_k: int | None = None) -> list[str]:
    """Per-row conditions of SPEC §2 (sampling parameters, logprobs recorded; with truncated sampling the support must be
    recorded too). One reason per append with a non-empty generation."""
    from arcforge.tito.export import rl_sampling_issue

    replay = sampling_replay(top_p, top_k)
    out = []
    for a in seg.appends:
        if not a.gen_len:
            continue
        if (issue := rl_sampling_issue(a.sampling, temperature, top_p, top_k)) is not None:
            out.append(f"{seg.key} append {a.k}: {issue}")
        if a.logprobs is None:
            out.append(f"{seg.key} append {a.k}: {a.gen_len} generated tokens without logprobs ({a.logprobs_error!r})")
    return out


def support_missing(seg, top_p: float = 1.0, top_k: int | None = None) -> list[str]:
    """Appends with a non-empty generation but no sampling support under truncated sampling (the fork could not record it:
    sampling_mask_error gives the reason). This is a data problem of one episode, not a configuration error: that episode
    is dropped as ``no_support`` and the rest of its group trains as usual (the user's decision, 10-24), with one WARN line
    on stderr. When the support exceeds --sampling-mask-max-tokens (OVERFLOW) or a token outside the support is drawn
    (INVALID), the fork aborts the request outright and the ledger records an attempt, not an append, so this is never
    reached (10-24 arc2 speed A/B: 0 out of 2,460 responses). When the server did not enable
    --session-ledger-sampling-mask at all, every append lacks it: every episode is dropped, the groups are empty, and
    prepare follows the "no group left" rule."""
    if not sampling_replay(top_p, top_k):
        return []
    return [f"{seg.key} append {a.k}: {a.gen_len} generated tokens without a sampling support ({a.sampling_mask_error!r})"
            for a in seg.appends if a.gen_len and a.sampling_mask is None]


def mask_wrapup_spans(seg, post_rids: frozenset) -> tuple[int, int]:
    """``--wrapup-loss mask``: removes the wrap-up generations (the append's rid is in the outcome's
    ``wrapup_post_rids``: requests sent after the wrap-up line; the turn cut across the line does not count) from
    ``seg.gen_spans`` -- they do not enter the loss, the tokens stay in the sequence as context. Loss mask, behavior
    logprob, support, the coef denominator and ``gen_tokens`` are all derived from ``gen_spans``, so they change together.
    The wrap-up is always the end of the segment: raises if a removed span is followed by a kept span. Modifies in place,
    returns (tokens removed, appends removed)."""
    by_k = {a.k: a for a in seg.appends}
    post = [by_k[s.k].rid in post_rids for s in seg.gen_spans]
    if not any(post):
        return 0, 0
    first = post.index(True)
    if not all(post[first:]):
        keep = [by_k[s.k].rid for s, p in zip(seg.gen_spans[first:], post[first:]) if not p]
        raise RuntimeError(f"{seg.key}: the wrap-up generation (rid {by_k[seg.gen_spans[first].k].rid}) is followed by non-wrap-up generations"
                           f" (rid {keep[:3]}): the wrap-up is not the end of this segment")
    gone = seg.gen_spans[first:]
    del seg.gen_spans[first:]
    return sum(s.end - s.start for s in gone), len({s.k for s in gone})


def post_close_tail(seg, same_sid: list, thread_last: int) -> bool:
    """``seg`` is the request that finished only after its session was closed (v6b 10-27 b13: 9 games, all clearing
    ones): the harness calls close_session when the episode ends while the agent's last request is still running; at
    intake it was an extension turn of this session (resume exact, resume point = previous segment's S−1, using the
    previous segment's KV), it finished after the close, and the fork recorded it as a new segment (cause evicted, one
    append, prefix_len 0, tail = the whole old S plus new tokens) with routing starting at the intake resume point -- all
    earlier positions are M (late_branch) and the whole game would be dropped. This turn was generated after the episode
    ended and does not affect the reward (tokens_at_clear / actions are in the manifest). The caller drops only this
    segment and keeps the game.
    Counts only when all five conditions hold (otherwise it is M as before):
      1. cause ``evicted``, exactly one append, prefix_len 0;
      2. the previous segment of the same sid in the same instance ended with ``close_session``;
      3. this one was taken in before that close (t0 < the close's t), resume exact, pos = previous segment length − 1 =
         routing.start;
      4. its first len(previous segment) tokens equal the previous segment's, token for token;
      5. it is this thread's last segment in this instance.
    ``same_sid``: the segments of the same sid in the same instance (increasing seg number, including ``seg`` itself);
    ``thread_last``: the largest seg number of the thread."""
    if seg.cause != "evicted" or len(seg.appends) != 1 or seg.appends[0].prefix_len != 0:
        return False
    i = next(i for i, x in enumerate(same_sid) if x is seg)
    if i == 0:
        return False
    prev, a = same_sid[i - 1], seg.appends[0]
    n = len(prev)
    if prev.close is None or prev.close["reason"] != "close_session" or not a.t0 < prev.close["t"]:
        return False
    if a.resume is None or a.resume.mode != "exact" or a.resume.pos != n - 1:
        return False
    if a.routing is None or a.routing.start != n - 1 or a.tail_len < n:
        return False
    return bool(np.array_equal(seg.ids[:n], prev.ids)) and seg.seg == thread_last


def parse_versions(arg: str) -> list[str]:
    """``--weight-version`` → the allowed set, sorted by version number. One value = v1 behavior (including
    ``unversioned``); comma-separated = async's {k−1, k} (SPEC §8). Version numbers must be non-negative decimal integers
    (written per token into int32), without duplicates."""
    vals = [v.strip() for v in arg.split(",")]
    if any(not v for v in vals) or len(set(vals)) != len(vals):
        raise ValueError(f"--weight-version {arg!r}: empty or duplicate values")
    if UNVERSIONED in vals:
        if len(vals) != 1:
            raise ValueError(f"--weight-version {arg!r}: {UNVERSIONED} cannot be mixed with version numbers")
        return vals
    for v in vals:
        if not is_version(v):
            raise ValueError(f"--weight-version {arg!r}: {v!r} is not a non-negative integer version number")
    return sorted(vals, key=int)


def is_version(v: str) -> bool:
    """Non-negative decimal integer, no leading 0, fits in int32 (written per token into ``weight_version.i32``)."""
    return v.isascii() and v.isdigit() and str(int(v)) == v and int(v) <= np.iinfo(np.int32).max


def window_versions(current: int, max_staleness: int) -> list[str]:
    """The streaming allowed window [V−S, V] (from 0 when V−S < 0), as strings sorted by version number."""
    if not 0 <= current <= np.iinfo(np.int32).max or max_staleness < 0:
        raise ValueError(f"--current-version {current} must be a non-negative int32, --max-staleness {max_staleness} must be ≥ 0")
    return [str(v) for v in range(max(0, current - max_staleness), current + 1)]


def check_window(seen: collections.Counter, current: int) -> None:
    """Conditions on ``seen`` (the sum of ``segment_versions``) under streaming (``--current-version``): every append has
    a version key, versions are version numbers, none newer than V (the server cannot run ahead of the trainer). Versions
    older than the window are not judged here: those episodes are dropped as ``stale``. ``UNRECORDED`` as in
    ``check_versions``: dropped and counted."""
    if seen[ABSENT]:
        raise RuntimeError(f"the ledger has {seen[ABSENT]} generated tokens from appends without a version key (--current-version needs a ledger with versions)")
    bad = sorted(v for v in seen if v != UNRECORDED and not is_version(v))
    if bad:
        raise RuntimeError(f"ledger weight versions {bad} are not non-negative integer version numbers (server not started with --weight-version?)")
    ahead = {v: n for v, n in seen.items() if v != UNRECORDED and int(v) > current}
    if ahead:
        raise RuntimeError(f"weight versions {ahead} are newer than --current-version {current}: the server cannot run ahead of the trainer")


def oldest_version(versions: collections.Counter) -> int:
    """Oldest version among the episode's generated tokens (sum of ``segment_versions``; unrecorded ones do not count)."""
    return min(int(v) for v, n in versions.items() if n and v not in (ABSENT, UNRECORDED))


#: manifest.rl staleness histograms (all null with --weight-version)
STALENESS_KEYS = ("staleness_loss_tokens", "staleness_episodes", "staleness_episodes_stale", "staleness_stale_tokens")


def staleness_histograms(kept: list, stale: list, current: int) -> dict[str, dict[str, int]]:
    """Staleness = V − version → count (string keys, sorted by staleness): kept loss tokens, kept episodes (by their
    oldest loss token), episodes dropped as ``stale`` (by their oldest generated token), the generated tokens of episodes
    dropped as ``stale`` (each by its own staleness; tokens with unrecorded versions are keyed by ABSENT / UNRECORDED
    itself, sorted after the numbers)."""
    tokens, episodes, dropped, dropped_tokens = (collections.Counter() for _ in range(4))
    for ep in kept:
        for v, n in ep.version_tokens.items():
            tokens[current - int(v)] += n
        episodes[current - min(int(v) for v in ep.version_tokens)] += 1
    for ep in stale:
        dropped[current - oldest_version(ep.gen_versions)] += 1
        for v, n in ep.gen_versions.items():
            if n:
                dropped_tokens[v if v in (ABSENT, UNRECORDED) else current - int(v)] += n
    def ordered(c: collections.Counter) -> dict[str, int]:
        lags = sorted(k for k in c if isinstance(k, int))
        return {str(k): c[k] for k in lags + sorted(k for k in c if isinstance(k, str))}
    return {key: ordered(c) for key, c in zip(STALENESS_KEYS, (tokens, episodes, dropped, dropped_tokens))}


def segment_versions(seg) -> collections.Counter:
    """Weight version of appends with a non-empty generation → generated token count (including tails rewound later). An
    append with no version key at all (before the fork fix) counts under ``ABSENT``; ``weight_versions`` null (the fork
    failed to record it, with ``weight_versions_error``) counts under ``UNRECORDED``."""
    out = collections.Counter()
    for a in seg.appends:
        if not a.gen_len:
            continue
        if a.weight_versions is None:
            out[ABSENT if a.weight_versions_error is None else UNRECORDED] += a.gen_len
            continue
        ends = [s for s, _ in a.weight_versions[1:]] + [a.gen_len]
        for (start, v), end in zip(a.weight_versions, ends):
            out[v] += end - start
    return out


def check_versions(seen: collections.Counter, allowed: list[str]) -> None:
    """``seen`` (sum of ``segment_versions``) against the allowed set (SPEC §2 / §8). ``unversioned``: the ledger recorded
    no version at all. ``UNRECORDED`` is not judged here: those episodes are dropped and counted (the caller owns the
    gate)."""
    if allowed == [UNVERSIONED]:
        if set(seen) - {ABSENT}:
            raise RuntimeError(f"--weight-version unversioned, but the ledger recorded versions {dict(seen)}: use the real version numbers")
        return
    if seen[ABSENT]:
        raise RuntimeError(f"the ledger has {seen[ABSENT]} generated tokens from appends without a version key"
                           f" (ledgers from before the fork fix need --weight-version unversioned)")
    bad = {v: n for v, n in seen.items() if v not in allowed and v != UNRECORDED}
    if bad:
        raise RuntimeError(f"weight versions {bad} not in the --weight-version allowed set {allowed} (the async gate should make this impossible)")


def token_versions(seg, allowed: list[str]) -> np.ndarray:
    """``(len(seg),)`` int32: gen-span position p = the weight version when ``seg.ids[p]`` was sampled (from the append's
    ``weight_versions`` ranges), −1 elsewhere. ``unversioned``: −1 throughout. Values must be in the allowed set."""
    n = len(seg)
    out = np.full(n, -1, np.int32)
    if allowed == [UNVERSIONED]:
        return out
    by_k = {a.k: a for a in seg.appends}
    for span in seg.gen_spans:
        a = by_k[span.k]
        if a.weight_versions is None:
            raise RuntimeError(f"{seg.key}: append {a.k} has no weight version ({a.weight_versions_error!r}); this episode should already be dropped")
        ends = [s for s, _ in a.weight_versions[1:]] + [a.gen_len]
        for (start, v), end in zip(a.weight_versions, ends):
            lo, hi = max(a.gen_start + start, span.start), min(a.gen_start + end, span.end)
            if lo < hi:
                if v not in allowed:
                    raise RuntimeError(f"{seg.key}: version {v!r} not in the allowed set {allowed}")
                out[lo:hi] = int(v)
    mask = ptm.loss_mask_from_spans(n, [(s.start, s.end) for s in seg.gen_spans]).astype(bool)
    if not np.array_equal(out != -1, mask):
        raise RuntimeError(f"{seg.key}: per-token versions do not exactly cover the gen spans")
    return out


# ----------------------------------------------------------------------------- groups, minibatches, bins


def manifest_groups(manifests) -> tuple[str, dict[tuple[str, str], tuple[str, str]]]:
    """Join mode (``CLASSIC`` / ``GROUP``) and the ``(group, task)`` of every ``(run, item id)``.

    classic: group = task = item id (one task across runs). group mode: one manifest = one group, every item id is
    ``<tid>__s<k>``, one tid, each k = 0…n−1 once (no leading 0), group = ``<run dir name>:<tid>``, task = tid; when the
    run directory has ``results/<tid>__s<k>.json`` (``task`` = item id, join reconciles score) its ``tid`` must be this
    task. A manifest's items are either all ``<tid>__s<k>`` or none (no items counts as classic), and the manifests of one
    call must be of one kind; raise otherwise."""
    modes: dict[str, list[str]] = {CLASSIC: [], GROUP: []}
    out: dict[tuple[str, str], tuple[str, str]] = {}
    for m in manifests:
        parsed = {iid: GROUP_ITEM.match(iid) for iid in m.items}
        group_form = sorted(iid for iid, p in parsed.items() if p)
        if not group_form:
            modes[CLASSIC].append(f"{m.path} ({len(parsed)} items)")
            out.update({(m.run, iid): (iid, iid) for iid in m.items})
            continue
        if len(group_form) != len(parsed):
            raise RuntimeError(f"{m.path}: items mix group-mode <tid>__s<k> ({group_form[:3]}) and classic "
                               f"{sorted(iid for iid, p in parsed.items() if not p)[:3]}")
        modes[GROUP].append(str(m.path))
        tids = sorted({p["tid"] for p in parsed.values()})
        if len(tids) != 1:
            raise RuntimeError(f"{m.path}: a group-mode manifest must be one task, it has {tids}")
        ks = sorted(p["k"] for p in parsed.values())
        if any(str(int(k)) != k for k in ks) or sorted(int(k) for k in ks) != list(range(len(ks))):
            raise RuntimeError(f"{m.path}: group-mode sample numbers must be 0…{len(ks) - 1} once each (no leading 0), got {ks}")
        for iid in m.items:
            results = m.run_dir / "results" / f"{iid}.json"
            if results.is_file():                 # same rule as join.Manifest.outcome (its task field = the item id)
                r = json.loads(results.read_text())
                if "tid" not in r or r["tid"] != tids[0]:
                    raise RuntimeError(f"{results}: tid={r.get('tid')!r}, group-mode item {iid} is task {tids[0]}")
        group = f"{m.run}:{tids[0]}"
        out.update({(m.run, iid): (group, tids[0]) for iid in m.items})
    if modes[CLASSIC] and modes[GROUP]:
        raise RuntimeError(f"one call mixes classic manifests {modes[CLASSIC][:2]} and group-mode ones {modes[GROUP][:2]}")
    return (GROUP if modes[GROUP] else CLASSIC), out


ADV_NORMALIZATIONS = ("std", "none")
LOSS_REDUCTIONS = ("sequence", "prompt")


def group_advantages(rewards, normalization: str = "std") -> np.ndarray | None:
    """``std``: ``(R − mean) / std`` (population std); ``none``: ``R − mean``. All rewards equal → None (the whole group
    is dropped)."""
    if normalization not in ADV_NORMALIZATIONS:
        raise ValueError(f"advantage normalization {normalization!r} not in {ADV_NORMALIZATIONS}")
    r = np.asarray(rewards, dtype=np.float64)
    if len(r) < 2:
        raise ValueError("a group needs at least 2 episodes")
    if not np.isfinite(r).all():
        raise ValueError(f"non-finite rewards: {r.tolist()}")
    if np.ptp(r) == 0:
        return None
    centered = r - r.mean()
    return centered / r.std() if normalization == "std" else centered


def efficiency_advantages(rewards, output_tokens, *, beta: float, budget: int, guard: float) -> tuple[np.ndarray, dict]:
    """Efficiency term: only among episodes with reward 1 (all correct) compares ``e_i = 1 − L_i / budget`` (L = the
    ledger's generated tokens, negative beyond budget), ``A_eff_i = β_g · (e_i − mean_allcorrect e)``, summing to 0 over
    the all-correct episodes; all 0 with fewer than 2 all-correct.
    β_g: all-correct group (every reward 1, task term all 0) = β; mixed group = min(β, guard · (1 − mean_g R) /
    max_allcorrect(mean e − e_i)): the longest all-correct episode loses at most guard times its task advantage
    (1 − mean_g R), and never flips sign.
    Returns (A_eff, {"correct": number of all-correct, "beta": β_g, "bound": whether the mixed-group bound bit})."""
    if not beta > 0 or not budget > 0 or not 0 < guard < 1:
        raise ValueError(f"efficiency: beta={beta} must be > 0, budget={budget} must be > 0, guard={guard} must be in (0, 1)")
    r = np.asarray(rewards, dtype=np.float64)
    n = np.asarray(output_tokens, dtype=np.float64)
    if r.shape != n.shape or r.ndim != 1:
        raise ValueError(f"rewards {r.shape} and output_tokens {n.shape} must be 1-D of equal length")
    if not np.isfinite(r).all() or (r < 0).any() or (r > 1).any():
        raise ValueError(f"rewards must be in [0, 1]: {r.tolist()}")
    if not np.isfinite(n).all() or (n <= 0).any():
        raise ValueError(f"output_tokens must be positive: {n.tolist()}")
    out = np.zeros_like(r)
    correct = r == 1.0
    if correct.sum() < 2:
        return out, {"correct": int(correct.sum()), "beta": 0.0, "bound": False}
    e = 1.0 - n[correct] / budget
    d = e - e.mean()
    beta_g, bound = beta, False
    if not correct.all():
        worst = float(-d.min())           # the longest correct episode, below the mean by this much
        if worst > 0 and guard * (1.0 - r.mean()) / worst < beta:
            beta_g, bound = guard * (1.0 - r.mean()) / worst, True
    out[correct] = beta_g * d
    return out, {"correct": int(correct.sum()), "beta": float(beta_g), "bound": bound}


def check_arc3_eff_params(*, beta_actions: float, beta_tokens: float, w_start: float, clip: float) -> None:
    """The four parameters of ``--arc3-reward eff``: β_a ≥ 0, β_t ≥ 0, 0 ≤ w_start < 1, clip > 0, all finite. Raise on
    violation."""
    vals = {"beta_actions": beta_actions, "beta_tokens": beta_tokens, "w_start": w_start, "clip": clip}
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in vals.values()):
        raise ValueError(f"arc3 eff: parameters must be finite numbers: {vals}")
    if not (beta_actions >= 0 and beta_tokens >= 0 and 0 <= w_start < 1 and clip > 0):
        raise ValueError(f"arc3 eff: beta_actions={beta_actions}, beta_tokens={beta_tokens} must be ≥ 0, w_start={w_start} must be in "
                         f"[0, 1), clip={clip} must be > 0")


def _plain_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def check_arc3_episodes(scores, correct, actions, tokens_at_clear, mode: str) -> int:
    """One arc3 eff / c0 group: four columns of equal length with ≥ 2 episodes; score is 0 / 1, correct is a bool equal
    to score == 1; actions are integers (≥ 1 if cleared, ≥ 0 if not); tokens_at_clear is an integer > 0 if cleared,
    None if not. Raise on violation; returns the count."""
    n = len(scores)
    if n < 2 or not len(correct) == len(actions) == len(tokens_at_clear) == n:
        raise ValueError(f"arc3 {mode}: scores / correct / actions / tokens_at_clear must be of equal length, at least 2 ({len(scores)}, {len(correct)}, "
                         f"{len(actions)}, {len(tokens_at_clear)})")
    for i, (s, c, a, t) in enumerate(zip(scores, correct, actions, tokens_at_clear)):
        if s not in (0.0, 1.0) or not isinstance(c, bool) or c != (s == 1.0):
            raise ValueError(f"arc3 {mode}: episode {i} has score={s!r}, correct={c!r}: score must be 0 / 1 and correct a bool equal to "
                             "score == 1")
        if not _plain_int(a) or a < (1 if c else 0):
            raise ValueError(f"arc3 {mode}: episode {i} actions={a!r} must be an integer (≥ 1 if cleared, ≥ 0 if not)")
        if c != (t is not None) or (c and (not _plain_int(t) or t <= 0)):
            raise ValueError(f"arc3 {mode}: episode {i} (correct={c}) tokens_at_clear={t!r}: must be an integer > 0 if cleared, None if not")
    return n


def arc3_eff_rewards(scores, correct, actions, tokens_at_clear, *, beta_actions: float, beta_tokens: float,
                     w_start: float, clip: float) -> tuple[np.ndarray, dict]:
    """``--arc3-reward eff``: one group's kept episodes → (R, info). The caller takes the advantage as ``R − mean R``
    (adv none).

    - c_i = cleared (outcome ``correct``: bool, must equal ``score == 1``), p̂ = mean c, C = the clearing episodes.
    - |C| ≥ 2: x_i = ln a_i − mean_C ln a (a = the actions counted for this level, ≥ 1), y_i = ln t_i − mean_C ln t
      (t = ``tokens_at_clear``, the output tokens the proxy counted at the moment of clearing, > 0);
      E_i = −(2 β_a x_i + β_t y_i), clipped to [−clip, clip], then minus mean_C E (re-centred).
      E = 0 with |C| < 2 and for non-clearing episodes. Actions times 2: the official per-level score ∝
      (baseline / actions)², and within a group the baseline cancels. Tokens of non-clearing episodes are not penalized
      (``tokens_at_clear`` is None); otherwise giving up early would be rewarded.
    - w = 0 (p̂ ≤ w_start), else (p̂ − w_start) / (1 − w_start).
    - R_i = c_i · (1 + w E_i); E averages 0 over C, so A_i = R_i − mean R = (c_i − p̂) + w c_i E_i: the efficiency term
      only redistributes among clearing episodes and does not touch "cleared vs not".
    - Asserted: R > 0 when cleared, R = 0 when not. After re-centring |E| < 2·clip, so this always holds for clip ≤ 0.5;
      the assertion guards larger clips.

    info: ``p_hat``, ``w``, ``cleared`` (|C|), ``clip_hits`` (clearing episodes that were clipped), ``task`` = c − p̂,
    ``eff`` = w c E (the two per-episode components)."""
    check_arc3_eff_params(beta_actions=beta_actions, beta_tokens=beta_tokens, w_start=w_start, clip=clip)
    n = check_arc3_episodes(scores, correct, actions, tokens_at_clear, "eff")
    cleared = np.array(correct, dtype=bool)
    p_hat = float(cleared.mean())
    e = np.zeros(n)
    clip_hits = 0
    if cleared.sum() >= 2:
        la = np.log(np.array([a for a, c in zip(actions, correct) if c], dtype=np.float64))
        lt = np.log(np.array([t for t, c in zip(tokens_at_clear, correct) if c], dtype=np.float64))
        raw = -(2.0 * beta_actions * (la - la.mean()) + beta_tokens * (lt - lt.mean()))
        clip_hits = int((np.abs(raw) > clip).sum())
        clipped = np.clip(raw, -clip, clip)
        e[cleared] = clipped - clipped.mean()
    w = 0.0 if p_hat <= w_start else (p_hat - w_start) / (1.0 - w_start)
    r = cleared * (1.0 + w * e)
    if not ((r[cleared] > 0).all() and (r[~cleared] == 0).all()):
            raise RuntimeError(f"arc3 eff: R={r.tolist()} (clip={clip}, w={w}): R must be > 0 when cleared and 0 when not -- clip too large?")
    return r, {"p_hat": p_hat, "w": w, "cleared": int(cleared.sum()), "clip_hits": clip_hits,
               "task": cleared - p_hat, "eff": w * cleared * e}


def check_arc3_c0_params(*, lambda_actions: float, kappa_actions: float, lambda_tokens: float, kappa_tokens: float,
                         w_start: float, success_floor: float) -> None:
    """The six parameters of ``--arc3-reward c0``: λ_a, λ_t ≥ 0, κ_a, κ_t > 0, 0 ≤ w_start < 1, 0 < success_floor < 1,
    all finite. Raise on violation."""
    vals = {"lambda_actions": lambda_actions, "kappa_actions": kappa_actions, "lambda_tokens": lambda_tokens,
            "kappa_tokens": kappa_tokens, "w_start": w_start, "success_floor": success_floor}
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) for v in vals.values()):
        raise ValueError(f"arc3 c0: parameters must be finite numbers: {vals}")
    if not (lambda_actions >= 0 and lambda_tokens >= 0 and kappa_actions > 0 and kappa_tokens > 0 and 0 <= w_start < 1
            and 0 < success_floor < 1):
        raise ValueError(f"arc3 c0: lambda must be ≥ 0, kappa > 0, w_start in [0, 1), success_floor in (0, 1): {vals}")


def success_quality(cost, kappa: float) -> np.ndarray:
    """Costs of clearing episodes → ``q = σ(κ (mean ln cost − ln cost))``: 0.5 at the geometric mean, closer to 1 the
    cheaper (the tanh form does not overflow)."""
    lc = np.log(np.asarray(cost, dtype=np.float64))
    return 0.5 * (1.0 + np.tanh(0.5 * kappa * (lc.mean() - lc)))


def arc3_c0_rewards(scores, correct, actions, tokens_at_clear, *, lambda_actions: float, kappa_actions: float,
                    lambda_tokens: float, kappa_tokens: float, w_start: float, success_floor: float
                    ) -> tuple[np.ndarray, dict]:
    """``--arc3-reward c0``: one group's kept episodes → (R, info). The caller takes the advantage as ``R − mean R``
    (adv none).

    - c, p̂, C, w(p̂) as in eff (``arc3_eff_rewards``); the input checks are the same too (``check_arc3_episodes``).
    - |C| ≥ 2: ``q_a = σ(κ_a (mean_C ln a − ln a_i))``, ``q_t`` likewise with ``tokens_at_clear`` (``success_quality``);
      ``D_i = λ_a (q_a − mean_C q_a) + λ_t (q_t − mean_C q_t)``, summing to 0 over C. D = 0 with |C| < 2 and for
      non-clearing episodes. No clipping: the sigmoid saturates smoothly. Actions are not multiplied by 2 (the scale is
      set by κ_a).
    - Guard: ``h = w D``, ``s = min(1, (1 − success_floor) / max_C(−h))`` (s = 1 when no h is negative).
    - ``R_i = c_i (1 + s h_i)``: R ≥ success_floor when cleared, R = 0 when not. The mean R of the clearing episodes is
      1, so ``A_i = R_i − mean R = (c_i − p̂) + s w c_i D_i``: cleared vs not cleared always differ by 1 (unlike a
      positive bonus, which grows with efficiency), the efficiency term only redistributes among clearing episodes, and
      an inefficient clear can have a negative advantage.
    - Asserted: R ≥ success_floor when cleared, R = 0 when not.

    info: ``p_hat``, ``w``, ``cleared`` (|C|), ``guard_scale`` (s), ``task`` = c − p̂, ``eff`` = s w c D, and its two
    components ``eff_actions``, ``eff_tokens`` (per episode)."""
    check_arc3_c0_params(lambda_actions=lambda_actions, kappa_actions=kappa_actions, lambda_tokens=lambda_tokens,
                         kappa_tokens=kappa_tokens, w_start=w_start, success_floor=success_floor)
    n = check_arc3_episodes(scores, correct, actions, tokens_at_clear, "c0")
    cleared = np.array(correct, dtype=bool)
    p_hat = float(cleared.mean())
    da, dt = np.zeros(n), np.zeros(n)
    if cleared.sum() >= 2:
        qa = success_quality([a for a, c in zip(actions, correct) if c], kappa_actions)
        qt = success_quality([t for t, c in zip(tokens_at_clear, correct) if c], kappa_tokens)
        da[cleared] = lambda_actions * (qa - qa.mean())
        dt[cleared] = lambda_tokens * (qt - qt.mean())
    w = 0.0 if p_hat <= w_start else (p_hat - w_start) / (1.0 - w_start)
    h = w * (da + dt)
    down = max(0.0, -float(h[cleared].min())) if cleared.any() else 0.0
    s = min(1.0, (1.0 - success_floor) / down) if down > 0 else 1.0
    r = cleared * (1.0 + s * h)
    if not ((r[cleared] >= success_floor - 1e-12).all() and (r[~cleared] == 0).all()):
        raise RuntimeError(f"arc3 c0: R={r.tolist()} (s={s}, w={w}): R must be ≥ {success_floor} when cleared and 0 when not")
    return r, {"p_hat": p_hat, "w": w, "cleared": int(cleared.sum()), "guard_scale": s, "task": cleared - p_hat,
               "eff": s * h, "eff_actions": s * w * da, "eff_tokens": s * w * dt}


def arc3_official_rewards(scores, correct, actions, baseline_actions, *, top_score: float, knee_score: float,
                          knee_reward: float, zero_score: float) -> tuple[np.ndarray, dict]:
    """``--arc3-reward official``: one group's kept episodes → (R, info). The caller takes the advantage as
    ``R − mean R`` (adv none).

    - ``R_i = c_i f(S_i)``, ``S_i = min(100 (b_i / a_i)², 115)`` (the official per-level score,
      ``arcforge.arc_code.score.level_score``), f piecewise linear through (top_score, 1), (knee_score, knee_reward),
      (zero_score, 0), flat at both ends (``arcforge.arc3_official``).
      No token term, no within-group centring: a clear much slower than the human (S ≤ zero_score) is 0, the same as not
      clearing.
    - score / correct checks as in eff / c0; actions are integers (≥ 1 if cleared); baseline_actions is an integer ≥ 1,
      identical within a group (one level).
    - info: ``p_hat`` (clear rate), ``cleared``, ``full`` (count with R = 1), ``cleared_zero`` (cleared but R = 0),
      ``partial`` (0 < R < 1), ``task`` = c − p̂ and ``eff`` = (R − c) − mean(R − c): A = R − mean R = task + eff, eff
      being what the official score (actions) redistributes beyond clearing (numerator of gradient_share)."""
    from arcforge.arc3_official import check_params, episode_reward
    params = {"top_score": top_score, "knee_score": knee_score, "knee_reward": knee_reward, "zero_score": zero_score}
    check_params(**params)
    n = len(scores)
    if n < 2 or not len(correct) == len(actions) == len(baseline_actions) == n:
        raise ValueError(f"arc3 official: scores / correct / actions / baseline_actions must be of equal length, at least 2 ({len(scores)}, "
                         f"{len(correct)}, {len(actions)}, {len(baseline_actions)})")
    for i, (s, c) in enumerate(zip(scores, correct)):
        if s not in (0.0, 1.0) or not isinstance(c, bool) or c != (s == 1.0):
            raise ValueError(f"arc3 official: episode {i} has score={s!r}, correct={c!r}: score must be 0 / 1 and correct a bool equal to "
                             "score == 1")
    if len(set(baseline_actions)) != 1:
        raise ValueError(f"arc3 official: a group is one level, but baseline_actions differ: {sorted(set(baseline_actions))}")
    r = np.array([episode_reward(c, a, b, **params) for c, a, b in zip(correct, actions, baseline_actions)])
    cleared = np.array(correct, dtype=bool)
    p_hat = float(cleared.mean())
    extra = r - cleared
    return r, {"p_hat": p_hat, "cleared": int(cleared.sum()), "full": int((r == 1.0).sum()),
               "cleared_zero": int((cleared & (r == 0.0)).sum()), "partial": int(((r > 0) & (r < 1)).sum()),
               "task": cleared - p_hat, "eff": extra - extra.mean()}


def arc3_rank_rewards(scores, correct, actions, tokens_at_clear, *, rank_actions: float, rank_tokens: float,
                      rank_pass_scale: int, rank_warmup_versions: int, launch_version: int | None
                      ) -> tuple[np.ndarray, dict]:
    """``--arc3-reward rank``: one group's kept episodes → (R, info). The caller takes the advantage as ``R − mean R``
    (adv none).

    - ``R_i = s_i (1 − f (α r_act,i + β r_tok,i))`` (``arcforge.arc3_rank.group_rewards``; the orchestrator's group_live
      and wandb_sync use the same definition): s = the outcome's score (g2seed's 1[cleared] min(1, (κ b / a)²)), r = the
      mean rank of action count / ``tokens_at_clear`` among the kept clearing episodes, normalized to [0,1] (0 best);
      r = 0 with fewer than 2 clears. α = rank_actions, β = rank_tokens.
      f = 0 (this group's launch version ``launch_version`` < rank_warmup_versions), otherwise = the clear fraction ρ of
      the kept episodes with rank_pass_scale 1, = 1 with 0.
    - Ranks and ρ are computed only among the episodes kept here (dropped episodes take no rank and do not enter ρ).
    - info: ``p_hat`` (clear rate), ``cleared``, ``ranked`` (≥ 2 clears = ranks take effect), ``factor`` = f, ``warmup``
      (f is the warm-up 0), ``task`` = c − p̂ and ``eff`` = (R − c) − mean(R − c): A = R − mean R = task + eff, eff being
      what the κ score and ranks redistribute beyond clearing (numerator of gradient_share)."""
    from arcforge.arc3_rank import group_rewards
    if len(scores) < 2:
        raise ValueError(f"arc3 rank: a group needs at least 2 episodes ({len(scores)})")
    r, _, _, f = group_rewards(scores, correct, actions, tokens_at_clear, rank_actions=rank_actions, rank_tokens=rank_tokens,
                               rank_pass_scale=rank_pass_scale, rank_warmup_versions=rank_warmup_versions,
                               launch_version=launch_version)
    r = np.array(r, dtype=np.float64)
    cleared = np.array(correct, dtype=bool)
    p_hat = float(cleared.mean())
    extra = r - cleared
    return r, {"p_hat": p_hat, "cleared": int(cleared.sum()), "ranked": bool(cleared.sum() >= 2), "factor": f,
               "warmup": bool(rank_warmup_versions > 0 and launch_version < rank_warmup_versions),
               "task": cleared - p_hat, "eff": extra - extra.mean()}


def arc3_hinge_rewards(scores, correct, actions, tokens_at_clear, *, hinge_tokens: float, hinge_actions: float,
                       hinge_warmup_versions: int, hinge_refs: str, task: str, gid: int, launch_version: int,
                       refs: dict) -> tuple[np.ndarray, dict]:
    """``--arc3-reward hinge``: one group's kept episodes → (R, info) (``arcforge.arc3_hinge.group_rewards``; the
    orchestrator's group_live and wandb_sync use the same definition; ρ and the mean cost of the clearing episodes are
    computed only among the kept episodes). ``refs`` = the refs file content, read once at the start of this prepare.
    info as in rank: ``p_hat``, ``cleared``, ``factor`` = f, ``hinged`` (this group used frozen thresholds), ``task``,
    ``eff``."""
    from arcforge.arc3_hinge import group_rewards
    if len(scores) < 2:
        raise ValueError(f"arc3 hinge: a group needs at least 2 episodes ({len(scores)})")
    r, _, f, rec = group_rewards(scores, correct, actions, tokens_at_clear, hinge_tokens=hinge_tokens,
                                 hinge_actions=hinge_actions, hinge_warmup_versions=hinge_warmup_versions,
                                 hinge_refs=hinge_refs, task=task, gid=gid, launch_version=launch_version, refs=refs)
    r = np.array(r, dtype=np.float64)
    cleared = np.array(correct, dtype=bool)
    p_hat = float(cleared.mean())
    extra = r - cleared
    return r, {"p_hat": p_hat, "cleared": int(cleared.sum()), "factor": f, "hinged": rec is not None,
               "task": cleared - p_hat, "eff": extra - extra.mean()}


def arc3_note_rewards(stops, correct, actions, baseline_actions, window_causes, judged) -> tuple[np.ndarray, dict]:
    """``--arc3-reward note``: one group's kept episodes → (R, info) (``arcforge.arc3_note.episode_reward``; the
    orchestrator's group_live uses the same definition). ``judged``: each episode's reward from judge.json (only episodes
    with stop window and window_cause threshold have one, None otherwise). info: ``p_hat``, ``cleared``,
    ``task`` = c − p̂, ``eff`` = (R − c) − mean(R − c) (what the official score and the note redistribute beyond
    clearing), plus ``window`` / ``early`` (stop-window episodes, and those among them that changed window early)."""
    from arcforge.arc3_note import WINDOW_STOP, episode_reward
    n = len(stops)
    if n < 2 or not len(correct) == len(actions) == len(baseline_actions) == len(window_causes) == len(judged) == n:
        raise ValueError(f"arc3 note: stops / correct / actions / baseline_actions / window_causes / judged must be of equal length, at least 2 ({n})")
    if len(set(baseline_actions)) != 1:
        raise ValueError(f"arc3 note: a group is one level, but baseline_actions differ: {sorted(set(baseline_actions))}")
    r = np.array([episode_reward(*args) for args in zip(stops, correct, actions, baseline_actions, window_causes, judged)])
    cleared = np.array(correct, dtype=bool)
    p_hat = float(cleared.mean())
    extra = r - cleared
    window = [s == WINDOW_STOP for s in stops]
    return r, {"p_hat": p_hat, "cleared": int(cleared.sum()), "task": cleared - p_hat, "eff": extra - extra.mean(),
               "window": sum(window), "early": sum(w and c == "early" for w, c in zip(window, window_causes))}


#: the four sums of arc3_reward.gradient_share (kept groups): Σ|c − p̂|, Σ|w c E|, and their versions weighted by the
#: episode's loss-token count L_i
#: ARC3_*_RULE: rule texts recorded as rl.arc3_reward.rule, kept verbatim (the orchestrator requires the rule of every
#: packet to be identical when it merges arc3_reward summaries). In English:
#: eff: R_i = c_i (1 + w E_i), c = cleared; E_i = clip(-(2 beta_actions x_i + beta_tokens y_i), ±eff_clip) minus the mean
#:   over clearing episodes, x / y = ln actions / ln tokens_at_clear minus the clearing-episode mean (E = 0 with fewer than
#:   2 clears and when not cleared); w = 0 (p̂ ≤ w_start) else (p̂ - w_start)/(1 - w_start); A_i = R_i - mean R =
#:   (c_i - p̂) + w c_i E_i.
#: c0: R_i = c_i (1 + s w D_i), c = cleared; D_i = lambda_actions (q_a - mean q_a) + lambda_tokens (q_t - mean q_t),
#:   q = sigmoid(kappa (mean ln cost over clearing episodes - ln cost)) (cost = actions / tokens_at_clear; D = 0 with
#:   fewer than 2 clears and when not cleared); w as in eff; s = min(1, (1 - success_floor) / max_C(-w D));
#:   A_i = R_i - mean R = (c_i - p̂) + s w c_i D_i.
#: official: R_i = c_i f(S_i), c = cleared, S = min(100 (baseline_actions/actions)², 115) (official per-level score);
#:   f piecewise linear through (top_score, 1), (knee_score, knee_reward), (zero_score, 0), flat at both ends; no token
#:   term; A_i = R_i - mean R.
#: rank: R_i = s_i (1 - f (rank_actions r_act + rank_tokens r_tok)), f = 0 (launch version < rank_warmup_versions), else
#:   the clear fraction of the kept episodes with rank_pass_scale 1, 1 with 0; s = outcome score; r = the mean rank of
#:   actions / tokens_at_clear among the kept clearing episodes divided by (clears - 1) (0 best; r = 0 with fewer than 2
#:   clears and when not cleared); A_i = R_i - mean R.
#: hinge: R_i = s_i + f (mean_cleared D - D_i), 0 when not cleared; D_i = s_i (hinge_tokens h_t + hinge_actions h_a),
#:   h_t = clip((tokens_at_clear - T)/(2T), 0, 1), h_a = clip((actions - A)/A, 0, 1) (0 when A is null), T / A = the
#:   level's frozen thresholds in the hinge_refs file; f = the clear fraction of the kept episodes when this group's
#:   gid >= the level's applies_from_gid, else 0; A_i = R_i - mean R.
#: note: a clear gets R = clear_reward(actions, baseline): with r = actions/baseline ≤ 1, 0.9 + 0.1 (L − 100)/15,
#:   L = min(115, 100/r²); with r > 1, 0.9 exp(−5 (r − 1)). Stop window with window_cause threshold:
#:   R = min(TIERS[grade], clear_reward(actions + remaining_actions, baseline)), grade / remaining_actions read from the
#:   group directory's judge.json (recomputed, must equal the reward in the file); window_cause early (called new_context
#:   itself before the window change point) and other stops: R = 0 (arcforge.arc3_note); A_i = R_i - mean R.
ARC3_SHARE_SUMS = ("task_abs", "eff_abs", "task_abs_loss_tokens", "eff_abs_loss_tokens")
ARC3_EFF_RULE = ("R_i = c_i (1 + w E_i)，c＝過關；E_i = clip(-(2 beta_actions x_i + beta_tokens y_i), ±eff_clip) 再減過關局的平均，"
                 "x／y＝ln actions／ln tokens_at_clear 減過關局的平均（過關不到 2 條、沒過關的 E = 0）；"
                 "w = 0（p̂ ≤ w_start）否則 (p̂ - w_start)/(1 - w_start)；A_i = R_i - mean R = (c_i - p̂) + w c_i E_i")
ARC3_C0_RULE = ("R_i = c_i (1 + s w D_i)，c＝過關；D_i = lambda_actions (q_a - mean q_a) + lambda_tokens (q_t - mean q_t)，"
                "q = sigmoid(kappa (過關局的 mean ln 成本 - ln 成本))（成本＝actions／tokens_at_clear；過關不到 2 條、沒過關的 D = 0）；"
                "w 同 eff；s = min(1, (1 - success_floor) / max_C(-w D))；A_i = R_i - mean R = (c_i - p̂) + s w c_i D_i")
ARC3_OFFICIAL_RULE = ("R_i = c_i f(S_i)，c＝過關，S＝min(100 (baseline_actions/actions)², 115)（官方單關分）；f 過 (top_score, 1)、"
                      "(knee_score, knee_reward)、(zero_score, 0) 分段線性、兩端持平；沒有 token 項；A_i = R_i - mean R")
ARC3_RANK_RULE = ("R_i = s_i (1 - f (rank_actions r_act + rank_tokens r_tok))，f＝0（派出版本 < rank_warmup_versions）否則 "
                  "rank_pass_scale 1 時＝留下的局的過關比例、0 時＝1；s＝outcome score；r＝留下的過關局之間 actions／"
                  "tokens_at_clear 的平均名次除以 (過關數 - 1)（0 最好；過關不到 2 條、沒過關的 r = 0）；A_i = R_i - mean R")
ARC3_HINGE_RULE = ("R_i = s_i + f (mean_cleared D - D_i)、沒過關 0；D_i = s_i (hinge_tokens h_t + hinge_actions h_a)，"
                   "h_t = clip((tokens_at_clear - T)/(2T), 0, 1)、h_a = clip((actions - A)/A, 0, 1)（A 是 null 時 0），T／A＝"
                   "hinge_refs 檔該關的凍結門檻；f＝這組 gid >= 該關 applies_from_gid 時是留下的局的過關比例，否則 0；"
                   "A_i = R_i - mean R")
ARC3_NOTE_RULE = ("過關 R = clear_reward(actions, baseline)：r = actions/baseline ≤ 1 時 0.9 + 0.1 (L − 100)/15，"
                  "L = min(115, 100/r²)；r > 1 時 0.9 exp(−5 (r − 1))。stop window 且 window_cause threshold：R = min(TIERS[grade], "
                  "clear_reward(actions + remaining_actions, baseline))，grade／remaining_actions 讀 group 目錄的 judge.json（重算、"
                  "要等於檔裡的 reward）；window_cause early（換窗點之前自己叫 new_context）與其他 stop R = 0（arcforge.arc3_note）；"
                  "A_i = R_i - mean R")
ARC3_RULES = {"eff": ARC3_EFF_RULE, "c0": ARC3_C0_RULE, "official": ARC3_OFFICIAL_RULE, "rank": ARC3_RANK_RULE,
              "hinge": ARC3_HINGE_RULE, "note": ARC3_NOTE_RULE}
#: extra counts per mode: eff = clipped clearing episodes (only groups with w > 0); c0 = groups where the guard (s < 1)
#: bit; official = groups with R = 1 throughout (considered; the orchestrator's retire uses the same rule), episodes
#: cleared with R = 0 and with 0 < R < 1 (kept); rank = groups with ≥ 2 clears where ranks take effect, and groups whose
#: launch version is still in warm-up (efficiency term f = 0) (considered)
ARC3_MODE_STATS = {"eff": ("clip_hits",), "c0": ("guard_hits",),
                   "official": ("full_groups", "cleared_zero_episodes", "partial_episodes"),
                   "rank": ("ranked_groups", "warmup_groups"),
                   "hinge": ("hinged_groups", "unfrozen_groups"),
                   # note: stop-window episodes in the considered groups, and those that changed window early (R 0, not judged)
                   "note": ("window_episodes", "early_window_episodes")}
#: Σ|·| of c0's two efficiency components over the kept groups (numerator and denominator of token_share)
ARC3_C0_SUMS = ("eff_actions_abs", "eff_tokens_abs")


def arc3_reward_summary(mode: str, params: dict, stats: collections.Counter, hist: collections.Counter,
                        sums: dict) -> dict:
    """Summary and ``rl.arc3_reward``: mode, parameters, group counts (considered = reached the advantage, kept, nonzero
    efficiency term, all cleared), the mode's own counts (``ARC3_MODE_STATS``), the p̂ histogram (considered, keys with 4
    decimals), and the efficiency term's share of the gradient Σ|eff| / (Σ|c − p̂| + Σ|eff|) (kept groups; unweighted and
    weighted by L_i; denominator 0 = no kept group, None). c0 also records ``token_share`` = the token component's share
    of the efficiency term Σ|eff_tokens| / (Σ|eff_actions| + Σ|eff_tokens|) (None when the denominator is 0)."""
    def share(eff: str, task: str) -> float | None:
        total = sums[task] + sums[eff]
        return sums[eff] / total if total > 0 else None

    out = {"mode": mode, **params, "rule": ARC3_RULES[mode],
           **{k: stats[k] for k in ("groups_considered", "groups_kept", "groups_eff_nonzero", "all_cleared_groups",
                                    "all_cleared_groups_kept", *ARC3_MODE_STATS[mode])},
           "p_hat_histogram": {k: hist[k] for k in sorted(hist)},
           "gradient_share": {"unweighted": share("eff_abs", "task_abs"),
                              "loss_token_weighted": share("eff_abs_loss_tokens", "task_abs_loss_tokens"),
                              **{k: sums[k] for k in ARC3_SHARE_SUMS}}}
    if mode == "c0":
        eff_total = sums["eff_actions_abs"] + sums["eff_tokens_abs"]
        out["token_share"] = sums["eff_tokens_abs"] / eff_total if eff_total > 0 else None
        out.update({k: sums[k] for k in ARC3_C0_SUMS})
    return out


def load_judge_files(manifests, episodes: dict, join_mode: str) -> dict[str, str]:
    """``--arc3-reward note``: each group directory's (the manifest's) judge.json → those episodes' ``judged`` (rewards
    recomputed by ``arcforge.arc3_note.judge_rewards``). Episodes to judge = harness finished (has a reward), stop window,
    window_cause threshold; when a group has episodes to judge, judge.json must exist and hold exactly those episodes
    (action and RESET counts must also match the outcome); a judge.json in a group with nothing to judge also raises.
    Returns {run: sha256 of judge.json} (groups that have one)."""
    from arcforge.arc3_note import judge_rewards, judged_episode
    if join_mode != GROUP:
        raise RuntimeError("--arc3-reward note is group mode only (one judge.json per group, in the group's run directory)")
    files = {}
    for m in manifests:
        eps = [episodes[f"{m.run}/{iid}"] for iid in sorted(m.items)]
        scored = [ep for ep in eps if ep.reward is not None]
        want = {ep.key.split("/", 1)[1]: ep for ep in scored if judged_episode(ep.stop, ep.window_cause)}
        path = m.run_dir / JUDGE_FILE
        if not want:
            if path.exists():
                raise RuntimeError(f"{path}: the group has no episode to judge (stop window at the threshold), yet the file "
                                   "is there")
            continue
        if not path.is_file():
            raise RuntimeError(f"{m.path}: episodes {sorted(want)[:3]} stopped at the window change and need the note judge, "
                               f"but {path} does not exist")
        level = {(m.items[iid]["outcome"]["game"], m.items[iid]["outcome"]["level"], ep.baseline_actions)
                 for iid, ep in want.items()}
        if len(level) != 1:
            raise RuntimeError(f"{m.path}: one group is one level, its judged episodes are {sorted(level)}")
        raw = path.read_bytes()
        got = judge_rewards(json.loads(raw), *level.pop())
        if set(got) != set(want):
            raise RuntimeError(f"{path}: grades {sorted(got)[:4]}, the group's episodes to judge are {sorted(want)[:4]} "
                               f"(missing {sorted(set(want) - set(got))[:3]}, extra {sorted(set(got) - set(want))[:3]})")
        for iid, ep in want.items():
            if (got[iid]["actions"], got[iid]["resets"]) != (ep.actions, ep.resets):
                raise RuntimeError(f"{path}: note {iid} has actions {got[iid]['actions']} resets {got[iid]['resets']}, the "
                                   f"outcome {ep.actions} / {ep.resets}")
            ep.judged = got[iid]["reward"]
        files[m.run] = hashlib.sha256(raw).hexdigest()
    return files


def coef_denominators(eps, reduction: str) -> dict[str, int]:
    """One minibatch's episodes → coef denominators (episode key → int). ``sequence``: E · L_i; ``prompt``: H · T_g."""
    if reduction not in LOSS_REDUCTIONS:
        raise ValueError(f"loss reduction {reduction!r} not in {LOSS_REDUCTIONS}")
    if reduction == "sequence":
        return {ep.key: len(eps) * ep.gen_tokens for ep in eps}
    group_tokens = collections.Counter()
    for ep in eps:
        group_tokens[ep.group] += ep.gen_tokens
    return {ep.key: len(group_tokens) * group_tokens[ep.group] for ep in eps}


def split_minibatches(group_tokens: dict[str, int], k: int, seed: int) -> list[list[str]]:
    """Groups are not split; LPT-assigned by token count to k minibatches (ties ordered by a seeded shuffle); the
    minibatch order is shuffled too."""
    if k < 1 or k > len(group_tokens):
        raise ValueError(f"--minibatches {k}: only {len(group_tokens)} groups")
    rng = random.Random(seed)
    order = sorted(group_tokens)
    rng.shuffle(order)
    order.sort(key=lambda g: -group_tokens[g])          # stable: ties keep the shuffled order
    heap = [(0, b) for b in range(k)]
    members: list[list[str]] = [[] for _ in range(k)]
    for g in order:
        load, b = heapq.heappop(heap)
        members[b].append(g)
        heapq.heappush(heap, (load + group_tokens[g], b))
    rng.shuffle(members)
    return members


def lpt_bins(lengths, cap: int, world: int) -> list[list[int]]:
    """Samples (not split) LPT into B bins: B starts at the best-fit bin count rounded up to a multiple of world, +world
    when they do not fit. Returns B bins (sample indices, increasing); with fewer samples than B some bins are empty (the
    caller pads them)."""
    from arcforge.train.packed_text import plan_bins

    lengths = [int(n) for n in lengths]
    if not lengths or world < 1:
        raise ValueError("no samples or world < 1")
    if max(lengths) > cap:
        raise ValueError(f"a sample has {max(lengths)} tokens > bin {cap}")
    b = -(-len(plan_bins(lengths, cap)) // world) * world
    order = sorted(range(len(lengths)), key=lambda i: (-lengths[i], i))
    while True:
        heap = [(0, j) for j in range(b)]
        bins: list[list[int]] = [[] for _ in range(b)]
        for i in order:
            load, j = heapq.heappop(heap)
            if load + lengths[i] > cap:
                break
            bins[j].append(i)
            heapq.heappush(heap, (load + lengths[i], j))
        else:
            return [sorted(x) for x in bins]
        b += world


def bins_to_steps(bins: list[list[int]], lengths, world: int) -> list[list[list[int]]]:
    """Bins sorted by token count, largest first; each run of world consecutive bins is one micro-step (bins of one step
    are of similar size)."""
    if len(bins) % world:
        raise ValueError(f"{len(bins)} bins while world={world}: not a multiple")
    order = sorted(bins, key=lambda x: (-sum(int(lengths[i]) for i in x), x))
    return [order[s:s + world] for s in range(0, len(order), world)]


# ----------------------------------------------------------------------------- records


def dropped_reward_sum(episodes: dict) -> float:
    """Sum of the rewards of the dropped_detail episodes (episode-level drops; episodes without a reward excluded): how
    much signal was dropped, plottable in wandb."""
    return round(sum(d["reward"] for d in dropped_detail(episodes) if d["reward"] is not None), 6)


def dropped_detail(episodes: dict) -> list[dict]:
    """Episodes dropped for an episode-level reason (not the group rules group_std0 / group_lt2): reason, reward, stop and
    generated tokens, so a drop that removed a group's only success (10-24 arc2 g000006: two too_long episodes, the
    group's only R = 1) is visible in the manifest."""
    return [{"episode": ep.key, "reason": ep.drop, "reward": ep.reward, "stop": ep.stop, "gen_tokens": ep.gen_tokens}
            for ep in sorted(episodes.values(), key=lambda e: e.key)
            if ep.drop is not None and not ep.drop.startswith("group_")]


#: max fraction of a batch made of episodes dropped whole because the arc3 rid join did not match; beyond it = systematic
#: (fork / harness bug), raise as before
RID_DROP_MAX_FRACTION = 0.25


@dataclass
class Episode:
    key: str                              # <run>/<item id>
    run: str
    task: str                             # classic: the item id; group mode: <tid>
    group: str                            # classic: the item id; group mode: <run>:<tid> (manifest_groups)
    status: str
    stop: str | None
    reward: float | None
    segments: list = field(default_factory=list)     # (Instance, Segment)
    m_reasons: collections.Counter = field(default_factory=collections.Counter)
    d_positions: collections.Counter = field(default_factory=collections.Counter)   # D_same / D_unknown / D_cross → position count
    d_enforced: bool | None = None        # its segments' server ran --session-exact-resume (None: no segment yet)
    too_long: bool = False
    no_support: int = 0                   # appends without a sampling support (truncated sampling): dropped as no_support
    compacted: bool = False               # a thread of it has a compaction request or a second window
    context_overflows: int = 0            # harness sessions that hit the context length (outcome.context_overflows)
    unrecorded: int = 0                   # generated tokens whose append has weight_versions null
    stray_media_pads: int = 0             # sampled image/video pad runs in its segments (ledger Segment.stray_media_pads)
    gen_versions: collections.Counter = field(default_factory=collections.Counter)   # segment_versions summed
    version_tokens: dict[str, int] = field(default_factory=dict)   # loss tokens per weight version
    drop: str | None = None
    advantage: float | None = None
    # the R the advantage is centred (and with adv std, scaled) from: the shaped R (--arc3-reward eff / c0) or the reward;
    # None with the --efficiency-beta term (its advantage is not a function of one R per episode)
    training_reward: float | None = None
    gen_tokens: int = 0
    output_tokens: int | None = None      # outcome output_tokens (proxy count); read only with the efficiency term, logged
    # outcome correct / actions / tokens_at_clear (run_arc3_level.tito_items); read only with --arc3-reward eff / c0 / rank;
    # official reads correct / actions / baseline_actions
    correct: bool | None = None
    actions: int | None = None
    tokens_at_clear: int | None = None
    baseline_actions: int | None = None
    # outcome window_cause / resets and the judge.json reward of its note; read only with --arc3-reward note
    window_cause: str | None = None
    resets: int | None = None
    judged: float | None = None
    # the policy version its group was dispatched at (--arc3-launch-versions); read only with rank warmup > 0
    launch_version: int | None = None
    # outcome tool_use (run_arc2 tool_use_stats, arcforge.arc2_tooluse.STATS_KEYS); read only with the --tool-* term
    tool_use: dict | None = None
    minibatch: int | None = None
    coef: float | None = None             # float32 value
    # --wrapup-loss mask: outcome wrapup_post_rids (the turns after the wrap-up line); their gen spans leave the loss
    wrapup_post_rids: frozenset | None = None
    wrapup_masked_tokens: int = 0
    wrapup_masked_rids: int = 0


@dataclass
class Sample:
    key: str                              # the segment key; a pad copy is "<segment key>#pad<n>"
    inst: object
    seg: object
    episode: Episode | None               # None: a pad copy
    minibatch: int
    pad_of: str | None = None             # the copied segment's key


def _index_row(s: Sample, media: list[dict], weight_version: str | list[str], harness: str) -> dict:
    seg = s.seg
    thread = None if seg.sid is None else seg.thread
    e = s.episode
    pad = s.pad_of is not None
    return {
        "key": s.key, "instance": seg.instance, "seg": seg.seg, "sid": seg.sid, "thread": thread,
        "window": -1 if seg.sid is None else seg.window, "task": None if pad else e.task, "harness": harness,
        "cause": seg.cause, "split_reason": seg.split_reason,
        "close_reason": None if seg.close is None else seg.close["reason"], "closed": seg.status == "closed",
        "length": len(seg), "n_appends": len(seg.appends), "n_rewind_appends": sum(a.rewind > 0 for a in seg.appends),
        "n_stop_substituted": sum(a.stop_substituted for a in seg.appends),
        "n_dropped": sum(sp.dropped for sp in seg.gen_spans),
        "n_length_finish": sum(a.finish["type"] == "length" for a in seg.appends), "media": media,
        "n_gen_spans": len(seg.gen_spans),
        "episode": None if pad else e.key, "group": None if pad else e.group, "reward": None if pad else e.reward,
        "advantage": 0.0 if pad else e.advantage, "training_reward": None if pad else e.training_reward,
        "gen_tokens": 0 if pad else e.gen_tokens,
        "weight_version": weight_version, "weight_version_tokens": {} if pad else e.version_tokens,
        "run": None if pad else e.run, "coef": 0.0 if pad else e.coef,
        "minibatch": s.minibatch, "pad": pad, "pad_of": s.pad_of,
    }


def _expected_loss_and_behavior(seg, replay: bool = False,
                                skip_rids: frozenset = frozenset()) -> tuple[np.ndarray, np.ndarray]:
    """For read-back verification: re-derive the loss mask and behavior logprob from the append records only (not via
    gen_spans) (``replay``: the ones over the support). ``skip_rids``: the wrap-up appends of --wrapup-loss mask (the
    outcome's wrapup_post_rids), not loss."""
    n = len(seg)
    loss = np.zeros(n, bool)
    behavior = np.full(n, np.nan, np.float32)
    for k, a in enumerate(seg.appends):
        rewound = seg.appends[k + 1].rewind if k + 1 < len(seg.appends) else 0
        m = a.gen_len - rewound
        if m <= 0 or a.rid in skip_rids:
            continue
        loss[a.gen_start:a.gen_start + m] = True
        behavior[a.gen_start:a.gen_start + m] = (a.sampling_mask.logprobs if replay else a.logprobs)[:m]
        if not np.array_equal(seg.ids[a.gen_start:a.gen_start + m], a.gen[:m]):
            raise RuntimeError(f"{seg.key}: read-back check: append {k}'s gen is not at S[{a.gen_start}, +{m})")
    return loss, behavior


# ----------------------------------------------------------------------------- main


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ledger", nargs="+", required=True, help="ledger instance directories (or their parent)")
    ap.add_argument("--manifest", action="append", required=True,
                    help="tito_manifest.json (repeatable): classic = one sample run; group mode = one group (items <tid>__s<k>)")
    ap.add_argument("--out", required=True, help="output directory (must not exist)")
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--weight-version", help=f"allowed weight versions (comma-separated); {UNVERSIONED} = ledgers from before the fork fix. "
                                             "Mutually exclusive with --current-version / --max-staleness")
    ap.add_argument("--current-version", type=int, help="streaming: the trainer's version V (raises on token versions > V)")
    ap.add_argument("--max-staleness", type=int, help="streaming: allows [V−S, V]; episodes with older tokens are dropped as stale")
    ap.add_argument("--temperature", type=float, required=True, help="rollout sampling temperature (must be the same on every ledger row)")
    ap.add_argument("--top-p", type=float, default=1.0,
                    help="rollout top_p (the same on every ledger row; < 1 needs the ledger to record the sampling support, batch is replay: mask)")
    ap.add_argument("--top-k", type=int, default=-1,
                    help="rollout top_k (-1 = off; the same on every ledger row; when on, the ledger must record the sampling support, batch is replay: mask)")
    ap.add_argument("--tokenizer-dir", required=True, help="model directory (tokenizer.json, config.json, processor settings)")
    ap.add_argument("--minibatches", type=int, required=True, help="number of optimizer steps (groups are not split)")
    ap.add_argument("--world", type=int, default=4, help="trainer rank count: one bin per rank per micro-step")
    ap.add_argument("--bin-tokens", type=int, default=120_000)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--adv-normalization", choices=ADV_NORMALIZATIONS, default="std",
                    help="std: (R − mean_g) / std_g; none: R − mean_g (no division by std)")
    ap.add_argument("--loss-reduction", choices=LOSS_REDUCTIONS, default="sequence",
                    help="sequence: coef = A/(E·L_i) (each episode averaged on its own); prompt: coef = A/(H·T_g) (one group's tokens averaged together)")
    ap.add_argument("--efficiency-beta", type=float, default=0.0,
                    help="β of the efficiency term (0 = off): prefers the shorter all-correct episodes (efficiency_advantages); only with --adv-normalization none")
    ap.add_argument("--efficiency-budget", type=int, help="B of the efficiency term: the harness's output budget (output_token_budget), required when β > 0")
    ap.add_argument("--efficiency-guard", type=float, default=0.25,
                    help="in a mixed group, the longest all-correct episode loses at most this many times its task advantage")
    ap.add_argument("--tool-hand-weight", type=float, default=0.0,
                    help="tool-use term (arcforge.arc2_tooluse.tool_advantages): weight of the hand-calculation term min(hand_chars/scale, cap) (0 = off)")
    ap.add_argument("--tool-toolkit-weight", type=float, default=0.0,
                    help="tool-use term: weight of u, the fraction of code-carrying commands that call toolkit functions (0 = off)")
    ap.add_argument("--tool-hand-scale", type=float, help="tool-use term: scale of hand-calculation characters (required when either weight > 0)")
    ap.add_argument("--tool-hand-cap", type=float, help="tool-use term: cap of the hand-calculation term (required when either weight > 0)")
    ap.add_argument("--arc3-reward", choices=ARC3_REWARDS, default="score",
                    help="score: reward = the outcome's score; eff: within-group relative clear + efficiency (arc3_eff_rewards); c0: smooth efficiency "
                         "term centred among clearing episodes (arc3_c0_rewards); official: piecewise linear of the official per-level score "
                         "(arc3_official_rewards); rank: score times a discount by within-group efficiency rank (arc3_rank_rewards); hinge "
                         "(arc3_hinge_rewards); note: the first window's clear score or judge.json's note score (arc3_note_rewards, group mode). "
                         "eff / c0 / official / rank / hinge / note only with harness arc3 and --adv-normalization none, "
                         "not with --efficiency-beta, and their own parameters (ARC3_MODE_PARAMS) must all be given explicitly")
    ap.add_argument("--arc3-beta-actions", type=float, help="eff: β_a of the action term (≥ 0; ln actions is also multiplied by 2)")
    ap.add_argument("--arc3-beta-tokens", type=float, help="eff: β_t of the token term (≥ 0; tokens_at_clear)")
    ap.add_argument("--arc3-w-start", type=float, help="eff / c0: w(p̂) is 0 for p̂ ≤ this, then rises linearly to 1 ([0, 1); 0 = w = p̂)")
    ap.add_argument("--arc3-eff-clip", type=float, help="eff: E is first clipped to ±this, then re-centred (> 0; ≤ 0.5 guarantees R > 0 for a clear)")
    ap.add_argument("--arc3-lambda-actions", type=float, help="c0: action term λ_a (≥ 0)")
    ap.add_argument("--arc3-kappa-actions", type=float, help="c0: slope κ_a of the action-term sigmoid (> 0, over ln action count)")
    ap.add_argument("--arc3-lambda-tokens", type=float, help="c0: token term λ_t (≥ 0; tokens_at_clear)")
    ap.add_argument("--arc3-kappa-tokens", type=float, help="c0: slope κ_t of the token-term sigmoid (> 0, over ln tokens_at_clear)")
    ap.add_argument("--arc3-success-floor", type=float, help="c0: every clear's R is at least this ((0, 1))")
    ap.add_argument("--arc3-top-score", type=float, help="official: official per-level score ≥ this → R = 1 (≤ 115)")
    ap.add_argument("--arc3-knee-score", type=float, help="official: knee of the piecewise linear (official score)")
    ap.add_argument("--arc3-knee-reward", type=float, help="official: R at the knee ((0, 1))")
    ap.add_argument("--arc3-zero-score", type=float, help="official: a clear with official score ≤ this → R = 0")
    ap.add_argument("--arc3-rank-actions", type=float, help="rank: weight α of the action rank (≥ 0, α + β < 1)")
    ap.add_argument("--arc3-rank-tokens", type=float, help="rank: weight β of the tokens_at_clear rank (≥ 0, α + β < 1)")
    ap.add_argument("--arc3-rank-pass-scale", type=int, help="rank: 1 = efficiency term times the within-group clear fraction ρ (of the kept episodes); 0 = no factor")
    ap.add_argument("--arc3-rank-warmup-versions", type=int,
                    help="rank: groups with launch version < this get efficiency term 0 (R = s); > 0 needs --arc3-launch-versions")
    ap.add_argument("--arc3-hinge-tokens", type=float, help="hinge: weight λ_t of the token threshold term (≥ 0, λ_t + λ_a < 1)")
    ap.add_argument("--arc3-hinge-actions", type=float, help="hinge: weight λ_a of the action threshold term (≥ 0)")
    ap.add_argument("--arc3-hinge-warmup-versions", type=int, help="hinge: thresholds come from the warm-up groups with launch version < this (≥ 1)")
    ap.add_argument("--arc3-hinge-refs", type=str,
                    help="hinge: absolute path of the frozen threshold file (written by work/qwen38fn_rl/hinge_refs.py; absent = no level frozen yet)")
    ap.add_argument("--arc3-launch-versions", type=Path,
                    help="rank warm-up / hinge: JSON {manifest path (the same string as given to --manifest): this group's launch version}, one for every manifest")
    ap.add_argument("--algo", choices=tuple(ALGOS), default="bpo",
                    help="the trainer's loss (SPEC §1): bpo (the four parameters below) or mis (--mis-low / --mis-high); parameters only with their own algo, "
                         "unset ones take the rl_loss.ALGOS defaults")
    for algo, params in ALGOS.items():
        for name, default in params.items():
            ap.add_argument("--" + name.replace("_", "-"), type=float, help=f"{algo} (default {default})")
    ap.add_argument("--media-device", choices=tuple(ptm.DEVICE_PATHS),
                    help="required when the batch has media: cuda = deployed as is (one GPU); cpu-exact = bit-exact CUDA bicubic reproduction on CPU (no GPU); "
                         "cpu-emulated = approximation for when the GPU is off limits (see prepare_tito_mm)")
    ap.add_argument("--allow-emulated-media", action="store_true",
                    help="with cpu-emulated, do not raise on bicubic-resized media (pixels may differ from deployment by ±1; the manifest records the count)")
    ap.add_argument("--workers", type=int, default=16, help="worker count for media / M-RoPE")
    ap.add_argument("--allow-compaction", action="store_true",
                    help="train episodes with several context windows (whole-game arc3_game): one segment per window, sharing the episode's advantage; "
                         "not given = dropped as compacted (RL v1)")
    ap.add_argument("--wrapup-loss", choices=WRAPUP_LOSS_MODES, default="train",
                    help="train = wrap-up generations are loss as usual (default, the original behavior); mask = tokens generated by the outcome's wrapup_post_rids "
                         "(requests sent after the wrap-up line, run_arc2) do not enter the loss and stay in the sequence as context (mask_wrapup_spans)")
    ap.add_argument("--defer-fraction-gates", action="store_true",
                    help="do not judge here the 0.5%% gates for routing M / unrecorded versions or the D_cross fraction gate (the summary's quality gives the counts; "
                         "the caller judges them over the whole optimizer step)")
    from arcforge.rl_storage import DEFAULT_ROUTING_GB, DEFAULT_RESERVE_GB, MAX_D_CROSS_FRACTION
    ap.add_argument("--drop-stops", default="",
                    help="comma list of outcome stops dropped as infra failures (drop reason infra_stop) instead of "
                         "training their reward, e.g. arc2 wall,deadline,cancelled,proxy_fatal (10-24 arc2 RL review A8). "
                         "Default: none")
    ap.add_argument("--max-d-cross-fraction", type=float, default=MAX_D_CROSS_FRACTION,
                    help="cap on the fraction of episodes dropped for D_cross (positions forwarded twice across weight versions); raises beyond it (counts only "
                         "episodes whose server had session_exact_resume on)")
    ap.add_argument("--max-routing-m-fraction", type=float, default=MAX_M_DROP_FRACTION,
                    help="cap on the fraction of episodes dropped for routing M (late_branch included); raises beyond it (default: SPEC §5's 0.5%%)")
    ap.add_argument("--routing-workers", type=int, default=16,
                    help="worker count for routing status checks and writing (one job per segment; CPU, no GPU)")
    ap.add_argument("--max-gb", type=float, default=DEFAULT_ROUTING_GB, help="routing output cap (decimal GB)")
    ap.add_argument("--reserve-gb", type=float, default=DEFAULT_RESERVE_GB, help="workspace / export headroom beyond the routing cap (decimal GB)")
    ap.add_argument("--error-json", type=Path,
                    help="(for the orchestrator) single-group data problems (a ManifestError that points at manifests: kind group_data) and routing over "
                         "--max-gb (kind routing_over_max_gb) are written to this JSON file and exit with PREPARE_EXIT_CODES; other errors still raise")
    return ap


def main(argv: list[str] | None = None) -> dict:
    args = build_parser().parse_args(argv)
    from arcforge.rl_storage import required_free_bytes
    from arcforge.tito.join import Joiner, load_manifests, rl_rid_join
    from arcforge.tito.ledger import (ROUTING_D_CROSS, ROUTING_D_SAME, ROUTING_D_UNKNOWN, ROUTING_M, ROUTING_NAMES,
                                      ROUTING_X, Vocab, instance_dirs, load_instance, resume_violations)

    t_start = time.time()
    required_bytes = required_free_bytes(args.max_gb, args.reserve_gb)
    timings: dict[str, float] = {}
    if min(args.minibatches, args.world, args.bin_tokens, args.routing_workers) < 1 or args.temperature <= 0:
        raise ValueError("--minibatches / --world / --bin-tokens / --routing-workers must be positive, --temperature > 0")
    if not 0 < args.top_p <= 1 or not (args.top_k == -1 or args.top_k >= 1):
        raise ValueError(f"--top-p={args.top_p} must be in (0, 1], --top-k={args.top_k} must be -1 or ≥ 1")
    top_k_rollout = None if args.top_k == -1 else args.top_k
    replay = sampling_replay(args.top_p, top_k_rollout)
    if args.efficiency_beta < 0:
        raise ValueError(f"--efficiency-beta={args.efficiency_beta} cannot be negative")
    if not 0 <= args.max_d_cross_fraction <= 1:
        raise ValueError(f"--max-d-cross-fraction={args.max_d_cross_fraction} must be in [0, 1]")
    if not 0 <= args.max_routing_m_fraction <= 1:
        raise ValueError(f"--max-routing-m-fraction={args.max_routing_m_fraction} must be in [0, 1]")
    foreign = ["--" + k.replace("_", "-") for algo, params in ALGOS.items() if algo != args.algo for k in params
               if getattr(args, k) is not None]
    if foreign:
        raise ValueError(f"{foreign} are not --algo {args.algo} parameters")
    algo_params = {k: default if getattr(args, k) is None else getattr(args, k) for k, default in ALGOS[args.algo].items()}
    check_algo_params(args.algo, tuple(algo_params.values()))
    efficiency = args.efficiency_beta > 0
    if efficiency and (args.adv_normalization != "none" or args.efficiency_budget is None):
        raise ValueError("--efficiency-beta > 0 needs --adv-normalization none (dividing by std would cancel β) and --efficiency-budget")
    if not efficiency and args.efficiency_budget is not None:
        raise ValueError("--efficiency-budget is only used with --efficiency-beta > 0")
    arc3_mode = None if args.arc3_reward == "score" else args.arc3_reward    # eff / c0 / official / rank: the shaped R
    arc3_flag = {k: "--arc3-" + k.replace("_", "-") for keys in ARC3_MODE_PARAMS.values() for k in keys}
    arc3_given = {k for k in arc3_flag if getattr(args, "arc3_" + k) is not None}
    arc3_params = {}
    if arc3_mode is not None:
        missing = [arc3_flag[k] for k in ARC3_MODE_PARAMS[arc3_mode] if k not in arc3_given]
        if missing:
            raise ValueError(f"--arc3-reward {arc3_mode} needs {missing} given explicitly (no defaults)")
        if args.adv_normalization != "none" or efficiency:
            raise ValueError(f"--arc3-reward {arc3_mode} needs --adv-normalization none (dividing by std would cancel the efficiency term's scale) and no "
                             "--efficiency-beta (it assumes rewards are only 0/1)")
        arc3_params = {k: getattr(args, "arc3_" + k) for k in ARC3_MODE_PARAMS[arc3_mode]}
        if arc3_mode == "eff":
            check_arc3_eff_params(beta_actions=args.arc3_beta_actions, beta_tokens=args.arc3_beta_tokens,
                                  w_start=args.arc3_w_start, clip=args.arc3_eff_clip)
        elif arc3_mode == "c0":
            check_arc3_c0_params(**arc3_params)
        elif arc3_mode == "official":
            from arcforge.arc3_official import check_params
            check_params(**arc3_params)
        elif arc3_mode == "rank":
            from arcforge.arc3_rank import check_params
            check_params(**arc3_params)
        elif arc3_mode == "hinge":
            from arcforge.arc3_hinge import check_params
            check_params(**arc3_params)
    launch_versions = None
    if (arc3_mode == "rank" and arc3_params["rank_warmup_versions"] > 0) or arc3_mode == "hinge":
        if args.arc3_launch_versions is None:
            raise ValueError(f"--arc3-reward {arc3_mode}{' (warm-up > 0)' if arc3_mode == 'rank' else ''} needs --arc3-launch-versions"
                             " (each group's launch version; prepare cannot see it)")
        launch_versions = json.loads(args.arc3_launch_versions.read_text())
        if not (isinstance(launch_versions, dict) and all(
                isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in launch_versions.values())):
            raise ValueError(f"{args.arc3_launch_versions}: want a JSON object {{manifest path: int >= 0}}")
        if set(launch_versions) != set(args.manifest):
            raise ValueError(f"{args.arc3_launch_versions}: its manifests differ from --manifest (only there: "
                             f"{sorted(set(launch_versions) - set(args.manifest))[:3]}, missing: "
                             f"{sorted(set(args.manifest) - set(launch_versions))[:3]})")
    elif args.arc3_launch_versions is not None:
        raise ValueError("--arc3-launch-versions only goes with --arc3-reward hinge, or rank with --arc3-rank-warmup-versions > 0")
    hinge_refs, hinge_refs_sha256 = None, None
    if arc3_mode == "hinge":                       # read once: every group of this batch sees the same frozen levels
        from arcforge.arc3_hinge import load_refs
        hinge_refs = load_refs(arc3_params["hinge_refs"])
        if hinge_refs:
            hinge_refs_sha256 = hashlib.sha256(Path(arc3_params["hinge_refs"]).read_bytes()).hexdigest()
    foreign = [arc3_flag[k] for k in sorted(arc3_given - set(ARC3_MODE_PARAMS.get(arc3_mode, ())))]
    if foreign:
        raise ValueError(f"{foreign} are not --arc3-reward {args.arc3_reward} parameters")
    tool_on = args.tool_hand_weight != 0 or args.tool_toolkit_weight != 0
    tool_params = {}
    if tool_on:
        if args.adv_normalization != "none" or arc3_mode is not None:
            raise ValueError("--tool-*-weight needs --adv-normalization none and no --arc3-reward (the tool term is added on top of R − mean R)")
        if args.tool_hand_scale is None or args.tool_hand_cap is None:
            raise ValueError("--tool-*-weight needs --tool-hand-scale and --tool-hand-cap given explicitly")
        from arcforge.arc2_tooluse import check_params as check_tool_params, tool_advantages
        tool_params = {"hand_weight": args.tool_hand_weight, "toolkit_weight": args.tool_toolkit_weight,
                       "hand_scale": args.tool_hand_scale, "hand_cap": args.tool_hand_cap}
        check_tool_params(**tool_params)
    elif args.tool_hand_scale is not None or args.tool_hand_cap is not None:
        raise ValueError("--tool-hand-scale / --tool-hand-cap are only used when a --tool-*-weight is nonzero")
    windowed = args.current_version is not None or args.max_staleness is not None
    if windowed and (args.weight_version is not None or args.current_version is None or args.max_staleness is None):
        raise ValueError("--current-version and --max-staleness must be given together, and not together with --weight-version")
    if not windowed and args.weight_version is None:
        raise ValueError("give --weight-version, or --current-version plus --max-staleness")
    if windowed:
        # streaming (SPEC §8): token versions must be in [V-S, V]; the batch-level value is set from the versions
        # actually present once the kept episodes are known
        allowed = window_versions(args.current_version, args.max_staleness)
        weight_version = None
    else:
        allowed = parse_versions(args.weight_version)
        # v1 (one value, including unversioned) keeps the scalar string; an async set is the sorted list (SPEC §8)
        weight_version = allowed[0] if len(allowed) == 1 else allowed
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f"{out} already exists: batches are not mixed")
    model_dir = Path(args.tokenizer_dir)
    vocab = Vocab(model_dir)
    text_cfg = json.loads((model_dir / "config.json").read_text())["text_config"]
    layers, top_k, num_experts = (int(text_cfg[k]) for k in ("num_hidden_layers", "num_experts_per_tok", "num_experts"))

    # ---- 1. manifests, ledger, rid join
    t0 = time.time()
    manifests = load_manifests(args.manifest)
    runs = [m.run for m in manifests]
    if len(set(runs)) != len(runs):
        raise RuntimeError(f"duplicate manifest run directory names: {runs} (the episode key is <run>/<item id>)")
    compacted: set[str] = set()           # episode keys; dropped unless --allow-compaction (whole-game episodes)
    harnesses = sorted({m.harness for m in manifests})
    if len(harnesses) != 1 or harnesses[0] not in RL_HARNESSES:
        raise RuntimeError(f"manifest harnesses are {harnesses}: a batch must be exactly one of {RL_HARNESSES}, not a mix")
    harness = harnesses[0]
    if arc3_mode is not None and harness != ARC3:
        raise RuntimeError(f"--arc3-reward {arc3_mode} is only for arc3 single-level episodes (run_arc3_level.py), this batch is {harness}")
    for m in manifests:
        if harness == ARC3:
            # whole-game arc3 manifests (collected by run_arm) have no reward: mixed in, they would all be silently
            # dropped as harness_failed
            unscored = [iid for iid, item in m.items.items()
                        if item["outcome"]["status"] == "finished" and "score" not in item["outcome"]]
            if unscored:
                raise RuntimeError(f"{m.path}: arc3 items {unscored[:3]} have no score -- not a manifest of single-level episodes"
                                   " (run_arc3_level.py)")
        for thread, info in m.threads.items():
            if len(info.windows) > 1 or any(w.compaction_response_id is not None for w in info.windows):
                compacted.add(f"{m.run}/{info.item}")
    dirs = instance_dirs(args.ledger)
    wanted_threads = {t for m in manifests for t in m.threads}
    instances = [load_instance(d, vocab, strict=True, threads=wanted_threads) for d in dirs]
    names = {inst.name for inst in instances}
    for m in manifests:
        lack = [i for srv in m.servers for i in srv["instances"] if i not in names]
        if lack:
            raise RuntimeError(f"{m.path}: rollout ledger instances {lack} are not in --ledger")
    join_mode, group_of = manifest_groups(manifests)
    # arc3: an rid mismatch drops only that game (rl_rid_join drop_episodes); arc2 raises as before (ManifestError →
    # quarantine those groups)
    drop_rid = bool(manifests) and all(m.harness == "arc3" for m in manifests)
    rj = rl_rid_join(instances, manifests, drop_episodes=drop_rid)
    rid_dropped = rj.get("dropped_episodes", {})
    for key, reasons in sorted(rid_dropped.items()):
        print(f"DROP EPISODE {key} (rid_join): {reasons[0]}" + (f" (+{len(reasons) - 1} more)" if len(reasons) > 1 else ""),
              file=sys.stderr, flush=True)
    joiner = Joiner(manifests)
    launch_of = None if launch_versions is None else {str(Path(k)): v for k, v in launch_versions.items()}

    episodes: dict[str, Episode] = {}
    for m in manifests:
        for iid in m.items:
            o = m.outcome(iid)
            group, task = group_of[(m.run, iid)]
            key = f"{m.run}/{iid}"
            arc3_fields = {}
            if arc3_mode is not None and o.reward is not None:
                lack = [k for k in ARC3_OUTCOME[arc3_mode] if k not in o.detail]
                if lack:
                    raise RuntimeError(f"{m.path}: item {iid} outcome lacks {lack}: --arc3-reward {arc3_mode} needs the manifest to carry these fields"
                                       " (run_arc3_level.tito_items)")
                arc3_fields = {k: o.detail[k] for k in ARC3_OUTCOME[arc3_mode]}
            if tool_on and o.reward is not None:
                if "tool_use" not in o.detail:
                    raise RuntimeError(f"{m.path}: item {iid} outcome has no tool_use: --tool-*-weight needs the harness config to enable "
                                       "tool_use_stats (run_arc2.py)")
                arc3_fields["tool_use"] = o.detail["tool_use"]
            if args.wrapup_loss == "mask" and o.reward is not None:
                post = o.detail.get("wrapup_post_rids")
                if not isinstance(post, list):
                    raise RuntimeError(f"{m.path}: item {iid} outcome has no wrapup_post_rids ({post!r}): --wrapup-loss mask "
                                       "needs the harness to enable wrap-up (wrapup_tokens) and run_arc2 to record the requests after the wrap-up line")
                if o.detail.get("wrapup_cut_rid") is not None and o.detail["wrapup_cut_rid"] in post:
                    raise RuntimeError(f"{m.path}: item {iid} wrapup_cut_rid {o.detail['wrapup_cut_rid']} is in "
                                       "wrapup_post_rids: the turn cut across the line is still within budget, not wrap-up")
                arc3_fields["wrapup_post_rids"] = frozenset(post)
            episodes[key] = Episode(key=key, run=m.run, task=task, group=group, status=o.status,
                                    stop=o.detail.get("stop"), reward=o.reward, compacted=key in compacted,
                                    context_overflows=o.detail["context_overflows"] if o.reward is not None else 0,
                                    output_tokens=(o.detail["output_tokens"] if efficiency and o.reward is not None
                                                   else None),
                                    launch_version=None if launch_of is None else launch_of[str(m.path)], **arc3_fields)

    judge_files = load_judge_files(manifests, episodes, join_mode) if arc3_mode == "note" else None

    if len(rid_dropped) > RID_DROP_MAX_FRACTION * len(episodes):
        raise RuntimeError(f"rid join: {len(rid_dropped)}/{len(episodes)} episodes do not match (> {RID_DROP_MAX_FRACTION:.0%})"
                           f" = a systematic problem, not individual broken games; e.g. {sorted(rid_dropped.items())[:2]}")

    # ---- 2. per segment: conditions, alignment, routing status
    stats = collections.Counter()
    stats.update(ledger_total_segments=sum(inst.total_segments for inst in instances),
                 ledger_read_bytes=sum(inst.events_read_bytes for inst in instances))
    problems: list[str] = []
    missing_support: list[str] = []       # appends without a sampling support (their episodes are dropped: no_support)
    contract: list[str] = []              # session resume contract (ledger.resume_violations): a fork bug, raised below
    versions = collections.Counter()
    temperatures = collections.Counter()
    routing_positions = collections.Counter()
    routed = []                           # (inst, seg, episode) whose routing status is computed (pool below)
    for inst in instances:
        by_sid: dict[str, list] = collections.defaultdict(list)
        thread_last: dict[str, int] = {}
        for seg in sorted(inst.segments.values(), key=lambda x: x.seg):
            if seg.sid is not None:
                by_sid[seg.sid].append(seg)
                thread_last[seg.thread] = seg.seg
        for seg in inst.segments.values():
            stats["segments"] += 1
            if not seg.appends:
                stats["empty_segments"] += 1
                continue
            hit = joiner.lookup(seg)
            if hit is None:
                stats["unjoined_segments"] += 1
                continue
            if hit.episode in rid_dropped:          # dropped whole (rid_join): its segments are neither checked nor batched
                stats["rid_join_dropped_segments"] += 1
                continue
            if seg.sid is not None and post_close_tail(seg, by_sid[seg.sid], thread_last[seg.thread]):
                stats["post_close_tail_segments"] += 1   # only this segment goes; the episode keeps its others
                continue
            stats["joined_segments"] += 1
            contract += resume_violations(seg)
            if seg.stray_media_pads:                     # sampled pad tokens: no media to feed (ledger Segment)
                episodes[hit.episode].stray_media_pads += len(seg.stray_media_pads)
                stats["stray_media_pad_segments"] += 1
            if seg.window != 0:
                if not episodes[hit.episode].compacted:
                    raise RuntimeError(f"{seg.key}: sid {seg.sid} is window {seg.window}, but the manifest does not record {hit.episode}"
                                       f"'s compaction")
                stats["compacted_window_segments"] += 1
                if not args.allow_compaction:
                    continue                  # the episode is dropped as compacted below
            if episodes[hit.episode].wrapup_post_rids:      # --wrapup-loss mask: before anything reads gen_spans
                ep = episodes[hit.episode]
                n_tok, n_rid = mask_wrapup_spans(seg, ep.wrapup_post_rids)
                ep.wrapup_masked_tokens += n_tok
                ep.wrapup_masked_rids += n_rid
                stats["wrapup_masked_tokens"] += n_tok
                stats["wrapup_masked_rids"] += n_rid
            if not seg.gen_spans:
                stats["no_gen_segments"] += 1
                continue
            ep = episodes[hit.episode]
            problems += segment_problems(seg, args.temperature, args.top_p, top_k_rollout)
            if miss := support_missing(seg, args.top_p, top_k_rollout):
                missing_support += miss
                ep.no_support += len(miss)
            temperatures.update(a.sampling.get("temperature") for a in seg.appends if a.gen_len)
            seg_versions = segment_versions(seg)
            versions.update(seg_versions)
            ep.gen_versions.update(seg_versions)
            ep.unrecorded += seg_versions[UNRECORDED]
            if problems:
                continue                      # raised below, after every segment is listed
            if ep.no_support:
                continue                      # the episode is dropped as no_support below
            behavior_logprobs(seg, replay)    # alignment assertions (recomputed when writing)
            if replay:
                support_arrays(seg)
            routed.append((inst, seg, ep))
    # routing status of every segment before the first problem, in segment order (the first X raises, as inline did)
    with ptm.FailLoudPool(mp_ctx.get_context("spawn"), args.routing_workers, _no_init, ()) as pool:
        results = pool.imap(_routing_task, [(str(inst.path), seg, layers, top_k, num_experts, None, 0)
                                            for inst, seg, _ in routed])
        for (inst, seg, ep), (status, reasons) in zip(routed, results, strict=True):
            counts = np.bincount(status, minlength=max(ROUTING_NAMES) + 1)
            if len(counts) != len(ROUTING_NAMES):
                raise RuntimeError(f"{seg.key}: routing status has unknown values {np.unique(status).tolist()}")
            for code, name in ROUTING_NAMES.items():
                routing_positions[name] += int(counts[code])
            if counts[ROUTING_X]:
                raise RuntimeError(f"{seg.key} ({ep.key}): invalid routing (X): {reasons}")
            if counts[ROUTING_M]:
                ep.m_reasons.update(reasons)          # only M reasons are left: X raised above
            for code in (ROUTING_D_SAME, ROUTING_D_UNKNOWN, ROUTING_D_CROSS):
                if counts[code]:
                    ep.d_positions[ROUTING_NAMES[code]] += int(counts[code])
            if len(seg) > args.bin_tokens:
                ep.too_long = True
            if ep.d_enforced is None:
                ep.d_enforced = seg.exact_resume
            elif ep.d_enforced != seg.exact_resume:
                raise RuntimeError(f"{ep.key}: segments come from servers with session_exact_resume on and off ({seg.key} is "
                                   f"{seg.exact_resume}, earlier segments {ep.d_enforced}): the forward-twice rules cannot be enforced on half")
            ep.segments.append((inst, seg))
    if contract:
        raise RuntimeError(f"session resume contract violated {len(contract)} times (fork bug), e.g.:\n  " + "\n  ".join(contract[:8]))
    if problems:
        raise RuntimeError(f"{len(problems)} SPEC §2 violations, e.g.:\n  " + "\n  ".join(problems[:8]))
    if missing_support:
        print(f"WARN {len(missing_support)} appends without a sampling support; their episodes are dropped as no_support"
              f" ({sorted({ep.key for ep in episodes.values() if ep.no_support})[:8]}); e.g.:\n  "
              + "\n  ".join(missing_support[:8]), file=sys.stderr, flush=True)
    if windowed:
        check_window(versions, args.current_version)
    else:
        check_versions(versions, allowed)
    timings["ledger_seconds"] = time.time() - t0

    # ---- 3. episodes and groups
    drop_stops = {s for s in args.drop_stops.split(",") if s}
    for ep in episodes.values():
        ep.segments.sort(key=lambda x: (x[0].name, x[1].seg))
        ep.gen_tokens = sum(seg.n_gen for _, seg in ep.segments)
        if ep.key in rid_dropped:
            ep.drop = "rid_join"
        elif ep.reward is None:
            ep.drop = "harness_failed"
        elif ep.stop in drop_stops:
            ep.drop = "infra_stop"
        elif ep.no_support:
            ep.drop = "no_support"
        elif ep.stray_media_pads:
            ep.drop = "stray_media_pad"
        elif ep.compacted and not args.allow_compaction:
            ep.drop = "compacted"
        elif ep.context_overflows:
            ep.drop = "context_overflow"
        elif not ep.segments:
            ep.drop = "no_tokens"
        elif ep.m_reasons:
            ep.drop = "routing_M"
        elif ep.unrecorded:
            ep.drop = "weight_version_unrecorded"
        elif ep.d_positions["D_cross"] and ep.d_enforced:
            ep.drop = "routing_D_cross"
        elif windowed and oldest_version(ep.gen_versions) < int(allowed[0]):
            ep.drop = "stale"
        elif ep.too_long:
            ep.drop = "too_long"
    with_tokens = [ep for ep in episodes.values() if ep.segments]
    m_dropped = [ep for ep in with_tokens if ep.m_reasons]
    unrecorded = [ep for ep in with_tokens if ep.unrecorded]
    # the forward-twice rules (drop D_cross, its gate, the D_unknown raise) apply only to episodes whose server ran
    # --session-exact-resume; without it (the old fork, or the flag off) they are only counted and written as R --
    # what prepare did before the rules existed (SPEC §5)
    d_cross = [ep for ep in with_tokens if ep.d_positions["D_cross"] and ep.d_enforced]
    d_cross_unenforced = [ep for ep in with_tokens if ep.d_positions["D_cross"] and not ep.d_enforced]
    if d_cross_unenforced:
        print(f"WARN {len(d_cross_unenforced)} episodes have positions forwarded twice across weight versions (D_cross), "
              f"{sum(ep.d_positions['D_cross'] for ep in d_cross_unenforced)} positions, but the ledger's server did not enable "
              f"session_exact_resume: only counted, kept, written as R (e.g. {[ep.key for ep in d_cross_unenforced[:3]]})",
              file=sys.stderr, flush=True)
    if allowed != [UNVERSIONED]:
        # with versions, D_unknown is only expected where an append's weight_versions is null: those episodes are
        # dropped as weight_version_unrecorded. Anything else is a version this batch cannot see.
        lost = [ep.key for ep in with_tokens if ep.d_positions["D_unknown"] and not ep.unrecorded and ep.d_enforced]
        if lost:
            raise RuntimeError(f"{len(lost)} episodes have positions forwarded twice with an unknown version (D_unknown) not caused by "
                               f"unrecorded weight_versions: {lost[:5]} (appends that generated 0 tokens lack a version key?)")
    reforward = {f"{name}_{unit}": n for name in D_CLASSES for unit, n in (
        ("positions", routing_positions[name]), ("episodes", sum(bool(ep.d_positions[name]) for ep in with_tokens)))}
    reforward["D_cross_unenforced_episodes"] = len(d_cross_unenforced)
    # the counts behind the fraction gates; --defer-fraction-gates leaves the gates to the caller (orchestrator
    # stream.group_ready sums them over every packet of one optimizer step, the unit the gates were defined on)
    # post_close_tail_segments is not a gate: the trailing segments post_close_tail dropped (their episodes kept), here so
    # that batches.jsonl group_ready.quality shows it per batch
    quality = {"with_tokens": len(with_tokens), "routing_M": len(m_dropped), "weight_version_unrecorded": len(unrecorded),
               "post_close_tail_segments": stats["post_close_tail_segments"],
               "routing_D_cross": len(d_cross)}
    if not args.defer_fraction_gates and len(m_dropped) > args.max_routing_m_fraction * len(with_tokens):
        raise RuntimeError(f"routing M dropped {len(m_dropped)}/{len(with_tokens)} episodes"
                           f" (> --max-routing-m-fraction {args.max_routing_m_fraction:.1%}): "
                           f"{sum((ep.m_reasons for ep in m_dropped), collections.Counter())}")
    if not args.defer_fraction_gates and len(unrecorded) > MAX_UNRECORDED_FRACTION * len(with_tokens):
        raise RuntimeError(f"unrecorded weight_versions (null) dropped {len(unrecorded)}/{len(with_tokens)} episodes"
                           f" (> {MAX_UNRECORDED_FRACTION:.1%})")
    if not args.defer_fraction_gates and len(d_cross) > args.max_d_cross_fraction * len(with_tokens):
        raise RuntimeError(f"routing D_cross (positions forwarded twice across weight versions) dropped {len(d_cross)}/{len(with_tokens)} "
                           f"episodes (> --max-d-cross-fraction {args.max_d_cross_fraction:.1%}), e.g. "
                           f"{[ep.key for ep in d_cross[:3]]}; for the recompute windows see the audit report's window")
    by_group: dict[str, list[Episode]] = collections.defaultdict(list)
    for ep in episodes.values():
        by_group[ep.group].append(ep)
    kept: dict[str, list[Episode]] = {}
    group_drop = collections.Counter()
    eff_stats = collections.Counter()     # groups with a nonzero efficiency term / all correct / guard bound
    eff_abs: list[float] = []
    tool_stats = collections.Counter()    # --tool-*: groups considered / with a nonzero term / whose only signal it is
    tool_abs: list[float] = []
    tool_means: dict[str, list[float]] = {"q_mean": [], "p_mean": [], "u_mean": []}
    arc3_stats = collections.Counter()    # --arc3-reward eff / c0: group counts, clip / guard hits (arc3_reward_summary)
    arc3_hist = collections.Counter()     # p̂ of the groups that reached the advantage
    # Σ|c − p̂| and Σ|eff| of the kept groups, also weighted by L_i; c0 also the two parts of eff
    arc3_sums = dict.fromkeys(ARC3_SHARE_SUMS + ARC3_C0_SUMS, 0.0)
    for group in sorted(by_group):
        live = sorted((ep for ep in by_group[group] if ep.drop is None), key=lambda e: e.key)
        if len(live) < 2:
            group_drop["lt2_episodes"] += 1
            for ep in live:
                ep.drop = "group_lt2"
            continue
        arc3_info = None
        if arc3_mode is not None:
            group_args = ([ep.reward for ep in live], [ep.correct for ep in live], [ep.actions for ep in live],
                          [ep.baseline_actions if arc3_mode == "official" else ep.tokens_at_clear for ep in live])
            if arc3_mode == "note":
                shaped, arc3_info = arc3_note_rewards([ep.stop for ep in live], [ep.correct for ep in live],
                                                      [ep.actions for ep in live], [ep.baseline_actions for ep in live],
                                                      [ep.window_cause for ep in live], [ep.judged for ep in live])
            elif arc3_mode == "eff":
                shaped, arc3_info = arc3_eff_rewards(*group_args, beta_actions=args.arc3_beta_actions,
                                                     beta_tokens=args.arc3_beta_tokens, w_start=args.arc3_w_start,
                                                     clip=args.arc3_eff_clip)
            elif arc3_mode == "c0":
                shaped, arc3_info = arc3_c0_rewards(*group_args, **arc3_params)
            elif arc3_mode == "official":
                shaped, arc3_info = arc3_official_rewards(*group_args, **arc3_params)
            else:
                launched = {ep.launch_version for ep in live}
                if len(launched) != 1:
                    raise RuntimeError(f"group {group}: its episodes have launch versions {sorted(launched, key=str)}, "
                                       "want one (one manifest per group)")
                if arc3_mode == "rank":
                    shaped, arc3_info = arc3_rank_rewards(*group_args, **arc3_params, launch_version=launched.pop())
                else:
                    run_name, tid = group.split(":", 1)
                    gm = GROUP_RUN.match(run_name)
                    if gm is None or {ep.task for ep in live} != {tid}:
                        raise RuntimeError(f"group {group}: hinge needs the orchestrator's g<NNNNNN> group runs of one task")
                    shaped, arc3_info = arc3_hinge_rewards(*group_args, **arc3_params, task=tid, gid=int(gm["gid"]),
                                                           launch_version=launched.pop(), refs=hinge_refs)
            adv = group_advantages(shaped, "none")
            arc3_stats["groups_considered"] += 1
            arc3_stats["all_cleared_groups"] += arc3_info["cleared"] == len(live)
            arc3_stats["groups_eff_nonzero"] += bool(arc3_info["eff"].any())
            if arc3_mode == "eff":
                arc3_stats["clip_hits"] += arc3_info["clip_hits"] if arc3_info["w"] > 0 else 0   # w = 0: the clip changes nothing
            elif arc3_mode == "c0":
                arc3_stats["guard_hits"] += arc3_info["guard_scale"] < 1
            elif arc3_mode == "official":
                arc3_stats["full_groups"] += arc3_info["full"] == len(live)
            elif arc3_mode == "rank":
                arc3_stats["ranked_groups"] += arc3_info["ranked"]
                arc3_stats["warmup_groups"] += arc3_info["warmup"]
            elif arc3_mode == "note":
                arc3_stats["window_episodes"] += arc3_info["window"]
                arc3_stats["early_window_episodes"] += arc3_info["early"]
            else:
                arc3_stats["hinged_groups"] += arc3_info["hinged"]
                arc3_stats["unfrozen_groups"] += not arc3_info["hinged"]
            arc3_hist[f"{arc3_info['p_hat']:.4f}"] += 1
        else:
            shaped = None if efficiency or tool_on else [ep.reward for ep in live]
            adv = group_advantages([ep.reward for ep in live], args.adv_normalization)
        if efficiency:
            eff, info = efficiency_advantages([ep.reward for ep in live], [ep.gen_tokens for ep in live],
                                              beta=args.efficiency_beta, budget=args.efficiency_budget,
                                              guard=args.efficiency_guard)
            if eff.any():
                eff_stats["groups"] += 1
                eff_stats["all_correct_groups"] += info["correct"] == len(live)
                eff_stats["guard_bound"] += info["bound"]
                eff_abs.extend(np.abs(eff[eff != 0]).tolist())
                adv = eff if adv is None else adv + eff
        if tool_on:
            a_tool, tinfo = tool_advantages([ep.tool_use for ep in live], **tool_params)
            tool_stats["groups"] += 1
            for k in ("q_mean", "p_mean", "u_mean"):
                tool_means[k].append(tinfo[k])
            if a_tool.any():
                tool_stats["groups_nonzero"] += 1
                tool_stats["groups_only_tool"] += adv is None
                tool_abs.extend(np.abs(a_tool).tolist())
                adv = a_tool if adv is None else adv + a_tool
        if adv is None:
            group_drop["std0"] += 1
            if len({ep.reward for ep in by_group[group] if ep.reward is not None}) > 1:
                group_drop["std0_after_drop"] += 1          # the episode-level drops removed its only signal
            for ep in live:
                ep.drop = "group_std0"
            continue
        for i, (ep, a) in enumerate(zip(live, adv)):
            ep.advantage = float(a)
            ep.training_reward = None if shaped is None else float(shaped[i])
        kept[group] = live
        if arc3_info is not None:
            arc3_stats["groups_kept"] += 1
            arc3_stats["all_cleared_groups_kept"] += arc3_info["cleared"] == len(live)
            loss_tokens = np.array([ep.gen_tokens for ep in live], dtype=np.float64)
            for name, part in (("task", np.abs(arc3_info["task"])), ("eff", np.abs(arc3_info["eff"]))):
                arc3_sums[f"{name}_abs"] += float(part.sum())
                arc3_sums[f"{name}_abs_loss_tokens"] += float((loss_tokens * part).sum())
            if arc3_mode == "c0":
                arc3_sums["eff_actions_abs"] += float(np.abs(arc3_info["eff_actions"]).sum())
                arc3_sums["eff_tokens_abs"] += float(np.abs(arc3_info["eff_tokens"]).sum())
            if arc3_mode == "official":
                arc3_stats["cleared_zero_episodes"] += arc3_info["cleared_zero"]
                arc3_stats["partial_episodes"] += arc3_info["partial"]
    arc3_summary = None if arc3_mode is None else arc3_reward_summary(arc3_mode, arc3_params, arc3_stats, arc3_hist,
                                                                      arc3_sums)
    if arc3_mode == "hinge":
        arc3_summary["hinge_refs_snapshot"] = {"sha256": hinge_refs_sha256, "levels": len(hinge_refs)}
    if arc3_mode == "note":
        arc3_summary["judge_files"] = judge_files
    if not kept:
        if not windowed:
            raise RuntimeError("no group left (all std 0 or all dropped)")
        summary = {"status": "EMPTY_ELIGIBLE", "out": str(out), "round": args.round,
                   "current_version": args.current_version, "max_staleness": args.max_staleness,
                   "episodes": len(episodes), "episodes_kept": 0, "groups": len(by_group), "groups_kept": 0, "quality": quality,
                   "reforward": reforward,
                   "dropped_episodes": dict(collections.Counter(ep.drop for ep in episodes.values())),
                   "dropped_groups": dict(group_drop), "arc3_reward": arc3_summary,
                   "groups_detail": {g: [{"episode": ep.key, "reason": ep.drop} for ep in eps]
                                     for g, eps in sorted(by_group.items())},
                   "dropped_detail": dropped_detail(episodes), "dropped_reward_sum": dropped_reward_sum(episodes),
                   "groups_std0_after_drop": group_drop.get("std0_after_drop", 0),
                   "ledger_total_segments": stats["ledger_total_segments"], "joined_segments": stats["joined_segments"],
                   "ledger_read_bytes": stats["ledger_read_bytes"], "prepare_ledger_seconds": round(timings["ledger_seconds"], 3),
                   "seconds": round(time.time() - t_start, 1)}
        out.mkdir(parents=True)
        (out / "report.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return summary

    # ---- 4. per-token weight versions, minibatches, coef, bins
    versions_of: dict[str, np.ndarray] = {}
    for eps in kept.values():
        for ep in eps:
            counts = collections.Counter()
            for _, seg in ep.segments:
                v = versions_of[seg.key] = token_versions(seg, allowed)
                vals, cnts = np.unique(v[v >= 0], return_counts=True)
                counts.update(dict(zip(vals.tolist(), cnts.tolist())))
            ep.version_tokens = ({str(k): int(c) for k, c in sorted(counts.items())} if allowed != [UNVERSIONED]
                                 else {UNVERSIONED: ep.gen_tokens})
            if sum(ep.version_tokens.values()) != ep.gen_tokens:
                raise RuntimeError(f"{ep.key}: per-version token counts {ep.version_tokens} do not add up to L={ep.gen_tokens}")
    staleness = dict.fromkeys(STALENESS_KEYS)
    if windowed:
        present = sorted({v for eps in kept.values() for ep in eps for v in ep.version_tokens}, key=int)
        # same rule as --weight-version: one version is the scalar string, several the sorted list (RLBatch, trainer)
        weight_version = present[0] if len(present) == 1 else present
        staleness = staleness_histograms([ep for eps in kept.values() for ep in eps],
                                         [ep for ep in episodes.values() if ep.drop == "stale"], args.current_version)
    members = split_minibatches({g: sum(len(seg) for ep in eps for _, seg in ep.segments) for g, eps in kept.items()},
                                args.minibatches, args.seed)
    samples: list[Sample] = []
    plan = []
    for b, groups in enumerate(members):
        eps = sorted((ep for g in groups for ep in kept[g]), key=lambda e: e.key)
        n_ep = len(eps)
        denominator = coef_denominators(eps, args.loss_reduction)
        for ep in eps:
            ep.minibatch = b
            ep.coef = float(np.float32(ep.advantage / denominator[ep.key]))
        group_tokens = collections.Counter()
        for ep in eps:
            group_tokens[ep.group] += ep.gen_tokens
        local = [Sample(key=seg.key, inst=inst, seg=seg, episode=ep, minibatch=b)
                 for ep in eps for inst, seg in ep.segments]
        lengths = [len(s.seg) for s in local]
        bins = lpt_bins(lengths, args.bin_tokens, args.world)
        shortest = min(range(len(local)), key=lambda i: (lengths[i], i))
        n_pad = 0
        for j, x in enumerate(bins):
            if not x:
                src = local[shortest]
                local.append(Sample(key=f"{src.seg.key}#pad{n_pad}", inst=src.inst, seg=src.seg, episode=None,
                                    minibatch=b, pad_of=src.seg.key))
                lengths.append(lengths[shortest])
                bins[j] = [len(local) - 1]
                n_pad += 1
        steps_local = bins_to_steps(bins, lengths, args.world)
        steps = []
        for step in steps_local:
            ranks = []
            for bin_ in step:
                ranks.append([])
                for i in bin_:
                    ranks[-1].append(len(samples))
                    samples.append(local[i])
            steps.append(ranks)
        sum_coef = sum(ep.advantage * ep.gen_tokens / denominator[ep.key] for ep in eps)
        plan.append({"steps": steps, "episodes": n_ep, "groups": sorted(groups), "samples": len(local) - n_pad,
                     "pad_samples": n_pad, "tokens": int(sum(lengths)),
                     "loss_tokens": int(sum(s.seg.n_gen for s in local if s.pad_of is None)),
                     "group_tokens": {g: int(group_tokens[g]) for g in sorted(groups)},
                     "sum_coef_expected": sum_coef,
                     "max_bin_tokens": int(max(sum(lengths[i] for i in bin_) for step in steps_local for bin_ in step))})
    stats["samples"] = len(samples)
    stats["pad_samples"] = sum(s.pad_of is not None for s in samples)
    n_tokens = sum(len(s.seg) for s in samples)
    routing_bytes = n_tokens * layers * top_k * 2
    if routing_bytes > args.max_gb * 1e9:
        raise RoutingOverMaxGB(routing_bytes, args.max_gb)
    if shutil.disk_usage(out.parent if out.parent.exists() else Path.cwd()).free < required_bytes:
        raise RuntimeError(f"{out.parent} has less free space than required_free_bytes={required_bytes} "
                           f"(--max-gb {args.max_gb} + --reserve-gb {args.reserve_gb}, decimal GB)")
    out.mkdir(parents=True)
    (out / SPLIT).mkdir()
    print(f"episodes {len(episodes)} (kept {sum(len(v) for v in kept.values())}), groups {len(by_group)} (kept {len(kept)}), "
          f"samples {len(samples)} (pad {stats['pad_samples']}), {n_tokens:,} tokens ({time.time() - t_start:.0f}s); "
          f"positions forwarded twice {reforward}", flush=True)

    # ---- 5. media and M-RoPE (once per real sample; pad samples reuse the original's result)
    t0 = time.time()
    real = {s.seg.key: s for s in samples if s.pad_of is None}
    media = {k: ptm.segment_media(s.seg) for k, s in real.items()}
    checks = collections.Counter()
    records: dict[str, dict] = {}
    computed: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    env = None
    media_src = {}
    for k, s in real.items():
        for m in media[k]:
            media_src.setdefault(m["sha256"], (m["modality"], str(s.inst.media_path(m["sha256"]))))
            if media_src[m["sha256"]][0] != m["modality"]:
                raise RuntimeError(f"{m['sha256']} is both image and video")
    if media_src:
        if args.media_device is None:
            raise RuntimeError(f"the batch has {len(media_src)} media: give --media-device (run via prepare_rl.sh, needs sglang)")
        visible = ptm.check_device(args.media_device)
        env = ptm._environment(model_dir, [d / "server.json" for d in dirs], args.media_device)
        # the processor must be the same sglang tree as the collecting server (ARC_SGLANG_FORK must point to the fork the
        # rollout server used)
        for d in dirs:
            served = Path(json.loads((d / "server.json").read_text())["sglang_file"]).resolve().parents[2]
            if served != Path(env["fork"]).resolve():
                raise RuntimeError(f"{d}: the server used the sglang at {served}, the media processor here is {env['fork']}"
                                   f" (set ARC_SGLANG_FORK={served} before prepare_rl.sh)")
        print(f"fork {env['fork']} commit={env['fork_commit']} dirty={env['fork_dirty']}  device {args.media_device}"
              f" (CUDA_VISIBLE_DEVICES={visible!r})", flush=True)
        ctx = mp_ctx.get_context("spawn")
        with ptm.FailLoudPool(ctx, args.workers, ptm._init_worker,
                              (str(dirs[0] / "server.json"), 1, args.media_device)) as pool:
            records = ptm.process_media(pool, media_src, out, checks)
            tasks = [ptm.segment_task(k, s.seg.ids.astype("<i4"), [(sp.start, sp.end) for sp in s.seg.gen_spans],
                                      media[k], records) for k, s in real.items()]
            for key, ids, loss, pos, seg_checks, _ in pool.imap(ptm._segment_task, tasks, chunksize=2):
                if not np.array_equal(ids, real[key].seg.ids):
                    raise RuntimeError(f"{key}: ids returned by the worker differ from the ledger")
                checks.update(seg_checks)
                computed[key] = (loss, pos)
        resized = sorted(sha for sha, r in records.items() if r["bicubic_calls"])
        if args.media_device == "cpu-emulated" and resized and not args.allow_emulated_media:
            raise RuntimeError(f"cpu-emulated: {len(resized)} media went through bicubic resizing, pixels may differ from deployment by ±1"
                               f" (use --media-device cuda, or knowingly add --allow-emulated-media)")
        with open(out / "media.jsonl", "w") as f:
            for sha in sorted(records):
                f.write(json.dumps(records[sha]) + "\n")
    else:
        for k, s in real.items():
            ids = s.seg.ids
            if ((ids == vocab.image_pad) | (ids == vocab.video_pad)).any():
                raise RuntimeError(f"{k}: media pads but no media records")
            loss = ptm.loss_mask_from_spans(len(ids), [(sp.start, sp.end) for sp in s.seg.gen_spans])
            computed[k] = (loss, np.broadcast_to(np.arange(len(ids), dtype="<i4"), (3, len(ids))))
            checks["mrope_text_only_is_arange"] += 1
        (out / "media.jsonl").write_text("")
    timings["media_mrope_seconds"] = time.time() - t0

    # ---- 6. write
    t0 = time.time()
    names = {"ids": "input_ids.i32", "loss": "loss_mask.u8", "pos": "mrope.i32", "routing": "routing.i16",
             "status": "routing_status.u8", "behavior": "behavior_logprob.f32", "coef": "coef.f32",
             "version": "weight_version.i32"}
    if replay:
        names.update(support_len="support_len.u16", support_ids="support_ids.i32",
                     behavior_full="behavior_logprob_full.f32")
    support_offset = 0
    routing_path = out / SPLIT / names.pop("routing")
    starts = np.concatenate([[0], np.cumsum([len(s.seg) for s in samples])]).tolist()
    with open(routing_path, "wb") as f:           # workers pwrite each sample's rows at its offset
        f.truncate(starts[-1] * layers * top_k * 2)
    files = {k: open(out / SPLIT / v, "wb") for k, v in names.items()}
    offset = 0
    rows = []
    with open(out / SPLIT / "index.jsonl", "w") as index, \
            ptm.FailLoudPool(mp_ctx.get_context("spawn"), args.routing_workers, _no_init, ()) as pool:
        written = pool.imap(_routing_task, [(str(s.inst.path), s.seg, layers, top_k, num_experts, str(routing_path), start)
                                            for s, start in zip(samples, starts)])
        for s, (status, reasons) in zip(samples, written, strict=True):
            seg = s.seg
            loss, pos = computed[seg.key]
            n = len(seg)
            twice = {name: int((status == code).sum()) for name, code in
                     (("D_same", ROUTING_D_SAME), ("D_unknown", ROUTING_D_UNKNOWN), ("D_cross", ROUTING_D_CROSS))}
            if seg.exact_resume and (twice["D_cross"] or (twice["D_unknown"] and allowed != [UNVERSIONED])):
                raise RuntimeError(f"{seg.key}: still {twice} at write time (the episode should have been dropped or raised)")
            status = np.where(np.isin(status, (ROUTING_D_SAME, ROUTING_D_UNKNOWN, ROUTING_D_CROSS)), ROUTING_R,
                              status).astype(np.uint8)
            if (status[:n - 1] != ROUTING_R).any() or status[n - 1] != ROUTING_N:
                raise RuntimeError(f"{seg.key}: routing status at write time is not all R + N at the segment end: {reasons}")
            behavior = behavior_logprobs(seg, replay)
            mask = loss.astype(bool)
            if not np.array_equal(np.isfinite(behavior), mask):
                raise RuntimeError(f"{seg.key}: valid positions of the behavior logprob differ from the loss mask")
            coef = np.zeros(n, np.float32)
            if s.pad_of is None:
                coef[mask] = np.float32(s.episode.coef)
            files["ids"].write(seg.ids.astype("<i4").tobytes())
            files["loss"].write(loss.astype(np.uint8).tobytes())
            files["pos"].write(np.ascontiguousarray(pos, dtype="<i4").tobytes())
            files["status"].write(status.astype(np.uint8).tobytes())
            files["behavior"].write(behavior.astype("<f4").tobytes())
            files["coef"].write(coef.astype("<f4").tobytes())
            files["version"].write(versions_of[seg.key].astype("<i4").tobytes())
            row = _index_row(s, media[seg.key], weight_version, harness)
            row.update(offset=offset, n_loss=int(mask.sum()), routing_D_same=twice["D_same"],
                       routing_D_unknown=twice["D_unknown"], routing_D_cross=twice["D_cross"])
            if replay:
                sup_lens, sup_ids, full = support_arrays(seg)
                if not np.array_equal(sup_lens > 0, mask):
                    raise RuntimeError(f"{seg.key}: positions with a support differ from the loss mask")
                files["support_len"].write(sup_lens.astype("<u2").tobytes())
                files["support_ids"].write(sup_ids.astype("<i4").tobytes())
                files["behavior_full"].write(full.astype("<f4").tobytes())
                row.update(support_offset=support_offset, support_tokens=int(len(sup_ids)))
                support_offset += len(sup_ids)
            index.write(json.dumps(row, ensure_ascii=False) + "\n")
            rows.append(row)
            offset += n
    for fh in files.values():
        fh.close()
    timings["write_seconds"] = time.time() - t0

    # ---- 7. manifest / report
    kept_eps = [ep for eps in kept.values() for ep in eps]
    rewarded = [ep.reward for ep in episodes.values() if ep.reward is not None]
    correct_eps = [ep for ep in kept_eps if ep.reward == 1.0] if efficiency else []
    eff_summary = {"groups": eff_stats["groups"], "all_correct_groups": eff_stats["all_correct_groups"],
                   "guard_bound": eff_stats["guard_bound"],
                   "abs_mean": float(np.mean(eff_abs)) if eff_abs else 0.0,
                   # L (ledger) next to the proxy count: the difference is the tokens codex resent and discarded, never in
                   # the trajectory
                   "correct_gen_tokens_mean": float(np.mean([ep.gen_tokens for ep in correct_eps])) if correct_eps else None,
                   "correct_output_tokens_mean": (float(np.mean([ep.output_tokens for ep in correct_eps]))
                                                  if correct_eps else None)}
    ep_drop = collections.Counter(ep.drop for ep in episodes.values() if ep.drop is not None)
    real_rows = [r for r in rows if not r["pad"]]
    split_stats = {"samples": len(rows), "tokens": offset, "loss_tokens": sum(r["n_loss"] for r in rows),
                   "pad_samples": len(rows) - len(real_rows),
                   "real_tokens": sum(r["length"] for r in real_rows),
                   "real_loss_tokens": sum(r["n_loss"] for r in real_rows),
                   "groups": len(kept), "with_images": sum(any(m["modality"] == "image" for m in r["media"]) for r in rows),
                   "with_videos": sum(any(m["modality"] == "video" for m in r["media"]) for r in rows)}
    resized = sorted(sha for sha, r in records.items() if r["bicubic_calls"])
    # The descriptive text values in rl / manifest below (group_key, advantage, efficiency.rule, tool_reward.rule, coef,
    # pad, behavior_logprob, support, weight_version_file, loss_mask, mrope, routing.rows / status / source) are recorded
    # batch metadata, kept verbatim as in existing batches; they restate the module docstring (§1, §4-§7) and rl_batch.py.
    rl = {
        "round": args.round, "weight_version": weight_version, "temperature": args.temperature, "algo": args.algo,
        "top_p": args.top_p, "top_k": top_k_rollout, "replay": "mask" if replay else "none",
        **algo_params,
        "world": args.world, "bin_tokens": args.bin_tokens, "seed": args.seed,
        "episodes": len(kept_eps), "groups": len(kept), "episodes_total": len(episodes), "groups_total": len(by_group),
        "dropped": {"episodes": dict(ep_drop), "groups": dict(group_drop),
                    "routing_M_reasons": dict(sum((ep.m_reasons for ep in episodes.values()), collections.Counter()))},
        "reward_mean": float(np.mean(rewarded)) if rewarded else None,
        "reward_mean_kept": float(np.mean([ep.reward for ep in kept_eps])),
        "live_group_fraction": len(kept) / len(by_group),
        "join": join_mode,
        "group_key": "classic：task（item id，同一題跨 run）；group：<group run 目錄名>:<tid>（一份 manifest 一個 group）",
        "current_version": args.current_version, "max_staleness": args.max_staleness, **staleness,
        "adv_normalization": args.adv_normalization, "loss_reduction": args.loss_reduction,
        "advantage": ("A_i = (R_i - mean_g) / std_g，母體 std" if args.adv_normalization == "std" else
                      "A_i = R_i - mean_g（不除 std）") + "，g＝同一個 group 留下的 episode（切 minibatch 之前）"
                     + ("；加效率項 efficiency_advantages（efficiency 欄）" if efficiency else "")
                     + ("；加工具使用項 arc2_tooluse.tool_advantages（tool_reward 欄）" if tool_on else "")
                     + (f"；R 換成 arc3_{arc3_mode}_rewards 的組內相對過關＋效率（arc3_reward 欄；index 的 training_reward）"
                        if arc3_mode is not None else ""),
        "efficiency": None if not efficiency else {
            "beta": args.efficiency_beta, "budget": args.efficiency_budget, "guard": args.efficiency_guard,
            "rule": "全對（R=1）之間 A_eff = beta_g (e_i - mean e)，e = 1 - gen_tokens/budget（帳本的生成 token）；全對組 beta_g = beta，"
                    "混合組 beta_g = min(beta, guard (1 - mean R) / max(mean e - e_i))",
            **eff_summary},
        "arc3_reward": arc3_summary,
        "tool_reward": None if not tool_on else {
            **tool_params,
            "rule": "A_tool = q_i - mean_g q，q = -hand_weight min(hand_chars/hand_scale, hand_cap) + toolkit_weight u，"
                    "u = toolkit_cmds/code_cmds（沒有寫程式 0）；加在任務 advantage（與效率項）上（arcforge.arc2_tooluse）",
            "groups": tool_stats["groups"], "groups_nonzero": tool_stats["groups_nonzero"],
            "groups_only_tool": tool_stats["groups_only_tool"],
            "abs_mean": float(np.mean(tool_abs)) if tool_abs else 0.0,
            **{k: float(np.mean(v)) if v else None for k, v in tool_means.items()}},
        "coef": ("loss 位置 coef = A_i / (E * L_i)：E＝minibatch 的 episode 數，L_i＝episode i 的 loss token 數"
                 if args.loss_reduction == "sequence" else
                 "loss 位置 coef = A_i / (H * T_g)：H＝minibatch 的 group 數，T_g＝episode i 所在 group 的 loss token 總數"
                 "（minibatches[m].group_tokens）"),
        "pad": "補位樣本＝該 minibatch 最短樣本的複製，coef 全 0（index pad=true）",
        "behavior_logprob": ("位置 p＝取樣出 input_ids[p] 的那一筆 append 記的 support 上重新正規化的 logprob（溫度、"
                             "top-k／top-p 之後；全詞表的截斷前 logprob 在 behavior_logprob_full.f32）" if replay else
                             "位置 p＝取樣出 input_ids[p] 的那一筆 append 記的 logprob（溫度之後、top-p 之前）"),
        **({"support": "train/support_len.u16：loss 位置 p 取樣 input_ids[p] 的 support 大小（≥ 1），其餘 0；"
                       "train/support_ids.i32：各 loss 位置的 support id 依位置串接，樣本的起點＝index support_offset",
            "support_tokens": support_offset,
            "support_len_mean": (support_offset / max(sum(r["n_loss"] for r in rows), 1))} if replay else {}),
        "weight_version_tokens": dict(sum((collections.Counter(ep.version_tokens) for ep in kept_eps),
                                          collections.Counter())),
        "weight_version_file": "train/weight_version.i32：loss 位置＝取樣那個 token 時的權重版本（int），其餘 −1；"
                               "unversioned 整個 −1",
        "minibatches": plan,
        "rid_join": {"expected": rj["expected"], "budget_extra": len(rj["budget_extra"]),
                     "wrapup_cut_extra": len(rj["wrapup_cut_extra"]),
                     "aborted_not_in_manifest": len(rj["aborted_not_in_manifest"]),
                     "threads_not_in_manifest": len(rj["threads_not_in_manifest"])},
        "sampling": {"temperature_found": {str(k): v for k, v in temperatures.items()}},
        "routing_positions_joined": dict(routing_positions),
        "reforward": {**reforward, **{f"kept_{name}_positions": sum(r[f"routing_{name}"] for r in rows if not r["pad"])
                                      for name in D_CLASSES}},
        "max_d_cross_fraction": args.max_d_cross_fraction,
        "max_routing_m_fraction": args.max_routing_m_fraction,
        "reforward_rule": REFORWARD_RULE,
    }
    manifest = {
        "format": RL_FORMAT, "created_unix": time.time(), "argv": sys.argv, "tokenizer_sha256": vocab.sha256,
        "ledger": [str(d) for d in dirs], "manifests": [str(m.path) for m in manifests],
        "splits": {SPLIT: split_stats}, "vision": ptm.vision_spec(vocab),
        "processor_digest": None if env is None else hashlib.sha256(json.dumps(env, sort_keys=True).encode()).hexdigest(),
        "processor": env,
        "media": {"device": args.media_device if media_src else None, "unique": len(records),
                  "resized_by_bicubic": len(resized),
                  "deployment_exact": None if not media_src
                  else (args.media_device in ("cuda", "cpu-exact") or not resized)},
        "loss_mask": "帳本 Segment.gen_spans（真的取樣出來的 token；回捲尾端、模板補的 token、tool 輸出、prompt、pad 都不算）",
        "mrope": "sglang MRotaryEmbedding.get_rope_index（qwen4_exp），段內從 0 起；純文字＝arange；(3, T) int32",
        "routing": {"layers": layers, "top_k": top_k, "dtype": "int16", "num_experts": num_experts,
                    "layout": "(T, layers, top_k) 逐 token 串接，與 input_ids 同 offset",
                    "rows": "位置 p＝把 token p 當輸入時選到的 expert（全域 logical id）；段尾從沒 forward 過，填 -1",
                    "status": {"0": "R（含 forward 兩次的 D_same／D_unknown，以及 server 沒開 session_exact_resume 的 "
                                    "D_cross：第一次的 routing；見 rl.reforward_rule）",
                               "1": "N（只在段尾）"},
                    "source": "arcforge.tito.ledger.segment_routing_status（後一筆 append 蓋前一筆）"},
        "rl": rl,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")

    # ---- 8. read-back verification
    t0 = time.time()
    verified = verify_batch(out, samples, plan, layers=layers, top_k=top_k, num_experts=num_experts, allowed=allowed,
                            replay=replay)
    timings["verify_seconds"] = time.time() - t0
    timings["total_seconds"] = time.time() - t_start
    report = {"stats": dict(stats), "checks": dict(sorted(checks.items())), "timings": timings,
              "verified": verified, "media_resized": resized}
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    summary = {
        "out": str(out), "round": args.round, "weight_version": weight_version, "algo": args.algo,
        "adv_normalization": args.adv_normalization, "loss_reduction": args.loss_reduction,
        "allow_compaction": args.allow_compaction,
        "wrapup_loss": args.wrapup_loss,
        "wrapup_masked_tokens_kept": sum(ep.wrapup_masked_tokens for ep in kept_eps),
        "compacted_episodes_kept": sum(ep.compacted for ep in kept_eps),
        "segments_per_kept_episode_max": max((len(ep.segments) for ep in kept_eps), default=0),
        "efficiency": rl["efficiency"], "arc3_reward": rl["arc3_reward"], "tool_reward": rl["tool_reward"],
        "episodes": len(episodes), "episodes_kept": len(kept_eps), "groups": len(by_group), "groups_kept": len(kept),
        "live_group_fraction": round(len(kept) / len(by_group), 4), "dropped_episodes": dict(ep_drop),
        "dropped_detail": dropped_detail(episodes), "dropped_reward_sum": dropped_reward_sum(episodes),
        # groups whose rewards varied before the episode-level drops and are std 0 after them (10-24 arc2 g000006)
        "groups_std0_after_drop": group_drop.get("std0_after_drop", 0),
        "rid_join_dropped": {k: v[:3] for k, v in sorted(rid_dropped.items())},
        "dropped_groups": dict(group_drop), "reward_mean": rl["reward_mean"], "samples": len(rows),
        "pad_samples": stats["pad_samples"], "tokens": split_stats["real_tokens"],
        "loss_tokens": split_stats["real_loss_tokens"], "tokens_with_pad": offset,
        "minibatches": [{"episodes": p["episodes"], "steps": len(p["steps"]), "tokens": p["tokens"]} for p in plan],
        "routing_positions_joined": dict(routing_positions), "temperature_found": rl["sampling"]["temperature_found"],
        "quality": quality, "reforward": rl["reforward"],
        "weight_version_tokens": rl["weight_version_tokens"], "budget_extra_rids": len(rj["budget_extra"]),
        "wrapup_cut_rids": len(rj["wrapup_cut_extra"]), "unjoined_segments": stats["unjoined_segments"],
        "wrapup_loss": args.wrapup_loss, "wrapup_masked_tokens": stats["wrapup_masked_tokens"],
        "wrapup_masked_rids": stats["wrapup_masked_rids"],
        "media": manifest["media"], "join": join_mode, "current_version": args.current_version,
        "max_staleness": args.max_staleness, **staleness, "seconds": round(timings["total_seconds"], 1),
        "ledger_total_segments": stats["ledger_total_segments"], "joined_segments": stats["joined_segments"],
        "ledger_read_bytes": stats["ledger_read_bytes"], "prepare_ledger_seconds": round(timings["ledger_seconds"], 3),
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def _expected_versions(seg, allowed: list[str], skip_rids: frozenset = frozenset()) -> np.ndarray:
    """For read-back verification: re-derive per-token versions from the append records (versions expanded per gen index,
    not via gen_spans). ``skip_rids``: the wrap-up appends of --wrapup-loss mask (the outcome's wrapup_post_rids); their
    tokens are not loss and have version -1."""
    out = np.full(len(seg), -1, np.int32)
    if allowed == [UNVERSIONED]:
        return out
    for k, a in enumerate(seg.appends):
        rewound = seg.appends[k + 1].rewind if k + 1 < len(seg.appends) else 0
        m = a.gen_len - rewound
        if m <= 0 or a.rid in skip_rids:
            continue
        per = np.empty(a.gen_len, np.int32)
        for start, v in a.weight_versions:
            per[start:] = int(v)                  # runs are increasing: later runs overwrite the tail
        out[a.gen_start:a.gen_start + m] = per[:m]
    return out


def verify_batch(out: Path, samples: list[Sample], plan: list[dict], *, layers: int, top_k: int,
                 num_experts: int, allowed: list[str], replay: bool = False) -> dict:
    """``RLBatch`` reads back every row (the reader's own checks + the bit-for-bit comparison here). Routing is compared
    only for the 3 longest rows and every 16th row."""
    from arcforge.tito.ledger import segment_routing_status

    b = RLBatch(out)
    if len(b) != len(samples) or b.minibatches != [{k: v for k, v in p.items()} for p in plan]:
        raise RuntimeError("read-back sample count or minibatch plan differs from what was written")
    longest = set(np.argsort(-b.lengths)[:3].tolist())
    n_routing = 0
    coef_sum = collections.defaultdict(float)
    episode_versions: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    # --wrapup-loss mask: the wrap-up appends of each sample's episode (a pad copy: its original's)
    skip_of = {s.key: s.episode.wrapup_post_rids or frozenset() for s in samples if s.episode is not None}
    for i, s in enumerate(samples):
        item = b.tokens(i)
        seg = s.seg
        skip = skip_of[s.key if s.pad_of is None else s.pad_of]
        if item["key"] != s.key or not np.array_equal(item["input_ids"], seg.ids):
            raise RuntimeError(f"{s.key}: read-back input_ids differ from the ledger")
        loss, behavior = _expected_loss_and_behavior(seg, replay, skip)
        if not np.array_equal(item["loss_mask"], loss):
            raise RuntimeError(f"{s.key}: read-back loss_mask differs from the one re-derived from the append records")
        if not np.array_equal(item["behavior_logprob"][loss], behavior[loss]):
            raise RuntimeError(f"{s.key}: read-back behavior_logprob differs from the one re-derived from the append records")
        if replay:
            # straight from the append records (no gen_spans): each kept token's support, in position order
            want_lens, want_ids, want_full = np.zeros(len(seg), np.int64), [], np.full(len(seg), np.nan, np.float32)
            for k, a in enumerate(seg.appends):
                m = a.gen_len - (seg.appends[k + 1].rewind if k + 1 < len(seg.appends) else 0)
                if m <= 0 or a.rid in skip:
                    continue
                sup = a.sampling_mask.head(m)
                want_lens[a.gen_start:a.gen_start + m] = sup.lens
                want_ids.append(sup.ids)
                want_full[a.gen_start:a.gen_start + m] = a.logprobs[:m]
            want_ids = np.concatenate(want_ids) if want_ids else np.empty(0, np.int32)
            if not (np.array_equal(item["support_len"], want_lens) and np.array_equal(item["support_ids"], want_ids)
                    and np.array_equal(item["behavior_logprob_full"][loss], want_full[loss])):
                raise RuntimeError(f"{s.key}: read-back sampling support / full-vocabulary logprob differ from those re-derived from the append records")
        if s.pad_of is None:
            if item["coef"][loss].tolist() != [s.episode.coef] * int(loss.sum()):
                raise RuntimeError(f"{s.key}: read-back coef is not the written {s.episode.coef}")
            coef_sum[s.minibatch] += float(item["coef"].astype(np.float64).sum())
        elif item["coef"].any():
            raise RuntimeError(f"{s.key}: pad sample coef is not 0")
        if not np.array_equal(item["weight_version"], _expected_versions(seg, allowed, skip)):
            raise RuntimeError(f"{s.key}: read-back per-token weight versions differ from those re-derived from the append records")
        if s.pad_of is None:
            if allowed == [UNVERSIONED]:
                episode_versions[s.episode.key][UNVERSIONED] += int(loss.sum())
            else:
                vals, cnts = np.unique(item["weight_version"][loss], return_counts=True)
                episode_versions[s.episode.key].update({str(v): int(c) for v, c in zip(vals.tolist(), cnts.tolist())})
        if i in longest or i % 16 == 0:
            st = segment_routing_status(s.inst.path, seg, layers=layers, top_k=top_k, num_experts=num_experts)
            if not np.array_equal(item["routing"], st.routing):
                raise RuntimeError(f"{s.key}: read-back routing differs from the ledger")
            n_routing += 1
    for m, p in enumerate(plan):
        if abs(coef_sum[m] - p["sum_coef_expected"]) > 1e-4:
            raise RuntimeError(f"minibatch {m}: Σcoef={coef_sum[m]}, Σ_i A_i·L_i/denominator_i={p['sum_coef_expected']}")
    for r in b.index:
        if not r["pad"] and r["weight_version_tokens"] != dict(episode_versions[r["episode"]]):
            raise RuntimeError(f"{r['key']}: index weight_version_tokens {r['weight_version_tokens']} but "
                               f"{dict(episode_versions[r['episode']])} read back")
    with_media = [i for i, r in enumerate(b.index) if r["media"]][:2]
    for i in with_media:
        b[i]                                  # pixels load and match the grids (TitoMMDataset checks)
    return {"samples": len(samples), "routing_compared": n_routing, "pixels_loaded": len(with_media),
            "coef_sum": [coef_sum[m] for m in range(len(plan))]}


def cli(argv: list[str] | None = None) -> int:
    """Command-line entry: ``main``; with ``--error-json``, the two errors the caller can tell apart
    (``PREPARE_EXIT_CODES``) are written as one JSON, the traceback still goes to stderr, and the dedicated exit code is
    returned. A ``ManifestError`` that points at no manifest (belongs to no particular groups) and every other error
    still raise."""
    from arcforge.tito.join import ManifestError

    error_json = build_parser().parse_args(argv).error_json
    if error_json is None:
        main(argv)
        return 0
    if error_json.exists():
        raise FileExistsError(f"--error-json {error_json} already exists: one per prepare run")
    try:
        main(argv)
        return 0
    except ManifestError as error:
        if not error.manifests:
            raise
        doc = {"kind": "group_data", "error_type": type(error).__name__, "message": str(error),
               "manifests": [str(p) for p in error.manifests], "groups": sorted({p.parent.name for p in error.manifests})}
    except RoutingOverMaxGB as error:
        doc = {"kind": "routing_over_max_gb", "error_type": type(error).__name__, "message": str(error),
               "routing_bytes": error.routing_bytes, "max_gb": error.max_gb}
    traceback.print_exc()
    error_json.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    print(f"prepare_rl: {doc['kind']} -> {error_json} (exit {PREPARE_EXIT_CODES[doc['kind']]})", file=sys.stderr, flush=True)
    return PREPARE_EXIT_CODES[doc["kind"]]


if __name__ == "__main__":
    sys.exit(cli())
