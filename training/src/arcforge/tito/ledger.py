"""Read the on-disk files of the sglang TITO ledger (``DIR/<instance>/``) → ``Segment``, and check the invariants.

Spec (contract): "on-disk format v1" in ``docs/plans/tito_session_ledger.md``. One instance directory:

- ``server.json``: has at least ``instance`` (= the directory name). The spec names no other keys; they are only
  compared in audit.
- ``events.<n>.jsonl``: n consecutive from 0; one event per line. ids are base64 of little-endian int32, logprobs are
  float32.
- ``media/<sha256>``: raw bytes.
- ``routing.<n>.bin``: concatenated routing arrays, dtype/shape in the events.
- ``topk.<n>.bin`` (``--session-ledger-top-logprobs K``, fork ``arc/ledger-topk``): one block per append, layout
  ``ids_i4,lp_f2,tail_f4,flags_u1``: ``<i4`` ids ``[n_gen, K]``, ``<f2`` logprobs ``[n_gen, K]`` (each row
  descending), ``<f4`` tail ``[n_gen]`` (log of the probability outside the top-K, computed from fp32 values; -inf when
  the K values sum to 1 in fp32), ``u1`` flags ``[n_gen]`` (``TOPK_*``: what this row records is not the model's own
  distribution); the event's ``top_logprobs`` records ``{k, n, bytes, file, offset, layout}``. The distribution is
  stated in ``server.json``'s ``ledger.logprob_dist`` (``raw_t1`` = log_softmax(logits)).

Reconstruction rule (S = the segment's confirmed sequence)::

    append:  S = S[:prefix_len] + tail + gen      # prefix_len + rewind == len(S); gen span = [prefix_len+len(tail), len(S))

``rewind > 0``: the tail of the previous gen was rewound (the template filled in ``</think>``, or ``<|endoftext|>`` was
replaced with ``<|im_end|>`` = stop_substituted). The previous gen span keeps only ``[gen_start, prefix_len)`` in the
final S; the template tokens put in belong to this append's tail (not sampled). The original gen and logprobs are still
in the previous ``Append.gen`` / ``Append.logprobs`` (RL needs them).

Invariants (a violation is a ``LedgerInvariantError`` whose ``code`` is the name below; for each one the tests have a
broken file it catches):

- ``ids_in_vocab``: every tail/gen id is in ``[0, tokenizer id upper bound)``. The scheduler's hash pad (not a real pad
  id) gets caught here.
- ``append_order``: ``k`` consecutive from 0 within a segment.
- ``prefix_len``: when ``k == 0``, ``prefix_len == rewind == 0``; afterwards ``prefix_len + rewind == len(S)``.
- ``rewind``: ``rewind <= len(previous gen)`` (spec: the split point is inside the previous turn's generated span). tail
  may be 0: the last turn resent verbatim (codex dropping a compaction reply that carries a tool call and asking
  again). Checked before ``gen_overlap``.
- ``gen_overlap``: the newly written range ``[prefix_len, …)`` must not cover an existing gen span, only the rewound tail
  of the previous gen (``rewind`` already covers this; kept as a defence).
- ``stop_substituted``: ``rewind == 1``; what was replaced is the previous ``finish.matched`` (stop), not
  ``<|im_end|>``; the first token of the new tail is ``<|im_end|>``.
- ``finish``: ``finish.type`` is ``stop`` / ``length``; with ``stop`` and an int ``matched``, the last gen token is that
  token.
- ``logprobs_len``: when logprobs is not null, length = gen length, and no NaN.
- ``sampling_mask`` (``--session-ledger-sampling-mask``, fork ``rl-topp-mask``; optional key, earlier ledgers lack it):
  ``{"lens", "ids", "logprobs"}``, each gen token's sampling support (the ids top-k / top-p kept) and the logprobs
  renormalised over the support. ``lens`` has the gen's length, each ≥ 1, summing to the length of ``ids``; ``ids``
  are in the vocabulary, not repeated within a row, and include the sampled token; ``logprobs`` finite and ≤ 0.
- ``topk_len`` / ``topk_file``: top_logprobs ``n`` = gen length, ``bytes == n*k*6 + n*5``, ``layout`` recognised, the
  sidecar holds ``offset + bytes``.
- ``routing_len``: covers ``[start, start + n_tokens)``, ``start + n_tokens = prefix_len + len(tail) + len(gen) - 1``
  (spec routing section: the last token of this turn is left to the next turn); ``shape == [n_tokens, layers,
  top_k]``. Per the spec the start is ``max(prefix_len - 1, 0)`` (the last sampled token of the previous turn is only
  forwarded in this turn), but the fork decides the start from the prefix_len at **intake**: when intake is an
  extension that only becomes a split, or an attachment after a concurrent request, at finish (the spec's "late split"),
  the start does not match finish's prefix_len. Accepted when the end is right and the start lies in ``[0, end]``,
  recorded as ``Segment.routing_start_shifted`` (audit lists a warning); the positions where the start is later than
  the spec's have no routing, ``segment_routing_status`` marks them M (``late_branch``), and ``assemble_routing``,
  which assembles complete routing, still raises.
- ``routing_file``: dtype is an integer type, ``nbytes == prod(shape)*itemsize``, the sidecar file exists and
  ``offset + nbytes`` does not pass its end, and every record in a segment has the same ``shape[1:]`` / dtype.
- ``routing_overlap``: when ``rewind > 0``, this record's first row (position ``prefix_len - 1``, the last token kept)
  must also be in the previous record's range. The two records' **values** may differ (spec routing section: without
  the radix cache, or when the session was released, that row is recomputed, and serving is not batch-invariant); a
  difference is only recorded as ``Segment.routing_overlap_mismatch`` and listed by audit as a warning -- in the
  HiSparse G1b measurement all 15 were recomputations with ``cached_tokens=0``.
- ``media_range``: every media range (closed interval in segment coordinates) lies in the tail of the append that
  introduced it, and consists entirely of real pad ids of that modality.
- ``media_coverage``: at segment close, every pad run in S is covered by exactly one media range (no orphan pads, no
  partial ranges, no duplicates).
- ``media_blob``: for media whose ``sha256`` is not null, ``media/<sha256>`` exists. (The content sha256 is checked in
  audit / inline export.)
- ``close_len`` / ``close_sha256``: ``seg_close``'s ``n_tokens`` and sha256 (of S's ``<i4`` bytes) match the
  reconstructed S.
- ``sid_mismatch``: the sid of append / close equals seg_open's sid.
- ``unknown_segment`` / ``duplicate_segment`` / ``closed_segment``: an event points at a segment never opened, a
  segment number is opened twice, events arrive after a segment closed.

Segments without ``seg_close`` (server crash) are marked ``status == "open"``, not dropped. With ``strict=False``
broken segments are marked ``"broken"`` (later events belonging to them are skipped and counted) and the other segments
are read as usual -- so audit sees every problem at once; exports always use ``strict=True``. Format errors (bad JSON,
missing required keys, unknown event types, holes in events file numbers) always raise regardless of strict: the file
itself cannot be trusted. The one exception: the unterminated line at the end of the **last** events file -- the server
is writing it (RL spec §8: with async the ledger is written continuously) or crashed mid-write. It is ignored whether
or not it parses, and recorded in ``Instance.truncated_tail``; an unterminated end in any other file = raise. The
fork's writer writes media and routing before the event line, so the data a complete event line depends on is already
on disk (the routing file size is re-stat'ed when it exceeds the cached one).

Assumptions (not fixed by the spec; read per the fork WIP ``session_ledger.py``, see the report):

- gen includes the stop token (``S = input + output``).
- ``sid`` is null ⇔ ``seg_open.cause == "single"``; such segments' ``seg_close.reason`` may be ``"single"``.
- ``seg_open.cause`` is ``new`` / ``single`` / a split reason. The spec contradicts itself (the JSON example writes
  ``split:<reason>``, while the split-reason section and the fork WIP write a bare ``text_differs``), so both are
  accepted and normalised into ``Segment.split_reason``; a bare string must be one of ``SPLIT_REASONS`` (typos still
  blow up), the x of ``split:<x>`` is unrestricted (audit judges whether it is expected).
- An append's ``logprobs_error`` / ``routing_error`` (string, optional) = "this record lacks this item", and that item
  must then be null; not a broken file, counted by audit. media may carry ``unmapped: true`` (no sha256, only ranges)
  or ``sha256: null`` + ``src`` (received by reference, no bytes).
- ``attempt.kind`` is unrestricted (abort / orphaned / other finish types).
- Weight versions (RL spec §6.2, only after the fork fix; ledgers from before the fix have neither key →
  ``Append.weight_versions is None``): ``"weight_version": "<v>"`` (the whole gen on one version), or
  ``"weight_versions": [{"version", "start", "end"}, …]`` written by fork rl/v1 -- half-open intervals in gen
  coordinates, tiling ``[0, len(gen))`` in order (several only when weights changed mid-request; a version change
  between two forwards leaves an empty interval). When ``weight_versions`` is null it must come with
  ``weight_versions_error`` (this record was not captured) → None. Both keys present, a version that is not a non-empty
  string, or incomplete tiling are format errors. Normalised to ``((start within gen, v), …)``: empty intervals
  dropped, adjacent same versions merged. ``Append.prefill_versions`` is recorded too: the versions of every piece from
  gen position 0 on (empty intervals included), deduplicated -- when the version changed after the request came in but
  before the first token came out (``[{"3", 0, 0}, {"4", 0, n}]``) it is uncertain which version prefill used, so both
  are listed.
- ``replay``: ``{"spans": [...], "skip": string or null, "lcp0": int}``; every span has ``s`` / ``lcp`` / ``rec_end`` /
  ``key_end`` / ``delta`` (int). Since fork ``--session-replay-empty-think`` a span may carry ``"kind": "empty_think"``
  (a non-standard empty thinking at the start of the previous assistant turn -- ``<think>\n</think>\n\n`` or
  ``<think>\n`` followed directly by the answer -- that the client rendered as the standard ``<think>\n\n</think>\n\n``,
  swapped back to the sampled ids); since fork ``--session-replay-own-last-turn`` it may carry ``"kind": "own_turn"``
  (the message the session generated last, whose text the client rewrote -- a tool call truncated by the budget, a
  normalised bad call, truncated thinking where the template filled in ``</think>`` -- still swapped back to the
  sampled ids; docs/plans/replay_own_last_turn.md); such a span additionally has exactly ``closure`` (int ≥ 0: the
  number of tokens filled in after the truncated generation, at the start of this record's tail, not sampled),
  ``client_len`` (== ``key_end − s``) and ``client_sha256`` (sha256 of the ``<i4`` ids of the client's message, 64
  lowercase hex), and ``delta == rec_end − s + closure − client_len``; when later requests send ids identical position
  by position for that message, the sampled ids are swapped back in again (sticky): the span is still ``own_turn`` with
  an extra ``"sticky": true``, ``closure`` must be 0 (closure was already counted in the tail of the record that took
  over), and ``rec_end − s`` is the swapped-in length; no ``kind`` = re-tokenisation of the same text. Any other kind,
  or wrong keys, are format errors.
- ``resume`` (``docs/plans/session_exact_resume.md`` interface contract v2.1): the fork's
  ``meta_info["session_resume"]`` verbatim, ``{"mode": "exact"|"fallback"|"none", "reason": one of RESUME_REASONS or
  null, "pos": non-negative int or null, "prefill_version": non-empty string}``, exactly these four keys; null, any
  other shape, or a reason not in the enum is a format error. ``pos`` = where this request's prefill starts (positions
  before it were not recomputed); ``prefill_version`` = the weight version of the forward that computed the first
  extend chunk (null for an attempt aborted before ever being admitted). Whether this key exists is decided by the
  server's flag (``server_args.session_exact_resume`` in ``server.json``, ``server_exact_resume``; key absent = old fork
  = False; ``server_args`` itself missing = format error):

  - flag True: every append must have ``resume`` (missing = format error).
  - flag False: usually no ``resume`` (``Append.resume is None``, like the old fork); a resume with mode none is
    accepted (ser ledgers written before the fork changed to writing it only with the flag on); exact / fallback =
    format error (a server with the flag off never resumes exactly).

  Contract (``resume_problems`` / ``resume_violations``; audit fails, prepare raises; not a format error): requests
  without session_id are ``none``; ``none`` has null reason and pos; ``exact`` has null reason,
  ``pos == cached_tokens``, and extension turns with ``rewind == 0`` have ``window ∈ {0, −1}`` (``rewind > 0``: the
  ledger's S is the last commit and this request attached to only part of it -- with concurrent resends in the same
  session it attached to an earlier commit and the server resumed exactly from that park; stop_substituted / truncated
  turns are the client replacing or cutting the tail of the previous turn (``rewind_cause``) -- window drifts from
  {0, −1}, which is not a violation; positions with window > 0 really were forwarded twice and are classified as usual,
  with audit counting them separately by ``rewind_cause``);
  ``fallback`` must have a reason, and when reason ≠ retract, ``pos == cached_tokens``; retract only needs pos not null
  (re-admission after retract: cached_tokens reflects only the first admission, and a concurrent resend in the same
  session may have inserted the same ids into the tree first, making the re-admission prefix longer than the first
  cached_tokens -- pos and cached_tokens have no fixed ordering, and window is computed from pos as usual). exact /
  non-retract fallback with cached_tokens not recorded = unverifiable, also a violation.
- ``usage.cached_tokens``: non-negative int; missing or null → ``Append.cached_tokens is None`` (no guessing).
- ``attempt`` (requests that did not enter S, e.g. abort): ``usage``, and ``resume`` in three states
  (``AttemptRecord.resume_state``) -- key absent (old fork), null (no meta_info: an orphaned request left before
  finishing, or an abort never reached the scheduler; also null on a fork with the flag off), or the same shape and
  contract as for appends (but window is not checked -- its prefix_len is relative to S at intake, which concurrent
  requests may change later; ``prefill_version`` null = never admitted, and then pos must be null too). With the flag
  False an attempt cannot have exact / fallback (same as appends). Optional ``prefix_len``; parsed into
  ``Instance.attempt_records`` (``AttemptRecord``), with the raw dicts still in ``Instance.attempts``.

**Recompute window (session_exact_resume.md)**: an extension turn's (``k ≥ 1``) request = the previous turn's S + tail,
``window = max(prefix_len−1, 0) − reuse``, ``reuse`` = ``resume.pos`` (when there is a resume and pos is not null),
otherwise ``cached_tokens`` (old fork, or mode none). Without pos, D is a **lower bound**: cached_tokens reflects only
the first admission, and the part recomputed from the alignment point after a retract is invisible.
``window > 0``: positions ``[reuse, prefix_len−1)`` were already forwarded in the previous turn (or earlier); this turn
the radix matched only up to reuse and prefilled them again (forwarded twice); the ledger's routing starts at
``prefix_len−1``, so the routing of this range is the **first** forward's value, while the tokens after it in this turn
use the KV computed the second time. ``segment_reforward`` splits these positions by weight version into D_same /
D_cross / D_unknown, and ``segment_routing_status`` replaces the R of those positions with these three classes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Iterable

import numpy as np

LEDGER_VERSION = 1

#: tokens Vocab reads from the tokenizer (the strings are as the Qwen3.8 chat template writes them; ids are always asked
#: of the tokenizer).
TOKENS = {
    "im_start": "<|im_start|>",
    "im_end": "<|im_end|>",
    "endoftext": "<|endoftext|>",
    "vision_start": "<|vision_start|>",
    "vision_end": "<|vision_end|>",
    "image_pad": "<|image_pad|>",
    "video_pad": "<|video_pad|>",
    "think": "<think>",
    "think_end": "</think>",
    "tool_call": "<tool_call>",
    "tool_call_end": "</tool_call>",
    "tool_response": "<tool_response>",
    "tool_response_end": "</tool_response>",
}

CLOSE_REASONS = ("close_session", "evicted", "shutdown", "single")
OPEN_CAUSES = ("new", "single")
#: split reasons listed in the spec's "segment" section: the reasons replay stops + retry / concurrent / evicted.
#: the ``kind`` of a replay span (absent = re-tokenisation of the same text).
REPLAY_SPAN_KINDS = ("empty_think", "own_turn")
REPLAY_SPAN_KEYS = ("s", "lcp", "rec_end", "key_end", "delta")
#: a ``kind: own_turn`` span additionally has exactly these three keys (fork ``--session-replay-own-last-turn``,
#: docs/plans/replay_own_last_turn.md)
OWN_TURN_SPAN_KEYS = ("closure", "client_len", "client_sha256")
SPLIT_REASONS = ("text_differs", "stop", "no_end", "mm_overlap", "mm_shift", "mm_type", "input_len", "logprob_start",
                 "embeds", "position_data", "routed_experts", "retry", "concurrent", "evicted")
#: keys and modes of an append's ``resume`` (session_exact_resume.md interface contract)
RESUME_KEYS = ("mode", "reason", "pos", "prefill_version")
RESUME_MODES = ("exact", "fallback", "none")
#: fallback reasons (as settled in the interface contract); a reason not in here is a format error
RESUME_REASONS = ("no_park", "claimed", "ids_mismatch", "key_limit", "evicted", "retract", "abort", "prefill_finished",
                  "stop_string", "grammar", "think_budget", "snapshot_mismatch", "unsupported", "qsa_misaligned",
                  "spec_unsupported")
#: the three states of an attempt's ``resume`` (``AttemptRecord.resume_state``): key absent (old fork) / null (no
#: meta_info) / present
ATTEMPT_RESUME_ABSENT, ATTEMPT_RESUME_NULL, ATTEMPT_RESUME_PRESENT = "absent", "null", "present"
#: causes of ``rewind > 0`` in extension turns (``rewind_cause``)
REWIND_CAUSES = ("concurrent", "stop_substituted", "truncated")
MODALITIES = ("image", "video")
EVENT_KEYS = {
    "start": ("v", "instance", "t"),
    "seg_open": ("sid", "seg", "t", "cause"),
    "append": ("sid", "seg", "k", "rid", "t0", "t1", "prefix_len", "tail", "gen", "finish", "stop_substituted",
               "rewind", "sampling", "usage", "replay", "media", "logprobs", "routing"),
    "attempt": ("sid", "rid", "t", "kind", "input_len", "gen_len"),
    "seg_close": ("sid", "seg", "t", "reason", "n_tokens", "sha256"),
    "lossy": ("t", "dropped"),     # events the writer dropped (queue full), written once there is room again
    "status": ("t", "status"),     # /session_ledger/status every 300 s
    "end": ("t", "status"),        # graceful shutdown only: a killed server writes none
}
ROUTING_KEYS = ("file", "offset", "nbytes", "start", "n_tokens", "shape", "dtype")
TOPK_KEYS = ("k", "n", "bytes", "file", "offset", "layout")
TOPK_LAYOUT = "ids_i4,lp_f2,tail_f4,flags_u1"
# flags (TOPK_* of the fork's session_ledger.py, also in server.json ledger.topk_flags): rows with non-zero flags are
# not the model's own distribution
TOPK_HARD_MASKED = 1    # the K-th value is the mask floor: forcing (ARC_THINK_BUDGET) or grammar allowed fewer than K
TOPK_PENALTY = 2        # request has a frequency / presence / repetition penalty or min_new_tokens
TOPK_LOGIT_BIAS = 4     # request has logit_bias
TOPK_GRAMMAR = 8        # request has json_schema / regex / ebnf / structural_tag
TOPK_THINK_MASK = 16    # partial mask on the row after ARC_THINK_BUDGET forces a blank line (whitespace, <think>, </think>)
TOPK_CUSTOM = 32        # request has custom_logit_processor
_EVENTS_RE = re.compile(r"events\.(\d+)\.jsonl")
_SID_RE = re.compile(r"(.+):(\d+)")


class LedgerFormatError(RuntimeError):
    """The on-disk files themselves are malformed (bad JSON, missing keys, unknown events, holes in file numbers).
    Always raised, regardless of strict."""


class LedgerInvariantError(RuntimeError):
    """A reconstructed segment violates an invariant. ``code`` is one of the names listed in the module docstring."""

    def __init__(self, code: str, where: str, message: str):
        super().__init__(f"[{code}] {where}: {message}")
        self.code = code
        self.where = where


# ----------------------------------------------------------------------------- tokenizer


class Vocab:
    """Special token ids read from ``tokenizer.json``, and decode. Pad ids are not hard-coded: ask the tokenizer."""

    def __init__(self, path: str | Path):
        from tokenizers import Tokenizer

        path = Path(path)
        if path.is_dir():
            path = path / "tokenizer.json"
        if not path.is_file():
            raise FileNotFoundError(f"no tokenizer.json: {path}")
        self.path = path
        self.sha256 = sha256_file(path)
        self._tok = Tokenizer.from_file(str(path))
        self.ids: dict[str, int] = {}
        for name, text in TOKENS.items():
            tid = self._tok.token_to_id(text)
            if tid is None or self._tok.id_to_token(tid) != text:
                raise ValueError(f"{path} has no single token {text!r} (not a Qwen3.8-family tokenizer?)")
            self.ids[name] = int(tid)
        self.id_bound = max(self._tok.get_vocab(with_added_tokens=True).values()) + 1

    def __getattr__(self, name: str) -> int:
        ids = self.__dict__.get("ids")
        if ids is None or name not in ids:
            raise AttributeError(name)
        return ids[name]

    def decode(self, ids: Iterable[int]) -> str:
        return self._tok.decode([int(t) for t in ids], skip_special_tokens=False)

    def pad(self, modality: str) -> int:
        if modality == "image":
            return self.ids["image_pad"]
        if modality == "video":
            return self.ids["video_pad"]
        raise ValueError(f"unknown modality {modality!r}")


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ids_sha256(ids: np.ndarray) -> str:
    """The spec's S sha256: of S's little-endian int32 bytes."""
    return hashlib.sha256(np.ascontiguousarray(ids, dtype="<i4").tobytes()).hexdigest()


def parse_open_cause(cause: object, where: str) -> str | None:
    """seg_open.cause → split reason (None for new / single). See the assumptions in the module docstring."""
    if not isinstance(cause, str):
        raise LedgerFormatError(f"{where}: seg_open.cause must be a string, got {cause!r}")
    if cause in OPEN_CAUSES:
        return None
    if cause.startswith("split:") and re.fullmatch(r"[a-z_]+", cause[6:]):
        return cause[6:]
    if cause in SPLIT_REASONS:
        return cause
    raise LedgerFormatError(f"{where}: seg_open.cause {cause!r} is not new / single / a split reason {SPLIT_REASONS}")


def parse_close_reason(reason: object, where: str) -> str | None:
    """seg_close.reason → split reason (the x of ``split:<x>``; None otherwise)."""
    if not isinstance(reason, str):
        raise LedgerFormatError(f"{where}: seg_close.reason must be a string, got {reason!r}")
    if reason in CLOSE_REASONS:
        return None
    if reason.startswith("split:") and re.fullmatch(r"[a-z_]+", reason[6:]):
        return reason[6:]
    raise LedgerFormatError(f"{where}: seg_close.reason {reason!r} is not one of {CLOSE_REASONS} or split:<reason>")


def split_sid(sid: str) -> tuple[str, int]:
    """``<thread>:<n>`` (codex patch 0002) → (thread, n). Raises on a wrong format."""
    m = _SID_RE.fullmatch(sid)
    if m is None:
        raise ValueError(f"sid {sid!r} is not <thread>:<n>")
    return m.group(1), int(m.group(2))


# ----------------------------------------------------------------------------- records


@dataclass(frozen=True)
class MediaRef:
    modality: str                         # "image" | "video"
    ranges: tuple[tuple[int, int], ...]   # closed intervals (segment coords) covering pad runs (video: one per time group)
    sha256: str | None                    # None: unmapped or received by reference (no bytes)
    mime: str | None
    grid_thw: tuple[int, ...] | None
    unmapped: bool
    src: str | None
    k: int                                # the append that introduced it


@dataclass(frozen=True)
class RoutingRef:
    """Routing of one append: the experts chosen for each position in segment coordinates ``[start, start + n_tokens)``
    when it is the **input**.

    The routing of position p is computed by "the forward that processed p"; the record after a stop_substituted one
    overlaps the previous record by one position (both reads are of the same KV slot's value)."""

    file: str
    offset: int
    nbytes: int
    start: int
    n_tokens: int
    shape: tuple[int, ...]                # [n_tokens, layers, top_k]
    dtype: str


@dataclass(frozen=True)
class SamplingSupport:
    """Sampling support of one append (``--session-ledger-sampling-mask``): gen token j was sampled from the truncated,
    renormalised distribution over ``ids[offsets[j]:offsets[j+1]]``, and ``logprobs[j]`` is its log probability under
    that distribution. As long as gen (including a tail that may be rewound later)."""

    lens: np.ndarray        # int32 [n]
    offsets: np.ndarray     # int64 [n + 1]
    ids: np.ndarray         # int32 [offsets[-1]]
    logprobs: np.ndarray    # float32 [n]

    def head(self, n: int) -> "SamplingSupport":
        """Support of the first n tokens."""
        if not 0 <= n <= len(self.lens):
            raise ValueError(f"head({n}) of a {len(self.lens)}-token support")
        return SamplingSupport(lens=self.lens[:n], offsets=self.offsets[: n + 1], ids=self.ids[: self.offsets[n]],
                               logprobs=self.logprobs[:n])


@dataclass(frozen=True)
class TopkRef:
    """Top-k logprobs of one append: one row per gen position (including a tail that may be rewound later); ``read_topk``
    reads it into ``TopkRows``."""

    file: str
    offset: int
    nbytes: int
    k: int
    n: int


@dataclass(frozen=True)
class TopkRows:
    """Result of ``read_topk``; n = that append's gen length."""

    ids: np.ndarray        # int32 [n, k], each row descending
    logprobs: np.ndarray   # float32 [n, k] (stored as f16)
    tail: np.ndarray       # float32 [n]: log(1 - sum exp(top-k)), -inf = no remaining mass in fp32
    flags: np.ndarray      # uint8 [n]: bitfield of TOPK_*


@dataclass(frozen=True)
class Resume:
    """An append's ``resume``: this turn's prefill resumed exactly from where the previous turn ended (``exact``, ``pos``
    = the resume point P), fell back to an alignment point (``fallback``, ``reason`` says why), or the server has exact
    resume off / the request has no session (``none``)."""

    mode: str                             # RESUME_MODES
    reason: str | None                    # one of RESUME_REASONS, or None
    pos: int | None                       # prefill starts here (positions before it were not recomputed)
    prefill_version: str | None           # version of the forward computing the first extend chunk; None only in attempts

    @property
    def admitted(self) -> bool:
        """Forwarded at least once (prefill_version is None for an attempt aborted before ever being admitted)."""
        return self.prefill_version is not None


@dataclass(frozen=True)
class AttemptRecord:
    """Parsed ``attempt`` event (requests that did not enter S, e.g. abort). reuse = ``resume.pos`` (when present and not
    None), otherwise cached_tokens. When the fork does not write an attempt's prefix_len, the recompute window can only
    be measured against the session's S at the time it was written: ``window_vs_s`` =
    ``max(min(input_len, len(S)) − 1, 0) − reuse`` -- its recompute window when the request extends S (the turn that
    was cut), an upper bound otherwise. An attempt never admitted (``resume.prefill_version`` is None) forwarded no
    position: both windows are None."""

    sid: str | None
    rid: str
    kind: str
    input_len: int
    gen_len: int
    cached_tokens: int | None             # usage.cached_tokens; no usage, usage null or not recorded → None
    resume: Resume | None                 # resume key absent or null → None (resume_state tells the two apart)
    resume_state: str                     # ATTEMPT_RESUME_ABSENT / _NULL / _PRESENT
    prefix_len: int | None                # only when the fork writes it
    s_len: int | None                     # length of the session's open segment when written (no open segment → None)
    seg: int | None                       # that segment's number

    @property
    def admitted(self) -> bool | None:
        """Whether resume says it was admitted; no resume (old fork, or null: no meta_info) → None (unknown)."""
        return None if self.resume is None else self.resume.admitted

    @property
    def reuse(self) -> int | None:
        if self.admitted is False:
            return None
        return reuse_point(self.resume, self.cached_tokens)

    @property
    def window(self) -> int | None:
        """Recompute window ``max(prefix_len−1, 0) − reuse`` when prefix_len is known; None otherwise."""
        if self.prefix_len is None or self.reuse is None:
            return None
        return max(self.prefix_len - 1, 0) - self.reuse

    @property
    def window_vs_s(self) -> int | None:
        """Recompute window measured against S at write time (an upper bound, see the class docstring); S or reuse
        unknown → None."""
        if self.s_len is None or self.reuse is None:
            return None
        return max(min(self.input_len, self.s_len) - 1, 0) - self.reuse


def reuse_point(resume: Resume | None, cached_tokens: int | None) -> int | None:
    """Where this request's prefill starts: ``resume.pos`` (when there is a resume and pos is not None: only pos shows
    the part recomputed from the alignment point after a retract), otherwise cached_tokens (old fork, mode none).
    Neither → None."""
    if resume is not None and resume.pos is not None:
        return resume.pos
    return cached_tokens


@dataclass
class Append:
    k: int
    rid: str
    t0: float
    t1: float
    prefix_len: int
    rewind: int                           # tokens rewound off the previous gen's tail (prefix_len + rewind == previous len(S))
    tail_len: int
    gen: np.ndarray                       # original sampled ids (int32), including a tail the next record may rewind
    finish: dict
    stop_substituted: bool
    sampling: dict
    usage: dict
    replay: dict
    media: list[MediaRef]
    logprobs: np.ndarray | None           # float32, as long as gen (including a tail that may be rewound later)
    logprobs_error: str | None
    routing: RoutingRef | None
    routing_error: str | None
    weight_versions: tuple[tuple[int, str], ...] | None = None   # ((gen index, version), …); None = not recorded
    weight_versions_error: str | None = None   # fork's reason when weight_versions is null (None, no reason = pre-fix ledger)
    top_logprobs: TopkRef | None = None   # None: --session-ledger-top-logprobs off, or missing here (top_logprobs_error)
    top_logprobs_error: str | None = None
    prefill_versions: tuple[str, ...] | None = None   # version of every piece from gen pos 0 (empty ones too); None = not recorded
    resume: Resume | None = None          # None: old fork (no resume key)
    cached_tokens: int | None = None      # usage.cached_tokens; None = not recorded
    sampling_mask: SamplingSupport | None = None   # None: --session-ledger-sampling-mask off, untruncated sampling, or missing here
    sampling_mask_error: str | None = None

    @property
    def reuse(self) -> int | None:
        """Where prefill starts (``reuse_point``: resume.pos, otherwise cached_tokens); neither recorded → None."""
        return reuse_point(self.resume, self.cached_tokens)

    @property
    def window(self) -> int | None:
        """Recompute window of an extension turn (k ≥ 1), ``max(prefix_len−1, 0) − reuse`` (module docstring). > 0: the
        positions in ``reforward_span`` were forwarded once more this turn. The segment's first record (not an
        extension) or unknown reuse → None."""
        if self.k == 0 or self.reuse is None:
            return None
        return self.routing_expected_start - self.reuse

    @property
    def reforward_span(self) -> tuple[int, int] | None:
        """``[reuse, prefix_len−1)``: positions recomputed this turn that had already been forwarded before (half-open
        interval in segment coordinates). The empty interval ``(prefix_len−1, prefix_len−1)`` when ``window ≤ 0``;
        None when ``window`` is None (unknown)."""
        w = self.window
        if w is None:
            return None
        hi = self.routing_expected_start
        return hi - max(w, 0), hi

    @property
    def replay_kinds(self) -> list[str]:
        """Each message replay swapped back at this record's intake: ``"text"`` (re-tokenisation of the same text), the
        span's ``kind``, or ``"own_turn_sticky"`` (a message taken over earlier, swapped back again)."""
        return [span.get("kind", "text") + ("_sticky" if span.get("sticky") else "") for span in self.replay["spans"]]

    @property
    def replay_closure(self) -> int:
        """Total closure tokens of own_turn spans (filled in after the truncated previous turn, at the start of this
        record's tail, not sampled)."""
        return sum(span["closure"] for span in self.replay["spans"] if span.get("kind") == "own_turn")

    @property
    def routing_expected_start(self) -> int:
        """The spec's routing start; ``routing.start`` differing from it = late split (see ``routing_len`` in the module
        docstring)."""
        return max(self.prefix_len - 1, 0)

    @property
    def gen_len(self) -> int:
        return len(self.gen)

    @property
    def gen_start(self) -> int:
        """Start of this gen in segment coordinates at sampling time; S before it is never touched by later rewinds (a
        rewind only reaches into the previous gen)."""
        return self.prefix_len + self.tail_len


@dataclass
class GenSpan:
    """A span of the final S that really was sampled: ``[start, end)``. The tail rewound by the next record is not in it
    (``dropped`` tokens)."""

    start: int
    end: int                              # half-open, segment coordinates
    k: int                                # which append it comes from
    dropped: int = 0                      # tail tokens rewound by the next record


@dataclass
class Segment:
    instance: str
    seg: int
    sid: str | None
    cause: str                            # seg_open.cause verbatim
    split_reason: str | None              # normalised split reason; None for new / single
    t_open: float
    exact_resume: bool                    # the server that wrote it ran --session-exact-resume (Instance.exact_resume)
    appends: list[Append] = field(default_factory=list)
    gen_spans: list[GenSpan] = field(default_factory=list)
    close: dict | None = None
    status: str = "open"                  # "open" | "closed" | "broken"
    error: LedgerInvariantError | None = None
    skipped_events: int = 0
    routing_overlap_mismatch: int = 0     # overlap row at a rewind differs between records (recompute; warning, not broken)
    routing_start_shifted: int = 0        # records whose routing start is not max(prefix_len-1, 0) (late split; warning)
    # image / video pad tokens the model sampled itself (T=1.0 untruncated sampling: one video pad appeared in the
    # compaction summary of 2026-10-24 v4 cd82__s1), plus the ones sent back afterwards as prompt and recorded by the fork
    # as unmapped n_items 0: (modality, a, b) closed intervals. Not media, coverage does not count them; such a segment
    # has no media to feed, and the RL export must drop the whole episode (prepare_rl's stray_media_pad).
    stray_media_pads: list = field(default_factory=list)
    _chunks: list[np.ndarray] = field(default_factory=list, repr=False)
    _n: int = 0
    _ids: np.ndarray | None = field(default=None, repr=False)

    @property
    def key(self) -> str:
        return f"{self.instance}:{self.seg}"

    @property
    def ids(self) -> np.ndarray:
        if self._ids is None or len(self._ids) != self._n:
            self._ids = np.concatenate(self._chunks) if self._chunks else np.empty(0, np.int32)
            self._chunks = [self._ids]
        return self._ids

    def __len__(self) -> int:
        return self._n

    @property
    def thread(self) -> str:
        if self.sid is None:
            raise ValueError(f"{self.key}: a single segment has no thread")
        return split_sid(self.sid)[0]

    @property
    def window(self) -> int:
        if self.sid is None:
            raise ValueError(f"{self.key}: a single segment has no window")
        return split_sid(self.sid)[1]

    @property
    def media(self) -> list[MediaRef]:
        return [m for a in self.appends for m in a.media]

    @property
    def n_gen(self) -> int:
        return sum(s.end - s.start for s in self.gen_spans)

    def gen_logprobs(self) -> np.ndarray | None:
        """Logprobs of every gen span concatenated (same order and length as ``gen_spans``). None if any record lacks
        them."""
        by_k = {a.k: a for a in self.appends}
        parts = []
        for span in self.gen_spans:
            lp = by_k[span.k].logprobs
            if lp is None:
                return None
            parts.append(lp[: span.end - span.start])
        return np.concatenate(parts) if parts else np.empty(0, np.float32)

    def gen_sampling_mask(self) -> SamplingSupport | None:
        """Sampling supports of every gen span concatenated (same order and length as ``gen_spans``). None if any record
        lacks one."""
        by_k = {a.k: a for a in self.appends}
        parts = []
        for span in self.gen_spans:
            sup = by_k[span.k].sampling_mask
            if sup is None:
                return None
            parts.append(sup.head(span.end - span.start))
        if not parts:
            return SamplingSupport(lens=np.empty(0, np.int32), offsets=np.zeros(1, np.int64),
                                   ids=np.empty(0, np.int32), logprobs=np.empty(0, np.float32))
        lens = np.concatenate([q.lens for q in parts])
        offsets = np.zeros(len(lens) + 1, np.int64)
        np.cumsum(lens, out=offsets[1:])
        return SamplingSupport(lens=lens, offsets=offsets, ids=np.concatenate([q.ids for q in parts]),
                               logprobs=np.concatenate([q.logprobs for q in parts]))

    def rids(self) -> list[str]:
        return [a.rid for a in self.appends]

    def reforward_spans(self) -> list[tuple[int, int, int]]:
        """Every extension append with ``window > 0``: ``(k, lo, hi)``, ``[lo, hi)`` being the positions it forwarded a
        second time. Records without cached_tokens are not included (``unknown_windows`` of ``segment_reforward``)."""
        out = []
        for a in self.appends:
            span = a.reforward_span
            if span is not None and span[0] < span[1]:
                out.append((a.k, *span))
        return out


@dataclass
class Instance:
    path: Path
    name: str
    server: dict
    start: dict
    segments: dict[int, Segment]
    attempts: list[dict]
    errors: list[LedgerInvariantError]    # errors that belong to no segment (only with strict=False)
    truncated_tail: bool
    n_events: int
    exact_resume: bool                   # server_exact_resume(server): the server ran --session-exact-resume
    lossy_dropped: int = 0               # sum of on-disk "lossy" events
    last_status: Optional[dict] = None   # the last "status"/"end" event's status
    ended: bool = False                  # an "end" event: graceful shutdown
    events_read_bytes: int = 0           # JSON bytes scanned (even with a thread filter)
    total_segments: int = 0              # all seg_open events, before the thread filter
    attempt_records: list[AttemptRecord] = field(default_factory=list)   # the attempts, parsed (usage, resume)

    def media_path(self, sha256: str) -> Path:
        return self.path / "media" / sha256

    def routing_path(self, ref: RoutingRef) -> Path:
        return self.path / ref.file

    def read_routing(self, ref: RoutingRef) -> np.ndarray:
        return read_routing(self.path, ref, self.name)

    def read_topk(self, ref: TopkRef) -> TopkRows:
        return read_topk(self.path, ref, self.name)

    def segment_routing(self, seg: Segment) -> np.ndarray:
        """MoE routing of every position of the final S (``assemble_routing``)."""
        return assemble_routing(self.path, seg.key, [a.routing for a in seg.appends], len(seg))


def read_routing(directory: Path, ref: RoutingRef, name: str = "") -> np.ndarray:
    with open(Path(directory) / ref.file, "rb") as f:
        f.seek(ref.offset)
        raw = f.read(ref.nbytes)
    if len(raw) != ref.nbytes:
        raise LedgerFormatError(f"{name or directory}/{ref.file}: read {len(raw)} bytes, expected {ref.nbytes}")
    return np.frombuffer(raw, dtype=np.dtype(ref.dtype)).reshape(ref.shape)


def read_topk(directory: Path, ref: TopkRef, name: str = "") -> TopkRows:
    """Read TOPK_LAYOUT back into ``TopkRows``; raises on NaN in logprobs or tail (the server never writes NaN, so one
    is a bug)."""
    with open(Path(directory) / ref.file, "rb") as f:
        f.seek(ref.offset)
        raw = f.read(ref.nbytes)
    where = f"{name or directory}/{ref.file}@{ref.offset}"
    if len(raw) != ref.nbytes:
        raise LedgerFormatError(f"{where}: read {len(raw)} bytes, expected {ref.nbytes}")
    n, k = ref.n, ref.k
    a = n * k * 4
    b = a + n * k * 2
    c = b + n * 4
    ids = np.frombuffer(raw[:a], dtype="<i4").reshape(n, k).astype(np.int32)
    vals = np.frombuffer(raw[a:b], dtype="<f2").reshape(n, k).astype(np.float32)
    tail = np.frombuffer(raw[b:c], dtype="<f4").astype(np.float32)
    flags = np.frombuffer(raw[c:], dtype="u1").copy()
    if np.isnan(vals).any() or np.isnan(tail).any():
        raise LedgerFormatError(f"{where}: top-k logprob / tail has NaN")
    return TopkRows(ids=ids, logprobs=vals, tail=tail, flags=flags)


def assemble_routing(directory: Path, key: str, refs: list[RoutingRef | None], n: int) -> np.ndarray:
    """MoE routing of every position of a segment's final S (length ``n``): ``out[p]`` = the experts chosen by the
    forward that took token p as input, ``(n, layers, top_k)`` int16. ``refs`` is the routing of each append of the
    segment, in append order.

    A later record overwrites an earlier one (for rewound or recomputed positions the later record wins, the
    ``RoutingRef`` rule). The last position was never forwarded (only computed in the next turn) and is filled with -1;
    if an earlier record wrote it, that was the value of a rewound token, likewise discarded. Every other position must
    be covered by some record, otherwise raise."""
    if not refs or any(r is None for r in refs):
        raise LedgerFormatError(f"{key}: {sum(r is None for r in refs)}/{len(refs)} appends have no routing")
    out = np.full((n, *refs[0].shape[1:]), -1, np.int16)
    covered = np.zeros(n, np.bool_)
    for ref in refs:
        rows = read_routing(directory, ref, key)
        if rows.dtype != np.int16:
            raise LedgerFormatError(f"{key}: routing dtype {rows.dtype}, expected int16")
        k = min(ref.start + ref.n_tokens, n) - ref.start
        out[ref.start:ref.start + k] = rows[:k]
        covered[ref.start:ref.start + k] = True
    out[n - 1] = -1
    if not covered[:n - 1].all():
        first = int(np.flatnonzero(~covered[:n - 1])[0])
        raise LedgerFormatError(f"{key}: position {first} is not covered by any routing record "
                                f"({int((~covered[:n - 1]).sum())} in total)")
    if (out[:n - 1] < 0).any():
        raise LedgerFormatError(f"{key}: routing has negative expert ids")
    return out


#: per-position states of ``segment_routing_status`` (RL spec §5). R / N are written into the RL batch; M / X / D only
#: appear during prepare (D_same is written as R in the RL batch: the first forward's routing is used).
ROUTING_R, ROUTING_N, ROUTING_M, ROUTING_X = 0, 1, 2, 3
#: D = forwarded twice (recompute window, module docstring): the first forward's routing is recorded, while the tokens
#: after it this turn use the second forward's KV. Three classes by the weight versions of the two forwards; the larger
#: number wins (a position recomputed by several windows takes the most severe): D_cross > D_unknown > D_same.
ROUTING_D_SAME, ROUTING_D_UNKNOWN, ROUTING_D_CROSS = 4, 5, 6
ROUTING_NAMES = {ROUTING_R: "R", ROUTING_N: "N", ROUTING_M: "M", ROUTING_X: "X", ROUTING_D_SAME: "D_same",
                 ROUTING_D_UNKNOWN: "D_unknown", ROUTING_D_CROSS: "D_cross"}
#: reasons for M (missing: the ledger has no routing for this range) and for X (recorded but the value is invalid)
ROUTING_M_REASONS = ("routing_error", "routing_null", "late_branch", "sentinel")
ROUTING_X_REASONS = ("out_of_range", "zero_row", "duplicate")


@dataclass(frozen=True)
class ReforwardWindow:
    """Classification of one extension append's recompute window ``[lo, hi)`` (``Append.reforward_span``, non-empty)."""

    k: int
    lo: int
    hi: int
    from_gen: int                         # first forward was a decode of the earlier record (the position is its gen)
    from_prefill: int                     # first forward was the earlier record's prefill (its tail, or its prefix_len−1)
    same: int                             # both forwards on the same weight version
    cross: int                            # different versions (or either side uncertain, but certainly not the same)
    unknown: int                          # either side has no recorded version


@dataclass
class Reforward:
    """Result of ``segment_reforward``. ``code``: (n,) uint8, 0 = forwarded only once as far as the ledger knows,
    otherwise ROUTING_D_*."""

    code: np.ndarray
    windows: list[ReforwardWindow]
    unknown_windows: list[int]            # extension appends (k) whose reuse (resume.pos / cached_tokens) is unknown

    def count(self, code: int) -> int:
        return int((self.code == code).sum())


def segment_reforward(seg: Segment) -> Reforward:
    """For every position of a segment: whether it was forwarded twice, and whether both forwards used the same weight
    version. Looks only at the append records; does not read routing files.

    Replays record by record: record k replaces ``[prefix_len, …)`` with new tokens (the earlier "first forward" there
    is void), and its request first-forwards ``[max(prefix_len−1, 0), len(S_k) − 1)`` -- the same range routing covers
    (``RoutingRef``). In the recompute window ``[cached, prefix_len−1)`` of record m (an extension), the first forward
    of every position p belongs to some earlier record j (later records all start at their own prefix_len−1, and
    prefix_len never shrinks, so they cannot cover p again). Versions (``weight_versions`` is the version at which each
    gen token was **sampled**):

    - second forward (record m's prefill): m's ``resume.prefill_version`` (recorded by the fork: the forward that
      computed the first extend chunk); on an old fork without resume, inferred from the labels: ``prefill_versions(m)``.
    - first forward, p in j's prefill (j's tail or j's prefix_len−1): the same rule applied to j
      (``resume.prefill_version``, or ``prefill_versions(j)`` without it).
    - first forward, p is j's i-th gen token (input to a decode step): that step sampled token i+1, so the version is
      ``label(i+1)``; but during an in_place sync of the overlap scheduler, the token sampled by the step already issued
      before the pause is labelled with the new version while actually sampled with the old weights (RL spec §8 "the
      fork already has this") -- that slot's forward version is ``label(i)``. Both count: ``{label(i), label(i+1)}``.

    Both sides' version sets being the same single version = D_same; either side unrecorded (``weight_versions`` null, or
    a pre-fix ledger) = D_unknown; everything else = D_cross (including "one side has two candidate versions":
    conservative, better to drop the episode). A blind spot that only applies to old forks (no ``prefill_version``):
    when the prefill step itself is labelled one slot late (the first gen token's label is newer than the prefill), the
    label is taken as truth -- on the second-forward side this only adds cross verdicts (conservative), while on the
    first-forward side a really cross-version position may be judged same; it takes a first-forward prefill straddling
    a sync exactly, with the next turn's recompute window reaching back into that record's tail (98.5% of recompute
    windows fall in the previous turn's gen, per the plan's measurement), for this to happen. A version change midway
    through chunked prefill: ``prefill_version`` records only the first chunk, so a change in later chunks is still
    invisible. If record j's prefix_len−1 was read back from cache (cached_tokens ≥ prefix_len: exact resume, the
    previous record's last step already forwarded it), its first forward actually belongs to the previous record's last
    step; here it counts as j's prefill (this only matters if a later recompute window reaches back there). Forwards
    outside the ledger are invisible (an aborted attempt: the positions it recomputed need prefix_len to be pinned down,
    so they are not counted here and audit lists them separately; ``parallel_sample_num > 1`` is not recorded)."""
    n = len(seg)
    code = np.zeros(n, np.uint8)
    windows: list[ReforwardWindow] = []
    unknown: list[int] = []
    ids: dict[str, int] = {}

    def vid(v: str) -> int:
        return ids.setdefault(v, len(ids))

    def pv(a: Append) -> tuple[int, int]:
        """prefill version → two slots: the fork's prefill_version when recorded; otherwise the set inferred from the
        labels (a single version fills both slots, two or more take first and last, unrecorded is -1)."""
        if a.resume is not None:
            v = vid(a.resume.prefill_version)
            return v, v
        if a.prefill_versions is None:
            return -1, -1
        return vid(a.prefill_versions[0]), vid(a.prefill_versions[-1])

    owner = np.full(n, -1, np.int32)      # the append whose request first forwarded p (as the ledger sees it)
    va = np.full(n, -1, np.int32)         # two candidate version ids of that first forward (-1: not recorded)
    vb = np.full(n, -1, np.int32)
    decode = np.zeros(n, np.bool_)        # the first forward was a decode step (p is that append's gen)
    for j, a in enumerate(seg.appends):
        span = a.reforward_span
        if a.k > 0 and span is None:
            unknown.append(a.k)
        elif span is not None and span[0] < span[1]:
            lo, hi = span[0], min(span[1], n)
            if (owner[lo:hi] < 0).any():
                raise RuntimeError(f"{seg.key}: internal error: append {a.k}'s recompute window [{lo}, {hi}) has positions "
                                   f"without a first forward")
            ra, rb = pv(a)
            fa, fb = va[lo:hi], vb[lo:hi]
            known = (fa >= 0) & (fb >= 0) & (ra >= 0)
            same = known & (fa == ra) & (fb == ra) & (ra == rb)
            cross = known & ~same
            c = np.full(hi - lo, ROUTING_D_UNKNOWN, np.uint8)
            c[same] = ROUTING_D_SAME
            c[cross] = ROUTING_D_CROSS
            np.maximum(code[lo:hi], c, out=code[lo:hi])
            n_gen = int(decode[lo:hi].sum())
            windows.append(ReforwardWindow(k=a.k, lo=lo, hi=hi, from_gen=n_gen, from_prefill=hi - lo - n_gen,
                                           same=int(same.sum()), cross=int(cross.sum()), unknown=int((~known).sum())))
        p = a.prefix_len
        owner[p:] = -1
        va[p:] = -1
        vb[p:] = -1
        decode[p:] = False
        lo, gs = a.routing_expected_start, a.gen_start
        end = min(p + a.tail_len + a.gen_len - 1, n)
        if end <= lo:
            continue
        owner[lo:end] = j
        decode[lo:end] = False
        pa, pb = pv(a)
        pre = min(gs, end)
        va[lo:pre], vb[lo:pre] = pa, pb
        if end > gs:
            if a.weight_versions is None:
                va[gs:end], vb[gs:end] = -1, -1
            else:
                label = np.empty(a.gen_len, np.int32)
                runs = a.weight_versions
                for (start, v), stop in zip(runs, [s for s, _ in runs[1:]] + [a.gen_len]):
                    label[start:stop] = vid(v)
                i = np.arange(end - gs)
                va[gs:end], vb[gs:end] = label[i], label[i + 1]
            decode[gs:end] = True
    return Reforward(code=code, windows=windows, unknown_windows=unknown)


#: recompute windows allowed for an exact extension turn: 0 = resumed at prefix_len−1 (the previous turn's last token is
#: forwarded only in this turn), −1 = even that token was read back from cache
EXACT_WINDOWS = (0, -1)


def resume_problems(r: Resume, *, cached: int | None, window: int | None, extension: bool,
                    session: bool, rewind: int = 0) -> list[str]:
    """Contract v2.1 of one ``resume`` against its own cached_tokens / window (module docstring). ``extension``: the
    request extends S and ``window`` is measurable (an append with k ≥ 1) -- only then is exact's window checked, and
    only when ``rewind == 0`` (for ``rewind > 0`` see ``rewind_cause``; attempts never have window checked).
    ``session``: the request has a session_id (the segment's or attempt's sid is not null)."""
    out = []
    if not session and r.mode != "none":
        out.append(f"a request without session_id must have resume.mode none, got {r.mode}")
    if r.mode == "none":
        if r.reason is not None or r.pos is not None:
            out.append(f"resume.mode=none but reason={r.reason!r} / pos={r.pos!r} (both must be null)")
        return out
    if r.mode == "exact" and r.reason is not None:
        out.append(f"resume.mode=exact but reason={r.reason!r}")
    if r.mode == "fallback" and r.reason is None:
        out.append("resume.mode=fallback but no reason")
    if not r.admitted:                    # an attempt aborted before admission: it forwarded nothing
        if r.pos is not None:
            out.append(f"prefill_version is null (not admitted) but pos={r.pos}")
        if r.mode == "exact":
            out.append("prefill_version is null (not admitted) but mode is exact")
        return out
    if r.pos is None:
        out.append(f"resume.mode={r.mode} but no pos")
    elif r.reason != "retract":
        # (a retract re-admits: its pos and the first admission's cached_tokens are unrelated -- a concurrent resend
        # may have inserted the same ids meanwhile -- so only pos is required)
        if cached is None:
            out.append(f"resume.mode={r.mode} but cached_tokens not recorded (pos cannot be checked against cached_tokens)")
        elif r.pos != cached:
            out.append(f"resume.mode={r.mode} with pos={r.pos} ≠ cached_tokens={cached}")
    if r.mode == "exact" and extension and rewind == 0 and window is not None and window not in EXACT_WINDOWS:
        out.append(f"resume.mode=exact but window={window} (must be in {EXACT_WINDOWS}"
                   + (f": [{r.pos}, {r.pos + window}) forwarded twice)" if window > 0 and r.pos is not None else ")"))
    return out


def resume_violations(seg: Segment) -> list[str]:
    """``resume_problems`` of every append of a segment. A violation is a fork bug: one line each, with instance:seg /
    sid / k / rid and the numbers."""
    out = []
    for a in seg.appends:
        if a.resume is None:
            continue
        where = (f"{seg.key} sid={seg.sid} k={a.k} rid={a.rid} (prefix_len={a.prefix_len}, rewind={a.rewind}, "
                 f"cached_tokens={a.cached_tokens}, window={a.window}, resume.pos={a.resume.pos})")
        out += [f"{where}: {x}" for x in resume_problems(a.resume, cached=a.cached_tokens, window=a.window,
                                                          extension=a.k > 0, session=seg.sid is not None,
                                                          rewind=a.rewind)]
    return out


def attempt_violations(inst_name: str, at: AttemptRecord) -> list[str]:
    """``resume_problems`` of an attempt, but **without the window check**: attempts do not enter the training data, and
    an attempt's prefix_len is relative to S at intake, which concurrent requests may change later, so judging exact's
    window by it gives false alarms (only counted, listed separately by audit). The other rules apply as usual; an
    attempt whose resume is null / absent has nothing to check."""
    if at.resume is None:
        return []
    where = (f"{inst_name} attempt sid={at.sid} rid={at.rid} kind={at.kind} (prefix_len={at.prefix_len}, "
             f"cached_tokens={at.cached_tokens}, window={at.window}, resume.pos={at.resume.pos})")
    return [f"{where}: {x}" for x in resume_problems(at.resume, cached=at.cached_tokens, window=at.window,
                                                      extension=False, session=at.sid is not None)]


def rewind_cause(seg: Segment, a: Append) -> str | None:
    """Cause of ``rewind > 0`` in an extension turn (``REWIND_CAUSES``; ``rewind == 0`` → None):

    - ``stop_substituted``: the client replaced the previous turn's stop token with <|im_end|> (rewind 1; the fork clears
      this flag when the session changed after intake, so it never overlaps with concurrent).
    - ``concurrent``: the previous append was committed (``t1``) after this request's intake (``t0``) -- S was changed by
      another request of the same session while this one ran, and this one attached to an earlier commit (concurrent
      resend).
    - ``truncated``: everything else -- the previous turn was committed before intake, and this request cut into its gen
      tail (the template trim of a truncated turn, the same request resent after it was already committed).
    """
    if a.rewind == 0:
        return None
    if a.k == 0:
        raise RuntimeError(f"{seg.key}: internal error: the segment's first record has rewind={a.rewind}")
    if a.stop_substituted:
        return "stop_substituted"
    return "concurrent" if seg.appends[a.k - 1].t1 > a.t0 else "truncated"


@dataclass
class RoutingStatus:
    routing: np.ndarray                   # (n, layers, top_k) int16; -1 at positions other than R / D
    status: np.ndarray                    # (n,) uint8: ROUTING_R / N / M / X / D_*
    reasons: dict[str, int]               # reason → number of positions (M_REASONS and X_REASONS; for D see status)

    def count(self, code: int) -> int:
        return int((self.status == code).sum())


def segment_routing_status(directory: Path, seg: Segment, *, layers: int, top_k: int,
                           num_experts: int) -> RoutingStatus:
    """Routing and state of every position of a segment's final S (RL spec §5). The non-raising version of
    ``assemble_routing``: gaps are marked for the caller to decide.

    Replays the construction of S append by append: record k replaces ``[prefix_len, …)`` with new tokens, voiding any
    routing written there before; this record should cover ``[max(prefix_len-1, 0), len(S_k) - 1)``, and its routing is
    written when present (later overwrites earlier); otherwise the positions in that range not yet covered are marked
    missing, with reason ``routing_error`` (the fork explained why it is missing) or ``routing_null``; positions missing
    because the start is later than the spec's (late split) are ``late_branch``. Finally:

    - the segment end (never forwarded) → N.
    - uncovered positions → M (with the reasons above). Uncovered yet not in any record's "missing" range = an N outside
      the segment end → raise.
    - covered rows (each position's ``(layers, top_k)``): a -1 (the fork's sentinel initial value never overwritten) →
      M ``sentinel``; an id < -1 or ≥ ``num_experts`` → X ``out_of_range``; a layer whose top-k is all 0 → X
      ``zero_row``; a layer with repeated ids → X ``duplicate``.
    - among the rest (R), positions forwarded twice (``segment_reforward``) → D_same / D_unknown / D_cross; the routing
      is still the first forward's value. M / X take precedence (that position is unusable anyway). The classification
      does not depend on the server's flag (whether to drop episodes over it is the caller's business; prepare only acts
      on it when ``Segment.exact_resume``). An extension turn with unknown reuse (resume.pos / cached_tokens) = unknown
      which positions were forwarded twice → raise, with the flag on or off alike: sglang fills cached_tokens for every
      finished request (an append always has meta_info), so missing it means a broken ledger, not an old fork's shape
      (measured 2026-10-18: 58 local ledgers, 35,347 extension appends, old fork and flag-off ones included, none
      missing it).
    """
    n = len(seg)
    if n < 2:
        raise LedgerFormatError(f"{seg.key}: segment length {n}, no position that can be forwarded")
    out = np.full((n, layers, top_k), -1, np.int16)
    covered = np.zeros(n, np.bool_)
    missing = np.zeros(n, np.int8)        # 1 + index into ROUTING_M_REASONS[:3]; 0 = not missing
    for a in seg.appends:
        p = a.prefix_len
        covered[p:] = False               # tokens from p on are new: earlier rows there belong to replaced tokens
        missing[p:] = 0
        lo, hi = a.routing_expected_start, min(p + a.tail_len + a.gen_len - 1, n)
        ref = a.routing
        if ref is None:
            gap = missing[lo:hi]
            gap[~covered[lo:hi]] = 1 if a.routing_error is not None else 2
            continue
        if tuple(ref.shape[1:]) != (layers, top_k) or np.dtype(ref.dtype) != np.int16:
            raise LedgerFormatError(f"{seg.key}: routing {ref.dtype} {list(ref.shape)}, expected int16 [n, {layers}, {top_k}]")
        rows = read_routing(directory, ref, seg.key)
        e = min(ref.start + ref.n_tokens, n)
        out[ref.start:e] = rows[:e - ref.start]
        covered[ref.start:e] = True
        missing[ref.start:e] = 0
        if ref.start > lo:
            gap = missing[lo:min(ref.start, n)]
            gap[~covered[lo:min(ref.start, n)]] = 3
    if covered[n - 1]:
        raise RuntimeError(f"{seg.key}: internal error: the segment end is covered by a routing record")
    status = np.full(n, ROUTING_R, np.uint8)
    status[n - 1] = ROUTING_N
    reasons: dict[str, int] = {}
    unc = ~covered
    unc[n - 1] = False
    orphan = unc & (missing == 0)
    if orphan.any():
        first = int(np.flatnonzero(orphan)[0])
        raise LedgerFormatError(f"{seg.key}: position {first} has no routing and is not in any record's missing-routing range"
                                f" (N outside the segment end, {int(orphan.sum())} in total)")
    for code, name in enumerate(ROUTING_M_REASONS[:3], 1):
        if (hit := int((unc & (missing == code)).sum())):
            reasons[name] = hit
    status[unc] = ROUTING_M
    out[~covered] = -1
    idx = np.flatnonzero(covered)
    rows = out[idx]
    sentinel = (rows == -1).any(axis=(1, 2))
    oor = ((rows < -1) | (rows >= num_experts)).any(axis=(1, 2)) & ~sentinel
    zero = (rows == 0).all(axis=2).any(axis=1) & ~sentinel & ~oor
    srt = np.sort(rows, axis=2)
    dup = (srt[:, :, 1:] == srt[:, :, :-1]).any(axis=(1, 2)) & ~sentinel & ~oor & ~zero
    for name, hit, code in (("sentinel", sentinel, ROUTING_M), ("out_of_range", oor, ROUTING_X),
                            ("zero_row", zero, ROUTING_X), ("duplicate", dup, ROUTING_X)):
        if hit.any():
            reasons[name] = int(hit.sum())
            status[idx[hit]] = code
    rf = segment_reforward(seg)
    if rf.unknown_windows:
        raise LedgerFormatError(f"{seg.key}: append {rf.unknown_windows[:5]} ({len(rf.unknown_windows)} extensions in total) "
                                f"has no resume.pos and no usage.cached_tokens: unknown which positions these turns recomputed "
                                f"(which positions were forwarded twice)")
    twice = (status == ROUTING_R) & (rf.code > 0)
    status[twice] = rf.code[twice]
    return RoutingStatus(routing=out, status=status, reasons=reasons)


# ----------------------------------------------------------------------------- decode helpers


def _b64(value: str, dtype: str, where: str) -> np.ndarray:
    if not isinstance(value, str):
        raise LedgerFormatError(f"{where}: expected a base64 string, got {type(value).__name__}")
    raw = base64.b64decode(value, validate=True)
    if len(raw) % 4:
        raise LedgerFormatError(f"{where}: {len(raw)} bytes is not a multiple of 4")
    return np.frombuffer(raw, dtype=dtype).astype(dtype.lstrip("<"), copy=True)


def _sampling_support(r, gen: np.ndarray, id_bound: int, where: str) -> tuple[SamplingSupport | None, str | None]:
    """An append's ``sampling_mask`` → ``SamplingSupport``; raises on format errors, returns (None, reason) when the
    ``sampling_mask`` invariant is violated."""
    if not isinstance(r, dict) or set(r) != {"lens", "ids", "logprobs"}:
        raise LedgerFormatError(f"{where}: sampling_mask must be {{lens, ids, logprobs}}, got {r!r}")
    lens = _b64(r["lens"], "<i4", f"{where} sampling_mask.lens")
    ids = _b64(r["ids"], "<i4", f"{where} sampling_mask.ids")
    logprobs = _b64(r["logprobs"], "<f4", f"{where} sampling_mask.logprobs")
    n = len(gen)
    if len(lens) != n or len(logprobs) != n:
        return None, f"{len(lens)} supports, {len(logprobs)} support logprobs, gen length {n}"
    if n and int(lens.min()) < 1:
        return None, f"minimum support length is {int(lens.min())}"
    offsets = np.zeros(n + 1, np.int64)
    np.cumsum(lens, out=offsets[1:])
    if offsets[-1] != len(ids):
        return None, f"support lengths sum to {int(offsets[-1])} ≠ ids length {len(ids)}"
    if ids.size and (int(ids.min()) < 0 or int(ids.max()) >= id_bound):
        return None, f"support id outside [0, {id_bound})"
    if not np.isfinite(logprobs).all() or (n and float(logprobs.max()) > 0):
        return None, "support logprobs are not finite values ≤ 0"
    row = np.repeat(np.arange(n), lens)
    hit = np.zeros(n, np.int64)
    np.add.at(hit, row, ids == gen[row])
    if (hit != 1).any():
        j = int(np.flatnonzero(hit != 1)[0])
        return None, f"token {j} ({int(gen[j])}) appears in its support {int(hit[j])} times"
    order = np.lexsort((ids, row))
    dup = (row[order][1:] == row[order][:-1]) & (ids[order][1:] == ids[order][:-1])
    if dup.any():
        return None, f"token {int(row[order][1:][dup][0])}'s support has repeated ids"
    return SamplingSupport(lens=lens, offsets=offsets, ids=ids, logprobs=logprobs), None


def _int(value, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LedgerFormatError(f"{where}: expected int, got {value!r}")
    return value


def _weight_versions(ev: dict, n_gen: int, where: str
                     ) -> tuple[tuple[tuple[int, str], ...] | None, str | None, tuple[str, ...] | None]:
    """An append's weight versions → (``((gen index, version), …)``, None, prefill_versions); neither key (pre-fix
    ledger) → (None, None, None); ``weight_versions`` null (not captured for this record) → (None,
    ``weight_versions_error``, None). prefill_versions: the versions of every piece from gen position 0 on (empty
    intervals included), deduplicated, in order. See the module docstring."""
    one, many = "weight_version" in ev, "weight_versions" in ev
    if one and many:
        raise LedgerFormatError(f"{where}: weight_version and weight_versions both present")
    if "weight_versions_error" in ev and not many:
        raise LedgerFormatError(f"{where}: weight_versions_error without weight_versions")
    if not one and not many:
        return None, None, None
    if one:
        v = ev["weight_version"]
        if not isinstance(v, str) or not v:
            raise LedgerFormatError(f"{where}: weight_version must be a non-empty string, got {v!r}")
        return ((0, v),), None, (v,)
    spans = ev["weight_versions"]
    error = ev.get("weight_versions_error")
    if spans is None:
        if not isinstance(error, str) or not error:
            raise LedgerFormatError(f"{where}: weight_versions is null but there is no weight_versions_error")
        return None, error, None
    if error is not None:
        raise LedgerFormatError(f"{where}: weight_versions has a value but also weight_versions_error={error!r}")
    if not isinstance(spans, list) or not spans:
        raise LedgerFormatError(f"{where}: weight_versions must be a non-empty [{{version, start, end}}, …], got {spans!r}")
    out: list[tuple[int, str]] = []
    prefill: list[str] = []
    pos = 0
    for item in spans:
        if (not isinstance(item, dict) or set(item) != {"version", "start", "end"}
                or not isinstance(item["version"], str) or not item["version"]
                or any(isinstance(item[k], bool) or not isinstance(item[k], int) for k in ("start", "end"))):
            raise LedgerFormatError(f"{where}: weight_versions item {item!r} is not "
                                    f"{{version: non-empty string, start: int, end: int}}")
        if item["start"] != pos or item["end"] < item["start"]:
            raise LedgerFormatError(f"{where}: weight_versions {spans!r} does not tile [0, {n_gen}) in order")
        if item["start"] == 0 and item["version"] not in prefill:
            prefill.append(item["version"])               # every span from gen 0: the prefill ran under one of them
        pos = item["end"]
        if item["end"] == item["start"]:
            continue                                      # an update between two forwards leaves an empty span
        if out and out[-1][1] == item["version"]:
            continue                                      # same version again after an empty span: one run
        out.append((item["start"], item["version"]))
    if pos != n_gen:
        raise LedgerFormatError(f"{where}: weight_versions {spans!r} does not tile [0, {n_gen}) in order")
    if not out:                                           # n_gen == 0: the version is moot, keep the recorded one
        out.append((0, spans[0]["version"]))
    return tuple(out), None, tuple(prefill)


def server_exact_resume(server: dict, where: str) -> bool:
    """``server_args.session_exact_resume`` of ``server.json``: key absent (old fork) → False; not a bool, or
    ``server_args`` not an object = format error (the fork always writes ``server_args``)."""
    args = server.get("server_args")
    if not isinstance(args, dict):
        raise LedgerFormatError(f"{where}: server_args must be an object, got {args!r}")
    flag = args.get("session_exact_resume", False)
    if not isinstance(flag, bool):
        raise LedgerFormatError(f"{where}: server_args.session_exact_resume must be a bool, got {flag!r}")
    return flag


def _resume_value(r, where: str, *, attempt: bool) -> Resume:
    """The value (object) of ``resume`` → ``Resume``. A wrong shape is a format error; the contract is in
    ``resume_problems``."""
    if not isinstance(r, dict) or set(r) != set(RESUME_KEYS):
        raise LedgerFormatError(f"{where}: resume must be {{{', '.join(RESUME_KEYS)}}}, got {r!r}")
    if r["mode"] not in RESUME_MODES:
        raise LedgerFormatError(f"{where}: resume.mode={r['mode']!r} not in {RESUME_MODES}")
    if r["reason"] is not None and r["reason"] not in RESUME_REASONS:
        raise LedgerFormatError(f"{where}: resume.reason={r['reason']!r} is neither null nor in {RESUME_REASONS}")
    pos = r["pos"]
    if pos is not None and (isinstance(pos, bool) or not isinstance(pos, int) or pos < 0):
        raise LedgerFormatError(f"{where}: resume.pos must be a non-negative int or null, got {pos!r}")
    version = r["prefill_version"]
    if not (version is None and attempt) and (not isinstance(version, str) or not version):
        raise LedgerFormatError(f"{where}: resume.prefill_version must be a non-empty string (only an attempt may have "
                                f"null), got {version!r}")
    return Resume(mode=r["mode"], reason=r["reason"], pos=pos, prefill_version=version)


def _flag_consistent(resume: Resume | None, exact_resume: bool, where: str) -> None:
    """A server with the flag off never resumes exactly: its resume (if any) can only be none (module docstring)."""
    if not exact_resume and resume is not None and resume.mode != "none":
        raise LedgerFormatError(f"{where}: server has session_exact_resume off (server.json server_args), yet resume.mode is "
                                f"{resume.mode}")


def _append_resume(ev: dict, where: str, exact_resume: bool) -> Resume | None:
    """An append's ``resume``: required (object) with the flag True; with the flag False it may be absent (→ None, like
    the old fork), and if present can only be none."""
    if "resume" not in ev:
        if exact_resume:
            raise LedgerFormatError(f"{where}: server has session_exact_resume on (server.json server_args), yet the append "
                                    f"has no resume")
        return None
    resume = _resume_value(ev["resume"], where, attempt=False)
    _flag_consistent(resume, exact_resume, where)
    return resume


def _attempt(ev: dict, where: str, seg: Segment | None, exact_resume: bool) -> AttemptRecord:
    """attempt event → ``AttemptRecord``. ``usage`` / ``resume`` may lack the key (old fork) or be null (no meta_info:
    orphaned, an abort that never reached the scheduler, a fork with the flag off); a value gets its shape checked; with
    the flag off resume can only be none. ``prefix_len`` optional. ``seg`` = the last segment opened for this sid; its
    length is used as S only while it is still open."""
    for key in ("input_len", "gen_len"):
        if _int(ev[key], f"{where} attempt.{key}") < 0:
            raise LedgerFormatError(f"{where}: attempt.{key}={ev[key]} is negative")
    prefix_len = ev.get("prefix_len")
    if prefix_len is not None and _int(prefix_len, f"{where} attempt.prefix_len") < 0:
        raise LedgerFormatError(f"{where}: attempt.prefix_len={prefix_len} is negative")
    if not isinstance(ev["kind"], str):
        raise LedgerFormatError(f"{where}: attempt.kind must be a string, got {ev['kind']!r}")
    if "resume" not in ev:
        resume, state = None, ATTEMPT_RESUME_ABSENT
    elif ev["resume"] is None:
        resume, state = None, ATTEMPT_RESUME_NULL
    else:
        resume, state = _resume_value(ev["resume"], where, attempt=True), ATTEMPT_RESUME_PRESENT
        _flag_consistent(resume, exact_resume, where)
    usage = ev.get("usage")
    live = seg is not None and seg.status == "open"
    return AttemptRecord(sid=ev["sid"], rid=ev["rid"], kind=ev["kind"], input_len=ev["input_len"], gen_len=ev["gen_len"],
                         cached_tokens=None if usage is None else _cached_tokens(usage, where),
                         resume=resume, resume_state=state, prefix_len=prefix_len,
                         s_len=len(seg) if live else None, seg=seg.seg if live else None)


def _cached_tokens(usage, where: str) -> int | None:
    """``usage.cached_tokens``: non-negative int; missing or null → None (the fork copies meta_info, which may lack it;
    no guessing). usage must be an object (an attempt's null usage is handled in ``_attempt``)."""
    if not isinstance(usage, dict):
        raise LedgerFormatError(f"{where}: usage must be an object, got {usage!r}")
    value = usage.get("cached_tokens")
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LedgerFormatError(f"{where}: usage.cached_tokens must be a non-negative int or null, got {value!r}")
    return value


def _replay(replay, where: str) -> dict:
    """An append's ``replay``, returned as is after checking its shape (see the module docstring)."""
    if not isinstance(replay, dict) or set(replay) != {"spans", "skip", "lcp0"}:
        raise LedgerFormatError(f"{where}: replay must be {{spans, skip, lcp0}}, got {replay!r}")
    if replay["skip"] is not None and not isinstance(replay["skip"], str):
        raise LedgerFormatError(f"{where}: replay.skip must be a string or null, got {replay['skip']!r}")
    _int(replay["lcp0"], f"{where} replay.lcp0")
    if not isinstance(replay["spans"], list):
        raise LedgerFormatError(f"{where}: replay.spans must be a list")
    for span in replay["spans"]:
        own = isinstance(span, dict) and span.get("kind") == "own_turn"
        keys = set(REPLAY_SPAN_KEYS) | (set(OWN_TURN_SPAN_KEYS) if own else set())
        if not isinstance(span, dict) or set(span) - {"kind", *(("sticky",) if own else ())} != keys:
            raise LedgerFormatError(f"{where}: replay span {span!r} keys must be {REPLAY_SPAN_KEYS} (+ optional kind; "
                                    f"kind own_turn adds {OWN_TURN_SPAN_KEYS} and optional sticky)")
        if own and "sticky" in span and (span["sticky"] is not True or span["closure"] != 0):
            raise LedgerFormatError(f"{where}: own_turn span {span!r}: sticky can only be true, and closure must be 0")
        for key in REPLAY_SPAN_KEYS:
            _int(span[key], f"{where} replay span {key}")
        if "kind" in span and span["kind"] not in REPLAY_SPAN_KINDS:
            raise LedgerFormatError(f"{where}: unknown replay span kind {span['kind']!r} (known: {REPLAY_SPAN_KINDS})")
        if own:
            closure = _int(span["closure"], f"{where} replay span closure")
            client_len = _int(span["client_len"], f"{where} replay span client_len")
            sha = span["client_sha256"]
            if closure < 0 or client_len != span["key_end"] - span["s"] or not (
                    isinstance(sha, str) and len(sha) == 64 and all(c in "0123456789abcdef" for c in sha)):
                raise LedgerFormatError(f"{where}: own_turn span {span!r}: closure must be ≥ 0, client_len == key_end − s, "
                                        "client_sha256 must be 64 lowercase hex")
            if span["delta"] != span["rec_end"] - span["s"] + closure - client_len:
                raise LedgerFormatError(f"{where}: own_turn span {span!r}: delta must equal "
                                        "(rec_end − s + closure) − client_len")
    return replay


def _require(ev: dict, keys: Iterable[str], where: str) -> None:
    missing = [k for k in keys if k not in ev]
    if missing:
        raise LedgerFormatError(f"{where}: missing keys {missing}")


def _events_files(path: Path) -> list[Path]:
    found = {}
    for p in path.iterdir():
        m = _EVENTS_RE.fullmatch(p.name)
        if m:
            found[int(m.group(1))] = p
    if not found:
        raise LedgerFormatError(f"{path}: no events.<n>.jsonl")
    if sorted(found) != list(range(len(found))):
        raise LedgerFormatError(f"{path}: events file numbers not consecutive {sorted(found)} (files missing)")
    return [found[i] for i in range(len(found))]


class _EventStream:
    """Read events.<n>.jsonl line by line (files roll over at 1 GiB; never read whole into memory). ``truncated`` /
    ``n`` are meaningful only after reading to the end."""

    def __init__(self, path: Path):
        self.path = path
        self.files = _events_files(path)
        self.truncated = False
        self.n = 0
        self.read_bytes = 0

    def __iter__(self):
        for fi, fp in enumerate(self.files):
            with open(fp, "rb") as f:
                for li, line in enumerate(f, 1):
                    self.read_bytes += len(line)
                    where = f"{self.path.name}/{fp.name}:{li}"
                    if not line.endswith(b"\n"):
                        # the iterator only yields a line without "\n" at EOF: a line still being written (or cut by
                        # a crash) -- allowed only at the end of the last file, ignored even if it happens to parse
                        if fi != len(self.files) - 1:
                            raise LedgerFormatError(f"{where}: not the last events file, yet it ends with an "
                                                    f"unterminated partial line")
                        self.truncated = True
                        break
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError as e:
                        raise LedgerFormatError(f"{where}: bad JSON: {e}") from e
                    if not isinstance(ev, dict) or "ev" not in ev:
                        raise LedgerFormatError(f"{where}: not an event object")
                    if ev["ev"] not in EVENT_KEYS:
                        raise LedgerFormatError(f"{where}: unknown event type {ev['ev']!r}")
                    _require(ev, EVENT_KEYS[ev["ev"]], where)
                    self.n += 1
                    yield where, ev


# ----------------------------------------------------------------------------- reconstruction


class _Builder:
    def __init__(self, path: Path, vocab: Vocab, strict: bool, exact_resume: bool):
        self.path = path
        self.vocab = vocab
        self.strict = strict
        self.exact_resume = exact_resume
        self.segments: dict[int, Segment] = {}
        self.errors: list[LedgerInvariantError] = []
        self.attempts: list[dict] = []
        self.attempt_records: list[AttemptRecord] = []
        self.routing_sizes: dict[str, int] = {}

    def fail(self, seg: Segment | None, code: str, where: str, message: str) -> None:
        err = LedgerInvariantError(code, where, message)
        if self.strict:
            raise err
        if seg is None:
            self.errors.append(err)
        else:
            seg.status = "broken"
            seg.error = err

    def _segment_for(self, ev: dict, where: str) -> Segment | None:
        seg_id = _int(ev["seg"], f"{where} seg")
        seg = self.segments.get(seg_id)
        if seg is None:
            self.fail(None, "unknown_segment", where, f"seg {seg_id} has no seg_open")
            return None
        if seg.status == "broken":
            seg.skipped_events += 1
            return None
        if seg.status == "closed":
            self.fail(None, "closed_segment", where, f"seg {seg_id} already got seg_close")
            return None
        if ev["sid"] != seg.sid:
            self.fail(seg, "sid_mismatch", where, f"sid {ev['sid']!r} ≠ seg_open's {seg.sid!r}")
            return None
        return seg

    def seg_open(self, ev: dict, where: str, instance: str) -> None:
        seg_id = _int(ev["seg"], f"{where} seg")
        sid, cause = ev["sid"], ev["cause"]
        split_reason = parse_open_cause(cause, where)
        if (sid is None) != (cause == "single"):
            raise LedgerFormatError(f"{where}: sid={sid!r} and cause={cause!r} disagree (null sid ⇔ single)")
        if sid is not None and not isinstance(sid, str):
            raise LedgerFormatError(f"{where}: sid must be a string or null")
        if seg_id in self.segments:
            self.fail(None, "duplicate_segment", where, f"seg {seg_id} opened twice")
            return
        self.segments[seg_id] = Segment(instance=instance, seg=seg_id, sid=sid, cause=cause,
                                        split_reason=split_reason, t_open=float(ev["t"]),
                                        exact_resume=self.exact_resume)

    def append(self, ev: dict, where: str) -> None:
        seg = self._segment_for(ev, where)
        if seg is None:
            return
        v = self.vocab
        k = _int(ev["k"], f"{where} k")
        prefix_len = _int(ev["prefix_len"], f"{where} prefix_len")
        rewind = _int(ev["rewind"], f"{where} rewind")
        if rewind < 0:
            raise LedgerFormatError(f"{where}: rewind={rewind} is negative")
        tail = _b64(ev["tail"], "<i4", f"{where} tail")
        gen = _b64(ev["gen"], "<i4", f"{where} gen")
        sub = ev["stop_substituted"]
        if not isinstance(sub, bool):
            raise LedgerFormatError(f"{where}: stop_substituted must be a bool")
        finish = ev["finish"]
        if not isinstance(finish, dict) or "type" not in finish:
            raise LedgerFormatError(f"{where}: finish must be an object with type, got {finish!r}")

        for name, arr in (("tail", tail), ("gen", gen)):
            if arr.size and (int(arr.min()) < 0 or int(arr.max()) >= v.id_bound):
                bad = arr[(arr < 0) | (arr >= v.id_bound)]
                return self.fail(seg, "ids_in_vocab", where,
                                 f"{name} has {bad.size} ids outside [0, {v.id_bound}), e.g. {bad[:3].tolist()}")
        if k != len(seg.appends):
            return self.fail(seg, "append_order", where, f"k={k}, but {len(seg.appends)} records precede it")
        n_old = len(seg)
        prev = seg.appends[-1] if seg.appends else None
        if k == 0:
            if prefix_len != 0 or rewind != 0 or sub:
                return self.fail(seg, "prefix_len", where,
                                 f"a segment's first record needs prefix_len=0, rewind=0, no stop_substituted "
                                 f"({prefix_len}, {rewind}, {sub})")
        elif prefix_len + rewind != n_old:
            return self.fail(seg, "prefix_len", where, f"prefix_len={prefix_len} + rewind={rewind} ≠ S length {n_old}")
        # A rewind can only be inside the previous record's generated span. tail may be 0: the last turn resent verbatim
        # (codex drops a compaction reply that carries a tool call and asks again; the spec's "including a request exactly
        # equal to the previous turn's input"), and this record regenerates from the end of the previous record's input.
        if rewind and rewind > prev.gen_len:
            return self.fail(seg, "rewind", where,
                             f"rewind={rewind}: previous gen length {prev.gen_len}, this tail length {len(tail)} "
                             f"(a rewind can only be inside the previous record's generated span)")
        # gen spans do not overlap: the newly written range [prefix_len, …) may only cover the rewound tail of the previous
        # gen (the rewind rule above already covers this; kept as a defence).
        for span in seg.gen_spans:
            if span.end > prefix_len:
                allowed = (prev is not None and span is seg.gen_spans[-1] and span.k == prev.k
                           and span.end == n_old and span.start <= prefix_len)
                if not allowed:
                    return self.fail(seg, "gen_overlap", where,
                                     f"prefix_len={prefix_len} covers append {span.k}'s gen span [{span.start}, {span.end})")
        if sub:
            old = int(seg.ids[-1])
            matched = prev.finish.get("matched")       # a length finish has no matched: the type check below catches it
            if (rewind != 1 or prev.finish["type"] != "stop" or old == v.im_end
                    or (isinstance(matched, int) and matched != old) or int(tail[0]) != v.im_end):
                return self.fail(seg, "stop_substituted", where,
                                 f"rewind={rewind}, replaced token {old} (previous finish={prev.finish}), "
                                 f"tail[0]={int(tail[0]) if tail.size else None}, expected <|im_end|>={v.im_end}")
        ftype = finish["type"]
        if ftype not in ("stop", "length"):
            return self.fail(seg, "finish", where, f"append finish.type={ftype!r} (an abort must be written as an attempt)")
        if ftype == "stop":
            if "matched" not in finish:
                raise LedgerFormatError(f"{where}: finish stop without matched")
            matched = finish["matched"]
            if isinstance(matched, int) and not isinstance(matched, bool) and (not gen.size or int(gen[-1]) != matched):
                return self.fail(seg, "finish", where,
                                 f"finish.matched={matched}, but the last gen token is {int(gen[-1]) if gen.size else None}")

        errors = {}
        for name in ("logprobs", "routing", "top_logprobs", "sampling_mask"):
            err = ev.get(f"{name}_error")      # optional: the fork says why this item is missing (module docstring)
            if err is not None and (not isinstance(err, str) or ev.get(name) is not None):
                raise LedgerFormatError(f"{where}: {name}_error={err!r} must be a string, and {name} must then be null")
            errors[name] = err

        logprobs = None
        if ev["logprobs"] is not None:
            logprobs = _b64(ev["logprobs"], "<f4", f"{where} logprobs")
            if len(logprobs) != len(gen) or np.isnan(logprobs).any():
                return self.fail(seg, "logprobs_len", where,
                                 f"{len(logprobs)} logprobs (NaN {int(np.isnan(logprobs).sum())}), gen length {len(gen)}")

        sampling_mask = None
        if ev.get("sampling_mask") is not None:           # optional key: absent before --session-ledger-sampling-mask
            sampling_mask, why = _sampling_support(ev["sampling_mask"], gen, v.id_bound, where)
            if why is not None:
                return self.fail(seg, "sampling_mask", where, why)

        weight_versions, weight_versions_error, prefill_versions = _weight_versions(ev, len(gen), where)
        resume = _append_resume(ev, where, self.exact_resume)
        cached_tokens = _cached_tokens(ev["usage"], where)

        top_logprobs = None
        if ev.get("top_logprobs") is not None:            # optional key: ledgers from before top-k lack it
            top_logprobs = self._topk(seg, ev["top_logprobs"], len(gen), where)
            if top_logprobs is None:
                return

        routing = None
        if ev["routing"] is not None:
            routing = self._routing(seg, ev["routing"], prefix_len, len(tail) + len(gen), where)
            if routing is None:
                return
            before = prev.routing if prev is not None else None
            # position prefix_len-1 is the last token kept: the previous record forwarded it; if this record read it
            # back from cache the rows are equal, if it recomputed it (no radix, session released) they may differ
            # (stop_substituted is the rewind=1 special case). A record whose start is later than the spec's (late
            # split) does not contain this position, so there is nothing to compare.
            p = prefix_len - 1
            if rewind and before is not None and routing.start <= p < routing.start + routing.n_tokens:
                row = p - before.start
                if not 0 <= row < before.n_tokens:
                    return self.fail(seg, "routing_overlap", where,
                                     f"previous routing [{before.start}, {before.start + before.n_tokens}) does not "
                                     f"contain position {p}")
                a = self._routing_row(before, row)
                b = self._routing_row(routing, p - routing.start)
                if not np.array_equal(a, b):
                    seg.routing_overlap_mismatch += 1   # a recomputed row: the later record wins

        tail_start, tail_end = prefix_len, prefix_len + len(tail)
        media = []
        for mi, m in enumerate(ev["media"]):
            ref = self._media(m, k, f"{where} media[{mi}]")
            if ref is None:
                continue                          # unmapped, nothing at or after prefix_len (_media)
            if ref.unmapped and m.get("n_items") == 0:
                # pad tokens in this tail with no media item in the request: sampled pad text sent back (stray_media_pads)
                seg.stray_media_pads += [(ref.modality, a, b) for a, b in ref.ranges]
            for a, b in ref.ranges:
                if not (tail_start <= a <= b < tail_end):
                    return self.fail(seg, "media_range", where,
                                     f"{ref.modality} range [{a}, {b}] is not inside this record's tail "
                                     f"[{tail_start}, {tail_end})")
                run = tail[a - tail_start: b - tail_start + 1]
                if not (run == v.pad(ref.modality)).all():
                    return self.fail(seg, "media_range", where,
                                     f"{ref.modality} range [{a}, {b}] is not all pad {v.pad(ref.modality)}")
            if ref.sha256 is not None and not (self.path / "media" / ref.sha256).is_file():
                return self.fail(seg, "media_blob", where, f"media/{ref.sha256} does not exist")
            media.append(ref)

        # commit: rewind the previous gen's tail (the original gen stays in the previous Append.gen), then append
        # tail + gen.
        if rewind:
            last = seg.gen_spans[-1]
            last.end -= rewind
            last.dropped = rewind
            if last.end == last.start:
                seg.gen_spans.pop()
            if len(seg._chunks[-1]) < rewind:
                raise RuntimeError(f"{where}: internal error: the last chunk has {len(seg._chunks[-1])} tokens, not enough "
                                   f"to rewind {rewind}")
            seg._chunks[-1] = seg._chunks[-1][: len(seg._chunks[-1]) - rewind]
            seg._n -= rewind
        seg._chunks.extend([tail, gen])
        seg._n += len(tail) + len(gen)
        seg._ids = None
        if len(gen):
            seg.gen_spans.append(GenSpan(start=tail_end, end=tail_end + len(gen), k=k))
        seg.appends.append(Append(
            k=k, rid=ev["rid"], t0=float(ev["t0"]), t1=float(ev["t1"]), prefix_len=prefix_len, rewind=rewind,
            tail_len=len(tail), gen=gen, finish=finish, stop_substituted=sub, sampling=ev["sampling"],
            usage=ev["usage"], replay=_replay(ev["replay"], where), media=media, logprobs=logprobs, logprobs_error=errors["logprobs"],
            routing=routing, routing_error=errors["routing"], weight_versions=weight_versions,
            weight_versions_error=weight_versions_error, top_logprobs=top_logprobs,
            top_logprobs_error=errors["top_logprobs"], prefill_versions=prefill_versions, resume=resume,
            cached_tokens=cached_tokens, sampling_mask=sampling_mask, sampling_mask_error=errors["sampling_mask"]))
        if routing is not None and routing.start != max(prefix_len - 1, 0):
            seg.routing_start_shifted += 1

    def _topk(self, seg: Segment, r: dict, n_gen: int, where: str) -> TopkRef | None:
        if not isinstance(r, dict):
            raise LedgerFormatError(f"{where}: top_logprobs must be an object or null")
        _require(r, TOPK_KEYS, f"{where} top_logprobs")
        k, n, nbytes = (_int(r[x], f"{where} top_logprobs.{x}") for x in ("k", "n", "bytes"))
        if n != n_gen or k < 1:
            return self.fail(seg, "topk_len", where, f"top_logprobs k={k} n={n}, gen length {n_gen}")
        if r["layout"] != TOPK_LAYOUT:
            raise LedgerFormatError(f"{where}: top_logprobs.layout={r['layout']!r} not recognised (only {TOPK_LAYOUT!r} is read)")
        if nbytes != n * k * 6 + n * 5:
            return self.fail(seg, "topk_file", where,
                             f"top_logprobs.bytes={nbytes}, n×k×(4+2)+n×(4+1)={n * k * 6 + n * 5}")
        name, offset = r["file"], _int(r["offset"], f"{where} top_logprobs.offset")
        if not isinstance(name, str) or Path(name).name != name:
            raise LedgerFormatError(f"{where}: top_logprobs.file={name!r} must be a file name in the instance directory")
        size = self.routing_sizes.get(name, -1)
        if size < 0 or offset + nbytes > size:
            fp = self.path / name                       # a ledger being written: as for routing, re-stat
            size = self.routing_sizes[name] = fp.stat().st_size if fp.is_file() else -1
        if size < 0 or offset < 0 or offset + nbytes > size:
            return self.fail(seg, "topk_file", where, f"{name} ({size} bytes) cannot hold offset={offset} + {nbytes}")
        return TopkRef(file=name, offset=offset, nbytes=nbytes, k=k, n=n)

    def _routing_row(self, ref: RoutingRef, row: int) -> np.ndarray:
        item = np.dtype(ref.dtype).itemsize * ref.shape[1] * ref.shape[2]
        with open(self.path / ref.file, "rb") as f:
            f.seek(ref.offset + row * item)
            raw = f.read(item)
        return np.frombuffer(raw, dtype=np.dtype(ref.dtype)).reshape(ref.shape[1:])

    def _routing(self, seg: Segment, r: dict, prefix_len: int, n_new: int, where: str) -> RoutingRef | None:
        if not isinstance(r, dict):
            raise LedgerFormatError(f"{where}: routing must be an object or null")
        _require(r, ROUTING_KEYS, f"{where} routing")
        n_tokens = _int(r["n_tokens"], f"{where} routing.n_tokens")
        shape = r["shape"]
        dtype = r["dtype"]
        if (not isinstance(shape, list) or len(shape) != 3
                or not all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in shape)):
            return self.fail(seg, "routing_file", where, f"routing.shape={shape!r} is not [n_tokens, layers, top_k]")
        if not isinstance(dtype, str) or np.dtype(dtype).kind not in "iu":
            return self.fail(seg, "routing_file", where, f"routing.dtype={dtype!r} is not an integer type")
        start = _int(r["start"], f"{where} routing.start")
        end = prefix_len + n_new - 1
        # the fork picks the start at intake: a late branch shifts it, the end is always this request's last
        # forwarded position (module docstring, routing_len)
        if shape[0] != n_tokens or start + n_tokens != end or not 0 <= start <= end:
            return self.fail(seg, "routing_len", where,
                             f"routing start={start} n_tokens={n_tokens} shape={shape}; prefix_len={prefix_len}, "
                             f"tail+gen={n_new} → must cover up to {end} (spec start {max(prefix_len - 1, 0)})")
        nbytes = math.prod(shape) * np.dtype(dtype).itemsize
        if r["nbytes"] != nbytes:
            return self.fail(seg, "routing_file", where, f"routing.nbytes={r['nbytes']}, shape×dtype={nbytes}")
        prev = next((a.routing for a in reversed(seg.appends) if a.routing is not None), None)
        if prev is not None and (prev.shape[1:] != tuple(shape[1:]) or prev.dtype != dtype):
            return self.fail(seg, "routing_file", where,
                             f"routing shape/dtype {shape}/{dtype} differs from the previous record "
                             f"({list(prev.shape)}/{prev.dtype})")
        name = r["file"]
        offset = _int(r["offset"], f"{where} routing.offset")
        if not isinstance(name, str) or Path(name).name != name:
            raise LedgerFormatError(f"{where}: routing.file={name!r} must be a file name in the instance directory")
        size = self.routing_sizes.get(name, -1)
        if size < 0 or offset + nbytes > size:
            # a ledger being written grows its routing file after the size was cached: the writer puts the routing
            # bytes on disk before the event line, so a fresh stat must cover every event read so far
            fp = self.path / name
            size = self.routing_sizes[name] = fp.stat().st_size if fp.is_file() else -1
        if size < 0 or offset < 0 or offset + nbytes > size:
            return self.fail(seg, "routing_file", where,
                             f"{name} ({size} bytes) cannot hold offset={offset} + {nbytes}")
        return RoutingRef(file=name, offset=offset, nbytes=nbytes, start=start, n_tokens=n_tokens,
                          shape=tuple(shape), dtype=dtype)

    @staticmethod
    def _media(m: dict, k: int, where: str) -> MediaRef | None:
        """None: the fork's unmapped item has no pad run in this record's tail -- the request carried no data of this
        modality (n_items 0) and all pads are in the prefix (already recorded by earlier appends), yet the fork still
        writes an item with ``ranges: []`` (session_ledger._media filters out the runs before prefix_len first).
        2026-10-24 v4 r0 events.0.jsonl:22209 (the video of cd82__s1) was exactly this; audit raised on it and the whole
        run stopped."""
        if not isinstance(m, dict):
            raise LedgerFormatError(f"{where}: media must be an object")
        _require(m, ("modality", "ranges"), where)
        if m["modality"] not in MODALITIES:
            raise LedgerFormatError(f"{where}: modality {m['modality']!r}")
        ranges = m["ranges"]
        if m.get("unmapped", False) is True and ranges == [] and m.get("n_items") == 0:
            return None
        if (not isinstance(ranges, list) or not ranges
                or not all(isinstance(r, list) and len(r) == 2 and all(isinstance(x, int) for x in r) for r in ranges)):
            raise LedgerFormatError(f"{where}: ranges must be a non-empty [[a, b], …], got {ranges!r}")
        unmapped = m.get("unmapped", False) is True
        if unmapped:
            return MediaRef(modality=m["modality"], ranges=tuple((a, b) for a, b in ranges), sha256=None, mime=None,
                            grid_thw=None, unmapped=True, src=None, k=k)
        _require(m, ("sha256", "grid_thw"), where)
        sha = m["sha256"]
        if sha is None:
            _require(m, ("src",), where)
            mime = None
        else:
            if not (isinstance(sha, str) and re.fullmatch(r"[0-9a-f]{64}", sha)):
                raise LedgerFormatError(f"{where}: sha256 {sha!r}")
            _require(m, ("mime",), where)
            mime = m["mime"]
        if m["modality"] == "image" and len(ranges) != 1:
            raise LedgerFormatError(f"{where}: an image must have exactly one range, got {len(ranges)} ranges")
        grid = m["grid_thw"]
        return MediaRef(modality=m["modality"], ranges=tuple((a, b) for a, b in ranges), sha256=sha, mime=mime,
                        grid_thw=None if grid is None else tuple(int(x) for x in grid), unmapped=False,
                        src=m.get("src"), k=k)

    def seg_close(self, ev: dict, where: str) -> None:
        seg = self._segment_for(ev, where)
        if seg is None:
            return
        reason = ev["reason"]
        split = parse_close_reason(reason, where)
        if (reason == "single") != (seg.sid is None):
            raise LedgerFormatError(f"{where}: reason=single is only for segments with a null sid (sid={seg.sid!r})")
        if not self.finish_segment(seg, where):
            return
        if _int(ev["n_tokens"], f"{where} n_tokens") != len(seg):
            return self.fail(seg, "close_len", where, f"seg_close.n_tokens={ev['n_tokens']}, reconstructed {len(seg)}")
        got = ids_sha256(seg.ids)
        if ev["sha256"] != got:
            return self.fail(seg, "close_sha256", where, f"seg_close.sha256={ev['sha256']}, reconstructed {got}")
        seg.close = {"reason": reason, "split_reason": split, "t": float(ev["t"]), "n_tokens": len(seg), "sha256": got}
        seg.status = "closed"

    def finish_segment(self, seg: Segment, where: str) -> bool:
        """Closing check (close or EOF): pad runs and media ranges correspond one to one."""
        ids = seg.ids
        v = self.vocab
        for modality in MODALITIES:
            runs = set(_runs(ids, v.pad(modality)))
            sampled = {r for r in runs if any(sp.start <= r[0] and r[1] < sp.end for sp in seg.gen_spans)}
            if sampled:                           # the model sampled pad tokens: not media (Segment.stray_media_pads)
                seg.stray_media_pads += [(modality, a, b) for a, b in sorted(sampled)]
                runs -= sampled
            ranges = [r for m in seg.media if m.modality == modality for r in m.ranges]
            if len(ranges) != len(set(ranges)) or set(ranges) != runs:
                orphan = sorted(runs - set(ranges))[:3]
                partial = sorted(set(ranges) - runs)[:3]
                self.fail(seg, "media_coverage", where,
                          f"{modality}: {len(runs)} pad runs, {len(ranges)} ranges; uncovered {orphan}, "
                          f"not a whole run {partial}")
                return False
        return True


def _runs(ids: np.ndarray, pad: int) -> list[tuple[int, int]]:
    """Closed-interval runs of pad in ids."""
    hit = (ids == pad).astype(np.int8)
    if not hit.any():
        return []
    d = np.diff(np.concatenate(([0], hit, [0])))
    return [(int(a), int(b)) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1) - 1)]


def load_instance(path: str | Path, vocab: Vocab, *, strict: bool = True,
                  threads: set[str] | None = None) -> Instance:
    """Read an instance; optional threads reconstruct only the requested sessions.

    Event framing and global stream metadata remain checked. Token/routing invariants are checked
    only for reconstructed threads; callers requiring a collection audit must use the full reader.
    Filtering bounds decoded token memory/work, while JSON bytes still grow with ledger history.
    """
    path = Path(path)
    server_path = path / "server.json"
    if not server_path.is_file():
        raise LedgerFormatError(f"{path}: no server.json (not an instance directory)")
    server = json.loads(server_path.read_text())
    if not isinstance(server, dict) or server.get("instance") != path.name:
        raise LedgerFormatError(f"{server_path}: instance field {server.get('instance')!r} ≠ directory name {path.name!r}")
    stream = _EventStream(path)
    events = iter(stream)
    first = next(events, None)
    if first is None or first[1]["ev"] != "start":
        raise LedgerFormatError(f"{path}: first event is not start")
    start = first[1]
    if start["v"] != LEDGER_VERSION or start["instance"] != path.name:
        raise LedgerFormatError(f"{path}: start v={start['v']!r} instance={start['instance']!r}")
    exact_resume = server_exact_resume(server, str(server_path))
    b = _Builder(path, vocab, strict, exact_resume)
    lossy_dropped, last_status, ended = 0, None, False
    seen_segments: set[int] = set()
    open_by_sid: dict[str, Segment] = {}
    for where, ev in events:
        kind = ev["ev"]
        if kind == "start":
            raise LedgerFormatError(f"{where}: a second start event")
        if kind == "seg_open":
            sid, seg_id = ev["sid"], _int(ev["seg"], f"{where} seg")
            if seg_id in seen_segments:
                b.fail(None, "duplicate_segment", where, f"seg {seg_id} opened twice")
                continue
            seen_segments.add(seg_id)
            if threads is not None:
                if sid is not None and not isinstance(sid, str):
                    raise LedgerFormatError(f"{where}: seg_open.sid must be a string or null")
                if sid is None or split_sid(sid)[0] not in threads:
                    continue
        elif threads is not None and kind in ("append", "seg_close"):
            seg_id = _int(ev["seg"], f"{where} seg")
            if seg_id not in seen_segments:
                b.fail(None, "unknown_segment", where, f"seg {seg_id} has no seg_open")
                continue
            if seg_id not in b.segments:
                continue
        if kind == "seg_open":
            b.seg_open(ev, where, path.name)
            if ev["sid"] is not None and seg_id in b.segments:
                open_by_sid[ev["sid"]] = b.segments[seg_id]     # the session's S that a later attempt is measured against
        elif kind == "append":
            b.append(ev, where)
        elif kind == "seg_close":
            b.seg_close(ev, where)
        elif kind == "attempt":
            if ev["sid"] is not None and not isinstance(ev["sid"], str):
                raise LedgerFormatError(f"{where}: attempt.sid")
            if threads is None or (ev["sid"] is not None and split_sid(ev["sid"])[0] in threads):
                b.attempts.append(ev)
                b.attempt_records.append(_attempt(ev, where, open_by_sid.get(ev["sid"]), exact_resume))
        elif kind == "lossy":
            lossy_dropped += int(ev["dropped"])
        elif kind in ("status", "end"):
            if not isinstance(ev["status"], dict):
                raise LedgerFormatError(f"{where}: {kind}.status")
            last_status = ev["status"]
            ended = ended or kind == "end"
        else:
            raise LedgerFormatError(f"{where}: unknown event {kind!r}")
    for seg in b.segments.values():
        if seg.status == "open":
            b.finish_segment(seg, f"{path.name} EOF seg {seg.seg}")
    return Instance(path=path, name=path.name, server=server, start=start, segments=b.segments,
                    attempt_records=b.attempt_records,
                    attempts=b.attempts, errors=b.errors, truncated_tail=stream.truncated, n_events=stream.n,
                    exact_resume=exact_resume,
                    lossy_dropped=lossy_dropped, last_status=last_status, ended=ended,
                    events_read_bytes=stream.read_bytes, total_segments=len(seen_segments))


def instance_dirs(paths: Iterable[str | Path]) -> list[Path]:
    """Each path is an instance directory (has server.json) or a ledger root DIR (every instance directory under it is
    taken)."""
    out = []
    for p in map(Path, paths):
        if (p / "server.json").is_file():
            out.append(p)
            continue
        if not p.is_dir():
            raise FileNotFoundError(f"{p} does not exist")
        found = sorted(d for d in p.iterdir() if (d / "server.json").is_file())
        if not found:
            raise LedgerFormatError(f"{p}: neither an instance directory nor a directory containing any")
        out.extend(found)
    names = [d.name for d in out]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise LedgerFormatError(f"duplicate instances: {sorted(dup)}")
    return out


def load_ledger(paths: Iterable[str | Path], vocab: Vocab, *, strict: bool = True) -> list[Instance]:
    return [load_instance(d, vocab, strict=strict) for d in instance_dirs(paths)]


def iter_instances(paths: Iterable[str | Path], vocab: Vocab, *, strict: bool = True) -> Iterable[Instance]:
    """Read one instance at a time (for exports: memory only has to hold one instance's segments)."""
    for d in instance_dirs(paths):
        yield load_instance(d, vocab, strict=strict)


def iter_segments(instances: Iterable[Instance]) -> Iterable[Segment]:
    for inst in instances:
        yield from inst.segments.values()
