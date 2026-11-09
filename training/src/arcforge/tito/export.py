"""Ledger segments → SFT / DFlash / RL parquet. One row per segment. Input is always read with ``strict=True`` (broken
segments never get in).

Common rules:

- Empty segments (no append after seg_open) and segments without any gen are skipped (nothing to learn); the counts go
  into ``tito_export.json``.
- With a manifest, only segments that match a manifest thread are exported (the rest is other traffic on the same
  server), counted as ``unjoined``.
- Segments without seg_close (crash) are exported anyway, ``seg_closed=false``: S is assembled event by event; there is
  just no closing sha256 to check it against.
- The output directory must not exist (no mixing batches).

**SFT**: the columns are a superset of OUT_SCHEMA in ``work/qwen38fn_distill/rollout_runner.py:73-94``, ``ids`` +
``span_start/end/role`` (role: see roles.py). Three file-name prefixes:

- ``part-*.parquet``: segments that are pure text and pass ``work/qwen35_9b_sft/convert_distill_shards.py:_check_row``
  (:102-138) -- can be fed to convert directly (non-``--agentic`` mode).
- ``media-part-*.parquet``: segments with media (``n_media > 0``). The packing in ``src/arcforge/train/sft.py`` only
  takes text; a separate prefix means convert's ``part-*`` glob naturally never trains on image pads as text.
- ``other-part-*.parquet``: pure-text segments that convert would reject, with the reason in ``convert_issue``. What
  actually shows up is rewind: a turn truncated by max_tokens keeps only the part before the rewind point in the final
  S (role 1), and the ``\n</think>\n\n<|im_end|>`` filled in by the template is role 2, so the role-1 span in between
  neither ends with ``<|im_end|>`` nor is followed by it (convert :128-136 only lets the latter through). The data
  itself is correct (the filled-in tokens are not in the loss); convert's rules just did not anticipate it. Whether to
  take them is the downstream's decision.
Extra columns: instance / seg / sid / thread / window, fork and close reasons, per-token span_kind (run-length), media,
the join's task / reward. OUT_SCHEMA columns the ledger cannot provide (instr, grid_fmt, train_idx, code_* ...) are
filled with empty strings / -1; ``correct`` is null without an outcome.

**DFlash** (``part-*.parquet``): ``input_ids / loss_mask / n_prompt`` of ``work/qwen38fn_dflash``
(``data.validate_row``) + ``uid / effort / n_prompt / n_completion`` read by ``prepare_data.py``. ``loss_mask`` = gen
tokens; ``n_prompt`` = before the first gen. ``uid`` = ``sha256("tito-group-v1:" + group)``: prepare_data splits
train/heldout by uid, so it must be a **task-level** key (group = the joined task, otherwise the thread), putting every
segment of a task on the same side; the ``effort`` column holds ``tito:<instance>:<seg>`` -- prepare_data requires
(uid, effort) to be unique, and its effort is already the column that tells trajectories apart under one uid. Segments
with media are skipped (capture only takes 1-D positions, and pads in ``input_ids`` without pixels are garbage).

**RL** (``part-*.parquet``): no existing format; columns modelled on ``attic/b200/dsv4/tir_sft/toy_rl_rollout.py:136-168``
(old_logprobs, out_ids, routed_experts). Two levels:

- segment level: ids = the final S; ``gen_start/gen_end/gen_k`` are the spans in S that were really sampled (rewound
  tails not included), ``gen_logprobs`` concatenated in the same order.
- turn level (one ``turn_*`` per append with non-empty gen): ``turn_gen`` is the **original** sampled ids (including the
  tail rewound later), ``turn_logprobs`` its logprobs, ``turn_gen_start`` the start position at sampling time,
  ``turn_rewound`` the number of tokens the next append rewound. Policy gradient uses the turn level: context =
  ``ids[:turn_gen_start]`` (a rewind only touches the previous gen, so this part is unchanged in the final S), action =
  ``turn_gen``, old logprob = ``turn_logprobs``.
Routing is not put into the parquet; it stores the ``routing.<n>.bin`` file name + offset (``routing_dir`` is the
instance directory): position p of turn k uses "the last record at or before k that covers p"; sampling as JSON per
turn; group (GRPO grouping key = task) / episode; reward. ``logprob_is_sampling_dist`` checks the precondition the spec
states (the logprob is the distribution after temperature and before top-p/top-k/min-p filtering, so it equals the
sampling distribution only when all those filters are off; penalties must be at their defaults too).
**No weight_version**: the ledger does not record it (see the report).

Media: by default only sha256 + ``media_dir``; with ``inline_media=True`` read the blob, verify sha256 and store base64
(what the user asked for: "one token sequence + that segment's image/video base64"). Media without bytes (unmapped /
received by reference) skip the whole segment in inline mode, and are counted.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from . import roles
from .join import Joiner
from .ledger import Instance, MediaRef, Segment, Vocab, sha256_file

TOP_K_ALL = 1 << 30   # sglang normalises top_k=-1 to this (the whole vocabulary)

MEDIA_TYPE = pa.struct([
    ("modality", pa.string()), ("sha256", pa.string()), ("mime", pa.string()),
    ("ranges", pa.list_(pa.list_(pa.int32()))), ("grid_thw", pa.list_(pa.int32())),
    ("unmapped", pa.bool_()), ("src", pa.string()), ("k", pa.int16()), ("b64", pa.string()),
])
ROUTING_TYPE = pa.struct([
    ("k", pa.int16()), ("file", pa.string()), ("offset", pa.int64()), ("nbytes", pa.int64()),
    ("start", pa.int32()), ("n_tokens", pa.int32()), ("shape", pa.list_(pa.int32())), ("dtype", pa.string()),
])

_SEG_FIELDS = [
    ("instance", pa.string()), ("seg", pa.int32()), ("sid", pa.string()), ("thread", pa.string()),
    ("window", pa.int32()), ("seg_cause", pa.string()), ("split_reason", pa.string()),
    ("close_reason", pa.string()), ("seg_closed", pa.bool_()), ("seg_sha256", pa.string()),
    ("task", pa.string()), ("episode", pa.string()), ("group", pa.string()),
    ("reward", pa.float32()), ("outcome_status", pa.string()), ("outcome", pa.string()),
]

SFT_SCHEMA = pa.schema([
    # ---- same names and types as rollout_runner.py OUT_SCHEMA (convert_distill_shards takes them directly)
    ("uid", pa.string()), ("attempt", pa.int16()),
    ("mode", pa.string()), ("instr", pa.string()), ("grid_fmt", pa.string()),
    ("label_shape", pa.bool_()), ("effort", pa.string()), ("pack_sha", pa.string()),
    ("train_idx", pa.list_(pa.int16())), ("test_idx", pa.int16()),
    ("ids", pa.list_(pa.int32())),
    ("span_start", pa.list_(pa.int32())), ("span_end", pa.list_(pa.int32())), ("span_role", pa.list_(pa.int8())),
    ("n_turns", pa.int16()), ("n_tool_calls", pa.int16()), ("n_tool_errors", pa.int16()),
    ("n_prompt", pa.int32()), ("n_completion", pa.int32()),
    ("stop_reason", pa.string()), ("parsed", pa.bool_()), ("correct", pa.bool_()),
    ("correct_strict", pa.bool_()), ("answer_src", pa.string()),
    ("confidence", pa.int16()), ("code_train_ok", pa.int16()), ("code_train_n", pa.int16()),
    ("code_test_match", pa.int8()), ("retok_ok", pa.int8()), ("latency_s", pa.float32()),
    # ---- TITO
    *_SEG_FIELDS,
    ("n_appends", pa.int16()), ("n_stop_substituted", pa.int16()),
    ("kind_start", pa.list_(pa.int32())), ("kind_end", pa.list_(pa.int32())), ("kind", pa.list_(pa.int8())),
    ("n_media", pa.int16()), ("media_dir", pa.string()), ("media", pa.list_(MEDIA_TYPE)),
    ("n_rewound", pa.int32()), ("convert_issue", pa.string()),
], metadata={"tito.format": "sft-v1", "tito.kinds": json.dumps(roles.KINDS)})

DFLASH_SCHEMA = pa.schema([
    ("uid", pa.string()), ("effort", pa.string()), ("n_prompt", pa.int32()), ("n_completion", pa.int32()),
    ("input_ids", pa.list_(pa.int32())), ("loss_mask", pa.list_(pa.int8())),
    ("instance", pa.string()), ("seg", pa.int32()), ("sid", pa.string()), ("task", pa.string()), ("group", pa.string()),
], metadata={"tito.format": "dflash-v1"})

RL_SCHEMA = pa.schema([
    *_SEG_FIELDS,
    ("ids", pa.list_(pa.int32())),
    ("gen_start", pa.list_(pa.int32())), ("gen_end", pa.list_(pa.int32())), ("gen_k", pa.list_(pa.int16())),
    ("gen_logprobs", pa.list_(pa.float32())), ("logprobs_complete", pa.bool_()),
    ("turn_k", pa.list_(pa.int16())), ("turn_gen_start", pa.list_(pa.int32())),
    ("turn_gen", pa.list_(pa.list_(pa.int32()))), ("turn_logprobs", pa.list_(pa.list_(pa.float32()))),
    ("turn_rewound", pa.list_(pa.int32())), ("turn_finish", pa.list_(pa.string())), ("turn_sampling", pa.list_(pa.string())),
    ("logprob_is_sampling_dist", pa.bool_()), ("sampling_issue", pa.string()),
    ("routing_dir", pa.string()), ("routing", pa.list_(ROUTING_TYPE)), ("routing_complete", pa.bool_()),
    ("n_media", pa.int16()), ("media_dir", pa.string()), ("media", pa.list_(MEDIA_TYPE)),
], metadata={"tito.format": "rl-v1",
             # Written into the parquet schema metadata, kept verbatim as on-disk format: "position p of turn k: the last
             # routing record at or before k that covers p; on a rewind the two records for row prefix_len-1 are equal"
             "tito.routing": "回合 k 的位置 p：k 以前（含）涵蓋 p 的最後一筆 routing；回捲時 prefix_len-1 那一列兩筆相同"})


# ----------------------------------------------------------------------------- common


class _Shards:
    def __init__(self, out: Path, schema: pa.Schema, rows_per_shard: int):
        if rows_per_shard < 1:
            raise ValueError(f"rows_per_shard={rows_per_shard}")
        self.out, self.schema, self.rows_per_shard = out, schema, rows_per_shard
        self.buf: dict[str, list[dict]] = {}
        self.n: dict[str, int] = {}
        self.files: list[str] = []

    def add(self, prefix: str, row: dict) -> None:
        self.buf.setdefault(prefix, []).append(row)
        if len(self.buf[prefix]) >= self.rows_per_shard:
            self.flush(prefix)

    def flush(self, prefix: str) -> None:
        rows = self.buf.get(prefix)
        if not rows:
            return
        i = self.n.get(prefix, 0)
        table = pa.table({f.name: [r[f.name] for r in rows] for f in self.schema}, schema=self.schema)
        name = f"{prefix}part-{i:05d}.parquet"
        tmp = self.out / (name + ".tmp")
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, self.out / name)
        self.files.append(name)
        self.n[prefix] = i + 1
        self.buf[prefix] = []

    def close(self) -> None:
        for prefix in list(self.buf):
            self.flush(prefix)


def _prepare_out(out: str | Path) -> Path:
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"{out} already exists: exports do not mix batches, use another directory")
    out.mkdir(parents=True)
    return out


def _seg_info(seg: Segment, joiner: Joiner | None) -> dict | None:
    """Identity and join result of a segment; None when a joiner is given but the segment does not match the manifest."""
    thread, window = (None, -1) if seg.sid is None else (seg.thread, seg.window)
    hit = joiner.lookup(seg) if joiner is not None else None
    if joiner is not None and hit is None:
        return None
    task = hit.task if hit else None
    out = None if hit is None else joiner.outcome(hit)
    group = task if task is not None else (thread if thread is not None else f"single:{seg.key}")
    return {
        "instance": seg.instance, "seg": seg.seg, "sid": seg.sid, "thread": thread, "window": window,
        "seg_cause": seg.cause, "split_reason": seg.split_reason,
        "close_reason": None if seg.close is None else seg.close["reason"],
        "seg_closed": seg.status == "closed",
        "seg_sha256": None if seg.close is None else seg.close["sha256"],
        "task": task, "episode": hit.episode if hit else (thread or f"single:{seg.key}"), "group": group,
        "reward": None if out is None else out.reward,
        "outcome_status": None if out is None else out.status,
        "outcome": None if out is None else json.dumps(out.detail, sort_keys=True),
    }


def _media_rows(inst: Instance, media: list[MediaRef], inline: bool) -> list[dict] | None:
    """None = inline mode and some media has no bytes (the caller skips the whole segment)."""
    out = []
    for m in media:
        b64 = None
        if inline:
            if m.sha256 is None:
                return None
            raw = inst.media_path(m.sha256).read_bytes()
            got = hashlib.sha256(raw).hexdigest()
            if got != m.sha256:
                raise ValueError(f"{inst.name}: media/{m.sha256} content has sha256 {got}")
            b64 = base64.b64encode(raw).decode("ascii")
        out.append({"modality": m.modality, "sha256": m.sha256, "mime": m.mime,
                    "ranges": [list(r) for r in m.ranges],
                    "grid_thw": None if m.grid_thw is None else list(m.grid_thw),
                    "unmapped": m.unmapped, "src": m.src, "k": m.k, "b64": b64})
    return out


def _gen_tuples(seg: Segment) -> list[tuple[int, int]]:
    return [(s.start, s.end) for s in seg.gen_spans]


def _iter(instances: Iterable[Instance], joiner: Joiner | None, stats: dict, seen: list[str]):
    for inst in instances:
        seen.append(str(inst.path))
        for seg in inst.segments.values():
            if seg.status == "broken":
                raise ValueError(f"{seg.key} is a broken segment (exports must read with strict=True): {seg.error}")
            stats["segments"] += 1
            if not seg.appends:
                stats["empty"] += 1
                continue
            info = _seg_info(seg, joiner)
            if info is None:
                stats["unjoined"] += 1
                continue
            if not seg.gen_spans:
                stats["no_gen"] += 1
                continue
            yield inst, seg, info


def _finish(out: Path, kind: str, shards: _Shards, stats: dict, instances: list[str], vocab: Vocab,
            joiner: Joiner | None) -> dict:
    shards.close()
    doc = {"format": kind, "t": time.time(), "files": shards.files, "stats": stats,
           "instances": instances, "tokenizer": str(vocab.path),
           "tokenizer_sha256": vocab.sha256,
           "manifests": [] if joiner is None else [str(m.path) for m in joiner.manifests],
           "kinds": list(roles.KINDS)}
    (out / "tito_export.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
    return doc


def _new_stats() -> dict:
    return {"segments": 0, "empty": 0, "unjoined": 0, "no_gen": 0, "rows": 0, "skipped_media": 0,
            "media_without_bytes": 0, "open_segments": 0, "tokens": 0, "gen_tokens": 0}


# ----------------------------------------------------------------------------- SFT


def sft_row(inst: Instance, seg: Segment, info: dict, vocab: Vocab, *, inline_media: bool = False,
            harness_prefixes=roles.HARNESS_USER_PREFIXES) -> dict | None:
    ids = seg.ids
    gen = _gen_tuples(seg)
    spans = roles.role_spans(len(ids), gen)
    kinds = roles.token_kinds(ids, gen, vocab, harness_prefixes)
    kspans = roles.kind_spans(kinds)
    media = _media_rows(inst, seg.media, inline_media)
    if media is None:
        return None
    n_completion = sum(e - s for s, e, r in spans if r == roles.ROLE_ASSISTANT)
    effort = roles.reasoning_effort(ids, gen, vocab)
    blocks = roles.chat_blocks(ids, roles.gen_mask(len(ids), gen), vocab)
    system = next((b for b in blocks if b.role == "system"), None)
    pack_sha = "" if system is None else hashlib.sha256(
        np.ascontiguousarray(ids[system.start: system.end + 1], dtype="<i4").tobytes()).hexdigest()[:16]
    # arc2: score is partial credit over test pairs; only all tests correct counts as correct
    correct = None if info["reward"] is None else info["reward"] >= 1.0
    uid = info["task"] or info["thread"] or f"single:{seg.key}"
    return {
        "uid": uid, "attempt": max(info["window"], 0), "mode": "tito", "instr": "", "grid_fmt": "",
        "label_shape": False, "effort": effort or "", "pack_sha": pack_sha, "train_idx": [], "test_idx": -1,
        "ids": ids.tolist(), "span_start": [s for s, _, _ in spans], "span_end": [e for _, e, _ in spans],
        "span_role": [r for _, _, r in spans],
        "n_turns": sum(1 for a in seg.appends if a.gen_len), "n_tool_calls": roles.n_tool_calls(ids, kinds, vocab),
        "n_tool_errors": -1, "n_prompt": len(ids) - n_completion, "n_completion": n_completion,
        "stop_reason": seg.appends[-1].finish["type"], "parsed": False, "correct": correct,
        "correct_strict": correct, "answer_src": "", "confidence": -1, "code_train_ok": -1, "code_train_n": -1,
        "code_test_match": -1, "retok_ok": -1, "latency_s": float(sum(a.t1 - a.t0 for a in seg.appends)),
        **info,
        "n_appends": len(seg.appends), "n_stop_substituted": sum(a.stop_substituted for a in seg.appends),
        "kind_start": [s for s, _, _ in kspans], "kind_end": [e for _, e, _ in kspans],
        "kind": [k for _, _, k in kspans],
        "n_media": len(seg.media), "media_dir": str(inst.path / "media"), "media": media,
        "n_rewound": sum(a.rewind for a in seg.appends), "convert_issue": convert_issue(ids, spans, vocab),
    }


def convert_issue(ids: np.ndarray, spans: list[tuple[int, int, int]], vocab: Vocab) -> str | None:
    """Token-level rules of ``convert_distill_shards._check_row`` (:124-136; roles.check_role_spans already guarantees
    the structural rules): a role-1 span does not start with ``<think>``; a role-1 span other than the last ends with
    ``<|im_end|>``, or the next token is one."""
    asst = [(s, e) for s, e, r in spans if r == roles.ROLE_ASSISTANT]
    for i, (s, e) in enumerate(asst):
        if ids[s] == vocab.think:
            return f"assistant span [{s}, {e}) starts with <think>"
        if i < len(asst) - 1 and ids[e - 1] != vocab.im_end and not (e < len(ids) and ids[e] == vocab.im_end):
            return f"non-final assistant span [{s}, {e}) does not end with <|im_end|>, nor is the next token one"
    return None


def export_sft(instances: Iterable[Instance], vocab: Vocab, out: str | Path, *, joiner: Joiner | None = None,
               inline_media: bool = False, rows_per_shard: int = 256,
               harness_prefixes=roles.HARNESS_USER_PREFIXES) -> dict:
    out = _prepare_out(out)
    shards = _Shards(out, SFT_SCHEMA, rows_per_shard)
    stats = _new_stats()
    stats.update(text_rows=0, media_rows=0, other_rows=0)
    seen: list[str] = []
    for inst, seg, info in _iter(instances, joiner, stats, seen):
        row = sft_row(inst, seg, info, vocab, inline_media=inline_media, harness_prefixes=harness_prefixes)
        if row is None:
            stats["media_without_bytes"] += 1
            continue
        prefix, key = (("media-", "media_rows") if row["n_media"] else
                       ("other-", "other_rows") if row["convert_issue"] else ("", "text_rows"))
        stats[key] += 1
        stats["rows"] += 1
        stats["open_segments"] += int(not row["seg_closed"])
        stats["tokens"] += len(row["ids"])
        stats["gen_tokens"] += row["n_completion"]
        shards.add(prefix, row)
    return _finish(out, "sft", shards, stats, seen, vocab, joiner)


# ----------------------------------------------------------------------------- DFlash


def dflash_uid(group: str) -> str:
    return hashlib.sha256(("tito-group-v1:" + group).encode()).hexdigest()


def export_dflash(instances: Iterable[Instance], vocab: Vocab, out: str | Path, *, joiner: Joiner | None = None,
                  rows_per_shard: int = 256) -> dict:
    out = _prepare_out(out)
    shards = _Shards(out, DFLASH_SCHEMA, rows_per_shard)
    stats = _new_stats()
    seen: list[str] = []
    for inst, seg, info in _iter(instances, joiner, stats, seen):
        if seg.media:
            stats["skipped_media"] += 1
            continue
        ids = seg.ids
        mask = np.zeros(len(ids), np.int8)
        for s in seg.gen_spans:
            mask[s.start: s.end] = 1
        n_prompt = seg.gen_spans[0].start
        if not 0 < n_prompt <= len(ids) or mask[:n_prompt].any():
            raise RuntimeError(f"{seg.key}: n_prompt={n_prompt} inconsistent with loss_mask")
        stats["rows"] += 1
        stats["open_segments"] += int(not info["seg_closed"])
        stats["tokens"] += len(ids)
        stats["gen_tokens"] += int(mask.sum())
        shards.add("", {"uid": dflash_uid(info["group"]), "effort": f"tito:{seg.key}", "n_prompt": n_prompt,
                        "n_completion": int(mask.sum()), "input_ids": ids.tolist(), "loss_mask": mask.tolist(),
                        "instance": seg.instance, "seg": seg.seg, "sid": seg.sid, "task": info["task"],
                        "group": info["group"]})
    return _finish(out, "dflash", shards, stats, seen, vocab, joiner)


# ----------------------------------------------------------------------------- RL


def sampling_issue(sampling: dict) -> str | None:
    """Preconditions for the logprob (after temperature, before filtering) to equal the sampling distribution; returns
    the reason when they do not hold."""
    problems = []
    for key in ("temperature", "top_p", "top_k", "min_p"):
        if key not in sampling:
            problems.append(f"sampling does not record {key}")
    if "top_p" in sampling and sampling["top_p"] != 1.0:
        problems.append(f"top_p={sampling['top_p']}")
    if "top_k" in sampling and not (sampling["top_k"] == -1 or sampling["top_k"] >= TOP_K_ALL):
        problems.append(f"top_k={sampling['top_k']}")
    if "min_p" in sampling and sampling["min_p"] != 0.0:
        problems.append(f"min_p={sampling['min_p']}")
    for key, neutral in (("frequency_penalty", 0.0), ("presence_penalty", 0.0), ("repetition_penalty", 1.0)):
        if key in sampling and sampling[key] != neutral:
            problems.append(f"{key}={sampling[key]}")
    return "; ".join(problems) or None


#: sampling fields rl_sampling_issue requires (a subset of the fork's session_ledger.SAMPLING_FIELDS)
RL_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p", "frequency_penalty", "presence_penalty",
                    "repetition_penalty")


def rl_sampling_issue(sampling: dict, temperature: float, top_p: float = 1.0, top_k: int | None = None) -> str | None:
    """Sampling preconditions of RL spec §2 (strict ``sampling_issue``): temperature = expected, top_p = expected
    (default 1), top_k = expected (None = off), min_p = 0, the three penalties neutral, and all seven fields must be
    recorded (unrecorded means unprovable). Returns the reason when they do not hold. With top_p < 1 or top_k on
    (truncated sampling) the behaviour logprob must come from the ledger's ``sampling_mask`` (SPEC §1
    ``replay: mask``), not ``logprobs``."""
    problems = [f"sampling does not record {key}" for key in RL_SAMPLING_KEYS if key not in sampling]
    if "temperature" in sampling:
        t = sampling["temperature"]
        if isinstance(t, bool) or not isinstance(t, (int, float)) or abs(t - temperature) > 1e-6:
            problems.append(f"temperature={t!r}, expected {temperature}")
    if "top_p" in sampling:
        p = sampling["top_p"]
        if isinstance(p, bool) or not isinstance(p, (int, float)) or abs(p - top_p) > 1e-6:
            problems.append(f"top_p={p!r}, expected {top_p}")
    if "top_k" in sampling:
        k = sampling["top_k"]
        off = k == -1 or k >= TOP_K_ALL
        if (top_k is None and not off) or (top_k is not None and k != top_k):
            problems.append(f"top_k={k!r}, expected {'off' if top_k is None else top_k}")
    if "min_p" in sampling and sampling["min_p"] != 0.0:
        problems.append(f"min_p={sampling['min_p']}")
    for key, neutral in (("frequency_penalty", 0.0), ("presence_penalty", 0.0), ("repetition_penalty", 1.0)):
        if key in sampling and sampling[key] != neutral:
            problems.append(f"{key}={sampling[key]}")
    return "; ".join(problems) or None


def rl_row(inst: Instance, seg: Segment, info: dict, *, inline_media: bool = False) -> dict | None:
    media = _media_rows(inst, seg.media, inline_media)
    if media is None:
        return None
    lp = seg.gen_logprobs()
    turns = [a for a in seg.appends if a.gen_len]
    rewound = {a.k: (seg.appends[a.k + 1].rewind if a.k + 1 < len(seg.appends) else 0) for a in turns}
    issues = sorted({i for a in turns if (i := sampling_issue(a.sampling)) is not None})
    routing = [{"k": a.k, "file": a.routing.file, "offset": a.routing.offset, "nbytes": a.routing.nbytes,
                "start": a.routing.start, "n_tokens": a.routing.n_tokens, "shape": list(a.routing.shape),
                "dtype": a.routing.dtype} for a in seg.appends if a.routing is not None]
    return {
        **info,
        "ids": seg.ids.tolist(),
        "gen_start": [s.start for s in seg.gen_spans], "gen_end": [s.end for s in seg.gen_spans],
        "gen_k": [s.k for s in seg.gen_spans],
        "gen_logprobs": None if lp is None else lp.tolist(),
        "logprobs_complete": all(a.logprobs is not None for a in turns),
        "turn_k": [a.k for a in turns], "turn_gen_start": [a.gen_start for a in turns],
        "turn_gen": [a.gen.tolist() for a in turns],
        "turn_logprobs": [None if a.logprobs is None else a.logprobs.tolist() for a in turns],
        "turn_rewound": [rewound[a.k] for a in turns], "turn_finish": [a.finish["type"] for a in turns],
        "turn_sampling": [json.dumps(a.sampling, sort_keys=True) for a in turns],
        "logprob_is_sampling_dist": not issues, "sampling_issue": "; ".join(issues) or None,
        "routing_dir": str(inst.path), "routing": routing,
        "routing_complete": len(routing) == len(seg.appends),
        "n_media": len(seg.media), "media_dir": str(inst.path / "media"), "media": media,
    }


def export_rl(instances: Iterable[Instance], vocab: Vocab, out: str | Path, *, joiner: Joiner | None = None,
              inline_media: bool = False, rows_per_shard: int = 256) -> dict:
    out = _prepare_out(out)
    shards = _Shards(out, RL_SCHEMA, rows_per_shard)
    stats = _new_stats()
    stats.update(logprobs_complete=0, routing_complete=0, sampling_mismatch=0)
    seen: list[str] = []
    for inst, seg, info in _iter(instances, joiner, stats, seen):
        row = rl_row(inst, seg, info, inline_media=inline_media)
        if row is None:
            stats["media_without_bytes"] += 1
            continue
        stats["rows"] += 1
        stats["open_segments"] += int(not row["seg_closed"])
        stats["tokens"] += len(row["ids"])
        stats["gen_tokens"] += sum(e - s for s, e in zip(row["gen_start"], row["gen_end"]))
        stats["logprobs_complete"] += int(row["logprobs_complete"])
        stats["routing_complete"] += int(row["routing_complete"])
        stats["sampling_mismatch"] += int(not row["logprob_is_sampling_dist"])
        shards.add("", row)
    return _finish(out, "rl", shards, stats, seen, vocab, joiner)


def verify_media_blobs(inst: Instance) -> list[tuple[str, str]]:
    """List of (sha256, problem): files whose content sha256 does not match the file name."""
    bad = []
    for sha in sorted({m.sha256 for seg in inst.segments.values() for m in seg.media if m.sha256 is not None}):
        got = sha256_file(inst.media_path(sha))
        if got != sha:
            bad.append((sha, f"content sha256 is {got}"))
    return bad
