"""The DFlash data plan's contract: corpora, what one job's token ids and loss mask are, how items are grouped for the
held-out split, and the plan files (make_data_plan.py writes them on a CPU, capture_plan.py replays them).

loss_mask is 1 exactly on the tokens the generating model sampled at that position from that very prefix, and 0 on
prompts, tool outputs, tokens a template or harness inserted, and assistant turns copied into a context they were not
sampled in. The drafter's anchors are data.eligible_anchors(loss_mask). Per corpus (each builder checks its contract on
every row and raises on the first violation):

- ``node1``: TITO ledger segments of the 10-06 collection on node1 (codex on ARC-AGI-2 pools and ARC-AGI-3, T=0.7).
  The mask is the segment's gen spans (capture_tito.loss_mask): the ledger rebuilds S from each request's tail and
  sampled gen, and a token the next request rewound (a stop token the template replaced with <|im_end|>) is not in a
  gen span (arcforge/tito/ledger.py).
- ``agentic``: agentic_distill's rendered segments (render_segments.py, token backend, T=1.0). Role 1 is each
  /generate's output_ids verbatim (token_backend.py: an assistant message's ids are the cue, output_ids, and an
  <|im_end|> appended only when the output was truncated -- that one is role 2). A segment after a compaction starts
  with the last ``kept_rounds`` assistant rounds of the episode's previous segment, copied after the harness'
  announcement: those were sampled in the previous segment's context, not this one, and are masked here (they stay
  supervised in the previous segment, where they were sampled). make_data_plan finds them by comparing tokens.
- ``naive``: qwen38fn_distill (rollout_runner.py, T=1.0): role 1 is each /generate's output_ids appended verbatim; a
  turn cut by the budget ends the rollout, nothing is appended after it.
- ``nemotron``: qwen38fn_specdata (generate.py, T=0.8): the loss_mask column, 0 on the rendered prompt and 1 on the
  output_ids (EOS kept on a normal stop, nothing appended on a length stop).
- ``arc2`` / ``arc3``: TITO ledger segments of the 2026-10-14..16 codex collection (docs/research/
  tito_fp8_dflash_collect_20261015.md: arc2 codex2wr on independent1000, arc3 the v17 / c0L arms on community, official
  25 and synth games, T=0.7), masked like node1. One ``--codex1015`` instance list feeds both: a segment's corpus is the
  lane its workspace names (classify_codex1015); segments outside any workspace and the three arc3_test harness-test
  sessions are dropped before planning and counted in summary.json. The think budget's forced tokens (436 of the second half's 185M arc3 tokens; the first half kept no
  flags) stay in the mask: a known approximation, recorded in plan_meta.json.

Splits: the held-out set is drawn by group, so related items never straddle. ARC items group by their parent ARC task
(``arc:<task>``, one group in every corpus): a node1 training1000 thread's task, and the parent of an NVARC task --
its uid is ``<parent>_<partner>``, two ARC task ids, NVARC's mix of the parent's summary with the partner's
(nvarc_parent) -- in node1's nvarc_training pool, agentic and naive, so a task and every NVARC variant of it fall on
one side. Only the parent binds: every parent is also some uids' partner, and the pairs connect all parents into one
component (binding both would leave nothing to hold out), so a training item can have a held-out parent as its
partner. ARC-AGI-3 groups by game (both runs); nemotron by prompt uid through data.split_for_key, the old split
function, so its held-out is a subset of the old 1% held-out. A group held out in one corpus is excluded from training
in every corpus. Never held out: independent1000 and synth108 (built from ARC-AGI-2 public-eval parents) and agentic's
nvarc_full (public-eval parents, which must not be a held-out pool's parent). The codex1015 corpora are their own
plans' data (no NVARC corpus beside them), so they hold out by task / game: arc2 an independent1000 task with every
episode of it (``independent1000:<ind_NNNN>``; tasks share public-eval parents, so this is task- not parent-disjoint),
arc3 a game (``arc3:<game>``, every run and arm), drawn per subset (make_data_plan.choose_codex_heldout).

Which parents (make_data_plan.choose_heldout): node1's ARC-AGI-2 held-out takes only training1000 tasks without an
NVARC family in any corpus, so it excludes nothing elsewhere. agentic and naive hold out the same few parents (one
NVARC family is 0.05-12M tokens, the target a few M) and keep a fixed sample of each family's items as the held-out
jobs; the rest of those families is neither held out nor trained (``heldout_unsampled``).
"""

import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

CORPORA = ("node1", "arc2", "arc3", "agentic", "naive", "nemotron")
LEDGER_CORPORA = ("node1", "arc2", "arc3")  # jobs are TITO ledger segments: key <corpus>:<instance>:<seg>
BLOCK_SIZE = 8
MIN_SUPERVISED = 7
CUE = "<|im_start|>assistant\n<think>\n"  # every generation prompt of the four corpora ends with it
HELDOUT_SALT = "dflash-data-plan-v1"  # the held-out choice does not depend on --seed, so plans stay comparable
SKIP_REASONS = ("too_long", "media_token", "media_unrecorded", "few_supervised", "no_anchors", "heldout_elsewhere",
                "heldout_unsampled")

# node1: the first user message of a codex session names its workspace.
AGENTS_RE = re.compile(r"# AGENTS\.md instructions for (\S+)")
ARC2_RE = re.compile(r"/arc2/(?P<batch>\d{4}-(?P<pool>[a-z0-9_]+)-b\d{3})/tasks/(?P<task>[^/]+)$")
ARC3_RE = re.compile(r"/arc3/(?P<run>run-\d+)/games/(?P<game>[^/]+)$")
SMOKE_RE = re.compile(r"/smoke/(?P<batch>[^/]+)/(?:tasks|games)/(?P<task>[^/]+)$")  # pre-collection checks: train only
CLEAN_ARC2 = ("training1000", "nvarc_training")  # ARC-AGI-2 training and NVARC training: the held-out pools
TAINTED_ARC2 = ("independent1000", "synth108")  # derived from ARC-AGI-2 public-eval parents: train only
# codex1015 (20261015-fp8-dflash-collect): arc2/<run>/tasks/ind_NNNN__s<episode>, arc3/<run>/run/games/<game>, and
# the pre-collection gate/smoke-arc2 (train only, grouped with its task)
CODEX_ARC2_RE = re.compile(r"/arc2/(?P<run>ind-b\d{3}(?:-ref)?|inde-q\d{5})/tasks/(?P<task>ind_\d{4})__s(?P<ep>\d+)$")
CODEX_ARC3_RE = re.compile(
    r"/arc3/(?P<run>(?P<subset>comm-[AB](?=\d)|off25(?=-)|synth(?=-))[\w-]+)/run/games/(?P<game>[a-z0-9]+)$")
CODEX_SMOKE_RE = re.compile(r"/gate/(?P<run>smoke-arc2)/tasks/(?P<task>ind_\d{4})__s(?P<ep>\d+)$")
CODEX_TEST_RE = re.compile(r"/arc3_test/(?P<run>[\w-]+)/run/games/(?P<game>[a-z0-9]+)$")  # harness tests: dropped
CODEX_ARC3_SUBSETS = ("comm-A", "comm-B", "off25", "synth")
# int3c1019 (tito/20261019-int3-collect, INT3 W3A8): arc2/{qual,nvar}-q<5 digits>/tasks/<task>__s<episode>,
# arc3/runs/comm-{B,C}<3 digits>-<arm>/run/games/<game>. Train only (held out only through --heldout-from).
INT3C_ARC2_RE = re.compile(r"/arc2/(?P<run>(?P<src>qual|nvar)-q\d{5})/tasks/(?P<task>[^/]+?)__s(?P<ep>\d+)$")
INT3C_ARC3_RE = re.compile(r"/arc3/runs/(?P<run>comm-(?P<tier>[BC])\d{3}-[a-z0-9]+)/run/games/(?P<game>[a-z0-9]+)$")
INT3C_ARC2_POOLS = {"qual": "quality300", "nvar": "nvarc_training"}
# bench (arc_code/20261006-bench-fp8-base, 20261008-bench-fp8-ple, official FP8): arc2/{A,B}-r<k>/tasks/<task> (A the
# ARC-AGI-2 eval 120, B 360 of independent1000), arc3/C-r<k>/games/<game> (public-25). Train only, like int3c1019.
BENCH_ARC2_RE = re.compile(r"/\d{8}-bench-fp8-(?P<arm>base|ple)/arc2/(?P<run>(?P<lane>[AB])-r\d+)/tasks/(?P<task>[^/]+)$")
BENCH_ARC3_RE = re.compile(r"/\d{8}-bench-fp8-(?P<arm>base|ple)/arc3/(?P<run>C-r\d+)/(?:run/)?games/(?P<game>[a-z0-9]+)$")
BENCH_ARC2_POOLS = {"A": "eval120", "B": "independent1000"}
ARC_TASK_RE = re.compile(r"[0-9a-f]{8}")
NVARC_UID_RE = re.compile(r"(?P<parent>[0-9a-f]{8})_(?P<partner>[0-9a-f]{8})")
ANCHOR_RULE = "a >= 1 and mask[a] and mask[a + 1]"  # data.eligible_anchors; plan_meta.json records it


# ----------------------------------------------------------------------------- masks and anchors


def runs(mask):
    """Maximal runs [start, end) of True."""
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def anchor_count(mask):
    """len(data.eligible_anchors(mask)) without torch: anchor a >= 1 needs mask[a] and mask[a + 1], so a run [s, e)
    has the anchors max(s, 1) .. e - 2."""
    s, e = runs(mask)
    return int(np.maximum(e - 1 - np.maximum(s, 1), 0).sum())


def span_mask(n, spans):
    mask = np.zeros(n, bool)
    for s, e in spans:
        mask[s:e] = True
    return mask


def ids_sha256(ids):
    """sha256 of the ids as little-endian int32 (arcforge.tito.ledger.ids_sha256, the ledger's seg_close digest)."""
    return hashlib.sha256(np.ascontiguousarray(ids, dtype="<i4").tobytes()).hexdigest()


def unit_hash(*parts):
    """A deterministic number in [0, 1) from the parts."""
    return int(hashlib.sha256(":".join(map(str, parts)).encode()).hexdigest()[:16], 16) / 2**64


class Markers:
    """The token ids the mask contracts check, from the target's tokenizer (arcforge.tito.ledger.Vocab)."""

    def __init__(self, vocab):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(vocab.path))
        self.cue = np.array(tok.encode(CUE, add_special_tokens=False).ids, np.int64)
        self.newline = tok.encode("\n", add_special_tokens=False).ids
        if len(self.newline) != 1:
            raise ValueError(f"'\\n' is {self.newline}, not one token")
        self.newline = self.newline[0]
        self.im_start, self.im_end = vocab.im_start, vocab.im_end
        self.stops = (vocab.im_end, vocab.endoftext)
        self.media = (vocab.image_pad, vocab.video_pad)
        self.id_bound = vocab.id_bound


# ----------------------------------------------------------------------------- row contracts


class ContractError(ValueError):
    """A row breaks the mask contract its corpus is documented to keep: the semantics above would be wrong for it."""


def tito_mask(seg, where):
    """A ledger segment's loss mask (capture_tito.loss_mask, its gen spans), checked against the ledger's own record:
    each span is the sampled gen of the append it came from, and a stop token the template replaced (the next append's
    <|im_end|>, stop_substituted) is outside the mask. Returns (mask, appends with a substituted stop)."""
    mask = np.zeros(len(seg), bool)
    by_k = {a.k: a for a in seg.appends}
    for span in seg.gen_spans:
        if not np.array_equal(seg.ids[span.start:span.end], by_k[span.k].gen[:span.end - span.start]):
            raise ContractError(f"{where}: gen span [{span.start}, {span.end}) is not append {span.k}'s sampled gen")
        mask[span.start:span.end] = True
    substituted = [a for a in seg.appends if a.stop_substituted]
    if any(mask[a.prefix_len] for a in substituted):
        raise ContractError(f"{where}: a template-substituted <|im_end|> is inside the mask")
    return mask, len(substituted)


def _tiles(starts, ends, roles, n, where):
    if not (len(starts) == len(ends) == len(roles)) or not len(starts):
        raise ContractError(f"{where}: span columns disagree")
    if starts[0] != 0 or ends[-1] != n or any(e != s for e, s in zip(ends[:-1], starts[1:])) or any(
            e <= s for s, e in zip(starts, ends)):
        raise ContractError(f"{where}: spans do not tile [0, {n})")
    # rollout_runner.py and render_segments.normalize_roles: 0 (1 2)* 1?
    if roles[0] != 0 or any(r != (1 if i % 2 else 2) for i, r in enumerate(roles[1:], 1)):
        raise ContractError(f"{where}: role sequence {list(roles)[:8]}... is not 0 (1 2)* 1?")


def assistant_spans(ids, starts, ends, roles, markers, *, truncation, where):
    """The role-1 spans [(s, e)] of a token-backend row, checked: each follows the generation cue and either ends with a
    stop token or was cut by the budget. ``truncation`` is what the producer did after a cut turn: "appended_im_end"
    (agentic: an <|im_end|> that is not sampled, role 2) or "rollout_ends" (naive: the cut turn is the row's last)."""
    n = len(ids)
    _tiles(starts, ends, roles, n, where)
    out, cut = [], 0
    k = len(markers.cue)
    for i, (s, e, r) in enumerate(zip(starts, ends, roles)):
        if r != 1:
            continue
        if s < k or not np.array_equal(ids[s - k:s], markers.cue):
            raise ContractError(f"{where}: assistant span [{s}, {e}) does not follow the generation cue")
        if ids[e - 1] in markers.stops:
            if e < n and ids[e] != markers.newline:
                raise ContractError(f"{where}: {ids[e]} after the stop token at {e - 1}, not the template's newline")
        else:
            cut += 1
            if truncation == "appended_im_end":
                if e == n or ids[e] != markers.im_end:
                    raise ContractError(f"{where}: cut assistant span [{s}, {e}) without the appended <|im_end|>")
            elif truncation == "rollout_ends":
                if e != n:
                    raise ContractError(f"{where}: cut assistant span [{s}, {e}) is not the rollout's last")
            else:
                raise ValueError(truncation)
        out.append((int(s), int(e)))
    return out, cut


def nemotron_mask(ids, loss_mask, n_prompt, n_completion, finish, markers, where):
    """generate.py's row: prompt (the rendered conversation, ending with the generation cue), then output_ids."""
    n = len(ids)
    loss_mask = np.asarray(loss_mask)
    if len(loss_mask) != n or n_prompt + n_completion != n or n_completion < 1:
        raise ContractError(f"{where}: {n} ids, {len(loss_mask)} mask, {n_prompt} + {n_completion}")
    if loss_mask[:n_prompt].any() or not (loss_mask[n_prompt:] == 1).all():
        raise ContractError(f"{where}: loss_mask is not 0 on the {n_prompt} prompt tokens and 1 after")
    k = len(markers.cue)
    if n_prompt < k or not np.array_equal(ids[n_prompt - k:n_prompt], markers.cue):
        raise ContractError(f"{where}: the prompt does not end with the generation cue")
    if finish == "stop":
        if ids[-1] not in markers.stops:
            raise ContractError(f"{where}: a stop without the EOS kept in output_ids")
    elif finish != "length":
        raise ContractError(f"{where}: finish_reason {finish!r}")
    return loss_mask.astype(bool)


def kept_rounds(previous, current):
    """How many of current's leading assistant rounds are the last rounds of the episode's previous segment, token for
    token (the compact harness keeps the most recent rounds when it folds). Arguments: role-1 span digests in order."""
    for m in range(min(len(previous), len(current)), 0, -1):
        if current[:m] == previous[-m:]:
            return m
    return 0


# ----------------------------------------------------------------------------- parquet corpora


PARQUET_COLUMNS = {
    "agentic": ["uid", "attempt", "episode", "lineage", "segment_idx", "n_segments", "run", "source", "ids",
                "span_start", "span_end", "span_role", "n_completion"],
    "naive": ["uid", "attempt", "mode", "ids", "span_start", "span_end", "span_role", "n_completion"],
    "nemotron": ["uid", "effort", "input_ids", "loss_mask", "n_prompt", "n_completion", "finish_reason"],
}


def parquet_key(corpus, row, batch):
    if corpus == "agentic":
        return f"agentic:{row['uid']}:{row['attempt']}:{row['episode']}:{row['lineage']}:{row['segment_idx']}"
    if corpus == "naive":
        return f"naive:{row['uid']}:{row['attempt']}"
    if corpus == "nemotron":
        return f"nemotron:{batch}:{row['uid']}:{row['effort']}"
    raise ValueError(corpus)


def arc_group(task):
    """The split group of an ARC task and of everything derived from it, in every corpus."""
    if ARC_TASK_RE.fullmatch(task) is None:
        raise ContractError(f"ARC task id {task!r} is not 8 hex digits")
    return f"arc:{task}"


def nvarc_parent(uid):
    """The ARC task an NVARC task was made from: its uid is <parent>_<partner>, two distinct ARC task ids."""
    m = NVARC_UID_RE.fullmatch(uid)
    if m is None or m["parent"] == m["partner"]:
        raise ContractError(f"NVARC uid {uid!r} is not <parent>_<partner>, two distinct 8-hex ARC task ids")
    return m["parent"]


def parquet_group(corpus, row):
    """(group, held-out bucket or None)."""
    if corpus == "agentic":
        if row["source"] == "nvarc_training":
            return arc_group(nvarc_parent(row["uid"])), "nvarc"
        if row["source"] == "nvarc_full":
            return arc_group(nvarc_parent(row["uid"])), None
        raise ContractError(f"agentic source {row['source']!r}")
    if corpus == "naive":
        return arc_group(nvarc_parent(row["uid"])), "nvarc"
    if corpus == "nemotron":
        return f"nemotron:{row['uid']}", "nemotron"
    raise ValueError(corpus)


def parquet_subset(corpus, row, batch):
    return {"agentic": lambda: row["source"], "naive": lambda: row["mode"], "nemotron": lambda: batch}[corpus]()


def row_ids_mask(corpus, row, markers, where, kept=0):
    """A parquet row's ids (int64), loss mask, role-1 spans (agentic/naive; [] for nemotron) and how many of its
    assistant turns the budget cut."""
    if corpus == "nemotron":
        ids = np.asarray(row["input_ids"], np.int64)
        mask = nemotron_mask(ids, row["loss_mask"], row["n_prompt"], row["n_completion"], row["finish_reason"],
                             markers, where)
        return ids, mask, [], int(row["finish_reason"] == "length")
    ids = np.asarray(row["ids"], np.int64)
    spans, cut = assistant_spans(ids, row["span_start"], row["span_end"], row["span_role"], markers,
                                 truncation="appended_im_end" if corpus == "agentic" else "rollout_ends", where=where)
    if sum(e - s for s, e in spans) != row["n_completion"]:
        raise ContractError(f"{where}: n_completion {row['n_completion']} != the assistant spans' length")
    if kept > len(spans):
        raise ContractError(f"{where}: {kept} kept rounds of {len(spans)}")
    return ids, span_mask(len(ids), spans[kept:]), spans, cut


def text_skip(ids, mask, markers, context_length):
    """Why a text row cannot be a job, or None."""
    if len(ids) + 1 > context_length:
        return "too_long"
    if np.isin(ids, markers.media).any():
        return "media_token"  # a media pad in a text-only request would be read as a placeholder
    if int(mask.sum()) < MIN_SUPERVISED:
        return "few_supervised"
    if anchor_count(mask) == 0:
        return "no_anchors"
    return None


class RowReader:
    """Random access to parquet rows, keeping the last few decoded files (a file is one row group)."""

    def __init__(self, sources, keep=4):
        self.sources = sources
        self.keep = keep
        self.cache = OrderedDict()

    def table(self, index):
        if index not in self.cache:
            src = self.sources[index]
            path = Path(src["path"])
            if path.stat().st_size != src["bytes"]:
                raise RuntimeError(f"{path} is {path.stat().st_size} bytes, the plan read {src['bytes']}")
            table = pq.read_table(path, columns=PARQUET_COLUMNS[src["corpus"]])
            if table.num_rows != src["rows"]:
                raise RuntimeError(f"{path} has {table.num_rows} rows, the plan read {src['rows']}")
            self.cache[index] = table
            while len(self.cache) > self.keep:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def row(self, index, row):
        return self.table(index).slice(row, 1).to_pylist()[0]


# ----------------------------------------------------------------------------- node1 ledgers


def workspace(seg, vocab, head=512):
    """The AGENTS.md workspace path in the segment's first user message, or None (sessions that are not codex runs)."""
    ids = seg.ids
    starts = np.flatnonzero(ids[:1 << 16] == vocab.im_start)
    if len(starts) < 2:
        return None
    text = vocab.decode(ids[starts[1]:starts[1] + head])
    if not text.startswith("<|im_start|>user\n"):
        return None
    m = AGENTS_RE.match(text[len("<|im_start|>user\n"):])
    return None if m is None else m.group(1)


def classify_workspace(path):
    """{lane, pool, task, run, batch} of a node1 workspace path; raises on a path that is neither lane."""
    m = ARC2_RE.search(path)
    if m is not None:
        if m["pool"] not in CLEAN_ARC2 + TAINTED_ARC2:
            raise ContractError(f"unknown ARC-AGI-2 pool {m['pool']!r} in {path}")
        return {"lane": "arc2", "pool": m["pool"], "task": m["task"], "batch": m["batch"], "run": None}
    m = ARC3_RE.search(path)
    if m is not None:
        return {"lane": "arc3", "pool": "arc3", "task": m["game"], "batch": None, "run": m["run"]}
    m = SMOKE_RE.search(path)
    if m is not None:
        return {"lane": "smoke", "pool": "smoke", "task": m["task"], "batch": m["batch"], "run": None}
    raise ContractError(f"workspace {path} is neither an ARC-AGI-2 task, an ARC-AGI-3 game nor a smoke run")


def node1_group(where):
    """(group, held-out bucket or None) of a classified segment."""
    if where["pool"] in ("probe", "smoke"):
        return f"{where['pool']}:{where['task']}", None
    if where["pool"] == "nvarc_training":
        return arc_group(nvarc_parent(where["task"])), "arc2"
    if where["pool"] == "training1000":
        return arc_group(where["task"]), "arc2"
    if where["pool"] in TAINTED_ARC2:
        return f"{where['pool']}:{where['task']}", None
    if where["pool"] == "arc3":
        return f"arc3:{where['task']}", "arc3"
    raise ValueError(where)


def classify_codex1015(path):
    """{lane, pool, task, run, batch} of a codex1015 workspace path (CODEX_*_RE); raises on any other path. arc2's
    task drops the episode suffix (__s<k>: s0, s2, the refiners), so a task's episodes share one group."""
    m = CODEX_ARC2_RE.search(path)
    if m is not None:
        return {"lane": "arc2", "pool": "independent1000", "task": m["task"], "batch": None, "run": m["run"]}
    m = CODEX_ARC3_RE.search(path)
    if m is not None:
        return {"lane": "arc3", "pool": m["subset"], "task": m["game"], "batch": None, "run": m["run"]}
    m = CODEX_SMOKE_RE.search(path)
    if m is not None:
        return {"lane": "arc2", "pool": "smoke", "task": m["task"], "batch": None, "run": m["run"]}
    m = CODEX_TEST_RE.search(path)
    if m is not None:
        return {"lane": "arc3", "pool": "test", "task": m["game"], "batch": None, "run": m["run"]}
    raise ContractError(f"workspace {path} is neither a codex1015 ARC-AGI-2 task, an ARC-AGI-3 game, its smoke nor a "
                        f"harness test")


def codex1015_group(where):
    """(group, held-out bucket or None) of a classified codex1015 segment: arc2 by task, arc3 by game within its
    subset; the smoke threads join their task's group but are never drawn (a held-out task excludes them)."""
    if where["lane"] == "arc2":
        return f"independent1000:{where['task']}", (None if where["pool"] == "smoke" else "arc2")
    if where["lane"] == "arc3" and where["pool"] in CODEX_ARC3_SUBSETS:
        return f"arc3:{where['task']}", where["pool"]
    raise ValueError(where)


def classify_int3c1019(path):
    """{lane, pool, task, run, batch} of an int3c1019 workspace path; raises on any other path. The pool names its
    source ("i3-..."), so the plan's subsets keep it apart from codex1015's."""
    m = INT3C_ARC2_RE.search(path)
    if m is not None:
        return {"lane": "arc2", "pool": f"i3-{INT3C_ARC2_POOLS[m['src']]}", "task": m["task"], "batch": None,
                "run": m["run"]}
    m = INT3C_ARC3_RE.search(path)
    if m is not None:
        return {"lane": "arc3", "pool": f"i3-comm-{m['tier']}", "task": m["game"], "batch": None, "run": m["run"]}
    raise ContractError(f"workspace {path} is neither an int3c1019 ARC-AGI-2 task nor an ARC-AGI-3 game")


def classify_bench(path):
    """{lane, pool, task, run, batch} of a bench workspace path (base or PLE arm); raises on any other path."""
    m = BENCH_ARC2_RE.search(path)
    if m is not None:
        return {"lane": "arc2", "pool": f"bench{m['arm']}-{BENCH_ARC2_POOLS[m['lane']]}", "task": m["task"],
                "batch": None, "run": m["run"]}
    m = BENCH_ARC3_RE.search(path)
    if m is not None:
        return {"lane": "arc3", "pool": f"bench{m['arm']}-public25", "task": m["game"], "batch": None, "run": m["run"]}
    raise ContractError(f"workspace {path} is neither a bench ARC-AGI-2 task nor an ARC-AGI-3 game")


def train_only_group(where):
    """(group, None) of a classified int3c1019 / bench segment: ARC-AGI-2 by task (an NVARC task by its parent, as in
    every corpus), ARC-AGI-3 by game. No bucket -- these segments are never drawn as held-out."""
    pool = where["pool"]
    if where["lane"] == "arc3":
        return f"arc3:{where['task']}", None
    if pool == "i3-nvarc_training":
        return arc_group(nvarc_parent(where["task"])), None
    if pool.endswith("-eval120"):
        return arc_group(where["task"]), None
    for name in ("quality300", "independent1000"):
        if pool.endswith(f"-{name}"):
            return f"{name}:{where['task']}", None
    raise ValueError(where)


def classify_threads(segments):
    """segment key -> classification, from (key, thread or None, classification or None) triples. Every segment of a
    thread takes the thread's one classification (later windows need not repeat the AGENTS.md message); a thread
    naming two workspaces raises. Requests outside any codex workspace -- the server checks before the collection
    (node1: 8 sessionless requests and 6 "smoke*" threads, 3k tokens) -- are pool "probe", train only, one group a
    thread (a sessionless request is its own group)."""
    by_thread = {}
    for key, thread, where in segments:
        if thread is not None and where is not None:
            by_thread.setdefault(thread, set()).add(json.dumps(where, sort_keys=True))
    out = {}
    for key, thread, where in segments:
        if thread is None:
            found = set() if where is None else {json.dumps(where, sort_keys=True)}
        else:
            found = by_thread.get(thread, set())
        if len(found) > 1:
            raise ContractError(f"thread {thread} names {len(found)} workspaces: {sorted(found)[:2]}")
        if found:
            out[key] = json.loads(next(iter(found)))
        else:
            out[key] = {"lane": None, "pool": "probe", "task": thread or key, "batch": None, "run": None}
    return out


# ----------------------------------------------------------------------------- plan files


def job_id(split, index):
    return f"{split}-{index:09d}"


def load_plan(plan_path):
    """(meta, jobs) of a plan: plan.jsonl and plan_meta.json beside it, checked against each other."""
    plan_path = Path(plan_path)
    meta = json.loads((plan_path.parent / "plan_meta.json").read_text())
    raw = plan_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != meta["plan_sha256"]:
        raise RuntimeError(f"{plan_path} is not the plan plan_meta.json describes (sha256 differs)")
    jobs = [json.loads(line) for line in raw.decode().splitlines()]
    if len(jobs) != meta["jobs"] or any(job["index"] != i for i, job in enumerate(jobs)):
        raise RuntimeError(f"{plan_path}: {len(jobs)} jobs, meta says {meta['jobs']}, or the indices are not 0..n-1")
    return meta, jobs


def shard_of(jobs, shards):
    """index -> shard, balanced by tokens: in plan order, each job goes to the shard with the fewest tokens so far
    (ties to the lowest shard), so the producers move through the plan together."""
    load = [0] * shards
    out = {}
    for job in jobs:
        i = min(range(shards), key=lambda k: (load[k], k))
        out[job["index"]] = i
        load[i] += job["num_tokens"]
    return out


def split_queue_gib(jobs, split, shards, gr_bytes_per_token):
    """Per shard, the --max-queue-gib under which capture_plan.py captures the whole split without waiting for a
    consumer: every job's GR stays in the root (held-out has no consumer until pipeline.sh moves it out), plus the
    producer's 3x headroom for the largest job it still has to assemble (capture_tito.wait_for_room). PR #6 F1."""
    split_jobs = [job for job in jobs if job["split"] == split]
    owner = shard_of(split_jobs, shards)
    held, largest = [0] * shards, [0] * shards
    for job in split_jobs:
        size = job["num_tokens"] * gr_bytes_per_token
        held[owner[job["index"]]] += size
        largest[owner[job["index"]]] = max(largest[owner[job["index"]]], size)
    return [(h + 3 * m) / 2**30 * 1.02 + 1 for h, m in zip(held, largest, strict=True)]


OPD_KEY_PREFIX = "opd:"  # make_opd_plan.py's keys


def read_consumed(path, jobs):
    """{(key, pass)} from the trainer's JSONL of finished jobs. Every line must name a job of the plan; only a last line
    without its newline may fail to parse (a writer killed mid-line), and that one is reported, not read."""
    known = {(job["key"], job["pass"]) for job in jobs}
    raw = Path(path).read_bytes()
    lines = raw.split(b"\n")
    torn = None
    if lines and lines[-1] == b"":
        lines.pop()
    elif lines:
        try:
            json.loads(lines[-1])
        except json.JSONDecodeError:
            torn = lines.pop().decode(errors="replace")
    consumed = set()
    for number, line in enumerate(lines, 1):
        entry = json.loads(line)
        if not isinstance(entry, dict) or not {"key", "pass"} <= set(entry):
            raise ValueError(f"{path}:{number}: a consumed line needs key and pass: {line[:200]!r}")
        if entry["key"].startswith(OPD_KEY_PREFIX):
            continue  # an OPD rollout job (opd_assemble.py) trained in the same run: not this plan's to capture
        pair = (entry["key"], entry["pass"])
        if pair not in known:
            raise ValueError(f"{path}:{number}: {pair} is not a job of this plan")
        consumed.add(pair)
    return consumed, torn
