"""role and span_kind of one S.

**role** (for the existing SFT downstream; ``span_role`` in ``work/qwen38fn_distill/rollout_runner.py`` OUT_SCHEMA):

- 1 = gen span (tokens the model actually sampled in this span; adjacent gen spans merge into one)
- 0 = every token before the first gen span
- 2 = all other non-generated tokens (tool responses, user messages, ``<|im_end|>`` filled in by the template, the next
  cue, ...)

So the role sequence is always ``0 (1 2)* 1?`` -- the shape ``work/qwen35_9b_sft/convert_distill_shards.py:_check_row``
expects (the same normalisation as ``work/agentic_distill/render_segments.py:normalize_roles``). The spec's "role 2 =
user blocks holding tool responses; everything else role 0" cannot produce this shape (a user message between two gen
spans would become ``1 0 1``), so per the task instructions the definition above is used.

**span_kind** (where each token comes from, parsed from the special tokens of the Qwen3.8 chat template; ids are always
asked of the tokenizer):

==================  ==================================================================================
``other``           non-generated tokens outside any chat block (the raw prompt received by ``--session-ledger-all``)
``sep``             newlines between blocks (after ``<|im_end|>``, before the next ``<|im_start|>``)
``system``          the system block. Developer messages land here too: sglang Responses tags developer with
                    ``Developer instructions:`` and merges it into the single system (``serving_responses.py``)
``user``            ordinary user block
``harness_user``    user block whose content starts with one of ``harness_prefixes`` (codex / harness injection;
                    defaults to the codex prefixes)
``tool_response``   user block whose content starts with ``<tool_response>`` (the template renders tool messages as
                    user)
``assistant_history``  assistant block without any generated token (history re-rendered by the template)
``assistant_cue``   non-generated template tokens inside an assistant block that contains generated tokens
                    (``<|im_start|>assistant\\n<think>\\n``, tokens the template fills in after a rewind
                    (``\\n</think>\\n\\n<|im_end|>``, the ``<|im_end|>`` put in by stop_substituted))
``image`` / ``video``  ``<|vision_start|>`` pad run ``<|vision_end|>``
``video_timestamp`` the ``<t seconds>`` text before a video time group (how the Qwen3-VL processor expands it;
                    **qwen4_exp path not verified**)
``gen_reasoning``   generated, between ``<think>`` and ``</think>`` (including ``</think>``)
``gen_content``     generated, outside reasoning and outside tool calls
``gen_tool_call``   generated, ``<tool_call>`` ... ``</tool_call>`` (both ends included)
``gen_stop``        the last generated token is ``<|im_end|>`` / ``<|endoftext|>``
==================  ==================================================================================

A broken template contract (a block opened before the previous one is closed, an unknown role, a pad not wrapped in
vision markers) always raises ``ValueError``.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

ROLE_PROMPT, ROLE_ASSISTANT, ROLE_TOOL = 0, 1, 2

KINDS = ("other", "sep", "system", "user", "harness_user", "tool_response", "assistant_history", "assistant_cue",
         "image", "video", "video_timestamp", "gen_reasoning", "gen_content", "gen_tool_call", "gen_stop")
K = {name: i for i, name in enumerate(KINDS)}
GEN_KINDS = frozenset(K[k] for k in ("gen_reasoning", "gen_content", "gen_tool_call", "gen_stop"))

#: prefixes codex injects as user messages (actually seen in the 2026-09 codex rollouts: AGENTS.md,
#: environment_context; the rest are fixed codex-rs markers). The compaction summary prefix has not been seen in real
#: data yet, so it is not listed -- see the report.
HARNESS_USER_PREFIXES = (
    "# AGENTS.md instructions for ",
    "<environment_context>",
    "<user_instructions>",
    "<user_shell_command>",
    "<turn_aborted>",
)
ROLES = ("system", "user", "assistant")
_TIMESTAMP_RE = re.compile(r"<\d+(?:\.\d+)? seconds>")
_EFFORT_RE = re.compile(r"Reasoning effort is set to (\w+)\.")


# ----------------------------------------------------------------------------- roles


def merge_gen_spans(gen: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge adjacent gen spans (previous end == next start); they must be sorted, non-overlapping and non-empty."""
    out: list[tuple[int, int]] = []
    for s, e in gen:
        if e <= s:
            raise ValueError(f"empty gen span [{s}, {e})")
        if out and s < out[-1][1]:
            raise ValueError(f"gen span [{s}, {e}) overlaps the previous {out[-1]} or is out of order")
        if out and s == out[-1][1]:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return out


def role_spans(n: int, gen: Iterable[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """``[(start, end, role)]`` covering ``[0, n)``, shape ``0 (1 2)* 1?``. Without gen it is a single role 0."""
    merged = merge_gen_spans(gen)
    if not merged:
        return [(0, n, ROLE_PROMPT)] if n else []
    if merged[0][0] == 0:
        raise ValueError("first gen span starts at 0: no prompt")
    if merged[-1][1] > n:
        raise ValueError(f"gen span {merged[-1]} exceeds S length {n}")
    out = [(0, merged[0][0], ROLE_PROMPT)]
    for i, (s, e) in enumerate(merged):
        out.append((s, e, ROLE_ASSISTANT))
        nxt = merged[i + 1][0] if i + 1 < len(merged) else n
        if nxt > e:
            out.append((e, nxt, ROLE_TOOL))
    check_role_spans(out, n)
    return out


def check_role_spans(spans: Sequence[tuple[int, int, int]], n: int) -> None:
    """The structural part of ``convert_distill_shards._check_row`` (full cover, contiguous, non-empty,
    ``0 (1 2)* 1?``)."""
    if not spans:
        raise ValueError("empty span table")
    if spans[0][0] != 0 or spans[-1][1] != n:
        raise ValueError(f"spans do not cover [0, {n}): {spans[0][0]}..{spans[-1][1]}")
    for (s0, e0, _), (s1, _, _) in zip(spans, spans[1:]):
        if e0 != s1:
            raise ValueError(f"spans have a gap or overlap: {e0} → {s1}")
    if any(e <= s for s, e, _ in spans):
        raise ValueError("empty span present")
    roles = [r for _, _, r in spans]
    if roles[0] != ROLE_PROMPT or any(r != ROLE_ASSISTANT for r in roles[1::2]) or any(r != ROLE_TOOL for r in roles[2::2]):
        raise ValueError(f"role sequence {roles[:12]} is not 0 (1 2)* 1?")


# ----------------------------------------------------------------------------- chat blocks


@dataclass(frozen=True)
class Block:
    start: int        # position of <|im_start|>
    end: int          # the <|im_end|> that closes it (inclusive); len(S)-1 if not closed before the end of S
    role: str
    content: int      # first content token (after the newline of the role line)


def gen_mask(n: int, gen: Iterable[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n, bool)
    for s, e in gen:
        mask[s:e] = True
    return mask


def chat_blocks(ids: np.ndarray, is_gen: np.ndarray, vocab) -> list[Block]:
    """A non-generated ``<|im_start|>`` opens a block that runs to the next ``<|im_end|>`` (generated ones count too)."""
    n = len(ids)
    openers = np.flatnonzero((ids == vocab.im_start) & ~is_gen)
    ends = np.flatnonzero(ids == vocab.im_end)
    out = []
    for i, p in enumerate(openers.tolist()):
        j = int(np.searchsorted(ends, p))
        e = int(ends[j]) if j < len(ends) else n - 1
        nxt = int(openers[i + 1]) if i + 1 < len(openers) else n
        if e >= nxt:
            raise ValueError(f"<|im_start|> block at position {p} is not closed before the next <|im_start|> ({nxt})")
        text = ""
        content = None
        for q in range(p + 1, min(p + 17, e + 1)):     # role name + newline; a byte-level tokenizer has one token per character
            text += vocab.decode([ids[q]])
            if "\n" in text:
                content = q + 1
                break
        role = text.partition("\n")[0]
        if content is None or role not in ROLES:
            raise ValueError(f"block at position {p}: role line {text!r} is not one of {ROLES} (unknown role)")
        out.append(Block(start=p, end=e, role=role, content=content))
    return out


def _user_kind(ids: np.ndarray, b: Block, vocab, prefixes: Sequence[str]) -> int:
    if b.content <= b.end and ids[b.content] == vocab.tool_response:
        return K["tool_response"]
    head = vocab.decode(ids[b.content: min(b.content + 64, b.end)])
    if any(head.startswith(p) for p in prefixes):
        return K["harness_user"]
    return K["user"]


def _label_gen(ids: np.ndarray, s: int, e: int, in_think: bool, in_tool: bool, kinds: np.ndarray, v) -> None:
    span = ids[s:e]
    marks = s + np.flatnonzero(np.isin(span, (v.think, v.think_end, v.tool_call, v.tool_call_end)))
    cur = s

    def kind() -> int:
        return K["gen_reasoning"] if in_think else (K["gen_tool_call"] if in_tool else K["gen_content"])

    for m in marks.tolist():
        kinds[cur:m] = kind()
        t = ids[m]
        if t == v.think:
            in_think = True
            kinds[m] = K["gen_reasoning"]
        elif t == v.think_end:
            kinds[m] = K["gen_reasoning"]
            in_think = False
        elif t == v.tool_call:
            in_tool = in_tool or not in_think
            kinds[m] = kind()
        else:  # </tool_call>
            kinds[m] = kind()
            in_tool = False
        cur = m + 1
    kinds[cur:e] = kind()
    if ids[e - 1] in (v.im_end, v.endoftext):
        kinds[e - 1] = K["gen_stop"]


def _state_before(ids: np.ndarray, lo: int, hi: int, v) -> tuple[bool, bool]:
    """think / tool_call state between the block start and the gen span start (including earlier gen spans in the same
    block)."""
    seg = ids[lo:hi]
    think = np.flatnonzero((seg == v.think) | (seg == v.think_end))
    in_think = bool(think.size) and seg[think[-1]] == v.think
    tool = np.flatnonzero((seg == v.tool_call) | (seg == v.tool_call_end))
    in_tool = (not in_think) and bool(tool.size) and seg[tool[-1]] == v.tool_call
    return in_think, in_tool


def token_kinds(ids: np.ndarray, gen: Sequence[tuple[int, int]], vocab,
                harness_prefixes: Sequence[str] = HARNESS_USER_PREFIXES) -> np.ndarray:
    """span_kind code of every token (index into ``KINDS``, int8)."""
    ids = np.asarray(ids)
    n = len(ids)
    v = vocab
    is_gen = gen_mask(n, gen)
    kinds = np.full(n, K["other"], np.int8)
    blocks = chat_blocks(ids, is_gen, v)
    for i, b in enumerate(blocks):
        nxt = blocks[i + 1].start if i + 1 < len(blocks) else n
        kinds[b.end + 1: nxt] = K["sep"]
        if b.role == "system":
            k = K["system"]
        elif b.role == "user":
            k = _user_kind(ids, b, v, harness_prefixes)
        else:
            k = K["assistant_cue"] if is_gen[b.start: b.end + 1].any() else K["assistant_history"]
        kinds[b.start: b.end + 1] = k

    for vs in np.flatnonzero((ids == v.vision_start) & ~is_gen).tolist():
        q = vs + 1
        if q >= n or ids[q] not in (v.image_pad, v.video_pad):
            raise ValueError(f"<|vision_start|> at position {vs} is not followed by a pad")
        pad = ids[q]
        while q < n and ids[q] == pad:
            q += 1
        if q >= n or ids[q] != v.vision_end:
            raise ValueError(f"pad run at position {vs} does not end with <|vision_end|>")
        video = pad == v.video_pad
        kinds[vs: q + 1] = K["video"] if video else K["image"]
        if video:
            for back in range(1, 33):         # real tokenizer: one token per digit; synthetic byte-level: one per char
                lo = vs - back
                if lo < 0 or is_gen[lo] or kinds[lo] in (K["image"], K["video"]):
                    break
                if _TIMESTAMP_RE.fullmatch(v.decode(ids[lo:vs])):
                    kinds[lo:vs] = K["video_timestamp"]
                    break
    stray = np.flatnonzero(np.isin(ids, (v.image_pad, v.video_pad)) & ~np.isin(kinds, (K["image"], K["video"])))
    if stray.size:
        raise ValueError(f"found {stray.size} pads not inside <|vision_start|>…<|vision_end|>, "
                         f"e.g. positions {stray[:3].tolist()}")

    starts = [b.start for b in blocks]
    for s, e in gen:
        i = bisect.bisect_right(starts, s - 1) - 1
        if i >= 0 and s <= blocks[i].end:
            in_think, in_tool = _state_before(ids, blocks[i].start, s, v)
        else:
            in_think, in_tool = False, False
        _label_gen(ids, s, e, in_think, in_tool, kinds, v)
    return kinds


def kind_spans(kinds: np.ndarray) -> list[tuple[int, int, int]]:
    """Per-token codes → run-length ``[(start, end, code)]``."""
    n = len(kinds)
    if not n:
        return []
    cut = np.flatnonzero(np.diff(kinds)) + 1
    starts = np.concatenate(([0], cut))
    ends = np.concatenate((cut, [n]))
    return [(int(s), int(e), int(kinds[s])) for s, e in zip(starts, ends)]


def n_tool_calls(ids: np.ndarray, kinds: np.ndarray, vocab) -> int:
    """Generated ``<tool_call>``s (ones written inside reasoning do not count)."""
    return int(np.count_nonzero((np.asarray(ids) == vocab.tool_call) & (kinds == K["gen_tool_call"])))


def reasoning_effort(ids: np.ndarray, gen: Sequence[tuple[int, int]], vocab) -> str | None:
    """Read the Qwen3.8 template's reasoning effort from the tokens.

    The template writes ``Reasoning effort is set to …`` into system only for xhigh / low; medium and
    ``enable_thinking=false`` write nothing, and the latter's cue is ``<think>\\n\\n</think>\\n\\n``. Returns
    ``xhigh`` / ``low`` / ``medium`` / ``none`` (thinking off); None when there is neither a system nor any cue (e.g. a
    raw prompt)."""
    ids = np.asarray(ids)
    is_gen = gen_mask(len(ids), gen)
    blocks = chat_blocks(ids, is_gen, vocab)
    system = [b for b in blocks if b.role == "system"]
    if system:
        b = system[0]
        m = _EFFORT_RE.search(vocab.decode(ids[b.content: b.end]))
        if m:
            return m.group(1)
    for b in blocks:
        if b.role == "assistant" and is_gen[b.start: b.end + 1].any():
            g = b.start + int(np.argmax(is_gen[b.start: b.end + 1]))
            return "none" if (ids[b.content: g] == vocab.think_end).any() else "medium"
    return None
