"""One model proxy per task: sitting between codex ↔ sglang, the only place on the harness side that touches requests.

What it does:
  budget   Rewrites ``max_output_tokens = remaining budget`` (reasoning included) on every POST /v1/responses. codex
           does not send this field itself, so sglang allows the whole rest of the context -- smoke1 measured one xhigh
           response at 57,706 tokens in one go, so killing the process from rollout accounting overshoots by up to a
           whole response. With the rewrite the budget is exact.
  count    Reads usage from the SSE response.completed / response.incomplete and accumulates this task's output tokens.
  log      One jsonl line per request: input/cached/output/reasoning tokens, status, TTFT, seconds. Also logged when
           codex is killed mid-stream (``client_disconnected``; that row has no usage).
  salt     (optional, config ``cache_salt: "episode"``) writes ``salt_field = salt`` into every /responses body: one
           value per episode, the same across all turns and sessions, so sglang's prefix cache is not shared across
           episodes. If the body already has this field: return 500 and record it in ``fatal`` (the harness's next poll
           sees it and fails this episode); never overwrite.

  DP       When ``dp_rank`` is not None, injects ``routed_dp_rank`` into every /responses body (one fixed DP per
           episode: sglang's session KV lives only on the DP serving it). The fork's ResponsesRequest has this field,
           and the DP controller's maybe_external_dp_rank_routing sends it straight to that DP (Kaggle line, confirmed
           by reading sglang-arc-2 on 10-04).
  rid      Sets ``request_id`` (sglang's rid) = ``resp_arc2_<task>_<n>_<random>`` in every /responses body and tracks
           the in-flight ones; when the episode ends, first POST /abort_request to stop those still generating, then
           drop the upstream, then close the session -- sglang detects disconnects asynchronously (a mock measured
           close arriving 2 s before disconnect detection), while abort and close travel the same
           tokenizer→scheduler queue, so their order is guaranteed.
  session  Records each /responses body's ``session_id`` (the window id of codex patch 0002) and every /close_session
           that passes through here and returns 200. A killed codex does not close (L4 HiSparse's idle timeout is 0, so
           an unclosed session keeps holding logical tokens); the harness sends those afterwards with
           ``abort_upstreams`` + ``close_open_sessions``.
  identity When ``allowed_uid`` is not None (posix fence, TCP proxy without netns: Kaggle), only connections from that
           uid are accepted: the owner of the peer socket is looked up in /proc/net/tcp.

  routes   When ``routes`` is not None (arc3 single-level episodes, run_arc3_level.py), only the (method, path) pairs in
           the list are forwarded; the rest get 403 and a ``forbidden`` row: through this proxy the agent in the jail
           can reach every sglang route (weight swap, flush, ledger state). ``require_session``: /responses must carry
           codex's ``session_id`` -- otherwise the agent's own scripts could call the model as a subroutine, spending
           budget tokens that are not in the trajectory. Once ``stop_dir`` (level_episode.py) holds any stop file,
           /responses always gets 503.

  charge   When ``charge`` is not None (arc3 g2seed's per-action charge, run_arc3_level.ActionCharge): ``charge()`` =
           tokens deducted from the budget besides model output (converted from the action count); remaining budget =
           budget − output − charge(). Asked once at the start of each request and on every harness poll; actions
           happen between requests (codex executing tool calls), so each request's cap is still exact. Each request's
           log row gains a column ``action_charge`` (the charge at that request's start). Not combined with wrapup.
  context  When ``context_limit`` is not None (arc3 single-level episodes, round 2): once a thread's context (usage
           input + output returned by the server) reaches this number, the episode ends. Each thread (``session_id``)
           records its context at the end of the previous response, and the next request's ``max_output_tokens`` is
           further capped to ``context_limit − that number`` (new tool output is not counted yet, so the actual
           overshoot is at most this round's new input); if after a response the context is less than
           ``EXHAUSTED_BELOW`` from the limit, it counts as used up: later /responses always get 503 and the harness
           writes the stop file ``context`` (handled the same way as an exhausted output budget; the episode is still
           a sample that did not clear the level). A response whose cap was the remaining context (not the budget, not
           what codex asked for) ending incomplete (max_output_tokens) also counts as used up (``context_cut``): codex
           treats incomplete as a stream error and resends after dropping the output; the dropped round stays in the
           ledger but not in the manifest, and since it is not the last entry rl_rid_join cannot take it (c0b-lr5e-5
           g000111 vc33_L06 s5: 0116, with 1996 left, was cut off; the resent 0117 fit, and the episode went on).
  window   ``stop_after_first_window`` (arc3 g2seed's RL v8, the same-named key in run_arc3_level's harness config): the
           game ends at the first window change. The first /responses with window ≥ 1 (``session_id`` =
           ``<thread>:<n>``, n ≥ 1: codex's first request after a window change) is not forwarded upstream; it gets 503
           and ``window_cut`` {sid, t} is recorded; later /responses always get 503. It is blocked before request_id,
           note_session and accounting: neither the ledger nor the jsonl has it, the ledger has only window 0. When
           the harness sees ``window_exhausted()`` it writes the stop file ``window``.
  pace     When ``pace_path`` is not None (arc3 g2seed's ``pace_line``, run_arc3_level.pace_path): before each
           /responses is sent upstream, and before its end event is forwarded to codex, writes
           ``{"level_output_tokens": cumulative output, "level_output_budget": budget}`` -- the last line of gen-2
           act's ``do`` / ``status`` (act_client's PACE_HOOK). The cumulative output is the number the budget uses
           (``charge`` excluded); the end-time write adds this response's output (account() runs only after the
           stream is fully read, and codex may execute a tool call first). The file is in the agent's workspace:
           O_EXCL|O_NOFOLLOW temp file + os.replace, no symlinks followed; if the workspace is not a regular directory
           or the write fails, it is recorded in ``fatal`` (the harness's next poll fails the episode), then raised.

All other requests (/v1/models, /health, /close_session) are forwarded as is (when ``routes`` is None, even those not
in the list). Responses always end with Connection: close (no keep-alive); bodies are streamed as is.
"""

from __future__ import annotations

import contextlib
import copy
import http.client
import json
import os
import re
import socket
import socketserver
import threading
import time
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler
from pathlib import Path

HOP = {"connection", "keep-alive", "transfer-encoding", "content-length", "host", "proxy-connection", "te",
       "trailer", "upgrade"}


#: Fewer than this many left counts as used up. sglang's Responses subtracts 2 from max_new_tokens up front (BOS/EOS
#: margin); with 2 left it returns incomplete with 0 tokens -- the count never reaches the cap, while codex treats
#: incomplete as a stream error and keeps resending (v1's 20270e3b: stuck at cum 99,998).
EXHAUSTED_BELOW = 64
#: Stop-loss: this many consecutive responses ending incomplete with 0 output tokens count as an exhausted budget
#: (remaining() returns 0, later requests get 503, the harness records stop budget). 10-24 arc2 RL: when the wrap-up
#: line was missed by 2 tokens, codex retried a request capped at 2 until the 4 h wall (up to 253 times).
ZERO_OUTPUT_STREAK = 5
#: sglang's Qwen-VL preprocessing (transformers smart_resize) rejects images with aspect ratio >= 200, returning
#: response.failed before generating. The image stays in the thread, so codex's retries and the harness resuming the same
#: thread both resend the same image and fail the same way: run_arc2 treats it as a model error, waits for /health and
#: resumes, up to 20 times (hq741 10-31 q340__s1: the model wrote its own PNG encoder, drew a 4508x18 image and viewed it,
#: and was stuck 50 minutes before being stopped by hand).
#: The user, 10-31: this is unwanted behavior; the episode ends right away (end_reason image_rejected) and is scored on
#: the answers on disk.
IMAGE_REJECTED = b"absolute aspect ratio must be smaller than"

#: sglang's Qwen-VL preprocessing also treats a literal ``<image>`` (the old /generate placeholder) as an image position
#: (image_token_regex in qwen_vl.py). In a request with images, once the model's own reasoning or a tool output
#: contains this string, the image count no longer matches ("More 'IMAGE' tokens found than corresponding data
#: provided"), and every later request in this thread fails (v8img 7ed72f31, 10-04). Before sending, a zero-width space
#: is inserted in the middle (JSON-escaped); the substitution is identical every time, so the prefix cache is unaffected.
#: For Responses requests with images / video sglang passes max_output_tokens through unchanged (without images it is
#: clamped to context − prompt); if input + max_output_tokens exceeds the context it returns response.failed before
#: generating (the fork's ARC_0018 comment). Every codex request carries the first message's images, so when "remaining
#: budget < context" the cap the proxy sets may not fit (10-11 INT3 CTXLEN 150k: 150 times in one run, sessions failing
#: after 12 retries). The error message carries the actual input token count: the proxy shrinks the cap to context −
#: input − this margin accordingly and resends once.
OVERFLOW_RE = re.compile(rb"Requested token count exceeds the model's maximum context length.*?"
                         rb"You requested a total of \d+ tokens: (\d+) tokens from the input messages")
CAP_MARGIN = 64

IMAGE_SENTINEL = b"<image>"
IMAGE_SENTINEL_SAFE = b"<image\\u200b>"

#: The file of codex patch 0005 ($CODEX_HOME/compact_reminder): the first line is the threshold (context token count),
#: the rest is the user message to insert into the in-progress turn. codex rereads it at every follow-up (after a tool
#: result comes back, before sending the next request); an empty message does nothing.
REMINDER_FILE = "compact_reminder"
REMINDER_DORMANT = "0\n"

#: The proxy of an arc3 single-level episode lets through only the routes codex uses: /responses, patch 0002's
#: /close_session (server root), the read-only /v1/models and /health. Everything else (/update_weights_*, /flush_cache,
#: /abort_request, /session_ledger/*, /metrics, ...) gets 403.
CODEX_ROUTES = frozenset({("POST", "/v1/responses"), ("POST", "/close_session"),
                          ("GET", "/v1/models"), ("GET", "/health")})


def user_texts(items) -> list[str]:
    """The input_text of every user message in a Responses input."""
    out = []
    for item in items or []:
        if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), list):
            out += [c.get("text") or "" for c in item["content"] if isinstance(c, dict) and c.get("type") == "input_text"]
    return out


class Wrapup:
    """Wrap-up (config ``wrapup_tokens``): once output usage passes ``soft`` (= the config's output_token_budget), ask the
    model to wrap up immediately; the proxy's cap is ``hard`` = soft + wrapup_tokens.

    codex is not killed: **before** the end event of the response that crossed the line is forwarded to codex, ``cross()``
    writes the wrap-up message into compact_reminder (threshold 0), and codex inserts it into the same turn at its next
    follow-up -- same thread, same effort, the prefix cache still hits. The first request after that whose input carries
    this user message (``carries()``; inserted by patch 0005, or the prompt when the harness resumes) and that the server
    starts answering is recorded by ``mark_seen()``, and the file is reset to DORMANT (otherwise a compaction during
    wrap-up would shrink the context and insert it again). Requests the proxy blocks (budget, context full) do not count
    -- the model did not see it then. If codex ends the turn on its own before the insertion, run_arc2.solve() resumes the
    same thread once with ``text``.

    ``cut`` (config ``wrapup_cut``): before the line is crossed each request's cap is ``soft − usage``, not
    ``hard − usage``, so the response that crosses the line is cut off exactly at soft (even mid-thinking); only after the
    line is crossed is the cap relaxed to hard. Without it the crossing response can generate all the way to hard, and the
    wrap-up message can be inserted only after it ends on its own.
    With cut the proxy inserts the wrap-up message itself (``inject``), not via patch 0005: on a cut-off (incomplete)
    response codex stores the thinking as a reasoning item and resends immediately (not a follow-up, so 0005 cannot
    insert), and an ending that is only reasoning renders on the server as an empty user message, dropping the thinking
    (measured 10-15 on GPU 7). The proxy inserts a user wrap-up message after the last item (the cut-off part) of the
    first request after the line, and in every later request at the same position, so the prompt stays an extension of
    one sequence: the cut-off thinking, ``</think>``, the wrap-up message.

    ``context`` (config ``wrapup_context``, must be enabled together with cut): a second line -- when any thread's
    context (usage input + output returned by the server) reaches this number, wrap up too, however much output was used,
    so an episode never compacts and the sequence never exceeds the trainer's bin (10-24 arc2 RL 200k: games that used
    the full budget were compacted by codex or exceeded the bin at server context 192k–248k, and not a single game made
    it into training). Before the line is crossed each request's cap is further lowered to ``context − this thread's
    context at the end of its previous response`` (new tool output on the same thread is not counted yet, so the cut-off
    point overshoots the line by at most this round's new input); the crossing response is still cut off at the line,
    and the wrap-up message is inserted right after it.
    When the context line triggers, wrap-up gets only another hard − soft (wrapup_tokens): the cap is
    ``cap()`` = min(hard, armed_at + hard − soft); when the output line triggers it is hard as before."""

    def __init__(self, path: Path, soft: int, hard: int, template: str, cut: bool = False,
                 context: int | None = None) -> None:
        if not 0 < soft < hard:
            raise ValueError(f"need 0 < soft < hard, not {soft} / {hard}")
        if context is not None and not (cut and isinstance(context, int) and not isinstance(context, bool) and context > 0):
            raise ValueError(f"the context wrap-up line {context!r} must be a positive int and needs cut")
        self.path, self.soft, self.hard, self.template, self.cut = path, soft, hard, template, cut
        self.context = context
        self.lock = threading.Lock()
        self.armed_at: int | None = None      # cumulative output when the line was crossed
        self.armed_by: str | None = None      # which line triggered: "output" or "context"
        self.armed_context: int | None = None  # its thread's context at the end of the triggering response (if usage gave it)
        self.seen_at: int | None = None       # cumulative output when the model first saw the wrap-up message
        self.text: str | None = None
        self.resumed = False                  # solve() has resumed with text
        self.anchor: str | None = None        # cut: the wrap-up message is inserted after this item id in the input
        self.injected = 0                     # cut: number of requests the proxy inserted the wrap-up message into
        self.anchor_missing = 0               # cut: requests not inserted into because the anchor was missing from the input (compaction replaced the history)
        #: cut: the rid of the request cut off at the soft line. codex does not write the id of an incomplete response
        #: into the rollout, though the ledger has this round; the manifest records it so that RL's rid join
        #: (arcforge.tito.join.rl_rid_join) can let it through by name
        self.cut_rid: str | None = None
        #: rids of requests sent only after the line was crossed (wrap-up-phase generation, including new rids from
        #: shrink-and-resend); the crossing request is not among them. The manifest's wrapup_post_rids; prepare_rl
        #: --wrapup-loss masks them out of training
        self.post_rids: list[str] = []

    def _write(self, text: str) -> None:
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(text)
        os.replace(tmp, self.path)

    def dormant(self) -> None:
        self._write(REMINDER_DORMANT)

    def cap(self) -> int:
        """Output cap: hard; when the context line triggered (usage not yet at soft) it is min(hard, cumulative output when
        the line was crossed + hard − soft)."""
        with self.lock:
            if self.armed_by != "context":
                return self.hard
            return min(self.hard, self.armed_at + self.hard - self.soft)

    def context_room(self, context: int) -> int | None:
        """Before the context line triggers: how much this request may generate on a thread whose previous response ended
        at ``context``; None when there is no context line or it has already been crossed. Being less than
        EXHAUSTED_BELOW from the line without having triggered is a bug (cross() runs before that response's end event
        is forwarded to codex)."""
        with self.lock:
            if self.context is None or self.armed_at is not None:
                return None
        room = self.context - context
        if room < EXHAUSTED_BELOW:
            raise RuntimeError(f"a thread's context {context} is past the wrap-up line {self.context} "
                               f"but the wrap-up never armed")
        return room

    def cross(self, total: int, cut_rid: str | None = None, context: int | None = None) -> bool:
        """A response ended, the cumulative output is now total, and its thread's context is now ``context`` (None if
        usage did not give it). The first time either line is crossed, write the wrap-up message and return True.
        ``cut_rid``: this response's rid if it was cut off at the cap (incomplete, max_output_tokens) -- with cut, that is
        the round cut off at the line.
        With cut, being less than EXHAUSTED_BELOW from the line counts as crossed: sglang under-reports usage by 2 for a
        response cut off at max_output_tokens, so the crossing response leaves the total at soft − 2, and a plain ≥ soft
        test would never trigger (the next request is capped at 2, returns 0 tokens, codex retries until the wall;
        10-24 arc2 RL)."""
        line = self.soft - EXHAUSTED_BELOW + 1 if self.cut else self.soft
        by_context = self.context is not None and context is not None and context > self.context - EXHAUSTED_BELOW
        with self.lock:
            if self.armed_at is not None or (total < line and not by_context):
                return False
            if self.cut:
                self.cut_rid = cut_rid
            self.armed_by = "output" if total >= line else "context"
            self.armed_context = context
            left = (self.hard if self.armed_by == "output" else min(self.hard, total + self.hard - self.soft)) - total
            self.text = self.template.format(left_k=max(0, left) // 1000).strip()
            if not self.text:
                raise ValueError("the wrap-up message is empty")
            if not self.cut:                  # cut: the proxy inserts it itself (inject); compact_reminder stays DORMANT
                self._write(f"0\n{self.text}\n")
            self.armed_at = total
            return True

    def inject(self, req: dict) -> bool:
        """cut: insert the wrap-up message into this request's input (modifying ``req`` in place); return True if
        inserted. Not inserted if the input already carries it (the harness resumed with it). For the first request to
        insert into after the line, the last item of its input is the anchor; after that, always insert after the
        anchor."""
        with self.lock:
            if not self.cut or self.text is None:
                return False
            text = self.text
        items = req.get("input")
        if not isinstance(items, list) or not items:
            raise RuntimeError(f"wrap-up: a /responses request after the line has no input list: {type(items).__name__}")
        if any(t.strip() == text for t in user_texts(items)):
            return False
        with self.lock:
            if self.anchor is None:
                anchor = items[-1].get("id") if isinstance(items[-1], dict) else None
                if not anchor:
                    raise RuntimeError(f"wrap-up: the last input item has no id to anchor the message to: {items[-1]!r:.200}")
                self.anchor = anchor
            ids = [item.get("id") if isinstance(item, dict) else None for item in items]
            if self.anchor not in ids:
                self.anchor_missing += 1
                return False
            items.insert(ids.index(self.anchor) + 1, {"type": "message", "role": "user",
                                                      "content": [{"type": "input_text", "text": text}]})
            self.injected += 1
            return True

    def carries(self, req: dict) -> bool:
        """The request's input carries the wrap-up message that has not been delivered yet (no state change)."""
        with self.lock:
            if self.text is None or self.seen_at is not None:
                return False
            text = self.text
        return any(t.strip() == text for t in user_texts(req.get("input")))

    def mark_seen(self, total: int) -> bool:
        """The server started answering a request carrying the wrap-up message: the first time, record total, reset the
        file to DORMANT and return True."""
        with self.lock:
            if self.text is None or self.seen_at is not None:
                return False
            self.seen_at = total
            self._write(REMINDER_DORMANT)
            return True

    def armed(self) -> bool:
        with self.lock:
            return self.armed_at is not None

    def note_post(self, rid: str) -> None:
        """``rid`` is a request sent only after the line was crossed (``post_rids``)."""
        with self.lock:
            if self.armed_at is None:
                raise RuntimeError(f"wrap-up: request {rid} noted as after the line, but the wrap-up never armed")
            self.post_rids.append(rid)

    def record(self, total: int) -> dict:
        with self.lock:
            return {"soft": self.soft, "hard": self.hard, "armed_at": self.armed_at, "seen_at": self.seen_at,
                    "resumed": self.resumed, "after_soft": max(0, total - self.soft), "post_rids": list(self.post_rids),
                    **({"context": self.context, "armed_by": self.armed_by, "armed_context": self.armed_context,
                        "after_armed": None if self.armed_at is None else total - self.armed_at}
                       if self.context is not None else {}),
                    **({"cut": True, "anchor": self.anchor, "injected": self.injected,
                        "anchor_missing": self.anchor_missing, "cut_rid": self.cut_rid} if self.cut else {})}


class ImageReattach:
    """Put the first message's images back after compaction (10-29). The user messages codex keeps across compaction
    retain only input_text: the kickoff message's images (and <image name=...> tags) are all dropped, leaving only the
    "The N images attached to this message ..." sentence -- the model reads that images are attached when there are none.
    The first request carrying the kickoff with its images records the whole content; later, a message with only the
    kickoff text is swapped back to the whole content (same images, same order). A shape mismatch (other text next to
    the kickoff, or a text-only kickoff seen before the images were recorded) stops loudly."""

    def __init__(self, kickoff: str) -> None:
        if not kickoff.strip():
            raise ValueError("ImageReattach needs the kickoff text")
        self.kickoff = kickoff.strip()
        self.content: list | None = None
        self.reattached = 0
        self.lock = threading.Lock()

    def apply(self, req: dict) -> int:
        """Modifies req in place; returns the number of messages restored (0 or 1)."""
        items = req.get("input")
        if not isinstance(items, list):
            raise RuntimeError(f"image reattach: a /responses request has no input list: {type(items).__name__}")
        n = 0
        for item in items:
            if not (isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), list)):
                continue
            content = item["content"]
            texts = [(c.get("text") or "").strip() for c in content if isinstance(c, dict) and c.get("type") == "input_text"]
            if self.kickoff not in texts:
                continue
            has_image = any(isinstance(c, dict) and c.get("type") == "input_image" for c in content)
            with self.lock:
                if has_image:
                    if self.content is None:
                        self.content = copy.deepcopy(content)
                    continue
                if self.content is None:
                    raise RuntimeError("image reattach: the kickoff message came without its images before any "
                                       "request carried them")
                if len(content) != 1:
                    raise RuntimeError(f"image reattach: the image-less kickoff message has {len(content)} parts, "
                                       f"expected only its text: {json.dumps(content)[:300]}")
                item["content"] = copy.deepcopy(self.content)
                self.reattached += 1
            n += 1
        if n > 1:
            raise RuntimeError(f"image reattach: {n} image-less kickoff messages in one request")
        return n


def tcp_peer_uid(client_port: int, server_port: int) -> int:
    """Which uid owns the client-side socket of the connection 127.0.0.1:client_port → 127.0.0.1:server_port.

    /proc/net/tcp has one socket per line: local_address rem_address (hex IP:PORT) ...; column 8 is the uid.
    On the client's line, local is client_port and remote is server_port. Not found means the peer is unknown -- raise.
    """
    with open("/proc/net/tcp") as f:
        next(f)
        for line in f:
            cols = line.split()
            local, remote = cols[1], cols[2]
            if int(local.split(":")[1], 16) == client_port and int(remote.split(":")[1], 16) == server_port:
                return int(cols[7])
    raise LookupError(f"no /proc/net/tcp entry for 127.0.0.1:{client_port} -> :{server_port}")


class TaskProxy:
    def __init__(self, task: str, budget: int, upstream: tuple[str, int], log_path: Path, server_context: int,
                 images=None, checker=None, salt_field: str | None = None, salt: str | None = None,
                 dp_rank: int | None = None, allowed_uid: int | None = None, wrapup: Wrapup | None = None,
                 routes: frozenset[tuple[str, str]] | None = None, require_session: bool = False,
                 stop_dir: Path | None = None, context_limit: int | None = None, charge=None,
                 pace_path: Path | None = None, reattach: ImageReattach | None = None,
                 stop_after_first_window: bool = False) -> None:
        #: task: the "task" of each jsonl row and the thread name; the harness passes the episode name (the task id in
        #: classic mode).
        self.task, self.budget, self.upstream, self.log_path = task, budget, upstream, log_path
        #: See "charge" in the module docstring. None = no charge (arc2, v5 / v5k g2seed). The wrap-up's soft line
        #: counts output only, so the two are not mixed
        if charge is not None and wrapup is not None:
            raise ValueError("charge does not go with a wrap-up (its soft line counts output tokens only)")
        self.charge = charge
        #: The most recently asked charge: never decreases (the action count never decreases); a decrease is a charge bug
        self.charge_seen = 0
        #: See "pace" in the module docstring. None = not written (arc2, arc3 without pace_line)
        self.pace_path = pace_path
        #: See "routes" in the module docstring. None / False / None = arc2's behavior (forward everything, ignore the
        #: session, no stop files).
        if require_session and routes is None:
            raise ValueError("require_session goes with a route allowlist")
        self.routes, self.require_session, self.stop_dir = routes, require_session, stop_dir
        #: See "window" in the module docstring. A session_id is required to tell the window apart
        if stop_after_first_window and not require_session:
            raise ValueError("stop_after_first_window reads the window from the session_id: it needs require_session")
        self.stop_after_first_window = stop_after_first_window
        #: The first blocked request with window ≥ 1, {"sid", "t"}; once set, this game ended at the first window change
        self.window_cut: dict | None = None
        #: Number of requests blocked by the route allowlist or require_session (each also writes a forbidden row to the jsonl)
        self.forbidden = 0
        if wrapup is not None and wrapup.hard != budget:
            raise ValueError(f"the wrap-up hard cap {wrapup.hard} must be the proxy budget {budget}")
        #: Wrap-up (see Wrapup): budget is its hard; for the response whose usage passes soft, cross() runs before the
        #: end event is forwarded to codex.
        self.wrapup = wrapup
        #: Put the kickoff images back after compaction (ImageReattach); None = don't. Image-only mode (images) has its
        #: own images and is not combined with it
        if reattach is not None and images is not None:
            raise ValueError("reattach is for the text view; the image view brings its own images")
        self.reattach = reattach
        if (salt_field is None) != (salt is None):
            raise ValueError("salt_field and salt go together")
        if salt is not None and not (isinstance(salt, str) and salt and isinstance(salt_field, str) and salt_field):
            raise ValueError(f"salt_field and salt must be non-empty strings: {salt_field!r} {salt!r}")
        self.salt_field, self.salt = salt_field, salt
        #: Things that should never happen (currently only: the request carried the salt field itself). The harness's
        #: run_session kills codex and fails the episode when it sees this.
        self.fatal: str | None = None
        #: The proxy decided this episode should end (not a harness fault; currently only IMAGE_REJECTED): run_session
        #: kills codex at the next poll, and the episode ends with this stop, collecting answers and scoring as usual.
        self.end_reason: str | None = None
        #: Image-only mode (image_check): images is an ImageStore, and [[image:NAME]] in requests is replaced by the
        #: image; checker() returns the check result text for the POST /arc2/check that the workspace's check.py makes.
        #: Both None means normal mode.
        if (images is None) != (checker is None):
            raise ValueError("images and checker go together")
        self.images, self.checker = images, checker
        #: With remaining budget ≥ this value max_output_tokens is not set: sglang checks input + max_output ≤ context,
        #: so setting 1M gets the request rejected outright (v6's first try: every request "exceeds the model's maximum
        #: context length"). Unset, the server uses context − input, which is necessarily below the remaining budget,
        #: so the budget still holds.
        self.server_context = server_context
        #: See "context" in the module docstring. None = not set (arc2, round 1). Must be below the server's context:
        #: equal or above is meaningless
        if context_limit is not None and not (isinstance(context_limit, int) and 0 < context_limit < server_context):
            raise ValueError(f"context_limit {context_limit!r} must be a positive int below the server context {server_context}")
        self.context_limit = context_limit
        #: session_id → that thread's context (usage input + output) at the end of its previous response; the maximum
        #: is kept separately
        self.context_end: dict[str, int] = {}
        self.context_peak = 0
        #: rid of the first response "whose cap was the remaining context and that ended incomplete
        #: (max_output_tokens)": once set, the context counts as used up
        self.context_cut: str | None = None
        self.lock = threading.Lock()
        self.output_tokens = 0
        #: How many consecutive responses ended incomplete with 0 output tokens (the ZERO_OUTPUT_STREAK stop-loss)
        self.zero_output_streak = 0
        self.requests = 0
        self.refused = 0
        #: session_id → the input item count of the request whose response the context truncated (incomplete without
        #: max_output_tokens set). codex treats incomplete as a stream error and resends unchanged 12 times, truncated
        #: at the same point each time (v6b e12f9a14: 13 × 4,977 tokens, 15 minutes). After that, requests on the same
        #: thread whose input is no shorter get a 400 overflow message directly, and the harness opens a new thread.
        self.context_full: dict[str, int] = {}
        self.dp_rank = dp_rank
        self.allowed_uid = allowed_uid
        self.rejected_peers = 0
        #: session_id → "open" / "closed" (closed only once a /close_session passing through here returns 200)
        self.sessions: dict[str, str] = {}
        self.upstreams: set[http.client.HTTPConnection] = set()
        #: request_ids that were sent but whose end event (response.completed / incomplete / failed, or a non-streaming
        #: response fully forwarded) has not been seen yet. When codex is killed the forwarding thread exits first on
        #: BrokenPipe, so "the thread is still alive" cannot be used to tell what is in flight.
        self.unfinished: set[str] = set()
        self.serial = 0
        self.aborted = False

    def cap(self) -> int:
        """Output cap: budget; with a wrap-up it is Wrapup.cap() (after the context line triggers, only wrapup_tokens more)."""
        return self.budget if self.wrapup is None else self.wrapup.cap()

    def charged(self) -> int:
        """Tokens deducted from the budget besides model output (see "charge" in the module docstring; 0 without charge)."""
        if self.charge is None:
            return 0
        # Ask and record under the same lock: when the handler thread and the harness poll ask at the same time, a
        # smaller value asked first cannot be recorded after a larger one (a false "went down")
        with self.lock:
            value = self.charge()
            if not (isinstance(value, int) and not isinstance(value, bool) and value >= 0):
                raise RuntimeError(f"{self.task}: charge() returned {value!r}, want an int >= 0")
            if value < self.charge_seen:
                raise RuntimeError(f"{self.task}: charge() went down from {self.charge_seen} to {value}")
            self.charge_seen = value
        return value

    def remaining(self) -> int:
        cap, charged = self.cap(), self.charged()
        with self.lock:
            if self.zero_output_streak >= ZERO_OUTPUT_STREAK:
                return 0
            return cap - self.output_tokens - charged

    def request_left(self) -> int:
        """How much this request may generate: the remaining budget; with cut, before the wrap-up triggers, up to the soft
        line (see Wrapup.cut)."""
        cap, charged = self.cap(), self.charged()
        with self.lock:
            used = self.output_tokens
        left = cap - used - charged
        w = self.wrapup
        if w is None or not w.cut:
            return left
        with w.lock:
            armed = w.armed_at is not None
        if armed:
            return left
        if used > w.soft - EXHAUSTED_BELOW:
            # When the crossing response ends cross() always triggers first (it runs before the end event is forwarded
            # to codex); only then can there be a next request
            raise RuntimeError(f"{self.task}: output {used} is past the wrap-up line {w.soft} but the wrap-up never armed")
        return w.soft - used

    def exhausted(self) -> bool:
        return self.remaining() < EXHAUSTED_BELOW

    def tracks_context(self) -> bool:
        """Whether each thread's context must be tracked: there is a context_limit, or the wrap-up has a context line."""
        return self.context_limit is not None or (self.wrapup is not None and self.wrapup.context is not None)

    def wrapup_room(self, session: str | None) -> int | None:
        """Before the wrap-up's context line triggers: how much this request on this thread may generate (see
        Wrapup.context_room)."""
        if self.wrapup is None or self.wrapup.context is None:
            return None
        with self.lock:
            end = self.context_end.get(session, 0)
        return self.wrapup.context_room(end)

    def context_left(self, session: str | None) -> int | None:
        """How much this thread has left to the context limit (None if no limit is set; an unseen thread counts from 0)."""
        if self.context_limit is None:
            return None
        with self.lock:
            return self.context_limit - self.context_end.get(session, 0)

    def context_exhausted(self) -> bool:
        """Some thread's context reached the limit (less than EXHAUSTED_BELOW from it, or a response was cut off by the
        remaining context)."""
        if self.context_limit is None:
            return False
        with self.lock:
            return self.context_cut is not None or self.context_peak > self.context_limit - EXHAUSTED_BELOW

    def window_exhausted(self) -> bool:
        """stop_after_first_window: the first request with window ≥ 1 has already been blocked (see "window" in the
        module docstring)."""
        with self.lock:
            return self.window_cut is not None

    def session_window(self, session: str) -> int | None:
        """The n of ``session`` = ``<thread>:<n>``; any other shape means codex's contract changed: record it in ``fatal``
        (the harness's next poll fails this game) and return None; the caller returns 500 and then raises."""
        thread, sep, n = session.rpartition(":")
        if not (sep and thread and n.isdigit()):
            self.fail(f"{self.task}: session_id {session!r} is not <thread>:<window>; cannot tell its window")
            return None
        return int(n)

    def cut_window(self, session: str, window: int) -> bool:
        """With stop_after_first_window, asked once per /responses: whether to block this request. The first one with
        window ≥ 1 is recorded as window_cut; after that, every request (whatever its window) is blocked."""
        if not self.stop_after_first_window:
            raise RuntimeError(f"{self.task}: cut_window without stop_after_first_window")
        with self.lock:
            if self.window_cut is None and window >= 1:
                self.window_cut = {"sid": session, "t": time.time()}
            return self.window_cut is not None

    def cut_context(self, rid: str) -> None:
        """A response whose cap was the remaining context ended incomplete (max_output_tokens): the context is used up
        (see the module docstring). Called before the end event is forwarded to codex, so that codex's resend is blocked
        with 503 instead of becoming the next entry in the ledger."""
        if self.context_limit is None:
            raise RuntimeError(f"{self.task}: context cut {rid} without a context_limit")
        with self.lock:
            if self.context_cut is None:
                self.context_cut = rid

    def stopped(self) -> bool:
        """The stop directory holds any stop file (level_episode.py; names starting with ``.`` are half-written temp files)."""
        return self.stop_dir is not None and any(not p.name.startswith(".") for p in self.stop_dir.iterdir())

    def note_forbidden(self, method: str, path: str, reason: str) -> None:
        with self.lock:
            self.forbidden += 1
            with self.log_path.open("a") as f:
                f.write(json.dumps({"t": time.time(), "task": self.task, "event": "forbidden", "method": method,
                                    "path": path[:300], "reason": reason}) + "\n")

    def end(self, reason: str) -> None:
        with self.lock:
            if self.end_reason is None:
                self.end_reason = reason

    def fail(self, message: str) -> None:
        with self.lock:
            if self.fatal is None:
                self.fatal = message

    def write_pace(self, unaccounted: int) -> None:
        """``pace_path`` (see "pace" in the module docstring): the cumulative output (plus ``unaccounted`` = the output of
        this response, which has ended but not yet been recorded by account()) and the budget. Does nothing without
        pace_path."""
        if self.pace_path is None:
            return
        with self.lock:
            used = self.output_tokens + unaccounted
        ws = self.pace_path.parent
        try:
            if ws.is_symlink() or not ws.is_dir():
                raise NotADirectoryError(f"{ws} is not a plain directory")
            body = json.dumps({"level_output_tokens": used, "level_output_budget": self.budget}).encode()
            tmp = ws / f".{self.pace_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
            try:
                if os.write(fd, body) != len(body):
                    raise OSError(f"short write to {tmp}")
            finally:
                os.close(fd)
            os.replace(tmp, self.pace_path)
        except OSError as exc:
            self.fail(f"cannot write the pace file {self.pace_path} (harness bug): {exc!r}")
            raise

    def account(self, row: dict) -> None:
        with self.lock:
            if (row.get("status") == "incomplete" and row.get("max_output_tokens") is None
                    and (row.get("incomplete") or {}).get("reason") == "max_output_tokens"):
                self.context_full[row["session_id"]] = row["n_input_items"]
            self.output_tokens += int(row.get("output_tokens") or 0)
            self.requests += 1
            row["cum_output_tokens"] = self.output_tokens
            usage = row.get("usage") or {}
            if self.tracks_context() and "input_tokens" in usage and row.get("session_id") is not None:
                end = int(usage["input_tokens"]) + int(usage.get("output_tokens") or 0)
                self.context_end[row["session_id"]] = end
                self.context_peak = max(self.context_peak, end)
                row["context_end"] = end
            with self.log_path.open("a") as f:
                f.write(json.dumps(row) + "\n")

    def next_rid(self) -> str:
        with self.lock:
            self.serial += 1
            return f"resp_arc2_{self.task}_{self.serial:04d}_{uuid.uuid4().hex[:8]}"

    def admit(self, conn: http.client.HTTPConnection, rid: str | None, replaces=None) -> bool:
        """Register an upstream connection (and its rid); once the episode has ended (after abort_upstreams), return
        False without registering."""
        with self.lock:
            if replaces is not None:
                self.upstreams.discard(replaces[0])
                self.unfinished.discard(replaces[1])
            if self.aborted:
                return False
            self.upstreams.add(conn)
            if rid is not None:
                self.unfinished.add(rid)
            return True

    def finished(self, rid: str) -> None:
        with self.lock:
            self.unfinished.discard(rid)

    def release(self, conn: http.client.HTTPConnection) -> None:
        with self.lock:
            self.upstreams.discard(conn)

    def note_session(self, sid: str, state: str) -> None:
        with self.lock:
            self.sessions[sid] = state

    def open_sessions(self) -> list[str]:
        with self.lock:
            return sorted(s for s, state in self.sessions.items() if state == "open")

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        row = {"t": time.time(), "task": self.task, "event": f"harness{path}", **payload}
        request = urllib.request.Request(f"http://{self.upstream[0]}:{self.upstream[1]}{path}",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout) as r:
                r.read()
                row["http"] = r.status
        except OSError as error:
            row["error"] = repr(error)
        with self.lock, self.log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row

    def abort_unfinished(self) -> list[dict]:
        """POST /abort_request for every request whose end event has not been seen (a no-op on the server side for an
        rid that already finished)."""
        with self.lock:
            rids = sorted(self.unfinished)
            self.unfinished.clear()
        return [self._post("/abort_request", {"rid": rid}, 30.0) for rid in rids]

    def abort_upstreams(self, wait_seconds: float = 30.0) -> list[dict]:
        """Episode over: forward no more requests (reply 503), abort unfinished requests, cut connections still open to
        sglang, wait for the forwarding threads to exit. Returns the abort records. Close sessions only after this --
        otherwise a close could land ahead of a request that is still generating."""
        with self.lock:
            self.aborted = True
            conns = list(self.upstreams)
        aborts = self.abort_unfinished()
        for conn in conns:
            sock = conn.sock
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass              # already disconnected: exactly the desired result
        deadline = time.monotonic() + wait_seconds
        while True:
            with self.lock:
                left = len(self.upstreams)
            if not left:
                return aborts
            if time.monotonic() > deadline:
                raise RuntimeError(f"{self.task}: {left} upstream request(s) still open "
                                   f"{wait_seconds:.0f} s after abort")
            time.sleep(0.1)

    def close_open_sessions(self, timeout: float = 30.0) -> list[dict]:
        """Send the /close_session calls codex did not send. One record row per session (also written to this episode's
        proxy jsonl).

        A failed close does not raise: the cost is that session's KV staying protected, not a wrong answer -- the caller
        must print the failure count."""
        rows = []
        for sid in self.open_sessions():
            row = self._post("/close_session", {"session_id": sid}, timeout)
            if row.get("http") == 200:
                self.note_session(sid, "closed")
            rows.append(row)
        return rows


def _handler(proxy: TaskProxy):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: D401 -- quiet; rows go to the jsonl
            pass

        def address_string(self):
            return "unix" if isinstance(self.client_address, (str, bytes)) else str(self.client_address[0])

        def _body(self) -> bytes:
            if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
                out = bytearray()
                while True:
                    size = int(self.rfile.readline().split(b";")[0], 16)
                    if size == 0:
                        self.rfile.readline()
                        return bytes(out)
                    out += self.rfile.read(size)
                    self.rfile.readline()
            return self.rfile.read(int(self.headers.get("Content-Length") or 0))

        def _reply(self, status: int, data: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(data)
            self.close_connection = True

        def _forward(self) -> None:
            if proxy.allowed_uid is not None:
                uid = tcp_peer_uid(self.client_address[1], self.server.server_address[1])
                if uid != proxy.allowed_uid:
                    proxy.rejected_peers += 1
                    print(f"[proxy {proxy.task}] refused a connection from uid {uid} "
                          f"(only uid {proxy.allowed_uid} may use it)", flush=True)
                    self._reply(403, json.dumps({"error": {"message": "arc2 harness: this proxy belongs to another "
                                                                      "episode"}}).encode(), "application/json")
                    return
            body = self._body() if self.command in ("POST", "PUT") else b""
            route = self.path.split("?", 1)[0].rstrip("/") or "/"
            if proxy.routes is not None and (self.command, route) not in proxy.routes:
                proxy.note_forbidden(self.command, self.path, "route")
                self._reply(403, json.dumps({"error": {"message": f"harness: {self.command} {route} is not "
                                                                  "allowed"}}).encode(), "application/json")
                return
            closing = (json.loads(body)["session_id"]
                       if self.command == "POST" and self.path.rstrip("/") == "/close_session" else None)
            if self.path.rstrip("/") == "/arc2/check":
                if proxy.checker is None:
                    self._reply(404, b"no checker in this mode\n", "text/plain; charset=utf-8")
                else:
                    self._reply(200, proxy.checker().encode(), "text/plain; charset=utf-8")
                return
            is_responses = self.command == "POST" and self.path.rstrip("/").endswith("/responses")
            row = None
            capped = False
            req = None
            wrapup_carried = False
            wrapup_injected = False
            post_wrapup = False
            if is_responses:
                req = json.loads(body)
                if proxy.require_session and not (isinstance(req.get("session_id"), str) and req["session_id"]):
                    proxy.note_forbidden(self.command, self.path, "no session_id")
                    self._reply(403, json.dumps({"error": {"message": "harness: /responses without a session_id is "
                                                                      "not allowed"}}).encode(), "application/json")
                    return
                if proxy.stopped():
                    # Level cleared, action cap, budget: the episode is already over, no more generation (run_arm's
                    # over() closes the session right after)
                    proxy.refused += 1
                    self._reply(503, json.dumps({"error": {"message": "harness: this episode is over"}}).encode(),
                                "application/json")
                    return
                if proxy.salt is not None:
                    if proxy.salt_field in req:
                        message = (f"arc2 harness: the request body already has the cache-salt field "
                                   f"{proxy.salt_field!r} = {json.dumps(req[proxy.salt_field])[:120]}; "
                                   f"refusing to overwrite it with {proxy.salt!r}")
                        proxy.fail(message)
                        self._reply(500, json.dumps({"error": {"message": message}}).encode(), "application/json")
                        raise RuntimeError(message)
                    req[proxy.salt_field] = proxy.salt
                left = proxy.remaining()
                if left < EXHAUSTED_BELOW:
                    proxy.refused += 1
                    msg = json.dumps({"error": {"message": "arc2 harness: output-token budget used up"}}).encode()
                    self.send_response(503)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(msg)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(msg)
                    self.close_connection = True
                    return
                session, n_items = req.get("session_id"), len(req.get("input") or [])
                if proxy.stop_after_first_window:
                    window = proxy.session_window(session)
                    if window is None:
                        self._reply(500, json.dumps({"error": {"message": proxy.fatal}}).encode(), "application/json")
                        raise RuntimeError(proxy.fatal)
                    if proxy.cut_window(session, window):
                        # First window change (stop_after_first_window): not forwarded, not counted; the harness
                        # writes the stop file window
                        proxy.refused += 1
                        self._reply(503, json.dumps({"error": {"message": "harness: this episode ends at its first "
                                                                          "window change"}}).encode(), "application/json")
                        return
                room = proxy.context_left(session)
                if proxy.context_exhausted() or (room is not None and room < EXHAUSTED_BELOW):
                    # The context limit (context_limit) is reached: like an exhausted budget, the episode ends (the
                    # harness writes the stop file context)
                    proxy.refused += 1
                    self._reply(503, json.dumps({"error": {"message": "harness: the context quota is used up"}}).encode(),
                                "application/json")
                    return
                full_at = proxy.context_full.get(session)
                if full_at is not None and n_items >= full_at:
                    # The message must contain the run_arc2.CONTEXT_OVERFLOW string: the harness uses it to detect the
                    # overflow and open a new thread.
                    proxy.refused += 1
                    msg = json.dumps({"error": {
                        "message": "arc2 harness: input plus response exceeds the model's maximum context length "
                                   f"(the previous response on this thread was cut off at the context limit with "
                                   f"{full_at} input items)", "type": "BadRequestError", "code": 400}}).encode()
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(msg)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(msg)
                    self.close_connection = True
                    return
                # after the refusals above: a refused request never reaches the model
                # armed before this request is sent: everything it generates is wrap-up (the crossing response's request
                # was sent before cross() armed it)
                post_wrapup = proxy.wrapup is not None and proxy.wrapup.armed()
                wrapup_injected = proxy.wrapup is not None and proxy.wrapup.inject(req)
                wrapup_carried = proxy.wrapup is not None and proxy.wrapup.carries(req)
                reattached = proxy.reattach.apply(req) if proxy.reattach is not None else None
                asked = req.get("max_output_tokens")
                budget_left = proxy.request_left()
                if proxy.charge is not None and budget_left < EXHAUSTED_BELOW:
                    # The charge grew after the remaining() above (an action was recorded in the ledger in between):
                    # same as an exhausted budget. charge is not combined with wrapup, so the two wrap-up lines above
                    # are no-ops for it; this is still a refusal before the request reaches the model
                    proxy.refused += 1
                    self._reply(503, json.dumps({"error": {"message": "arc2 harness: output-token budget used up"}}).encode(),
                                "application/json")
                    return
                left = budget_left if room is None else min(budget_left, room)
                #: The wrap-up's context line (Wrapup.context): before it is crossed the cap is also lowered to the
                #: line, so the crossing response is cut off at the line
                wroom = proxy.wrapup_room(session)
                if wroom is not None and wroom < left:
                    left = wroom
                #: the proxy set (or lowered) the cap itself: only such requests get shrunk and resent when the server
                #: finds the cap too large
                capped = left < proxy.server_context and (asked is None or int(asked) > left)
                if left < proxy.server_context:
                    req["max_output_tokens"] = left if asked is None else min(int(asked), left)
                #: The cap is exactly the remaining context (smaller than the budget, and codex did not ask for less):
                #: being cut off by it = context used up (cut_context)
                context_bound = room is not None and room < budget_left and req.get("max_output_tokens") == room
                if proxy.dp_rank is not None:
                    if req.get("routed_dp_rank") not in (None, proxy.dp_rank):
                        raise RuntimeError(f"client asked for DP {req['routed_dp_rank']}, "
                                           f"the episode owns DP {proxy.dp_rank}")
                    req["routed_dp_rank"] = proxy.dp_rank
                if session is not None:
                    proxy.note_session(session, "open")
                if req.get("request_id") is not None:
                    raise RuntimeError(f"client set request_id {req['request_id']!r}; the proxy owns it")
                req["request_id"] = proxy.next_rid()
                if post_wrapup:
                    proxy.wrapup.note_post(req["request_id"])
                images = 0
                if proxy.images is not None:
                    import image_check
                    images = image_check.inject_images(req, proxy.images)
                raw = json.dumps(req).encode()
                sentinels = raw.count(IMAGE_SENTINEL) if b'"input_image"' in raw else 0
                body = _encode(req)
                row = {"images": images, "image_sentinels": sentinels, "t": time.time(), "task": proxy.task, "max_output_tokens": req.get("max_output_tokens"),
                       "session_id": session, "n_input_items": n_items, "dp_rank": req.get("routed_dp_rank"),
                       "request_id": req["request_id"],
                       **({"images_reattached": reattached} if reattached is not None else {}),
                       **({"context_bound": True} if context_bound else {}),
                       **({"action_charge": proxy.charge_seen} if proxy.charge is not None else {}),
                       **({"cache_salt": proxy.salt} if proxy.salt is not None else {})}
                # Before sending upstream (pace, see the module docstring): refused requests never get here
                proxy.write_pace(0)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in HOP}
            headers["Content-Length"] = str(len(body))
            conn = http.client.HTTPConnection(*proxy.upstream, timeout=None)
            if not proxy.admit(conn, None if row is None else row["request_id"]):
                self._reply(503, json.dumps({"error": {"message": "arc2 harness: this episode has ended"}}).encode(),
                            "application/json")
                return
            began = time.monotonic()
            resp = first = None
            #: codex was already gone when writing back to it (killed mid-stream). Only the client-write sites count --
            #: an upstream ConnectionReset (sglang down) is not a disconnect.
            gone = []

            @contextlib.contextmanager
            def to_client():
                try:
                    yield
                except (BrokenPipeError, ConnectionResetError):
                    gone.append(True)
                    raise

            # The accounting must always happen: when the client disconnects midway (BrokenPipe / ConnectionReset) the
            # request still goes into the jsonl and the counts, marked client_disconnected; other exceptions are
            # recorded as error. The exception itself is still raised (_Quiet prints one line).
            error = None
            try:
                conn.request(self.command, self.path, body=body if body else None, headers=headers)
                resp = conn.getresponse()
                if closing is not None and resp.status == 200:
                    proxy.note_session(closing, "closed")
                stream = row is not None and "text/event-stream" in (resp.getheader("Content-Type") or "")
                held: list[bytes] = []
                if stream and capped:
                    held = _head_events(resp)
                    n_input = _overflow_input(held)
                    if n_input is not None:
                        # The cap was set by the proxy and the server finds it too large: shrink it to fit and send once
                        # more (still ≤ remaining budget, so the accounting stays exact)
                        cap = proxy.server_context - n_input - CAP_MARGIN
                        row["cap_retry"] = {"from": req["max_output_tokens"], "to": cap, "input_tokens": n_input}
                        conn.close()
                        if cap < EXHAUSTED_BELOW:
                            # The input alone nearly fills the context: reply with the overflow message the harness
                            # recognizes (CONTEXT_OVERFLOW); it opens a new thread
                            proxy.refused += 1
                            row["status"] = "refused: context full"
                            self._reply(400, json.dumps({"error": {
                                "message": "arc2 harness: input plus response exceeds the model's maximum context "
                                           f"length ({n_input} input tokens)", "type": "BadRequestError",
                                "code": 400}}).encode(), "application/json")
                            return
                        req["max_output_tokens"] = cap
                        row["max_output_tokens"] = cap
                        if proxy.context_limit is not None:
                            row["context_bound"] = True     # the server's context is the real cap: being cut off by it also counts as used up
                        # New rid: the first one already ended as failed on the server; not reused
                        old = (conn, row["request_id"])
                        req["request_id"] = row["request_id"] = proxy.next_rid()
                        if post_wrapup:
                            proxy.wrapup.note_post(req["request_id"])
                        body = _encode(req)
                        headers["Content-Length"] = str(len(body))
                        conn = http.client.HTTPConnection(*proxy.upstream, timeout=None)
                        if not proxy.admit(conn, row["request_id"], replaces=old):
                            row["status"] = "refused: episode ended"
                            self._reply(503, json.dumps({"error": {"message": "arc2 harness: this episode has "
                                                                              "ended"}}).encode(), "application/json")
                            return
                        conn.request(self.command, self.path, body=body, headers=headers)
                        resp = conn.getresponse()
                        held = _head_events(resp)
                with to_client():
                    self.send_response(resp.status, resp.reason)
                    for k, v in resp.getheaders():
                        if k.lower() not in HOP:
                            self.send_header(k, v)
                    self.send_header("Connection", "close")
                    self.end_headers()
                self.close_connection = True
                if wrapup_injected:
                    row["wrapup_injected"] = True
                if stream and wrapup_carried and resp.status == 200 and proxy.wrapup.mark_seen(proxy.output_tokens):
                    row["wrapup_seen"] = True
                if stream:
                    pending = iter(held)
                    while True:
                        line = next(pending, None)
                        if line is None:
                            line = resp.readline()
                        if not line:
                            break
                        if first is None:
                            first = time.monotonic() - began
                        # Handle end events before forwarding them to codex: the wrap-up message (Wrapup.cross) must be
                        # written before codex's next follow-up
                        if line.startswith(b"data: ") and b'"response.' in line:
                            event = json.loads(line[6:])
                            if event.get("type") in ("response.completed", "response.incomplete", "response.failed"):
                                proxy.finished(row["request_id"])
                                if event["type"] == "response.failed" and IMAGE_REJECTED in line:
                                    row["image_rejected"] = True
                                    proxy.end("image_rejected")
                                r = event.get("response") or {}
                                usage = r.get("usage") or {}
                                row.update(status=r.get("status") or event["type"], usage=usage,
                                           output_tokens=usage.get("output_tokens") or 0,
                                           incomplete=(r.get("incomplete_details") or None))
                                truncated = (row["status"] == "incomplete"
                                             and (row["incomplete"] or {}).get("reason") == "max_output_tokens")
                                with proxy.lock:
                                    if row["status"] == "incomplete" and not row["output_tokens"]:
                                        proxy.zero_output_streak += 1
                                        if proxy.zero_output_streak >= ZERO_OUTPUT_STREAK:
                                            row["zero_output_stall"] = True
                                    else:
                                        proxy.zero_output_streak = 0
                                context_now = (int(usage["input_tokens"]) + int(row["output_tokens"])
                                               if "input_tokens" in usage else None)
                                if context_now is not None and session is not None and proxy.tracks_context():
                                    # Record it before forwarding to codex: account() runs only after the stream is
                                    # fully read, and codex's next request may arrive first
                                    with proxy.lock:
                                        proxy.context_end[session] = context_now
                                if proxy.wrapup is not None and proxy.wrapup.cross(
                                        proxy.output_tokens + row["output_tokens"],
                                        cut_rid=row["request_id"] if truncated else None, context=context_now):
                                    row["wrapup_armed"] = True
                                if truncated and row.get("context_bound"):
                                    proxy.cut_context(row["request_id"])
                                    row["context_cut"] = True
                                # Before the end event is forwarded to codex (pace): account() has not recorded this
                                # response yet, so add its output
                                proxy.write_pace(int(row["output_tokens"]))
                        with to_client():
                            self.wfile.write(line)
                            self.wfile.flush()
                else:
                    while True:
                        chunk = resp.read1(65536)
                        if not chunk:
                            break
                        with to_client():
                            self.wfile.write(chunk)
                    if row is not None:
                        row["status"] = f"http {resp.status} (not a stream)"
                        proxy.finished(row["request_id"])
            except BaseException as exc:
                if not gone:
                    error = repr(exc)
                raise
            finally:
                conn.close()
                proxy.release(conn)
                if row is not None:
                    row.update(ttft=None if first is None else round(first, 3),
                               seconds=round(time.monotonic() - began, 3),
                               http=None if resp is None else resp.status, client_disconnected=bool(gone),
                               **({"error": error} if error is not None else {}))
                    proxy.account(row)

        do_GET = do_POST = do_PUT = do_DELETE = _forward

    return Handler


def _encode(req: dict) -> bytes:
    body = json.dumps(req).encode()
    if b'"input_image"' in body:
        body = body.replace(IMAGE_SENTINEL, IMAGE_SENTINEL_SAFE)
    return body


def _head_events(resp) -> list[bytes]:
    """Read the SSE opening lines up to and including the first event that is not response.created / in_progress, or
    until the stream ends. These lines are later forwarded to codex as is."""
    held = []
    while True:
        line = resp.readline()
        held.append(line)
        if not line:
            return held
        if line.startswith(b"data: ") and b'"type":' in line:
            kind = json.loads(line[6:]).get("type")
            if kind not in ("response.created", "response.in_progress"):
                return held


def _overflow_input(held: list[bytes]) -> int | None:
    """If the opening is a response.failed for "input + cap exceeds the context", return the input token count the
    server reported, else None."""
    last = next((l for l in reversed(held) if l.startswith(b"data: ")), None)
    if last is None or b'"response.failed"' not in last:
        return None
    m = OVERFLOW_RE.search(last)
    return None if m is None else int(m.group(1))


class _Quiet:
    """When codex is killed mid-stream, the write back at that moment raises BrokenPipe -- expected; print one line, no
    traceback. When the harness cuts the upstream itself (abort_upstreams), reading that end gives OSError /
    IncompleteRead, equally expected."""

    def handle_error(self, request, client_address):
        import sys
        error = sys.exc_info()[1]
        if isinstance(error, (BrokenPipeError, ConnectionResetError)):
            print(f"[proxy] client went away mid-response ({type(error).__name__})", flush=True)
            return
        if isinstance(error, (OSError, http.client.HTTPException)) and getattr(self, "proxy_aborted", lambda: False)():
            print(f"[proxy] upstream cut by the harness ({type(error).__name__})", flush=True)
            return
        super().handle_error(request, client_address)


class _UnixServer(_Quiet, socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


class _TcpServer(_Quiet, socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve_unix(path: Path, proxy: TaskProxy, mode: int = 0o600) -> socketserver.BaseServer:
    if path.exists():
        path.unlink()
    server = _UnixServer(str(path), _handler(proxy))
    server.proxy_aborted = lambda: proxy.aborted
    os.chmod(path, mode)
    threading.Thread(target=server.serve_forever, daemon=True, name=f"proxy-{proxy.task}").start()
    return server


def serve_tcp(proxy: TaskProxy) -> tuple[socketserver.BaseServer, int]:
    """A random port on 127.0.0.1. For fence none (local testing), or the posix fence without netns (Kaggle) -- the
    latter must set ``proxy.allowed_uid``: every uid on the same host can connect to this port."""
    server = _TcpServer(("127.0.0.1", 0), _handler(proxy))
    server.proxy_aborted = lambda: proxy.aborted
    threading.Thread(target=server.serve_forever, daemon=True, name=f"proxy-{proxy.task}").start()
    return server, server.server_address[1]


def unix_path_ok(path: Path) -> None:
    if len(os.fsencode(path)) >= 108:
        raise ValueError(f"unix socket path too long for sun_path: {path}")

