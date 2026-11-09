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

All other requests (/v1/models, /health, /close_session) are forwarded as is. Responses always end with
Connection: close (no keep-alive); bodies are streamed as is.
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
#: A response cut off at the context edge (input + output within this many of the server context) also counts as
#: context full, even when the cap was set by the proxy: 10-24 pod 88bcf3b4__s0 at input 149,546, the proxy set the cap
#: to 3,986 from "remaining context", every response came back incomplete, and codex resent it unchanged 13 times at
#: ~290 s each (one slot idling for an hour).
CONTEXT_FULL_SLACK = 256
#: The server rejects requests whose "input alone exceeds the context" (returns response.failed): resending on the same
#: thread only fails again (10-24 pod: 4 episodes, 13 times each). Later requests on the same thread get a 400 overflow
#: directly, and the harness opens a new thread.
INPUT_TOO_LONG = "longer than the model's context length"

IMAGE_SENTINEL = b"<image>"
IMAGE_SENTINEL_SAFE = b"<image\\u200b>"

#: The file of codex patch 0005 ($CODEX_HOME/compact_reminder): the first line is the threshold (context token count),
#: the rest is the user message to insert into the in-progress turn. codex rereads it at every follow-up (after a tool
#: result comes back, before sending the next request); an empty message does nothing.
REMINDER_FILE = "compact_reminder"
REMINDER_DORMANT = "0\n"


def user_texts(items) -> list[str]:
    """The input_text of every user message in a Responses input."""
    out = []
    for item in items or []:
        if isinstance(item, dict) and item.get("role") == "user" and isinstance(item.get("content"), list):
            out += [c.get("text") or "" for c in item["content"] if isinstance(c, dict) and c.get("type") == "input_text"]
    return out


class Extension:
    """Extension (config ``budget_extension``): the first time output usage passes Wrapup's soft, ask ``gate(total)``
    (returns a dict that always has bool ``granted``). If granted, no wrap-up: soft and hard both move back by
    ``extra``, the proxy's cap rises with them, and wrap-up comes at the new soft; if refused, wrap up as usual.

    **No message is inserted** (the model sees the new cap only in budget.txt): patch 0005 inserts only once per context
    window -- after inserting, it reopens only once compaction has made the context smaller -- so a message inserted on
    extension would keep the wrap-up message at the new soft from being inserted (10-25 real codex test: the wrap-up
    message was never delivered, the episode hit hard and was killed).

    ``repeat`` (config budget_extension's ``idle``, 10-30 improvement plan §2.4): after a grant, ask again at the new
    soft, until the gate refuses (count limit and idleness are both for the gate to judge); the later answers are
    recorded in ``later``."""

    def __init__(self, extra: int, gate, repeat: bool = False) -> None:
        if not (isinstance(extra, int) and extra > 0):
            raise ValueError(f"extra must be a positive int, not {extra!r}")
        self.extra, self.gate, self.repeat = extra, gate, repeat
        self.decision: dict | None = None     # the first gate answer, plus at (the cumulative output when asked)
        self.later: list[dict] = []           # repeat: each later answer
        self.open = True                      # will ask again (closed after one refusal, or after one ask without repeat)

    def ask(self, total: int) -> bool:
        decision = dict(self.gate(total))
        if not isinstance(decision.get("granted"), bool):
            raise ValueError(f"the extension gate must answer granted true/false, not {decision!r}")
        decision["at"] = total
        if self.decision is None:
            self.decision = decision
        else:
            self.later.append(decision)
        self.open = decision["granted"] and self.repeat
        return decision["granted"]

    def record(self) -> dict:
        return {"extra": self.extra, "decision": self.decision, **({"later": self.later} if self.later else {})}


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
    same thread once with ``text``."""

    def __init__(self, path: Path, soft: int, hard: int, template: str, extension: Extension | None = None,
                 time_template: str | None = None) -> None:
        if not 0 < soft < hard:
            raise ValueError(f"need 0 < soft < hard, not {soft} / {hard}")
        self.path, self.soft, self.hard, self.template = path, soft, hard, template
        #: config wrapup_time_seconds (10-30 improvement plan §2.5a): when the wall clock is nearly up, arm_by_time uses
        #: this template ({minutes} = "3 minutes", {left_k})
        self.time_template = time_template
        self.armed_by: str | None = None      # "tokens" / "time"
        #: soft / hard move back once an extension is granted; base_soft is the original budget (for the record)
        self.base_soft = soft
        self.extension = extension
        #: the "raise the proxy cap" hook TaskProxy attaches (called when an extension is granted; argument = new hard)
        self.raise_cap = None
        self.lock = threading.Lock()
        self.armed_at: int | None = None      # cumulative output when the line was crossed
        self.seen_at: int | None = None       # cumulative output when the model first saw the wrap-up message
        self.text: str | None = None
        self.resumed = False                  # solve() has resumed with text

    def _write(self, text: str) -> None:
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(text)
        os.replace(tmp, self.path)

    def dormant(self) -> None:
        self._write(REMINDER_DORMANT)

    def cross(self, total: int) -> bool:
        """A response ended and the cumulative output is now total. The first time it is ≥ soft, write the wrap-up
        message and return True."""
        with self.lock:
            if self.armed_at is not None or total < self.soft:
                return False
            ext = self.extension
            while ext is not None and ext.open and total >= self.soft:
                if not ext.ask(total):
                    break
                if self.raise_cap is None:
                    raise RuntimeError("an extension needs the proxy's raise_cap (TaskProxy binds it)")
                self.soft += ext.extra
                self.hard += ext.extra
                self.raise_cap(self.hard)
                # a single response crossed the extended soft (extra smaller than one response): ask again with repeat,
                # otherwise wrap up right away
            if total < self.soft:
                return False
            self.text = self.template.format(left_k=max(0, self.hard - total) // 1000).strip()
            if not self.text:
                raise ValueError("the wrap-up message is empty")
            self._write(f"0\n{self.text}\n")
            self.armed_at = total
            self.armed_by = "tokens"
            return True

    def arm_by_time(self, total: int, seconds_left: float) -> bool:
        """Wall clock (episode limit or deadline) has less than the config's wrapup_time_seconds left: if not wrapping up
        yet, write the wrap-up message from the time template, along the same path as crossing the line (codex inserts it
        into the turn at its next follow-up; if the turn ends first, solve() resumes once). The token cap is unchanged."""
        if self.time_template is None:
            raise RuntimeError("arm_by_time needs the time template (config wrapup_time_seconds)")
        with self.lock:
            if self.armed_at is not None:
                return False
            m = max(1, int(seconds_left // 60))
            self.text = self.time_template.format(minutes=f"{m} minute{'' if m == 1 else 's'}",
                                                  left_k=max(0, self.hard - total) // 1000).strip()
            if not self.text:
                raise ValueError("the time wrap-up message is empty")
            self._write(f"0\n{self.text}\n")
            self.armed_at = total
            self.armed_by = "time"
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

    def record(self, total: int) -> dict:
        with self.lock:
            return {"soft": self.soft, "hard": self.hard, "armed_at": self.armed_at, "seen_at": self.seen_at,
                    "resumed": self.resumed, "after_soft": max(0, total - self.soft),
                    **({"base_soft": self.base_soft, "extension": self.extension.record()}
                       if self.extension is not None else {}),
                    **({"armed_by": "time"} if self.armed_by == "time" else {})}


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
                 reattach: ImageReattach | None = None) -> None:
        #: task: the "task" of each jsonl row and the thread name; the harness passes the episode name (the task id in
        #: classic mode).
        self.task, self.budget, self.upstream, self.log_path = task, budget, upstream, log_path
        if wrapup is not None and wrapup.hard != budget:
            raise ValueError(f"the wrap-up hard cap {wrapup.hard} must be the proxy budget {budget}")
        #: Wrap-up (see Wrapup): budget is its hard; for the response whose usage passes soft, cross() runs before the
        #: end event is forwarded to codex. When an extension is granted, Wrapup raises budget to the new hard via
        #: raise_cap.
        self.wrapup = wrapup
        if wrapup is not None:
            wrapup.raise_cap = self.raise_budget
        if (salt_field is None) != (salt is None):
            raise ValueError("salt_field and salt go together")
        if salt is not None and not (isinstance(salt, str) and salt and isinstance(salt_field, str) and salt_field):
            raise ValueError(f"salt_field and salt must be non-empty strings: {salt_field!r} {salt!r}")
        self.salt_field, self.salt = salt_field, salt
        #: Things that should never happen (currently only: the request carried the salt field itself). The harness's
        #: run_session kills codex and fails the episode when it sees this.
        self.fatal: str | None = None
        #: Image-only mode (image_check): images is an ImageStore, and [[image:NAME]] in requests is replaced by the
        #: image; checker() returns the check result text for the POST /arc2/check that the workspace's check.py makes.
        #: Both None means normal mode.
        if (images is None) != (checker is None):
            raise ValueError("images and checker go together")
        self.images, self.checker = images, checker
        #: Put the kickoff images back after compaction (ImageReattach); None = don't. Image-only mode (images) has its
        #: own images and is not combined with it
        if reattach is not None and images is not None:
            raise ValueError("reattach is for the text view; the image view brings its own images")
        self.reattach = reattach
        #: With remaining budget ≥ this value max_output_tokens is not set: sglang checks input + max_output ≤ context,
        #: so setting 1M gets the request rejected outright (v6's first try: every request "exceeds the model's maximum
        #: context length"). Unset, the server uses context − input, which is necessarily below the remaining budget,
        #: so the budget still holds.
        self.server_context = server_context
        self.lock = threading.Lock()
        self.output_tokens = 0
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

    def raise_budget(self, hard: int) -> None:
        with self.lock:
            if hard <= self.budget:
                raise ValueError(f"raise_budget {hard} is not above the budget {self.budget}")
            self.budget = hard

    def remaining(self) -> int:
        with self.lock:
            return self.budget - self.output_tokens

    def exhausted(self) -> bool:
        return self.remaining() < EXHAUSTED_BELOW

    def fail(self, message: str) -> None:
        with self.lock:
            if self.fatal is None:
                self.fatal = message

    def account(self, row: dict) -> None:
        with self.lock:
            if row.get("status") == "incomplete" and (row.get("incomplete") or {}).get("reason") == "max_output_tokens":
                usage = row.get("usage") or {}
                edge = (usage.get("input_tokens") or 0) + (usage.get("output_tokens") or 0)
                if row.get("max_output_tokens") is None or edge >= self.server_context - CONTEXT_FULL_SLACK:
                    self.context_full[row["session_id"]] = row["n_input_items"]
            if row.get("status") == "failed" and INPUT_TOO_LONG in json.dumps(row.get("failure") or {}):
                self.context_full[row["session_id"]] = row["n_input_items"]
            self.output_tokens += int(row.get("output_tokens") or 0)
            self.requests += 1
            row["cum_output_tokens"] = self.output_tokens
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
            if is_responses:
                req = json.loads(body)
                wrapup_carried = proxy.wrapup is not None and proxy.wrapup.carries(req)
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
                reattached = proxy.reattach.apply(req) if proxy.reattach is not None else None
                asked = req.get("max_output_tokens")
                #: the proxy set (or lowered) the cap itself: only such requests get shrunk and resent when the server
                #: finds the cap too large
                capped = left < proxy.server_context and (asked is None or int(asked) > left)
                if left < proxy.server_context:
                    req["max_output_tokens"] = left if asked is None else min(int(asked), left)
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
                       **({"cache_salt": proxy.salt} if proxy.salt is not None else {})}
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
                        # New rid: the first one already ended as failed on the server; not reused
                        old = (conn, row["request_id"])
                        req["request_id"] = row["request_id"] = proxy.next_rid()
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
                                r = event.get("response") or {}
                                usage = r.get("usage") or {}
                                row.update(status=r.get("status") or event["type"], usage=usage,
                                           output_tokens=usage.get("output_tokens") or 0,
                                           incomplete=(r.get("incomplete_details") or None))
                                if row["status"] == "failed":
                                    row["failure"] = r.get("error") or event.get("error")
                                if proxy.wrapup is not None and proxy.wrapup.cross(proxy.output_tokens
                                                                                   + row["output_tokens"]):
                                    row["wrapup_armed"] = True
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

