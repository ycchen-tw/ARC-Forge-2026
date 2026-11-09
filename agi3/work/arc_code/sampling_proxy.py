#!/usr/bin/env python3
"""Write the sampling parameters **explicitly** into every model request, and record exactly what was written.

Why this exists: only on 2026-09-19 did we find that codex never sends `temperature` / `top_p` /
`top_k` (measured: the body it sends has only input / instructions / tools / reasoning / …),
so the sampling parameters were always decided by sglang's **class defaults**, which differ from what anyone assumed:

  - `ResponsesRequest._DEFAULT_SAMPLING_PARAMS` is **temperature 0.7** / top_p 1.0
    (it is `ChatCompletionRequest` that has 1.0; the two APIs have different defaults)
  - `--sampling-defaults openai` means the model's shipped generation_config (1.0 / 0.95 / top_k 20)
    is not applied at all
  - `--preferred-sampling-params` cannot fix it: tokenizer_manager does
    ``{**preferred, **request}``, and the Responses path **always** fills in temperature and top_p,
    so the request side always wins

In other words, every earlier codex-arm run used temp 0.7 / top_p 1.0 / top_k off, and no artifact
recorded it. This proxy turns "which sampling parameters this run uses" into something **the arm declares
itself and that can be audited**: values given explicitly on the command line, written into the manifest, actually
overridden on every request, and counted for you.

    python work/arc_code/sampling_proxy.py --listen 31862 --upstream http://127.0.0.1:30862 \
        --set temperature=1.0 --set top_p=0.95 --set top_k=20 --manifest sampling.json

Then point the arm's `--model-base-url` at `http://127.0.0.1:31862/v1`.

Contract:
  - On the matched paths (default `/v1/responses` and `/v1/chat/completions`) the body must be a
    JSON object; otherwise answer 400 and say so loudly in the log. **Do not** let it through silently -- that would
    be back to "not knowing what we are using".
  - `--set` values **override** what the client sends (that is the point: the arm decides), and each override
    records the original value.
  - Other paths, other methods: forwarded byte for byte.
  - Responses are always streamed through (SSE must not be buffered), with **no socket timeout** -- the lesson of
    bridge.py's 30-second timeout: a timeout left on the socket silently cuts long generations.
  - ``--image-ref-key`` (act_image, ``act_images.py``): ``./act do`` markers in ``/v1/responses`` tool outputs are
    replaced by images; needs ``PYTHONPATH=src`` (signature checks and drawing live in arcforge).
  - ``--act-nudge-every N`` (``act_nudge.py``): after the N-th, 2N-th, … tool call since the last ``./act do``, append
    a "time to act" line to its output.
  - ``--effort-schedule LOW,HIGH,TOKENS`` (``effort_schedule.py``): while the output tokens spent on the current level
    are below TOKENS, send reasoning effort LOW, otherwise HIGH; reset on a level clear (``--effort-run-dir`` reads
    each game's record/).
  - ``--window-at HIGH --window-keep LOW`` (``context_window.py``): once the context passes HIGH, stop sending the
    oldest history wholesale and replace it with one message carrying a notes.md snapshot, keeping only about LOW
    tokens; the usage codex sees never reaches its own compaction threshold (``--window-run-dir`` reads each game's
    notes.md).
  - ``--level-watch FIRST,FLOOR,F1,F2`` (``level_watch.py``): when the output tokens spent on a level pass the
    threshold (FIRST for the first level, then max(F × median of earlier levels, FLOOR)), append "first write down
    what is certain and the untested hypotheses" after the tool output, at most twice per level.
  - ``--review-run-dir <run-dir>`` (``review_notice.py``): after a read-only reviewer's report is delivered
    (``reviews/notices/``), append "go read review/Rnn.md" after that game's next tool output; the reviewer's own
    requests never get it.
  - ``--pace-run-dir <run-dir> --pace-game-tokens N`` (``game_pace.py``): the whole-game pace line. Before each request
    goes upstream and before its terminal event is forwarded to codex, write ``.pace.json`` in that game's workspace
    (output spent on this level, this level's allowance), for the act enabled by run_arm's ``pace_line`` to print;
    harness errors such as an unrecognized game return 500.
"""

from __future__ import annotations

import argparse
import http.client
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from compaction_tail import CompactionTail
from reasoning_trim import ReasoningTrim, UsageScanner

DEFAULT_PATHS = ("/v1/responses", "/v1/chat/completions")
#: how long a request waits at most for upstream to start responding. Generation itself is not limited by this
#: (streaming starts with the response).
UPSTREAM_TIMEOUT = 600.0


def parse_set(text: str) -> tuple[str, object]:
    """``top_p=0.95`` -> ``("top_p", 0.95)``. Types are parsed as JSON, falling back to a string."""
    if "=" not in text:
        raise ValueError(f"--set must be key=value, got {text!r}")
    key, _, raw = text.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"--set has an empty key: {text!r}")
    try:
        return key, json.loads(raw)
    except json.JSONDecodeError:
        return key, raw


class Stats:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.seen = 0
        self.rewritten = 0
        self.passthrough = 0
        self.bad_body = 0
        self.overridden: dict[str, int] = {}
        self.first_original: dict[str, object] = {}

    def record(self, original: dict, applied: dict) -> None:
        with self.lock:
            self.rewritten += 1
            for key in applied:
                if key in original and original[key] != applied[key]:
                    self.overridden[key] = self.overridden.get(key, 0) + 1
                    self.first_original.setdefault(key, original[key])
                elif key not in original:
                    self.overridden.setdefault(f"{key} (not sent)", 0)
                    self.overridden[f"{key} (not sent)"] += 1

    def snapshot(self) -> dict:
        with self.lock:
            return {"seen": self.seen, "rewritten": self.rewritten,
                    "passthrough": self.passthrough, "bad_body": self.bad_body,
                    "overridden": dict(self.overridden),
                    "first_original_value": dict(self.first_original)}


def make_handler(upstream_host: str, upstream_port: int, sampling: dict,
                 paths: tuple[str, ...], stats: Stats, trim: ReasoningTrim | None = None,
                 usage_log: Path | None = None, tail: CompactionTail | None = None, images=None,
                 nudge=None, effort=None, window=None, level_watch=None, review=None, pace=None):
    usage_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "arcforge-sampling-proxy"

        def log_message(self, fmt, *args):  # noqa: D102 - by default every request is printed to stderr
            pass

        def _fail(self, code: int, message: str) -> None:
            body = json.dumps({"error": {"message": message,
                                         "type": "sampling_proxy"}}).encode()
            self.send_response(code)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.send_header("connection", "close")
            self.end_headers()
            self.wfile.write(body)
            print(f"[sampling_proxy] {code} {message}", file=sys.stderr, flush=True)

        def _body(self) -> bytes:
            if self.headers.get("transfer-encoding", "").lower() == "chunked":
                # Kept verbatim: this message goes back to the client (codex) as the HTTP 400 error body. Gist: "this
                # proxy does not support a chunked request body"
                raise ValueError("這支 proxy 不支援 chunked 請求 body")
            length = int(self.headers.get("content-length") or 0)
            return self.rfile.read(length) if length else b""

        def _relay(self, method: str) -> None:
            with stats.lock:
                stats.seen += 1
            try:
                body = self._body()
            except ValueError as error:
                self._fail(400, str(error))
                return

            matched = method == "POST" and self.path.rstrip("/") in paths
            session = None
            n_images = n_nudges = n_dropped = n_level = n_review = None
            effort_game = effort_level = watch_game = pace_game = None
            if matched and (sampling or trim or tail or images or nudge or effort or window or level_watch or review
                            or pace):
                try:
                    payload = json.loads(body)
                except json.JSONDecodeError as error:
                    with stats.lock:
                        stats.bad_body += 1
                    # Kept verbatim (HTTP 400 error body returned to codex). Gist: "<path>'s body is not JSON: <error>"
                    self._fail(400, f"{self.path} 的 body 不是 JSON：{error}")
                    return
                if not isinstance(payload, dict):
                    with stats.lock:
                        stats.bad_body += 1
                    # Kept verbatim (HTTP 400 error body returned to codex). Gist: "<path>'s body is not a JSON object"
                    self._fail(400, f"{self.path} 的 body 不是 JSON 物件")
                    return
                original = dict(payload)
                payload.update(sampling)
                stats.record(original, sampling)
                session = payload.get("session_id")
                if self.path.rstrip("/") == "/v1/responses":
                    # tail first, then trim: the trim anchor looks at the input actually sent
                    if tail is not None:
                        tail.apply(payload)
                    if trim is not None:
                        trim.apply(payload)
                    # the cut point is keyed by call_id, unaffected by trim removing reasoning; nudge / images only
                    # see what is left
                    if window is not None:
                        n_dropped = window.apply(payload)
                    if nudge is not None:
                        n_nudges = nudge.apply(payload)
                    if level_watch is not None:
                        watch_game, n_level = level_watch.apply(payload)
                    # as with level_watch: appended after tool outputs and re-applied on the same call_ids every time;
                    # before images swaps in pictures, in a fixed order
                    if review is not None:
                        _, n_review = review.apply(payload)
                    if effort is not None:
                        effort_game, effort_level = effort.apply(payload)
                    if images is not None:
                        n_images = images.apply(payload)
                    # write this game's .pace.json before sending upstream (game_pace); payload unchanged
                    if pace is not None:
                        try:
                            pace_game = pace.apply(payload)
                        except (OSError, RuntimeError, KeyError, ValueError) as error:
                            # Kept verbatim (HTTP 500 error body returned to codex). Gist: "game_pace (harness bug): <error>"
                            self._fail(500, f"game_pace（harness bug）：{error!r}")
                            return
                body = json.dumps(payload).encode()
            else:
                with stats.lock:
                    stats.passthrough += 1

            headers = {k: v for k, v in self.headers.items()
                       if k.lower() not in ("content-length", "connection",
                                            "transfer-encoding", "host")}
            headers["content-length"] = str(len(body))
            headers["host"] = f"{upstream_host}:{upstream_port}"

            conn = http.client.HTTPConnection(upstream_host, upstream_port,
                                              timeout=UPSTREAM_TIMEOUT)
            try:
                conn.request(method, self.path, body=body, headers=headers)
                # no timeout once the response has started: long generations can go silent for minutes between
                # chunks, and a leftover timeout would repeat bridge.py's 30-second defect.
                conn.sock.settimeout(None)
                response = conn.getresponse()
                self.send_response(response.status)
                for key, value in response.getheaders():
                    if key.lower() in ("content-length", "transfer-encoding", "connection"):
                        continue
                    self.send_header(key, value)
                self.send_header("transfer-encoding", "chunked")
                self.end_headers()
                scanner = (UsageScanner() if matched and (trim or usage_log or effort or window or level_watch or pace_game)
                           else None)
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    if scanner is not None and scanner.usage is None:
                        scanner.feed(chunk)
                        if scanner.usage is not None:
                            # record **before** forwarding the completed event: codex sends the next request as soon as
                            # it arrives, and recording late would make the next request's trim decision use the old context
                            if trim is not None:
                                trim.observe(session, scanner.usage)
                            if effort is not None:
                                effort.observe(effort_game, scanner.usage)
                            if window is not None:
                                window.observe(session, scanner.usage)
                            if level_watch is not None:
                                level_watch.observe(watch_game, scanner.usage)
                            if pace_game is not None:
                                pace.observe(pace_game, scanner.usage)
                            if usage_log is not None:
                                row = {"t": round(time.time(), 3), "session": session, "usage": scanner.usage}
                                if scanner.status != "response.completed":
                                    row["status"] = scanner.status      # a response that hit the output cap is response.incomplete
                                if n_images is not None:
                                    row["act_images"] = n_images
                                if n_nudges is not None:
                                    row["act_nudges"] = n_nudges
                                if effort is not None:
                                    row["game"], row["effort"] = effort_game, effort_level
                                if n_dropped is not None:
                                    row["window_dropped"] = n_dropped
                                if n_level is not None:
                                    row["level_notes"] = n_level
                                if n_review:
                                    row["review_notices"] = n_review
                                with usage_lock, usage_log.open("a") as fh:
                                    fh.write(json.dumps(row) + "\n")
                    self.wfile.write(b"%x\r\n" % len(chunk) + chunk + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
            except (OSError, http.client.HTTPException) as error:
                print(f"[sampling_proxy] upstream failed {method} {self.path}: {error}",
                      file=sys.stderr, flush=True)
                try:
                    self._fail(502, f"upstream: {error}")
                except OSError:
                    pass
            finally:
                conn.close()

        def do_POST(self):  # noqa: N802
            self._relay("POST")

        def do_GET(self):  # noqa: N802
            self._relay("GET")

        def do_DELETE(self):  # noqa: N802
            self._relay("DELETE")

    return Handler


def serve(listen_port: int, upstream: str, sampling: dict, paths: tuple[str, ...],
          manifest: Path | None, stats: Stats | None = None, trim: ReasoningTrim | None = None,
          usage_log: Path | None = None, tail: CompactionTail | None = None, images=None,
          nudge=None, effort=None, window=None, level_watch=None, review=None, pace=None) -> ThreadingHTTPServer:
    host, _, port = upstream.replace("http://", "").rstrip("/").rpartition(":")
    stats = stats if stats is not None else Stats()
    handler = make_handler(host or "127.0.0.1", int(port), sampling, paths, stats, trim, usage_log, tail, images,
                           nudge, effort, window, level_watch, review, pace)
    server = ThreadingHTTPServer(("127.0.0.1", listen_port), handler)
    server.daemon_threads = True
    server.stats = stats                      # type: ignore[attr-defined]
    server.trim = trim                        # type: ignore[attr-defined]
    server.tail = tail                        # type: ignore[attr-defined]
    server.images = images                    # type: ignore[attr-defined]
    server.nudge = nudge                      # type: ignore[attr-defined]
    server.effort = effort                    # type: ignore[attr-defined]
    server.window = window                    # type: ignore[attr-defined]
    server.level_watch = level_watch          # type: ignore[attr-defined]
    server.review = review                    # type: ignore[attr-defined]
    server.pace = pace                        # type: ignore[attr-defined]
    if manifest:
        manifest.write_text(json.dumps(
            {"listen": listen_port, "upstream": upstream, "paths": list(paths),
             "sampling": sampling, "started": time.strftime("%FT%TZ", time.gmtime()),
             "reasoning_trim": None if trim is None else
             {"trim_at": trim.trim_at, "keep_items": trim.keep_items, "min_gain": trim.min_gain,
              "chars_per_token": trim.chars_per_token,
              "log": None if trim.log_path is None else str(trim.log_path)},
             "compaction_tail": None if tail is None else
             {"budget": tail.budget, "chars_per_token": tail.chars_per_token,
              "log": None if tail.log_path is None else str(tail.log_path)},
             "act_images": None if images is None else
             {"run_dir": str(images.games.parent), "png_size": images.png_size},
             "act_nudge": None if nudge is None else {"every": nudge.every},
             "effort_schedule": None if effort is None else
             {"low": effort.low, "high": effort.high, "level_tokens": effort.level_tokens,
              "run_dir": str(effort.games.parent)},
             "context_window": None if window is None else
             {"high": window.high, "keep": window.keep, "notes_chars": window.notes_chars,
              "notes_after": window.notes_after, "hard": window.hard,
              "chars_per_token": window.chars_per_token, "run_dir": str(window.run_dir),
              "log": None if window.log_path is None else str(window.log_path)},
             "level_watch": None if level_watch is None else
             {"first": level_watch.first, "floor": level_watch.floor, "factors": list(level_watch.factors),
              "run_dir": str(level_watch.games.parent), "stuck": list(getattr(level_watch, "stuck", ()))},
             "review_notice": None if review is None else {"run_dir": str(review.run_dir)},
             "game_pace": None if pace is None else
             {"run_dir": str(pace.games.parent), "game_tokens": pace.game_tokens,
              "log": None if pace.log_path is None else str(pace.log_path)},
             "usage_log": None if usage_log is None else str(usage_log)},
            indent=2, ensure_ascii=False))
    return server


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--listen", type=int, required=True)
    p.add_argument("--upstream", required=True, help="http://127.0.0.1:<model port>")
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                   help="sampling parameter written explicitly into every request (repeatable); overrides the client's value")
    p.add_argument("--paths", default=",".join(DEFAULT_PATHS))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--report-every", type=int, default=200)
    p.add_argument("--trim-reasoning-at", type=int,
                   help="once the context (the previous response's input+output tokens) passes this, trim old reasoning (reasoning_trim.py)")
    p.add_argument("--trim-keep-items", type=int, default=8, help="reasoning in the last this-many input items is not trimmed")
    p.add_argument("--trim-min-gain", type=int, default=15000, help="jump only if it trims at least this many tokens (estimated)")
    p.add_argument("--trim-log", type=Path, help="one JSONL line per boundary jump")
    p.add_argument("--usage-log", type=Path, help="one JSONL line with the usage of every /v1/responses response")
    p.add_argument("--carry-tail-tokens", type=int,
                   help="after a compaction, append the last this-many tokens of the previous window's raw text after the summary (compaction_tail.py)")
    p.add_argument("--tail-log", type=Path, help="one JSONL line per window with what was carried")
    p.add_argument("--image-ref-key", type=Path, help="act_image: the middleware's signing key; when given, ./act do markers are replaced by images")
    p.add_argument("--image-run-dir", type=Path, help="act_image: run_arm's --run-dir (images are read from games/<game>/record/)")
    p.add_argument("--image-png-size", type=int, default=640, help="act_image: image side length (= config record.png_size)")
    p.add_argument("--act-nudge-every", type=int, help="after the last ./act do, remind every N tool calls that it is time to act (act_nudge.py)")
    p.add_argument("--effort-schedule", help="LOW,HIGH,TOKENS: send LOW effort while the output tokens spent on the current level are below TOKENS, otherwise HIGH"
                   " (effort_schedule.py)")
    p.add_argument("--effort-run-dir", type=Path, help="effort_schedule: run_arm's --run-dir (reads games/<game>/record/)")
    p.add_argument("--window-at", type=int, help="once the context passes this, stop sending the oldest history wholesale (context_window.py)")
    p.add_argument("--window-keep", type=int, help="keep about this many tokens after a cut (including the opening and the notes.md snapshot)")
    p.add_argument("--window-run-dir", type=Path, help="context_window: run_arm's --run-dir (reads games/<game>/notes.md)")
    p.add_argument("--window-log", type=Path, help="one JSONL line per cut")
    p.add_argument("--window-notes-after", type=int,
                   help="past this context (codex's pre-compaction reminder threshold), do not cut until notes.md has been written (context_window.py)")
    p.add_argument("--window-hard", type=int, help="wait for notes.md at most until here, then cut regardless")
    p.add_argument("--level-watch", help="FIRST,FLOOR,F1,F2: remind when a level spends too many output tokens (level_watch.py)")
    p.add_argument("--level-watch-run-dir", type=Path, help="level_watch: run_arm's --run-dir (reads games/<game>/record/)")
    p.add_argument("--level-watch-stuck", help="TOKENS,ACTIONS: remind only when a level's output passes k×TOKENS with actions ≤ k×ACTIONS (k=1,2; "
                                             "level_watch.StuckWatch, RSI round 3 rf)")
    p.add_argument("--review-run-dir", type=Path,
                   help="review_notice: run_arm's --run-dir (reviewer notices in reviews/notices/, appended after the player's next tool output)")
    p.add_argument("--pace-run-dir", type=Path, help="game_pace: run_arm's --run-dir (writes games/<game>/.pace.json)")
    p.add_argument("--pace-game-tokens", type=int, help="game_pace: per-game output budget (= the arm config's game_output_tokens)")
    p.add_argument("--pace-log", type=Path, help="game_pace: one JSONL line at the start of every level (when Y is fixed)")
    a = p.parse_args(argv)

    sampling = dict(parse_set(s) for s in a.set)
    if not sampling:
        raise SystemExit("without --set nothing is written explicitly, and this proxy is pointless")
    paths = tuple(x.strip().rstrip("/") for x in a.paths.split(",") if x.strip())
    stats = Stats()
    trim = None
    if a.trim_reasoning_at is not None:
        trim = ReasoningTrim(a.trim_reasoning_at, a.trim_keep_items, a.trim_min_gain, log_path=a.trim_log)
    tail = None
    if a.carry_tail_tokens is not None:
        tail = CompactionTail(a.carry_tail_tokens, log_path=a.tail_log)
    images = None
    if (a.image_ref_key is None) != (a.image_run_dir is None):
        raise SystemExit("--image-ref-key and --image-run-dir go together")
    if a.image_ref_key is not None:
        from act_images import ActImages
        from arcforge.arc_code.image_ref import load_key
        images = ActImages(load_key(a.image_ref_key), a.image_run_dir, a.image_png_size)
    nudge = None
    if a.act_nudge_every is not None:
        from act_nudge import ActNudge
        nudge = ActNudge(a.act_nudge_every)
    effort = None
    if (a.effort_schedule is None) != (a.effort_run_dir is None):
        raise SystemExit("--effort-schedule and --effort-run-dir go together")
    if a.effort_schedule is not None:
        from effort_schedule import EffortSchedule, parse
        effort = EffortSchedule(a.effort_run_dir, *parse(a.effort_schedule))
    window = None
    given = [x is not None for x in (a.window_at, a.window_keep, a.window_run_dir)]
    if any(given) and not all(given):
        raise SystemExit("--window-at, --window-keep and --window-run-dir go together")
    if a.window_at is not None:
        from context_window import ContextWindow
        window = ContextWindow(a.window_run_dir, a.window_at, a.window_keep, notes_after=a.window_notes_after,
                               hard=a.window_hard, log_path=a.window_log)
    level_watch = None
    if a.level_watch is not None and a.level_watch_stuck is not None:
        raise SystemExit("--level-watch and --level-watch-stuck are mutually exclusive")
    if (a.level_watch is None and a.level_watch_stuck is None) != (a.level_watch_run_dir is None):
        raise SystemExit("--level-watch (or --level-watch-stuck) and --level-watch-run-dir go together")
    if a.level_watch is not None:
        import level_watch as lw
        level_watch = lw.LevelWatch(a.level_watch_run_dir, *lw.parse(a.level_watch))
    if a.level_watch_stuck is not None:
        import level_watch as lw
        level_watch = lw.StuckWatch(a.level_watch_run_dir, *lw.parse_stuck(a.level_watch_stuck))
    review = None
    if a.review_run_dir is not None:
        from review_notice import ReviewNotice
        review = ReviewNotice(a.review_run_dir)
    pace = None
    if (a.pace_run_dir is None) != (a.pace_game_tokens is None) or (a.pace_log is not None and a.pace_run_dir is None):
        raise SystemExit("--pace-run-dir and --pace-game-tokens go together (--pace-log needs them)")
    if a.pace_run_dir is not None:
        from game_pace import GamePace
        pace = GamePace(a.pace_run_dir, a.pace_game_tokens, log_path=a.pace_log)
    server = serve(a.listen, a.upstream, sampling, paths, a.manifest, stats, trim, a.usage_log, tail, images, nudge,
                   effort, window, level_watch, review, pace)
    print(f"[sampling_proxy] :{a.listen} → {a.upstream}  explicit {sampling}  paths {list(paths)}"
          f"{'' if trim is None else f'  reasoning_trim at {trim.trim_at} keep {trim.keep_items} min_gain {trim.min_gain}'}"
          f"{'' if tail is None else f'  carry_tail {tail.budget}'}"
          f"{'' if images is None else f'  act_images {a.image_run_dir} {a.image_png_size}px'}"
          f"{'' if nudge is None else f'  act_nudge every {nudge.every}'}"
          f"{'' if effort is None else f'  effort {effort.low}<{effort.level_tokens}<={effort.high}'}"
          f"{'' if window is None else f'  context_window at {window.high} keep {window.keep} notes_after {window.notes_after} hard {window.hard}'}"
          f"{'' if level_watch is None else f'  level_watch first {level_watch.first} floor {level_watch.floor} x{level_watch.factors}'}"
          f"{'' if review is None else f'  review_notice {review.run_dir} (reloaded {len(review.sent)})'}"
          f"{'' if pace is None else f'  game_pace {a.pace_run_dir} {pace.game_tokens:,}'}",
          flush=True)

    def report():
        while True:
            time.sleep(60)
            print(f"[sampling_proxy] {json.dumps(stats.snapshot(), ensure_ascii=False)}"
                  f"{'' if trim is None else ' trim ' + json.dumps(trim.snapshot())}"
                  f"{'' if tail is None else ' tail ' + json.dumps(tail.snapshot())}"
                  f"{'' if images is None else ' act_images ' + json.dumps(images.snapshot())}"
                  f"{'' if nudge is None else ' act_nudge ' + json.dumps(nudge.snapshot())}"
                  f"{'' if effort is None else ' effort ' + json.dumps(effort.snapshot())}"
                  f"{'' if window is None else ' window ' + json.dumps(window.snapshot())}"
                  f"{'' if level_watch is None else ' level_watch ' + json.dumps(level_watch.snapshot())}"
                  f"{'' if review is None else ' review_notice ' + json.dumps(review.snapshot())}"
                  f"{'' if pace is None else ' game_pace ' + json.dumps(pace.snapshot())}", flush=True)

    threading.Thread(target=report, daemon=True).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    print(f"[sampling_proxy] done {json.dumps(stats.snapshot(), ensure_ascii=False)}"
          f"{'' if images is None else ' act_images ' + json.dumps(images.snapshot())}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
