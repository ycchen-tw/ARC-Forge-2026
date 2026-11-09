"""After a reviewer's report is delivered, append a one-line notice after the player's next tool output: hooked into
``sampling_proxy.py``'s ``/v1/responses`` (``--review-run-dir <run_arm's run-dir>``;
docs/plans/arc3_readonly_reviewer_20261014.md).

After run_arm's ReviewScheduler (review_scheduler.py) writes the report into the player workspace's ``review/Rnn.md``,
it drops a notice ``<run-dir>/reviews/notices/<game>/Rnn.json`` (``{"id", "game", "text"}``, written tmp + rename).
Here:

- Each request is matched to its game by the AGENTS.md header in its input (as in level_watch.py). A header under
  ``<run-dir>/reviews/…`` is the reviewer's own request: **never attach**.
- If the game has an undelivered notice: append it after the latest tool output (keyed by call_id), recording it in
  ``notices/delivered.jsonl`` before modifying the payload. Every later request gets the same bytes re-applied on the
  same call_id -- the same history always yields the same bytes, and the prefix cache only serves appends.
- When the proxy restarts (the notebook's ProxyKeeper) it reads delivered.jsonl back: delivered notices are not resent,
  but are re-applied as before.
- A request with no tool output yet (the first request of a new thread) leaves it for the next request.
- If the notice was pushed out of context (compaction or a new thread: none of the call_ids it was attached to is in
  the input) and the player has not opened the report yet (no tool call in the input mentions ``review/Rnn``), attach
  the same sentence again to the latest tool output, at most ``AGAIN_MAX`` times. posix smoke run: right after cn04's
  notice was attached, the player's next step was writing notes before compaction; after the compaction it never
  mentioned it again nor opened the report. Once opened, it is not attached again.

On start-up, write ``notices/PROXY.json``: run_arm only turns the reviewer on once it sees it (without this proxy a
report would be delivered and nobody told).

``F01`` is not a report: it is the "N minutes left, clear levels at full speed" that run_arm's config ``final_call``
(hs110 v39) drops for every game a few minutes before the global deadline, taking the same path (the player never
"opens" it, so when pushed out of context it is attached again, at most ``AGAIN_MAX`` times).
"""

from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path

OUTPUTS = ("function_call_output", "custom_tool_call_output")
CALLS = ("function_call", "custom_tool_call")
#: how many more times a notice is attached after being pushed out of context
AGAIN_MAX = 2
#: run_arm checks this file before turning the reviewer on (the proxy runs with --review-run-dir)
MARKER = "PROXY.json"
DELIVERED = "delivered.jsonl"


def notices_dir(run_dir: Path) -> Path:
    return run_dir / "reviews" / "notices"


def write_notice(run_dir: Path, game: str, review_id: str, text: str) -> Path:
    """Scheduler side: one notice (new file, never overwritten). ``review_id`` is like ``R01`` (a reviewer report) or
    run_arm final_call's ``F01``, unique within a game."""
    if not re.fullmatch(r"[RF]\d\d", review_id):
        raise ValueError(f"review id {review_id!r} is not Rnn or Fnn")
    out = notices_dir(run_dir) / game
    out.mkdir(parents=True, exist_ok=True)
    target = out / f"{review_id}.json"
    if target.exists():
        raise FileExistsError(target)
    tmp = out / f".{review_id}.json.tmp"
    tmp.write_text(json.dumps({"id": review_id, "game": game, "text": text, "written": time.time()}))
    os.replace(tmp, target)
    return target


class ReviewNotice:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.games = run_dir / "games"
        self.notices = notices_dir(run_dir)
        self.player = re.compile(r"# AGENTS\.md instructions for " + re.escape(str(self.games)) + r"/([A-Za-z0-9_.-]+)")
        self.reviewer = re.compile(r"# AGENTS\.md instructions for " + re.escape(str(run_dir / "reviews")) + "/")
        self.lock = threading.Lock()
        #: game -> call_id -> [full notice text…] (in delivery order)
        self.attached: dict[str, dict[str, list[str]]] = {}
        #: (game, id) delivered
        self.sent: set[tuple[str, str]] = set()
        #: (game, id) -> call_ids it was attached to (the first + re-attachments)
        self.where: dict[tuple[str, str], list[str]] = {}
        #: (game, id) already opened by the player (a tool call mentioned review/Rnn)
        self.opened: set[tuple[str, str]] = set()
        self.counts = {"requests": 0, "player_requests": 0, "reviewer_requests": 0, "unknown": 0,
                       "delivered": 0, "again": 0, "opened": 0, "attached": 0, "odd_outputs": 0}
        self.notices.mkdir(parents=True, exist_ok=True)
        self.log = self.notices / DELIVERED
        if self.log.exists():
            raw = self.log.read_bytes()
            for line in raw[:raw.rfind(b"\n") + 1].splitlines():
                if line.strip():
                    row = json.loads(line)
                    key = (row["game"], row["id"])
                    if row.get("opened"):
                        self.opened.add(key)
                    elif row.get("again"):
                        self._again(*key, row["call_id"], self._text(*key))
                    else:
                        self._attach(*key, row["call_id"], self._text(*key))
        (self.notices / MARKER).write_text(json.dumps({"pid": os.getpid(), "started": time.time(),
                                                       "run_dir": str(run_dir), "reloaded": len(self.sent)}))

    def _text(self, game: str, review_id: str) -> str:
        spec = json.loads((self.notices / game / f"{review_id}.json").read_text())
        if spec["game"] != game or spec["id"] != review_id or not spec["text"].strip():
            raise ValueError(f"notice {game}/{review_id} is malformed")
        return spec["text"]

    def _attach(self, game: str, review_id: str, call_id: str, text: str) -> None:
        if (game, review_id) in self.sent:
            raise RuntimeError(f"notice {game}/{review_id} delivered twice")
        self.sent.add((game, review_id))
        self.where[(game, review_id)] = [call_id]
        self.attached.setdefault(game, {}).setdefault(call_id, []).append(text)

    def _again(self, game: str, review_id: str, call_id: str, text: str) -> None:
        where = self.where[(game, review_id)]
        if call_id in where or len(where) > AGAIN_MAX:
            raise RuntimeError(f"notice {game}/{review_id} re-attached to {call_id} after {where}")
        where.append(call_id)
        self.attached.setdefault(game, {}).setdefault(call_id, []).append(text)

    def _record(self, row: dict) -> None:
        # persist before sending: if the proxy dies after this, on restart it re-applies to the same call_id
        with self.log.open("a") as f:
            f.write(json.dumps({**row, "t": time.time()}) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.counts, "games": len(self.attached)}

    def _who(self, items: list) -> tuple[str | None, str | None]:
        """("player", game), ("reviewer", None) or (None, None)."""
        for item in items:
            if not isinstance(item, dict) or item.get("type", "message") != "message" or item.get("role") != "user":
                continue
            content = item.get("content")
            parts = content if isinstance(content, list) else [{"text": content}]
            for part in parts:
                text = part.get("text") if isinstance(part, dict) else None
                if not isinstance(text, str):
                    continue
                if self.reviewer.search(text):
                    return "reviewer", None
                m = self.player.search(text)
                if m:
                    return "player", m.group(1)
        return None, None

    def _pending(self, game: str) -> list[str]:
        folder = self.notices / game
        if not folder.is_dir():
            return []
        return sorted(p.stem for p in folder.glob("[RF][0-9][0-9].json"))

    def apply(self, payload: dict) -> tuple[str | None, int]:
        """Modifies payload in place; returns (game, number of notices attached to this request, re-applied ones
        included)."""
        items = payload.get("input")
        items = items if isinstance(items, list) else []
        who, game = self._who(items)
        with self.lock:
            self.counts["requests"] += 1
            if who is None:
                self.counts["unknown"] += 1
            else:
                self.counts[f"{who}_requests"] += 1
        if game is None:
            return None, 0
        pending = self._pending(game)
        with self.lock:
            new = [rid for rid in pending if (game, rid) not in self.sent]
            last = max((i for i, it in enumerate(items) if isinstance(it, dict) and it.get("type") in OUTPUTS),
                       default=None)
            if new and last is not None:
                call_id = items[last]["call_id"]
                for rid in new:
                    text = self._text(game, rid)
                    self._record({"game": game, "id": rid, "call_id": call_id})
                    self._attach(game, rid, call_id, text)
                    self.counts["delivered"] += 1
                    print(f"[review_notice] {game} {rid} -> call {call_id}", flush=True)
            present = {it.get("call_id") for it in items if isinstance(it, dict) and it.get("type") in OUTPUTS}
            calls = [str(it.get("arguments", "")) + str(it.get("input", ""))
                     for it in items if isinstance(it, dict) and it.get("type") in CALLS]
            for rid in pending:
                key = (game, rid)
                if key not in self.sent or key in self.opened:
                    continue
                if any(f"review/{rid}" in c for c in calls):
                    self._record({"game": game, "id": rid, "opened": True})
                    self.opened.add(key)
                    self.counts["opened"] += 1
                    continue
                where = self.where[key]
                if last is None or present & set(where) or len(where) > AGAIN_MAX:
                    continue
                call_id = items[last]["call_id"]
                self._record({"game": game, "id": rid, "call_id": call_id, "again": True})
                self._again(game, rid, call_id, self._text(game, rid))
                self.counts["again"] += 1
                print(f"[review_notice] {game} {rid} again -> call {call_id} (left the context unopened)", flush=True)
            notes = {cid: list(texts) for cid, texts in self.attached.get(game, {}).items()}
        n = 0
        for item in items:
            if not (isinstance(item, dict) and item.get("type") in OUTPUTS and item.get("call_id") in notes):
                continue
            output = item.get("output")
            texts = notes[item["call_id"]]
            if isinstance(output, str):
                item["output"] = output + "".join(f"\n\n{t}" for t in texts)
            elif isinstance(output, list):
                item["output"] = [*output, *({"type": "input_text", "text": f"\n\n{t}"} for t in texts)]
            else:
                # Responses API tool output is only a string or a content array; another shape must not fail all of
                # this game's requests, so log it loudly
                print(f"[review_notice] {game}: call {item['call_id']} output is {type(output).__name__}, "
                      "notice not attached", flush=True)
                with self.lock:
                    self.counts["odd_outputs"] += 1
                continue
            n += len(texts)
        with self.lock:
            self.counts["attached"] += n
        return game, n
