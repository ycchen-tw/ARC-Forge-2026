"""Scheduling for the idle-slot read-only reviewer: who is reviewed, when, and how the result is delivered
(docs/plans/arc3_readonly_reviewer_20261014.md).

The arm config's ``reviewer`` block (absent = nothing starts at all, behavior unchanged). Every key is required, each
checked by ``check_spec``:

- ``prompts``: ``review_prompts/<name>/``; ``output_tokens``, ``wall_seconds``: output-token and seconds cap of one review
- launch window: new reviews open only after ``start_after_seconds`` into the run and while the global time left >
  ``stop_launch_before_end_seconds``; each review's deadline = min(now + wall_seconds, global deadline)
- capacity: open only while players (running + resume pool + main queue) + running reviewers < ``max_total``; at most
  ``max_reviewers`` reviewers; at most one opened per ``start_interval_seconds``. Players never yield: the cap only
  blocks reviewer launches
- game selection: not won, running or in the resume pool; ≥ ``min_level_seconds`` spent on the current level, and (no
  level clear on record, or ≥ ``min_level_factor`` × median of earlier levels; factor 0 = ignore); at most
  ``max_reviews_per_game`` per game, one at a time; a re-review waits at least ``repeat_after_seconds`` after the
  previous one ended, and needs new actions since. ``order``: ``stalled`` picks by seconds spent on the current level,
  most first; ``round_robin`` picks never-reviewed games first, then those whose last review ended earliest (stalls
  are ignored; with both thresholds at 0 it simply reviews in rotation)

One review (``run_review``): snapshot (``review.build_snapshot``, record via hardlinks; during the review
``review.refresh_snapshot`` moves it forward every ``REFRESH_SECONDS``) -> reviewer brief -> driver home ->
driver env file (outside the workspace) -> run inside the fence (bwrap: jail.sh; posix: fence_posix.sh's reviewer
role, one uid at a time) -> once ``REVIEW.md`` is written, deliver it to the player workspace's ``review/Rnn.md``, then
write a notice for the proxy (``review_notice.py``; the proxy must run with ``--review-run-dir``, and run_arm checks
its ``PROXY.json`` before the run). Every review is recorded in ARM_STATE's ``reviews`` and
``<run-dir>/reviews/reviews.jsonl``; log lines always start with ``REVIEW`` (Kaggle's LIVE filters on that word).

Card writing (the arm config's ``cards`` block, which requires ``reviewer``; the user 2026-10-31, hs110 v40): once the
run's cumulative level clears exceed ``min_total_clears``, whenever a game clears a new level and the player has
finished the notes for that level, a card write starts (``run_card``, prompt ``review_prompts/<cards.prompts>/card.md``):
the cleared-level documents, notes and statistics from the snapshot are assembled into one request, the model (no
thinking, no codex; the user 2026-10-31: codex took 12 minutes per card, and 4 card writers reached only 24 of 106
games) returns the whole card in one go, and if the format is right it is published as
``<run-dir>/cards/live/<game>.md`` (card_library). ``output_tokens`` is the request's max_tokens, ``wall_seconds`` its
timeout. Cards are written only after the player clears a level and finishes the notes (the user 2026-10-31: update
only after the clear's note done, not mid-level). This arm has no note_gate (note_gate does not go with reviewer), so
"notes finished" uses ``./act note_done``'s own condition: ``levels/LNN.md`` and ``notes.md`` both modified after the
clearing step (act_client.cmd_note_done). A game is written again only once it has cleared more levels than at its
previous card write (successful or not); levels cleared during a write are written right after it. Card writing and
reviews share ``max_total`` (players + reviews + card writes); at most ``max_writers`` writers and
``max_cards_per_game`` per game; games whose just-cleared level took longest go first. When a review starts,
``card_library.assemble`` puts the static cards and the current live cards (excluding the same base game) into the
reviewer workspace's ``game_cards/``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import stat
import string
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

import card_library
import review_notice
import reviewer
from arcforge.arc_code import review

SPEC_KEYS = frozenset({"prompts", "start_after_seconds", "stop_launch_before_end_seconds", "max_total",
                       "max_reviewers", "start_interval_seconds", "output_tokens", "wall_seconds",
                       "min_level_seconds", "min_level_factor", "max_reviews_per_game", "repeat_after_seconds",
                       "order"})
ORDERS = ("stalled", "round_robin")
CARD_SPEC_KEYS = frozenset({"prompts", "max_writers", "min_total_clears", "max_cards_per_game", "output_tokens",
                            "wall_seconds", "stop_launch_before_end_seconds"})
#: card writing: the prompt file (``review_prompts/<cards.prompts>/card.md``) and its fields; the card kept in
#: ``cards/<game>/Cnn/``
CARD_PROMPT = "card.md"
CARD_FIELDS = ("game", "levels_completed", "win_levels", "level", "step", "documents")
CARD_FILE = "CARD.md"
#: example card shown for format in the card-writing request (one of the static cards)
EXAMPLE_CARD = "KA59.md"
#: document cap (characters) of one card-writing request: tiny's 25 cards had a median of 17k, at most 28k; above it
#: the request is not sent and counts as failed
CARD_INPUT_MAX_CHARS = 200_000
#: minimum seconds for a card-writing request (tiny's estimate: 5-8k tokens prefill, about 1.6k tokens output, about
#: 33 tokens/s per stream)
MIN_CARD_SECONDS = 60
#: floor: reviewer.run's exploration session needs at least 60 s (finish_fraction 0.2 -> window ≥ 75 s); plus room for
#: fence start-up and wrap-up
MIN_WINDOW_SECONDS = 300
POLL_SECONDS = 10.0
#: at wrap-up, the longest wait for running reviews to be killed (the launcher checks for cancellation every 5 s)
STOP_SECONDS = 180.0
#: how often during a review to check whether the player has left that level (reads history.jsonl)
LEFT_LEVEL_SECONDS = 30.0
#: how often during a review the snapshot is moved forward (review.refresh_snapshot: new steps in record/, notes and
#: code in record/live/)
REFRESH_SECONDS = 60.0


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] REVIEW {message}", flush=True)


def check_spec(spec, cfg: dict, driver_name: str) -> dict:
    """The ``reviewer`` block and the arm settings it depends on; SystemExit on anything wrong, nothing guessed."""
    if not isinstance(spec, dict):
        raise SystemExit(f"reviewer must be an object, not {type(spec).__name__}")
    if set(spec) != SPEC_KEYS:
        raise SystemExit(f"reviewer keys: missing {sorted(SPEC_KEYS - set(spec))}, unknown {sorted(set(spec) - SPEC_KEYS)}")
    if not isinstance(spec["prompts"], str) or not spec["prompts"]:
        raise SystemExit("reviewer.prompts must name a set under work/arc_code/review_prompts")
    try:
        reviewer.ReviewPrompts(spec["prompts"])
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"reviewer.prompts: {error}") from error
    positive = ("max_total", "max_reviewers", "output_tokens", "wall_seconds", "max_reviews_per_game")
    if spec["order"] not in ORDERS:
        raise SystemExit(f"reviewer.order must be one of {ORDERS}, not {spec['order']!r}")
    for key in SPEC_KEYS - {"prompts", "min_level_factor", "order"}:
        value = spec[key]
        if type(value) is not int or value < (1 if key in positive else 0):
            raise SystemExit(f"reviewer.{key} must be an int >= {1 if key in positive else 0}, not {value!r}")
    factor = spec["min_level_factor"]
    if type(factor) not in (int, float) or not (factor == 0 or factor >= 1):
        raise SystemExit(f"reviewer.min_level_factor must be 0 (off) or a number >= 1, not {factor!r}")
    for key in ("wall_seconds", "stop_launch_before_end_seconds"):
        if spec[key] < MIN_WINDOW_SECONDS:
            raise SystemExit(f"reviewer.{key} {spec[key]} < {MIN_WINDOW_SECONDS}: a review window would be too short")
    if spec["max_reviewers"] >= spec["max_total"]:
        raise SystemExit(f"reviewer.max_reviewers {spec['max_reviewers']} must be below max_total {spec['max_total']}")
    if driver_name != "codex":
        raise SystemExit(f"reviewer counts codex rollouts and resumes a codex thread, not {driver_name}")
    if not cfg.get("record"):
        raise SystemExit("reviewer needs the game record (config record): the snapshot and the stall signal read it")
    if not isinstance(cfg.get("watchdog_cpu_seconds"), int) or cfg["watchdog_cpu_seconds"] <= 0:
        raise SystemExit("reviewer's brief states the per-process CPU limit: config needs watchdog_cpu_seconds")
    return dict(spec)


def check_cards_spec(spec, review_spec: dict | None) -> dict:
    """The ``cards`` block: requires the reviewer (shared scheduling and capacity; cards are read only by reviewers),
    with exactly the expected keys."""
    if review_spec is None:
        raise SystemExit("cards needs the reviewer block: cards run in the reviewer's scheduler and only reviewers read them")
    if not isinstance(spec, dict) or set(spec) != CARD_SPEC_KEYS:
        got = set(spec) if isinstance(spec, dict) else set()
        raise SystemExit(f"cards keys: missing {sorted(CARD_SPEC_KEYS - got)}, unknown {sorted(got - CARD_SPEC_KEYS)}")
    try:
        load_card_prompt(spec["prompts"])
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"cards.prompts: {error}") from error
    for key in CARD_SPEC_KEYS - {"prompts"}:
        value = spec[key]
        low = 0 if key == "min_total_clears" else 1
        if type(value) is not int or value < low:
            raise SystemExit(f"cards.{key} must be an int >= {low}, not {value!r}")
    # cap of one request; a request sent before the launch window closes always ends before the global deadline
    if not MIN_CARD_SECONDS <= spec["wall_seconds"] <= spec["stop_launch_before_end_seconds"]:
        raise SystemExit(f"cards.wall_seconds {spec['wall_seconds']} must be in [{MIN_CARD_SECONDS}, "
                         f"stop_launch_before_end_seconds {spec['stop_launch_before_end_seconds']}]")
    if spec["max_writers"] + review_spec["max_reviewers"] >= review_spec["max_total"]:
        raise SystemExit(f"cards.max_writers {spec['max_writers']} + reviewer.max_reviewers "
                         f"{review_spec['max_reviewers']} must be below max_total {review_spec['max_total']}")
    card_library.static_cards(card_library.STATIC_DIR)     # broken packaged static cards blow up before the run
    return dict(spec)


def check_proxy(run_dir: Path) -> dict:
    """The proxy must run with ``--review-run-dir <the same run-dir>``: without it, reports get delivered and nobody
    tells the player."""
    marker = review_notice.notices_dir(run_dir) / review_notice.MARKER
    if not marker.is_file():
        raise SystemExit(f"reviewer needs the model proxy started with --review-run-dir {run_dir} (no {marker})")
    info = json.loads(marker.read_text())
    if info["run_dir"] != str(run_dir):
        raise SystemExit(f"the model proxy watches {info['run_dir']}, this arm runs in {run_dir}: "
                         "the AGENTS.md header would never match")
    return info


# ---------------------------------------------------------------- scheduling

def notes_done(ws: Path, level: int, cleared_at: float) -> bool:
    """``./act note_done``'s condition (act_client.cmd_note_done): ``levels/LNN.md`` and ``notes.md`` are both regular
    files, both modified after level ``level`` was cleared (``cleared_at``, history's wall clock). The reader is the
    harness: symlinks are not followed. A player who made ``levels`` a regular file (NotADirectoryError) has not
    finished either, and note_done would not pass."""
    for path in (ws / "levels" / f"L{level:02d}.md", ws / "notes.md"):
        try:
            st = os.lstat(path)
        except (FileNotFoundError, NotADirectoryError):
            return False
        if not stat.S_ISREG(st.st_mode) or st.st_mtime <= cleared_at:
            return False
    return True


class ReviewScheduler(threading.Thread):
    """``review_one(job, stopping) -> dict`` runs one review (``run_review`` live, a fake in tests); ``record(row,
    summary)`` is called once at the end of each review (writes ARM_STATE and reviews.jsonl). ``occupancy`` is handed
    to run_slots."""

    def __init__(self, spec: dict, *, games: dict[str, Path], deadline: float, started: float, review_one,
                 record, cancelled: threading.Event, clock=time.monotonic, wall=time.time,
                 poll: float = POLL_SECONDS, cards: dict | None = None, card_one=None, card_record=None) -> None:
        super().__init__(daemon=True, name="review-scheduler")
        if not ((cards is None) == (card_one is None) == (card_record is None)):
            raise ValueError("cards, card_one and card_record go together")
        self.spec, self.games, self.deadline, self.started = spec, games, deadline, started
        self.review_one, self.record, self.cancelled = review_one, record, cancelled
        self.cards, self.card_one, self.card_record = cards, card_one, card_record
        self.card_running: dict[str, dict] = {}      # game -> its running card write
        self.card_past: dict[str, list[dict]] = {}   # game -> each card write {"id", "levels" (up to which level), "ended", "ok"}
        self.card_counts = {"launched": 0, "delivered": 0, "failed": 0}
        self.card_error: str | None = None           # card scheduling blew up: no more card writes, reviews continue
        self.clock, self.wall, self.poll = clock, wall, poll
        self.halt = threading.Event()
        self.lock = threading.Lock()
        self.occ = {"running": frozenset(), "pooled": frozenset(), "queued": 0}
        self.running: dict[str, dict] = {}           # game -> its running review
        self.past: dict[str, list[dict]] = {}        # game -> each review {"id", "step", "ended"}
        self.threads: list[threading.Thread] = []
        self.counts = {"launched": 0, "delivered": 0, "failed": 0}
        self.last_launch: float | None = None
        self.error: str | None = None

    # run_slots calls this inside its lock: just copy it
    def occupancy(self, snapshot: dict) -> None:
        with self.lock:
            self.occ = snapshot

    def stopping(self) -> bool:
        return self.halt.is_set() or self.cancelled.is_set()

    def stop(self) -> None:
        self.halt.set()

    def summary(self) -> dict:
        with self.lock:
            occ = self.occ
            return {**self.counts, "running": len(self.running),
                    "players": {"running": len(occ["running"]), "pooled": len(occ["pooled"]), "queued": occ["queued"]},
                    "max_total": self.spec["max_total"], "error": self.error,
                    **({"cards": {**self.card_counts, "running": len(self.card_running), "error": self.card_error}}
                       if self.cards else {})}

    def candidates(self, active, now: float) -> list[tuple[str, dict]]:
        """Games that can be reviewed, sorted by ``order`` (see the module docstring)."""
        spec, out = self.spec, []
        wall_now = self.wall()
        for game in sorted(active):
            with self.lock:
                if game in self.running:
                    continue
                past = list(self.past.get(game, []))
            s = review.stall(self.games[game] / "record" / "history.jsonl", wall_now)
            if s is None or s["state"] == "WIN" or len(past) >= spec["max_reviews_per_game"]:
                continue
            if s["level_seconds"] < spec["min_level_seconds"]:
                continue
            median = s["median_previous_seconds"]
            if median is not None and s["level_seconds"] < spec["min_level_factor"] * median:
                continue
            if past:
                last = past[-1]
                if now - last["ended"] < spec["repeat_after_seconds"] or s["step"] <= last["step"]:
                    continue
            out.append((game, s, past[-1]["ended"] if past else None))
        if spec["order"] == "stalled":
            out.sort(key=lambda gsl: -gsl[1]["level_seconds"])
        else:
            out.sort(key=lambda gsl: (gsl[2] is not None, gsl[2] or 0.0, gsl[0]))
        return [(game, s) for game, s, _ in out]

    def card_candidates(self, now: float) -> list[tuple[str, dict]]:
        """Games due for a card (all games, including finished ones), the one whose just-cleared level took longest
        first; empty while the run's total clears are below the threshold."""
        cards, wall_now = self.cards, self.wall()
        found, total = [], 0
        for game in sorted(self.games):
            s = review.stall(self.games[game] / "record" / "history.jsonl", wall_now)
            if s is None:
                continue
            total += s["levels_completed"]
            with self.lock:
                if game in self.card_running:
                    continue
                past = list(self.card_past.get(game, []))
            done = s["levels_completed"]
            # write only on clears: only once more levels are cleared than at the previous card write (success or
            # failure both count); the new card's range starts after the last successful card
            if done <= (past[-1]["levels"] if past else 0) or len(past) >= cards["max_cards_per_game"]:
                continue
            covered = max([p["levels"] for p in past if p["ok"]], default=0)
            if not notes_done(self.games[game], done, s["level_start_time"]):
                continue
            s["new_levels_seconds"] = max(s["previous_level_seconds"][covered:done])
            s["covered"] = covered
            found.append((game, s))
        if total <= cards["min_total_clears"]:
            return []
        found.sort(key=lambda gs: -gs[1]["new_levels_seconds"])
        return found

    def tick_card(self, now: float) -> dict | None:
        """One decision on whether to write a card: returns the job if one was launched. Shares max_total and the
        launch interval with reviews."""
        cards = self.cards
        if self.deadline - now <= cards["stop_launch_before_end_seconds"]:
            return None
        with self.lock:
            occ = self.occ
            players = len(occ["running"]) + len(occ["pooled"]) + occ["queued"]
            if len(self.card_running) >= cards["max_writers"] \
                    or players + len(self.running) + len(self.card_running) >= self.spec["max_total"]:
                return None
            if self.last_launch is not None and now - self.last_launch < self.spec["start_interval_seconds"]:
                return None
        found = self.card_candidates(now)
        if not found:
            return None
        game, s = found[0]
        with self.lock:
            number = len(self.card_past.get(game, [])) + 1
            job = {"kind": "card", "game": game, "ws": self.games[game], "card_id": f"C{number:02d}",
                   "deadline": min(now + cards["wall_seconds"], self.deadline), "stall": s,
                   "launched": now, "launched_unix": self.wall(), "players": players,
                   "writers_before": len(self.card_running)}
            self.card_running[game] = job
            self.card_past.setdefault(game, []).append({"id": job["card_id"], "levels": s["levels_completed"],
                                                       "ended": None, "ok": False})
            self.card_counts["launched"] += 1
            self.last_launch = now
        log(f"card {game} {job['card_id']}: levels {s['covered'] + 1}-{s['levels_completed']} cleared "
            f"(longest {s['new_levels_seconds']:.0f}s); players {players}, writers "
            f"{job['writers_before'] + 1}/{cards['max_writers']}, window {job['deadline'] - now:.0f}s")
        thread = threading.Thread(target=self._run_card, args=(job,), daemon=True, name=f"card-{game}")
        self.threads.append(thread)
        thread.start()
        return job

    def _run_card(self, job: dict) -> None:
        game = job["game"]
        try:
            out = self.card_one(job, self.stopping)
        except BaseException as error:  # noqa: BLE001 — one failed card write must not stop the scheduler; the
                                        # cause goes into the row and the log
            out = {"delivered": None, "failure": f"{type(error).__name__}: {error}"[:600]}
            log(f"card {game} {job['card_id']} raised:\n{traceback.format_exc()[-3000:]}")
        ended = self.clock()
        row = {"kind": "card", "game": game, "card": job["card_id"], "launched_unix": job["launched_unix"],
               "ended_unix": self.wall(), "seconds": round(ended - job["launched"], 1),
               "levels_completed": job["stall"]["levels_completed"], "players_at_launch": job["players"], **out}
        with self.lock:
            del self.card_running[game]
            entry = self.card_past[game][-1]
            entry["ended"], entry["ok"] = ended, bool(out.get("delivered"))
            if out.get("levels_completed") is not None:
                entry["levels"] = out["levels_completed"]
            self.card_counts["delivered" if out.get("delivered") else "failed"] += 1
        log(f"card {game} {job['card_id']} " + (f"published {out['delivered']}" if out.get("delivered")
                                                 else f"failed: {out.get('failure')}")
            + f" after {row['seconds']:.0f}s, {out.get('output_tokens', 0):,} output tokens")
        self.card_record(row, self.summary())

    def tick(self) -> dict | None:
        """One decision: returns the job of a card write or review if one was launched (card writes first)."""
        spec = self.spec
        now = self.clock()
        if self.stopping():
            return None
        if self.cards is not None and self.card_error is None:
            try:
                job = self.tick_card(now)
            except Exception as error:  # noqa: BLE001 — card writing is new: if its scheduling breaks,
                                        # only card writing stops (logged loudly); v39's reviews go on
                self.card_error = f"{type(error).__name__}: {error}"[:600]
                log(f"CARDS FAILED, no more cards (reviews go on):\n{traceback.format_exc()[-3000:]}")
                job = None
            if job is not None:
                return job
        if now - self.started < spec["start_after_seconds"] \
                or self.deadline - now <= spec["stop_launch_before_end_seconds"]:
            return None
        with self.lock:
            occ = self.occ
            players = len(occ["running"]) + len(occ["pooled"]) + occ["queued"]
            if len(self.running) >= spec["max_reviewers"] \
                    or players + len(self.running) + len(self.card_running) >= spec["max_total"]:
                return None
            if self.last_launch is not None and now - self.last_launch < spec["start_interval_seconds"]:
                return None
            active = occ["running"] | occ["pooled"]
        found = self.candidates(active, now)
        if not found:
            return None
        game, s = found[0]
        with self.lock:
            number = len(self.past.get(game, [])) + 1
            job = {"game": game, "ws": self.games[game], "review_id": f"R{number:02d}",
                   "deadline": min(now + spec["wall_seconds"], self.deadline), "stall": s,
                   "launched": now, "launched_unix": self.wall(), "players": players,
                   "reviewers_before": len(self.running)}
            self.running[game] = job
            self.past.setdefault(game, []).append({"id": job["review_id"], "step": s["step"], "ended": None})
            self.counts["launched"] += 1
            self.last_launch = now
        log(f"launch {game} {job['review_id']}: level {s['level']} for {s['level_seconds']:.0f}s "
            f"(median before {s['median_previous_seconds'] if s['median_previous_seconds'] is None else round(s['median_previous_seconds'])}"
            f"), step {s['step']}; players {players}, reviewers {job['reviewers_before'] + 1}/{spec['max_reviewers']}, "
            f"window {job['deadline'] - now:.0f}s")
        thread = threading.Thread(target=self._run, args=(job,), daemon=True, name=f"review-{game}")
        self.threads.append(thread)
        thread.start()
        return job

    def _run(self, job: dict) -> None:
        game = job["game"]
        try:
            out = self.review_one(job, self.stopping)
        except BaseException as error:  # noqa: BLE001 — one failed review must not stop the scheduler; the
                                        # cause goes into the row and the log
            out = {"delivered": None, "failure": f"{type(error).__name__}: {error}"[:600]}
            log(f"{game} {job['review_id']} raised:\n{traceback.format_exc()[-3000:]}")
        ended = self.clock()
        row = {"game": game, "review": job["review_id"], "launched_unix": job["launched_unix"],
               "ended_unix": self.wall(), "seconds": round(ended - job["launched"], 1),
               "window_seconds": round(job["deadline"] - job["launched"], 1),
               "players_at_launch": job["players"],
               "stall": {k: job["stall"][k] for k in ("step", "level", "levels_completed", "level_seconds",
                                                       "median_previous_seconds", "since_action_seconds")},
               **out}
        with self.lock:
            del self.running[game]
            entry = self.past[game][-1]
            entry["ended"] = ended
            if out.get("snapshot_step") is not None:
                entry["step"] = out["snapshot_step"]
            self.counts["delivered" if out.get("delivered") else "failed"] += 1
        log(f"{game} {job['review_id']} " + (f"delivered {out['delivered']}" if out.get("delivered")
                                              else f"failed: {out.get('failure')}")
            + f" after {row['seconds']:.0f}s, {out.get('output_tokens', 0):,} output tokens")
        self.record(row, self.summary())

    def run(self) -> None:
        try:
            while not self.halt.wait(self.poll):
                if self.cancelled.is_set():
                    break
                self.tick()
        except BaseException as error:  # noqa: BLE001 — scheduler broken: log it loudly, open no more
                                        # reviews (players keep going)
            self.error = f"{type(error).__name__}: {error}"[:600]
            log(f"SCHEDULER FAILED, no more reviews:\n{traceback.format_exc()[-3000:]}")
        self.halt.set()
        for thread in list(self.threads):
            thread.join()


# ---------------------------------------------------------------- one review

@dataclasses.dataclass
class ReviewContext:
    run_dir: Path
    spec: dict
    pset: reviewer.ReviewPrompts
    driver: object
    knobs: object                      # reviewer.reviewer_knobs(...)
    paths: object                      # arms.Paths
    agent_runtime: Path                # parents[1] of --agent-python, not resolved (ARCSEC_RUNTIME for jail.sh / the fence)
    driver_bin: Path
    limits: dict[str, str]             # the brief's process_limit / cpu_limit
    process_data_kb: int
    cpu_seconds: int
    memory_max: str
    model_port: int                    # bwrap: the model port inside the jail
    sockets: Path                      # bwrap: holds model.sock
    model_health_url: str              # posix: the reviewer GETs it as itself
    library: Path | None               # act_library's agent_lib
    deliver_library: Path | None       # arms whose player has no library: the reviewer's copy is delivered along (in review_work/)
    pyshadow: Path | None              # posix
    fence: object | None               # None = bwrap; posix: run_arm.PosixFence
    cards_static: Path | None = None   # card writing on: the static-card (card_library.STATIC_DIR) and live-card directories
    cards_live: Path | None = None


def prepare_dirs(run_dir: Path, posix: bool) -> Path:
    """``reviews/``: under posix every reviewer uid must be able to traverse it (711); notices are root-only (700)."""
    root = run_dir / "reviews"
    root.mkdir(exist_ok=True)
    review_notice.notices_dir(run_dir).mkdir(parents=True, exist_ok=True)
    if posix:
        root.chmod(0o711)
        review_notice.notices_dir(run_dir).chmod(0o700)
    return root


def _driver_env(ctx: ReviewContext, ws: Path, name: str) -> Path:
    denv = ctx.driver.env(ws / ".driver", ws, ctx.paths)
    if ctx.library is not None:
        # as for the player: posix's PYTHONPATH must keep pyshadow (which masks the image's sitecustomize)
        denv["PYTHONPATH"] = str(ctx.library) if ctx.fence is None else f"{ctx.pyshadow}:{ctx.library}"
    for key, value in denv.items():
        if "\n" in key or "\n" in value or "=" in key:
            raise ValueError(f"driver env {key!r} cannot be one KEY=VALUE line")
    path = ctx.run_dir / "driver_env" / f"{name}.env"
    path.parent.mkdir(exist_ok=True)
    path.write_text("".join(f"{k}={v}\n" for k, v in denv.items()))
    return path


def _launcher(ctx: ReviewContext, ws: Path, env_file: Path, name: str):
    driver_files = [path for _, path in ctx.knobs.driver_files]
    if ctx.fence is None:
        env = reviewer.jail_env(ws, runtime=ctx.agent_runtime, paths=ctx.paths, sockets=ctx.sockets,
                                driver_env_file=env_file,
                                extra_ro=[*(str(m) for m in ctx.driver.mounts(ctx.paths)), *driver_files,
                                          *([str(ctx.library)] if ctx.library is not None else [])],
                                process_data_kb=ctx.process_data_kb, cpu_seconds=ctx.cpu_seconds,
                                model_port=ctx.model_port)
        return reviewer.bwrap_launcher(ws, env, unit_prefix=f"arcreview-{os.urandom(6).hex()}",
                                       memory_max=ctx.memory_max), None
    import posix_fence
    # shared fence-level settings (deny, pyshadow, oom) as for the player; the environment port removed (refused if given
    # to a reviewer), and the must-be-readable files replaced by the reviewer's own driver files
    env = {k: v for k, v in ctx.fence.env.items() if k != "ARCSEC_ENV_PORT"}
    env.update({"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "ARCSEC_ROLE": "reviewer",
                "ARCSEC_RUNTIME": str(ctx.agent_runtime), "ARCSEC_DRIVER_BIN": str(ctx.driver_bin),
                "ARCSEC_DRIVER_HOME": str(ws / ".driver"), "ARCSEC_DRIVER_ENV": str(env_file),
                "ARCSEC_PROCESS_DATA_KB": str(ctx.process_data_kb), "ARCSEC_RECORD": str(ws / "record"),
                "ARCSEC_MODEL_HEALTH_URL": ctx.model_health_url,
                "ARCSEC_FENCE_REQUIRE": ":".join(driver_files)})
    if ctx.library is not None:
        env["ARCSEC_AGENT_LIB"] = str(ctx.library)
    launcher = reviewer.PosixLauncher(ws, env, user=posix_fence.game_user(name), watchdog=ctx.fence.watchdog,
                                      label=name)
    return launcher, launcher


def run_review(ctx: ReviewContext, job: dict, stopping) -> dict:
    """One review; returns the result to record (``delivered`` is the delivered path or None, ``failure`` the reason or
    None)."""
    game, rid, player_ws = job["game"], job["review_id"], job["ws"]
    posix = ctx.fence is not None
    base = ctx.run_dir / "reviews" / game / rid
    if posix and os.lstat(player_ws / "record").st_uid != 0:
        # the record directory is written by the middleware (root); if it is not root's, the player replaced it
        # (fence_posix.sh would also refuse at the next session)
        raise RuntimeError(f"{player_ws}/record is not root's: the record was replaced")
    base.mkdir(parents=True)                                   # the same id is never opened twice
    if posix:
        base.parent.chmod(0o711)
        base.chmod(0o711)
    ws = base / "ws"
    meta = review.build_snapshot(player_ws, ws, link=True)
    # record/live/ and LIVE.json exist from the first moment (the brief mentions them); afterwards refreshed every
    # REFRESH_SECONDS in the launcher's polling
    live = {"status": review.refresh_snapshot(player_ws, ws, meta["step"]), "at": time.monotonic(), "count": 1}
    if posix:
        ws.chmod(0o700)                                        # nobody else gets in before fence_posix hands it to the reviewer uid
    out: dict = {"snapshot_step": meta["step"],
                 "snapshot": {k: meta[k] for k in ("step", "level", "levels_completed", "win_levels", "state",
                                                   "record_files", "code_bytes", "skipped")}}
    reviewer.prepare(ws, ctx.pset, meta, game, ctx.limits)
    if ctx.cards_live is not None:
        out["cards"] = card_library.assemble(ws / "game_cards", game, static=ctx.cards_static, live=ctx.cards_live)
    ctx.driver.prepare(ws / ".driver", ws, ctx.knobs)
    name = f"rv-{game}-{rid}"
    env_file = _driver_env(ctx, ws, name)
    launch, closer = _launcher(ctx, ws, env_file, name)

    history = player_ws / "record" / "history.jsonl"
    moved = {"checked": 0.0, "left": False}

    def left_level() -> bool:
        # the player left the level during the review (e2e2: level 1 was reviewed for 13 minutes, and on delivery the
        # player was on level 3): a review about this level is mostly void, so stop and give the slot to games still
        # stuck. Reads history once per LEFT_LEVEL_SECONDS
        now = time.monotonic()
        if not moved["left"] and now - moved["checked"] >= LEFT_LEVEL_SECONDS:
            moved["checked"] = now
            current = review.stall(history, time.time())
            moved["left"] = current is not None and current["level"] != meta["level"]
        return moved["left"]

    def refresh() -> None:
        # the refresh blew up (the player replaced the record, a step is missing): this review is void. Do not raise
        # inside the polling -- bwrap's launcher would leave an uncollected scope -- but record the reason and take the
        # budget path so the launcher kills it as usual; afterwards it is recorded as failed (no delivery)
        now = time.monotonic()
        if live.get("error") is None and now - live["at"] >= REFRESH_SECONDS:
            live["at"] = now
            try:
                live["status"] = review.refresh_snapshot(player_ws, ws, live["status"]["step"])
                live["count"] += 1
            except Exception as error:  # noqa: BLE001 — the cause goes into failure and the log;
                                        # this review is not delivered
                live["error"] = f"{type(error).__name__}: {error}"[:600]
                log(f"{game} {rid} snapshot refresh failed, stopping the review:\n{traceback.format_exc()[-2000:]}")

    def polled(over) -> bool:
        refresh()
        return live.get("error") is not None or stopping() or left_level() or over()

    def guarded(attempt, argv, left, over):
        if left_level() or live.get("error") is not None:
            return reviewer.TOKEN_BUDGET_CODE, "left-level"          # open no further stage
        # wrap-up (global cancel or scheduler stop) and the player leaving the level both take the budget path: the
        # launcher kills at its next poll
        return launch(attempt, argv, left, lambda: polled(over))

    try:
        result = reviewer.run(ws, driver=ctx.driver, knobs=ctx.knobs, pset=ctx.pset, launch=guarded,
                              deadline=job["deadline"], output_tokens=ctx.spec["output_tokens"])
    finally:
        if closer is not None:
            closer.close()
    # delivery and re-review go by the last step the reviewer could see (how far the snapshot was moved), not the step
    # at launch
    seen = {**meta, "step": live["status"]["step"]}
    out.update(output_tokens=result["output_tokens"], sessions=result["sessions"], report=result["report"],
               delivered=None, failure=None, snapshot_step=seen["step"],
               live={"refreshes": live["count"], "step": seen["step"], "error": live.get("error")})
    (base / "result.json").write_text(json.dumps({"job": {k: v for k, v in job.items() if k != "ws"} | {"ws": str(player_ws)},
                                                  "meta": meta, "live": live, "left_level": moved["left"], **result},
                                                 indent=1, default=str))
    if live.get("error") is not None:
        out["failure"] = f"snapshot refresh failed: {live['error']}"
        return out
    if moved["left"]:
        out["failure"] = f"the player left level {meta['level']} during the review"
        return out
    if not result["report"]:
        out["failure"] = "cancelled before REVIEW.md" if stopping() else "no REVIEW.md"
        return out
    text = review.report_text(ws / review.REPORT)
    (base / review.REPORT).write_text(text, encoding="utf-8")   # the delivered copy, kept outside the workspace
    current = review.stall(player_ws / "record" / "history.jsonl", time.time())
    problem = claim_delivery_dir(player_ws / review.DELIVERY_DIR, posix)
    if problem is not None:
        out["failure"] = problem
        return out
    delivered = review.deliver(text, player_ws, seen, current, review_id=rid, reviewer_ws=ws,
                               library=ctx.deliver_library, tail=ctx.pset.text.get("delivery_tail.md"))
    # posix: the player workspace is 700 and owned by the player uid; the report and frozen workspace root wrote into it
    # stay root's, readable but not modifiable by the player
    for path in (delivered["report"], delivered["frozen"]):
        if path is not None:
            seal(path, posix)
    review_notice.write_notice(ctx.run_dir, game, rid, review.notice(delivered["report"], seen))
    out.update(delivered=str(delivered["report"]), frozen=None if delivered["frozen"] is None else str(delivered["frozen"]),
               work=delivered["work"], current_step=None if current is None else current["step"])
    return out


@dataclasses.dataclass(frozen=True)
class CardPrompt:
    """``review_prompts/<name>/card.md``: the single-request card-writing prompt, whose ``{}`` fields are exactly
    ``CARD_FIELDS``."""
    name: str
    text: str
    fingerprint: str


def load_card_prompt(name: str) -> CardPrompt:
    path = reviewer.PROMPT_ROOT / name / CARD_PROMPT
    text = path.read_text(encoding="utf-8")
    fields = {field for _, field, _, _ in string.Formatter().parse(text) if field is not None}
    if fields != set(CARD_FIELDS):
        raise ValueError(f"{path}: fields {sorted(fields)}, want exactly {sorted(CARD_FIELDS)}")
    return CardPrompt(name, text, hashlib.sha256(text.encode()).hexdigest()[:16])


@dataclasses.dataclass(frozen=True)
class CardContext:
    run_dir: Path
    spec: dict                         # the cards block from check_cards_spec
    prompt: CardPrompt
    model_base_url: str                # run_arm's --model-base-url (the model proxy: sampling parameters overridden as for the arm)
    model: str
    cards_static: Path
    cards_live: Path
    posix: bool


def card_documents(ws: Path, meta: dict, previous: str | None, example: str) -> str:
    """The documents in a card-writing request, read from the snapshot (build_snapshot: root's copy, player symlinks not
    followed, size caps on code and documents): LEVEL_STATS computed from the record, ``levels/LNN.md`` of each cleared
    level, ``notes.md``, this game's previous live card, and the example card. Documents of the level still in play are
    left out."""
    stats = card_library.level_stats(review.read_history(ws / "record" / "history.jsonl"))
    parts = [f"<level_stats>\n{stats.strip()}\n</level_stats>"]
    for n in range(1, meta["levels_completed"] + 1):
        doc = ws / "levels" / f"L{n:02d}.md"
        body = (doc.read_text(encoding="utf-8", errors="replace").strip() if doc.is_file()
                else "(the player wrote no document for this level)")
        parts.append(f'<level_document level="{n}">\n{body}\n</level_document>')
    notes = ws / "notes.md"
    parts.append("<notes>\n" + (notes.read_text(encoding="utf-8", errors="replace").strip() if notes.is_file()
                                else "(no notes.md)") + "\n</notes>")
    if previous is not None:
        parts.append(f"<previous_card>\n{previous.strip()}\n</previous_card>")
    parts.append(f"<example_card>\n{example.strip()}\n</example_card>")
    return "\n\n".join(parts)


def unfence(text: str) -> tuple[str, bool]:
    """The prompt says not to, but the model occasionally wraps the whole card in a pair of ```: strip only that
    outermost pair (returning whether it did), touch nothing else."""
    s = text.strip()
    if s.startswith("```") and s.endswith("```") and s.count("```") == 2 and "\n" in s:
        return s.split("\n", 1)[1].rsplit("```", 1)[0].strip(), True
    return s, False


def _chat(base_url: str, body: dict, timeout: float) -> dict:
    """One ``/chat/completions`` call (not streamed). HTTP errors are raised together with the response body."""
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP {error.code} from {base_url}: {error.read()[:500]!r}") from error


def run_card(ctx: CardContext, job: dict, stopping) -> dict:
    """One card write (the user 2026-10-31: no codex, a single request, no thinking): snapshot -> documents assembled
    into ``ctx.prompt`` -> the model returns the whole card in one go -> published to ``ctx.cards_live`` if the format
    is right. Request, response and card are kept in ``cards/<game>/Cnn/``. Returns the result to record."""
    game, cid, player_ws = job["game"], job["card_id"], job["ws"]
    if ctx.posix and os.lstat(player_ws / "record").st_uid != 0:
        # the record directory is written by the middleware (root); if it is not root's, the player replaced it
        raise RuntimeError(f"{player_ws}/record is not root's: the record was replaced")
    base = ctx.run_dir / "cards" / game / cid
    base.mkdir(parents=True)                                   # the same id is never opened twice
    meta = review.build_snapshot(player_ws, base / "ws", link=True)
    out: dict = {"snapshot_step": meta["step"], "levels_completed": meta["levels_completed"], "output_tokens": 0,
                 "delivered": None, "failure": None}
    previous = ctx.cards_live / f"{game}.md"
    documents = card_documents(base / "ws", meta, previous.read_text(encoding="utf-8") if previous.is_file() else None,
                               (ctx.cards_static / EXAMPLE_CARD).read_text(encoding="utf-8"))
    out["input_chars"] = len(documents)
    if len(documents) > CARD_INPUT_MAX_CHARS:
        out["failure"] = f"the documents are {len(documents):,} chars, over {CARD_INPUT_MAX_CHARS:,}: not sent"
        return out
    prompt = ctx.prompt.text.format_map({"game": game, "levels_completed": meta["levels_completed"],
                                         "win_levels": meta["win_levels"], "level": meta["level"],
                                         "step": meta["step"], "documents": documents})
    left = min(ctx.spec["wall_seconds"], job["deadline"] - time.monotonic())
    if stopping() or left < MIN_CARD_SECONDS:
        out["failure"] = "stopping before the request" if stopping() else f"only {left:.0f}s left for the request"
        return out
    body = {"model": ctx.model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": ctx.spec["output_tokens"], "chat_template_kwargs": {"enable_thinking": False}}
    (base / "request.json").write_text(json.dumps(body, indent=1))
    t0 = time.monotonic()
    try:
        response = _chat(ctx.model_base_url, body, left)
    except (OSError, RuntimeError, ValueError) as error:       # unreachable / timeout is OSError, HTTP error RuntimeError,
                                                               # bad JSON ValueError
        out["failure"] = f"request failed after {time.monotonic() - t0:.0f}s: {type(error).__name__}: {error}"[:600]
        return out
    out["request_seconds"] = round(time.monotonic() - t0, 1)
    (base / "response.json").write_text(json.dumps(response, indent=1))
    choice, usage = response["choices"][0], response["usage"]
    out.update(prompt_tokens=usage["prompt_tokens"], output_tokens=usage["completion_tokens"],
               finish_reason=choice["finish_reason"],
               reasoning_chars=len(choice["message"].get("reasoning_content") or ""))
    if choice["finish_reason"] != "stop":
        out["failure"] = (f"the reply ended with {choice['finish_reason']!r} (max_tokens {ctx.spec['output_tokens']}): "
                          "a cut card is not published")
        return out
    text, out["unfenced"] = unfence(choice["message"]["content"] or "")
    problem = card_library.check_card(text)
    if problem is not None:
        out["failure"] = f"the reply is not a card: {problem}"
        return out
    (base / CARD_FILE).write_text(text, encoding="utf-8")
    card = card_library.publish(ctx.cards_live, game, text, {"levels_completed": meta["levels_completed"],
                                                             "win_levels": meta["win_levels"], "step": meta["step"],
                                                             "card": cid})
    out["delivered"] = str(card)
    return out


def claim_delivery_dir(path: Path, posix: bool) -> str | None:
    """Before delivery: ``review/`` in the player workspace must be a real directory with no symlinks inside (the player
    takes back the whole workspace at the start of every session and can write into it). On posix, first reclaim it for
    root (via fd, not following symlinks), so only root can add things afterwards. Returns the reason delivery is
    impossible, if any."""
    try:
        os.mkdir(path, 0o755)
    except FileExistsError:
        pass
    try:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    except OSError as error:
        return f"{path} is not a plain directory ({error.strerror})"
    try:
        if posix:
            os.fchown(fd, 0, 0)
        os.fchmod(fd, 0o755)
        links = [name for name in os.listdir(fd) if stat.S_ISLNK(os.stat(name, dir_fd=fd, follow_symlinks=False).st_mode)]
    finally:
        os.close(fd)
    return f"{path} holds symlinks {links[:5]}" if links else None


def seal(path: Path, posix: bool) -> None:
    """``chown -R root`` (posix) + ``chmod -R a+rX,go-w``, without following symlinks (delivered content has none
    anyway)."""
    def fix(p: str) -> None:
        st = os.lstat(p)
        if stat.S_ISLNK(st.st_mode):
            raise RuntimeError(f"{p} is a symlink in a delivered tree")
        if posix and (st.st_uid, st.st_gid) != (0, 0):
            os.chown(p, 0, 0, follow_symlinks=False)
        mode = stat.S_IMODE(st.st_mode)
        want = (mode | 0o444 | (0o111 if stat.S_ISDIR(st.st_mode) or mode & 0o100 else 0)) & ~0o022
        if want != mode:
            os.chmod(p, want)

    fix(str(path))
    if path.is_dir():
        for root, dirs, files in os.walk(path):
            for name in dirs + files:
                fix(os.path.join(root, name))


def card_recorder(supervisor, run_dir: Path, posix: bool):
    """``card_record`` for card writing: ARM_STATE's ``cards`` (one row per write + counts) and ``cards/cards.jsonl``
    (kept separate from the review records)."""
    path = run_dir / "cards" / "cards.jsonl"
    lock = threading.Lock()

    def record(row: dict, summary: dict) -> None:
        with lock:
            with path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            if posix:
                path.chmod(0o600)
        with supervisor.lock:
            state = supervisor.state["cards"]
            state["rows"].append(row)
            state["counts"] = summary.get("cards", {})
            supervisor.update()

    return record


def recorder(supervisor, run_dir: Path, posix: bool):
    """``record`` callback: ARM_STATE's ``reviews`` (one row per review + counts) and ``reviews/reviews.jsonl``."""
    path = run_dir / "reviews" / "reviews.jsonl"
    lock = threading.Lock()

    def record(row: dict, summary: dict) -> None:
        with lock:
            with path.open("a") as f:
                f.write(json.dumps(row, default=str) + "\n")
            if posix:
                path.chmod(0o600)
        with supervisor.lock:
            state = supervisor.state["reviews"]
            state["rows"].append(row)
            state["counts"] = summary
            supervisor.update()

    return record
