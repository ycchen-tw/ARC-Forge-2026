"""One review by the idle-slot read-only reviewer: prompts, workspace, running codex inside the fence
(docs/plans/arc3_readonly_reviewer_20261014.md).

One review = a snapshot (``arcforge.arc_code.review.build_snapshot``) + the reviewer's own brief (``AGENTS.md``) +
one codex thread. The reviewer has no game: the fence uses ``ARCSEC_ROLE=reviewer`` (no act.py mounted, no token, no
environment bridge), and the snapshot's ``record/`` is mounted back read-only (sharing inodes with the main game). The
only output is ``REVIEW.md`` in the workspace.

If exploration uses up ``explore_fraction`` of the budget without a finished ``REVIEW.md``, send ``finish.md`` in the
same thread; if there is still none, send ``last.md`` at low effort (``run``). Output tokens are counted from codex's
rollout (the same rule as run_arm's ``GameOutputTokens``), with a kill when the budget is reached.
Two fences: ``bwrap_launcher`` (jail.sh, B200 / PC) and ``PosixLauncher`` (fence_posix.sh, Kaggle).

Scheduling (who is reviewed, and when) lives in ``review_scheduler.py``; this file only handles how one review runs.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import signal
import string
import subprocess
import time
from pathlib import Path

UNIT = Path(__file__).resolve().parent
PROMPT_ROOT = UNIT / "review_prompts"
FILES = ("brief.md", "kickoff.md", "finish.md", "developer.md", "instructions.md", "compact_prompt.md")
#: optional: ``last.md`` (from r2) -- when there is still no report after the wrap-up, call once more at low effort,
#: allowed only to write the report; ``delivery_tail.md`` (from r8) -- on delivery, inserted before the last sentence
#: of the harness's paragraph (review.delivery_header)
OPTIONAL = ("last.md", "delivery_tail.md")
#: effort for the last step: it only writes down what is already known, no more thinking needed (Qwen3.8's template
#: accepts low / medium / xhigh)
LAST_EFFORT = "low"
#: guaranteed output tokens for the last step: one long reasoning stretch in the wrap-up can use up the budget (Kaggle
#: fence smoke run with the real model, budget 30k: the wrap-up shot to 30,503 in one go, no report), so the last step
#: still runs, capped at max(budget, used + this much)
LAST_TOKENS = 6000
#: guaranteed time for the last step: the wrap-up stops at the latest this long before the deadline (e2e3: a review
#: opened only 8 minutes before wrap-up had a ~470 s window; at 92% the wrap-up had only 38 s left, short of the last
#: step's 60 s, and two reviews had no report)
LAST_SECONDS = 90.0
BRIEF_VARIABLES = frozenset({"game", "step", "level", "levels_completed", "win_levels",
                             "process_limit", "cpu_limit"})
#: codex settings for the player only: the reviewer's driver home has no compact_reminder file and should not be
#: reminded to write notes.md
PLAYER_ONLY_CONFIG = ("features.pre_compact_reminder", "developer_instructions")
#: codex settings of the gen-2 player (run_arm.gen2_driver_config): the hooks would assemble a status pack in the
#: reviewer's workspace, block it from ending, and hook_error on a missing gen2_budget.json; token_budget makes window
#: changes skip the summary. The reviewer wants none of it and runs with ordinary codex compaction.
PLAYER_ONLY_PREFIXES = ("hooks.", "features.token_budget.")
POLL_SECONDS = 5.0
TOKEN_BUDGET_CODE = 125
REPORT_MIN_CHARS = 200


class ReviewPrompts:
    """``review_prompts/<name>/``: all six files must exist; the brief's variables must match ``BRIEF_VARIABLES``
    exactly."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.root = PROMPT_ROOT / name
        if not self.root.is_dir():
            raise FileNotFoundError(f"no reviewer prompt set {self.root}")
        self.text = {}
        for f in FILES + tuple(f for f in OPTIONAL if (self.root / f).exists()):
            body = (self.root / f).read_text(encoding="utf-8")
            if not body.strip():
                raise ValueError(f"{self.root / f} is empty")
            self.text[f] = body
        used = {field for _, field, _, _ in string.Formatter().parse(self.text["brief.md"]) if field}
        if used != BRIEF_VARIABLES:
            raise ValueError(f"{self.root}/brief.md uses {sorted(used)}, expected {sorted(BRIEF_VARIABLES)}")
        for f in self.text:
            if f != "brief.md" and any(field for _, field, _, _ in string.Formatter().parse(self.text[f])):
                raise ValueError(f"{self.root / f} must not have template fields")
        digest = hashlib.sha256()
        for f in sorted(self.text):
            digest.update(f.encode() + b"\0" + self.text[f].encode() + b"\0")
        self.fingerprint = digest.hexdigest()[:16]

    def brief(self, **values) -> str:
        if set(values) != BRIEF_VARIABLES:
            raise ValueError(f"brief values {sorted(values)} != {sorted(BRIEF_VARIABLES)}")
        return self.text["brief.md"].format(**values)

    def driver_files(self) -> dict[str, Path]:
        return {"model_instructions_file": self.root / "instructions.md",
                "experimental_compact_prompt_file": self.root / "compact_prompt.md"}


def reviewer_knobs(player, pset: ReviewPrompts, auto_compact: int | None = None):
    """Turn the player's knobs into the reviewer's: the same model, the same effort and context settings; base
    instructions, compaction prompt and developer rules replaced by the reviewer's, player-only settings removed
    (gen-2's hooks, token_budget and hook trust too). ``tool_output_tokenizer`` is kept. ``auto_compact``: the config's
    auto_compact_token_limit (the gen-2 player's knobs add the window-change last line of defense; the reviewer does
    not change windows and uses ordinary compaction); when not given, keep the player's."""
    files = {k: v for k, v in player.driver_files if k not in pset.driver_files()}
    files.update({k: str(v) for k, v in pset.driver_files().items()})
    config = [(k, v) for k, v in player.driver_config
              if k not in PLAYER_ONLY_CONFIG and not k.startswith(PLAYER_ONLY_PREFIXES)]
    config.append(("developer_instructions", json.dumps(pset.text["developer.md"])))
    return dataclasses.replace(player, driver_files=tuple(sorted(files.items())),
                               driver_config=tuple(sorted(config)), hook_trust_bypass=False,
                               auto_compact=player.auto_compact if auto_compact is None else auto_compact)


def prepare(ws: Path, pset: ReviewPrompts, meta: dict, game: str, limits: dict[str, str]) -> str:
    """Turn the snapshot directory into the reviewer's workspace: write the brief, create scratch. Returns the full
    brief (also AGENTS.md)."""
    brief = pset.brief(game=game, step=meta["step"], level=meta["level"],
                       levels_completed=meta["levels_completed"], win_levels=meta["win_levels"],
                       process_limit=limits["process_limit"], cpu_limit=limits["cpu_limit"])
    if (ws / "AGENTS.md").exists():
        raise FileExistsError(ws / "AGENTS.md")
    (ws / "AGENTS.md").write_text(brief, encoding="utf-8")
    (ws / "review_work").mkdir()
    return brief


class OutputTokens:
    """Sum of output_tokens over every distinct ``response_id`` in the codex rollouts (same as run_arm.GameOutputTokens)."""

    def __init__(self, ws: Path) -> None:
        self.root = ws / ".driver" / "sessions"
        self.offsets: dict[Path, int] = {}
        self.seen: set[str] = set()
        self.total = 0

    def read(self) -> int:
        for path in sorted(self.root.glob("**/rollout*.jsonl")):
            start = self.offsets.get(path, 0)
            with path.open("rb") as f:
                f.seek(start)
                data = f.read()
            end = data.rfind(b"\n") + 1
            self.offsets[path] = start + end
            for line in data[:end].splitlines():
                if b'"token_usage_record"' not in line:
                    continue
                row = json.loads(line)
                if row.get("type") != "token_usage_record":
                    continue
                payload = row["payload"]
                if payload["response_id"] not in self.seen:
                    self.seen.add(payload["response_id"])
                    self.total += int(payload["usage"]["output_tokens"])
        return self.total


def report_ready(ws: Path) -> bool:
    path = ws / "REVIEW.md"
    return path.is_file() and len(path.read_text(encoding="utf-8", errors="replace").strip()) >= REPORT_MIN_CHARS


def _handle(driver, log: Path) -> str | None:
    import arms
    return arms.session_handle(driver, log) if log.exists() else None


def run(ws: Path, *, driver, knobs, pset: ReviewPrompts, launch, deadline: float, output_tokens: int,
        explore_fraction: float = 0.75, finish_fraction: float = 0.17, min_session_seconds: float = 60.0) -> dict:
    """Run one review: explore -> wrap-up -> (if there is still no report and the prompt set has ``last.md``) last step.

    All three stages run in the same thread, each capped at a cumulative fraction of output tokens and time:
    exploration up to ``explore_fraction``, wrap-up up to ``explore_fraction + finish_fraction``, the last step up to
    the whole. If exploration ends normally by itself with the report written, done; otherwise send ``finish.md``.
    r1's 24 offline reviews: with 80% exploration / 20% wrap-up, 4 had no report -- the wrap-up stage did not do as
    told and kept verifying (one reasoning stretch of 17k characters) -- so from r2 a further low-effort ``last.md``,
    allowed only to write the report, was added. The last step's token cap is at least used + ``LAST_TOKENS`` (the
    budget is a soft cap; going a little over beats wasting the whole review), and the wrap-up stops at the latest
    ``LAST_SECONDS`` before the deadline.

    ``launch(attempt, argv, left, over) -> (returncode, unit_result)``: run one session inside the fence, killed when
    ``over()`` is true or ``left`` seconds pass (returns TOKEN_BUDGET_CODE / 124).
    """
    if not (0 < explore_fraction < 1 and 0 < finish_fraction and explore_fraction + finish_fraction <= 1):
        raise ValueError(f"fractions explore {explore_fraction} finish {finish_fraction}")
    tokens = OutputTokens(ws)
    began = time.monotonic()
    window = deadline - began
    sessions = []

    def until(fraction: float) -> tuple[int, float]:
        return int(output_tokens * fraction), began + fraction * window

    def session(attempt: int, phase: str, argv: list[str], limit_tokens: int, end: float) -> tuple[int, str | None]:
        t0 = time.monotonic()
        code, result = launch(attempt, argv, end - t0, lambda: tokens.read() >= limit_tokens)
        log = ws / ("driver.log" if attempt == 1 else f"driver.{attempt}.log")
        handle = _handle(driver, log)
        sessions.append({"attempt": attempt, "phase": phase, "returncode": code, "unit_result": result,
                         "seconds": round(time.monotonic() - t0, 1), "handle": handle,
                         "output_tokens": tokens.read(), "report": report_ready(ws)})
        return code, handle

    def room(limit_tokens: int, end: float) -> bool:
        return end - time.monotonic() >= min_session_seconds and tokens.read() < limit_tokens

    if not room(*until(explore_fraction)):
        raise ValueError(f"review window {window:.0f}s is too short")
    code, handle = session(1, "explore", driver.argv(pset.text["kickoff.md"], knobs), *until(explore_fraction))
    if code not in (TOKEN_BUDGET_CODE, 124) and report_ready(ws):
        handle = None                                  # exploration finished by itself: no more calls
    finish_until = min(1.0, explore_fraction + finish_fraction)
    finish_tokens, finish_end = until(finish_until)
    if "last.md" in pset.text:
        finish_end = min(finish_end, deadline - LAST_SECONDS)
    if handle is not None and room(finish_tokens, finish_end):
        _, resumed = session(len(sessions) + 1, "finish", driver.resume_argv(handle, pset.text["finish.md"], knobs),
                             finish_tokens, finish_end)
        handle = resumed or handle          # the wrap-up never came up (fence or resume died early, no thread id
                                            # printed): the last step still continues the same thread
    # a single response in exploration or wrap-up can overshoot the next stage's cap (posix smoke run: exploration with
    # a 22.5k line stopped at 27,643, leaving the wrap-up no room); the last step does not care whether the wrap-up ran,
    # and its cap is guaranteed at used + LAST_TOKENS
    last = handle is not None and not report_ready(ws) and "last.md" in pset.text and finish_until < 1
    last_limit = (max(output_tokens, tokens.read() + LAST_TOKENS), deadline)
    if last and room(*last_limit):
        low = dataclasses.replace(knobs, effort=LAST_EFFORT)
        session(len(sessions) + 1, "last", driver.resume_argv(handle, pset.text["last.md"], low), *last_limit)
    return {"sessions": sessions, "output_tokens": tokens.read(), "report": report_ready(ws),
            "seconds": round(time.monotonic() - began, 1)}


# ---------------------------------------------------------------- bwrap fence (B200, PC)

def jail_env(ws: Path, *, runtime: Path, paths, sockets: Path, driver_env_file: Path, extra_ro: list[str],
             process_data_kb: int, cpu_seconds: int | None, model_port: int) -> dict[str, str]:
    """jail.sh's resource declaration for the reviewer. No ARCSEC_ACT / ARCSEC_TOKEN / ARCSEC_ENV_PORT (jail.sh refuses
    them if given).

    ``runtime`` is the agent venv's root (``parents[1]`` of ``--agent-python``, **not resolved**):
    ``Paths.agent_runtime`` is resolved and follows the symlink to uv's base interpreter, which has no numpy (run_arm's
    ARCSEC_RUNTIME is not resolved either)."""
    env = {
        "PATH": "/usr/bin:/bin",
        "ARCSEC_ROLE": "reviewer",
        "ARCSEC_RUNTIME": str(runtime),
        "ARCSEC_SOCKETS": str(sockets),
        "ARCSEC_BRIDGE": str(UNIT.parents[1] / "src/arcforge/arc_code/bridge.py"),
        "ARCSEC_DRIVER_BIN": str(paths.driver_bin),
        "ARCSEC_DRIVER_HOME": str(ws / ".driver"),
        "ARCSEC_DRIVER_ENV": str(driver_env_file),
        "ARCSEC_EXTRA_RO": ":".join(extra_ro),
        "ARCSEC_PROCESS_DATA_KB": str(process_data_kb),
        "ARCSEC_MODEL_PORT": str(model_port),
        "ARCSEC_RECORD": str(ws / "record"),
        "ARCSEC_JAIL_PROC": os.environ.get("ARCSEC_JAIL_PROC", "new"),
        "XDG_RUNTIME_DIR": os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"),
    }
    if cpu_seconds:
        env["ARCSEC_CPU_SECONDS"] = str(cpu_seconds)
    if "DBUS_SESSION_BUS_ADDRESS" in os.environ:
        env["DBUS_SESSION_BUS_ADDRESS"] = os.environ["DBUS_SESSION_BUS_ADDRESS"]
    return env


def bwrap_launcher(ws: Path, env: dict[str, str], *, unit_prefix: str, memory_max: str):
    """Returns the launch for ``run``: one session is jail.sh in a systemd user scope."""

    def launch(attempt: int, argv: list[str], left: float, over) -> tuple[int, str]:
        unit = f"{unit_prefix}-s{attempt}"
        cmd = ["systemd-run", "--user", "--scope", "--quiet", f"--unit={unit}",
               f"--property=MemoryMax={memory_max}", "--property=MemorySwapMax=0",
               f"--property=RuntimeMaxSec={left:.0f}s", "--property=TimeoutStopSec=15s",
               "/bin/bash", str(UNIT / "jail.sh"), str(ws), *argv]
        log = ws / ("driver.log" if attempt == 1 else f"driver.{attempt}.log")
        with log.open("w") as out:
            proc = subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            end = time.monotonic() + left
            code = None
            while code is None:
                try:
                    code = proc.wait(timeout=POLL_SECONDS)
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= end:
                        code = 124
                    elif over():
                        code = TOKEN_BUDGET_CODE
                    if code is not None:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=30)
        stop = subprocess.run(["systemctl", "--user", "stop", f"{unit}.scope"], capture_output=True, text=True)
        result = "stopped" if stop.returncode == 0 else "gone"
        return code, result

    return launch


# ---------------------------------------------------------------- posix fence (Kaggle)

class PosixLauncher:
    """The launch for ``run``: fence_posix.sh in the reviewer role, one dedicated uid per review (as in
    run_arm._session_posix).

    The three things a systemd scope does each have a stand-in here: memory = RLIMIT_DATA + the watchdog summing per
    uid; time = polling here; reaping = killing everything of this uid when each session ends. The player's posix
    sessions do not poll output tokens (game_output_tokens works only with bwrap), but the reviewer must: ``over()`` is
    asked every ``POLL_SECONDS``. After the review, call ``close()``: kill everything, and the watchdog stops watching
    this uid.
    """

    def __init__(self, ws: Path, env: dict[str, str], *, user: str, watchdog, label: str) -> None:
        import posix_fence
        self.posix_fence = posix_fence
        if len(user) > 32:
            raise ValueError(f"user name {user} longer than 32")
        if env.get("ARCSEC_ROLE") != "reviewer":
            raise ValueError("PosixLauncher is for ARCSEC_ROLE=reviewer")
        self.ws, self.env, self.user, self.watchdog, self.label = ws, env, user, watchdog, label
        self.uid: int | None = None

    def _ensure_user(self) -> int:
        if self.uid is None:
            uid = self.posix_fence.uid_of(self.user)
            if uid is None:
                subprocess.run(["useradd", "-M", "-N", "-s", "/bin/bash", self.user], check=True, capture_output=True)
                uid = self.posix_fence.uid_of(self.user)
            if uid is None or uid == 0:
                raise RuntimeError(f"no usable uid for {self.user}: {uid!r}")
            self.uid = uid
            self.watchdog.watch(uid, self.label)
        if self.watchdog.error is not None or not self.watchdog.is_alive():
            raise RuntimeError(f"posix watchdog is down: {self.watchdog.error!r}")
        return self.uid

    def __call__(self, attempt: int, argv: list[str], left: float, over) -> tuple[int, str]:
        uid = self._ensure_user()
        fenv = {**self.env, "ARCSEC_GAME_USER": self.user, "ARCSEC_FENCE_SCAN": "1" if attempt == 1 else ""}
        log = self.ws / ("driver.log" if attempt == 1 else f"driver.{attempt}.log")
        since = time.time()
        code = None
        with log.open("w") as out:
            proc = subprocess.Popen(["/bin/bash", str(UNIT / "fence_posix.sh"), str(self.ws), *argv], env=fenv,
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            end = time.monotonic() + left
            try:
                while code is None:
                    try:
                        code = proc.wait(timeout=max(0.0, min(POLL_SECONDS, end - time.monotonic())))
                    except subprocess.TimeoutExpired:
                        if time.monotonic() >= end:
                            code = 124
                        elif over():
                            code = TOKEN_BUDGET_CODE
            finally:
                self.posix_fence.kill_user(uid)
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=30)
        if self.watchdog.oom_killed_since(self.label, since):
            return code, "oom-kill"
        if code == 124:
            return code, "timeout"
        if code == TOKEN_BUDGET_CODE:
            return code, "token-budget"
        return code, "success" if code == 0 else "exit-code"

    def close(self) -> None:
        if self.uid is not None:
            self.posix_fence.kill_user(self.uid)
            self.watchdog.unwatch(self.uid)
            self.uid = None
