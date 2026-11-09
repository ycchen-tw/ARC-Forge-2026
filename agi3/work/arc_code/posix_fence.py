"""The launcher side of ``fence: posix``: one uid per game, cleanup after each session, memory watchdog.

In bwrap mode these three are done by the systemd scope (``MemoryMax``, ``RuntimeMaxSec``, the whole
cgroup reaped when the scope stops). Kaggle has no systemd, a read-only ``/sys/fs/cgroup`` and no pid
namespace, so uids take their place: one user per game, and "all processes of this game" is "all
processes of this uid".

- :func:`kill_user`: after a session ends (or times out), kill everything of this uid -- orphans of
  ``nohup … &`` included.
- :class:`Watchdog`: reads ``/proc`` every ``interval`` seconds.
  * A game's (uid's) total RSS above ``game_memory_max`` -> kill that whole game, recorded as
    ``oom-kill``, so the launcher takes the same path as when the scope is OOM-killed (reattach,
    with oom.md).
  * Container free memory below ``headroom`` -> kill the game with the largest RSS. In v4 (2026-09-23)
    an agent's script filled the 175 GiB container and the OOM killer picked sglang's scheduler.
  * A single process's CPU time above ``cpu_seconds`` -> kill only that one (a runaway search
    script); the session stays.

The old arm's watchdog (``Watchdog`` in ``work/arc_agi3/arc_code_qwen/run_arm.py``) is the prototype
of this one; it grouped by the fence user name ``arc-<game>``, this uses the uid directly.
"""

from __future__ import annotations

import hashlib
import json
import os
import pwd
import signal
import threading
import time
from pathlib import Path

PAGE_KB = os.sysconf("SC_PAGE_SIZE") // 1024
CLOCK = os.sysconf("SC_CLK_TCK")


def game_user(game: str) -> str:
    """Linux user names are capped at 32 chars; beyond that use a hash, not truncation (truncation could make two games collide into one user)."""
    name = f"arc-{game}"
    if len(name) <= 32 and all(c.isalnum() or c in "-_" for c in game):
        return name
    return "arc-" + hashlib.sha256(game.encode()).hexdigest()[:16]


def uid_of(user: str) -> int | None:
    try:
        return pwd.getpwnam(user).pw_uid
    except KeyError:
        return None


def _procs():
    """(pid, uid, rss_kb, cpu_seconds, comm). Processes vanishing mid-read are normal; skipped."""
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            with open(f"/proc/{pid}/stat") as f:
                stat = f.read()
            status = Path(f"/proc/{pid}/status").read_text()
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        comm = stat[stat.index("(") + 1:stat.rindex(")")]
        fields = stat[stat.rindex(")") + 2:].split()
        cpu = (int(fields[11]) + int(fields[12])) / CLOCK
        uid = rss = None
        for line in status.splitlines():
            if line.startswith("Uid:"):
                uid = int(line.split()[1])
            elif line.startswith("VmRSS:"):
                rss = int(line.split()[1])
        if uid is None:
            continue
        yield pid, uid, rss or 0, cpu, comm


def kill_user(uid: int, grace: float = 5.0) -> int:
    """SIGTERM every process of this uid, wait ``grace`` seconds, SIGKILL the rest. Returns how many were killed."""
    if uid == 0:
        raise ValueError("refusing to kill uid 0")
    killed = 0
    for sig, wait in ((signal.SIGTERM, grace), (signal.SIGKILL, 0.0)):
        pids = [pid for pid, u, *_ in _procs() if u == uid]
        for pid in pids:
            try:
                os.kill(pid, sig)
                killed += sig == signal.SIGTERM
            except ProcessLookupError:
                pass
        if not pids:
            break
        deadline = time.monotonic() + wait
        while wait and time.monotonic() < deadline and any(u == uid for _, u, *_ in _procs()):
            time.sleep(0.2)
    return killed


def cgroup_headroom_kb(root: Path = Path("/sys/fs/cgroup"),
                       self_cgroup: Path = Path("/proc/self/cgroup"),
                       meminfo: Path = Path("/proc/meminfo")) -> int:
    """What the kernel can still hand out before its OOM killer runs, in KiB.

    **Anonymous memory and shmem count; page cache does not** -- the kernel
    reclaims cache before it kills anything. The old arm's watchdog says exactly
    this (work/arc_agi3/arc_code_qwen/run_arm.py); the first port of it here used
    ``memory.max - memory.current`` instead, and ``memory.current`` includes the
    checkpoint the notebook deliberately keeps in the page cache (rewarm). On the
    Kaggle v2 run (2026-09-25) that read as < 12 GiB of headroom from the first
    tick, so the watchdog killed the largest game every two seconds and every
    session died as "oom-kill" within seconds of starting.

    cgroup v2 only (Kaggle, b200, PC). Without a ceiling (``memory.max`` = max),
    MemAvailable already is this number.
    """
    rel = self_cgroup.read_text().strip().splitlines()[-1].split(":", 2)[2]
    base = root / rel.lstrip("/")
    ceiling = (base / "memory.max").read_text().strip()
    if ceiling == "max":
        return next(int(line.split()[1]) for line in meminfo.read_text().splitlines()
                    if line.startswith("MemAvailable:"))
    stat = dict(line.split() for line in (base / "memory.stat").read_text().splitlines())
    return (int(ceiling) - int(stat["anon"]) - int(stat["shmem"])) // 1024


class Watchdog(threading.Thread):
    def __init__(self, log_path: Path, *, game_max_kb: int, headroom_kb: int | None,
                 cpu_seconds: float | None, interval: float = 2.0) -> None:
        super().__init__(daemon=True, name="posix-watchdog")
        self.log_path = log_path
        self.game_max_kb, self.headroom_kb = game_max_kb, headroom_kb
        self.cpu_seconds, self.interval = cpu_seconds, interval
        self.lock = threading.Lock()
        self.games: dict[int, str] = {}          # uid -> game, only registered ones are watched
        self.kills: list[dict] = []
        self.peak_kb: dict[str, int] = {}
        self.min_headroom_kb: int | None = None
        self.error: BaseException | None = None
        self.ticks = 0
        self._stop = threading.Event()

    def watch(self, uid: int, game: str) -> None:
        with self.lock:
            self.games[uid] = game

    def unwatch(self, uid: int) -> None:
        """reviewer: one uid per review, no longer watched once the review is done (after killing everything). Player uids are watched until wrap-up."""
        with self.lock:
            if uid not in self.games:
                raise KeyError(f"uid {uid} is not watched")
            del self.games[uid]

    def oom_killed_since(self, game: str, since: float) -> bool:
        with self.lock:
            return any(k["game"] == game and k["unix"] >= since and k["reason"] != "cpu"
                       for k in self.kills)

    def stop(self) -> None:
        self._stop.set()

    def _record(self, row: dict) -> None:
        row["unix"] = time.time()
        with self.lock:
            self.kills.append(row)
        with self.log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        # Print it too: stdout is the only thing visible on Kaggle while running. v2's false kills only
        # showed up as "kills N" in the heartbeat, with no reason given.
        print(f"[{time.strftime('%H:%M:%S')}] WATCHDOG kill {json.dumps(row)}", flush=True)

    def tick(self) -> None:
        with self.lock:
            games = dict(self.games)
        per_game: dict[int, int] = {}
        for pid, uid, rss, cpu, comm in _procs():
            if uid not in games:
                continue
            per_game[uid] = per_game.get(uid, 0) + rss
            if self.cpu_seconds and cpu > self.cpu_seconds:
                try:
                    os.kill(pid, signal.SIGKILL)
                    self._record({"game": games[uid], "reason": "cpu", "pid": pid,
                                  "comm": comm, "cpu_s": round(cpu), "rss_kb": rss})
                except ProcessLookupError:
                    pass
        for uid, rss in per_game.items():
            game = games[uid]
            self.peak_kb[game] = max(self.peak_kb.get(game, 0), rss)
            if rss > self.game_max_kb:
                self._record({"game": game, "reason": "game_memory", "rss_kb": rss,
                              "limit_kb": self.game_max_kb, "killed": kill_user(uid, grace=2.0)})
        headroom = cgroup_headroom_kb()
        if headroom is not None:  # tests stub this to None
            self.min_headroom_kb = headroom if self.min_headroom_kb is None \
                else min(self.min_headroom_kb, headroom)
            if self.headroom_kb and headroom < self.headroom_kb and per_game:
                uid = max(per_game, key=per_game.get)
                self._record({"game": games[uid], "reason": "headroom", "rss_kb": per_game[uid],
                              "headroom_kb": headroom, "limit_kb": self.headroom_kb,
                              "killed": kill_user(uid, grace=2.0)})
        self.ticks += 1

    def run(self) -> None:
        try:
            while not self._stop.wait(self.interval):
                self.tick()
        except BaseException as error:  # noqa: BLE001 — recorded; the launcher checks it every session
            self.error = error
            raise

    def summary(self) -> dict:
        with self.lock:
            return {"kills": list(self.kills), "ticks": self.ticks,
                    "peak_game_rss_gib": {g: round(kb / 2**20, 2) for g, kb in
                                          sorted(self.peak_kb.items(), key=lambda x: -x[1])[:10]},
                    "min_headroom_gib": None if self.min_headroom_kb is None
                    else round(self.min_headroom_kb / 2**20, 1),
                    "error": repr(self.error) if self.error else None}
