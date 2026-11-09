#!/usr/bin/env python3
"""Run one arc-code arm: one session per game, wrap up when the wall clock runs out.

**Always a single pass.** The competition allows only one pass (the real gateway answers GET scorecard
with 403 and opening a second scorecard with 409, so `--resume` is structurally impossible); a local
multi-pass run would measure something the competition cannot do. If a weak arm's sessions collapse early
and leave wall clock unused, that is itself the signal. The leftover time is deliberately given up.

Drivers differ only in `arms/<driver>.py` (protocol in `arms/__init__.py`); prompts live in
`prompts/<set>/` (named by the config's `prompts`, fingerprinted in ARM_STATE). The gateway, ledger,
fence and act client are all in `src/arcforge/arc_code/`, because CONVENTIONS forbids imports across work/ units.

    python run_arm.py --config configs/b200-smoke.json \
        --run-dir $ARC_RUNS/arc_code/<tag> \
        --corpus <.../environment_files> \
        --prep <.../arc-code-prep> \
        --python <venv with arc_agi>/bin/python \
        --agent-python <agent runtime>/bin/python \
        --driver-bin <codex install prefix> --driver codex \
        --model-base-url http://127.0.0.1:30811/v1 --model <served name>
"""

from __future__ import annotations

import argparse
import dataclasses
import errno
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid
from pathlib import Path

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[1]


def _own_arcforge():
    """Always use **this repo's** arcforge, never another worktree's.

    `.venv` is shared across worktrees, and the editable install of `arcforge` points at whichever
    worktree ran `install -e` last. A plain `sys.path.insert` only works while "nobody has imported it
    yet"; once it was loaded elsewhere first, this script silently runs on someone else's code.
    """
    import importlib
    import importlib.util

    existing = sys.modules.get("arcforge")
    if existing and Path(existing.__file__).resolve() == ROOT / "src" / "arcforge" / "__init__.py":
        return
    for name in [n for n in sys.modules if n == "arcforge" or n.startswith("arcforge.")]:
        del sys.modules[name]
    pkg = ROOT / "src" / "arcforge"
    spec = importlib.util.spec_from_file_location(
        "arcforge", pkg / "__init__.py", submodule_search_locations=[str(pkg)])
    module = importlib.util.module_from_spec(spec)
    sys.modules["arcforge"] = module
    spec.loader.exec_module(module)
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))


_own_arcforge()

from arcforge.arc_code import act_client, bridge, image_ref, manifest, record, recovery, workspace_tools  # noqa: E402
from arcforge.arc_code import level_episode as episode_stops  # noqa: E402
from arcforge.arc_code import level_seed  # noqa: E402
from arcforge.arc_code.admission import Admission  # noqa: E402
from arcforge.arc_code.ledger import Ledger  # noqa: E402
from arcforge.arc_code.score import baselines, score_run, short_index  # noqa: E402

sys.path.insert(0, str(UNIT))
import arms  # noqa: E402
import posix_fence  # noqa: E402
import prompts  # noqa: E402

#: The two renderings of the fence. Same resource declaration (ARCSEC_*), different mechanism -- S2 of
#: docs/plans/arc_code_gateway.md. ``bwrap``: jail.sh inside a systemd scope (b200, PC). ``posix``:
#: fence_posix.sh, one uid per game, a watchdog enforcing memory (Kaggle: no namespaces, no systemd).
FENCES = ("bwrap", "posix")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_json(path: Path, value: dict) -> None:
    """Readers only ever see a complete checkpoint; temp files of different workers never collide."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x") as handle:
            json.dump(value, handle, indent=2, default=str)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


#: Timeout of one ``systemctl --user show`` while polling scope state after stop: at most 10 s and no later than
#: the polling deadline, but at least 1 s (a timeout shorter than systemctl's own latency is bound to fail). When
#: a group's episodes wrap up together systemd --user is busy and one show takes over 1 s: in v4b g000051 and
#: g000060 (three games) the old timeout min(1 s, remaining) raised TimeoutExpired and the whole game was judged a
#: harness failure.
SCOPE_SHOW_TIMEOUT_SECONDS = 10.0
#: Deadline for the scope to become inactive after stop. In v4b (48 games per machine) one show during wrap-up
#: already took over 1 s; v5 runs 128 games per machine with groups of 24 wrapping up together, so systemd is
#: busier: allow 30 s (TimeoutStopSec 15 s + margin) so that the per-show cap and the retries actually get used
SCOPE_STOP_DEADLINE_SECONDS = 30.0
SCOPE_SHOW_FLOOR_SECONDS = 1.0


def stop_owned_scope(unit: str, description: str) -> dict:
    """Verify the Description first, then stop this one scope; refuse to touch it if ownership is not established.

    After stop, poll until the deadline: a show timeout (systemd busy) is retried, not a failure; raise only when
    the deadline passes without verifying that the scope stopped (including a timeout on the last show). Every
    timeout is recorded in the returned ``show_timeouts``."""
    env = {**os.environ, "XDG_RUNTIME_DIR": os.environ.get(
        "XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")}

    def inspect(timeout=10) -> dict:
        proc = subprocess.run(
            ["systemctl", "--user", "show", unit,
             "-p", "LoadState", "-p", "ActiveState", "-p", "Description", "-p", "Result", "-p", "ControlGroup"],
            capture_output=True, text=True, timeout=timeout, env=env)
        properties = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
        if properties.get("LoadState") == "not-found":
            return properties
        if proc.returncode or properties.get("Description") != description:
            raise RuntimeError(f"refusing scope cleanup: ownership not verified for {unit}")
        return properties

    population = []

    def empty_group(properties: dict) -> bool:
        group = properties.get("ControlGroup")
        observation = {"control_group": group, "observed_monotonic": time.monotonic()}
        population.append(observation)
        if group == "":
            observation.update(status="no_control_group", empty=True)
            return True
        mount = Path("/sys/fs/cgroup")
        if (not isinstance(group, str) or not group.startswith("/") or ".." in Path(group).parts
                or Path(group).name != unit):
            raise RuntimeError(f"invalid owned scope cgroup path: {unit}: {group!r}")
        path = mount / group.lstrip("/")
        if path.resolve() != path or not path.is_relative_to(mount):
            raise RuntimeError(f"noncanonical owned scope cgroup path: {unit}: {group!r}")
        observation["path"] = str(path)
        try:
            observation["events_before"] = (path / "cgroup.events").read_text()
            observation["procs"] = (path / "cgroup.procs").read_text()
            observation["events_after"] = (path / "cgroup.events").read_text()
        except FileNotFoundError:
            if path.exists():
                raise RuntimeError(f"owned cgroup exists but lacks population counters: {path}") from None
            observation.update(status="removed", empty=True)
            return True
        except OSError as error:
            # ENODEV: kernfs has already withdrawn this cgroup's files = the cgroup is being deleted, and the kernel
            # only deletes cgroups with no processes and no child cgroups, so it is empty. 2026-10-18 arc3 RL: the
            # scope was killed after a level clear, cleanup happened to read a cgroup mid-deletion, ENODEV reached
            # the session loop, the whole group failed and the orchestrator stopped (about once per 3,000 episodes).
            if error.errno != errno.ENODEV:
                raise
            observation.update(status="removing", empty=True)
            return True
        recursive = []
        for key in ("events_before", "events_after"):
            rows = [line.split() for line in observation[key].splitlines() if line.strip()]
            if any(len(row) != 2 for row in rows) or len({row[0] for row in rows}) != len(rows):
                raise RuntimeError(f"malformed owned cgroup events: {path}")
            events = dict(rows)
            if events.get("populated") not in {"0", "1"}:
                raise RuntimeError(f"missing recursive cgroup population: {path}")
            recursive.append(int(events["populated"]))
        pids = observation["procs"].split()
        if any(not pid.isdecimal() or int(pid) <= 0 for pid in pids):
            raise RuntimeError(f"malformed owned cgroup process list: {path}")
        empty = recursive == [0, 0] and not pids
        observation.update(status="observed", recursive_populated=recursive, direct_pids=pids, empty=empty)
        return empty

    def terminated(properties: dict) -> bool:
        return (properties.get("LoadState") == "not-found"
                or properties.get("ActiveState") in {"inactive", "failed"} and empty_group(properties))

    before = inspect()
    result = before.get("Result") or None
    if before.get("LoadState") == "not-found":
        return {"outcome": "absent", "unit_result": result, "before": before, "population_observations": population}
    if terminated(before):
        return {"outcome": "inactive", "unit_result": result, "before": before, "population_observations": population}
    # RuntimeMaxSec may collect a transient scope between show and stop. The
    # stop exit code alone cannot establish whether owned cleanup succeeded.
    try:
        stopped = subprocess.run(["systemctl", "--user", "stop", unit],
                                 capture_output=True, text=True, timeout=30, check=False, env=env)
        stop = {"returncode": stopped.returncode, "stdout": stopped.stdout, "stderr": stopped.stderr,
                "timed_out": False}
    except subprocess.TimeoutExpired as exc:
        stop = {"returncode": None, "stdout": exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else exc.stdout,
                "stderr": exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr,
                "timed_out": True}
    after = inspect()
    deadline = time.monotonic() + SCOPE_STOP_DEADLINE_SECONDS
    show_timeouts = []
    while not terminated(after):
        if after.get("LoadState") != "not-found" and result in (None, "success"):
            result = after.get("Result") or result
        if after.get("ActiveState") not in {"inactive", "failed", "deactivating"} or time.monotonic() >= deadline:
            raise RuntimeError(f"owned scope remains active after stop: {unit}; stop={stop!r}; after={after!r}; "
                               f"population={population!r}; show_timeouts={show_timeouts!r}")
        if after.get("ActiveState") == "deactivating":
            empty_group(after)  # Retain population evidence while waiting for the terminal state.
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
        while True:
            timeout = max(SCOPE_SHOW_FLOOR_SECONDS, min(SCOPE_SHOW_TIMEOUT_SECONDS, deadline - time.monotonic()))
            try:
                after = inspect(timeout=timeout)
                break
            except subprocess.TimeoutExpired as error:
                show_timeouts.append(round(timeout, 3))
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        f"owned scope not verified stopped by the deadline: {unit}: systemctl show timed out "
                        f"{len(show_timeouts)} times {show_timeouts!r}; stop={stop!r}; last after={after!r}; "
                        f"population={population!r}") from error
    outcome = "absent" if after.get("LoadState") == "not-found" else "inactive" if stop["returncode"] != 0 else "stopped"
    after_result = after.get("Result") if after.get("LoadState") != "not-found" else None
    terminal_result = result if result not in (None, "success") else after_result or result
    return {"outcome": outcome, "unit_result": terminal_result, "stop": stop, "before": before, "after": after,
            "population_observations": population, "show_timeouts": show_timeouts}


class PosixFence:
    """Shared state in posix mode: the ARCSEC_* common to every session, and the one watchdog."""

    def __init__(self, env: dict[str, str], watchdog: posix_fence.Watchdog) -> None:
        self.env, self.watchdog = env, watchdog


class ArmSupervisor:
    """Serializes ARM_STATE updates; scope receipts are written outside the jail, to disk before start-up, for
    outer cleanup."""

    def __init__(self, run_dir: Path, state: dict):
        self.run_dir = run_dir.resolve()
        self.state = state
        self.lock = threading.RLock()
        self.cancelled = threading.Event()
        #: None = bwrap (systemd scope); in posix mode a :class:`PosixFence`.
        self.fence: PosixFence | None = None
        self.scopes: dict[str, dict] = {}
        self.receipts = self.run_dir / "supervision" / "scopes"
        self.run_nonce = uuid.uuid4().hex
        self.update(status="starting", plays=[], supervision={
            "run_nonce": self.run_nonce, "scopes_dir": str(self.receipts),
            "launcher_pid": os.getpid()})

    def update(self, **fields) -> None:
        with self.lock:
            self.state.update(fields)
            self.state["updated"] = time.time()
            atomic_json(self.run_dir / "ARM_STATE.json", self.state)

    def play(self, record: dict) -> None:
        with self.lock:
            plays = self.state["plays"]
            existing = next((i for i, row in enumerate(plays)
                             if row["game"] == record["game"]), None)
            if existing is None:
                plays.append(record)
            else:
                plays[existing] = record
            self.update()

    def scope(self, game: str, attempt: int, runtime_max_seconds: float | None = None) -> dict:
        nonce = uuid.uuid4().hex
        unit = f"arccode-{nonce}.scope"
        receipt = {"unit": unit, "description": f"arc-code:{self.run_dir}:{nonce}",
                   "run_nonce": self.run_nonce, "run_dir": str(self.run_dir),
                   "game": game, "attempt": attempt, "status": "starting",
                   "started": time.time(), "runtime_max_seconds": runtime_max_seconds}
        with self.lock:
            self.scopes[unit] = receipt
            atomic_json(self.receipts / f"{unit}.json", receipt)
        return receipt

    def cleanup_scope(self, receipt: dict, **fields) -> dict:
        try:
            cleanup = stop_owned_scope(receipt["unit"], receipt["description"])
        except BaseException as error:
            with self.lock:
                receipt.update(fields, status="cleanup_failed", cleanup_error=repr(error))
                atomic_json(self.receipts / f"{receipt['unit']}.json", receipt)
            raise
        with self.lock:
            receipt.update(fields, status="cleaned", cleanup=cleanup, cleaned=time.time())
            atomic_json(self.receipts / f"{receipt['unit']}.json", receipt)
        return cleanup

    def cleanup_all(self) -> list[str]:
        errors = []
        with self.lock:
            pending = [r for r in self.scopes.values() if r["status"] != "cleaned"]
        for receipt in pending:
            try:
                self.cleanup_scope(receipt)
            except Exception as error:  # noqa: BLE001 — try to clean up every owned scope
                errors.append(repr(error))
        return errors


def mint_tokens(run_dir: Path, games: list[str]) -> tuple[Path, str, dict[str, str]]:
    """The repo used to have no generator; every run typed the tokens by hand -- a missing one surfaced 60 s
    later as an ARC_SERVER_NEVER_READY that points nowhere near the real cause. Now it is version-controlled."""
    token_dir = run_dir / "gateway"
    token_dir.mkdir(parents=True, exist_ok=True)
    launcher = secrets.token_hex(24)
    per_game = {game: secrets.token_hex(24) for game in games}
    lines = [f"*={launcher}"] + [f"{g}={t}" for g, t in per_game.items()]
    path = token_dir / "tokens"
    old = os.umask(0o077)
    try:
        path.write_text("\n".join(lines) + "\n")
    finally:
        os.umask(old)
    return path, launcher, per_game


def wait_ready(url: str, key: str, timeout: float) -> list[dict]:
    """Ready = `GET /api/games` returns a non-empty list. "TCP connects" means nothing on someone else's container."""
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url + "/api/games")
            req.add_header("X-API-Key", key)
            with urllib.request.urlopen(req, timeout=10) as response:
                games = json.loads(response.read())
            if games:
                return games
        except Exception as error:      # noqa: BLE001 — readiness polling is expected to fail
            last = f"{type(error).__name__}: {error}"
        time.sleep(1.0)
    raise SystemExit(f"gateway never became ready in {timeout}s — last: {last}")


def wait_http(url: str, timeout: float, proc: subprocess.Popen | None = None) -> None:
    """Wait for a GET to return 200. **If proc dies, report it at once, not at the timeout** -- the previous
    version had no `poll()` anywhere in the run, so a gateway that failed to start showed up as "polling timed
    out after 300 s" while the real cause sat in the log."""
    deadline = time.monotonic() + timeout
    last = ""
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"{url}: process exited {proc.returncode} during startup")
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                if r.status == 200:
                    return
                last = f"HTTP {r.status}"
        except Exception as error:  # noqa: BLE001
            last = repr(error)
        time.sleep(1.0)
    raise RuntimeError(f"{url} never became ready in {timeout:.0f}s — last: {last}")


#: sglang's Qwen video preprocessing (FPS in multimodal/processors/qwen_vl.py) samples mp4 frames at this rate
VIDEO_SERVER_FPS = 2.0


def record_config(cfg: dict, pset: prompts.PromptSet) -> dict | None:
    """``record``: whether the game record is on. Whether the brief talks about ``record/`` must agree with it --
    the record on with a v6 brief (which tells the agent to read a logs.txt that does not exist), or the reverse,
    stops before the run starts."""
    rec = cfg.get("record")
    speaks = "record/README.md" in pset.texts["brief.md"]
    if rec is None:
        if speaks:
            raise SystemExit(f"prompts {pset.name} talks about record/, but the config has no record")
        return None
    if not speaks:
        raise SystemExit(f"config turns record on, but prompts {pset.name} does not mention record/ (use a set from v7 on)")
    if not isinstance(rec, dict) or not {"video_fps", "video_size"} <= set(rec) \
            or not set(rec) <= {"video_fps", "video_size", "video_max_frames", "png_size", "video_repeat"}:
        raise SystemExit(f"record must be {{video_fps, video_size[, video_max_frames][, png_size][, video_repeat]}}, "
                         f"got {rec!r}")
    repeat = rec.get("video_repeat", 1)
    if not isinstance(repeat, int) or repeat < 1:
        raise SystemExit(f"record.video_repeat must be an int >= 1, got {repeat!r}")
    if repeat > 1 and rec["video_fps"] > VIDEO_SERVER_FPS:
        # sglang qwen_vl smart_nframes: takes total / mp4 fps x 2.0 frames (linear sampling); with an mp4 faster
        # than 2 fps the repeated frames get dropped
        raise SystemExit(f"record.video_repeat {repeat} needs video_fps <= {VIDEO_SERVER_FPS}: the server samples at "
                         f"{VIDEO_SERVER_FPS} fps, so a faster mp4 drops the repeated frames")
    if (cfg.get("driver_config") or {}).get("features.view_video") != "true":
        raise SystemExit("the record brief tells the agent to use view_video; driver_config needs features.view_video=true")
    return rec


def layout_config(cfg: dict, pset: prompts.PromptSet) -> bool:
    """``workspace_layout``: create lib/ scripts/ levels/, and act reminds the agent to write levels/Lnn.md on a
    level clear. As with record, the switch and the brief must agree."""
    layout = cfg.get("workspace_layout", False)
    speaks = "`levels/L01.md`" in pset.texts["brief.md"]
    if not isinstance(layout, bool):
        raise SystemExit(f"workspace_layout must be true/false, got {layout!r}")
    if layout != speaks:
        raise SystemExit(f"workspace_layout={layout}, but prompts {pset.name} "
                         f"{'does' if speaks else 'does not'} state the levels/ rules")
    if layout and cfg.get("record") is None:
        raise SystemExit("workspace_layout's level-clear docs cite record/; the config must turn record on")
    return layout


def observation_config(cfg: dict, pset: prompts.PromptSet) -> bool:
    """Opt-in exact queries; availability and the advertised interface must agree."""
    # 2026-09-30: judged probably not useful, not in Kaggle: docs/research/arc_code_g4_native_20260929.md
    enabled = cfg.get("observation_tools", False)
    if not isinstance(enabled, bool):
        raise SystemExit("observation_tools must be true or false")
    if enabled and cfg.get("record") is None:
        raise SystemExit("observation_tools requires record")
    advertised = "./act inspect" in pset.texts["brief.md"]
    if enabled != advertised:
        raise SystemExit("observation_tools must match the prompt's ./act inspect interface")
    return enabled


def receipt_config(cfg: dict, pset: prompts.PromptSet) -> bool:
    """``act_receipt`` (prompts v16): when ``./act do`` finishes it prints the batch's object-level changes.

    Motivation: after each action call Codex calls the model 3 more times on average (Super Duck 1.2), the
    extra calls being image views and hand-written diffs: docs/research/super_duck_codex_roundtrips_20261008.md.
    The brief must describe the receipt for the two sides to agree.
    """
    enabled = cfg.get("act_receipt", False)
    if not isinstance(enabled, bool):
        raise SystemExit("act_receipt must be true or false")
    if enabled and cfg.get("record") is None:
        raise SystemExit("act_receipt requires record")
    advertised = "what changed over" in pset.texts["brief.md"]
    if enabled != advertised:
        raise SystemExit(f"act_receipt={enabled} but prompts {pset.name} "
                         f"{'does' if advertised else 'does not'} describe the ./act do change summary")
    return enabled


def receipt_scene_config(cfg: dict, pset: prompts.PromptSet, receipt: bool) -> bool:
    """``act_receipt_scene`` (prompts v17): on a level clear the receipt lists the new level's objects instead
    (receipt_scene in receipt.py). k1 measured that on easy levels codex spends about 8k more per level than
    Super Duck, 2.5-5.5k of it from level start to the first action; Duck gets an object segmentation every turn."""
    enabled = cfg.get("act_receipt_scene", False)
    if not isinstance(enabled, bool):
        raise SystemExit("act_receipt_scene must be true or false")
    if enabled and not receipt:
        raise SystemExit("act_receipt_scene requires act_receipt")
    advertised = "lists the objects on the new level" in pset.texts["brief.md"]
    if enabled != advertised:
        raise SystemExit(f"act_receipt_scene={enabled} but prompts {pset.name} "
                         f"{'does' if advertised else 'does not'} describe the new-level object list")
    return enabled


def image_config(cfg: dict, pset: prompts.PromptSet, key_file: Path | None) -> bool:
    """``act_image`` (prompts v14i, duckx c0i): ``./act do`` ends with markers for the board image after the batch
    and the diff image, which the model proxy (work/arc_code/act_images.py) replaces with images -- Super Duck
    attaches an image every turn, Codex needs one more model call to view_image. Needs record (images are read
    from the record) and ``--image-ref-key`` (the middleware signs refs, the proxy verifies them; give the proxy
    the same file)."""
    enabled = cfg.get("act_image", False)
    if not isinstance(enabled, bool):
        raise SystemExit("act_image must be true or false")
    if enabled and cfg.get("record") is None:
        raise SystemExit("act_image requires record")
    advertised = ("shows you the board after the batch" in pset.texts["brief.md"]
                  or LIBRARY_PHRASE in pset.texts["brief.md"])
    if enabled != advertised:
        raise SystemExit(f"act_image={enabled} but prompts {pset.name} "
                         f"{'does' if advertised else 'does not'} say that ./act do (or arcgame) shows the board")
    if enabled != (key_file is not None):
        raise SystemExit("act_image and --image-ref-key go together (the proxy needs the same key file)")
    if key_file is not None:
        image_ref.load_key(key_file)
    return enabled


#: the brief's sentence about arcgame; ``act_library`` must agree with it.
LIBRARY_PHRASE = "from arcgame import"


def library_config(cfg: dict, pset: prompts.PromptSet, images: bool, driver_name: str) -> bool:
    """``act_library`` (prompts v14L, duckx c0L): actions go only through the Python arcgame (``./act do``
    removed), so one program acts, reads the board and computes diffs; when a process that acted exits, arcgame
    writes the board and diff image markers straight to the stdout of the shell the driver started (the agent's
    filtering cannot swallow them), and the proxy swaps in images as for act_image. The library sits in the run
    directory's ``agent_lib/`` with PYTHONPATH pointing at it: bwrap mounts it read-only into the jail; under posix
    (Kaggle) it is a root-owned 0755 directory, and fence_posix.sh verifies as the game's user that it is readable
    and not writable (``ARCSEC_AGENT_LIB``)."""
    enabled = cfg.get("act_library", False)
    if not isinstance(enabled, bool):
        raise SystemExit("act_library must be true or false")
    advertised = LIBRARY_PHRASE in pset.texts["brief.md"]
    if enabled != advertised:
        raise SystemExit(f"act_library={enabled} but prompts {pset.name} "
                         f"{'does' if advertised else 'does not'} teach `{LIBRARY_PHRASE} ...`")
    if not enabled:
        if "arcgame" in cfg:
            raise SystemExit("config arcgame (a frozen library version) goes with act_library")
        return False
    if not images:
        raise SystemExit("act_library shows its images through act_image: turn act_image on")
    if cfg.get("fence") not in FENCES:
        raise SystemExit(f"act_library needs a fence in {FENCES}, not {cfg.get('fence')!r}")
    if driver_name != "codex":
        raise SystemExit(f"act_library finds the command's output through the codex process, not {driver_name}")
    if cfg.get("observation_tools") or cfg.get("act_receipt"):
        raise SystemExit("act_library replaces ./act do: observation_tools / act_receipt do not apply")
    return True


def arcgame_version(cfg: dict, library: bool) -> str | None:
    """config ``arcgame``: which frozen arcgame version is mounted into the jail (``act_client.ARCGAME_FROZEN``);
    absent means the current arcgame.py. Kaggle v19 uses ``c0L-r1`` -- the copy duckx measured at 69.99, not
    the later edits."""
    if "arcgame" not in cfg:
        return None
    version = cfg["arcgame"]
    if not library:
        raise SystemExit("config arcgame (a frozen library version) goes with act_library")
    if version not in act_client.ARCGAME_FROZEN:
        raise SystemExit(f"config arcgame {version!r} is not a frozen version: {sorted(act_client.ARCGAME_FROZEN)}")
    return version


def expect_config(cfg: dict, pset: prompts.PromptSet) -> bool:
    """``act_expect`` (RSI batch 1 item A, prompts v14ra): ``./act do --expect-change / --expect-boards``, with the
    summary line always last. Qwen pipes 40-60% of act output through tail, which hid the step at which a long
    batch diverged from the expectation (docs/plans/rsi_harness_batch1.md)."""
    enabled = cfg.get("act_expect", False)
    if not isinstance(enabled, bool):
        raise SystemExit("act_expect must be true or false")
    if enabled and cfg.get("record") is None:
        raise SystemExit("act_expect requires record")
    advertised = "--expect-boards" in pset.texts["brief.md"]
    if enabled != advertised:
        raise SystemExit(f"act_expect={enabled} but prompts {pset.name} "
                         f"{'does' if advertised else 'does not'} describe --expect-boards")
    return enabled


def digest_config(cfg: dict, pset: prompts.PromptSet, expect: bool) -> bool:
    """``act_digest`` (RSI round 3, prompts v14re): ``./act do`` prints a ``what changed:`` line after each action,
    and ``--expect-change`` treats a change only at the board's edge as no change
    (src/arcforge/arc_code/act_client.py ``with_digest``)."""
    enabled = cfg.get("act_digest", False)
    if not isinstance(enabled, bool):
        raise SystemExit("act_digest must be true or false")
    if enabled and not expect:
        raise SystemExit("act_digest goes on act_expect: turn act_expect on")
    advertised = "`what changed:`" in pset.texts["brief.md"]
    if enabled != advertised:
        raise SystemExit(f"act_digest={enabled} but prompts {pset.name} "
                         f"{'does' if advertised else 'does not'} describe `what changed:`")
    return enabled


#: the brief's sentence on the notes.md cap; the number must equal the config's ``notes_budget``.
NOTES_BUDGET_RE = re.compile(r"Keep `notes.md` under ([\d,]+) characters")


def notes_budget_config(cfg: dict, pset: prompts.PromptSet) -> int | None:
    """``notes_budget`` (RSI batch 1 item B, prompts v14rb): when notes.md exceeds this many characters, act warns
    after every command and asks for a cleanup on a level clear. 82% of DeepSeek's notes.md reads were cut by the
    2.5k-token tool output cap, and what got cut was exactly the middle."""
    limit = cfg.get("notes_budget")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise SystemExit(f"notes_budget must be a positive int (characters), not {limit!r}")
    if limit is not None and cfg.get("record") is None:
        raise SystemExit("notes_budget requires record")
    m = NOTES_BUDGET_RE.search(pset.texts["brief.md"])
    said = int(m.group(1).replace(",", "")) if m else None
    if said != limit:
        raise SystemExit(f"notes_budget={limit} but prompts {pset.name} says {said}")
    return limit


def tools_config(cfg: dict, pset: prompts.PromptSet) -> bool:
    """``workspace_tools`` (RSI batch 1 item C, prompts v14rc): at game start, put lib/arc.py (tools that read the
    record) and ./bg in place (src/arcforge/arc_code/workspace_tools.py)."""
    enabled = cfg.get("workspace_tools", False)
    if not isinstance(enabled, bool):
        raise SystemExit("workspace_tools must be true or false")
    if enabled and not (cfg.get("workspace_layout") and cfg.get("record") is not None):
        raise SystemExit("workspace_tools puts lib/arc.py in the workspace layout and reads record/: "
                         "turn workspace_layout and record on")
    advertised = "lib/arc.py" in pset.texts["brief.md"]
    if (cfg.get("gen2_toolkit") is not None or cfg.get("act_toolkit") is not None) and not enabled:
        return False                 # the toolkit places lib/arc.py itself;
                                     # gen2_toolkit_config checks the brief against it
    if enabled != advertised:
        raise SystemExit(f"workspace_tools={enabled} but prompts {pset.name} "
                         f"{'does' if advertised else 'does not'} describe lib/arc.py")
    return enabled


def recap_config(cfg: dict, pset: prompts.PromptSet, library: bool) -> bool:
    """``act_recap`` (prompts v14k, duckx c0i4): ``./act recap`` prints, for free, every batch of the current level
    so far, the cleared levels and the objects on the current board (recap.py, the same code as arcgame.recap()).
    In c0i-r2 each compaction was followed by a median of 12 calls before the next action, mostly spent rebuilding
    the level from record/."""
    enabled = cfg.get("act_recap", False)
    if not isinstance(enabled, bool):
        raise SystemExit("act_recap must be true or false")
    if enabled and cfg.get("record") is None:
        raise SystemExit("act_recap requires record")
    if enabled and library:
        raise SystemExit("act_recap is ./act's; act_library has arcgame.recap()")
    advertised = "./act recap" in pset.texts["brief.md"]
    if enabled != advertised:
        raise SystemExit(f"act_recap={enabled} but prompts {pset.name} "
                         f"{'does' if advertised else 'does not'} describe ./act recap")
    return enabled


#: ``act_library``: where arcgame.py goes under the run directory (read-only in the jail, on PYTHONPATH)
AGENT_LIB = "agent_lib"
#: the copy given to the reviewer in arms whose player does not use the library (frozen: later arcgame edits
#: cannot silently change the reviewer)
REVIEW_LIB = "review_lib"
REVIEW_ARCGAME = "c0L-r1"


def context_recovery_config(cfg: dict, driver_name: str) -> int:
    """Explicit, bounded recovery from proven context-limit failures only."""
    # 2026-09-30: judged probably not useful, not in Kaggle (at most one game per run hits the limit):
    # docs/research/arc_code_g4_native_20260929.md
    setting = cfg.get("context_error_recovery")
    if setting is None:
        return 0
    if not isinstance(setting, dict) or set(setting) != {"max_fresh_sessions"}:
        raise SystemExit("context_error_recovery requires only max_fresh_sessions")
    maximum = setting["max_fresh_sessions"]
    if type(maximum) is not int or not 1 <= maximum <= recovery.MAX_FRESH_RECOVERIES:
        raise SystemExit("context_error_recovery.max_fresh_sessions must be an integer in 1..2")
    if driver_name != "codex" or cfg["resume_mode"] != "thread" or cfg.get("record") is None:
        raise SystemExit("context_error_recovery requires codex, thread resume and record")
    if cfg.get("keep_driver_home") is not True:
        raise SystemExit("context_error_recovery requires keep_driver_home=true to preserve failed traces")
    return maximum


def developer_rules(knobs: "arms.Knobs", pset: "prompts.PromptSet", driver_name: str) -> "arms.Knobs":
    """If the prompt set has ``developer.md``, put it into codex's ``developer_instructions``.

    The old arm (RULES in arc_agi3/arc_code_qwen/run_arm.py) always had this; the new arm lost it in the
    migration, and in v10's Save & Run the agent enumerated server endpoints, wrote its own client and ran
    ``find /`` for source code. The value goes through ``-c key=value`` and is parsed as TOML -- hence a JSON
    string (also a valid TOML basic string).
    """
    if not pset.has("developer.md"):
        return knobs
    if driver_name != "codex":
        raise SystemExit(f"prompts {pset.name} has developer.md, but {driver_name} has no "
                         "developer_instructions to put it in")
    if any(k == "developer_instructions" for k, _ in knobs.driver_config):
        raise SystemExit("developer_instructions comes from both driver_config and developer.md")
    rules = ("developer_instructions", json.dumps(pset.optional("developer.md")))
    return dataclasses.replace(knobs, driver_config=tuple(sorted(knobs.driver_config + (rules,))))


def compact_reminder_config(cfg: dict, pset: prompts.PromptSet) -> int | None:
    """``compact_reminder_tokens`` (prompts v10): once the context passes this number, codex
    (features.pre_compact_reminder from patch 0005) sends prompts/<set>/compact_reminder.md once, so the agent
    writes up its notes before the automatic compaction. The three must be all present or all absent."""
    tokens = cfg.get("compact_reminder_tokens")
    has_text = pset.has("compact_reminder.md")
    feature = (cfg.get("driver_config") or {}).get("features.pre_compact_reminder") == "true"
    if tokens is None and not has_text and not feature:
        return None
    if tokens is None or not has_text or not feature:
        raise SystemExit(f"compact_reminder_tokens={tokens!r}, compact_reminder.md of prompts {pset.name}"
                         f" ({'present' if has_text else 'absent'}) and features.pre_compact_reminder ({feature}) must be set together")
    limit = cfg.get("auto_compact_token_limit")
    if not isinstance(tokens, int) or not 0 < tokens < limit:
        raise SystemExit(f"compact_reminder_tokens {tokens!r} must be below auto_compact_token_limit {limit} and positive")
    return tokens


#: config ``final_call``: the notice text (``{minutes}`` filled in when sent), and its id on the reviewer-notice
#: path (review_notice.py)
FINAL_CALL_TEXT = prompts.ROOT_DIR / "_notices" / "final_call.md"
FINAL_CALL_ID = "F01"


def final_call_config(cfg: dict, review_spec: dict | None) -> dict | None:
    """``final_call`` (hs110 v39, the user 2026-10-31): ``{"before_end_seconds": N}``; N seconds before the global
    deadline, tell every game "only a few minutes left, clear levels at full speed". The notice goes the reviewer's
    way (the model proxy's review_notice.py appends it after the game's next tool output), so a reviewer is
    required; N must lie within wall."""
    spec = cfg.get("final_call")
    if spec is None:
        return None
    if not isinstance(spec, dict) or sorted(spec) != ["before_end_seconds"]:
        raise SystemExit(f"final_call has exactly one key, before_end_seconds, not {spec!r}")
    seconds = spec["before_end_seconds"]
    if not (isinstance(seconds, int) and not isinstance(seconds, bool) and 60 <= seconds < cfg["wall_seconds"]):
        raise SystemExit(f"final_call.before_end_seconds {seconds!r} must be an int >= 60 and below wall_seconds {cfg['wall_seconds']} (int)")
    if review_spec is None:
        raise SystemExit("final_call goes through the reviewer's notices (the model proxy's review_notice.py, reviews/notices/): the config needs a reviewer")
    text = FINAL_CALL_TEXT.read_text()
    if text.count("{minutes}") != 1:
        raise SystemExit(f"{FINAL_CALL_TEXT} must contain exactly one {{minutes}}")
    return {"before_end_seconds": seconds, "text": text}


def final_call(stop: threading.Event, run_dir: Path, games: list[str], deadline: float, spec: dict,
               supervisor: "ArmSupervisor", state: dict) -> None:
    """config ``final_call``: ``before_end_seconds`` before the deadline, drop an ``F01`` notice for every game
    (prompts/_notices/final_call.md, with the minutes left filled in). The model proxy appends it after that game's
    next tool output; finished games send no more requests, so only games still playing read it."""
    import review_notice
    if stop.wait(max(0.0, deadline - spec["before_end_seconds"] - time.monotonic())):
        return
    left = deadline - time.monotonic()
    text = spec["text"].replace("{minutes}", str(max(1, round(left / 60)))).strip()
    for game in games:
        review_notice.write_notice(run_dir, game, FINAL_CALL_ID, text)
    with supervisor.lock:
        state["final_call"] = {"games": len(games), "seconds_left": round(left, 1), "text": text}
    supervisor.update()
    log(f"FINAL CALL: {len(games)} games told {left:.0f}s are left ({FINAL_CALL_ID}, reviews/notices/)")


def level_episode_config(cfg: dict, pset: prompts.PromptSet, args) -> None:
    """``--level-episode`` (run_arc3_level.py): check the preconditions and put the settings in
    ``cfg["level_episode"]`` (_play reads them from there; also recorded in state).
    Prompt set has ``level.md`` <=> single-level episode: a brief saying "it ends when you clear the level" while the
    whole game is played, or the reverse, lies to the agent."""
    if args.agent_model_url is not None and cfg["fence"] != "bwrap":
        raise SystemExit("--agent-model-url only plugs into bwrap's model bridge (under posix the agent connects straight to --model-base-url)")
    if args.level_episode is None:
        if pset.has("level.md"):
            raise SystemExit(f"prompt set {pset.name} carries level.md: usable only with --level-episode")
        if args.level_seed is not None:
            raise SystemExit("--level-seed goes only with --level-episode")
        if "level_episode" in cfg:
            raise SystemExit("the config must not set level_episode itself: --level-episode produces it")
        return
    if not pset.has("level.md"):
        raise SystemExit(f"--level-episode needs a prompt set with level.md ({pset.name} has none)")
    if args.upstream:
        raise SystemExit("--level-episode only on the local corpus: the official gateway is started here so --start-level can attach")
    if cfg["fence"] != "bwrap":
        raise SystemExit("--level-episode needs fence bwrap (the stop files and the proxy are on the host side)")
    if not args.games or len(args.games) != 1:
        raise SystemExit(f"--level-episode is one game: --games must name exactly one, got {args.games}")
    if cfg.get("tail_resume"):
        raise SystemExit("--level-episode does not resume play (tail_resume)")
    if args.agent_model_url is None:
        raise SystemExit("--level-episode needs --agent-model-url: the output budget is counted by run_arc3_level.py's proxy")
    if not args.stop_dir.is_dir() or any(args.stop_dir.iterdir()):
        raise SystemExit(f"--stop-dir {args.stop_dir} must be an empty directory")
    cfg["level_episode"] = {"level": args.level_episode, "action_cap": args.action_cap,
                            "stop_dir": str(args.stop_dir.resolve()), "start_hash": args.expect_start_hash}
    # Prior-levels seed (--level-seed; sets with level_seed.md such as v14seed): level 1 has no prior levels, every
    # later level does. An opening saying "earlier levels were cleared by previous sessions and the notes are all
    # here" over an empty workspace, or the reverse, lies to the agent
    seeded = pset.has("level_seed.md")
    if args.level_seed is not None and not seeded:
        raise SystemExit(f"--level-seed needs a prompt set with level_seed.md ({pset.name} has none)")
    if seeded and (args.level_seed is not None) != (args.level_episode >= 2):
        raise SystemExit(f"prompt set {pset.name}: level 1 takes no seed, levels 2+ always need one (level {args.level_episode}, "
                         f"--level-seed {'given' if args.level_seed is not None else 'missing'})")
    if args.level_seed is not None:
        spec = level_seed.SeedSpec.from_json(json.loads(args.level_seed.read_text()))
        if spec.level != args.level_episode:
            raise SystemExit(f"--level-seed is a seed for level {spec.level}, --level-episode is {args.level_episode}")
        cfg["level_episode"]["seed"] = spec.to_json()


def game_episode_config(cfg: dict, args) -> None:
    """``--stop-dir`` without ``--level-episode`` (run_arc3_level.py's arc3_game, whole-game RL): one game played from
    start to end, with the stop-file directory on the host side -- the proxy writes ``budget`` (output budget), this
    script writes ``died`` (codex died, no resume); when either appears the session is closed and no new one is
    opened. Check the preconditions and put the settings in ``cfg["game_episode"]`` (_play reads them from there;
    also recorded in state).

    gen-2 single-level RL (run_arc3_level.py's arc3_g2seed) takes this path too: it adds ``--game-seed`` +
    ``--episode-action-cap`` and starts from the seed's level K. The middleware's stop files (``cleared`` /
    ``action_cap`` / ``start_failed``) go into the same host-side directory (no separate ``seed_stops``); as soon
    as the proxy writes ``budget`` / ``context`` the middleware refuses actions."""
    if "game_episode" in cfg:
        raise SystemExit("the config must not set game_episode itself: --stop-dir (without --level-episode) produces it")
    if args.stop_dir is None or args.level_episode is not None:
        if args.proxy_log is not None:
            raise SystemExit("--proxy-log is only for whole-game episodes (--stop-dir without --level-episode)")
        return
    if args.proxy_log is None:
        raise SystemExit("a whole-game episode needs --proxy-log: the reference when the digest's windows have a gap (repair_windows_from_proxy)")
    if args.upstream:
        raise SystemExit("whole-game episodes only on the local corpus (the official gateway is started here)")
    if cfg["fence"] != "bwrap":
        raise SystemExit("a whole-game episode needs fence bwrap (the stop files and the proxy are on the host side)")
    if not args.games or len(args.games) != 1:
        raise SystemExit(f"a whole-game episode is one game: --games must name exactly one, got {args.games}")
    if cfg.get("tail_resume") or cfg.get("game_output_tokens"):
        raise SystemExit("a whole-game episode does not resume play (tail_resume); its output budget is in the proxy (no game_output_tokens)")
    if args.agent_model_url is None:
        raise SystemExit("a whole-game episode needs --agent-model-url: the output budget is counted by run_arc3_level.py's proxy")
    if args.game_seed is not None and args.episode_action_cap is None:
        raise SystemExit("a whole-game episode plays from the game start: --game-seed goes only with --episode-action-cap (gen-2 single-level episode)")
    if not args.stop_dir.is_dir() or any(args.stop_dir.iterdir()):
        raise SystemExit(f"--stop-dir {args.stop_dir} must be an empty directory")
    cfg["game_episode"] = {"stop_dir": str(args.stop_dir.resolve()), "proxy_log": str(args.proxy_log.resolve())}


def repair_windows_from_proxy(digest: Path, proxy_log: Path) -> list[int]:
    """Before a whole-game episode's TITO manifest: when the window numbers in ``digest`` (driver_digest.jsonl) have a
    gap, re-split the response ids into windows using the per-episode proxy's request log (the ``session_id`` =
    ``<thread>:<n>`` each request actually sent, codex patch 0002), and return the window numbers filled in (no gap =
    [], file untouched).

    2026-10-24 v4 g000000/tn36__s16: the rollout lacked one ``compacted`` record (window 2's), so the digest merged
    window 2's 41 responses into window 1 and the numbers became [0, 1, 3, …]; the manifest raised and the whole
    group was quarantined (about an 18% chance per 32-game group). If the missing one is the **last** compaction
    (g000030/cd82__s1: [0, 1] still contiguous), the numbers do not show it, and only the audit caught "wrong
    window" at the rid join and stopped the whole run -- so numbers are not trusted; every response is checked
    against the proxy's window. The ledger and the proxy both record ``:2``; only the digest, copied from the
    rollout, lacks that record, so it can be repaired:
    - every proxy row must be a session of this thread, with no duplicate request id; every response id of the
      digest must be in the proxy log;
    - in digest order the proxy's window numbers must never go backwards, and after re-splitting they must be
      0..N contiguous;
    - when a window ends in a compaction, that compaction request must be its last request (the summary request
      still uses the old window's sid): every compaction_response_id the digest recorded is checked against this,
      and the missing ones take the last response of the proxy window by the same rule.
    If any of these fails, raise (if it cannot be repaired, the game fails as before). The original file is kept as
    ``<digest>.unrepaired``, and the usage row gains ``windows_repaired_from_proxy``."""
    rows = [json.loads(line) for line in digest.read_text().splitlines()]
    usage = [i for i, row in enumerate(rows) if row["kind"] == "usage"]
    if len(usage) != 1:
        raise RuntimeError(f"{digest}: a whole-game episode's digest must have exactly one usage row, found {len(usage)} rows")
    row = rows[usage[0]]
    numbers = [w["window_number"] for w in row["windows"]]
    thread = row["thread_id"]
    window_of: dict[str, int] = {}
    for line in proxy_log.read_text().splitlines():
        req = json.loads(line)
        if "event" in req:
            continue                  # the harness's own calls (harness/close_session, harness/abort_request): no request
        sid_thread, _, n = req["session_id"].rpartition(":")
        if sid_thread != thread or not n.isdigit():
            raise RuntimeError(f"{proxy_log}: session {req['session_id']!r} is not the digest's thread {thread}")
        if req["request_id"] in window_of:
            raise RuntimeError(f"{proxy_log}: request id {req['request_id']} appears twice")
        window_of[req["request_id"]] = int(n)
    rids = [rid for w in row["windows"] for rid in w["response_ids"]]
    lost = [rid for rid in rids if rid not in window_of]
    if lost:
        raise RuntimeError(f"{digest}: {len(lost)} response ids are not in the proxy log (e.g. {lost[:3]}), cannot repair windows {numbers}")
    order = [window_of[rid] for rid in rids]
    if order != sorted(order):
        raise RuntimeError(f"{digest}: in digest order the proxy's window numbers go backwards, cannot repair windows {numbers}")
    windows: dict[int, list[str]] = {}
    for rid in rids:
        windows.setdefault(window_of[rid], []).append(rid)
    if sorted(windows) != list(range(len(windows))):
        raise RuntimeError(f"{digest}: the proxy's windows are not contiguous either ({sorted(windows)}), cannot repair windows {numbers}")
    if all(window_of[rid] == w["window_number"] for w in row["windows"] for rid in w["response_ids"]):
        return []                     # the digest's windows are the proxy's: nothing to repair
    closing: dict[int, str] = {}
    for w in row["windows"]:
        rid = w["compaction_response_id"]
        if rid is None:
            continue
        n = window_of.get(rid)
        if n is None or windows[n][-1] != rid:
            raise RuntimeError(f"{digest}: compaction request {rid} is not the last request of proxy window {n}, cannot repair windows {numbers}")
        closing[n] = rid
    last = len(windows) - 1
    repaired = [n + 1 for n in range(last) if n not in closing]
    if not repaired or sorted(set(numbers) | set(repaired)) != list(range(len(windows))):
        raise RuntimeError(f"{digest}: repaired windows {repaired} plus the digest's {numbers} are not 0..{last}")
    for n in range(last):
        closing.setdefault(n, windows[n][-1])
    row["windows"] = [{"window_number": n, "response_ids": windows[n], "compaction_response_id": closing.get(n)}
                      for n in range(len(windows))]
    row["windows_repaired_from_proxy"] = repaired
    shutil.copyfile(digest, digest.with_name(digest.name + ".unrepaired"))
    tmp = digest.with_name(digest.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(digest)
    return repaired


#: gen-2 (docs/plans/arc3_harness_gen2_20261020.md): codex hooks manage the session (SessionStart hands over the
#: status pack, Stop tells it to keep going), token_budget makes every window change skip the summary
#: (new_context / thresholds), the harness starts the game, and act has note_gate built in. The hook scripts live
#: here and run inside the fence with the agent's python (mounted read-only into the jail; under posix the repo is
#: already readable by the game uid).
GEN2_HOOKS = UNIT / "gen2" / "hooks"
#: first-generation opt-ins and their codex patch knobs: gen-2 does not stack them (note_gate is built in; the
#: compaction family is dead code under token_budget)
GEN2_REFUSED_KEYS = ("note_gate", "compact_reminder_tokens", "act_expect", "notes_budget", "workspace_tools",
                     "act_digest", "act_image", "act_library", "act_recap", "act_receipt", "act_receipt_scene",
                     "observation_tools", "level_gate", "arcgame", "act_click_info", "act_toolkit")
GEN2_REFUSED_DRIVER = ("features.compaction_reuse_prefix", "features.compaction_strip_analysis",
                       "features.pre_compact_reminder", "features.external_compact_request")
#: Past the window-change point, once "window full" has been sent, allow this many more tokens, then switch to a new
#: window (no summary written). The hook reads the previous request's token count (the rollout writes the current
#: request's token_count only after its tool finishes); over the 776 windows of 2026-10-20/21 the first crossing of the
#: change point overshot by up to 27.8k: 30k gives every window at least one chance to read the message and write notes.
GEN2_FORCE_BUFFER = 30_000
#: Both "window full" and the window change are done by the PostToolUse hook (gen2/hooks/post_tool_use.py): the message
#: is appended to the tool output, from then on act refuses ``do`` (the workspace's ``.context_full``, removed by the
#: next window's SessionStart), and the window change writes compact_request (patch 0004).
#: 2026-10-21 analysis (section "window changes not obeyed" of the gen-2 plan doc): the old approach reminded 15k before
#: the change point and sent a backstop at the point; the model wrote notes within 1-2 steps, but only 45% called
#: new_context themselves, the rest first did "one more cheap thing" until forced over (58% of the forced ones still
#: acted after their last notes write, and those results were not saved).
#: codex's own token_budget reminder is a developer message, and sglang's Responses merges every developer message into
#: the leading system message (serving_responses.py ``_construct_input_messages``): the model sees it only at the very
#: top of the prompt, and the whole prefix cache is invalidated (2026-10-20 round 1). So codex's threshold = the
#: change-point cap + this much; it changes windows only when the hook's number lags (a long think on one step jumps
#: past the forcing point) or the hook fails; its reminder is set at 1 token left, at the same checkpoint as its window
#: change, so in practice the model never reads it.
GEN2_BACKSTOP_TOKENS = GEN2_FORCE_BUFFER + 8192
GEN2_FULL = ("<context_window_full>\nThis context window is full. Stop playing: from now until the next window, "
             "`./act do` is refused. Finish this window by bringing your notes up to date, then call `new_context`:\n"
             "- notes.md: Current state, the status of each hypothesis, and under Next the test or batch you were "
             "about to run -- the next window starts from it. If a level cleared, its file in levels/ too.\n"
             "- To get them right you may read your files, `./act status` and `record/` to recall what happened "
             "since you last wrote them. Take the steps you need for that, and nothing else: no new experiments, "
             "analyses or scripts.\n"
             "What is not in your notes does not reach the next window.\n</context_window_full>")
#: the brief's "Context windows" must describe act's gate (compared with whitespace collapsed): the old brief (g2),
#: which describes a reminder, would say the wrong thing under this mechanism
GEN2_BRIEF_GATE = "from then until the next window `./act do` is refused"
#: the notes.md that SessionStart puts into the workspace of a brand-new game: the sections the g2b brief's
#: "Remembering: your workspace" describes (moved from hooks/common on 2026-10-21).
GEN2_NOTES_SKELETON = """# Notes

## Current state
(Where you are in this level and what your last test showed. Rewrite this before every `new_context`. The levels
cleared and actions spent come from the server at the start of every window: do not copy them here.)

## Confirmed rules
(Each rule with the step that showed it. Rules that hold across levels belong here.)

## Hypotheses
(One line each: the hypothesis -- untested, supported or refuted -- and the steps behind it.)

## Ruled out
(What you tried that does not work, and why.)

## Next
(The next one to three things to try, most promising first.)
"""
GEN2_PACKET_TAIL = "Continue playing from here."
#: Stuck review (the prompt set's stuck.md): from this window number on within the same level, the status pack's last
#: section is replaced by it. Counting in hooks/session_start.py. 2026-10-21 reading of six stuck stretches in
#: g2b / g2inl: a wrong explanation gets written more confidently window after window, and a new window spends the whole
#: window rebuilding without acting; most visible from the 3rd window on.
GEN2_STUCK_WINDOWS = 3
#: Long-think reminder (the prompt set's idle.md): sent each time this many output tokens pass within a window since its
#: start or the last action (hooks/post_tool_use.py). 2026-10-21 g2b: about 6.5 thinks of 25k+ per game (median 39k),
#: the excess being 17-18% of output; 20k would add 50% more such stretches.
GEN2_IDLE_TOKENS = 25_000
#: texts written into gen2_budget.json that the hook sends to the model: the prompt set's matching optional file if it
#: has one, else the g2b originals above.
GEN2_TEXT_FILES = {"full": "window_full.md", "skeleton": "notes_skeleton.md", "tail": "packet_tail.md"}
#: codex's token_budget reminder template (i.e. full) rejects empty text and text over this many bytes
CODEX_REMINDER_MAX_BYTES = 2000


def gen2_texts(pset: prompts.PromptSet) -> dict:
    defaults = {"full": GEN2_FULL, "skeleton": GEN2_NOTES_SKELETON, "tail": GEN2_PACKET_TAIL}
    texts = {key: pset.optional(f) or defaults[key] for key, f in GEN2_TEXT_FILES.items()}
    if "new_context" not in texts["full"] or not 0 < len(texts["full"].encode()) <= CODEX_REMINDER_MAX_BYTES:
        raise SystemExit(f"gen2: prompts {pset.name} window_full.md must name new_context and fit in "
                         f"{CODEX_REMINDER_MAX_BYTES} bytes (codex's reminder template)")
    texts["stuck"] = pset.texts.get("stuck.md")          # template; the hook fills in level / windows / actions
    texts["stuck_windows"] = GEN2_STUCK_WINDOWS
    texts["idle"] = pset.texts.get("idle.md")            # template; the hook fills in tokens
    texts["idle_tokens"] = GEN2_IDLE_TOKENS
    return texts


def gen2_config(cfg: dict, pset: prompts.PromptSet, driver_name: str) -> bool:
    harness = cfg.get("harness", "gen1")
    if harness not in ("gen1", "gen2"):
        raise SystemExit(f"harness must be gen1 or gen2, not {harness!r}")
    if harness == "gen1":
        if "new_context" in pset.texts["brief.md"]:
            raise SystemExit(f"prompts {pset.name} is a gen-2 brief (new_context) but the config is gen1")
        gen2_only = [f for f in (*GEN2_TEXT_FILES.values(), "stuck.md", "idle.md") if pset.has(f)]
        if gen2_only:
            raise SystemExit(f"prompts {pset.name} has the gen-2 hook texts {gen2_only} but the config is gen1")
        return False
    if driver_name != "codex":
        raise SystemExit(f"gen2 runs on codex's hooks and token_budget, not {driver_name}")
    clash = [k for k in GEN2_REFUSED_KEYS if k in cfg]
    if clash:
        raise SystemExit(f"gen2 does not take the gen-1 keys {clash}")
    clash = [k for k in GEN2_REFUSED_DRIVER if k in (cfg.get("driver_config") or {})]
    clash += [k for k in (cfg.get("driver_config") or {}) if k.startswith(("features.token_budget", "hooks."))]
    if clash:
        raise SystemExit(f"gen2 sets these itself or they are dead under token_budget: {clash}")
    if cfg.get("record") is None or not cfg.get("workspace_layout") or cfg.get("resume_mode") != "thread":
        raise SystemExit("gen2 needs record, workspace_layout and resume_mode thread")
    brief = pset.texts["brief.md"]
    for phrase in ("new_context", "./act note_done"):
        if phrase not in brief:
            raise SystemExit(f"gen2: prompts {pset.name} does not mention {phrase}")
    if GEN2_BRIEF_GATE not in " ".join(brief.split()):
        raise SystemExit(f"gen2: prompts {pset.name} does not say that `./act do` is refused once the window is full "
                         f"(g2 is the 10-20 brief, written for a reminder before the change point: use g2b)")
    if "./act init" in brief or pset.features:
        raise SystemExit(f"gen2: prompts {pset.name} is not a gen-2 set (./act init, or feature fragments)")
    if cfg["auto_compact_token_limit"] + GEN2_BACKSTOP_TOKENS > 0.95 * cfg["model_context_window"]:
        raise SystemExit("gen2: auto_compact_token_limit + the backstop must stay under 0.95 x the window")
    return True


#: each game's window-change point (written by the runner, read by the PostToolUse hook): the config's
#: auto_compact_token_limit; with dynamic context (plan #6) computed from the number of running games, capped at
#: auto_compact_token_limit.
GEN2_BUDGET = "gen2_budget.json"
DYNAMIC_KEYS = frozenset({"min_tokens", "pool_share", "safety"})
#: optional key for gen-1 only: the reminder point when the compaction point is at the cap (CompactionBudget.reminder_for)
DYNAMIC_GEN1_OPTIONAL = frozenset({"reminder_max_tokens"})


def dynamic_context_config(cfg: dict, gen2: bool) -> dict | None:
    """``dynamic_context: {min_tokens, pool_share, safety}``: point = clamp(safety × pool_share × KV pool ÷ running
    games, min_tokens, auto_compact_token_limit). ``pool_share`` is this run's share of the KV pool (0.5 each when two
    runs share one server). For gen-2 it is the window-change point (WindowBudget); for gen-1 the compaction point
    (CompactionBudget, see there)."""
    dyn = cfg.get("dynamic_context")
    if dyn is None:
        return None
    if not DYNAMIC_KEYS <= set(dyn) <= DYNAMIC_KEYS | (frozenset() if gen2 else DYNAMIC_GEN1_OPTIONAL):
        raise SystemExit(f"dynamic_context takes exactly {sorted(DYNAMIC_KEYS)}"
                         + ("" if gen2 else f" (gen-1: and optionally {sorted(DYNAMIC_GEN1_OPTIONAL)})")
                         + f", not {sorted(dyn)}")
    if not 0 < dyn["min_tokens"] < cfg["auto_compact_token_limit"]:
        raise SystemExit("dynamic_context.min_tokens must be in (0, auto_compact_token_limit)")
    for key in ("pool_share", "safety"):
        if not 0 < dyn[key] <= 1:
            raise SystemExit(f"dynamic_context.{key} must be in (0, 1]")
    if not gen2:
        reminder = cfg.get("compact_reminder_tokens")
        if not isinstance(reminder, int) or not 0 < reminder < dyn["min_tokens"]:
            raise SystemExit("gen-1 dynamic_context moves the pre-compaction reminder with the compaction point: "
                             "it needs compact_reminder_tokens below min_tokens")
        if (cfg.get("driver_config") or {}).get("features.external_compact_request") != "true":
            raise SystemExit("gen-1 dynamic_context compacts through compact_request: driver_config "
                             "features.external_compact_request=true (codex patch 0004)")
        if "level_gate" in cfg:
            raise SystemExit("gen-1 dynamic_context does not go with level_gate (both write compact_request)")
        top = dyn.get("reminder_max_tokens")
        if top is not None and not (isinstance(top, int) and reminder <= top < cfg["auto_compact_token_limit"]):
            raise SystemExit("gen-1 dynamic_context.reminder_max_tokens must be an int in "
                             "[compact_reminder_tokens, auto_compact_token_limit)")
        if cfg["auto_compact_token_limit"] > 0.9 * cfg["model_context_window"]:
            raise SystemExit("gen-1 dynamic_context: codex compacts at min(auto_compact_token_limit, 0.9 x window), "
                             "so a cap above 0.9 x model_context_window would compact before the dynamic point")
    return dict(dyn)


#: the two gen-2 switches of the 2026-10-26 run (prompts g2f): ``gen2_click_info`` = act prints the cell hit by every
#: ACTION6 (act_client.with_click_info); ``gen2_packet_levels`` = the status pack carries the full text of only this many
#: latest level files, older ones list only their Goal (hooks/common.packet). The brief must agree with the switches: on
#: but not described, or described but off, are both blocked (compared with whitespace collapsed).
GEN2_CLICK_BRIEF = "`./act do` prints what it hit"
GEN2_PACKET_BRIEF = "the goal of each older one"
#: what the brief must say with ``packet_level_docs: latest`` (moved from arc3-v6b 6675fecd3, 2026-10-28 hs110 v36):
#: the latest one in full, older ones by file name only
GEN2_LATEST_BRIEF = "the names of the older ones"
#: which levels/ files the gen-2 status pack carries: PACKET_LEVEL_DOCS in hooks/common.py (the hook uses only the
#: standard library and does not import this file; a test compares the two copies)
PACKET_LEVEL_DOCS = ("all", "latest")


def packet_level_docs_config(cfg: dict, gen2: bool, pset: prompts.PromptSet | None = None) -> str | None:
    """``packet_level_docs`` (since the 2026-10-27 RL single-level games; deployed 2026-10-28 in hs110 v36): all /
    latest, written into gen2_budget.json for SessionStart. Absent = None: the budget file lacks the key as before and
    the hook treats it as all. Mutually exclusive with ``gen2_packet_levels``. With ``pset`` given, check the brief:
    latest on without saying "file names only", or said without being on, are both blocked."""
    docs = cfg.get("packet_level_docs")
    if "packet_level_docs" in cfg and docs not in PACKET_LEVEL_DOCS:
        raise SystemExit(f"packet_level_docs must be one of {PACKET_LEVEL_DOCS}, not {docs!r}")
    if docs is not None and not gen2:
        raise SystemExit("packet_level_docs is a gen2 setting (the SessionStart hook builds the packet)")
    if docs is not None and "gen2_packet_levels" in cfg:
        raise SystemExit("packet_level_docs and gen2_packet_levels both set: pick one way to cut the packet's level files")
    if gen2 and pset is not None:
        on = docs == "latest"
        if on != (GEN2_LATEST_BRIEF in " ".join(pset.texts["brief.md"].split())):
            raise SystemExit(f"gen2: packet_level_docs is {docs or 'all'} but the brief of prompts {pset.name} "
                             f"{'does not say' if on else 'says'} {GEN2_LATEST_BRIEF!r}")
    return docs




def gen2_options_config(cfg: dict, gen2: bool, pset: prompts.PromptSet) -> tuple[bool, int | None]:
    click = cfg.get("gen2_click_info", False)
    levels = cfg.get("gen2_packet_levels")
    if not isinstance(click, bool):
        raise SystemExit(f"gen2_click_info must be true or false, not {click!r}")
    if levels is not None and (isinstance(levels, bool) or not isinstance(levels, int) or levels < 1):
        raise SystemExit(f"gen2_packet_levels must be a positive int, not {levels!r}")
    if not gen2:
        if "gen2_click_info" in cfg or "gen2_packet_levels" in cfg:
            raise SystemExit("gen2_click_info / gen2_packet_levels are gen2 features")
        return False, None
    brief = " ".join(pset.texts["brief.md"].split())
    for key, on, phrase in (("gen2_click_info", click, GEN2_CLICK_BRIEF),
                            ("gen2_packet_levels", levels is not None, GEN2_PACKET_BRIEF)):
        if on != (phrase in brief):
            raise SystemExit(f"gen2: {key} is {'on' if on else 'off'} but the brief of prompts {pset.name} "
                             f"{'does not say' if on else 'says'} {phrase!r}")
    return click, levels


#: gen-2 toolkit (config ``gen2_toolkit``, the toolkit loop since 2026-10-27): ``gen2/toolkit/<version>/MANIFEST.json``
#: lists the files placed verbatim into the workspace before game start (workspace-relative path -> {"from": file name
#: in that directory, "mode": "644"|"755"}) and a sentence the brief must contain (``brief``, compared with whitespace
#: collapsed). The files count toward the prompt fingerprint; each version is frozen once used, changes need a new one.
GEN2_TOOLKITS = UNIT / "gen2" / "toolkit"
#: MANIFEST's ``act`` (optional): act-side features to turn on (act_client.TOOLKIT_ACT); for each one the brief must
#: contain the start of its output (table below).
GEN2_TOOLKIT_ACT_BRIEF = {"anim": "animation:", "shift": "view shift:", "view": "lib/view.py",
                          "events": "without a RESET", "map": "view: the picture slid",
                          "map2": "load_map()", "movers": "movers(step)",
                          "state": "state(board, level)", "digest": "this call, steps"}
#: what a toolkit may not overwrite: the harness's act, the record written by the server, the notes skeleton placed by
#: the hook, the note_gate and window-change flags, and the map and state written by act's map feature
GEN2_TOOLKIT_RESERVED = ("act", "act.py", "record", "notes.md", "AGENTS.md", ".note_gate", ".context_full",
                         ".driver", "map", ".act_view")


def gen2_toolkit_config(cfg: dict, gen2: bool, pset: prompts.PromptSet, click_info: bool) -> dict | None:
    """The gen-2 key is ``gen2_toolkit``; gen-1 (2026-10-30 hs110 v38 = v35's v14 arm + tk5) uses ``act_toolkit``, with
    the same toolkit directory and the same checks, the act side stacked on ``act_click_info`` (gen1 in
    act_client.with_toolkit_act)."""
    key, other = ("gen2_toolkit", "act_toolkit") if gen2 else ("act_toolkit", "gen2_toolkit")
    if other in cfg:
        raise SystemExit(f"{other} is the {'gen-1' if gen2 else 'gen-2'} key; this config is "
                         f"{'gen2' if gen2 else 'gen1'}: use {key}")
    name = cfg.get(key)
    if name is None:
        return None
    if not gen2 and (cfg.get("workspace_tools") or not cfg.get("workspace_layout") or cfg.get("record") is None):
        raise SystemExit("act_toolkit puts lib/arc.py (which reads record/) in the workspace layout: turn "
                         "workspace_layout and record on, and workspace_tools (the RSI lib/arc.py) off")
    root = GEN2_TOOLKITS / str(name)
    if not (root / "MANIFEST.json").is_file():
        have = sorted(p.name for p in GEN2_TOOLKITS.iterdir()) if GEN2_TOOLKITS.is_dir() else []
        raise SystemExit(f"gen2_toolkit {name!r}: no {root}/MANIFEST.json (have {have})")
    manifest = json.loads((root / "MANIFEST.json").read_text())
    if not {"files", "brief"} <= set(manifest) <= {"files", "brief", "act"} or not manifest["files"] \
            or not manifest["brief"].strip():
        raise SystemExit(f"gen2_toolkit {name}: MANIFEST.json takes files (non-empty), brief and optionally act")
    act = tuple(manifest.get("act", ()))
    if not set(act) <= set(GEN2_TOOLKIT_ACT_BRIEF) or len(set(act)) != len(act):
        raise SystemExit(f"gen2_toolkit {name}: act must be distinct names from {sorted(GEN2_TOOLKIT_ACT_BRIEF)}")
    if act and not click_info:
        raise SystemExit(f"{key} {name}: the act features go on top of "
                         f"{'gen2_click_info' if gen2 else 'act_click_info'}")
    files = {}
    for rel, spec in manifest["files"].items():
        parts = Path(rel).parts
        if (Path(rel).is_absolute() or ".." in parts or not parts or parts[0] in GEN2_TOOLKIT_RESERVED
                or set(spec) != {"from", "mode"} or spec["mode"] not in ("644", "755")):
            raise SystemExit(f"gen2_toolkit {name}: bad entry {rel!r}: {spec!r}")
        files[rel] = ((root / spec["from"]).read_text(encoding="utf-8"), int(spec["mode"], 8))
    brief = " ".join(pset.texts["brief.md"].split())
    for phrase in (manifest["brief"], *(GEN2_TOOLKIT_ACT_BRIEF[a] for a in act)):
        if " ".join(phrase.split()) not in brief:
            raise SystemExit(f"gen2_toolkit {name}: the brief of prompts {pset.name} does not say {phrase!r}")
    return {"name": str(name), "files": files, "act": act}


def install_toolkit(ws: Path, toolkit: dict) -> None:
    for rel, (text, mode) in toolkit["files"].items():
        path = ws / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise RuntimeError(f"toolkit {toolkit['name']}: {path} already exists")
        path.write_text(text, encoding="utf-8")
        path.chmod(mode)


def seed_snapshots_config(cfg: dict, gen2: bool, recording: dict | None) -> bool:
    """``seed_snapshots: true`` (gen-2 seed collection, ``src/arcforge/arc_code/seed_snapshot.py``): the middleware
    snapshots the workspace before a level change, and the SessionStart hook asks it for a snapshot at the start of a
    new window; both go to ``<run>/seeds/``. Needs gen2 and record (a snapshot is the workspace plus the record)."""
    on = cfg.get("seed_snapshots", False)
    if not isinstance(on, bool):
        raise SystemExit(f"seed_snapshots must be true or false, not {on!r}")
    if on and not gen2:
        raise SystemExit("seed_snapshots is a gen2 feature (the window snapshots come from its SessionStart hook)")
    if on and recording is None:
        raise SystemExit("seed_snapshots needs record (a snapshot is the workspace with its record)")
    return on


def seed_note_done_config(cfg: dict, game_seed: dict | None) -> bool:
    """``seed_note_done: true`` (2026-10-27, RL single-level games): when loading a g2seed, write
    ``.note_gate/L(K-1).done`` (``note_done`` of level_seed.load) so level K can act right from its start. Only with
    ``--game-seed`` (gen-2 seeds are taken when note_done passes, so the notes are already written)."""
    on = cfg.get("seed_note_done", False)
    if not isinstance(on, bool):
        raise SystemExit(f"seed_note_done must be true or false, not {on!r}")
    if on and game_seed is None:
        raise SystemExit("seed_note_done goes with --game-seed (a gen-2 single-level episode)")
    return on


def pace_line_config(cfg: dict, gen2: bool) -> bool:
    """``pace_line: true`` (2026-10-27, RL single-level games): the last line of the gen-2 act's ``do`` / ``status``
    is the output tokens spent on this level (act_client's ``pace``). The number comes from the workspace's
    ``.pace.json``, written by run_arc3_level's proxy at the start and end of every request; act refuses if the file
    is missing (a harness bug), so turn it on only in a harness whose proxy writes the file (passed in through
    run_arc3_level's harness config)."""
    on = cfg.get("pace_line", False)
    if not isinstance(on, bool):
        raise SystemExit(f"pace_line must be true or false, not {on!r}")
    if on and not gen2:
        raise SystemExit("pace_line is a gen2 setting (the gen-2 act prints it)")
    return on


def level_snapshot_at_config(cfg: dict, seed_snapshots: bool) -> str:
    """``level_snapshot_at`` (2026-10-24): when the level-change snapshot is taken. ``first_action`` (default, the
    collection since 2026-10-21) = before level K's first action; ``note_done`` = when act's note_done passes
    (middleware ``--level-snapshot-at``). On 2026-10-23 we found that 64% of first_action seeds carried preliminary
    work on level K, some with the answer outright (docs/research/arc3_g2seed_collection_difficulty_20261023.md)."""
    at = cfg.get("level_snapshot_at", "first_action")
    if at not in ("first_action", "note_done"):
        raise SystemExit(f"level_snapshot_at must be first_action or note_done, not {at!r}")
    if "level_snapshot_at" in cfg and not seed_snapshots:
        raise SystemExit("level_snapshot_at goes with seed_snapshots: true")
    return at


def dynamic_limit(dyn: dict, cap: int, pool: int, active: int) -> int:
    """clamp(safety × pool_share × pool ÷ active, min_tokens, cap): gen-2's window-change point, gen-1's compaction
    point."""
    raw = int(dyn["safety"] * dyn["pool_share"] * pool / max(active, 1))
    return max(dyn["min_tokens"], min(cap, raw))


def server_kv_pool(base_url: str) -> int:
    """The server's KV pool (tokens; ``max_total_num_tokens`` of a single replica)."""
    root = base_url.split("/v1")[0]
    with urllib.request.urlopen(root + "/get_server_info", timeout=120) as r:
        pool = json.loads(r.read()).get("max_total_num_tokens")
    if not isinstance(pool, int) or pool <= 0:
        raise RuntimeError(f"{root}/get_server_info has no max_total_num_tokens (got {pool!r})")
    return pool


class WindowBudget:
    """Each game's window-change point, written into the driver home of every running game (``GEN2_BUDGET``). Written
    in run_slots' occupancy (inside the lock, before run_one), so the file exists before a game's first session
    starts. Without ``dyn`` it is always cap; with it, recomputed whenever the number of running games changes."""

    def __init__(self, dyn: dict | None, cap: int, pool: int | None, workspaces: dict[str, Path], texts: dict):
        if (dyn is None) != (pool is None):
            raise ValueError("a dynamic budget needs the KV pool, a fixed one does not")
        self.dyn, self.cap, self.pool, self.workspaces, self.texts = dyn, cap, pool, workspaces, texts
        self.limit: int | None = None

    def limit_for(self, active: int) -> int:
        if self.dyn is None:
            return self.cap
        return dynamic_limit(self.dyn, self.cap, self.pool, active)

    def occupancy(self, snapshot: dict) -> None:
        running = sorted(snapshot["running"])
        if not running:
            return
        limit = self.limit_for(len(running))
        if limit != self.limit:
            log(f"window budget: {len(running)} game(s) running, window change point {limit:,} tokens")
            self.limit = limit
        body = json.dumps({"limit": limit, "cap": self.cap, "active": len(running), "pool": self.pool,
                           "buffer": GEN2_FORCE_BUFFER, **self.texts})
        for game in running:
            home = self.workspaces[game] / ".driver"
            home.mkdir(parents=True, exist_ok=True)
            write_driver_file(home, GEN2_BUDGET, body)


def write_driver_file(home: Path, name: str, body: str) -> None:
    """The runner (root under posix) writes into a game's driver home: that directory belongs to the game uid, and the
    agent can put symlinks in it. The temp file is created with O_EXCL|O_NOFOLLOW (if something was planted first,
    raise rather than write through it), and os.replace swaps the directory entry itself without following symlinks.
    Also raise if home itself is a symlink."""
    if home.is_symlink() or not home.is_dir():
        raise RuntimeError(f"{home} is not a plain directory")
    tmp = home / f".{name}.{os.getpid()}.{threading.get_ident()}.tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644)
    data = body.encode()
    try:
        if os.write(fd, data) != len(data):
            raise RuntimeError(f"short write to {tmp}")
    finally:
        os.close(fd)
    os.replace(tmp, home / name)


#: the two files of gen-1 dynamic compaction, in codex's $CODEX_HOME (each game's .driver): patch 0004 consumes
#: compact_request at every turn boundary (its content is the threshold; below it the file is consumed without
#: compacting), patch 0005 rereads the first line of compact_reminder on every check.
COMPACT_REQUEST = "compact_request"
COMPACT_REMINDER = "compact_reminder"
#: how soon a consumed compact_request is put back. Two turn boundaries are at least one sampling plus one tool apart,
#: so a missing file only delays compaction by one boundary; if it is never put back, codex's own
#: auto_compact_token_limit (the cap) is the fallback.
COMPACTION_SYNC_SECONDS = 1.0


class CompactionBudget:
    """gen-1 dynamic compaction (config ``dynamic_context``, harness gen1). codex's auto_compact_token_limit is the cap,
    only a last line of defense; the compaction point L = dynamic_limit(running games) is triggered by patch 0004's
    compact_request (content L) at the first turn boundary after the context passes L -- through the same
    run_auto_compact as codex's own threshold (ContextLimit / MidTurn: summary, patches 0003 and 0006 all the same).
    patch 0005's reminder moves with it (reminder_for): linear interpolation between (min_tokens,
    compact_reminder_tokens) and (cap, reminder_max_tokens); without reminder_max_tokens it keeps the fixed distance
    min_tokens − compact_reminder_tokens. When so many games run that the computed point falls below min_tokens, it is
    the fixed min_tokens / compact_reminder_tokens (Kaggle v30: 100k / 85k). The only difference is at the start of a
    turn: codex's own pre-turn compaction looks only at the cap, so when a new turn starts with the context between L
    and the cap, compaction waits for the boundary after that turn's first sampling.

    occupancy (inside the run_slots lock) only records the running games and L. The files are reconciled by the
    ``start()`` thread every COMPACTION_SYNC_SECONDS: if compact_request is not L (consumed, or L changed), write L; if
    compact_reminder is not reminder_for(L) plus the prompt text, rewrite it (each play segment writes back the fixed
    value at its start). Games whose driver home is not prepared yet (no config.toml) are skipped; no directory is
    created for them. If the thread errors, it prints the traceback and ends; from then on only the cap remains."""

    def __init__(self, dyn: dict, cap: int, pool: int, reminder: int, text: str, workspaces: dict[str, Path]):
        top = dyn.get("reminder_max_tokens", cap - (dyn["min_tokens"] - reminder))
        if not 0 < reminder < dyn["min_tokens"] < cap or not reminder <= top < cap or not text.strip():
            raise ValueError("the reminder must sit below the compaction point at both ends, with a message")
        self.dyn, self.cap, self.pool, self.text, self.workspaces = dyn, cap, pool, text, workspaces
        self.reminder, self.top = reminder, top
        self.lock = threading.Lock()
        self.running: frozenset[str] = frozenset()
        self.limit: int | None = None
        self.stop = threading.Event()
        self.writes = {"compact_request": 0, "compact_reminder": 0}

    def limit_for(self, active: int) -> int:
        return dynamic_limit(self.dyn, self.cap, self.pool, active)

    def reminder_for(self, limit: int) -> int:
        low = self.dyn["min_tokens"]
        return self.reminder + (limit - low) * (self.top - self.reminder) // (self.cap - low)

    def occupancy(self, snapshot: dict) -> None:
        running = frozenset(snapshot["running"])
        with self.lock:
            self.running = running
            if not running:
                return
            limit = self.limit_for(len(running))
            if limit != self.limit:
                log(f"compaction budget: {len(running)} game(s) running, compaction at {limit:,}, "
                    f"reminder at {self.reminder_for(limit):,} tokens")
                self.limit = limit

    def sync(self) -> None:
        with self.lock:
            running, limit = self.running, self.limit
        if limit is None:
            return
        request, reminder = f"{limit}\n", f"{self.reminder_for(limit)}\n{self.text}"
        for game in sorted(running):
            home = self.workspaces[game] / ".driver"
            if not (home / "config.toml").is_file():
                continue
            for name, body in ((COMPACT_REMINDER, reminder), (COMPACT_REQUEST, request)):
                if self._read(home / name) != body:
                    self._replace(home / name, body)
                    self.writes[name] += 1

    @staticmethod
    def _read(path: Path) -> str | None:
        """None = no such file: codex consumes and deletes compact_request by design, and it can also vanish between
        exists and the read."""
        try:
            return path.read_text()
        except FileNotFoundError:
            return None

    @staticmethod
    def _replace(path: Path, body: str) -> None:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(body)
        tmp.replace(path)

    def start(self) -> threading.Thread:
        def loop() -> None:
            try:
                while not self.stop.wait(COMPACTION_SYNC_SECONDS):
                    self.sync()
            except Exception:
                log(f"COMPACTION BUDGET FAILED (only codex's own limit {self.cap:,} is left):\n"
                    f"{traceback.format_exc()}")
                raise
        thread = threading.Thread(target=loop, name="compaction-budget", daemon=True)
        thread.start()
        return thread


def gen2_driver_config(agent_python: str, full: str) -> tuple[tuple[str, str], ...]:
    """codex's -c: token_budget (no summary on any window change; codex's own reminder only shows up at the last line
    of defense, see GEN2_BACKSTOP_TOKENS), patch 0004 (the hook writes compact_request to change windows) and the three
    hooks. Values are TOML."""
    def hook(script: str) -> str:
        path = GEN2_HOOKS / script
        if not path.is_file():
            raise SystemExit(f"gen2: no hook {path}")
        return json.dumps(f"{shlex.quote(agent_python)} -I {shlex.quote(str(path))}")
    return (
        ("features.token_budget.enabled", "true"),
        ("features.token_budget.reminder_threshold_tokens", "1"),       # codex requires > 0: only when R reaches zero
        ("features.token_budget.reminder_message_template", json.dumps(full)),
        ("features.external_compact_request", "true"),
        ("hooks.SessionStart", '[{matcher="startup|resume|compact",hooks=[{type="command",command=%s,'
                               'timeout=60,additionalContextLimit=0}]}]' % hook("session_start.py")),
        ("hooks.Stop", '[{hooks=[{type="command",command=%s,timeout=60}]}]' % hook("stop.py")),
        ("hooks.PostToolUse", '[{hooks=[{type="command",command=%s,timeout=60}]}]' % hook("post_tool_use.py")),
    )


#: the middleware opens one game at a time (``_start`` waits for the official RESET inside its lock), and all slots
#: open at the same moment: later ones wait for every game ahead. About 20 ms per game on Kaggle, but 60 s is not
#: enough when the official server is slow; waiting longer costs nothing.
OPEN_GAME_SECONDS = 1800
#: retries when the connection is refused or reset (the middleware never received the request). 2026-10-20 Kaggle
#: v29g: 110 games opened at once, the middleware's listen backlog was only 5, and 5 games got ConnectionResetError
#: and died before supervisor.play (the middleware now uses 1024; this is a second safety net).
OPEN_GAME_TRIES = 5


def _never_reached(error: Exception) -> bool:
    """Connection-level failure: the request never reached the middleware. HTTP errors and timeouts do not count --
    the request may already have been processed, and resending RESET would reset the level."""
    if isinstance(error, urllib.error.HTTPError):
        return False
    reason = error.reason if isinstance(error, urllib.error.URLError) else error
    return isinstance(reason, ConnectionError)


def open_game(url: str, key: str, game: str) -> dict:
    """gen-2: the harness opens the game before the first session (the agent need not ./act init). If it is already
    open (continued session, tail resume), send nothing -- a RESET to an open game resets the level and costs an
    action."""
    def call(method: str, path: str, body: bytes | None, timeout: float) -> dict:
        req = urllib.request.Request(url + path, method=method, data=body)
        req.add_header("X-API-Key", key)
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    for attempt in range(1, OPEN_GAME_TRIES + 1):
        try:
            status = call("GET", "/api/status", None, 60)
            if status.get("started"):
                return status
            obs = call("POST", "/api/cmd/RESET", b"{}", OPEN_GAME_SECONDS)
            break
        except (ConnectionError, urllib.error.URLError) as error:
            if not _never_reached(error) or attempt == OPEN_GAME_TRIES:
                raise
            log(f"{game}: opening the game: {error!r}; asking again ({attempt}/{OPEN_GAME_TRIES})")
            time.sleep(attempt)
    if obs.get("state") != "NOT_FINISHED":
        raise RuntimeError(f"{game}: the harness opened the game but its state is {obs.get('state')!r}")
    return obs


def note_gate_features(cfg: dict) -> tuple[str, ...]:
    """Which features that bring their own prompt fragment (``prompts/_features/``: note_gate, click_info) the config
    turns on, in the order they are appended."""
    return tuple(name for name, on in (("note_gate", cfg.get("note_gate") is True),
                                       ("click_info", cfg.get("act_click_info") is True)) if on)


def act_click_info_config(cfg: dict, pset: prompts.PromptSet) -> bool:
    """``act_click_info`` (gen-1, 2026-10-27 hs110 v35 = v33 + this item): act prints the cell hit under every ACTION6
    (act_client.with_click_info, same as gen-2's ``gen2_click_info``). The text describing it is the feature fragment
    ``prompts/_features/click_info/``, appended to any set; a set that describes it itself is blocked (it would be
    said twice). Stacks only on the plain record actuator (checked by act_client.render)."""
    on = cfg.get("act_click_info", False)
    if not isinstance(on, bool):
        raise SystemExit(f"act_click_info must be true or false, not {on!r}")
    if on and GEN2_CLICK_BRIEF in " ".join(pset.texts["brief.md"].split()):
        raise SystemExit(f"prompts {pset.name} describes the click readout itself; "
                         "the text comes with the feature (prompts/_features/click_info)")
    return on


def note_gate_config(cfg: dict, pset: prompts.PromptSet, driver_name: str) -> bool:
    """``note_gate``: on a level clear, act withholds the next level and stops the game's other processes;
    ``./act note_done`` releases it only after confirming that levels/Lnn.md and notes.md were modified after the
    clear (src/arcforge/arc_code/act_client.py ``NOTE_GATE_HOOK``). The text describing it is the feature's own
    fragment ``prompts/_features/note_gate/``, appended to any set; sets do not describe it themselves.
    Lives only in act; it recognizes codex by its executable (``ARC_DRIVER_EXE``), so codex only."""
    enabled = cfg.get("note_gate", False)
    if not isinstance(enabled, bool):
        raise SystemExit("note_gate must be true or false")
    if "./act note_done" in pset.texts["brief.md"]:
        raise SystemExit(f"prompts {pset.name} describes ./act note_done itself; "
                         "the text comes with the feature (prompts/_features/note_gate)")
    if enabled != ("note_gate" in pset.features):
        raise SystemExit(f"note_gate={enabled} but the prompts were loaded with features {pset.features}")
    if not enabled:
        return False
    if cfg.get("record") is None or not cfg.get("workspace_layout"):
        raise SystemExit("note_gate requires record and workspace_layout (it reads record/, asks for levels/Lnn.md)")
    if driver_name != "codex":
        raise SystemExit(f"note_gate finds the driver by its executable: codex only, not {driver_name}")
    for key in ("level_gate", "reviewer"):
        if key in cfg:
            raise SystemExit(f"note_gate does not go with {key}")
    return True


def level_gate_config(cfg: dict, pset: prompts.PromptSet) -> dict | None:
    """``level_gate`` (prompts v9): after a level clear, the middleware waits for levels/Lnn.md before letting play
    continue, then asks codex to compact (features.external_compact_request from patch 0004). The brief, the layout
    and the codex feature must all match -- with one missing, the agent is blocked by a gate it does not know about,
    or the compaction never comes."""
    gate = cfg.get("level_gate")
    speaks = "The server takes no further action until that file exists" in pset.texts["brief.md"]
    if gate is None:
        if speaks:
            raise SystemExit(f"prompts {pset.name} describes the level gate, but the config has no level_gate")
        return None
    if not speaks:
        raise SystemExit(f"config has level_gate, but prompts {pset.name} does not tell the agent")
    if not isinstance(gate, dict) or set(gate) != {"min_chars", "compact_min_tokens"} \
            or not all(isinstance(v, int) and v >= 0 for v in gate.values()):
        raise SystemExit(f"level_gate must be {{min_chars, compact_min_tokens}} (non-negative ints), got {gate!r}")
    if not cfg.get("workspace_layout"):
        raise SystemExit("level_gate needs workspace_layout (it creates levels/)")
    if (cfg.get("driver_config") or {}).get("features.external_compact_request") != "true":
        raise SystemExit("level_gate needs driver_config features.external_compact_request=true (codex patch 0004)")
    return gate


def check_video_input(base_url: str, name: str, ffmpeg: str, tmp_dir: Path) -> int:
    """Check that the model server really decodes our mp4 and counts it in the prompt. Returns the tokens the video adds.

    Decoding happens on the sglang side (torchcodec + FFmpeg libraries); if they are missing, view_video blows up only
    at decode time, and the agent just sees an incomprehensible error. So before the run, send one clip encoded the
    same way.
    """
    import base64

    import numpy as np
    frames = np.zeros((4, 64, 64), dtype=np.uint8)
    for i in range(4):
        frames[i, 20:30, 10 * i:10 * i + 10] = 8
    clip = record.mp4(frames, ffmpeg=ffmpeg, fps=2.0, size=384, tmp_dir=tmp_dir)
    url = "data:video/mp4;base64," + base64.b64encode(clip).decode()

    def prompt_tokens(content) -> int:
        body = json.dumps({"model": name, "max_tokens": 1,
                           "messages": [{"role": "user", "content": content}]}).encode()
        req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return int(json.loads(r.read())["usage"]["prompt_tokens"])

    text = {"type": "text", "text": "What moves in this clip?"}
    extra = prompt_tokens([{"type": "video_url", "video_url": {"url": url}}, text]) - prompt_tokens([text])
    # 4 frames at 384² = 2 temporal patches × 144 tokens (docs/research/codex_video_input_20260921.md).
    if extra < 200:
        raise RuntimeError(f"{base_url}: a 4-frame clip added only {extra} prompt tokens; "
                           "the server is not decoding the video")
    log(f"video check: a 4-frame 384px clip adds {extra} prompt tokens")
    return extra


def failed_play(previous: dict, game: str, outcome: BaseException) -> dict:
    """The ARM_STATE play row of a game whose session loop raised: ``error`` is a one-line repr (run_arc3_level puts it
    into failures.jsonl and the orchestrator computes quarantine signatures from it, so nothing multi-line goes
    there); the full traceback is in ``error_traceback``."""
    return {"sessions": [], **previous, "game": game, "status": "failed", "error": repr(outcome),
            "error_traceback": "".join(traceback.format_exception(outcome))}


#: the harness's own driver logs (each session's stdout / stderr) live in the run directory's ``driver_logs/<game>/``,
#: **not** in the workspace. The workspace is agent-writable (jail.sh --binds it into the jail; fence_posix.sh chowns
#: it to the game's user): in 2026-10-20 RL an agent mv'd driver.2.log into trash/, and in v4b su15__s9 an agent mv'd
#: driver.log into old/; both times the whole game was judged a harness failure. jail.sh mounts only the workspace and
#: the driver home, so this is invisible there; under the posix fence this directory is 700 and owned by the harness
#: (like gateway/, driver_env/).
DRIVER_LOGS = "driver_logs"


def driver_log_path(run_dir: Path, game: str, attempt: int) -> Path:
    """Driver log of session number ``attempt``: ``driver_logs/<game>/driver.log``, ``driver.2.log``, ..."""
    return run_dir / DRIVER_LOGS / game / ("driver.log" if attempt == 1 else f"driver.{attempt}.log")


def require_driver_log(log_path: Path, attempt: int) -> Path:
    """After a session, the driver log the harness itself opened (``log_path``, written by _session_bwrap /
    _session_posix) must exist. It is outside the workspace (``DRIVER_LOGS``) where the agent cannot reach it, so its
    absence can only be a harness bug: raise."""
    if not log_path.is_file():
        raise RuntimeError(f"session {attempt}: the harness's own driver log {log_path} is missing after the session; "
                           f"it lives outside the agent's workspace, so this is a harness bug")
    return log_path


def last_turn_failure(driver_log: Path) -> str | None:
    """The last ``turn.failed`` / ``error`` message of codex --json; None if the session ended normally. Raise if the
    log is missing (it always exists after a session, require_driver_log)."""
    if not driver_log.is_file():
        raise RuntimeError(f"driver log {driver_log} does not exist: it is written by every session (require_driver_log)")
    failure = None
    for line in driver_log.read_text(errors="replace").splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.failed":
            failure = str((event.get("error") or {}).get("message") or event)
        elif event.get("type") == "turn.completed":
            failure = None
    return failure


#: /dev, /tmp, /run, /proc, which jail.sh mounts after the extra read-only mounts: paths below them get covered even
#: if mounted, and are invisible inside the jail
JAIL_SHADOWED = ("/dev", "/tmp", "/run", "/proc")


def jail_ro_paths(paths: list[str], fence: str) -> list[str]:
    """Paths for ARCSEC_EXTRA_RO; under bwrap, raise if one falls under a directory the jail covers (codex could not
    read the driver files and would exit 1 at game start). posix has no such covering (/tmp is shared; Kaggle's
    tokenizer is in /tmp/arcrun); readability there is checked by fence_posix.sh's ARCSEC_FENCE_REQUIRE as the player."""
    if fence == "posix":
        return paths
    if fence != "bwrap":
        raise ValueError(f"fence must be bwrap or posix, not {fence!r}")
    shadowed = [p for p in paths if any(p == d or p.startswith(d + "/") for d in JAIL_SHADOWED)]
    if shadowed:
        raise SystemExit(f"jail read-only paths under {JAIL_SHADOWED} are hidden inside the jail: {shadowed}; "
                         "copy them out first")
    return paths


#: jail.sh's own exit codes (fence signals, not agent behavior): 70 corpus visible, 71 bridge / health check failed,
#: 64 broken resource declaration
JAIL_CODES = (70, 71, 64)


def level_death(code: int | None, unit_result: str | None, driver_log: Path, started: bool) -> dict | None:
    """In a single-level episode (RL), whether this session counts as codex "dying": an abnormal end caused by the
    model itself.

    - OOM-killed after hitting the scope's MemoryMax (systemd then tears down the whole scope, codex with it): the
      agent's program blew up memory -> ``oom``;
    - codex exited non-zero, not with a jail.sh fence code, and the last turn had no model-side error
      (``turn.failed``) -> ``exit`` (e.g. the agent killed codex).

    A normal end (0, the agent stopped by itself) is nudged and resumed as before; fence codes and model-side errors
    are infrastructure matters, left to the existing branches. wall (124) and the stop files are judged by the caller
    first.

    ``started`` = the driver got this session up (codex's thread.started). A non-zero exit without that cannot be the
    model's behavior (2026-10-18 prefilter: the jail could not see the tokenizer, and all 400 episodes exited 1 in
    1.5 s): raise; it must not become a negative sample."""
    if not started and code not in (0, None) and code not in JAIL_CODES and not was_oom(unit_result):
        raise RuntimeError(f"level episode: the driver exited {code} without starting a session "
                           f"(infrastructure, not the model); see {driver_log}")
    if was_oom(unit_result):
        return {"cause": "oom", "returncode": code, "unit_result": unit_result}
    if code in (0, None) or code in JAIL_CODES:
        return None
    if last_turn_failure(driver_log) is not None:
        return None
    return {"cause": "exit", "returncode": code, "unit_result": unit_result}


def model_healthy(base_url: str) -> bool:
    root = base_url.split("/v1")[0]
    try:
        with urllib.request.urlopen(root + "/health", timeout=10) as r:
            return r.status == 200
    except OSError:
        return False


def wait_model_healthy(base_url: str, deadline: float, game: str, every: float = 10.0) -> None:
    """Do not open the next session until the model server is back (the notebook's watchdog restarts it).
    If it is still not back at this game's deadline, let the caller's deadline check wrap up."""
    if model_healthy(base_url):
        return
    began = time.monotonic()
    log(f"{game}: model server down, waiting for it to come back")
    while time.monotonic() < deadline:
        time.sleep(every)
        if model_healthy(base_url):
            log(f"{game}: model server is back (waited {time.monotonic() - began:.0f}s)")
            return
    log(f"{game}: model server still not back at the deadline")


def check_model(base_url: str, name: str) -> None:
    """The model name served on the port must be the one we want. A TCP connection means nothing. Timeout 120 s: on a
    fully loaded replica the front end (event loop) queues for over 10 s when a group of 16 episodes start at once
    (2026-10-19 RL A/B, see server_context_len)."""
    root = base_url.split("/v1")[0]
    with urllib.request.urlopen(root + "/v1/models", timeout=120) as r:
        served = {m.get("id") for m in json.loads(r.read()).get("data", [])}
    if name not in served:
        raise RuntimeError(f"{root}/v1/models serves {sorted(served)}, not {name!r}")
    log(f"model check: {root} serves {name}")


def close_card(url: str, key: str) -> dict:
    req = urllib.request.Request(url + "/api/scorecard/close", method="POST",
                                 data=b"{}")
    req.add_header("X-API-Key", key)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read())


def official_final(card: dict, summary: dict | None, games: list[str],
                   roster: list[str]) -> dict:
    """Score = the official scorecard. The ledger's RHAE is only a cross-check, and a mismatch must be **visible**.

    ``summary`` is None: no corpus (the rerun's shape), only the card -- then there is nothing to cross-check, and the
    ledger fields stay None, not 0."""
    by_id = {e["id"]: e for e in card.get("environments", [])}
    ours = {row["game_id"]: row for row in (summary or {}).get("games", [])}
    rows, mismatches = [], []
    for game in roster:
        env = by_id.get(game)
        official = float(env["score"]) if env else 0.0
        mine = float(ours.get(game, {}).get("score", 0.0)) if summary is not None else None
        row = {"game_id": game, "official_score": official,
               "ledger_rhae": mine, "diff": round(official - mine, 6) if mine is not None else None,
               "actions": env.get("actions") if env else 0,
               "levels_completed": env.get("levels_completed") if env else 0,
               "level_count": env.get("level_count") if env else None,
               "completed": bool(env.get("completed")) if env else False,
               "played": env is not None}
        if row["diff"] is not None and abs(row["diff"]) > 1e-6:
            mismatches.append(game)
        rows.append(row)
    mean = sum(r["official_score"] for r in rows) / len(rows) if rows else 0.0
    return {
        "source": "official-scorecard",
        "mean_score": round(mean, 4),
        "card_score_as_reported": card.get("score"),
        "denominator": len(rows),
        "games_played": sum(1 for r in rows if r["played"]),
        "wins": sum(1 for r in rows if r["completed"]),
        "total_levels": sum(r["levels_completed"] or 0 for r in rows),
        "total_actions": sum(r["actions"] or 0 for r in rows),
        "ledger_mean": (summary or {}).get("mean_score"),
        "score_mismatches": mismatches,
        "games": rows,
    }


def render_final(final: dict) -> str:
    lines = [f"=== FINAL (official card; denominator {final['denominator']} games) ===",
             f"mean {final['mean_score']:.2f}   wins {final['wins']}   "
             f"levels {final['total_levels']}   actions {final['total_actions']}   "
             f"played {final['games_played']}/{final['denominator']}",
             f"ledger RHAE mean {final['ledger_mean']}  "
             + ("(matches per game)" if not final["score_mismatches"]
                else f"**mismatch: {final['score_mismatches']}**"),
             "", f"{'game':16s} {'official':>9s} {'ledger':>9s} {'acts':>7s} {'lv':>6s} win"]
    for r in final["games"]:
        ledger = f"{r['ledger_rhae']:9.2f}" if r["ledger_rhae"] is not None else f"{'-':>9s}"
        lines.append(f"{r['game_id']:16s} {r['official_score']:9.2f} {ledger} "
                     f"{r['actions'] or 0:7d} {str(r['levels_completed'])+'/'+str(r['level_count']):>6s} "
                     f"{'WIN' if r['completed'] else ''}")
    return "\n".join(lines) + "\n"


def open_card(url: str, key: str) -> str:
    req = urllib.request.Request(
        url + "/api/scorecard/open", method="POST",
        data=json.dumps({"tags": ["arc-code"]}).encode())
    req.add_header("X-API-Key", key)
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.loads(response.read())["card_id"]


def open_seeded_game(url: str, key: str, game: str, level: int, prior: int) -> dict:
    """A prior-levels-seed game is opened by the launcher (middleware ``--prior-actions``: the agent's key gets 409 for
    everything before the game is open), before codex starts, so that the opening message's "the game is already open"
    is true. When the opening frame does not match the seed, the middleware writes ``start_failed`` and returns 500,
    and this raises in turn."""
    req = urllib.request.Request(url + "/api/cmd/RESET", method="POST", data=json.dumps({"game_id": game}).encode())
    req.add_header("X-API-Key", key)
    req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            obs = json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{game}: the seeded level episode did not open: {error.code} "
                           f"{error.read().decode(errors='replace')[:400]}") from None
    if obs.get("levels_completed") != level - 1 or obs.get("record") != {"step": prior, "resumed_from_seed": True}:
        raise RuntimeError(f"{game}: the seeded open returned levels_completed {obs.get('levels_completed')} / "
                           f"record {obs.get('record')}, want {level - 1} / step {prior}")
    return obs


#: ``workspace_layout`` (prompts v8 on): directories created before game start; the brief says what goes where.
LAYOUT_DIRS = ("lib", "scripts", "levels")


def workspace(run_dir: Path, game: str, brief: str, brief_file: str | None,
              fence: str = "bwrap", recording: bool = False, layout: bool = False,
              observation_tools: bool = False, receipt: bool = False, scene: bool = False,
              images: bool = False, library: bool = False, recap: bool = False, *,
              expect: bool = False, notes_budget: int | None = None, tools: bool = False,
              digest: bool = False, note_gate: bool = False, gen2: bool = False,
              note_done_snapshot: bool = False, click_info: bool = False, toolkit: dict | None = None,
              toolkit_act: tuple[str, ...] = (), pace: bool = False) -> Path:
    ws = run_dir / "games" / game
    ws.mkdir(parents=True, exist_ok=True)
    act_client.write(ws / "act.py", record=recording, observation_tools=observation_tools, receipt=receipt,
                     scene=scene, images=images, library=library, recap=recap,
                     expect=expect, notes_budget=notes_budget, digest=digest, note_gate=note_gate, gen2=gen2,
                     note_done_snapshot=note_done_snapshot, click_info=click_info, toolkit_act=toolkit_act, pace=pace)
    if layout:
        for name in LAYOUT_DIRS:
            (ws / name).mkdir(exist_ok=True)
    if toolkit is not None:
        install_toolkit(ws, toolkit)
    if tools:
        # bwrap: the workspace is the player's; posix (Kaggle): fence_posix.sh chowns the workspace to the game's user
        # (only act / act.py go back to root), so lib/arc.py, lib/bg.py and bg all belong to the player, consistent
        # with the brief's "if it is wrong, fix it in place"
        workspace_tools.install(ws)
    if recording:
        # written by the middleware, read-only for the agent (jail.sh mounts it back read-only; fence_posix.sh leaves
        # it to root). 755 must be explicit: the local umask 002 would make it group-writable, which record.py rejects.
        rec = ws / "record"
        if rec.is_symlink():
            raise RuntimeError(f"{rec} is a symlink; the record must be a real directory")
        rec.mkdir(exist_ok=True)
        rec.chmod(0o755)
    # act is a shim pointing at the agent runtime -- there is no harness venv in the jail. bwrap mounts act.py
    # read-only at /act.py; posix has no mounts, so it uses the copy in the workspace (the fence leaves it to root).
    act_path = "/act.py" if fence == "bwrap" else str(ws / "act.py")
    (ws / "act").write_text(f'#!/bin/sh\nexec python3 {act_path} "$@"\n')
    (ws / "act").chmod(0o755)
    # No more scorecard.json: the new act.py does not need card_id (the middleware injects it), and audit #1
    # saw ka59 agonize for a whole turn over "AGENTS.md says the score is hidden, but here is a scorecard.json".
    # The brief is the first prompt and is **also** written as the file the driver re-injects automatically
    # (AGENTS.md for both codex and pi): the only mechanism that keeps the brief in context after a compaction.
    # A driver without such a file (BRIEF_FILE=None) only gets the prompt; ARM_STATE records the named downgrade.
    if brief_file is not None:
        (ws / brief_file).write_text(brief)
    return ws


#: The two ways to continue. ``thread``: keep going in the same conversation (codex ``exec resume`` / pi
#: ``--session``, plus a one-line nudge), the driver decides whether to compact first; ``fresh``: open a new
#: session with a prompts/<set>/resume.md handover (the only way before 2026-09-18; the model re-derives from scratch).
RESUME_MODES = ("thread", "fresh")


def session_mode(resume_mode: str, attempt: int, last_handle: str | None) -> str:
    """How this session starts: ``fresh`` (new conversation) or ``resume`` (continue the old one).

    The first session is always fresh. In ``thread`` mode with no conversation to continue (the driver never came
    up -- a failed health check, killed at start-up) it falls back to fresh, and the caller must log that.
    """
    if resume_mode not in RESUME_MODES:
        raise ValueError(f"resume_mode must be one of {RESUME_MODES}, got {resume_mode!r}")
    if attempt == 1 or resume_mode == "fresh" or last_handle is None:
        return "fresh"
    return "resume"


def size_kb(text: str | int) -> int:
    """``"16G"`` / ``"16384M"`` / ``"1000K"`` / int KB -> KB. Raises on anything else."""
    if isinstance(text, int):
        return text
    s = str(text).strip().upper()
    units = {"K": 1, "M": 1024, "G": 1024 * 1024, "T": 1024 ** 3}
    if s[-1:] in units and s[:-1].isdigit():
        return int(s[:-1]) * units[s[-1]]
    if s.isdigit():
        return int(s)
    raise ValueError(f"unparseable size {text!r} (expected like 16G / 16384M / int KB)")


def was_oom(unit_result: str | None) -> bool:
    return bool(unit_result) and "oom" in unit_result


def capability_problems(driver, cfg: dict) -> list[str]:
    """Whatever the config asks for that the driver cannot provide is reported before the run, not silently downgraded."""
    caps = driver.CAPABILITIES
    problems = []
    if cfg["resume_mode"] == "thread" and "resume" not in caps:
        problems.append("resume_mode=thread but the driver has no resume (use fresh, or another driver)")
    if cfg.get("auto_compact_token_limit") is not None and "auto_compact" not in caps:
        problems.append("auto_compact_token_limit is set but the driver has no auto_compact")
    if cfg.get("effort") is not None and "effort" not in caps:
        problems.append("effort is set but the driver has no effort")
    if cfg.get("driver_files") and "driver_files" not in caps:
        problems.append(f"driver_files is set but {driver.NAME} has no driver_files")
    if cfg.get("driver_config") and "driver_config" not in caps:
        problems.append(
            f"driver_config {sorted(cfg['driver_config'])} is set but {driver.NAME} "
            "has no driver_config -- those keys would be ignored entirely and the whole run would finish believing they were set")
    return problems


def server_context_len(base_url: str) -> int:
    """The server's real context_length. pi derives max_completion_tokens from it (see arms/pi.py), and preflight also
    blocks < 200000. Timeout 120 s: a fully loaded replica's front end (event loop) answers one at a time, and a group
    of 16 episodes starting at once queues for over 20 s (2026-10-19 RL A/B: 0.7 s per /v1/models)."""
    root = base_url.split("/v1")[0]
    with urllib.request.urlopen(root + "/get_server_info", timeout=120) as r:
        info = json.loads(r.read())
    ctx = next((info[k] for k in ("context_length", "max_context_len", "context_len")
                if isinstance(info.get(k), int)), None)
    if ctx is None:
        raise RuntimeError(f"{root}/get_server_info has no context_length")
    return ctx


#: ``game_output_tokens``: a game's session stopping with this exit code = output tokens reached the cap (124 is the
#: wall clock).
TOKEN_BUDGET_CODE = 125
TOKEN_POLL_SECONDS = 15.0


class GameOutputTokens:
    """A game's cumulative codex output tokens: the sum of ``output_tokens`` of the ``token_usage_record`` of every
    distinct ``response_id`` in the rollouts -- the same rule as ``extract_codex`` in analysis/roundtrips.py, which is
    the clock used for comparisons. Keeps an offset per file and reads only newly completed lines each time."""

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


def wait_session(proc: subprocess.Popen, left: float, over, poll: float = TOKEN_POLL_SECONDS) -> int:
    """Wait for the session to end. At ``left`` seconds, like ``subprocess.run(timeout=)``: kill it and return 124;
    when ``over()`` is true (this game's output reached game_output_tokens): kill it and return TOKEN_BUDGET_CODE.
    The scope is cleaned up by the caller's cleanup_scope (the same path as a timeout)."""
    deadline = time.monotonic() + left
    while True:
        try:
            return proc.wait(timeout=max(0.0, min(poll, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            pass
        code = 124 if time.monotonic() >= deadline else TOKEN_BUDGET_CODE if over() else None
        if code is not None:
            proc.kill()
            proc.wait()
            return code


def game_progress(ledger_path: Path, game: str) -> tuple[bool, int]:
    """(did this game win, counted actions). Reads the ledger, because it is the only authoritative record."""
    won, actions, _ = _progress(ledger_path, game)
    return won, actions


def _progress(ledger_path: Path, game: str) -> tuple[bool, int, int]:
    """Plus the highest level so far, as handover information for a continued session."""
    won, actions, levels = False, 0, 0
    if not ledger_path.exists():
        return won, actions, levels
    for line in ledger_path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue                      # the server is writing; the last line may be partial
        # A full name is matched exactly. The old code compared short names, so Save & Run's copies (ar25-0c556536,
        # ar25-c20c556536, ...) shared one progress: in v4 all five ar25 copies were recorded with the five copies'
        # combined 894 actions, and ft09-c40 was recorded as WIN without clearing because another copy won. Only
        # short names (b200 corpus, where short names are unique) are prefix-matched.
        mine = row["game"] == game if "-" in game else row["game"].split("-")[0] == game
        if not mine:
            continue
        if row.get("counted"):
            actions += 1
        levels = max(levels, int(row.get("levels_completed") or 0))
        if row.get("state") == "WIN":
            won = True
    return won, actions, levels


def ledger_standing(ledger_path: Path, game: str) -> tuple[bool, int, int, int | None]:
    """(won, counted actions, levels_completed, win_levels), for ``tail_resume`` to order the resumed games.

    Reads the committed part of the ledger (the last line may be partial; skip it); any other bad line raises. While a
    game has no line yet (the agent never opened it), ``win_levels`` is None and it sorts last."""
    raw = ledger_path.read_bytes()
    rows = []
    for line in raw[:raw.rfind(b"\n") + 1].splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row["game"] == game if "-" in game else row["game"].split("-")[0] == game):
            rows.append(row)
    if not rows:
        return False, 0, 0, None
    last = rows[-1]
    return (any(r["state"] == "WIN" for r in rows), sum(1 for r in rows if r["counted"]),
            int(last["levels_completed"]), int(last["win_levels"]))


def recovery_checkpoint(ledger_path: Path, game: str) -> dict:
    """One committed ledger prefix; never reconstruct state from agent prose.

    Other games may still append. Only an unfinished last line is excluded;
    malformed committed rows or multiple game identities fail loudly.
    """
    raw = ledger_path.read_bytes()
    boundary = raw.rfind(b"\n") + 1
    committed = raw[:boundary]
    rows = []
    for number, line in enumerate(committed.splitlines(keepends=True), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        mine = row["game"] == game if "-" in game else row["game"].split("-")[0] == game
        if mine:
            rows.append((number, row, line))
    if not rows:
        raise RuntimeError(f"{game}: no authoritative game checkpoint for context recovery")
    identities = {(r["game"], r["guid"], r["play"], r["card_id"], r["authority"]) for _, r, _ in rows}
    if len(identities) != 1:
        raise RuntimeError(f"{game}: ambiguous game identity in recovery ledger")
    number, latest, line = rows[-1]
    if not latest["guid"] or latest["authority"] not in {"middleware", "local-gateway"}:
        raise RuntimeError(f"{game}: recovery requires an observed live game GUID")
    if any(type(r["counted"]) is not bool for _, r, _ in rows):
        raise ValueError("recovery ledger counted must be boolean")
    return {"step": sum(r["counted"] for _, r, _ in rows),
            "levels_completed": latest["levels_completed"], "win_levels": latest["win_levels"],
            "game_id": latest["game"], "guid": latest["guid"], "play": latest["play"],
            "card_id": latest["card_id"], "authority": latest["authority"],
            "source": {"path": str(ledger_path.resolve()), "prefix_bytes": boundary,
                       "prefix_sha256": hashlib.sha256(committed).hexdigest(),
                       "excluded_tail_bytes": len(raw) - boundary, "line": number,
                       "line_sha256": hashlib.sha256(line).hexdigest()}}


GAME_ORDERS = ("sorted", "interleave")


def interleave_copies(ids: list[str]) -> list[str]:
    """``game_order: interleave``: keep copies of the same game (Save & Run resamples 25 games into 110) apart. First one
    original of each game, then a second of each, ... Sorted order aaaabbbb puts only six distinct games in the first
    28; this way the first 25 are 25 different games. The original is the copy with the shortest id (copies add
    ``c<k>`` before the version), ties broken alphabetically. The rerun's games are all distinct, so there this order
    equals sorting by short name."""
    groups: dict[str, list[str]] = {}
    for game in ids:
        groups.setdefault(game.split("-")[0], []).append(game)
    columns = [sorted(g, key=lambda x: (len(x), x)) for _, g in sorted(groups.items())]
    return [col[i] for i in range(max(map(len, columns), default=0)) for col in columns if i < len(col)]


def run_slots(work: list, concurrency: int, base_seconds: float | None, deadline: float,
              run_one, *, standing=None, min_resume_seconds: float | None = None, occupancy=None) -> list:
    """``concurrency`` slots work through ``work`` in order; returns results in the same order as ``work``.

    With ``base_seconds`` given, each slot is its own timeline: a game gets the base budget plus the time **the
    previous game on the same slot left unused**, counted from the moment it actually starts. A game that WINs at
    4,000 s hands its remaining 4,000 s to the next one (8,000 + 4,000), whose end still falls on the original round
    boundary, so no slot ever runs past "rounds × base budget" -- which is already ≤ wall. WIN, a give-up verdict,
    sessions used up, killed: any early end hands over the same way. ``None`` (admission mode): only the global
    deadline; admission computes each game's time itself, no carry-over.

    ``standing(game) -> (won, actions, levels, win_levels)`` (``tail_resume``, ``base_seconds`` mode only): once the
    main queue is empty, freed slots do not sit idle. A game that ended without WIN (time up, gave up, sessions used
    up all count; ``run_one`` raising does not) enters the **resume pool**; a free slot picks the game closest to
    clearing (smallest ``win_levels - levels``, ties to larger ``levels``, then to whoever entered the pool first),
    with the global deadline as its deadline, and ``run_one``'s ``slot_info`` carries ``resume`` = which segment this
    is (≥1). A resume segment that adds no action (gave up again) does not return to the pool, to avoid spinning; a
    game is on at most one slot at a time. ``min_resume_seconds``: a segment opens only if the remaining global time
    is **greater** than this (the caller passes the config's ``tail_resume_min_seconds``). A slot waits while the
    pool is empty but other slots still have games running: those may still enter the pool.

    Exceptions raised by ``run_one(game, ws, deadline, slot_info)`` go into the results as they are, for the caller to
    record -- one game blowing up must not stop its slot.

    ``occupancy(snapshot)`` (optional, used by the reviewer's scheduler): called inside the lock whenever a slot's
    state changes, with ``snapshot`` = ``{"running": games running, "pooled": games in the resume pool, "queued":
    games left in the main queue}``. Read-only; it does not change the scheduling.
    """
    if standing is not None and (base_seconds is None or min_resume_seconds is None):
        raise ValueError("tail resume needs base_seconds mode and min_resume_seconds")
    queue, results = list(work), {}
    cond = threading.Condition()
    pool: list[tuple[tuple, int, str, object]] = []   # (priority, entry order, game, ws)
    segments: dict[str, int] = {}
    running = 0
    entries = 0
    active: set[str] = set()

    def report() -> None:
        """The caller holds cond."""
        if occupancy is not None:
            occupancy({"running": frozenset(active), "pooled": frozenset(e[2] for e in pool),
                       "queued": len(queue)})

    def enter_pool(game: str, ws, acted_before: int | None) -> None:
        """After a segment ends, decide whether this game enters the pool (the caller holds cond)."""
        nonlocal entries
        won, actions, levels, win_levels = standing(game)
        if won or (acted_before is not None and actions <= acted_before):
            return
        remaining = float("inf") if win_levels is None else win_levels - levels
        entries += 1
        pool.append(((remaining, -levels), entries, game, ws))

    def take(index: int):
        """(game, ws, resume segment) or None = this slot is done."""
        nonlocal running
        with cond:
            while True:
                if queue:
                    game, ws = queue.pop(0)
                    running += 1
                    active.add(game)
                    report()
                    return game, ws, 0
                if standing is not None:
                    if deadline - time.monotonic() <= min_resume_seconds:
                        return None
                    if pool:
                        pool.sort(key=lambda e: (e[0], e[1]))
                        _, _, game, ws = pool.pop(0)
                        segments[game] += 1
                        running += 1
                        active.add(game)
                        report()
                        return game, ws, segments[game]
                    if running:
                        cond.wait(timeout=1.0)
                        continue
                return None

    def slot(index: int) -> None:
        nonlocal running
        carried = 0.0
        while True:
            picked = take(index)
            if picked is None:
                return
            game, ws, segment = picked
            if segment:
                game_deadline, info = deadline, {"slot": index, "resume": segment}
            elif base_seconds is None:
                game_deadline, info = deadline, {"slot": index}
            else:
                game_deadline = min(deadline, time.monotonic() + base_seconds + carried)
                info = {"slot": index, "base_seconds": base_seconds,
                        "carried_seconds": round(carried, 1)}
            acted_before = standing(game)[1] if segment else None
            failed = False
            try:
                results[game] = run_one(game, ws, game_deadline, info)
            except BaseException as error:  # noqa: BLE001 — the caller records it
                # Also log a line right away: the caller records only after every slot is done, so a game that dies
                # before supervisor.play would otherwise just show as queued the whole time. BaseException: a config
                # check's SystemExit inside a thread is silently swallowed by threading, the slot dies, running never
                # returns to zero, the other slots only finish at the deadline, and it ends in a KeyError here
                # (2026-10-27: the jail could not see the tokenizer in /dev/shm, and a smoke run idled for 10 minutes)
                log(f"{game}: run_one raised {error!r} (the slot moves on; listed again at the end)")
                results[game] = error
                failed = True
            if base_seconds is not None:
                carried = 0.0 if segment else max(0.0, game_deadline - time.monotonic())
            with cond:
                running -= 1
                active.discard(game)
                if standing is not None and not failed:
                    segments.setdefault(game, 0)
                    try:
                        enter_pool(game, ws, acted_before)
                    except Exception as error:  # noqa: BLE001 — progress unreadable: record it loudly
                                                # in the results; the slot must not die
                        results[game] = error
                report()
                cond.notify_all()

    with cond:
        report()
    threads = [threading.Thread(target=slot, args=(i,), name=f"slot-{i}", daemon=True)
               for i in range(concurrency)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return [results[game] for game, _ in work]


def play(game: str, cfg: dict, args, ws: Path, token: str, sockets: Path,
         deadline: float, ledger_path: Path, *, driver, knobs: arms.Knobs,
         paths: arms.Paths, pset: prompts.PromptSet, supervisor: ArmSupervisor,
         earlier: dict | None = None) -> dict:
    directory = getattr(args, "admission_dir", None)
    if directory is None:
        # the deadline is already this game's own (run_slots: the slot's base budget plus what the previous game left).
        return _play(game, cfg, args, ws, token, sockets, deadline, ledger_path,
                     driver=driver, knobs=knobs, paths=paths, pset=pset, supervisor=supervisor,
                     earlier=earlier)
    if earlier is not None:
        raise ValueError("tail resume does not exist in admission mode")
    admission = Admission(directory, args.admission_owner)
    agent = f"{args.replica_id}/{game}"
    supervisor.play({"game": game, "sessions": [], "status": "queued", "agent_id": agent})
    # A delayed gateway/queue must never admit a game whose full budget plus
    # bounded scope cleanup would exceed the independent campaign supervisor.
    admitted = admission.acquire(agent, required_seconds=cfg["wall_seconds"] + 60)
    deadline = admitted["admitted_monotonic"] + cfg["wall_seconds"]
    record, error = None, None
    try:
        record = _play(game, cfg, args, ws, token, sockets, deadline, ledger_path,
                       driver=driver, knobs=knobs, paths=paths, pset=pset, supervisor=supervisor)
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        with supervisor.lock:
            if record is None:
                previous = next((row for row in supervisor.state["plays"] if row["game"] == game), {})
                record = {"sessions": [], **previous, "game": game, "status": "failed", "error": error}
            sessions = record["sessions"]
            won = bool(sessions and sessions[-1].get("won"))
            censored = (any(row.get("returncode") == 124 or row.get("unit_result") == "timeout" for row in sessions)
                        or not won and time.monotonic() >= deadline)
            cleaned = all(row["status"] == "cleaned" for row in supervisor.scopes.values() if row["game"] == game)
        outcome = "failed" if error else "time_budget_exhausted" if censored else "won" if won else "retired_before_deadline"
        record["admission"] = admission.finish(agent, outcome=outcome, deadline_censored=censored,
                                                game_won=won, release=cleaned)
        record["agent_id"] = agent
        supervisor.play(record)
    return record


def _session_bwrap(supervisor, cfg, game, attempt, ws, log_path, env, driver_argv, left, over=None):
    """One session: jail.sh inside a systemd scope, output written to ``log_path`` (outside the workspace,
    driver_log_path). Returns (returncode, unit_result, unit, cleanup)."""
    # One systemd scope per game, with a memory cap. The old arm learned this the hard way: an unbounded BFS the
    # agent wrote itself reached 274 GB RSS. Harmless with eight games, but four arms of 25 games each running at once
    # would take the whole machine down, and the OOM killer does not only kill our processes -- this machine is
    # shared. A cgroup is more reliable than a hand-written watchdog: the kernel enforces it, and kills stay inside
    # the scope.
    cap = cfg.get("game_memory_max", "24G")
    # The name must be unique: while a unit of the same name exists, systemd-run fails outright, which the loop below
    # would read as "this session produced nothing" and stop opening sessions -- losing the game.
    receipt = supervisor.scope(game, attempt, runtime_max_seconds=left)
    unit = receipt["unit"]
    argv = ["systemd-run", "--user", "--scope", "--quiet",
            f"--unit={unit}", f"--description={receipt['description']}",
            f"--property=MemoryMax={cap}", "--property=MemorySwapMax=0",
            # native for scopes in systemd 244+; reclaimed on expiry even if the launcher is SIGKILLed.
            f"--property=RuntimeMaxSec={left:.6f}s", "--property=TimeoutStopSec=15s",
            "/bin/bash", str(UNIT / "jail.sh"), str(ws), *driver_argv]
    code, failure = None, None
    try:
        with log_path.open("w") as out:
            if over is None:
                proc = subprocess.run(argv, env=env, stdin=subprocess.DEVNULL,
                                      stdout=out, stderr=subprocess.STDOUT,
                                      timeout=left, check=False)
                code = proc.returncode
            else:
                code = wait_session(subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL,
                                                     stdout=out, stderr=subprocess.STDOUT), left, over)
    except subprocess.TimeoutExpired:
        code = 124                           # wall clock reached: a normal end
    except BaseException as error:
        failure = repr(error)
        raise
    finally:
        cleanup = supervisor.cleanup_scope(receipt, returncode=code, error=failure)
    # rc=-15 alone does not say who killed it. 6 h real run, cn04: the agent's own analysis script hit MemoryMax, the
    # OOM killer killed it, then systemd SIGTERMed the whole scope -- the journal says so plainly, yet run.log only
    # had a -15. Record systemd's Result too.
    # (Hence systemd-run no longer passes --collect: once collected, the Result can no longer be queried.)
    return code, cleanup["unit_result"], unit, cleanup["outcome"]


def _session_posix(supervisor, cfg, game, attempt, ws, log_path, env, driver_argv, left, began):
    """One session: fence_posix.sh, run as this game's dedicated uid, output written to ``log_path``. Same return
    shape as _session_bwrap.

    The three things a systemd scope does each have a stand-in here: memory cap = RLIMIT_DATA (per process) + the
    watchdog summing per uid (whole game); RuntimeMaxSec = ``wait(timeout)``; the whole cgroup going down with the
    scope = killing everything of this uid afterwards (orphans included). ``unit_result`` uses systemd's vocabulary,
    so downstream (oom.md, the give-up verdict) need not know which fence it was.
    """
    fence = supervisor.fence
    user = posix_fence.game_user(game)
    uid = posix_fence.uid_of(user)
    if uid is None:
        subprocess.run(["useradd", "-M", "-N", "-s", "/bin/bash", user], check=True,
                       capture_output=True)
        uid = posix_fence.uid_of(user)
    fence.watchdog.watch(uid, game)
    if fence.watchdog.error is not None or not fence.watchdog.is_alive():
        raise RuntimeError(f"posix watchdog is down: {fence.watchdog.error!r}")
    # fence-level settings (deny, caps, the middleware port this game sees) override the values bwrap uses.
    fenv = {**{k: v for k, v in env.items() if k.startswith("ARCSEC_")}, **fence.env,
            "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
            "ARCSEC_GAME_USER": user,
            "ARCSEC_FENCE_SCAN": "1" if attempt == 1 else "",
            "ARCSEC_WORKSPACE_TOOLS": "1" if cfg.get("workspace_tools") else ""}
    argv = ["/bin/bash", str(UNIT / "fence_posix.sh"), str(ws), *driver_argv]
    code = None
    with log_path.open("w") as out:
        proc = subprocess.Popen(argv, env=fenv, stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=left)
        except subprocess.TimeoutExpired:
            code = 124                       # wall clock reached: a normal end
        finally:
            killed = posix_fence.kill_user(uid)
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=30)
    if code != 124 and proc.returncode is not None:
        code = proc.returncode
    if fence.watchdog.oom_killed_since(game, time.time() - (time.monotonic() - began)):
        result = "oom-kill"
    elif code == 124:
        result = "timeout"
    else:
        result = "success" if code == 0 else "exit-code"
    return code, result, f"uid:{uid}", f"killed {killed} leftover"


def _play(game: str, cfg: dict, args, ws: Path, token: str, sockets: Path,
          deadline: float, ledger_path: Path, *, driver, knobs: arms.Knobs,
          paths: arms.Paths, pset: prompts.PromptSet, supervisor: ArmSupervisor,
          earlier: dict | None = None) -> dict:
    """Play this game until the wall clock runs out or it really ends.

    ``earlier`` (``tail_resume``): the record of this game's previous segment. Sessions reopen in the same workspace
    with the same guid, session numbers continue from the previous segment (no clash in log file names or scopes),
    and ``sessions`` is the sum over all segments; there is no thread to continue (the driver home was deleted by
    harvest, or resuming is off), so it is a fresh conversation + a ``resume.md`` handover. Resume segments are not
    limited by ``max_sessions`` (they play until the deadline, a give-up verdict or a win).

    **The driver exiting early does not mean the game is over.** Measured 2026-09-17: in one game the model emitted a
    23-token degenerate response in its first turn without calling any tool, and `codex exec` exited normally --
    that game scored zero for the whole run, which in the four-arm comparison is a random zero.

    So when the driver exits before the game ends, a new session is opened. **This is not a replay**: the workspace,
    guid and play number all stay the same, and the agent continues from its own logs.txt. A replay is "opening a new
    environment for a game already touched", which the gateway blocks (RESET without a guid returns 400); there is
    no way around it here.
    """
    driver_home = ws / ".driver"
    driver.prepare(driver_home, ws, knobs)
    reminder = compact_reminder_config(cfg, pset)
    if reminder is not None:
        # codex patch 0005 reads this file: the first line is the threshold, the rest is the message to send.
        (driver_home / "compact_reminder").write_text(
            f"{reminder}\n{pset.optional('compact_reminder.md')}")
    # the driver's own environment variables go into a KEY=VALUE file outside games/ (the jail mounts only ws).
    env_file = ws.parents[1] / "driver_env" / f"{game}.env"
    env_file.parent.mkdir(parents=True, exist_ok=True)
    # the driver log is outside games/ too (DRIVER_LOGS): an agent tidying its workspace cannot move or delete it
    driver_log_path(ws.parents[1], game, 1).parent.mkdir(parents=True, exist_ok=True)
    driver_env = driver.env(driver_home, ws, paths)
    library = ws.parents[1] / AGENT_LIB
    if cfg.get("act_library"):
        # posix: fence_posix.sh's PYTHONPATH is pyshadow (which masks the image's sitecustomize), and the driver env
        # applied after it replaces it wholesale -- without it every agent command gets a wrapt warning line again (on
        # 2026-09-17 an agent ran pip install over it).
        driver_env["PYTHONPATH"] = (str(library) if cfg["fence"] == "bwrap"
                                    else f"{args.pyshadow}:{library}")
    gen2 = cfg.get("harness") == "gen2"
    if cfg.get("note_gate") or gen2:
        # act's note_gate recognizes codex by this executable's inode when stopping processes (same path in the jail:
        # jail.sh mounts the driver bin read-only at its original path)
        exe = Path(args.driver_bin) / "bin" / "codex"
        if not exe.is_file():
            raise RuntimeError(f"note_gate: no codex executable at {exe}")
        driver_env["ARC_DRIVER_EXE"] = str(exe)
    for key, value in driver_env.items():
        if "\n" in key or "\n" in value or "=" in key:
            raise ValueError(f"{driver.NAME}.env(): {key!r} cannot be a one-line KEY=VALUE")
    env_file.write_text("".join(f"{k}={v}\n" for k, v in driver_env.items()))
    env = {
        "PATH": "/usr/bin:/bin",
        "ARCSEC_RUNTIME": str(Path(args.agent_python).parents[1]),
        "ARCSEC_ACT": str(ws / "act.py"),
        "ARCSEC_SOCKETS": str(sockets),
        "ARCSEC_BRIDGE": str(ROOT / "src/arcforge/arc_code/bridge.py"),
        "ARCSEC_DRIVER_BIN": str(args.driver_bin),
        "ARCSEC_DRIVER_HOME": str(driver_home),
        "ARCSEC_DRIVER_ENV": str(env_file),
        # driver_files (codex's instruction files) must be mounted read-only too: the config points at absolute paths
        # in the unit, and if the jail cannot see them codex exits 1 at start-up (the first PC end-to-end run lost 30
        # games this way).
        "ARCSEC_EXTRA_RO": ":".join(jail_ro_paths([*(str(m) for m in driver.mounts(paths)),
                                                   *(str(path) for _, path in knobs.driver_files),
                                                   *([str(library)] if cfg.get("act_library") else []),
                                                   *([str(GEN2_HOOKS)] if gen2 else [])],
                                                  cfg["fence"])),
        "ARCSEC_TOKEN": token,
        "ARCSEC_PROCESS_DATA_KB": str(size_kb(cfg["process_data_max"])),
        # posix's watchdog kills any single process over its CPU seconds; bwrap does the same with RLIMIT_CPU (jail.sh).
        **({"ARCSEC_CPU_SECONDS": str(cfg["watchdog_cpu_seconds"])}
           if cfg.get("watchdog_cpu_seconds") else {}),
        "ARCSEC_ENV_PORT": str(cfg["env_port"]),
        "ARCSEC_MODEL_PORT": str(cfg["model_port"]),
        "ARCSEC_JAIL_PROC": os.environ.get("ARCSEC_JAIL_PROC", "new"),
        **({"ARCSEC_RECORD": str(ws / "record")} if cfg.get("record") else {}),
        # posix has no read-only mounts: fence_posix.sh verifies as the game's user that the library is readable, not
        # writable
        **({"ARCSEC_AGENT_LIB": str(library)} if cfg.get("act_library") else {}),
        **({"ARCSEC_OBSERVATION_TOOLS": "1"} if cfg.get("observation_tools") else {}),
        # systemd-run --user needs these two to reach the user bus, otherwise it fails straight away with
        # `Failed to connect to bus: No medium found` and the game ends instantly.
        # Passing them is safe: jail.sh uses --clearenv, so inside the jail only explicitly setenv'd values are visible.
        "XDG_RUNTIME_DIR": os.environ.get(
            "XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"),
    }
    if "DBUS_SESSION_BUS_ADDRESS" in os.environ:
        env["DBUS_SESSION_BUS_ADDRESS"] = os.environ["DBUS_SESSION_BUS_ADDRESS"]
    if gen2:
        opened = open_game(f"http://127.0.0.1:{cfg['gateway_port']}", token, game)
        log(f"{game}: gen2 harness open: levels_completed {opened.get('levels_completed')} "
            f"state {opened.get('state')}")
    sessions = list(earlier["sessions"]) if earlier else []
    first_attempt = max((row["attempt"] for row in sessions), default=0)
    started = time.monotonic() - (earlier["seconds"] if earlier else 0.0)
    _, before = game_progress(ledger_path, game)
    idle = 0
    last_handle: str | None = None
    last_result: str | None = None
    brief = pset.brief(game, driver.brief_addendum(), limits=brief_limits(pset, cfg))
    # From v12 (kickoff.md) the first message is only an opening line: the brief is already in AGENTS.md, which the
    # driver injects itself and re-injects after compaction, so no second copy is needed in context.
    opening = pset.optional("kickoff.md") or brief
    # single-level episode (--level-episode): the opening gains a paragraph "starting from level K, it ends when you
    # clear it"; once a stop file appears (level cleared, action cap, output budget, failed start) no more sessions are
    # opened, and the running one is closed by over()
    episode = cfg.get("level_episode")
    stop_dir = Path(episode["stop_dir"]) if episode is not None else None
    if "game_episode" in cfg:                      # whole-game episode (RL arc3_game): same stop files,
                                                   # nothing added to the opening
        stop_dir = Path(cfg["game_episode"]["stop_dir"])
    # prior-levels seed (--level-seed): the actions the first K−1 levels took in the real full game. The ledger only
    # has this episode's, so the "actions used so far" stated by the handover messages (opening, resume, nudge) must
    # add them back -- the same number as the actions_used the middleware shows the agent
    seed_actions = episode["seed"]["actions"] if episode is not None and "seed" in episode else 0
    kickoff = opening
    if episode is not None and seed_actions:
        _, done_actions, done_levels = _progress(ledger_path, game)
        opening = kickoff + "\n" + pset.render("level_seed.md", level=episode["level"], levels=done_levels,
                                                actions=seed_actions + done_actions)
    elif episode is not None:
        opening = opening + pset.render("level.md", level=episode["level"])
    supervisor.play({"game": game, "sessions": list(sessions), "status": "running",
                     **({"seconds": earlier["seconds"]} if earlier else {})})
    model_errors = 0
    last_overflow: tuple | None = None   # (input, limit) when the previous zero-action session was
                                         # rejected by the server for exceeding the context
    recovery_limit = context_recovery_config(cfg, driver.NAME)
    recovery_count = 0
    pending_recovery_note = ""
    attempt = first_attempt
    budget = cfg.get("game_output_tokens")
    tokens = GameOutputTokens(ws) if budget else None
    over = (lambda: tokens.read() >= budget) if budget else None
    if stop_dir is not None:
        if budget:
            raise SystemExit("--level-episode's output budget is in the proxy (stop file budget); the config must not also set game_output_tokens")
        over = lambda: bool(episode_stops.stops(stop_dir))  # noqa: E731
    # gen-2 single-level episode (--episode-action-cap): a stop file (level cleared, action cap, start mismatch) or the
    # output budget, whichever comes first, closes the session
    g2_stops = Path(cfg["g2_episode_stop_dir"]) if cfg.get("g2_episode_stop_dir") else None
    if g2_stops is not None:
        token_over = over
        over = lambda: bool(episode_stops.stops(g2_stops)) or bool(token_over is not None and token_over())  # noqa: E731
    counted = 0                       # sessions counted toward max_sessions; those with model-side errors do not count
    # resume segments (earlier) are not limited by max_sessions: while time remains, keep opening sessions; the give-up
    # verdict (two short zero-action sessions in a row), the deadline and min_session_seconds still end it.
    while earlier or counted < cfg.get("max_sessions", 6):
        attempt += 1
        counted += 1
        if supervisor.cancelled.is_set():
            break
        left = deadline - time.monotonic()
        if left < cfg.get("min_session_seconds", 120):
            break
        if stop_dir is not None and episode_stops.stops(stop_dir):
            log(f"{game}: episode already stopped ({sorted(episode_stops.stops(stop_dir))}), not reopening")
            break
        if g2_stops is not None and episode_stops.stops(g2_stops):
            log(f"{game}: gen-2 single-level episode already stopped ({sorted(episode_stops.stops(g2_stops))}), not reopening")
            break
        if over is not None and over():
            log(f"{game}: output {tokens.total:,} tokens reached game_output_tokens {budget:,}, not reopening")
            break
        mode = session_mode(cfg["resume_mode"], attempt, last_handle)
        # The previous session was killed by the memory cap: the agent saw nothing (the whole scope died with it); the
        # launcher knows, so it must tell the agent. Only prompt sets with oom.md have this paragraph; v3 does not.
        aftermath = (pset.optional("oom.md", limit=str(cfg.get("game_memory_max", "24G")))
                     if was_oom(last_result) else "")
        recovering_context = bool(pending_recovery_note)
        if recovering_context:
            if mode != "fresh":
                raise RuntimeError("context recovery must start a fresh conversation")
            aftermath += pending_recovery_note
        if mode == "resume":
            # Continue the same conversation. The reasoning step cut off by the context is still in the history; the
            # driver looks at usage to decide whether to compact first or continue directly (see arms/<driver>.py).
            _, done_actions, done_levels = _progress(ledger_path, game)
            nudge = aftermath + pset.nudge(levels=done_levels, actions=seed_actions + done_actions)
            driver_argv = driver.resume_argv(last_handle, nudge, knobs)
            # the first 8 chars of a UUIDv7 are a millisecond timestamp, identical when 25 games start together; print
            # the last 8.
            log(f"{game}: session {attempt} resumes {driver.NAME} session …{last_handle[-8:]} (nudge"
                + (", with the OOM note" if aftermath else "") + ")")
        else:
            if attempt > 1:
                if recovering_context:
                    log(f"{game}: session {attempt} context recovery in the same workspace and game")
                elif attempt == first_attempt + 1 and earlier:
                    log(f"{game}: session {attempt} tail resume in the same workspace and game (new conversation + resume handover)")
                elif cfg["resume_mode"] == "thread":
                    log(f"{game}: session {attempt} has no session to resume ({driver.NAME} never came up), opening a new session")
                _, done_actions, done_levels = _progress(ledger_path, game)
                if seed_actions:
                    # a seeded game's opening is itself a resume handover: on a new conversation, rewrite it from the
                    # current state rather than stacking two
                    task = (kickoff + "\n" + aftermath
                            + pset.render("level_seed.md", level=episode["level"], levels=done_levels,
                                          actions=seed_actions + done_actions))
                else:
                    task = opening + "\n" + aftermath + pset.resume(levels=done_levels, actions=done_actions)
            else:
                task = opening
            driver_argv = driver.argv(task, knobs)
        left = deadline - time.monotonic()
        if left <= 0:
            break
        began = time.monotonic()
        log_path = driver_log_path(ws.parents[1], game, attempt)
        if supervisor.cancelled.is_set():
            break
        if supervisor.fence is None:
            code, result, unit, outcome = _session_bwrap(
                supervisor, cfg, game, attempt, ws, log_path, env, driver_argv, left,
                **({"over": over} if over is not None else {}))
        else:
            code, result, unit, outcome = _session_posix(
                supervisor, cfg, game, attempt, ws, log_path, env, driver_argv, left, began)
        last_result = result
        won, actions = game_progress(ledger_path, game)
        require_driver_log(log_path, attempt)
        # This session's conversation id: the next resume continues it. None = the driver never came up; then keep
        # the last one that was alive (e.g. after exit 71).
        handle = arms.session_handle(driver, log_path)
        if gen2 and handle:
            # codex's --json does not show hooks; if they did not run (trust, path, permissions) the whole run is
            # wasted, so stop right here
            hook_log = driver_home / "gen2_hooks.jsonl"
            hook_text = hook_log.read_text() if hook_log.is_file() else ""
            if '"SessionStart"' not in hook_text:
                raise RuntimeError(f"{game}: gen2 hooks did not run in session {attempt} (no {hook_log})")
            failed = [line for line in hook_text.splitlines() if '"hook_error"' in line]
            if failed:
                raise RuntimeError(f"{game}: a gen2 hook failed in session {attempt}: {failed[-1]}")
        if handle:
            last_handle = handle
            if recovering_context:
                prior = next(s for s in reversed(sessions) if "context_recovery" in s)
                prior["context_recovery"].update(status="driver_started", to_handle=handle,
                                                 resumed_attempt=attempt)
                atomic_json(supervisor.receipts.parent / "context_recovery" / game
                            / f"attempt-{prior['attempt']:04d}.json", prior["context_recovery"])
                pending_recovery_note = ""
        sessions.append({"attempt": attempt, "returncode": code,
                         "unit_result": result, "mode": mode, "handle": handle,
                         "scope": unit, "cleanup": outcome,
                         "seconds": round(time.monotonic() - began, 1),
                         "won": won, "actions_after": actions})
        supervisor.play({"game": game, "sessions": list(sessions), "status": "running",
                         "seconds": round(time.monotonic() - started, 1)})
        if result not in ("success", None):
            log(f"{game}: session {attempt} unit result = {result}"
                + (" (hit MemoryMax; the agent's processes were OOM-killed)" if "oom" in result else ""))
        if stop_dir is not None and episode_stops.stops(stop_dir):
            # a session closed by over() also returns TOKEN_BUDGET_CODE: check the stop files first, do not read it as
            # game_output_tokens
            sessions[-1]["episode_stops"] = sorted(episode_stops.stops(stop_dir))
            log(f"{game}: episode stopped ({sessions[-1]['episode_stops']}, rc={code}), wrapping up")
            break
        if g2_stops is not None and episode_stops.stops(g2_stops):
            sessions[-1]["episode_stops"] = sorted(episode_stops.stops(g2_stops))
            log(f"{game}: gen-2 single-level episode stopped ({sessions[-1]['episode_stops']}, rc={code}), wrapping up")
            break
        if code == 124:
            break
        if code == TOKEN_BUDGET_CODE:
            log(f"{game}: output {tokens.total:,} tokens reached game_output_tokens {budget:,}, wrapping up")
            break
        death = level_death(code, result, log_path, handle is not None) if stop_dir is not None else None
        if death is not None:
            # single-level episode (RL): codex died = failure, no resume (died in level_episode.py)
            episode_stops.write_stop(stop_dir, episode_stops.DIED,
                                     {**death, "attempt": attempt, "actions": actions, "unix": time.time()})
            sessions[-1]["episode_stops"] = [episode_stops.DIED]
            log(f"{game}: RL episode: codex died ({death['cause']}, rc={code}, {result}), episode over")
            break
        # A win ends the game. Measured 2026-09-17: after clearing all eight levels, ar25 was reopened three more
        # times, each answering "Already solved, nothing left to do" and burning wall clock, and one of them even
        # RESET the environment (the score was not polluted, but by luck, not by design).
        if won:
            log(f"{game}: WIN ({actions} actions), not reopening")
            break
        # This session produced no action, and the next one probably will not either. Reopening is for "the driver
        # dropped but the game is not over", not for "the agent does not want to play anymore".
        session_seconds = sessions[-1]["seconds"]
        # jail.sh's exit codes are the fence's own signals, not agent behavior:
        #   70 = corpus/engine visible inside the jail -> this game's score may come from reading the corpus; stop,
        #        loudly.
        #   71 = bridge/health check failed -> a harness problem; retry, **not counted as giving up**.
        # 6 h real run sk48: the health check broke because the agent's own struct.py shadowed a module -> 71 -> it was
        # retired as zero-action.
        if code == 70:
            log(f"{game}: FENCE BREACH (jail.sh exit 70) -- corpus or engine visible inside the jail; this game stops, "
                "and the whole run's attributability must be re-assessed")
            break
        if code == 71:
            # Operator log kept verbatim (04_main.py's LIVE filter matches "放棄"). Gist: "bridge/health check failed
            # (jail.sh exit 71) -- a harness problem, not counted as giving up, retrying in 10 s"
            log(f"{game}: session {attempt} 橋/健檢不通（jail.sh exit 71）——harness 的問題，"
                "不計入放棄，10 秒後重試")
            time.sleep(10)
            continue
        if code == 64:
            # jail.sh's input contract is broken (ARCSEC_* missing or malformed). jail.sh rereads the disk every
            # session, so a rebase onto a new contract mid-run does this -- the harness's fault, not the agent giving
            # up; be loud.
            # Operator log kept verbatim (04_main.py's LIVE filter matches "放棄"). Gist: "jail.sh exit 64 (broken
            # resource declaration, see <log>) -- a harness problem, not counted as giving up, retrying in 30 s"
            log(f"{game}: session {attempt} jail.sh exit 64（資源宣告壞了，看 {log_path}）——harness 的問題，"
                "不計入放棄，30 秒後重試")
            time.sleep(30)
            continue
        if was_oom(result):
            # Being killed by the memory cap is not giving up: re86 re-ran the same broken script five times and wanted
            # to play all along. The next session carries the oom.md explanation (if any); max_sessions bounds this.
            log(f"{game}: session {attempt} OOM-killed ({actions} actions), not counted as giving up; resuming with an explanation")
            before = actions
            continue
        if recovery_limit and code != 0:
            diagnosis = recovery.diagnosis(log_path)
            if diagnosis["fresh_recommended"]:
                sessions[-1]["context_error"] = diagnosis
                left = deadline - time.monotonic()
                if supervisor.cancelled.is_set() or left <= 0 or left < cfg.get("min_session_seconds", 120):
                    sessions[-1]["context_recovery_skipped"] = "cancelled or insufficient remaining time"
                    break
                if recovery_count >= recovery_limit:
                    sessions[-1]["context_recovery_exhausted"] = True
                    log(f"{game}: context recovery limit {recovery_limit} reached; refusing repeated fresh threads")
                    break
                checkpoint = recovery_checkpoint(ledger_path, game)
                actual_step, levels = checkpoint["step"], checkpoint["levels_completed"]
                pending_recovery_note = recovery.render_recovery_note(
                    diagnosis, game_id=checkpoint["game_id"], step=actual_step,
                    levels_completed=levels, win_levels=checkpoint["win_levels"],
                    recovery_number=recovery_count + 1,
                    remaining_seconds=max(0, deadline - time.monotonic()))
                recovery_count += 1
                receipt = {"recovery": recovery_count, "status": "scheduled", "from_handle": last_handle,
                           "action_step": actual_step, "levels_completed": levels,
                           "checkpoint": checkpoint,
                           "diagnosis": diagnosis, "note": pending_recovery_note,
                           "game_reset": False}
                sessions[-1]["context_recovery"] = receipt
                target = supervisor.receipts.parent / "context_recovery" / game
                atomic_json(target / f"attempt-{attempt:04d}.json", receipt)
                supervisor.play({"game": game, "sessions": list(sessions), "status": "recovering_context",
                                 "seconds": round(time.monotonic() - started, 1)})
                last_handle = None
                counted -= 1  # Separate bounded repair budget, not agent retirement.
                idle = 0
                before = actions
                log(f"{game}: proven context overflow; recovery {recovery_count}/{recovery_limit}, "
                    f"new conversation at the same game step {actual_step}")
                continue
        # The same history rejected by the server for exceeding the context in two sessions in a row (zero actions,
        # same input token count): resuming would only resend the same thing and hit it again. c0L4-r2 sk48: the proxy
        # missed one response's usage, the window let a 166k-token request through, codex retried 12 times per session
        # for about 14 minutes with zero actions, and kept being resumed by "zero actions but long = still analyzing"
        # below; codex does not compact on its own because of this error. With context_error_recovery on, a new thread
        # was already started above; this is the stop-loss when it is off.
        overflow = None
        if actions == before:
            report = recovery.diagnosis(log_path)
            if report["kind"] == "context_limit" and report["terminal"]:
                overflow = (report["input_tokens"], report["context_limit"])
        if overflow is not None and overflow == last_overflow:
            sessions[-1]["stuck_context_limit"] = {"input_tokens": overflow[0], "context_limit": overflow[1]}
            log(f"{game}: STUCK: two sessions in a row with zero actions, both rejected by the server for exceeding the context"
                f" (input {overflow[0]} > {overflow[1]}); resuming would only hit it again, stopping")
            break
        last_overflow = overflow
        gave_up = actions == before and session_seconds < cfg.get("idle_session_seconds", 600)
        failure = last_turn_failure(log_path) if gave_up else None
        if failure is not None:
            # The model server is in trouble (returns errors, retries exhausted), not the agent giving up: log the cause
            # loudly, wait for /health to recover, then resume the same thread; counted neither as giving up nor toward
            # max_sessions. The previous v10 submission wrapped up after about 5 hours of a 9-hour rerun with no log to
            # check; this path at least leaves the cause behind and does not retire the run's games one by one.
            model_errors += 1
            counted -= 1
            log(f"{game}: MODEL ERROR session {attempt} (#{model_errors}): {failure[:300]}")
            if model_errors > cfg.get("max_model_errors", 20):
                log(f"{game}: more than {cfg.get('max_model_errors', 20)} model-side errors, stopping")
                break
            wait_model_healthy(args.model_base_url, deadline, game)
            before = actions
            continue
        if gave_up:
            idle += 1
            # "Giving up" looks like a **short** zero-action session: tu93 wrote a report and exited 0 after 4 minutes.
            # Retiring permanently after one cost arm A 16.3 hours in the previous run, 2.7 game-slots across four arms.
            # A **long** zero-action session is not giving up: in the 6 h real run ka59 reached 6/7, and session 2 spent
            # 54 minutes purely analyzing the last level, ended normally, zero actions -- the old rule would have closed
            # a 6/7 game for good after the next such session, with 3.5 hours of budget left. In a single pass,
            # reopening costs no other game's time, only GPU; retiring saves tokens and loses a game.
            if idle >= 2:
                # Operator log kept verbatim (04_main.py's LIVE filter matches "放棄"). Gist: "<n> short zero-action
                # sessions in a row (<Ns): judged as giving up, no more sessions"
                log(f"{game}: 連續 {idle} 個短的零動作 session（<{cfg.get('idle_session_seconds', 600)}s），"
                    "判定放棄，停止續開")
                break
            log(f"{game}: session {attempt} {session_seconds:.0f}s produced no action, trying again")
        else:
            if actions == before:
                # Operator log kept verbatim (04_main.py's LIVE filter matches "放棄"). Gist: "zero actions but long --
                # treated as still analyzing, not counted as giving up"
                log(f"{game}: session {attempt} {session_seconds:.0f}s 零動作但很長——當成還在分析，不計入放棄")
            idle = 0
        log(f"{game}: session {attempt} ended (rc={code}, {actions} actions), "
            f"{deadline - time.monotonic():.0f}s left")
        before = actions
    record = {"game": game, "sessions": sessions, "status": "finished",
              "seconds": round(time.monotonic() - started, 1)}
    segment = len(earlier["segment_digests"]) if earlier else 0
    digest = harvest(driver, driver_home, ws, keep=cfg["keep_driver_home"], segment=segment,
                     stash=ws / ROLLOUT_STASH if "game_episode" in cfg else None)
    record["segment_digests"] = [*(earlier["segment_digests"] if earlier else []), digest]
    rows: dict[str, int] = {}
    for d in record["segment_digests"]:
        for kind, n in d["rows"].items():
            rows[kind] = rows.get(kind, 0) + n
    record["digest"] = {"rows": rows,
                        "driver_home_bytes": sum(d["driver_home_bytes"] for d in record["segment_digests"]),
                        "driver_home_kept": digest["driver_home_kept"],
                        "files": [d["file"] for d in record["segment_digests"] if d["file"]]}
    supervisor.play(record)
    return record


#: rollout stash of a whole-game episode (copied before harvest deletes the driver home; main decides its fate after
#: repairing windows)
ROLLOUT_STASH = "rollout-stash"


def harvest(driver, driver_home: Path, ws: Path, *, keep: bool, segment: int = 0, stash: Path | None = None) -> dict:
    """At game end: copy what the driver's records have and ``--json`` lacks into ``driver_digest.jsonl``, then
    (``keep_driver_home: false``) delete the driver home.

    ``stash`` (whole-game episode): copy the rollout files here before deleting. main decides their fate after
    ``repair_windows_from_proxy``: kept if windows were repaired (the only physical evidence of codex failing to write
    ``compacted``), deleted otherwise.

    A codex game's CODEX_HOME is about 30 MB (the rollout carries every image viewed); 28 games × 4 waves is more than
    Kaggle output can hold -- v20 on 2026-09-19 produced 1.5 GB of output and got a system error at submission. But
    compaction summaries and tool calls exist **only there**: after the 2026-09-24 rehearsal deleted it, not one of
    98 summaries could be read. So copy first, then delete.
    """
    digest = getattr(driver, "digest", None)
    counts: dict[str, int] = {}
    # resume segment n (tail_resume) writes driver_digest.<n>.jsonl, without overwriting earlier ones.
    file = "driver_digest.jsonl" if segment == 0 else f"driver_digest.{segment}.jsonl"
    if callable(digest):
        with (ws / file).open("w") as out:
            for row in digest(driver_home):
                counts[row["kind"]] = counts.get(row["kind"], 0) + 1
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif not keep:
        raise RuntimeError(f"{driver.NAME} has no digest(); deleting the driver home would leave nothing; "
                           "set keep_driver_home: true")
    size = sum(f.stat().st_size for f in driver_home.rglob("*") if f.is_file() and not f.is_symlink()) \
        if driver_home.exists() else 0
    if stash is not None and not keep:
        rollouts = sorted((driver_home / "sessions").rglob("rollout-*.jsonl")) if (driver_home / "sessions").is_dir() else []
        stash.mkdir(exist_ok=False)
        for rollout in rollouts:
            shutil.copy2(rollout, stash / rollout.name)
    if not keep and driver_home.exists():
        shutil.rmtree(driver_home)
    return {"rows": counts, "driver_home_bytes": size, "driver_home_kept": keep,
            "file": file if callable(digest) else None}


def tito_items(results: list[dict], workspaces: dict[str, Path], ledger_path: Path) -> list[dict]:
    """One ``tito_manifest.json`` item per game (called only when the config has ``tito``): thread = the sessions'
    handles, response ids read from the game's driver_digest, outcome read from the ledger (the only authoritative
    record)."""
    items = []
    for play in results:
        game = play["game"]
        if "digest" in play:
            digests = [workspaces[game] / name for name in play["digest"]["files"]]
        elif play["status"] == "failed":
            digests = []                 # the session loop raised before reaching harvest: this game has no digest
        else:
            raise RuntimeError(f"{game}: play has no digest and is not failed (status={play['status']!r})")
        won, actions, levels = _progress(ledger_path, game)
        items.append({"id": game, "threads": [s["handle"] for s in play["sessions"] if s["handle"] is not None],
                      "digests": digests,
                      "outcome": {"status": play["status"], "levels_completed": levels, "won": won,
                                  "actions": actions, **({"error": play["error"]} if "error" in play else {})}})
    return items


def brief_limits(pset: prompts.PromptSet, cfg: dict) -> dict[str, str] | None:
    """The memory caps in the brief = the caps this run actually enforces. v3/v4 hard-code b200's 16 GB / 24 GB;
    running them under a fence with different caps lies to the agent, so refuse outright."""
    limits: dict[str, str] | None = None
    if pset.templates_limits():
        limits = {"process_limit": human_size(cfg["process_data_max"]),
                  "workspace_limit": human_size(cfg["game_memory_max"])}
    elif cfg["fence"] != "bwrap" or size_kb(cfg["process_data_max"]) != size_kb("16G") \
            or size_kb(cfg["game_memory_max"]) != size_kb("24G"):
        raise SystemExit(f"prompt set {pset.name} hard-codes the caps as 16 GB / 24 GB, "
                         f"this run has {cfg['process_data_max']} / {cfg['game_memory_max']} ({cfg['fence']}); "
                         "use a set from v5 on")
    if pset.templates_cpu_limit():
        # v12: the posix watchdog's per-process CPU-seconds cap (watchdog_cpu_seconds).
        # If the brief mentions it, the cap must exist -- telling the agent it will be killed when it won't be lies to it.
        # posix: the watchdog kills; bwrap: jail.sh's RLIMIT_CPU (ARCSEC_CPU_SECONDS) kills. Both per single process.
        seconds = cfg.get("watchdog_cpu_seconds")
        if cfg["fence"] not in ("posix", "bwrap") or not isinstance(seconds, int) or seconds <= 0:
            raise SystemExit(f"prompt set {pset.name} mentions a per-process CPU cap, "
                             f"but this run has none (fence {cfg['fence']}, watchdog_cpu_seconds {seconds!r})")
        limits = {**(limits or {}), "cpu_limit": human_duration(seconds)}
    return limits


def human_duration(seconds: int) -> str:
    """``900`` -> ``"15 minutes"``: for the agent to read."""
    if seconds % 60:
        return f"{seconds} seconds"
    minutes = seconds // 60
    return f"{minutes} minute{'s' if minutes != 1 else ''}"


def human_size(text: str) -> str:
    """``"8G"`` -> ``"8 GB"``: for the agent to read."""
    kb = size_kb(text)
    return f"{kb / 2**20:g} GB" if kb >= 2**20 else f"{kb / 1024:g} MB"


def assert_unambiguous_corpus(corpus: Path) -> None:
    """Stop when a game has two versions -- **that signals a broken corpus, not a situation where one can be picked**.

    `short_index()`'s rule is "lexicographically smallest", which is **arbitrary**. In the 2026-09-18 four-arm
    comparison `ar25` had two versions (`0c556536` 2026-06-05, `e3c63847` 2026-04-07), and lexicographic order
    happened to pick the newer one -- `0 < e`, **pure luck**. With the hashes reversed it would silently have played
    the old version, and the score could not be compared with any result on the new one, with nothing to flag it.

    The canonical corpus (`runs/arc_agi3_harness/20260831-qwen38/shared/environment_files`) has **zero multi-version
    games** among its 25. So ambiguity only shows up in stale or merged corpora, which is exactly what must be stopped
    on the spot.
    """
    ambiguous = {}
    for d in sorted(corpus.iterdir()):
        if not d.is_dir():
            continue
        versions = sorted(v.name for v in d.iterdir() if v.is_dir())
        if len(versions) > 1:
            dates = {}
            for v in versions:
                meta = json.loads((d / v / "metadata.json").read_text())
                dates[v] = meta["date_downloaded"][:10]
            ambiguous[d.name] = dates
    if ambiguous:
        detail = "; ".join(f"{g}: {v}" for g, v in ambiguous.items())
        raise RuntimeError(
            f"the corpus has {len(ambiguous)} games with multiple versions, refusing to start: {detail}. "
            "Which version is picked decides the score, and the default rule (lexicographic) is arbitrary. "
            "Point at a single-version corpus, or explicitly delete the unwanted version directories.")


def corpus_provenance(corpus: Path, roster: list[str]) -> dict:
    """Which corpus, and which versions in it, this run actually played.

    The 2026-09-18 four-arm comparison used 78.60 as its baseline, and only after the run did it turn out that the
    baseline came from a different corpus: run5/run9/run10/run11 all used ``runs/arc_agi3_harness/20260831-qwen38/shared/
    environment_files`` (all downloaded 2026-08-31), while I used
    ``reference/polyphony-arc3-prep/vendor/arc/environment_files`` (**deleted on
    2026-09-18**) -- an April
    third-party vendored snapshot. **17 of the 25 games had different baseline_actions**, including games with
    identical code; RHAE is squared, so a 35% lower baseline costs 58% of the score for the same play.
    Most of that 18-point "regression" was this.

    So every run writes the corpus identity into ARM_STATE: the path, per-game versions and download dates,
    and a fingerprint of the whole selection. **Before comparing across runs, compare ``fingerprint``; if it differs,
    do not compare.**
    """
    import hashlib

    picked, dates = {}, []
    for full in roster:
        meta = json.loads((corpus / full.split("-")[0] / full.split("-", 1)[1]
                           / "metadata.json").read_text())
        picked[full] = {"date_downloaded": meta["date_downloaded"],
                        "levels": len(meta["baseline_actions"]),
                        "baseline_total": sum(meta["baseline_actions"])}
        dates.append(meta["date_downloaded"])
    blob = json.dumps({g: picked[g]["baseline_total"] for g in sorted(picked)},
                      sort_keys=True)
    return {"path": str(corpus.resolve()),
            "fingerprint": hashlib.sha256(blob.encode()).hexdigest()[:16],
            "date_downloaded_range": [min(dates), max(dates)] if dates else [],
            "games": picked}


def summarise(ledger: Ledger, games: list[str], corpus: Path) -> dict:
    """Scores come only from the ledger -- the agent's logs.txt is not a ledger.

    The formula was checked against the official EnvironmentScoreCalculator, matching on all 4000 random cases;
    rerun with probes/diff_score_official.py.

    ``games`` arrive as **short names** (workspace directory names are short names) while ledger keys are full names,
    so they must be resolved here. Before 2026-09-18 they were not: all 25 games were recorded as never played, all
    four arms' summary.json said `mean_score: 0.0`, with no error message at all -- FINAL.json is computed by another
    path, so this 0 survived until wrap-up before anyone noticed.
    Now score_run raises when it does not match the corpus, so this path can no longer silently return 0.
    """
    base = baselines(corpus)
    return score_run(ledger.rows(), base, full_ids(games, base))


def full_ids(games: list[str], base: dict[str, list[int]]) -> list[str]:
    """Game names are short names in local mode (corpus directory names) and full names in upstream mode; map them to
    the corpus's full names, and raise on anything that does not map."""
    index = short_index(base)
    out, missing = [], []
    for g in games:
        if g in base:
            out.append(g)
        elif g in index:
            out.append(index[g])
        else:
            missing.append(g)
    if missing:
        raise RuntimeError(f"roster has games not in the corpus: {missing}")
    return out


def start_posix_fence(supervisor: ArmSupervisor, cfg: dict, args, run_dir: Path,
                      driver_files: dict[str, Path]) -> posix_fence.Watchdog:
    """One-time setup of the posix fence: directory permissions, the watchdog, and the ARCSEC_* shared by every session.

    The run directory shares a file system with the agent, so permissions are the fence: the run directory and games/
    are traverse-only (711, other games cannot be listed), gateway/ (tokens, competition key, logs) and driver_logs/
    (the harness's driver logs) 700, the ledger 600 (other games' progress).
    Every ancestor must be traversable by others, or every game dies with exit 71 in the fence's health check -- say
    so here first.
    """
    for ancestor in [run_dir.resolve(), *run_dir.resolve().parents]:
        if not ancestor.stat().st_mode & 0o001:
            raise SystemExit(f"{ancestor} is not traversable by other users (mode {oct(ancestor.stat().st_mode & 0o777)}); "
                             "under fence=posix the agent cannot enter its own workspace")
    run_dir.chmod(0o711)
    (run_dir / "games").mkdir(exist_ok=True)
    (run_dir / "games").chmod(0o711)
    for private in ("gateway", "driver_env", DRIVER_LOGS):
        (run_dir / private).mkdir(exist_ok=True)
        (run_dir / private).chmod(0o700)
    ledger = run_dir / "ledger.jsonl"
    if ledger.exists():
        ledger.chmod(0o600)
    watchdog = posix_fence.Watchdog(
        run_dir / "watchdog.jsonl",
        game_max_kb=size_kb(cfg["game_memory_max"]),
        headroom_kb=size_kb(cfg["watchdog_headroom"]) if cfg.get("watchdog_headroom") else None,
        cpu_seconds=cfg.get("watchdog_cpu_seconds"),
        interval=cfg.get("watchdog_interval_seconds", 2.0))
    watchdog.tick()               # first tick right here: an unreadable /proc must blow up before games start
    watchdog.start()
    supervisor.fence = PosixFence({
        "ARCSEC_ENV_PORT": str(cfg["gateway_port"]),
        "ARCSEC_FENCE_DENY": args.fence_deny,
        "ARCSEC_FENCE_KNOWN_READABLE": args.fence_known_readable,
        "ARCSEC_PYSHADOW": str(args.pyshadow.resolve()),
        "ARCSEC_OOM_SCORE_ADJ": str(cfg.get("oom_score_adj", 1000)),
        # gen-2: codex runs the hook scripts as the game user; if unreadable, every hook silently fails -- verify as the
        # player before game start
        "ARCSEC_FENCE_REQUIRE": ":".join([*(str(p) for p in driver_files.values()),
                                          *(str(p) for p in (sorted(GEN2_HOOKS.glob("*.py"))
                                                             if cfg.get("harness") == "gen2" else []))]),
    }, watchdog)
    supervisor.update(fence_posix={"deny": args.fence_deny.split(":"),
                                   "known_readable": [p for p in args.fence_known_readable.split(":") if p],
                                   "pyshadow": str(args.pyshadow)})
    log(f"posix fence: deny {args.fence_deny}; known readable {args.fence_known_readable or '-'}; "
        f"game max {cfg['game_memory_max']}, process {cfg['process_data_max']}, "
        f"headroom {cfg.get('watchdog_headroom')}")
    return watchdog


#: instantaneous values (gauges) of sglang ``/metrics`` (``--enable-metrics``): written into heartbeat.jsonl's
# ``sglang``; the screen only prints running / queued / retracted / mamba. gen_tok_s is the last decode-log interval
# (20 forwards, about 1.5 s) and cache_hit the last prefill batch -- neither is an average, so the interval numbers on
# screen come from differencing the cumulative counters below instead.
# kv_usage is the full-attention KV pool. On hybrid models sglang:token_usage is max(KV, mamba), and the mamba pool (3
# slots per running request, 120 in total) is always fuller -- that number only reflects concurrency. Under QSA
# HiSparse kv_usage is the reservation (0.99 by minute 2 of a run); actual occupancy is the #cpu token of server.log's
# decode lines (ServerLogTail).
METRICS = {"sglang:gen_throughput": "gen_tok_s", "sglang:full_token_usage": "kv_usage",
           "sglang:num_running_reqs": "running", "sglang:num_queue_reqs": "queued",
           "sglang:cache_hit_rate": "cache_hit", "sglang:mamba_usage": "mamba_usage",
           "sglang:num_retracted_reqs": "retracted"}
#: cumulative counters: the difference between two heartbeats is the interval amount. Missing means missing (recorded
# in missing, not filled with 0) -- the only exception is num_aborted_requests_total: in prometheus multiprocess mode a
# metric never written has no TYPE line at all, so it is absent until the first abort.
COUNTERS = {"decode": ("sglang:realtime_tokens_total", {"mode": "decode"}),
            "prefill_compute": ("sglang:realtime_tokens_total", {"mode": "prefill_compute"}),
            "prefill_cache": ("sglang:realtime_tokens_total", {"mode": "prefill_cache"}),
            "generation": ("sglang:generation_tokens_total", {}),
            "spec_verify": ("sglang:spec_verify_calls_total", {}),
            "requests": ("sglang:num_requests_total", {}),
            "aborted": ("sglang:num_aborted_requests_total", {}),
            "idle_seconds": ("sglang:scheduler_idle_seconds_total", {})}
LAZY_COUNTERS = {"aborted"}
PROM_SAMPLE = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})? (\S+)")
PROM_LABEL = re.compile(r'(\w+)="((?:[^"\\]|\\.)*)"')


def parse_prometheus(text: str) -> dict[str, list[tuple[dict[str, str], float]]]:
    """The text exposition of ``/metrics``: name -> [(labels, value)]; comment lines skipped, raises on a line it
    cannot parse."""
    out: dict[str, list[tuple[dict[str, str], float]]] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = PROM_SAMPLE.match(line)
        if m is None:
            raise ValueError(f"unparsable /metrics line: {line[:120]!r}")
        out.setdefault(m[1], []).append((dict(PROM_LABEL.findall(m[2] or "")), float(m[3])))
    return out


def fetch_metrics(model_base_url: str) -> str:
    root = model_base_url.split("/v1")[0]
    with urllib.request.urlopen(root + "/metrics", timeout=10) as r:
        return r.read().decode(errors="replace")


def gauges(series: dict) -> dict[str, float]:
    """The METRICS gauges that are present, each summed over all label combinations."""
    return {key: sum(v for _, v in series[name]) for name, key in METRICS.items() if name in series}


def sglang_metrics(model_base_url: str) -> dict:
    return gauges(parse_prometheus(fetch_metrics(model_base_url)))


class MetricsSnapshot:
    """One ``/metrics`` read: monotonic time plus all series. A counter's interval amount = the difference of two
    snapshots. (Not a dataclass: a test loads this file with spec_from_file_location without registering it in
    sys.modules, where a dataclass cannot be built.)"""

    def __init__(self, t: float, series: dict):
        self.t, self.series = t, series

    def counter(self, key: str) -> float:
        """Value of COUNTERS[key] (sum of the series whose labels contain the given values). KeyError if the name is not
        in /metrics, except for LAZY_COUNTERS."""
        name, match = COUNTERS[key]
        if name not in self.series:
            if key in LAZY_COUNTERS:
                return 0.0
            raise KeyError(name)
        return sum(v for labels, v in self.series[name] if all(labels.get(k) == x for k, x in match.items()))

    def buckets(self, name: str) -> dict[float, float]:
        """histogram ``<name>_bucket``: le -> cumulative count (summed over all label combinations)."""
        if name + "_bucket" not in self.series:
            raise KeyError(name + "_bucket")
        out: dict[float, float] = {}
        for labels, v in self.series[name + "_bucket"]:
            le = float(labels["le"])
            out[le] = out.get(le, 0.0) + v
        return out


def histogram_quantile(cumulative: dict[float, float], q: float) -> float | None:
    """Prometheus-style quantile: cumulative is le -> cumulative count (including +Inf), linear interpolation within a
    bucket; inf if it falls in the +Inf bucket, None if the interval has no samples."""
    les = sorted(cumulative)
    if not les or not math.isinf(les[-1]):
        raise ValueError(f"histogram without a +Inf bucket: {les[-3:]}")
    total = cumulative[les[-1]]
    if total <= 0:
        return None
    target, lo, below = q * total, 0.0, 0.0
    for le in les:
        count = cumulative[le]
        if count >= target:
            if math.isinf(le):
                return math.inf
            return lo + (le - lo) * ((target - below) / (count - below) if count > below else 1.0)
        lo, below = le, count
    raise AssertionError("unreachable: the +Inf bucket holds the total")


def counter_values(snap: MetricsSnapshot) -> dict[str, float | None]:
    """For heartbeat.jsonl: the raw cumulative value of every COUNTERS entry (None when missing), so any interval can be
    recomputed later."""
    out: dict[str, float | None] = {}
    for key in COUNTERS:
        try:
            out[key] = snap.counter(key)
        except KeyError:
            out[key] = None
    return out


def server_window(prev: MetricsSnapshot, cur: MetricsSnapshot, start: MetricsSnapshot) -> dict:
    """Interval amounts prev -> cur, plus the decode average start -> cur. Missing counters go into ``missing``, shrunk
    ones (server restart) into ``reset``; that field is None, not filled with 0."""
    dt = cur.t - prev.t
    out: dict = {"seconds": round(dt, 1), "missing": [], "reset": []}

    def delta(key: str, a: MetricsSnapshot) -> float | None:
        try:
            d = cur.counter(key) - a.counter(key)
        except KeyError as error:
            if error.args[0] not in out["missing"]:
                out["missing"].append(error.args[0])
            return None
        if d < 0:
            out["reset"].append(key)
            return None
        return d

    d = {key: delta(key, prev) for key in COUNTERS}
    decode_all = delta("decode", start)
    out["decode_tok_s"] = None if d["decode"] is None else d["decode"] / dt
    out["decode_tok_s_since_start"] = (None if decode_all is None or cur.t <= start.t
                                       else decode_all / (cur.t - start.t))
    out["prefill_tok_s"] = None if d["prefill_compute"] is None else d["prefill_compute"] / dt
    seen = None if d["prefill_compute"] is None or d["prefill_cache"] is None \
        else d["prefill_compute"] + d["prefill_cache"]
    out["prefill_hit"] = d["prefill_cache"] / seen if seen else None
    # both only grow when a request finishes: this is the token-weighted accept (bonus included) of requests finished
    # in the interval, lagging decode a little
    out["accept"] = (d["generation"] / d["spec_verify"]
                     if d["generation"] is not None and d["spec_verify"] else None)
    out["finished"] = d["requests"]
    out["aborted"] = d["aborted"]
    out["idle_frac"] = None if d["idle_seconds"] is None else d["idle_seconds"] / dt
    for name, key, qs in (("sglang:e2e_request_latency_seconds", "req", (0.5, 0.9)),
                          ("sglang:time_to_first_token_seconds", "ttft", (0.5,))):
        try:
            a, b = prev.buckets(name), cur.buckets(name)
        except KeyError as error:
            out["missing"].append(error.args[0])
            continue
        window = {le: b[le] - a.get(le, 0.0) for le in b}
        for q in qs:
            out[f"{key}_p{round(q * 100)}"] = histogram_quantile(window, q)
    return out


#: lines of server.log (sglang-arc-2) that the heartbeat counts. A format change would count 0 -- so heartbeat_lines
# reconciles against the finished-request count from /metrics (says so explicitly when QSA is on and ≥ 50 requests
# finished in the interval without a single decision).
QSA_SESSION_ON = "QSA HiSparse session cache: snapshot"
QSA_DECISION = re.compile(r"QSA session \S+: (?:(resume)|mismatch \((\w+)\), \w+) at P=")
THINK_ON = re.compile(r"ARC think budget: (\d+) tokens \(")
THINK_ARMED = re.compile(r"ARC think budget: \S+ \(session \S+\) armed at \d+ output tokens")
THINK_CLOSED = re.compile(r"ARC think budget: \S+ \(session \S+\) closed at output token")
THINK_TOOL = "ended its thinking with its own <tool_call>"
DECODE_HOST = re.compile(r"Decode batch, .*#cpu token: (\d+), cpu token usage: ([\d.]+)")


class ServerLogTail:
    """Read server.log onward from where the last read stopped (an unfinished half line waits for the next read) and
    count, over that stretch, QSA session decisions, thinking-cap events, actual host KV occupancy (#cpu token of the
    decode lines) and Tracebacks. The first read starts at the head of the file, so start-up is seen too: whether the
    QSA session cache and ARC_THINK_BUDGET are on is stated by the log itself."""

    def __init__(self, path: Path):
        self.path, self.offset, self.rest = path, 0, b""
        self.qsa_on = False
        self.think_budget: int | None = None
        self.totals = {"resume": 0, "mismatch": 0, "armed": 0, "closed": 0, "tool_ended": 0, "tracebacks": 0}

    def read(self) -> dict:
        size = self.path.stat().st_size
        if size < self.offset:
            raise RuntimeError(f"{self.path} shrank from {self.offset} to {size} bytes")
        with self.path.open("rb") as f:
            f.seek(self.offset)
            data = f.read()
        self.offset += len(data)
        lines = (self.rest + data).split(b"\n")
        self.rest = lines.pop()
        w = {"resume": 0, "mismatch": 0, "armed": 0, "closed": 0, "tool_ended": 0, "tracebacks": 0}
        reasons: dict[str, int] = {}
        decode_lines, host, peak = 0, None, 0
        for raw in lines:
            line = raw.decode("utf-8", errors="replace")
            if "QSA session " in line:
                m = QSA_DECISION.search(line)
                if m:
                    if m[1]:
                        w["resume"] += 1
                    else:
                        w["mismatch"] += 1
                        reasons[m[2]] = reasons.get(m[2], 0) + 1
            elif "Decode batch, " in line:
                m = DECODE_HOST.search(line)
                if m:
                    decode_lines += 1
                    host = (int(m[1]), float(m[2]))
                    peak = max(peak, host[0])
            elif "ARC think budget: " in line:
                if THINK_ARMED.search(line):
                    w["armed"] += 1
                elif THINK_CLOSED.search(line):
                    w["closed"] += 1
                elif THINK_TOOL in line:
                    w["tool_ended"] += 1
                elif m := THINK_ON.search(line):
                    self.think_budget = int(m[1])
            elif QSA_SESSION_ON in line:
                self.qsa_on = True
            if "Traceback (most recent call last)" in line:
                w["tracebacks"] += 1
        for k, v in w.items():
            self.totals[k] += v
        return {**w, "mismatch_reasons": reasons, "decode_lines": decode_lines,
                "host_tokens": None if host is None else host[0], "host_usage": None if host is None else host[1],
                "host_peak": peak if decode_lines else None, "qsa_on": self.qsa_on,
                "think_budget": self.think_budget, "totals": dict(self.totals)}


def _tok_s(x: float | None) -> str:
    return "-" if x is None else f"{x:,.0f}"


def _secs(x: float | None) -> str:
    if x is None:
        return "-"
    if math.isinf(x):
        return ">top bucket"
    return f"{x:.0f}s" if x >= 10 else f"{x:.1f}s"


def heartbeat_lines(row: dict) -> list[str]:
    """The lines one heartbeat prints, all starting with HEARTBEAT (the notebook's LIVE and kaggle_log_watch both rely on
    that word): game progress / server interval amounts (/metrics differences) / session (server.log). The format of
    the first line's start is read by work/qsa_hisparse/probes/kaggle_run_report.py."""
    g, pl = row["games"], row["play"]
    if "error" in pl:
        play = f"progress {pl['error']}"
    else:
        play = (f"actions {pl['actions']}" + (f" (+{pl['d_actions']})" if "d_actions" in pl else "")
                + f" levels {pl['levels']}" + (f" (+{pl['d_levels']})" if "d_levels" in pl else "")
                + f" wins {pl['wins']}"
                + (f" rhae {pl['rhae_started']:.2f}" if "rhae_started" in pl else "")
                + (f" record_errors {pl['record_errors']}" if "record_errors" in pl else ""))
    lines = ["HEARTBEAT " + f"games run {g['running']} done {g['finished']} queued {g['queued']} fail {g['failed']} | "
             + play + (f" | headroom {row['memory']['min_headroom_gib']} GiB kills {row['memory']['kills']}"
                       if "memory" in row else "")]
    if "reviews" in row:
        rv = row["reviews"]
        lines.append(f"HEARTBEAT reviews | running {rv['running']} launched {rv['launched']} delivered {rv['delivered']} "
                     f"failed {rv['failed']} | players running {rv['players']['running']} pooled {rv['players']['pooled']} "
                     f"queued {rv['players']['queued']} | cap {rv['max_total']}"
                     + (f" | cards running {rv['cards']['running']} launched {rv['cards']['launched']} "
                        f"published {rv['cards']['delivered']} failed {rv['cards']['failed']}"
                        + (f" CARDS FAILED {rv['cards']['error']}" if rv["cards"]["error"] else "") if "cards" in rv else "")
                     + (f" | SCHEDULER FAILED {rv['error']}" if rv["error"] else ""))
    sv, s = row.get("server", {}), row.get("sglang", {})
    if "error" in sv:
        lines.append(f"HEARTBEAT server | error {sv['error']}")
    elif sv.get("baseline"):
        lines.append("HEARTBEAT server | /metrics baseline taken (interval numbers from the next heartbeat)")
    else:
        hit = "-" if sv["prefill_hit"] is None else f"{100 * sv['prefill_hit']:.1f}%"
        idle = "-" if sv["idle_frac"] is None else f"{100 * sv['idle_frac']:.1f}%"
        parts = [f"server {sv['seconds'] / 60:.1f}m",
                 f"decode {_tok_s(sv['decode_tok_s'])} tok/s (since start {_tok_s(sv['decode_tok_s_since_start'])})",
                 f"prefill {_tok_s(sv['prefill_tok_s'])} tok/s hit {hit}",
                 "accept " + ("-" if sv["accept"] is None else f"{sv['accept']:.2f}"),
                 f"req p50 {_secs(sv.get('req_p50'))} p90 {_secs(sv.get('req_p90'))} ttft p50 {_secs(sv.get('ttft_p50'))}",
                 f"finished {'-' if sv['finished'] is None else round(sv['finished'])} "
                 f"aborted {'-' if sv['aborted'] is None else round(sv['aborted'])}",
                 " ".join(f"{k} {s[k]:g}" for k in ("running", "queued", "retracted") if k in s),
                 f"idle {idle}"]
        if "mamba_usage" in s:
            parts.append(f"mamba {s['mamba_usage']:.2f}")
        if sv["missing"]:
            parts.append("missing " + ", ".join(sv["missing"]))
        if sv["reset"]:
            parts.append("counter reset " + ", ".join(sv["reset"]))
        lines.append("HEARTBEAT " + " | ".join(parts))
    se = row.get("session")
    if se is None:
        return lines
    if "error" in se:
        lines.append(f"HEARTBEAT session | error {se['error']}")
        return lines
    t = se["totals"]
    parts = ["session"]
    if se["qsa_on"]:
        qsa = f"QSA resume {se['resume']} mismatch {se['mismatch']}"
        if se["mismatch_reasons"]:
            qsa += " (" + ", ".join(f"{k} {v}" for k, v in sorted(se["mismatch_reasons"].items())) + ")"
        finished = sv.get("finished") if isinstance(sv, dict) else None
        if finished is not None and finished >= 50 and se["resume"] + se["mismatch"] == 0:
            qsa += f" -- no QSA decision for {round(finished)} finished requests: has the log line changed?"
        parts.append(qsa + f" (total {t['resume']}/{t['resume'] + t['mismatch']})")
    else:
        parts.append("QSA session cache off")
    if se["think_budget"] is not None:
        parts.append(f"think {se['think_budget']}: armed +{se['armed']} closed +{se['closed']}"
                     + (f" tool-ended +{se['tool_ended']}" if se["tool_ended"] else "")
                     + f" (total {t['closed'] + t['tool_ended']}/{t['armed']})")
    if se["decode_lines"]:
        parts.append(f"host KV {se['host_tokens'] / 1e6:.2f}M ({100 * se['host_usage']:.0f}%) "
                     f"peak {se['host_peak'] / 1e6:.2f}M")
    else:
        parts.append("no decode lines")
    parts.append(f"Traceback {se['tracebacks']} (total {t['tracebacks']})")
    lines.append("HEARTBEAT " + " | ".join(parts))
    return lines


def heartbeat(stop: threading.Event, every: float, url: str, key: str, model_base_url: str,
              supervisor: ArmSupervisor, run_dir: Path, watchdog, server_log: Path | None = None,
              reviews=None) -> None:
    """Once every ``every`` seconds (heartbeat_lines): scheduling and progress (middleware), sglang interval amounts
    (/metrics counter differences), plus QSA session / thinking cap / host KV when ``server_log`` is given. All raw
    values go into heartbeat.jsonl.

    The Kaggle notebook log used to show not a single line during the hours the arm runs; this is the only live window.
    An unreadable source is written as ``error`` as is, without stopping the heartbeat -- this is observation, not
    control.
    """
    started = time.monotonic()
    tail = ServerLogTail(server_log) if server_log is not None else None
    if tail is None:
        log("heartbeat: no --server-log, so no QSA session / thinking budget / host KV line")
    start = prev = None
    try:
        start = prev = MetricsSnapshot(time.monotonic(), parse_prometheus(fetch_metrics(model_base_url)))
    except Exception as error:  # noqa: BLE001 — observation, not control
        log(f"heartbeat: no /metrics baseline yet ({repr(error)[:200]})")
    last_play: dict | None = None
    while not stop.wait(every):
        row: dict = {"t": round(time.monotonic() - started)}
        with supervisor.lock:
            plays = list(supervisor.state.get("plays", []))
        games = supervisor.state.get("games", [])
        status = [p.get("status") for p in plays]
        row["games"] = {"total": len(games), "running": status.count("running"),
                        "finished": status.count("finished"), "failed": status.count("failed"),
                        "queued": len(games) - len(plays)}
        try:
            req = urllib.request.Request(url + "/api/progress")
            req.add_header("X-API-Key", key)
            with urllib.request.urlopen(req, timeout=10) as r:
                progress = json.loads(r.read())
            rows = progress["games"]
            row["play"] = {"actions": sum(g["actions_used"] for g in rows),
                           "levels": sum(g["levels_completed"] for g in rows),
                           "wins": sum(1 for g in rows if g.get("state") == "WIN")}
            if "rhae_mean_over_started" in progress:
                row["play"]["rhae_started"] = progress["rhae_mean_over_started"]
            if "record_errors" in progress:
                row["play"]["record_errors"] = progress["record_errors"]
        except Exception as error:  # noqa: BLE001 — observation, not control
            row["play"] = {"error": repr(error)[:200]}
        if "error" not in row["play"]:
            if last_play is not None:
                row["play"]["d_actions"] = row["play"]["actions"] - last_play["actions"]
                row["play"]["d_levels"] = row["play"]["levels"] - last_play["levels"]
            last_play = row["play"]
        try:
            snap = MetricsSnapshot(time.monotonic(), parse_prometheus(fetch_metrics(model_base_url)))
            row["sglang"] = {k: round(v, 3) for k, v in gauges(snap.series).items()}
            row["counters"] = counter_values(snap)
            if prev is None:
                start = prev = snap
                row["server"] = {"baseline": True}
            else:
                row["server"] = server_window(prev, snap, start)
                prev = snap
        except Exception as error:  # noqa: BLE001
            row["sglang"] = {"error": repr(error)[:200]}
            row["server"] = {"error": repr(error)[:200]}
        if tail is not None:
            try:
                row["session"] = tail.read()
            except Exception as error:  # noqa: BLE001
                row["session"] = {"error": repr(error)[:200]}
        if watchdog is not None:
            summary = watchdog.summary()
            row["memory"] = {"min_headroom_gib": summary["min_headroom_gib"],
                             "kills": len(summary["kills"])}
        if reviews is not None:
            row["reviews"] = reviews()
        with (run_dir / "heartbeat.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        try:
            lines = heartbeat_lines(row)
        except Exception as error:  # noqa: BLE001 — a broken format must still be visible; the heartbeat must not stop
            lines = [f"HEARTBEAT format error {repr(error)[:200]} | row {json.dumps(row)[:600]}"]
        for line in lines:
            log(line)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    # The corpus is part of the score. The machine-specific canonical path lives in env/<machine>.sh, so "which one"
    # is not a free parameter typed by hand every time. On 2026-09-18 I typed in an April vendored snapshot whose
    # baselines differed from the canonical ones in 17/25 games, and the whole run could not be compared across runs.
    p.add_argument("--corpus", type=Path,
                   default=(Path(os.environ["ARC_ARC3_CORPUS"])
                            if os.environ.get("ARC_ARC3_CORPUS") else None),
                   help="default $ARC_ARC3_CORPUS (see env/aimo2.sh)")
    p.add_argument("--expect-corpus", metavar="FINGERPRINT",
                   help="refuse to start if the corpus fingerprint differs. For cross-run comparisons pass the previous run's "
                        "corpus.fingerprint from ARM_STATE.json")
    p.add_argument("--prep", type=Path, required=True)
    p.add_argument("--python", required=True, help="venv with arc_agi (used by the gateway)")
    p.add_argument("--agent-python", required=True,
                   help="agent runtime; **without** arcengine, so the jail need not mount the harness venv")
    p.add_argument("--driver-bin", type=Path, required=True,
                   help="driver install prefix (contains bin/<driver>)")
    p.add_argument("--driver", default="codex", choices=arms.available())
    p.add_argument("--driver-runtime", type=Path,
                   help="extra runtime prefix the driver needs (pi needs Node ≥ 22); not for codex")
    p.add_argument("--model-base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--games", nargs="*", help="default: the config's first N games")
    # with several arms in parallel the host gateway ports must differ; the env_port seen inside the jail stays the same.
    p.add_argument("--gateway-port", type=int,
                   help="override the config's gateway_port (host side, must differ per arm)")
    p.add_argument("--official-port", type=int,
                   help="override the config's official_port (official gateway, must differ per arm)")
    p.add_argument("--tag", default="", help="recorded in the run state, for later reconciliation")
    # The environment is an address (S1): without it, start the official gateway locally on --corpus; with it, attach
    # to someone else's -- the Kaggle rerun's gateway:8001, or the stand-in the notebook starts during Save & Run.
    p.add_argument("--upstream", help="someone else's official gateway (e.g. http://gateway:8001); without it, start one locally")
    p.add_argument("--upstream-key-file", type=Path,
                   help="--upstream's key, read from a file (argv is visible to everyone on a host without a pid namespace)")
    p.add_argument("--wall-seconds", type=int, help="override the config's wall_seconds (Kaggle: the notebook computes it from the time left)")
    # posix fence (config fence=posix)
    p.add_argument("--fence-deny", default="", help="colon-separated: paths the agent must not be able to read")
    p.add_argument("--fence-known-readable", default="",
                   help="colon-separated: paths measured as readable and deliberately accepted (recorded in state)")
    p.add_argument("--pyshadow", type=Path, help="directory holding an empty sitecustomize.py (posix)")
    p.add_argument("--ffmpeg", default=shutil.which("ffmpeg"),
                   help="used to encode animation.mp4 when the config turns record on (default: ffmpeg on PATH)")
    p.add_argument("--server-log", type=Path,
                   help="sglang's server.log: the heartbeat counts QSA session decisions, thinking-cap events and host KV occupancy from it")
    p.add_argument("--image-ref-key", type=Path,
                   help="act_image: signing key file (somewhere the agent cannot read); the middleware signs, the model proxy verifies")
    p.add_argument("--admission-dir", type=Path, help="shared whole-game queue; each game budget starts at admission")
    p.add_argument("--admission-owner")
    p.add_argument("--replica-id")
    # single-level episode (run_arc3_level.py, RL): one game, from level K, closed on a level clear or when a stop file
    # appears (src/arcforge/arc_code/level_episode.py)
    p.add_argument("--level-episode", type=int, metavar="K", help="this game starts at level K (1-based) and plays only that level")
    p.add_argument("--action-cap", type=int, help="--level-episode: the most actions counted on this level")
    p.add_argument("--stop-dir", type=Path,
                   help="--level-episode: stop-file directory (host side; shared by the middleware, the proxy and this script). Alone = whole-game episode (RL arc3_game): "
                        "plays from the start to the end; the proxy writes the stop file budget for the output budget, codex dying writes died; after either, no further sessions")
    p.add_argument("--expect-start-hash", help="--level-episode: frame_hash of the opening frame (recorded by the stub)")
    p.add_argument("--level-seed", type=Path,
                   help="--level-episode: spec of the prior-levels seed (json, level_seed.SeedSpec; written by run_arc3_level.py). Before game start the seed is loaded into the "
                        "workspace, the middleware continues the seed's record, and the launcher opens the game")
    p.add_argument("--game-seed", type=Path,
                   help="continue a hard level from gen-2 seed collection: one game (--games names exactly one), the workspace loads this prior-levels seed (json of level_seed.SeedSpec), "
                        "the official gateway opens at its level K, the launcher opens the game, the middleware continues the seed's record, and play then goes on to the end "
                        "(middleware --level-episode K --through)")
    p.add_argument("--episode-action-cap", type=int,
                   help="with --game-seed: a gen-2 single-level episode (seed difficulty evaluation) -- opens at the seed's level K and stops on clearing it "
                        "(the middleware writes cleared) or when level K's actions reach this number (action_cap); output budget from the config's game_output_tokens; "
                        "stop files in <run>/seed_stops. Absent = play on to the end (--through)")
    p.add_argument("--proxy-log", type=Path,
                   help="whole-game episode (--stop-dir alone): run_arc3_level.py's per-episode proxy log (jsonl), "
                        "used to re-split windows when the digest's windows have a gap (repair_windows_from_proxy)")
    p.add_argument("--agent-model-url",
                   help="the jail's model bridge connects here (run_arc3_level.py's per-episode proxy) instead of --model-base-url; "
                        "the launcher's own checks still hit --model-base-url. bwrap only")
    args = p.parse_args(argv)
    episode_args = (args.level_episode, args.action_cap, args.stop_dir)
    if (any(x is not None for x in (args.level_episode, args.action_cap, args.expect_start_hash))
            and any(x is None for x in episode_args)):
        p.error("--level-episode, --action-cap and --stop-dir go together (--expect-start-hash needs them); "
                "--stop-dir alone is a whole-game episode")
    if any((args.admission_dir, args.admission_owner, args.replica_id)) and not all(
            (args.admission_dir, args.admission_owner, args.replica_id)):
        p.error("admission-dir, admission-owner and replica-id are required together")
    if args.server_log is not None and not args.server_log.is_file():
        p.error(f"--server-log {args.server_log} is not a file")

    cfg = json.loads(args.config.read_text())
    if cfg.get("resume_mode") not in RESUME_MODES:
        raise SystemExit(f"the config's resume_mode must be explicit, one of {RESUME_MODES}; "
                         f"got {cfg.get('resume_mode')!r} ({args.config})")
    if not cfg.get("prompts"):
        raise SystemExit(f"the config must name a prompt set in prompts (available: {prompts.available()})")
    if "process_data_max" not in cfg:
        raise SystemExit("the config needs process_data_max (RLIMIT_DATA of every process in the jail, e.g. \"16G\"; must be below game_memory_max)")
    if size_kb(cfg["process_data_max"]) < size_kb("4G"):
        # codex's allocator (static musl binary) reserves several GB of private anonymous mappings at start-up: under 2G
        # it aborts straight away with "memory allocation of 15 bytes failed" (rc=134). It only starts from 4G up.
        raise SystemExit(f"process_data_max {cfg['process_data_max']} is too small: codex needs ≥ 4G to start")
    if size_kb(cfg["process_data_max"]) >= size_kb(cfg.get("game_memory_max", "24G")):
        raise SystemExit(f"process_data_max {cfg['process_data_max']} must be below game_memory_max {cfg.get('game_memory_max', '24G')}, "
                         "otherwise a single process would still hit the scope's MemoryMax first")
    if cfg.get("fence") not in FENCES:
        raise SystemExit(f"the config's fence must be explicit, one of {FENCES}; got {cfg.get('fence')!r}")
    if not isinstance(cfg.get("game_seconds"), int) or cfg["game_seconds"] < 0:
        raise SystemExit("the config needs game_seconds (per-game budget in seconds; 0 = wall ÷ the number of rounds needed)")
    if not isinstance(cfg.get("keep_driver_home"), bool):
        raise SystemExit("the config needs keep_driver_home (true/false): whether to keep the driver home after a game ends")
    if "tail_resume" in cfg and not isinstance(cfg["tail_resume"], bool):
        raise SystemExit(f"tail_resume must be true/false, got {cfg['tail_resume']!r}")
    if cfg.get("tail_resume"):
        minimum = cfg.get("tail_resume_min_seconds")
        if not isinstance(minimum, int) or isinstance(minimum, bool) or minimum < 0:
            raise SystemExit("tail_resume needs tail_resume_min_seconds (int seconds: resume segments open only while the global time left exceeds it)")
    elif "tail_resume_min_seconds" in cfg:
        raise SystemExit("tail_resume_min_seconds only means something with tail_resume: true")
    if cfg.get("tail_resume") and getattr(args, "admission_dir", None):
        raise SystemExit("tail_resume only in base_seconds mode (no --admission-dir)")
    if args.wall_seconds:
        cfg["wall_seconds"] = args.wall_seconds
    if bool(args.upstream) != bool(args.upstream_key_file):
        raise SystemExit("--upstream and --upstream-key-file go together")
    if cfg["fence"] == "posix":
        if os.geteuid() != 0:
            raise SystemExit("fence=posix must start as root (fence_posix.sh drops privileges to each game's user)")
        if not args.fence_deny or args.pyshadow is None:
            raise SystemExit("fence=posix needs --fence-deny and --pyshadow")
        if not (args.pyshadow / "sitecustomize.py").is_file():
            raise SystemExit(f"{args.pyshadow}/sitecustomize.py does not exist")
    budget = cfg.get("game_output_tokens")
    if budget is not None:
        if type(budget) is not int or budget <= 0:
            raise SystemExit(f"game_output_tokens must be a positive int, got {budget!r}")
        if cfg["fence"] != "bwrap":
            raise SystemExit("game_output_tokens only works on bwrap sessions (fence=posix has no polling)")
    driver = arms.load(args.driver)
    if budget is not None and driver.NAME != "codex":
        raise SystemExit(f"game_output_tokens reads the rollout of codex, not of {driver.NAME}")
    context_recovery_config(cfg, driver.NAME)
    for problem in capability_problems(driver, cfg):
        raise SystemExit(f"arms/{driver.NAME} cannot support this config: {problem}")
    # TITO (docs/plans/tito_session_ledger.md) is opt-in: without tito in the config not one line here runs, and
    # arcforge.tito is not imported -- the Kaggle notebook does not ship that package (build_notebook rejects configs
    # with tito).
    tito = None
    if "tito" in cfg:
        from arcforge.tito import manifest as tito_manifest
        from arcforge.tito import preflight as tito_preflight
        if driver.NAME != "codex":
            raise SystemExit(f"tito needs codex (only patch 0002 sends session_id), not {driver.NAME}")
        tito_preflight.check_config(cfg["tito"])
        tito_preflight.check_codex_config(cfg.get("driver_config") or {})
        tito = {"config": tito_manifest.config_identity(args.config)}
    pset = prompts.load(cfg["prompts"], features=note_gate_features(cfg))
    recording = record_config(cfg, pset)
    layout = layout_config(cfg, pset)
    observation_tools = observation_config(cfg, pset)
    act_receipt = receipt_config(cfg, pset)
    act_scene = receipt_scene_config(cfg, pset, act_receipt)
    act_image = image_config(cfg, pset, args.image_ref_key)
    act_library = library_config(cfg, pset, act_image, driver.NAME)
    library_version = arcgame_version(cfg, act_library)
    act_recap = recap_config(cfg, pset, act_library)
    act_expect = expect_config(cfg, pset)
    act_digest = digest_config(cfg, pset, act_expect)
    notes_budget = notes_budget_config(cfg, pset)
    tools = tools_config(cfg, pset)
    if act_expect or notes_budget is not None or tools:
        # fixed string: grep-able after a Kaggle run (the server's require_log does not cover the arm's log)
        log(f"RSI_CHECK config act_expect={act_expect} notes_budget={notes_budget} workspace_tools={tools} "
            f"prompts={pset.name}" + (" act_digest=True" if act_digest else ""))
    gate = level_gate_config(cfg, pset)
    gen2 = gen2_config(cfg, pset, driver.NAME)
    # gen-2's act has note_gate built in (described in the g2 brief, not via a feature fragment)
    note_gate = True if gen2 else note_gate_config(cfg, pset, driver.NAME)
    dynamic = dynamic_context_config(cfg, gen2)
    seed_snapshots = seed_snapshots_config(cfg, gen2, recording)
    level_snapshot_at = level_snapshot_at_config(cfg, seed_snapshots)
    click_info, packet_levels = gen2_options_config(cfg, gen2, pset)
    if not gen2:
        click_info = act_click_info_config(cfg, pset)
    toolkit = gen2_toolkit_config(cfg, gen2, pset, click_info)
    toolkit_act = toolkit["act"] if toolkit is not None else ()
    if not gen2 and click_info:
        log(f"RSI_CHECK config act_click_info=True prompts={pset.name}+{'+'.join(pset.features)}"
            + (f" toolkit={toolkit['name']}:{','.join(sorted(toolkit['files']))} act={','.join(toolkit_act)}"
               if toolkit is not None else ""))
    level_docs = packet_level_docs_config(cfg, gen2, pset)
    pace_line = pace_line_config(cfg, gen2)
    hook_texts = ({**gen2_texts(pset), "seed_snapshots": seed_snapshots, "packet_levels": packet_levels,
                   **({"packet_level_docs": level_docs} if level_docs is not None else {})} if gen2 else None)
    if gen2:
        log(f"RSI_CHECK config harness=gen2 prompts={pset.name} hooks={GEN2_HOOKS}"
            + (f" seed_snapshots=True level_snapshot_at={level_snapshot_at}" if seed_snapshots else "")
            + (" click_info=True" if click_info else "")
            + (f" packet_levels={packet_levels}" if packet_levels is not None else "")
            + (f" packet_level_docs={level_docs}" if level_docs is not None else "")
            + (f" toolkit={toolkit['name']}:{','.join(sorted(toolkit['files']))} act={','.join(toolkit_act)}"
               if toolkit is not None else "")
            + (" pace_line=True" if pace_line else "")
            + (f" dynamic_context={json.dumps(dynamic, sort_keys=True)}" if dynamic else "")
            + " hook_texts=" + ",".join(f for f in (*GEN2_TEXT_FILES.values(), "stuck.md", "idle.md")
                                        if pset.has(f)))
        if args.level_episode is not None:
            raise SystemExit("gen2 does not go with --level-episode")
    elif note_gate:
        log(f"RSI_CHECK config note_gate=True prompts={pset.name}+{'+'.join(pset.features)}")
        if args.level_episode is not None:
            raise SystemExit("note_gate does not go with --level-episode: the episode ends at the clear")
    game_seed = None
    if args.episode_action_cap is not None and args.game_seed is None:
        raise SystemExit("--episode-action-cap goes with --game-seed (a gen-2 single-level episode starts from a seed)")
    if args.game_seed is not None:
        if not gen2:
            raise SystemExit("--game-seed is a gen2 feature (a new session on a played workspace is SessionStart startup)")
        if not args.games or len(args.games) != 1:
            raise SystemExit(f"--game-seed plays one game: --games must name exactly one, not {args.games}")
        game_seed = level_seed.SeedSpec.from_json(json.loads(args.game_seed.read_text())).to_json()
        if args.episode_action_cap is not None and args.episode_action_cap < 1:
            raise SystemExit(f"--episode-action-cap must be >= 1, not {args.episode_action_cap}")
        log(f"RSI_CHECK config game_seed={game_seed['label']} L{game_seed['level']:02d} actions={game_seed['actions']}")
    seed_note_done = seed_note_done_config(cfg, game_seed)
    if seed_note_done:
        log(f"RSI_CHECK config seed_note_done=True (.note_gate/L{game_seed['level'] - 1:02d}.done at load)")
    if dynamic is not None and not gen2:
        log(f"RSI_CHECK config harness=gen1 dynamic_context={json.dumps(dynamic, sort_keys=True)} "
            f"cap={cfg['auto_compact_token_limit']} reminder={cfg['compact_reminder_tokens']}")
        if args.level_episode is not None:
            raise SystemExit("gen-1 dynamic_context does not go with --level-episode (one episode, no slots)")
    compact_reminder_config(cfg, pset)
    level_episode_config(cfg, pset, args)
    game_episode_config(cfg, args)
    # idle-slot read-only reviewer (review_scheduler.py): without a reviewer block in the config, not one line here runs
    review_spec = None
    if "reviewer" in cfg:
        import review_scheduler
        import reviewer
        review_spec = review_scheduler.check_spec(cfg["reviewer"], cfg, driver.NAME)
        if getattr(args, "admission_dir", None):
            raise SystemExit("reviewer only in base_seconds mode (no --admission-dir)")
    card_spec = None
    if "cards" in cfg:
        import review_scheduler
        card_spec = review_scheduler.check_cards_spec(cfg["cards"], review_spec)
    final_spec = final_call_config(cfg, review_spec)
    if recording is not None and not args.ffmpeg:
        raise SystemExit("config turns record on, but ffmpeg was not found (not on PATH and no --ffmpeg given)")
    if args.gateway_port:
        cfg["gateway_port"] = args.gateway_port
    if args.official_port:
        cfg["official_port"] = args.official_port
    run_dir = args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    review_proxy = review_scheduler.check_proxy(run_dir) if review_spec is not None else None
    if act_library:
        act_client.write_library(run_dir / AGENT_LIB, driver.NAME, library_version)
    paths = arms.Paths(agent_runtime=Path(args.agent_python).resolve().parents[1],
                       driver_bin=args.driver_bin.resolve(),
                       runtime=args.driver_runtime.resolve() if args.driver_runtime else None)
    installed = driver.installed_version(paths)
    if installed != driver.VERSION:
        raise SystemExit(f"{driver.NAME} installed is {installed}, the adapter pins {driver.VERSION}")
    log(f"driver: {driver.NAME} {installed}  caps={sorted(driver.CAPABILITIES)}  brief_file={driver.BRIEF_FILE}")
    state = {"mode": "upstream" if args.upstream else "offline", "single_pass": True,
             "fence": cfg["fence"], "upstream": args.upstream,
             "driver": {"name": driver.NAME, "version": installed,
                        "capabilities": sorted(driver.CAPABILITIES),
                        "brief_file": driver.BRIEF_FILE,
                        "brief_delivery": "prompt+file" if driver.BRIEF_FILE else "prompt-only",
                        "bin": str(paths.driver_bin), "runtime": str(paths.runtime)},
             "tag": args.tag, "model": args.model,
             "model_base_url": args.model_base_url, "config": cfg,
             "competition_mode": cfg["competition_mode"], "auth": cfg["auth"],
             "started": time.time()}
    if act_library:
        library_text = (run_dir / AGENT_LIB / "arcgame.py").read_text(encoding="utf-8")
        state["arcgame"] = {"version": library_version or "current",
                            "md5": hashlib.md5(library_text.encode()).hexdigest()}
        log(f"arcgame: {state['arcgame']['version']} md5 {state['arcgame']['md5']}")

    report = manifest.verify_prep(args.prep)
    log("prep:\n" + report.render())
    if not report.ok:
        log("prep has FAILs -- continuing, but this run's attributability is discounted")
    state["prep"] = [f.__dict__ for f in report.findings]

    if args.upstream:
        # Games come from upstream (the rerun's 110 games are never on disk). Only GETs here -- in competition mode a
        # scorecard can be opened only once (the second gets 409), and /api/cmd would burn a game for good.
        upstream_key = args.upstream_key_file.read_text().strip()
        if not upstream_key:
            raise SystemExit(f"{args.upstream_key_file} is empty")
        listed = wait_ready(args.upstream, upstream_key, cfg.get("discover_seconds", 600))
        # Sorted: the upstream order means nothing (the stand-in's is dict order), and sorting makes the first N games
        # of Save & Run the same set as the previous rehearsal's, comparable across arms.
        all_games = sorted(g["game_id"] for g in listed)
        if len(set(all_games)) != len(all_games):
            raise RuntimeError(f"the upstream game list has duplicate ids: {all_games}")
    else:
        if args.corpus is None:
            raise RuntimeError(
                "no corpus: pass --corpus, or source env/env.sh so $ARC_ARC3_CORPUS takes effect")
        assert_unambiguous_corpus(args.corpus)
        upstream_key = secrets.token_hex(24)
        all_games = sorted(d.name for d in args.corpus.iterdir() if d.is_dir()) \
            if any(d.is_dir() for d in args.corpus.iterdir()) else \
            sorted(p.stem for p in args.corpus.glob("*.py"))
    order = cfg.get("game_order", "sorted")
    if order not in GAME_ORDERS:
        raise SystemExit(f"game_order must be one of {GAME_ORDERS}, got {order!r}")
    if order == "interleave":
        all_games = interleave_copies(all_games)
    games = args.games or (all_games[:cfg["games"]] if cfg["games"] else all_games)
    unknown = [g for g in games if g not in all_games]
    if unknown:
        raise SystemExit(f"--games has games not in the environment: {unknown}")
    log(f"games ({len(games)}): {games[:6]}{' …' if len(games) > 6 else ''}")

    # Provenance goes into state **before the run starts**. The 2026-09-18 run wrote it only at wrap-up, so the four
    # arms' ARM_STATE.json had no corpus field to compare at all. Upstream mode has no corpus and hence no
    # provenance -- that is the rerun, whose score exists only on the official card.
    if args.corpus is not None:
        base = baselines(args.corpus)
        state["corpus"] = corpus_provenance(args.corpus, full_ids(games, base))
        log(f"corpus: {state['corpus']['fingerprint']}  "
            f"{state['corpus']['date_downloaded_range'][0][:10]}"
            f"..{state['corpus']['date_downloaded_range'][1][:10]}  "
            f"{args.corpus}")
        if args.expect_corpus and state["corpus"]["fingerprint"] != args.expect_corpus:
            raise RuntimeError(
                f"corpus fingerprint mismatch: expected {args.expect_corpus}, "
                f"actual {state['corpus']['fingerprint']} ({args.corpus})")

    # Scheduling: each game gets a base budget; when a slot frees up the next game fills it and inherits the unused
    # time (run_slots). 0 = wall ÷ the number of rounds needed, guaranteeing every game its full base budget.
    rounds = math.ceil(len(games) / cfg["concurrency"])
    cfg["game_seconds_effective"] = cfg["game_seconds"] or cfg["wall_seconds"] // rounds
    state["schedule"] = {"games": len(games), "concurrency": cfg["concurrency"], "rounds": rounds,
                         "wall_seconds": cfg["wall_seconds"],
                         "game_seconds": cfg["game_seconds_effective"],
                         "tail_resume": bool(cfg.get("tail_resume"))}
    log(f"schedule: {len(games)} games, {cfg['concurrency']} at a time, "
        f"{cfg['game_seconds_effective']} s each, wall {cfg['wall_seconds']} s")

    tokens_path, launcher_key, per_game = mint_tokens(run_dir, games)
    ledger_path = run_dir / "ledger.jsonl"
    sockets = run_dir / "sockets"
    sockets.mkdir(parents=True, exist_ok=True)

    # Model: verify before the run that the port serves this arm's model -- the only independent variable of the
    # four-arm comparison is the port->model mapping, and not checking it means not checking the independent variable.
    # The old run.sh:27 had this; it was lost in the rewrite.
    check_model(args.model_base_url, args.model)
    server_context = server_context_len(args.model_base_url)
    kv_pool = server_kv_pool(args.model_base_url) if dynamic is not None else None
    if dynamic is not None:
        state["dynamic_context"] = {**dynamic, "cap": cfg["auto_compact_token_limit"], "kv_pool": kv_pool}
        log(f"dynamic context: KV pool {kv_pool:,} tokens, cap {cfg['auto_compact_token_limit']:,}")
    if tito is not None:
        # the server really is replaying and persisting the ledger; instance goes into state and into the end-of-run
        # tito_manifest.json.
        tito["server"] = tito_preflight.check_server(tito_preflight.server_root(args.model_base_url))
        state["tito"] = tito
        log(f"tito: ledger instance {tito['server']['instance']} in {tito['server']['ledger_dir']}")
    if recording is not None:
        record_tmp = run_dir / "gateway" / "record_tmp"
        record_tmp.mkdir(parents=True, exist_ok=True)
        record_tmp.chmod(0o700)
        record.check_ffmpeg(args.ffmpeg, record_tmp)
        log(f"record: ffmpeg {args.ffmpeg} round-trips losslessly; video {recording}")
        state["record"] = {**recording, "ffmpeg": args.ffmpeg,
                           "video_check_tokens": check_video_input(
                               args.model_base_url, args.model, args.ffmpeg, record_tmp)}
    # driver_files: config key -> file inside the unit (codex's base instructions, compaction prompt).
    # A missing or empty file stops here -- codex would refuse too, but as 28 games each blowing up inside the jail.
    driver_files = {}
    for key, rel in sorted((cfg.get("driver_files") or {}).items()):
        path = (UNIT / rel).resolve()
        if not path.is_file() or not path.read_text().strip():
            raise SystemExit(f"driver_files.{key} = {rel}: {path} does not exist or is empty")
        driver_files[key] = path
    # From v12 the prompt set ships codex's base instructions and compaction prompt, which change along with the set;
    # the same key may not come from both the config and the set (which one wins would become a matter of order).
    for key, path in pset.driver_files().items():
        if driver.NAME != "codex":
            raise SystemExit(f"prompts {pset.name} carries {path.name}, but {driver.NAME} is not codex")
        if key in driver_files:
            raise SystemExit(f"driver_files.{key} comes from both the config and prompts {pset.name}")
        driver_files[key] = path
    driver_files = dict(sorted(driver_files.items()))
    if pset.has("kickoff.md") and driver.BRIEF_FILE is None:
        raise SystemExit(f"prompts {pset.name} opens with kickoff.md and puts the brief only in the brief file of {driver.NAME}, "
                         f"but {driver.NAME} has no brief file: the agent would never get the brief")
    # bwrap: the bridge inside the jail connects model_port to the host; posix has no netns and connects straight to
    # the host's sglang.
    knobs = arms.Knobs(model=args.model,
                       base_url=(f"http://127.0.0.1:{cfg['model_port']}/v1" if cfg["fence"] == "bwrap"
                                 else args.model_base_url.rstrip("/")),
                       effort=cfg["effort"],
                       context_window=cfg["model_context_window"],
                       auto_compact=cfg["auto_compact_token_limit"] + (GEN2_BACKSTOP_TOKENS if gen2 else 0),
                       server_context=server_context,
                       driver_config=tuple(sorted(
                           (str(k), str(v))
                           for k, v in (cfg.get("driver_config") or {}).items()))
                       + (gen2_driver_config(str(Path(args.agent_python)), hook_texts["full"]) if gen2 else ()),
                       driver_files=tuple((k, str(v)) for k, v in driver_files.items()),
                       hook_trust_bypass=gen2)
    knobs = developer_rules(knobs, pset, driver.NAME)
    state["knobs"] = knobs.__dict__
    if review_spec is not None:
        review_pset = reviewer.ReviewPrompts(review_spec["prompts"])
        review_knobs = reviewer.reviewer_knobs(knobs, review_pset, auto_compact=cfg["auto_compact_token_limit"])
        state["reviews"] = {"spec": review_spec, "prompts": {"set": review_pset.name,
                                                             "fingerprint": review_pset.fingerprint},
                            "knobs": review_knobs.__dict__, "proxy": review_proxy, "rows": [], "counts": {}}
        log(f"REVIEW prompts {review_pset.name} {review_pset.fingerprint}; {json.dumps(review_spec)}")
    if card_spec is not None:
        card_prompt = review_scheduler.load_card_prompt(card_spec["prompts"])
        state["cards"] = {"spec": card_spec, "prompts": {"set": card_prompt.name, "fingerprint": card_prompt.fingerprint},
                          "static": sorted(review_scheduler.card_library.static_cards(
                              review_scheduler.card_library.STATIC_DIR)), "rows": [], "counts": {}}
        log(f"REVIEW cards prompts {card_prompt.name} {card_prompt.fingerprint}; {json.dumps(card_spec)}")
    # The values the driver actually applies may differ from the config's (pi ignores context_window and derives
    # three budgets itself). ``effective()`` is an optional protocol function; without it, record the config's values
    # and label the source.
    effective = getattr(driver, "effective", None)
    state["knobs_effective"] = ({"source": driver.NAME, **effective(knobs)} if callable(effective)
                                else {"source": "config"})
    # The prompts' identity must be recorded just like the corpus: act.py is something the brief tells the agent to
    # read, so it counts too. Keys carry no driver name: two arms given the same text must get the same fingerprint.
    state["prompts"] = pset.fingerprint({"act.py": act_client.render(recording is not None, observation_tools,
                                                                     act_receipt, act_scene, act_image,
                                                                     act_library, act_recap, expect=act_expect,
                                                                     notes_budget=notes_budget, digest=act_digest,
                                                                     note_gate=note_gate, gen2=gen2,
                                                                     note_done_snapshot=level_snapshot_at == "note_done",
                                                                     click_info=click_info, toolkit_act=toolkit_act,
                                                                     pace=pace_line),
                                         # gen-2's hooks decide what the model sees at the start of each window: prompts too
                                         **({f"gen2/hooks/{h.name}": h.read_text()
                                             for h in sorted(GEN2_HOOKS.glob("*.py"))} if gen2 else {}),
                                         # the brief tells the agent to use lib/arc.py and ./bg: prompts too
                                         **(workspace_tools.sources() if tools else {}),
                                         **({f"toolkit/{rel}": text for rel, (text, _) in toolkit["files"].items()}
                                            if toolkit is not None else {}),
                                         "addendum": driver.brief_addendum(),
                                         # the brief tells the agent to read help(arcgame): the library is a prompt too
                                         # (in the fingerprint from v19 on)
                                         **({"arcgame.py": act_client.library_source(driver.NAME, library_version)}
                                            if act_library else {}),
                                         **({"observation.py": act_client.observation_source()}
                                            if observation_tools else {}),
                                         **{f"driver_files/{k}": v.read_text()
                                            for k, v in driver_files.items()}})
    log(f"prompts: {'+'.join([state['prompts']['set'], *state['prompts'].get('features', [])])} "
        f"{state['prompts']['fingerprint']}")
    seed = cfg.get("level_episode", {}).get("seed") or game_seed
    if game_seed is not None:
        state["game_seed"] = game_seed
    if seed is not None:
        # Before the middleware starts: its record continues from the seed's last step; workspace() afterwards
        # overwrites AGENTS.md / act.py / act as usual.
        # gen-2 seeds (--game-seed): the record/steps of the first K−1 levels are hard links into the seed store (read,
        # never written; saves half the space); the RL stub still deep-copies.
        # seed_note_done: restore the note_gate marker act wrote at the moment the seed was taken (--game-seed only,
        # checked by seed_note_done_config)
        state["level_seed"] = level_seed.load(level_seed.SeedSpec.from_json(seed), run_dir / "games" / games[0],
                                              link_steps=game_seed is not None, note_done=seed_note_done)
        log(f"level seed {seed['label']} L{seed['level']:02d}: {state['level_seed']['files']} files, "
            f"{seed['actions']} prior actions, rewrote {len(state['level_seed']['rewritten'])} file(s)")

    state["games"] = games
    supervisor = ArmSupervisor(run_dir, state)
    owned_processes = []
    log_handles = []
    listeners = []
    watchdog = None
    scheduler = None
    heartbeat_stop = threading.Event()
    try:
        gw_env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        gateway_dir = run_dir / "gateway"
        gateway_dir.mkdir(parents=True, exist_ok=True)
        gateway_dir.chmod(0o700)
        step_timeout_log = None
        mw_log = (gateway_dir / "middleware.log").open("w")
        log_handles.append(mw_log)
        # Competition key: known only to the middleware, never visible to the agent -- hence a file, not argv.
        # Random locally; in upstream mode it is the organiser's (Kaggle rerun) or the stand-in's.
        key_file = gateway_dir / "upstream_key"
        old_umask = os.umask(0o077)
        try:
            key_file.write_text(upstream_key + "\n")
        finally:
            os.umask(old_umask)

        if args.upstream:
            upstream_url = args.upstream.rstrip("/")
        else:
            # 1) the official gateway, as is. Game semantics and scoring live here; we add not one line.
            official_log = (gateway_dir / "official.log").open("w")
            log_handles.append(official_log)
            upstream_url = f"http://127.0.0.1:{cfg['official_port']}"
            jail = cfg.get("gateway_jail")
            if jail not in (None, "bwrap"):
                raise SystemExit(f"gateway_jail must be absent or 'bwrap', got {jail!r}")
            # per-step timeout (step_guard.py): a hung game disables only that game, without dragging down the other
            # games on the same gateway
            step_timeout = cfg.get("gateway_step_timeout")
            if step_timeout is not None and (isinstance(step_timeout, bool) or not isinstance(step_timeout, (int, float))
                                             or not step_timeout > 0):
                raise SystemExit(f"gateway_step_timeout must be absent or a positive number, got {step_timeout!r}")
            if jail == "bwrap":
                # The game .py files are exec'd inside the gateway: untrusted (community) games must be locked up. No
                # network, no view of $HOME or the run directory (key file), writes only to recordings and
                # official_out; a host-side bridge connects official_port to the jail's unix socket.
                sock_dir, out_dir = gateway_dir / "jail_sock", gateway_dir / "official_out"
                sock_dir.mkdir(mode=0o700)
                out_dir.mkdir()
                (run_dir / "recordings").mkdir(parents=True, exist_ok=True)
                official_argv = ["/bin/bash", str(UNIT / "gateway_jail.sh"), args.python, str(ROOT / "src"),
                                 str(args.corpus.resolve()), str(cfg["official_port"]), str(sock_dir),
                                 str((run_dir / "recordings").resolve()), str(out_dir.resolve())]
            else:
                official_argv = [args.python, "-m", "arcforge.arc_code.official",
                                 "--corpus", str(args.corpus), "--port", str(cfg["official_port"]),
                                 "--recordings-dir", str(run_dir / "recordings"),
                                 "--scorecard-out", str(run_dir / "scorecard_official.json")]
            if step_timeout is not None:
                step_timeout_log = ((out_dir if jail == "bwrap" else gateway_dir) / "step_timeouts.jsonl").resolve()
                official_argv += ["--step-timeout", str(step_timeout), "--step-timeout-log", str(step_timeout_log)]
            if not cfg["competition_mode"]:
                official_argv.append("--no-competition-mode")
            if "level_episode" in cfg:
                official_argv += ["--start-level", f"{games[0]}:{cfg['level_episode']['level']}"]
            elif game_seed is not None:
                official_argv += ["--start-level", f"{games[0]}:{game_seed['level']}"]
            official = subprocess.Popen(official_argv, env=gw_env, stdout=official_log,
                                        stderr=subprocess.STDOUT)
            owned_processes.append(("official", official))
            if jail == "bwrap":
                official_bridge = subprocess.Popen([args.python, "-I", str(ROOT / "src/arcforge/arc_code/bridge.py"),
                                           "--listen-tcp", f"127.0.0.1:{cfg['official_port']}",
                                           "--connect-unix", str(sock_dir / "official.sock")],
                                          stdout=official_log, stderr=subprocess.STDOUT)
                owned_processes.append(("official_bridge", official_bridge))
            supervisor.update(services={"official": {"pid": official.pid, "status": "starting",
                                                     "jail": jail}})
            log(f"official gateway pid {official.pid} on {upstream_url}" + (" (bwrap jail)" if jail else "")
                + (f" step timeout {step_timeout}s" if step_timeout is not None else ""))
            wait_http(upstream_url + "/api/healthcheck", 120, official)

        # 2) middleware: per-game keys, opening games on the agent's behalf, illegal-action filtering, rate limiting.
        #    The card is opened by the launcher only once everything is ready (open_card below): competition mode
        #    allows only one open, and a gateway with a cleanup loop closes it for you if it idles too long after
        #    opening.
        mw_argv = [args.python, "-m", "arcforge.arc_code.middleware",
                   "--upstream", upstream_url, "--upstream-key-file", str(key_file),
                   "--tokens", str(tokens_path), "--port", str(cfg["gateway_port"]),
                   "--state", str(run_dir / "middleware_state.json"),
                   "--ledger", str(ledger_path),
                   "--rate", str(cfg.get("rate_limit", 10.0)),
                   "--burst", str(cfg.get("burst", 50.0))]
        if args.corpus is not None:
            # baselines only locally (or for a stand-in with a corpus): the launcher-only /api/progress then carries RHAE.
            # The rerun lacks this flag, so the interface honestly has no score; the agent sees it in neither case.
            mw_argv += ["--baselines", str(args.corpus)]
        if recording is not None:
            record_dirs = gateway_dir / "record_dirs.json"
            record_dirs.write_text(json.dumps(
                {g: str(run_dir / "games" / g / "record") for g in games}, indent=1))
            mw_argv += ["--record-dirs", str(record_dirs), "--ffmpeg", args.ffmpeg,
                        "--video-fps", str(recording["video_fps"]),
                        "--video-size", str(recording["video_size"]),
                        *(["--video-max-frames", str(recording["video_max_frames"])]
                          if "video_max_frames" in recording else []),
                        *(["--png-size", str(recording["png_size"])] if "png_size" in recording else []),
                        *(["--video-repeat", str(recording["video_repeat"])]
                          if "video_repeat" in recording else []),
                        "--record-tmp", str(record_tmp)]
        if act_image:
            mw_argv += ["--image-ref-key", str(args.image_ref_key)]
        if gate is not None:
            gates = gateway_dir / "level_gates.json"
            gates.write_text(json.dumps({g: str(run_dir / "games" / g) for g in games}, indent=1))
            mw_argv += ["--level-gates", str(gates),
                        "--level-doc-min-chars", str(gate["min_chars"]),
                        "--compact-min-tokens", str(gate["compact_min_tokens"])]
        if seed_snapshots:
            mw_argv += ["--seed-snapshots", str(run_dir / "seeds"), "--level-snapshot-at", level_snapshot_at]
        if args.upstream:
            # upstream mode always uses full names: workspaces, tokens and the ledger use the upstream ids; short names
            # only bring ambiguity.
            mw_argv += ["--full-ids-only", "--discover-seconds", str(cfg.get("discover_seconds", 600))]
        if game_seed is not None:
            # Open at the seed's level K and play to the end: no action cap, no stop on a clear; a start that does not
            # match the seed writes start_failed (nobody else reads it here: the opening call returns 500 and
            # open_seeded_game raises)
            # RL's gen-2 single-level episode (game_episode + --game-seed): shares the host-side stop-file directory with
            # the proxy (game_episode_config)
            seed_stops = Path(cfg["game_episode"]["stop_dir"]) if "game_episode" in cfg else run_dir / "seed_stops"
            seed_stops.mkdir(exist_ok=True)
            if args.episode_action_cap is None:
                mw_argv += ["--level-episode", str(game_seed["level"]), "--action-cap", "0", "--through"]
            else:
                # single-level episode: a clear, the action cap and a start mismatch all write stop files; when _play
                # sees one it closes the session and opens no more
                mw_argv += ["--level-episode", str(game_seed["level"]), "--action-cap", str(args.episode_action_cap)]
                cfg["g2_episode_stop_dir"] = str(seed_stops)
                state["g2_episode"] = {"level": game_seed["level"], "action_cap": args.episode_action_cap,
                                       "stop_dir": str(seed_stops)}
            mw_argv += ["--stop-dir", str(seed_stops), "--prior-actions", str(game_seed["actions"])]
        if "game_episode" in cfg and game_seed is None:
            # whole-game episode: once the proxy writes budget (or this script writes died) the middleware refuses
            # actions, without waiting for wait_session's TOKEN_POLL_SECONDS poll to close the scope (meanwhile a
            # running script could still send actions and change the score of a stop=budget game).
            # The seeded variant already got the same directory above
            mw_argv += ["--stop-dir", cfg["game_episode"]["stop_dir"]]
        if "level_episode" in cfg:
            episode = cfg["level_episode"]
            mw_argv += ["--level-episode", str(episode["level"]), "--action-cap", str(episode["action_cap"]),
                        "--stop-dir", episode["stop_dir"],
                        *(["--expect-start-hash", episode["start_hash"]] if episode["start_hash"] else []),
                        *(["--prior-actions", str(episode["seed"]["actions"])] if "seed" in episode else [])]
        gateway = subprocess.Popen(mw_argv, env=gw_env, stdout=mw_log,
                                   stderr=subprocess.STDOUT)
        owned_processes.append(("middleware", gateway))
        supervisor.update(services={name: {"pid": proc.pid, "status": "running"}
                                    for name, proc in owned_processes})
        log(f"middleware pid {gateway.pid} on 127.0.0.1:{cfg['gateway_port']} → {upstream_url}")

        url = f"http://127.0.0.1:{cfg['gateway_port']}"
        wait_ready(url, launcher_key, 300 + cfg.get("discover_seconds", 600) * bool(args.upstream))
        card = open_card(url, launcher_key)
        supervisor.update(card_id=card)
        log(f"scorecard {card}")

        if cfg["fence"] == "bwrap":
            # Host side: connect the two TCP services to unix sockets, which the jail connects back to loopback.
            # Host ports differ per arm (four arms in parallel must not clash), but **inside the jail it is always
            # env_port**. That is a design invariant: the address the agent sees is the same in every mode and every
            # arm, so switching upstream gives the agent no hint that its environment changed.
            host_bridges = []
            # --agent-model-url (single-level episode): all of the agent's model traffic goes through the per-episode
            # proxy (output budget, route allowlist, stop files)
            model_host, _, model_port = (args.agent_model_url or args.model_base_url).split("//")[1] \
                .split("/")[0].rpartition(":")
            for name, target in (("env", ("127.0.0.1", cfg["gateway_port"])),
                                 ("model", (model_host or "127.0.0.1", int(model_port)))):
                listener = bridge.listen_unix(sockets / f"{name}.sock")
                listeners.append(listener)
                threading.Thread(
                    target=bridge.serve,
                    args=(listener, lambda t=target: __import__("socket")
                          .create_connection(t, 30)),
                    daemon=True).start()
                host_bridges.append(name)
            log(f"bridges up: {host_bridges}")
        else:
            watchdog = start_posix_fence(supervisor, cfg, args, run_dir, driver_files)

        supervisor.update(status="running", games=games)
        deadline = time.monotonic() + cfg["wall_seconds"]
        work = [(g, workspace(run_dir, g,
                              pset.brief(g, driver.brief_addendum(), limits=brief_limits(pset, cfg)),
                              driver.BRIEF_FILE, cfg["fence"], recording is not None, layout,
                              observation_tools, act_receipt, act_scene, act_image, act_library,
                              act_recap, expect=act_expect, notes_budget=notes_budget, tools=tools,
                              digest=act_digest, note_gate=note_gate, gen2=gen2,
                              note_done_snapshot=level_snapshot_at == "note_done", click_info=click_info,
                              toolkit=toolkit, toolkit_act=toolkit_act, pace=pace_line))
                for g in games]
        if seed is not None:
            opened = open_seeded_game(url, launcher_key, games[0], seed["level"], seed["actions"])
            log(f"{games[0]}: seeded level {seed['level']} opened by the launcher "
                f"(levels_completed {opened['levels_completed']}, record continues after step {seed['actions']})")
        if review_spec is not None:
            posix = cfg["fence"] == "posix"
            review_scheduler.prepare_dirs(run_dir, posix)
            # The reviewer's brief says arcgame is readable (board / frames / history / objects / diff read record/
            # offline). Arms whose player does not use the library (v14, v17's arms) also give the reviewer a copy:
            # frozen c0L-r1, only on the reviewer's PYTHONPATH
            review_library = run_dir / AGENT_LIB
            if not act_library:
                review_library = run_dir / REVIEW_LIB
                act_client.write_library(review_library, driver.NAME, REVIEW_ARCGAME)
            context = review_scheduler.ReviewContext(
                run_dir=run_dir, spec=review_spec, pset=review_pset, driver=driver, knobs=review_knobs, paths=paths,
                agent_runtime=Path(args.agent_python).parents[1], driver_bin=args.driver_bin,
                limits={"process_limit": human_size(cfg["process_data_max"]),
                        "cpu_limit": human_duration(cfg["watchdog_cpu_seconds"])},
                process_data_kb=size_kb(cfg["process_data_max"]), cpu_seconds=cfg["watchdog_cpu_seconds"],
                memory_max=cfg.get("game_memory_max", "24G"), model_port=cfg["model_port"], sockets=sockets,
                model_health_url=args.model_base_url.split("/v1")[0] + "/health",
                library=review_library, deliver_library=None if act_library else review_library,
                pyshadow=args.pyshadow, fence=supervisor.fence,
                cards_static=None if card_spec is None else review_scheduler.card_library.STATIC_DIR,
                cards_live=None if card_spec is None else run_dir / "cards" / "live")
            card_kw = {}
            if card_spec is not None:
                (run_dir / "cards" / "live").mkdir(parents=True, exist_ok=True)
                # card-writing snapshots (the player's notes and record), requests and responses: read only by the
                # harness; the reviewer gets the assembled copy
                (run_dir / "cards").chmod(0o700)
                card_context = review_scheduler.CardContext(
                    run_dir=run_dir, spec=card_spec, prompt=card_prompt, model_base_url=args.model_base_url,
                    model=args.model, cards_static=review_scheduler.card_library.STATIC_DIR,
                    cards_live=run_dir / "cards" / "live", posix=posix)
                card_kw = {"cards": card_spec,
                           "card_one": lambda job, stopping: review_scheduler.run_card(card_context, job, stopping),
                           "card_record": review_scheduler.card_recorder(supervisor, run_dir, posix)}
            scheduler = review_scheduler.ReviewScheduler(
                review_spec, games=dict(work), deadline=deadline, started=deadline - cfg["wall_seconds"],
                review_one=lambda job, stopping: review_scheduler.run_review(context, job, stopping),
                record=review_scheduler.recorder(supervisor, run_dir, posix), cancelled=supervisor.cancelled,
                **card_kw)
            scheduler.start()
            log(f"REVIEW scheduler up: from {review_spec['start_after_seconds']}s, total <= {review_spec['max_total']}, "
                f"reviewers <= {review_spec['max_reviewers']}"
                + (f"; cards after {card_spec['min_total_clears']} clears, writers <= {card_spec['max_writers']}"
                   if card_spec is not None else ""))
        if final_spec is not None:
            # heartbeat_stop is set in finally: if the arm wraps up early, nothing is sent
            threading.Thread(target=final_call, daemon=True, name="final-call",
                             args=(heartbeat_stop, run_dir, [g for g, _ in work], deadline, final_spec,
                                   supervisor, state)).start()
            log(f"FINAL CALL armed: {final_spec['before_end_seconds']}s before the deadline")
        if cfg.get("heartbeat_seconds"):
            threading.Thread(target=heartbeat, daemon=True, name="heartbeat",
                             args=(heartbeat_stop, cfg["heartbeat_seconds"], url, launcher_key,
                                   args.model_base_url, supervisor, run_dir, watchdog, args.server_log),
                             kwargs={"reviews": scheduler.summary if scheduler is not None else None}).start()
        latest: dict[str, dict] = {}              # tail_resume: each game's record of its previous segment

        def run_one(game: str, ws: Path, game_deadline: float, slot_info: dict) -> dict:
            if "resume" in slot_info:
                log(f"{game}: slot {slot_info['slot']} resume segment {slot_info['resume']}, "
                    f"global deadline in {max(0.0, game_deadline - time.monotonic()):.0f}s")
            elif "base_seconds" in slot_info:
                log(f"{game}: slot {slot_info['slot']}, budget {slot_info['base_seconds']:.0f}s"
                    f" + carried {slot_info['carried_seconds']:.0f}s"
                    f" (effective {max(0.0, game_deadline - time.monotonic()):.0f}s, capped by the global deadline)")
            prior = latest[game] if "resume" in slot_info else None
            record = play(game, cfg, args, ws, per_game[game], sockets, game_deadline, ledger_path,
                          driver=driver, knobs=knobs, paths=paths, pset=pset, supervisor=supervisor,
                          earlier=prior)
            # every segment's scheduling info is kept: the top-level slot / base_seconds / carried_seconds are the
            # first segment's.
            record = {**record, **(prior["segments"][0] if prior else slot_info),
                      "segments": [*(prior["segments"] if prior else []), slot_info]}
            if prior:
                record = {**record, "resumed": slot_info["resume"]}
            supervisor.play(record)
            latest[game] = record
            return record

        results = []
        base = None if getattr(args, "admission_dir", None) else cfg["game_seconds_effective"]
        tail = ({"standing": lambda g: ledger_standing(ledger_path, g),
                 "min_resume_seconds": cfg["tail_resume_min_seconds"]}
                if cfg.get("tail_resume") else {})
        budget_files = (WindowBudget(dynamic, cfg["auto_compact_token_limit"], kv_pool, dict(work), hook_texts)
                        if gen2 else None)
        compaction = (CompactionBudget(dynamic, cfg["auto_compact_token_limit"], kv_pool,
                                       cfg["compact_reminder_tokens"],
                                       pset.optional("compact_reminder.md"), dict(work))
                      if dynamic is not None and not gen2 else None)
        watchers = [w.occupancy for w in (scheduler, budget_files, compaction) if w is not None]
        if watchers:
            tail["occupancy"] = lambda snapshot: [watch(snapshot) for watch in watchers]
        if compaction is not None:
            compaction.start()

        # SIGINT / SIGTERM: no new sessions, stop the running session scopes, collect each game's result as usual, then
        # close the card and write the summary and tito_manifest as usual (SIGINT used to raise KeyboardInterrupt
        # straight out of run_slots' join; no manifest was written and the ledger's segments could not be matched back
        # to games).
        def stop_sessions() -> None:
            with supervisor.lock:
                live = [r for r in supervisor.scopes.values() if r["status"] != "cleaned"]
            for receipt in live:
                try:
                    stop_owned_scope(receipt["unit"], receipt["description"])
                except Exception as error:  # noqa: BLE001 — the session's own finally cleans up
                                            # again and records the failure
                    log(f"signal stop: {receipt['unit']}: {error!r}")

        def on_signal(signum, _frame) -> None:
            if supervisor.cancelled.is_set():
                log(f"signal {signum}: already stopping")
                return
            log(f"signal {signum}: no new sessions; stopping the running ones, then scoring and the manifest")
            supervisor.cancelled.set()
            threading.Thread(target=stop_sessions, name="signal-stop", daemon=True).start()

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)
        for (game, _), outcome in zip(work, run_slots(work, cfg["concurrency"], base, deadline, run_one, **tail)):
            if not isinstance(outcome, BaseException):
                results.append(outcome)
                continue
            # One game blowing up must not take the whole run down, but it must not be swallowed silently either --
            # record it; the game naturally scores 0, and ARM_STATE can say why.
            with supervisor.lock:
                previous = next((row for row in state["plays"] if row["game"] == game), {})
                failed = failed_play(previous, game, outcome)
                results.append(failed)
                supervisor.play(failed)
            # Operator log kept verbatim (04_main.py's LIVE filter matches "拋出"). Gist: "<game>: the session loop raised
            # <exception>" followed by the traceback
            log(f"{game}: session 迴圈拋出 {outcome!r}\n{failed['error_traceback']}")
        if compaction is not None:
            compaction.stop.set()
            with supervisor.lock:
                state["dynamic_context"] = {**state["dynamic_context"], "writes": dict(compaction.writes)}
            log(f"compaction budget: files written {compaction.writes}")
        heartbeat_stop.set()
        if scheduler is not None:
            scheduler.stop()                      # all players are done: no new reviews;
                                                  # running ones are killed at the next poll
        supervisor.update(plays=results, status="scoring")
        # the remaining wall clock is deliberately given up -- the competition is single-pass only, and local runs
        # should be no different.
        state["stopped"] = "single_pass: up to max_sessions driver sessions per game, never a second environment"
        # Always close the card before wrapping up: it is the only authoritative score. The previous run did not, and
        # the score was only rescued because recordings/ happened to be on. **Close only here** -- after every game has
        # ended. In competition mode the card can be opened only once; closing early is what happened on 2026-09-18:
        # all later games died.
        try:
            state["official"] = close_card(url, launcher_key)
            (run_dir / "scorecard_official.json").write_text(
                json.dumps(state["official"], indent=2))
            log("official scorecard closed and saved")
        except Exception as error:  # noqa: BLE001
            log(f"CLOSE FAILED: {error!r} — no official score, only the ledger cross-check")
            state["official_error"] = repr(error)
        if step_timeout_log is not None and step_timeout_log.exists():
            rows = [json.loads(x) for x in step_timeout_log.read_text().splitlines() if x.strip()]
            state["step_timeouts"] = rows
            log(f"STEP TIMEOUTS: {len(rows)} event(s); games disabled: {sorted({str(r.get('game')) for r in rows})}")
        summary = summarise(Ledger(ledger_path), games, args.corpus) if args.corpus else None
        state["summary"] = summary
        if state.get("official"):
            state["final"] = official_final(
                state["official"], summary, games,
                full_ids(games, baselines(args.corpus)) if args.corpus else games)
            (run_dir / "FINAL.json").write_text(json.dumps(state["final"], indent=2))
            (run_dir / "FINAL.txt").write_text(render_final(state["final"]))
            log(render_final(state["final"]))
        supervisor.update()
        if summary is not None:
            (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
            log(json.dumps({k: v for k, v in summary.items() if k != "games"}, indent=2))
        if tito is not None:
            if "game_episode" in cfg:
                # do not name the variable play: a closure in main refers to the global play(), and assigning it here
                # would make it a local of main (2026-10-24 g000011 NameError)
                for played in results:
                    for name in (played["digest"]["files"] if "digest" in played else []):   # failed: no digest (tito_items)
                        repaired = repair_windows_from_proxy(dict(work)[played["game"]] / name,
                                                             Path(cfg["game_episode"]["proxy_log"]))
                        if repaired:
                            log(f"TITO REPAIR {played['game']}: {name} lacks the compacted record of window(s) {repaired}, "
                                f"re-split windows by the proxy's session ids (original file {name}.unrepaired)")
                            state.setdefault("tito_windows_repaired", {})[played["game"]] = repaired
                    if "digest" in played and not cfg["keep_driver_home"]:
                        stash = dict(work)[played["game"]] / ROLLOUT_STASH
                        if not stash.is_dir():
                            raise RuntimeError(f"{stash}: harvest left no rollout (a whole-game episode must have one)")
                        bad = played["digest"]["rows"].get("bad_line", 0)
                        if bad:
                            log(f"ROLLOUT BAD LINE {played['game']}: {bad} unparseable line(s) (the driver digest's bad_line rows), "
                                f"rollout kept in {stash}")
                        if played["game"] in state.get("tito_windows_repaired", {}):
                            log(f"TITO REPAIR {played['game']}: rollout kept in {stash} (physical evidence of codex failing to write compacted)")
                        elif not bad:
                            shutil.rmtree(stash)
            # read status once more: a server restart midway (instance changed) is recorded, not raised; unreadable raises.
            tito["server"] = tito_preflight.check_server_end(tito["server"])
            written = tito_manifest.write(run_dir / tito_manifest.FILE, harness="arc3", config=tito["config"],
                                          server=tito["server"],
                                          items=tito_items(results, dict(work), ledger_path))
            log(f"tito: {tito_manifest.FILE} {written['counts']} instances {written['server']['instances']}")
    except BaseException as error:
        supervisor.update(status="failed", error=repr(error), finished=time.time())
        raise
    finally:
        heartbeat_stop.set()
        supervisor.cancelled.set()
        if scheduler is not None:
            # a reviewer failure does not fail the whole arm (the score is on the card, unrelated to reviews); but it
            # must be visible
            scheduler.stop()
            scheduler.join(timeout=review_scheduler.STOP_SECONDS)
            with supervisor.lock:
                state["reviews"]["counts"] = scheduler.summary()
                if scheduler.is_alive():
                    state["reviews"]["stop_error"] = f"still running {review_scheduler.STOP_SECONDS:.0f}s after stop"
                    log(f"REVIEW scheduler did not stop: {state['reviews']['stop_error']}")
            supervisor.update()
        cleanup_errors = supervisor.cleanup_all()
        if watchdog is not None:
            watchdog.stop()
            for uid in list(watchdog.games):
                try:
                    posix_fence.kill_user(uid)
                except Exception as error:  # noqa: BLE001 — try to clean up every uid
                    cleanup_errors.append(f"uid {uid}: {error!r}")
            state["watchdog"] = watchdog.summary()
        for listener in listeners:
            listener.close()
        for name, proc in reversed(owned_processes):
            try:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=10)
            except Exception as error:  # noqa: BLE001 — one failure must not block cleanup of the other owned processes
                cleanup_errors.append(f"{name}: {error!r}")
        for handle in log_handles:
            handle.close()
        supervisor.update(
            services={name: {"pid": proc.pid, "returncode": proc.poll(),
                             "status": "stopped" if proc.poll() is not None else "cleanup_failed"}
                      for name, proc in owned_processes},
            cleanup_errors=cleanup_errors)
        if cleanup_errors:
            supervisor.update(status="failed", finished=time.time())
            if sys.exc_info()[0] is None:
                raise RuntimeError(f"owned resource cleanup failed: {cleanup_errors}")
    incomplete = state.get("official_error") or any(row.get("error") for row in state["plays"])
    supervisor.update(status="incomplete" if incomplete else "finished", finished=time.time())


if __name__ == "__main__":
    main()
