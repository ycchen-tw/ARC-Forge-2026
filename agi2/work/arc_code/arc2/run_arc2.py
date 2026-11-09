"""codex solves ARC-AGI-2: one codex run per task, pass@1.

    python work/arc_code/arc2/run_arc2.py --config work/arc_code/arc2/configs/<cfg>.json \\
        --run-dir <dir> --tasks-dir <ARC-AGI-2/data/evaluation> \\
        --agent-python <venv>/bin/python --driver-bin <codex prefix> \\
        --model-base-url http://127.0.0.1:<port>/v1 --model <served name> \\
        --tokenizer-json <the served model's tokenizer.json> --deny <directory holding the answers>[:...]

Per task:
  workspace  task/ (task.json without test outputs, per-grid .txt and .png), AGENTS.md (brief), check.py
  session    codex exec; the first message carries images of all grids (-i); full grids only in the files
  budget     the task's model proxy caps every request's max_output_tokens at the remaining budget (reasoning
             included), so the accounting is exact; when it runs out the task's uid is killed -- answers are
             collected from files, the model is not called again. Usage is written to the workspace's budget.txt.
             When the config has wrapup_tokens, output_token_budget means "time to wrap up": crossing it asks the
             model to wrap up immediately (model_proxy.Wrapup, inserted into the same turn by codex patch 0005,
             codex not killed), and the kill comes only wrapup_tokens later.
  resume     codex exited on its own but train is not all passed and budget remains: exec resume the same thread
             with a nudge (at most max_sessions)
  harvest    the harness runs its own copy of check.py --harvest as the task's uid: answer/test_J.json first, then
             transform(test input). attempt_1 = that grid; pass@1 looks only at it
  scoring    official partial credit: per task = fraction of test pairs right, total = mean over 120 tasks

Group mode (``--samples N --only <one task>``, one RL group): N copies of one task run concurrently in the same run
directory, episode name ``<task>__s<k>`` -- workspace tasks/<episode>/, result results/<episode>.json (task field =
episode, task id in tid, plus sample / episode), proxy/<episode>.jsonl, the id in tito_manifest; uid, socket, driver
env and watchdog are also one per episode (episode_resources). concurrency = N, no submission.json.
cache salt (config ``cache_salt: "episode"`` + ``--salt-prefix``): the proxy writes ``cache_salt_field`` (default
cache_salt) = ``<prefix>/<episode>`` into every /responses body, the same for all turns and sessions of an episode.

Fence (``fence: posix``, needs root): one uid per task (uid_base + index), workspace 700, task/ read-only,
RLIMIT_DATA, a per-uid memory watchdog (posix_fence.Watchdog), --deny paths unreadable.
With ``network_namespace`` codex runs in a user+net namespace created by that uid (loopback only; bridge.py connects
the model port from a host-side unix socket back to the same 127.0.0.1:<port>); before each session it verifies inside
that /health works and net_probe is unreachable; running solution.py at harvest also happens in the network-cut
namespace. Colab G4 has no CAP_NET_ADMIN and iptables cannot be used, hence this approach. ``fence: none`` is only for
local mock tests.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import hashlib
import importlib.util
import json
import os
import re
import shutil
import signal
import sqlite3
import urllib.parse
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNIT = HERE.parent
sys.path.insert(0, str(UNIT))
sys.path.insert(0, str(HERE))
import arms  # noqa: E402  (work/arc_code/arms)
import posix_fence  # noqa: E402
import arc2_render as render  # noqa: E402
import model_proxy  # noqa: E402
import image_check  # noqa: E402
import harvest_sandbox  # noqa: E402

SCHEMA_VERSION = 1
CONFIG_KEYS = {
    "schema_version", "concurrency", "fence", "effort", "model_context_window",
    "auto_compact_token_limit", "output_token_budget", "task_wall_seconds", "max_sessions",
    "min_resume_tokens", "prompts", "driver_config", "process_data_max", "game_memory_max",
    "watchdog_headroom", "watchdog_cpu_seconds", "oom_score_adj", "network_namespace", "net_probe",
    "uid_base", "budget_poll_seconds", "heartbeat_seconds", "keep_driver_home",
    "model_error_wait_seconds", "max_failures", "workspace_helper", "midpoint_reminder_tokens", "best_snapshot", "note",
    "view", "check_call_limit",
}
#: Keys that may be absent. ``tito`` ({"required": true}): before starting, confirm sglang is keeping the session
#: ledger; write tito_manifest.json at the end.
#: ``cache_salt`` ("none" = key absent, or "episode"): the proxy writes ``cache_salt_field`` (default cache_salt) =
#: ``<--salt-prefix>/<episode>`` into every /responses body, one value for all turns and sessions of an episode.
#: ``wrapup_tokens``: when output_token_budget is used up, ask the model to wrap up (write solution.py / notes.md, then
#: stop) and kill only after this many more output tokens (model_proxy.Wrapup). Shares codex patch 0005's file with
#: midpoint_reminder_tokens; the two cannot both be on.
#: Three read only by run_arc2_seq (2026-10-24, the user, the C128 Kaggle config): ``ind_prompts`` = the prompt set s2
#: uses (s0 follows ``prompts``); ``refine_prompts`` = the refiner's prompt set (replaces the --refine-prompts
#: default); ``second_solution`` = file name of the second answer in the solver workspace (solution2.py of prompts
#: v3dual), run like solution.py at harvest, counted only if it passes all train pairs.
#: ``driver_env_extra`` (10-24): extra variables written into the driver env file ({name: value}), e.g. codex's jemalloc
#: setting ``_RJEM_MALLOC_CONF``. The env file is expanded with ``$(cat …)``, so values cannot contain whitespace; the
#: driver's own variables cannot be overridden.
#: ``budget_extension`` (10-25, the user: more budget for hard tasks): {"extra_tokens": N, "kinds": [...],
#: "refine_reserve_seconds": S, "margin_seconds": M}. Needs wrapup_tokens: when output passes output_token_budget and the
#: episode's gate grants it, there is no wrap-up yet: the budget moves back by extra_tokens (no message inserted, only
#: budget.txt shows the new cap) and wrap-up comes then (model_proxy.Extension). kinds (first / ind) and the time gate
#: belong to run_arc2_seq; a single run_arc2 solver is always granted (no deadline). Optional ``idle`` (10-30
#: improvement plan §2.4): {"kinds": [...], "max_grants": N}, used only by run_arc2_seq: episodes of these kinds (extra
#: solvers allowed) that have not passed train get further extensions beyond the first while idle (no main-line work
#: queued, fewer running than concurrency), at most N times, again subject to the time gate (model_proxy.Extension's
#: repeat).
#: ``workspace_helper_file`` (10-27, the arc2 harness loop's toolkit): the workspace ``arc.py`` is replaced by this file
#: from the harness directory (e.g. ``arc_v6.py``); only together with workspace_helper. ``workspace_doc_file``: this file
#: from the harness directory is copied into the workspace as ``TOOLS.md`` (the toolkit's reference doc, read by arc_v6's
#: doc()); requires workspace_helper_file. Neither key = the original arc.py, no TOOLS.md.
#: ``workspace_pythonpath`` (arc_v10 / v3y): when true the agent's PYTHONPATH includes the workspace, so scripts in /tmp
#: or in subdirectories can import arc too (v3w: 174 `No module named 'arc'` over eight runs, v3y 0). fence none: the
#: driver env's PYTHONPATH = workspace; fence posix (Kaggle): fence_arc2.sh sets PYTHONPATH = pyshadow:workspace (the
#: empty sitecustomize stays first).
#: ``rerefine_skip_agreed`` (10-27, the user): read only by run_arc2_seq. When true, if s0 and s2 both pass all train
#: pairs and the attempt_1 of s0, s2 and refine_exp agree on every test (refine_exp wrote a1 itself), no re-refiner
#: (filler) is started. Four offline datasets: such tasks eat 52–81% of the re-refiner's tokens, while it gets only
#: 0–0.25 more tasks right, with no right-to-wrong changes; tasks where all three agree but train is not all passed still
#: get one (that is where it helps most).
#: ``reattach_images`` (true): the kickoff message codex keeps after compaction is text only ("The N images attached
#: ...", every image dropped); the proxy swaps the whole message (images and <image> tags) back in
#: (model_proxy.ImageReattach, 10-29). View text only. Key absent = not restored. Each kind (s0 / s2 / refiner / seq /
#: verify) uses the kickoff.md of its own prompt set, and ImageReattach recognizes that episode's kickoff; the handover
#: message of a new thread after a context overflow (resume_fresh.md) is not a kickoff and is not restored (same as
#: main).
#: ``refine_repair_tokens`` (10-30, improvement plan §1.1): read only by run_arc2_seq. If final/ is still incomplete when
#: the refiner stops (missing file, broken file, the two attempts identical), give this many more output tokens and
#: resume the same thread once with the refiner prompts' repair.md (solve's ``repair``). x34 b9e38dc0: the refiner,
#: taking the brief literally, wrote final/attempt_1/test_J.json, and nobody checked again after wrapped_up. Key absent =
#: no repair.
#: ``solver_report`` (10-30, improvement plan §2.1, e.g. "REPORT.md"): read only by run_arc2_seq. An s0, s2 or extra
#: solver workspace without this file (or with under 200 bytes) does not count as done (nudged; = --handoff-file, paired
#: with prompts v4y / v4ydual), and the refiner's candidates include it.
#: ``wrapup_time_seconds`` (10-30, improvement plan §2.5a): when a session's wall clock (the episode limit, including one
#: shortened by the deadline / the refiner cutoff) has fewer than this many seconds left and it is not wrapping up yet,
#: send the wrap-up message from the prompts' wrapup_time.md (model_proxy.Wrapup.arm_by_time); budget.txt also shows the
#: remaining wall clock. Needs wrapup_tokens. Before x34, hitting the wall clock killed outright with no message first.
#: ``time_scale`` (K of the G4 time-scaling simulation, written by g4sim/scale_config.py): run_arc2_seq / kaggle_run
#: divide their wall-clock constants by it; absent = real time.
OPTIONAL_CONFIG_KEYS = {"tito", "cache_salt", "cache_salt_field", "wrapup_tokens", "ind_prompts", "refine_prompts",
                        "second_solution", "driver_env_extra", "budget_extension",
                        "workspace_helper_file", "workspace_doc_file",
                        "workspace_pythonpath", "rerefine_skip_agreed", "reattach_images",
                        "extra_solvers", "extra_solvers_saverun", "refine_repair_tokens", "solver_report",
                        "wrapup_time_seconds", "time_scale", "filler_guard_seconds", "filler_min_left_seconds"}
#: solve's ``repair``: for these stop reasons, repair once when status_fn says the work is not done (wall / deadline /
#: cancelled have no time left; context_overflow has no thread left to resume)
REPAIR_STOPS = ("budget", "budget_low", "wrapped_up", "max_sessions")
EXTENSION_KINDS = ("first", "ind")
CACHE_SALT_MODES = ("none", "episode")
#: sglang's request field. agent-factory-3 sends extra_key, with the same matching semantics, but the fork's C++ radix
#: tree loudly rejects cache_salt and silently ignores extra_key, and extra_key also gets the LoRA id appended --
#: default cache_salt (the two are different namespaces)
DEFAULT_CACHE_SALT_FIELD = "cache_salt"
#: Fields the proxy writes itself: a salt under one of these names would be overwritten by them or overwrite them
PROXY_OWNED_FIELDS = {"max_output_tokens", "input"}
PROMPT_FILES = ("brief.md", "instructions.md", "developer.md", "compact_prompt.md",
                "kickoff.md", "nudge.md")
HARVEST_MARK = "ARC2_HARVEST "
#: sglang's messages rejecting over-long input (the validate and tokenizer paths). codex's auto compact looks at the last
#: reported usage; a truncated response reports no usage, yet its long reasoning still enters the history, so resuming
#: the same thread keeps overflowing (v6: 174k–194k > 155k).
CONTEXT_OVERFLOW = ("longer than the model's context length", "exceeds the model's maximum context length")
#: src/arcforge/arc_code/bridge.py (stdlib only); copied into the run's harness/, both ends of the namespace use the
#: same copy.
BRIDGE = UNIT.parents[1] / "src" / "arcforge" / "arc_code" / "bridge.py"
#: src/arcforge/tito/ (stdlib only), loaded only when the config has tito. Loaded by file path from this repo's copy:
#: .venv is shared across worktrees, so ``import arcforge`` could land on another worktree's editable install (same
#: reason as run_arm._own_arcforge).
TITO = UNIT.parents[1] / "src" / "arcforge" / "tito"
LOCK = threading.Lock()


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def size_kb(text: str) -> int:
    units = {"K": 1, "M": 1024, "G": 1024 ** 2}
    if text[-1] not in units or not text[:-1].isdigit():
        raise ValueError(f"size {text!r} must look like 8G / 512M")
    return int(text[:-1]) * units[text[-1]]


def tito_module(name: str):
    """``preflight`` or ``manifest`` (src/arcforge/tito/)."""
    key = f"arc2_tito_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, TITO / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def load_config(path: Path, concurrency: int | None = None) -> dict:
    """``concurrency``: when the config's concurrency is null, the caller supplies it (kaggle_run's --concurrency);
    giving both is an error."""
    cfg = json.loads(path.read_text())
    missing, unknown = CONFIG_KEYS - set(cfg), set(cfg) - CONFIG_KEYS - OPTIONAL_CONFIG_KEYS
    if missing or unknown:
        raise SystemExit(f"{path}: missing {sorted(missing)}, unknown {sorted(unknown)}")
    if "tito" in cfg:
        preflight = tito_module("preflight")
        preflight.check_config(cfg["tito"])
        preflight.check_codex_config(cfg["driver_config"])
    if cfg["schema_version"] != SCHEMA_VERSION:
        raise SystemExit(f"{path}: schema_version {cfg['schema_version']} != {SCHEMA_VERSION}")
    if "solver_report" in cfg and not (isinstance(cfg["solver_report"], str) and cfg["solver_report"].endswith(".md")
                                       and "/" not in cfg["solver_report"]):
        raise SystemExit(f"solver_report must be a workspace .md file name, not {cfg['solver_report']!r}")
    for key in ("ind_prompts", "refine_prompts"):
        if key in cfg and not (isinstance(cfg[key], str) and cfg[key] and (HERE / "prompts" / cfg[key]).is_dir()):
            raise SystemExit(f"{key} must name a directory under {HERE / 'prompts'}, not {cfg[key]!r}")
    if "reattach_images" in cfg and (cfg["reattach_images"] is not True or cfg["view"] != "text"):
        raise SystemExit("reattach_images must be true (leave it out to turn it off) and needs view text")
    if "second_solution" in cfg:
        name = cfg["second_solution"]
        if not (isinstance(name, str) and name.endswith(".py") and "/" not in name and name != "solution.py"):
            raise SystemExit(f"second_solution must be a workspace .py file name other than solution.py, not {name!r}")
        if cfg["view"] != "text":
            raise SystemExit("second_solution is for view text")
    extra = cfg.get("driver_env_extra", {})
    if not (isinstance(extra, dict) and all(isinstance(k, str) and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k)
                                            and isinstance(v, str) and v and not re.search(r"\s", v)
                                            for k, v in extra.items())):
        raise SystemExit(f"driver_env_extra must map env names to non-empty values without whitespace, not {extra!r}")
    if cfg["fence"] not in ("posix", "none"):
        raise SystemExit(f"fence must be posix or none, not {cfg['fence']!r}")
    if cfg["concurrency"] is None:
        if concurrency is None:
            raise SystemExit(f"{path}: concurrency is null, so the caller has to give it")
        cfg["concurrency"] = concurrency
    elif concurrency is not None:
        raise SystemExit(f"{path} fixes concurrency {cfg['concurrency']}; do not also pass one ({concurrency})")
    for key in ("concurrency", "output_token_budget", "task_wall_seconds", "max_sessions", "budget_poll_seconds"):
        if not (isinstance(cfg[key], (int, float)) and cfg[key] > 0):
            raise SystemExit(f"{key} must be positive, not {cfg[key]!r}")
    if not 0 <= cfg["min_resume_tokens"] < cfg["output_token_budget"]:
        raise SystemExit("need 0 <= min_resume_tokens < output_token_budget")
    for key in ("filler_guard_seconds", "filler_min_left_seconds"):   # x39: kaggle_run's filler thresholds (test configs)
        if key in cfg and (type(cfg[key]) not in (int, float) or isinstance(cfg[key], bool) or cfg[key] <= 0):
            raise SystemExit(f"{key} must be a positive number of (real) seconds")
    if ("filler_guard_seconds" in cfg) != ("filler_min_left_seconds" in cfg):
        raise SystemExit("filler_guard_seconds and filler_min_left_seconds go together")
    reminder = cfg["midpoint_reminder_tokens"]
    wrapup = cfg.get("wrapup_tokens")
    feature = cfg["driver_config"].get("features.pre_compact_reminder") == "true"
    if (reminder is None and wrapup is None) != (not feature):
        raise SystemExit("driver_config features.pre_compact_reminder is on exactly when midpoint_reminder_tokens "
                         "or wrapup_tokens is set")
    if reminder is not None and not (isinstance(reminder, int) and 0 < reminder < cfg["auto_compact_token_limit"]):
        raise SystemExit("midpoint_reminder_tokens must be a positive int below auto_compact_token_limit")
    if wrapup is not None:
        # Both use compact_reminder (patch 0005 inserts only once per context window): once the midway reminder has been
        # inserted, the wrap-up can no longer be inserted
        if reminder is not None:
            raise SystemExit("wrapup_tokens and midpoint_reminder_tokens share codex's compact_reminder: set one")
        if not (isinstance(wrapup, int) and wrapup > cfg["min_resume_tokens"]):
            raise SystemExit("wrapup_tokens must be an int above min_resume_tokens (the wrap-up may need a resume)")
        if cfg["view"] != "text":
            raise SystemExit("wrapup_tokens is for view text")
    if "time_scale" in cfg and not (type(cfg["time_scale"]) in (int, float) and cfg["time_scale"] > 0):
        raise SystemExit(f"time_scale must be a positive number, not {cfg['time_scale']!r}")
    if "wrapup_time_seconds" in cfg and not (type(cfg["wrapup_time_seconds"]) is int and cfg["wrapup_time_seconds"] > 0
                                             and wrapup is not None):
        raise SystemExit("wrapup_time_seconds must be a positive int and needs wrapup_tokens")
    ext = cfg.get("budget_extension")
    if ext is not None:
        keys = {"extra_tokens", "kinds", "refine_reserve_seconds", "margin_seconds"}
        if not (isinstance(ext, dict) and keys <= set(ext) <= keys | {"idle"}):
            raise SystemExit(f"budget_extension must have exactly the keys {sorted(keys)} (and maybe idle), not {ext!r}")
        idle = ext.get("idle")
        if idle is not None and not (isinstance(idle, dict) and set(idle) == {"kinds", "max_grants"}
                                     and isinstance(idle["kinds"], list) and idle["kinds"]
                                     and all(isinstance(k, str) for k in idle["kinds"])
                                     and len(set(idle["kinds"])) == len(idle["kinds"])
                                     and type(idle["max_grants"]) is int and idle["max_grants"] >= 1):
            raise SystemExit(f"budget_extension idle must be {{kinds: [...], max_grants: int >= 1}}, not {idle!r}")
        if wrapup is None:
            raise SystemExit("budget_extension needs wrapup_tokens (the wrap-up moves to the extended budget)")
        if not (isinstance(ext["extra_tokens"], int) and ext["extra_tokens"] > cfg["min_resume_tokens"]):
            raise SystemExit("budget_extension extra_tokens must be an int above min_resume_tokens")
        kinds = ext["kinds"]
        if not (isinstance(kinds, list) and kinds and len(set(kinds)) == len(kinds) and set(kinds) <= set(EXTENSION_KINDS)):
            raise SystemExit(f"budget_extension kinds must be a non-empty subset of {EXTENSION_KINDS}, not {kinds!r}")
        for key in ("refine_reserve_seconds", "margin_seconds"):
            if not (isinstance(ext[key], int) and ext[key] >= 0):
                raise SystemExit(f"budget_extension {key} must be a non-negative int, not {ext[key]!r}")
    if not 0 < cfg["auto_compact_token_limit"] < cfg["model_context_window"]:
        raise SystemExit("need 0 < auto_compact_token_limit < model_context_window")
    if cfg["network_namespace"] and (cfg["fence"] != "posix" or not cfg["net_probe"]):
        raise SystemExit("network_namespace needs fence posix and a net_probe host:port")
    if cfg["view"] not in ("text", "image"):
        raise SystemExit(f"view must be text or image, not {cfg['view']!r}")
    if cfg["view"] == "image":
        # check.py and image substitution both go through the proxy; under posix without a namespace codex connects to
        # sglang directly and never passes the proxy.
        if cfg["fence"] == "posix" and not cfg["network_namespace"]:
            raise SystemExit("view image needs network_namespace under fence posix (the proxy carries the images)")
        if cfg["workspace_helper"]:
            raise SystemExit("view image has no grid helper: set workspace_helper false")
        # "How many cells are wrong" can serve as an oracle: in the smoke run the agent binary-searched every cell of the
        # train outputs from wrong-cell counts (10-04). The text version calls check a median of 5 and at most 14 times
        # per task; that kind of search takes tens to hundreds of calls.
        if not (isinstance(cfg["check_call_limit"], int) and cfg["check_call_limit"] > 0):
            raise SystemExit("view image needs check_call_limit, a positive int")
    elif cfg["check_call_limit"] is not None:
        raise SystemExit("check_call_limit is only for view image")
    if "cache_salt" in cfg and cfg["cache_salt"] not in CACHE_SALT_MODES:
        raise SystemExit(f"cache_salt must be one of {CACHE_SALT_MODES}, not {cfg['cache_salt']!r}")
    if "cache_salt_field" in cfg:
        field = cfg["cache_salt_field"]
        if cfg.get("cache_salt") != "episode":
            raise SystemExit("cache_salt_field only goes with cache_salt episode")
        if not (isinstance(field, str) and field) or field in PROXY_OWNED_FIELDS:
            raise SystemExit(f"cache_salt_field must be a non-empty string other than {sorted(PROXY_OWNED_FIELDS)}, "
                             f"not {field!r}")
    if cache_salt_field(cfg) is not None and cfg["fence"] == "posix" and not cfg["network_namespace"]:
        # Same as view image: under posix without a namespace codex connects to sglang directly, and the salt is added
        # by the proxy.
        raise SystemExit("cache_salt episode needs network_namespace under fence posix (the proxy adds the salt)")
    if "workspace_helper_file" in cfg:
        name = cfg["workspace_helper_file"]
        if not cfg["workspace_helper"] or not isinstance(name, str) or "/" in name or not (HERE / name).is_file():
            raise SystemExit(f"workspace_helper_file needs workspace_helper and a file in {HERE}, not {name!r}")
    if "workspace_pythonpath" in cfg and (cfg["workspace_pythonpath"] is not True or "workspace_helper_file" not in cfg):
        raise SystemExit("workspace_pythonpath must be true (leave it out to turn it off) and needs workspace_helper_file")
    if "rerefine_skip_agreed" in cfg and cfg["rerefine_skip_agreed"] is not True:
        raise SystemExit("rerefine_skip_agreed must be true (leave it out to turn it off)")
    if "workspace_doc_file" in cfg:
        name = cfg["workspace_doc_file"]
        if "workspace_helper_file" not in cfg or not isinstance(name, str) or "/" in name or not (HERE / name).is_file():
            raise SystemExit(f"workspace_doc_file needs workspace_helper_file and a file in {HERE}, not {name!r}")
    return cfg


def cache_salt_field(cfg: dict) -> str | None:
    """Name of the salt field the proxy writes into every /responses body; None = no salt (cache_salt absent or
    "none")."""
    if cfg.get("cache_salt", "none") == "none":
        return None
    return cfg.get("cache_salt_field", DEFAULT_CACHE_SALT_FIELD)


# --------------------------------------------------------------------------- tasks

def load_tasks(tasks_dir: Path, limit: int | None, only: list[str] | None) -> dict[str, dict]:
    files = sorted(tasks_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no *.json in {tasks_dir}")
    tasks = {f.stem: json.loads(f.read_text()) for f in files}
    if only:
        missing = sorted(set(only) - set(tasks))
        if missing:
            raise SystemExit(f"tasks not in {tasks_dir}: {missing}")
        tasks = {t: tasks[t] for t in only}
    if limit is not None:
        tasks = dict(list(tasks.items())[:limit])
    check_tasks(tasks, scored=True)
    return tasks


def check_tasks(tasks: dict[str, dict], scored: bool) -> None:
    """scored: every test must have an output; otherwise none may (a mix means the wrong file was given)."""
    if not tasks:
        raise SystemExit("no tasks")
    for tid, task in tasks.items():
        if not task["train"] or not task["test"]:
            raise SystemExit(f"{tid}: needs train and test pairs")
        for pair in task["train"]:
            render.check_grid(pair["input"]), render.check_grid(pair["output"])
        for test in task["test"]:
            render.check_grid(test["input"])
            if scored != ("output" in test):
                raise SystemExit(f"{tid}: test pair {'has no' if scored else 'has an'} output "
                                 f"({'scored' if scored else 'unscored'} run)")
            if scored:
                render.check_grid(test["output"])


def load_challenges(challenges: Path, solutions: Path | None, limit: int | None) -> tuple[dict[str, dict], bool]:
    """Kaggle format: ``{task_id: {train: [...], test: [{input}]}}``, answers (if any) in a separate file
    ``{task_id: [test_1 output, ...]}``. Order follows the file. Returns (tasks, scored)."""
    data = json.loads(challenges.read_text())
    if not isinstance(data, dict) or not data:
        raise SystemExit(f"{challenges}: not a non-empty {{task_id: task}} object")
    tasks = {tid: {"train": t["train"], "test": [{"input": x["input"]} for x in t["test"]]} for tid, t in data.items()}
    if limit is not None:
        tasks = dict(list(tasks.items())[:limit])
    scored = solutions is not None
    if scored:
        answers = json.loads(solutions.read_text())
        missing = sorted(set(tasks) - set(answers))
        if missing:
            raise SystemExit(f"{solutions} has no answers for {missing[:5]} ({len(missing)} tasks)")
        for tid, task in tasks.items():
            if len(answers[tid]) != len(task["test"]):
                raise SystemExit(f"{tid}: {len(answers[tid])} answers for {len(task['test'])} test inputs")
            for test, grid in zip(task["test"], answers[tid], strict=True):
                test["output"] = grid
    check_tasks(tasks, scored)
    return tasks, scored


def image_names(task: dict) -> list[str]:
    names = []
    for k in range(1, len(task["train"]) + 1):
        names += [f"train_{k}_input.png", f"train_{k}_output.png"]
    names += [f"test_{j}_input.png" for j in range(1, len(task["test"]) + 1)]
    return names


def sizes_text(task: dict) -> str:
    def s(g):
        return f"{len(g)}x{len(g[0])}"
    parts = [f"train {k}: {s(p['input'])} -> {s(p['output'])}" for k, p in enumerate(task["train"], 1)]
    parts += [f"test {j}: {s(t['input'])}" for j, t in enumerate(task["test"], 1)]
    return "; ".join(parts)


def image_markers(task: dict, store: image_check.ImageStore) -> str:
    """Image-only mode: store the task's images in store; return marker text, one image per line (the proxy replaces
    the markers with images before they reach the model)."""
    lines = []
    for k, pair in enumerate(task["train"], 1):
        for side in ("input", "output"):
            lines.append(f"train {k} {side}: " + store.put(f"train_{k}_{side}", render.png(pair[side])))
    for j, test in enumerate(task["test"], 1):
        lines.append(f"test {j} input: " + store.put(f"test_{j}_input", render.png(test["input"])))
    return "\n".join(lines)


def build_image_workspace(ws: Path, task: dict, prompts: dict[str, str], cfg: dict, markers: str) -> None:
    """Image-only mode workspace: no cell values at all. task/IMAGES.md holds only markers; cat-ing it shows the task's
    images again. check.py (the client) is written only once the proxy is up and its port is known."""
    if ws.exists():
        raise RuntimeError(f"{ws} exists already; a run directory is used once")
    (ws / "task").mkdir(parents=True)
    (ws / "task" / "IMAGES.md").write_text(markers + "\n")
    (ws / "AGENTS.md").write_text(prompts["brief.md"].format(
        n_train=len(task["train"]), n_test=len(task["test"]), sizes=sizes_text(task),
        budget_k=cfg["output_token_budget"] // 1000, check_limit=cfg["check_call_limit"]))
    atomic_write(ws / "budget.txt", budget_line(0, cfg["output_token_budget"]))


def build_workspace(ws: Path, task: dict, prompts: dict[str, str], cfg: dict, harness_check: Path) -> list[Path]:
    """The workspace holds only what the task may show: tests have only inputs. Returns the images to attach to the
    first message (in order)."""
    if ws.exists():
        raise RuntimeError(f"{ws} exists already; a run directory is used once")
    (ws / "task").mkdir(parents=True)
    (ws / "answer").mkdir()
    visible = {"train": [{"input": p["input"], "output": p["output"]} for p in task["train"]],
               "test": [{"input": t["input"]} for t in task["test"]]}
    if cfg["workspace_helper"]:
        # Compact JSON (~2 tokens/cell; the default ", " is ~3) + a ready-parsed arc.py (in v1 reading task.json cost
        # 3.7k tokens per task on average)
        (ws / "task" / "task.json").write_text(json.dumps(visible, separators=(",", ":")))
        shutil.copyfile(HERE / cfg.get("workspace_helper_file", "arc.py"), ws / "arc.py")
        if "workspace_doc_file" in cfg:
            shutil.copyfile(HERE / cfg["workspace_doc_file"], ws / "TOOLS.md")
    else:
        (ws / "task" / "task.json").write_text(json.dumps(visible))
    images = []
    for k, pair in enumerate(task["train"], 1):
        for side in ("input", "output"):
            (ws / "task" / f"train_{k}_{side}.txt").write_text(render.grid_text(pair[side]))
            (ws / "task" / f"train_{k}_{side}.png").write_bytes(render.png(pair[side]))
            images.append(ws / "task" / f"train_{k}_{side}.png")
    for j, test in enumerate(task["test"], 1):
        (ws / "task" / f"test_{j}_input.txt").write_text(render.grid_text(test["input"]))
        (ws / "task" / f"test_{j}_input.png").write_bytes(render.png(test["input"]))
        images.append(ws / "task" / f"test_{j}_input.png")
    if cfg["best_snapshot"]:
        (ws / "task" / ".snapshot").write_text("check.py keeps .best/solution.py\n")
    if [p.name for p in images] != image_names(task):
        raise AssertionError("image order drifted from image_names()")
    (ws / "AGENTS.md").write_text(prompts["brief.md"].format(
        n_train=len(task["train"]), n_test=len(task["test"]), sizes=sizes_text(task),
        image_order=", ".join(image_names(task)), budget_k=cfg["output_token_budget"] // 1000))
    shutil.copyfile(harness_check, ws / "check.py")
    atomic_write(ws / "budget.txt", budget_line(0, cfg["output_token_budget"]))
    return images


def budget_line(used: int, budget: int, hard: int | None = None, seconds_left: float | None = None,
                by_time: bool = False) -> str:
    """hard: given only once wrap-up has started (usage past budget); states where it will be stopped. seconds_left:
    given when the config has wrapup_time_seconds, the remaining wall clock. by_time: the wrap-up was triggered by time
    (x39: previously written as over budget, so the refiner saw the wrong stop reason)."""
    if hard is None:
        over = ""
    elif by_time:
        over = f"; time is nearly up: wrap up now, you are stopped when it runs out (or at {hard:,} tokens)"
    else:
        over = f"; over budget: wrap up now, you are stopped at {hard:,}"
    clock = "" if seconds_left is None else f"; time left: about {max(0, int(seconds_left // 60))} min"
    return f"output tokens used: {used:,} of {budget:,}{over}{clock} (updated {time.strftime('%H:%M:%S')})\n"


# --------------------------------------------------------------------------- token meter

class TokenMeter:
    """codex rollout's token_usage_record, summed after deduplicating by response_id. Reads incrementally, consuming only
    up to the last newline."""

    def __init__(self, driver_home: Path) -> None:
        self.sessions = driver_home / "sessions"
        self.offsets: dict[Path, int] = {}
        self.seen: dict[str, dict] = {}

    def poll(self) -> dict:
        if self.sessions.is_dir():
            for path in sorted(self.sessions.rglob("rollout-*.jsonl")):
                self._read(path)
        total = {"output_tokens": 0, "reasoning_output_tokens": 0, "input_tokens": 0,
                 "cached_input_tokens": 0, "responses": len(self.seen)}
        for usage in self.seen.values():
            for key in ("output_tokens", "reasoning_output_tokens", "input_tokens", "cached_input_tokens"):
                total[key] += int(usage.get(key) or 0)
        return total

    def _read(self, path: Path) -> None:
        start = self.offsets.get(path, 0)
        with path.open("rb") as f:
            f.seek(start)
            chunk = f.read()
        end = chunk.rfind(b"\n")
        if end < 0:
            return
        self.offsets[path] = start + end + 1
        for line in chunk[:end].splitlines():
            if b'"token_usage_record"' not in line:
                continue
            row = json.loads(line)
            if row.get("type") != "token_usage_record":
                continue
            payload = row["payload"]
            rid = payload["response_id"]
            self.seen[rid] = payload["usage"]


# --------------------------------------------------------------------------- server

def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def server_root(base_url: str) -> str:
    if not base_url.endswith("/v1"):
        raise SystemExit(f"--model-base-url must end in /v1: {base_url}")
    return base_url[:-3]


def model_healthy(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(server_root(base_url) + "/health", timeout=10) as r:
            return r.status == 200
    except OSError:
        return False


def wait_model_healthy(base_url: str, seconds: float, what: str, cancelled: threading.Event | None = None) -> None:
    deadline = time.monotonic() + seconds
    while not model_healthy(base_url):
        if cancelled is not None and cancelled.is_set():
            return                      # deadline / SIGTERM: stop waiting; the next session sees cancelled at once
        if time.monotonic() > deadline:
            raise RuntimeError(f"{what}: model server not healthy for {seconds:.0f} s")
        time.sleep(10)


def sglang_metrics(base_url: str) -> dict:
    keys = {"sglang:gen_throughput": "gen_tok_s", "sglang:num_running_reqs": "running",
            "sglang:num_queue_reqs": "queued", "sglang:cache_hit_rate": "cache_hit"}
    with urllib.request.urlopen(server_root(base_url) + "/metrics", timeout=10) as r:
        text = r.read().decode()
    out = {}
    for line in text.splitlines():
        name = line.split("{")[0].split(" ")[0]
        if name in keys:
            out[keys[name]] = round(out.get(keys[name], 0.0) + float(line.rsplit(" ", 1)[1]), 3)
    return out


# --------------------------------------------------------------------------- fence

def task_user(index: int, cfg: dict) -> tuple[str, int]:
    uid = cfg["uid_base"] + index
    return f"arc2t-{uid}", uid           # the name follows the uid: two runs with different uid_base never collide


#: Image-only mode sandbox uid = this task's uid + this number (a uid different from the agent's: the transform cannot
#: write anywhere the agent can read).
SANDBOX_UID_OFFSET = 500


def sandbox_user(index: int, cfg: dict) -> tuple[str, int]:
    uid = cfg["uid_base"] + SANDBOX_UID_OFFSET + index
    return f"arc2s-{uid}", uid


def ensure_user(name: str, uid: int) -> None:
    have = posix_fence.uid_of(name)
    if have is None:
        subprocess.run(["useradd", "-M", "-N", "-u", str(uid), "-s", "/bin/bash", name],
                       check=True, capture_output=True)
    elif have != uid:
        raise SystemExit(f"user {name} exists with uid {have}, expected {uid}")


def model_host_port(base_url: str) -> tuple[str, int]:
    url = urllib.parse.urlsplit(base_url)
    if url.hostname != "127.0.0.1" or url.port is None:
        raise SystemExit(f"--model-base-url must be http://127.0.0.1:<port>/v1, not {base_url}")
    return url.hostname, url.port


# --------------------------------------------------------------------------- episodes

@dataclasses.dataclass(frozen=True)
class Episode:
    """One codex run. classic: one per task, name is the task id; group (``--samples N``): N copies of one task run
    concurrently, name = ``<task>__s<k>``. index is the sequence number within this run (uid, socket)."""
    index: int
    task: str
    sample: int | None       # None = classic

    @property
    def name(self) -> str:
        return self.task if self.sample is None else f"{self.task}__s{self.sample}"


def compact_driver_dbs(driver_home: Path) -> None:
    """After an episode ends (codex has exited), merge the WAL of every WAL-mode sqlite in the driver home back into the
    main file and truncate it to 0. codex's state_5.sqlite main file is ~0.3 MB, its WAL ~3.3 MB (the same few pages
    rewritten over and over); one RL round of 512 episodes means ~1.7 GB of duplicate pages. No content is lost
    (checkpoint, not deleting the WAL). Trajectories are the rollout jsonl in sessions/, unaffected."""
    for db in sorted(driver_home.glob("*.sqlite")):
        con = sqlite3.connect(db)
        try:
            busy, log, done = con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        finally:
            con.close()
        if busy != 0 or log != done:
            raise RuntimeError(f"{db}: wal_checkpoint(TRUNCATE) busy={busy} log={log} checkpointed={done} "
                               f"(another connection still open?)")
        for side in ("-wal", "-shm"):
            path = db.with_name(db.name + side)
            if path.exists() and path.stat().st_size != 0:
                raise RuntimeError(f"{path}: {path.stat().st_size} bytes left after checkpoint")


def item_fields(ep: Episode) -> dict:
    """Identity fields at the head of results/<episode>.json and of every failures.jsonl row. ``task`` always equals
    the item id (file name): TITO join (arcforge.tito.join.arc2_results) requires results' task field to equal the file
    name. classic: the task id; group: ``<task>__s<k>``, with the real task id in ``tid``."""
    if ep.sample is None:
        return {"task": ep.task}
    return {"task": ep.name, "tid": ep.task, "sample": ep.sample, "episode": ep.name}


def make_episodes(task_ids: list[str], samples: int | None) -> list[Episode]:
    if samples is None:
        return [Episode(i, tid, None) for i, tid in enumerate(task_ids)]
    if len(task_ids) != 1 or samples < 2:
        raise ValueError(f"group mode is one task x >= 2 samples, not {task_ids} x {samples}")
    return [Episode(k, task_ids[0], k) for k in range(samples)]


def episode_resources(run_dir: Path, cfg: dict, ep: Episode, salt_prefix: str | None) -> dict:
    """All of one episode's own resources. In group mode N copies of one task run concurrently in the same run
    directory: every item here must differ per episode (tests check each item), and solve() takes them only from here.
    Things in the workspace (budget.txt, .driver = CODEX_HOME, .best, check.py) follow the workspace."""
    user, uid = task_user(ep.index, cfg)
    sandbox, sandbox_uid = sandbox_user(ep.index, cfg)
    return {"workspace": run_dir / "tasks" / ep.name,
            "result": run_dir / "results" / f"{ep.name}.json",
            "proxy_log": run_dir / "proxy" / f"{ep.name}.jsonl",
            "socket": run_dir / "proxy" / f"{ep.index:04d}.sock",       # used only by fence posix
            "driver_env": run_dir / "driver_env" / f"{ep.name}.env",
            "user": user, "uid": uid,                                     # fence posix: one uid per episode
            "sandbox_user": sandbox, "sandbox_uid": sandbox_uid,          # posix + view image
            "watchdog": ep.name,                                          # the Watchdog's game name (peak, kill records)
            "salt": None if salt_prefix is None else f"{salt_prefix}/{ep.name}"}


@dataclasses.dataclass
class Context:
    cfg: dict
    args: argparse.Namespace
    run_dir: Path
    driver: object
    knobs: arms.Knobs
    paths: arms.Paths
    prompts: dict[str, str]
    harness: Path            # run_dir/harness: check.py, pyshadow, instruction files (readable by every uid)
    watchdog: posix_fence.Watchdog | None
    cancelled: threading.Event
    deny: list[str]
    require: list[str]
    sandbox_hide: list[str]  # image-only mode: directories covered with tmpfs in the sandbox
    salt_prefix: str | None  # cache_salt episode: salt = <salt_prefix>/<episode>; None without salt
    #: Kaggle: global deadline (unix seconds; None = none). Once reached, deadline_hit + cancelled: no new episodes,
    #: running ones are killed and harvested.
    deadline: float | None = None
    deadline_hit: threading.Event = dataclasses.field(default_factory=threading.Event)
    #: number of episodes running on each DP; None = no DP routing (solve's dp argument is chosen by the caller, see
    #: run_arc2_seq.pick_dp)
    dp_live: list[int] | None = None
    #: whether tests have outputs (the Kaggle hidden test does not: correct / score recorded as None)
    scored: bool = True
    proxies: dict = dataclasses.field(default_factory=dict)        # episode → TaskProxy (cumulative counts for the heartbeat)
    close_failures: list = dataclasses.field(default_factory=list)  # records of abort / close_session not returning 200


def fence_argv(ctx: Context, ws: Path, user: str, env_file: Path, driver_argv: list[str],
               sock: Path | None, model_url: str) -> tuple[list[str], dict]:
    """model_url: the /v1 codex uses. Under posix without netns it is this episode's TCP proxy: the fence fetches its
    /health as this uid, which also verifies that the proxy's uid check lets the uid through."""
    cfg = ctx.cfg
    if cfg["fence"] == "none":
        script = ('ulimit -d "$1"; cd "$2"; shift 2; exec env -i HOME="$PWD" TMPDIR="$PWD/.tmp" LANG=C.UTF-8 '
                  'TERM=dumb MPLBACKEND=Agg PATH="$ARC2_PATH" $(cat "$ARC2_DRIVER_ENV") "$@"')
        env = {"PATH": "/usr/bin:/bin", "ARC2_DRIVER_ENV": str(env_file),
               "ARC2_PATH": f"{ctx.paths.agent_runtime}/bin:{ctx.paths.driver_bin}/bin:/usr/bin:/bin"}
        (ws / ".tmp").mkdir(exist_ok=True)
        return ["/bin/bash", "-c", script, "fence-none", str(size_kb(cfg["process_data_max"])), str(ws),
                *driver_argv], env
    env = {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin",
        "ARC2_USER": user,
        "ARC2_RUNTIME": str(ctx.paths.agent_runtime),
        "ARC2_DRIVER_BIN": str(ctx.paths.driver_bin),
        "ARC2_DRIVER_ENV": str(env_file),
        "ARC2_PROCESS_DATA_KB": str(size_kb(cfg["process_data_max"])),
        "ARC2_DENY": ":".join(ctx.deny),
        "ARC2_REQUIRE": ":".join(ctx.require),
        "ARC2_PYSHADOW": str(ctx.harness / "pyshadow"),
        "ARC2_MODEL_HEALTH": server_root(ctx.args.model_base_url if cfg["network_namespace"] else model_url) + "/health",
        "ARC2_OOM_SCORE_ADJ": str(cfg["oom_score_adj"]),
        "ARC2_NETNS": "1" if cfg["network_namespace"] else "0",
    }
    if cfg.get("workspace_pythonpath"):
        env["ARC2_PYTHONPATH_WORKSPACE"] = "1"
    if cfg["network_namespace"]:
        env.update({"ARC2_INNER": str(ctx.harness / "netns_inner.sh"), "ARC2_BRIDGE": str(ctx.harness / "bridge.py"),
                    "ARC2_MODEL_PORT": str(model_host_port(ctx.args.model_base_url)[1]),
                    "ARC2_MODEL_SOCK": str(sock), "ARC2_NET_PROBE": cfg["net_probe"]})
    return ["/bin/bash", str(HERE / "fence_arc2.sh"), str(ws), *driver_argv], env


def kill_task(ctx: Context, uid: int, proc: subprocess.Popen) -> int:
    killed = posix_fence.kill_user(uid) if ctx.cfg["fence"] == "posix" else 0
    if proc.poll() is None:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
    return killed


# --------------------------------------------------------------------------- one task

def run_session(ctx: Context, ws: Path, uid: int, user: str, env_file: Path, driver_argv: list[str],
                log_name: str, proxy: model_proxy.TaskProxy, sock: Path | None, wall_left: float,
                model_url: str) -> dict:
    """One codex session. Stop reasons: exit (exited on its own), budget, wall, deadline, cancelled (SIGTERM / whole run
    cancelled)."""
    cfg = ctx.cfg
    budget = cfg["output_token_budget"]
    argv, env = fence_argv(ctx, ws, user, env_file, driver_argv, sock, model_url)
    began = time.monotonic()
    reason = None
    with (ws / log_name).open("w") as out:
        proc = subprocess.Popen(argv, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True)
        while proc.poll() is None:
            time.sleep(cfg["budget_poll_seconds"])
            used = proxy.output_tokens
            wrapping = proxy.wrapup is not None and proxy.wrapup.armed_at is not None
            # once an extension is granted, the budget the model sees is the extended soft
            shown = budget if proxy.wrapup is None else proxy.wrapup.soft
            time_left = wall_left - (time.monotonic() - began)
            atomic_write(ws / "budget.txt", budget_line(used, shown, proxy.budget if wrapping else None,
                                                        time_left if cfg.get("wrapup_time_seconds") else None,
                                                        by_time=wrapping and proxy.wrapup.armed_by == "time"))
            if cfg.get("wrapup_time_seconds") and time_left < cfg["wrapup_time_seconds"] \
                    and proxy.wrapup.arm_by_time(used, time_left):
                log(f"{proxy.task}: {time_left:.0f} s of wall time left: wrap-up message armed by time")
            if proxy.fatal is not None:
                reason = "proxy_fatal"
            elif proxy.exhausted():
                reason = "budget"
            elif time.monotonic() - began > wall_left:
                # for an episode whose wall clock was shortened by the deadline, what it hit is really the deadline
                # (the two clocks differ by less than one poll)
                near = ctx.deadline is not None and time.time() >= ctx.deadline - cfg["budget_poll_seconds"] - 1
                reason = "deadline" if near else "wall"
            elif ctx.cancelled.is_set():
                reason = "deadline" if ctx.deadline_hit.is_set() else "cancelled"
            if reason:
                break
        killed = kill_task(ctx, uid, proc)
    # requests killed mid-generation: abort right away (without waiting for the harvest) so the server does not finish them
    aborted = proxy.abort_unfinished() if reason else []
    failed_abort = [r for r in aborted if r.get("http") != 200]
    if failed_abort:
        with LOCK:
            ctx.close_failures.extend(failed_abort)
        log(f"{proxy.task}: ABORT_REQUEST FAILED: {failed_abort}")
    if proxy.fatal is not None:          # codex may also have given up and exited on its own before the next poll
        raise RuntimeError(f"model proxy: {proxy.fatal}")
    code = proc.returncode
    events = list(ctx.driver.events(ws / log_name))
    failed = [e for e in events if e.kind == "ended"]
    handle = next((e.data["handle"] for e in events if e.kind == "started"), None)
    return {"log": log_name, "returncode": code, "reason": reason or "exit", "killed_leftover": killed,
            "aborted_requests": aborted, "seconds": round(time.monotonic() - began, 1), "handle": handle,
            # codex's top-level error events can also be transient ("Reconnecting... 1/12", followed by turn.completed
            # as usual). A model-side error only when this session has no turn.completed at all (which arms.codex
            # translates to usage).
            "model_error": bool(failed) and reason is None and not any(e.kind == "usage" for e in events),
            "context_overflow": reason is None and any(m in json.dumps(e.data) for e in failed for m in CONTEXT_OVERFLOW),
            "ended": [e.data for e in failed]}


def harvest(ctx: Context, ws: Path, user: str, task: dict, solution: str | None = None, *,
            uid: int | None = None) -> dict:
    """Run the harness's own copy of check.py --harvest (out of the agent's reach) as this task's uid. solution: path
    relative to the workspace.

    If the agent's code hangs at import (check.py only sets an alarm around transform calls) it hits the timeout: kill
    this uid's processes (fence posix must pass uid) and harvest again as "no solution" (reading only
    answer/test_J.json), with load_error recorded as timeout."""
    payload = json.dumps({"train": task["train"], "test": [t["input"] for t in task["test"]]})
    seconds = 60 + 35 * (len(task["train"]) + 2 * len(task["test"]))
    try:
        return _harvest_once(ctx, ws, user, payload, solution, seconds)
    except subprocess.TimeoutExpired:
        if ctx.cfg["fence"] == "posix":
            if uid is None:
                raise RuntimeError(f"{ws.name}: harvest timed out and no uid was given to kill") from None
            posix_fence.kill_user(uid)
        log(f"{ws.name}: harvest of {solution or 'solution.py'} timed out after {seconds} s; answer files only")
        h = _harvest_once(ctx, ws, user, payload, ".harness-no-solution.py", seconds)
        if not (h["load_error"] or "").startswith("FileNotFoundError"):
            raise RuntimeError(f"{ws.name}: the answer-files-only harvest loaded something: {h['load_error']}")
        h["load_error"] = f"harvest timed out after {seconds} s"
        return h


def _harvest_once(ctx: Context, ws: Path, user: str, payload: str, solution: str | None, seconds: float) -> dict:
    py = [str(ctx.paths.agent_runtime / "bin" / "python"), "-I", str(ctx.harness / "check.py"), "--harvest",
          *([str(ws / solution)] if solution else [])]
    env_vars = [f"HOME={ws}", f"TMPDIR={ws}/.tmp", "LANG=C.UTF-8", "MPLBACKEND=Agg",
                f"PYTHONPATH={ctx.harness / 'pyshadow'}", "PATH=/usr/bin:/bin"]
    limit = str(size_kb(ctx.cfg["process_data_max"]))
    if ctx.cfg["network_namespace"]:
        # The agent's code also runs at harvest: network cut there too (lo not needed).
        py = ["unshare", "--user", "--map-root-user", "--net", "--", *py]
    if ctx.cfg["fence"] == "posix":
        argv = ["/bin/bash", "-c", 'ulimit -d "$1"; shift; exec runuser -u "$@"', "harvest", limit,
                user, "--", "env", "-i", *env_vars, "ARC2_HARVEST_SANDBOX=1", *py]
    else:
        isolated = harvest_sandbox.argv(ws, ctx.harness, ctx.paths.agent_runtime, py)
        argv = ["/bin/bash", "-c", 'ulimit -d "$1"; shift; exec "$@"', "harvest", limit, *isolated]
    out = subprocess.run(argv, cwd=ws, input=payload, capture_output=True, text=True, timeout=seconds)
    lines = [l for l in out.stdout.splitlines() if l.startswith(HARVEST_MARK)]
    if out.returncode != 0 or len(lines) != 1:
        raise RuntimeError(f"harvest failed (rc {out.returncode}): {out.stderr[-800:]}")
    return json.loads(lines[0][len(HARVEST_MARK):])


def status_text(h: dict) -> tuple[bool, str]:
    """(train all passed and every test has an output, one sentence for the nudge message)."""
    if h["load_error"]:
        return False, f"solution.py cannot be loaded ({h['load_error']})"
    verdicts = [t["verdict"] for t in h["train"]]
    passed = sum(v == "PASS" for v in verdicts)
    missing = [j for j, t in enumerate(h["test"], 1) if t["transform"] is None and t["answer_file"] is None]
    done = passed == len(verdicts) and not missing
    text = f"check.py passes {passed} of {len(verdicts)} training pairs"
    if missing:
        text += f" and gives no output for test input(s) {missing}"
    return done, text


def solve(ctx: Context, ep: Episode, task: dict, *, extra_files=None, status_fn=None, dp: int | None = None,
          wall_cap: float | None = None, second: str | None = None, extend_gate=None, repair: dict | None = None) -> dict:
    """One episode, from building the workspace to harvesting. Two hooks for run_arc2_seq.py (text mode only):
    ``extra_files(ws)`` places extra files (the previous session's solution, candidates) after the workspace is built and
    before codex starts; ``status_fn(h, ws) -> (done, one sentence for the nudge message)`` replaces status_text in
    judging whether this session counts as done.
    Kaggle: ``dp`` is this episode's routed_dp_rank (None = unspecified); ``wall_cap`` replaces task_wall_seconds (the
    caller has already clamped it to before the deadline). ``second``: file name of the second answer in the workspace
    (the config's second_solution); if that file exists it is harvested the same way after solution.py and recorded in
    the result's ``second`` (whether to adopt it is up to the caller, based on train). ``extend_gate(total) -> dict``:
    when the config has budget_extension, asked at the moment output passes the budget whether to grant an extension
    (model_proxy.Extension); None = this episode is never extended.
    ``repair`` ({"tokens": N, "text": template}, needs status_fn): when a session stops for one of REPAIR_STOPS and
    status_fn says the work is not done, raise the proxy cap to "used + N" (left alone if already enough) and resume the
    same thread once with text.format(status=…, left_k=…); the result's ``stop`` remains the reason that triggered the
    repair, and the repair leg is recorded in ``repair``."""
    cfg = ctx.cfg
    if repair is not None and (status_fn is None or set(repair) != {"tokens", "text"} or repair["tokens"] <= 0):
        raise ValueError(f"repair needs status_fn and exactly a positive tokens and a text, not {repair!r}")
    wall = cfg["task_wall_seconds"] if wall_cap is None else wall_cap
    if cfg["view"] == "image" and (extra_files is not None or status_fn is not None or second is not None):
        raise ValueError("extra_files / status_fn are for view text only")
    res = episode_resources(ctx.run_dir, cfg, ep, ctx.salt_prefix)
    label = ep.name  # messages, proxy and checker below all use the episode name (the task id in classic mode)
    ws, user, uid = res["workspace"], res["user"], res["uid"]
    image_mode = cfg["view"] == "image"
    store = checker = markers = None
    if image_mode:
        store = image_check.ImageStore()
        markers = image_markers(task, store)
        build_image_workspace(ws, task, ctx.prompts, cfg, markers)
        images = []
        posix = cfg["fence"] == "posix"
        sandbox = image_check.Sandbox(res["sandbox_user"] if posix else None,
                                      ctx.paths.agent_runtime / "bin" / "python", ctx.harness / "sandbox_runner.py",
                                      ctx.sandbox_hide, size_kb(cfg["process_data_max"]))
        checker = image_check.ImageChecker(label, task, ws, uid if posix else None, sandbox, store,
                                           cfg["check_call_limit"])
    else:
        images = build_workspace(ws, task, ctx.prompts, cfg, ctx.harness / "check.py")
        if extra_files is not None:
            extra_files(ws)

    def collect(solution: str | None = None) -> dict:
        if image_mode:
            return checker.harvest(None if solution is None else checker.best[1])
        return harvest(ctx, ws, user, task, solution, uid=uid)
    driver_home = ws / ".driver"
    ctx.driver.prepare(driver_home, ws, ctx.knobs)
    wrap = None
    if cfg.get("wrapup_tokens") is not None:
        # Wrap-up (model_proxy.Wrapup): patch 0005 reads compact_reminder at every follow-up; until the line is crossed
        # it holds an empty message (no insertion)
        extension = None
        if extend_gate is not None:
            if cfg.get("budget_extension") is None:
                raise RuntimeError(f"{label}: extend_gate without budget_extension in the config")
            extension = model_proxy.Extension(cfg["budget_extension"]["extra_tokens"], extend_gate,
                                              repeat="idle" in cfg["budget_extension"])
        wrap = model_proxy.Wrapup(driver_home / model_proxy.REMINDER_FILE, cfg["output_token_budget"],
                                  cfg["output_token_budget"] + cfg["wrapup_tokens"], ctx.prompts["wrapup.md"],
                                  extension=extension,
                                  time_template=ctx.prompts["wrapup_time.md"] if cfg.get("wrapup_time_seconds") else None)
        wrap.dormant()
    elif cfg["midpoint_reminder_tokens"] is not None:
        # codex patch 0005: once the context passes the token count on the first line, the remaining text is inserted
        # into the in-progress turn as a user message, once per context window. With compaction off (since v3) that
        # means once per task; context ≈ output used + prompt + tool outputs.
        (driver_home / model_proxy.REMINDER_FILE).write_text(
            f"{cfg['midpoint_reminder_tokens']}\n{ctx.prompts['midpoint_reminder.md']}")
    env_file = res["driver_env"]
    driver_env = ctx.driver.env(driver_home, ws, ctx.paths)
    clash = sorted(set(driver_env) & set(cfg.get("driver_env_extra", {})))
    if clash:
        raise RuntimeError(f"driver_env_extra would override the driver's own {clash}")
    driver_env.update(cfg.get("driver_env_extra", {}))
    if cfg.get("workspace_pythonpath"):
        if "PYTHONPATH" in driver_env or any(ch.isspace() for ch in str(ws)):
            raise RuntimeError(f"workspace_pythonpath: the driver env already has PYTHONPATH, or {ws} has a space")
        if cfg["fence"] == "none":
            driver_env["PYTHONPATH"] = str(ws)          # posix: fence_arc2.sh puts it after the pyshadow
    env_file.write_text("".join(f"{k}={v}\n" for k, v in driver_env.items()))
    if ctx.watchdog is not None:
        ctx.watchdog.watch(uid, res["watchdog"])
    meter = TokenMeter(driver_home)          # for reconciliation; the budget goes by the proxy's count
    salt = res["salt"]
    # posix without netns (Kaggle: unshare EPERM): codex goes through this episode's own TCP proxy, which admits only
    # this uid
    posix_tcp = cfg["fence"] == "posix" and not cfg["network_namespace"]
    # the kickoff text (ctx.prompts is this episode's own set: s0, s2 / ind, the refiners and seq/verify each replace it)
    # is built before the proxy: ImageReattach has to recognise exactly the message codex sends first
    kickoff = ctx.prompts["kickoff.md"].format(n_images=len(images), image_order=", ".join(image_names(task)),
                                               images=markers or "")
    reattach = model_proxy.ImageReattach(kickoff) if cfg.get("reattach_images") else None
    proxy = model_proxy.TaskProxy(label, cfg["output_token_budget"] if wrap is None else wrap.hard,
                                  model_host_port(ctx.args.model_base_url),
                                  res["proxy_log"], ctx.knobs.server_context,
                                  images=store, checker=checker.check if checker else None,
                                  salt_field=None if salt is None else cache_salt_field(cfg), salt=salt,
                                  dp_rank=dp, allowed_uid=uid if posix_tcp else None, wrapup=wrap,
                                  reattach=reattach)
    with LOCK:
        ctx.proxies[label] = proxy
    sock = None
    knobs = ctx.knobs
    if cfg["fence"] == "posix" and cfg["network_namespace"]:
        sock = res["socket"]
        model_proxy.unix_path_ok(sock)
        server = model_proxy.serve_unix(sock, proxy)
        os.chown(sock, uid, -1)              # only this task's uid can connect to its own proxy
    else:
        server, port = model_proxy.serve_tcp(proxy)
        knobs = dataclasses.replace(knobs, base_url=f"http://127.0.0.1:{port}/v1")
    if image_mode:
        # Inside the namespace the bridge listens on the model's port (codex's base_url unchanged); with fence none
        # it is the proxy's tcp port.
        (ws / "check.py").write_text(image_check.client_source(model_host_port(knobs.base_url)[1]))
    sessions: list[dict] = []
    began = time.monotonic()
    handle = None
    counted = 0
    model_errors = 0
    stop = None
    status = ""            # check result after the previous session, for the nudge message
    fresh_note = None      # handover message for a new thread after a context overflow
    resume_text = None     # the next resume uses this instead of the nudge (wrap-up message, repair message)
    repair_rec = None      # the repair's record once repaired (repaired only once)
    overflows = 0
    h = None               # the last harvest
    closed: list[dict] = []
    aborted: list[dict] = []
    #: Bookkeeping steps after the harvest that failed (upstreams still open 30 s after abort, driver db checkpoint):
    #: recorded loudly, answers kept (review 10-12: originally the whole episode was marked failed, and the answers
    #: already harvested were lost with it)
    teardown_errors: list[str] = []

    def repairing(reason: str, done: bool, used: int) -> bool:
        """``repair``: if a repair is due, prepare the resume message and the cap and return True (the caller
        continues)."""
        nonlocal resume_text, repair_rec
        if repair is None or repair_rec is not None or done or handle is None or reason not in REPAIR_STOPS:
            return False
        before = proxy.budget
        if before < used + repair["tokens"]:
            proxy.raise_budget(used + repair["tokens"])
        if wrap is not None and wrap.armed_at is not None:
            wrap.dormant()       # an undelivered wrap-up message is no longer inserted: the repair message replaces it
        resume_text = repair["text"].format(status=status, left_k=(proxy.budget - used) // 1000).strip()
        repair_rec = {"after": reason, "status": status, "used": used, "budget_before": before,
                      "budget": proxy.budget, "session": len(sessions) + 1}
        log(f"{label}: session {len(sessions)} stopped ({reason}) with {status}; one repair session, "
            f"budget {before:,} -> {proxy.budget:,}")
        return True

    try:
        while True:
            wall_left = wall - (time.monotonic() - began)
            if wall_left <= 0:
                stop = "wall"
                break
            if handle is None:
                first = kickoff if fresh_note is None else fresh_note
                argv = ctx.driver.argv(first, knobs)
                # Attach images to the first message. `codex exec -i <FILE>...` is greedy: placed before the prompt it
                # takes the prompt as an image too, and codex stalls at "Reading prompt from stdin". So it goes right
                # after exec, before the next --flag.
                if argv[:3] != ["codex", "exec", "--json"]:
                    raise RuntimeError(f"unexpected codex argv shape {argv[:3]}")
                driver_argv = argv if image_mode else argv[:2] + ["-i", ",".join(str(p) for p in images)] + argv[2:]
            else:
                if resume_text is not None:
                    message, resume_text = resume_text, None
                else:
                    shown = cfg["output_token_budget"] if wrap is None else wrap.soft
                    message = ctx.prompts["nudge.md"].format(status=status, used_k=proxy.output_tokens // 1000,
                                                              budget_k=shown // 1000)
                driver_argv = ctx.driver.resume_argv(handle, message, knobs)
            name = "driver.log" if not sessions else f"driver.{len(sessions) + 1}.log"
            row = run_session(ctx, ws, uid, user, env_file, driver_argv, name, proxy, sock, wall_left,
                              knobs.base_url)
            if row["returncode"] in (64, 70) and row["reason"] == "exit" and row["handle"] is None:
                # The fence refused to run (64 broken declaration, 70 answers or internet reachable): a harness fault;
                # stop the whole run.
                raise RuntimeError(f"{label}: fence refused with exit {row['returncode']}: "
                                   f"{(ws / name).read_text()[-600:]}")
            handle = row["handle"] or handle
            sessions.append(row)
            h = collect()
            done, status = status_text(h) if status_fn is None else status_fn(h, ws)
            row["status_after"] = status
            used = proxy.output_tokens
            if row["reason"] != "exit":
                stop = row["reason"]
                if repairing(stop, done, used):
                    continue
                break
            if proxy.exhausted():          # budget may also run out as codex exits by itself (e.g. gives up after 12 retries)
                stop = "budget"
                if repairing(stop, done, used):
                    continue
                break
            if ctx.cancelled.is_set():
                stop = "deadline" if ctx.deadline_hit.is_set() else "cancelled"
                break
            if row["context_overflow"]:
                # Resuming the same thread would only overflow again: open a new thread and carry on from the workspace
                # files (not counted toward max_sessions).
                overflows += 1
                if overflows > cfg["max_sessions"]:
                    stop = "context_overflow"
                    break
                handle = None
                fresh_note = ctx.prompts["resume_fresh.md"].format(
                    status=status, used_k=proxy.output_tokens // 1000, image_order=", ".join(image_names(task)),
                    images=markers or "")
                log(f"{label}: context overflow in {name}; fresh thread #{overflows} from the workspace")
                continue
            if row["model_error"] or (row["returncode"] == 71 and row["handle"] is None):
                model_errors += 1
                if model_errors > 20:
                    raise RuntimeError(f"{label}: more than 20 model errors")
                log(f"{label}: model error in {name}: {row['ended']}; waiting for /health")
                wait_model_healthy(ctx.args.model_base_url, cfg["model_error_wait_seconds"], label, ctx.cancelled)
                continue
            counted += 1
            if done:
                stop = "finished"
                break
            if wrap is not None and wrap.armed_at is not None:
                # Past the wrap-up line and codex stopped on its own: if the wrap-up message was delivered (inserted into
                # the turn by patch 0005, or by the previous resume), stop here; if not yet delivered (no follow-up after
                # the response that crossed the line), resume the same thread once with it, not limited by max_sessions
                if wrap.seen_at is not None or wrap.resumed:
                    stop = "wrapped_up"
                    if repairing(stop, done, used):
                        continue
                    break
                if proxy.budget - used < cfg["min_resume_tokens"]:
                    stop = "budget_low"
                    if repairing(stop, done, used):
                        continue
                    break
                wrap.resumed = True
                resume_text = wrap.text
                log(f"{label}: session {len(sessions)} ended past the budget before the wrap-up message reached it; "
                    f"resuming with it")
                continue
            if counted >= cfg["max_sessions"]:
                stop = "max_sessions"
                if repairing(stop, done, used):
                    continue
                break
            if proxy.budget - used < cfg["min_resume_tokens"]:
                stop = "budget_low"
                if repairing(stop, done, used):
                    continue
                break
            log(f"{label}: session {len(sessions)} exited unfinished ({status}); resuming")
    finally:
        # codex is gone (exited or killed). Abort requests still generating, cut the upstreams, then send the missing
        # closes -- a killed codex does not close, and L4 HiSparse has no idle timeout, so without this the session's
        # logical tokens stay held. abort_upstreams raises only if forwarding threads are still alive 30 s after the
        # disconnect (a harness bug): still close and shut down the proxy, then let this episode end as failed.
        try:
            aborted = proxy.abort_upstreams()
        except RuntimeError as error:
            teardown_errors.append(f"abort_upstreams: {error}")
            log(f"{label}: TEARDOWN FAILED {error} (answers are kept)")
        finally:
            closed = proxy.close_open_sessions()
            server.shutdown()
            server.server_close()
            failed_close = [r for r in closed + aborted if r.get("http") != 200]
            if failed_close:
                with LOCK:
                    ctx.close_failures.extend(failed_close)
                log(f"{label}: CLOSE_SESSION FAILED ({len(failed_close)} abort/close request(s)): {failed_close}")
    if h is None:
        raise RuntimeError(f"{label}: no session ran (stop={stop})")
    if repair_rec is not None:
        repair_rec.update(stop=stop, status_after=status, tokens=proxy.output_tokens - repair_rec["used"],
                          done=status_fn(h, ws)[0])
        stop = repair_rec["after"]
    fallback = None
    best_passed = None
    if cfg["best_snapshot"]:
        # When solution.py was overwritten by a half-finished version (v3: c7f57c3e's train-passing version was
        # overwritten amid garbled output), fall back to the snapshot check.py saved (image-only mode: the source the
        # checker keeps in memory).
        if image_mode:
            best_passed = None if checker.best is None else checker.best[0]
        elif (ws / ".best" / "solution.py").exists():
            best_passed = json.loads((ws / ".best" / "meta.json").read_text())["passed"]
    if best_passed is not None:
        passes = lambda x: sum(t["verdict"] == "PASS" for t in x["train"])  # noqa: E731
        if h["load_error"] or passes(h) < best_passed:
            hb = collect(solution=".best/solution.py")
            fallback = {"current_train_pass": passes(h), "snapshot_train_pass": passes(hb),
                        "current_load_error": h["load_error"], "used": passes(hb) > passes(h)}
            if fallback["used"]:
                h = hb
    second_record = None
    if second is not None and (ws / second).is_file():
        # Only take the transform's output: answer/test_J.json belongs to solution.py, not to the second answer
        h2 = collect(solution=second)
        second_record = {"file": second, "train": [t["verdict"] for t in h2["train"]], "load_error": h2["load_error"],
                         "outputs": [t["transform"] for t in h2["test"]],
                         "transform_errors": [t["transform_error"] for t in h2["test"]]}
    usage = meter.poll()
    usage["proxy_output_tokens"] = proxy.output_tokens
    usage["proxy_requests"] = proxy.requests
    usage["proxy_refused"] = proxy.refused
    usage["check_calls"] = None if checker is None else checker.calls
    usage["check_refused"] = None if checker is None else checker.refused
    usage["proxy_rejected_peers"] = proxy.rejected_peers
    atomic_write(ws / "budget.txt", budget_line(proxy.output_tokens,
                                                cfg["output_token_budget"] if wrap is None else wrap.soft,
                                                None if wrap is None or wrap.armed_at is None else wrap.hard))
    attempts, sources = [], []
    for entry in h["test"]:
        if entry["answer_file"] is not None:
            attempts.append(entry["answer_file"]), sources.append("answer_file")
        elif entry["transform"] is not None:
            attempts.append(entry["transform"]), sources.append("transform")
        else:
            attempts.append(None), sources.append("none")
    if ctx.scored:
        correct = [a is not None and a == t["output"] for a, t in zip(attempts, task["test"], strict=True)]
        scoring = {"correct": correct, "score": sum(correct) / len(correct),
                   # whether each source is right on its own (analysis: did answer_file override a better transform, or
                   # the reverse)
                   "transform_correct": [t["transform"] is not None and t["transform"] == g["output"]
                                         for t, g in zip(h["test"], task["test"], strict=True)],
                   "answer_file_correct": [None if t["answer_file"] is None else t["answer_file"] == g["output"]
                                           for t, g in zip(h["test"], task["test"], strict=True)]}
    else:
        scoring = {"correct": None, "score": None, "transform_correct": None, "answer_file_correct": None}
    digest = ws / "driver_digest.jsonl"
    try:
        digest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ctx.driver.digest(driver_home)))
        if cfg["keep_driver_home"]:
            compact_driver_dbs(driver_home)
        else:
            shutil.rmtree(driver_home)
    except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
        teardown_errors.append(f"driver home: {error!r}")
        log(f"{label}: TEARDOWN FAILED driver home {error!r} (answers are kept)")
    # classic-mode result fields are key-for-key the same as before group / salt were added (pinned by tests); with salt
    # on there is an extra cache_salt.
    return {**item_fields(ep), "index": ep.index, "stop": stop, "sessions": sessions, "usage": usage,
            "seconds": round(time.monotonic() - began, 1), "train": [t["verdict"] for t in h["train"]],
            "load_error": h["load_error"], "attempt_1": attempts, "source": sources, "snapshot_fallback": fallback,
            "transform_errors": [t["transform_error"] for t in h["test"]],
            "answer_errors": [t["answer_error"] for t in h["test"]], **scoring,
            # the proxy's session / abort records (Kaggle: unclosed sessions keep holding HiSparse's logical tokens)
            "wall_cap_seconds": round(wall, 1), "dp_rank": dp, "sessions_seen": sorted(proxy.sessions),
            "sessions_closed_by_harness": closed, "teardown_errors": teardown_errors,
            "requests_aborted_by_harness": [r for x in sessions for r in x["aborted_requests"]] + aborted,
            "vision_tokens": sum(render.vision_tokens(g) for p in task["train"] for g in (p["input"], p["output"]))
            + sum(render.vision_tokens(t["input"]) for t in task["test"]),
            **({"images_reattached": reattach.reattached} if reattach is not None else {}),
            **({} if salt is None else {"cache_salt": salt}),
            **({} if wrap is None else {"wrapup": wrap.record(proxy.output_tokens)}),
            **({} if repair is None else {"repair": repair_rec}),
            **({} if second is None else {"second": second_record})}


# --------------------------------------------------------------------------- run

def heartbeat(stop: threading.Event, ctx: Context, state: dict) -> None:
    while not stop.wait(ctx.cfg["heartbeat_seconds"]):
        with LOCK:
            done = list(state["done"])
            running = len(state["running"])
        score = sum(r["score"] for r in done)
        out_tok = sum(r["usage"]["proxy_output_tokens"] for r in done)
        try:
            metrics = sglang_metrics(ctx.args.model_base_url)
        except OSError as error:
            metrics = {"error": repr(error)}
        log(f"HEARTBEAT done {len(done)}/{state['total']} running {running} "
            f"score_sum {score:.2f} out_tok_done {out_tok:,} {json.dumps(metrics)}")


def write_summary(ctx: Context, tasks: dict, results: list[dict], failures: list[dict], started: float) -> dict:
    by_task = {r["task"]: r for r in results}
    submission = {}
    for tid, task in tasks.items():
        r = by_task.get(tid)
        submission[tid] = [{"attempt_1": (r["attempt_1"][j] if r and r["attempt_1"][j] is not None
                                          else t["input"]),
                            "attempt_2": t["input"]} for j, t in enumerate(task["test"])]
    atomic_write(ctx.run_dir / "submission.json", json.dumps(submission))
    out_tokens = [r["usage"]["proxy_output_tokens"] for r in results]
    summary = {
        "tasks": len(tasks), "finished_tasks": len(results), "failed_tasks": [f["task"] for f in failures],
        "score_pass1_partial": round(100 * sum(r["score"] for r in results) / len(tasks), 2),
        "tasks_fully_correct": sum(all(r["correct"]) for r in results),
        "test_pairs_correct": sum(sum(r["correct"]) for r in results),
        "test_pairs": sum(len(t["test"]) for t in tasks.values()),
        "train_all_pass": sum(all(v == "PASS" for v in r["train"]) for r in results),
        "stops": {s: sum(r["stop"] == s for r in results) for s in sorted({r["stop"] for r in results})},
        "output_tokens_total": sum(out_tokens),
        "output_tokens_mean": round(sum(out_tokens) / len(out_tokens)) if out_tokens else None,
        "wall_seconds": round(time.time() - started),
        "scoring": "official ARC-AGI-2 partial score on attempt_1 only (pass@1): per task = correct test pairs / test pairs, mean over all tasks; unfinished tasks count 0",
    }
    atomic_write(ctx.run_dir / "summary.json", json.dumps(summary, indent=1) + "\n")
    return summary


def write_group_summary(ctx: Context, tid: str, samples: int, results: list[dict], failures: list[dict],
                        started: float) -> dict:
    """Group mode: N copies of one task. No submission.json (the attempt_1 of N copies of one task is not one
    submission). Task id in ``tid`` (in group mode the ``task`` field is the episode, see item_fields)."""
    out_tokens = [r["usage"]["proxy_output_tokens"] for r in results]
    summary = {
        "tid": tid, "samples": samples, "finished_episodes": len(results),
        "failed_episodes": sorted(f["episode"] for f in failures),
        "score_pass1_partial": round(100 * sum(r["score"] for r in results) / samples, 2),
        "scores": {r["episode"]: r["score"] for r in sorted(results, key=lambda r: r["sample"])},
        "episodes_fully_correct": sum(all(r["correct"]) for r in results),
        "train_all_pass": sum(all(v == "PASS" for v in r["train"]) for r in results),
        "stops": {s: sum(r["stop"] == s for r in results) for s in sorted({r["stop"] for r in results})},
        "output_tokens_total": sum(out_tokens),
        "output_tokens_mean": round(sum(out_tokens) / len(out_tokens)) if out_tokens else None,
        "wall_seconds": round(time.time() - started),
        "scoring": "official ARC-AGI-2 partial score on attempt_1 of each sample (per sample = correct test pairs / test pairs), mean over the samples of the one task; failed samples count 0",
    }
    atomic_write(ctx.run_dir / "summary.json", json.dumps(summary, indent=1) + "\n")
    return summary


def tito_items(run_dir: Path, results: list[dict], failures: list[dict]) -> list[dict]:
    """One item per episode for ``tito_manifest.json`` (called only when the config has ``tito``): id and workspace are
    the ``task`` field (item_fields: the task id in classic, ``<task>__s<k>`` in group). thread = the sessions' handles
    (the started event; a resume continues the same one); response ids are read from this episode's driver_digest; the
    outcome is the score fields of results. An episode that blew up has neither result nor digest and is recorded
    truthfully as failed."""
    items = [{"id": r["task"], "threads": [s["handle"] for s in r["sessions"] if s["handle"] is not None],
              "digests": [run_dir / "tasks" / r["task"] / "driver_digest.jsonl"],
              "outcome": {"status": "finished", "score": r["score"], "correct": r["correct"], "stop": r["stop"]}}
             for r in results]
    items += [{"id": f["task"], "threads": [], "digests": [], "outcome": {"status": "failed", "error": f["error"]}}
              for f in failures]
    return sorted(items, key=lambda item: item["id"])


SELFTEST_SOURCE = """
import os
import re
def transform(grid):
    found = []
    for path in PATHS:
        try:
            open(path, "rb").read(1)
            found.append("read " + path)
        except OSError:
            pass
    for d in WRITE_DIRS:
        try:
            with open(os.path.join(d, "arc2-sandbox-probe"), "w") as f:
                f.write("x")
        except OSError:
            pass
    if found:
        raise RuntimeError(";".join(found))
    return grid
"""


def sandbox_selftest(ctx: Context, n_tasks: int) -> None:
    """Before an image-only run: the sandbox uid cannot read the answers, what it writes to "places the agent can read"
    is invisible outside, and outputs come back normally."""
    if ctx.cfg["fence"] != "posix":
        return
    for d in ctx.sandbox_hide:
        if not Path(d).is_dir():
            raise SystemExit(f"sandbox hide path {d} is not a directory")
    probe_files = [Path(d) / "arc2-sandbox-probe" for d in ctx.sandbox_hide]
    for f in probe_files:
        if f.exists():
            f.unlink()
    answers = [str(f) for d in ctx.deny for f in sorted(Path(d).rglob("*.json"))[:3]]
    if not answers:
        raise SystemExit(f"no *.json under --deny {ctx.deny} to probe the sandbox with")
    for i in range(n_tasks):
        name, _ = sandbox_user(i, ctx.cfg)
        for path in ctx.deny:
            if subprocess.run(["runuser", "-u", name, "--", "test", "-r", path]).returncode == 0:
                raise SystemExit(f"sandbox user {name} can read {path}")
    name, _ = sandbox_user(0, ctx.cfg)
    sandbox = image_check.Sandbox(name, ctx.paths.agent_runtime / "bin" / "python", ctx.harness / "sandbox_runner.py",
                                  ctx.sandbox_hide, size_kb(ctx.cfg["process_data_max"]))
    source = f"PATHS = {answers!r}\nWRITE_DIRS = {ctx.sandbox_hide!r}\n" + SELFTEST_SOURCE
    r = sandbox.run(source, [[[1, 2], [3, 4]]])
    if r["load_error"] or r["outputs"] != [[[[1, 2], [3, 4]], None]]:
        raise SystemExit(f"sandbox selftest failed: {r}")
    leaked = [str(f) for f in probe_files if f.exists()]
    if leaked:
        raise SystemExit(f"sandbox writes reached the host: {leaked}")
    log(f"sandbox selftest ok: {len(answers)} answer files unreadable, {len(ctx.sandbox_hide)} writable dirs hidden")


def prepare_run(args: argparse.Namespace, cfg: dict, episodes: list[Episode], deny: list[str],
                tito: dict | None, *, dp_size: int | None = None, deadline: float | None = None,
                scored: bool = True) -> tuple[Context, dict, str]:
    """Build the run directory (harness files, prompts), verify the codex version and the server, assemble Knobs, the
    posix fence and the watchdog; return (Context, server_info, codex version). When tito is not None, the server's
    ledger status is filled in here. Shared with run_arc2_seq.py."""
    if dp_size is not None and not (isinstance(dp_size, int) and dp_size > 0):
        raise SystemExit(f"dp_size must be a positive int, not {dp_size!r}")
    if deadline is not None and deadline <= time.time():
        raise SystemExit(f"deadline {deadline} is already past (now {time.time():.0f})")
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    for sub in ("tasks", "results", "driver_env", "proxy", "harness", "harness/pyshadow", "harness/prompts"):
        (run_dir / sub).mkdir()
    pdir = HERE / "prompts" / cfg["prompts"]
    prompts = {}
    names = PROMPT_FILES + (("midpoint_reminder.md",) if cfg["midpoint_reminder_tokens"] is not None else ()) \
        + (("wrapup.md",) if cfg.get("wrapup_tokens") is not None else ()) \
        + (("wrapup_time.md",) if cfg.get("wrapup_time_seconds") else ())
    if (pdir / "resume_fresh.md").exists():
        names += ("resume_fresh.md",)
    elif cfg["output_token_budget"] >= 120000:
        raise SystemExit(f"prompts {cfg['prompts']} has no resume_fresh.md, needed when the budget can outgrow the context")
    for name in names:
        text = (pdir / name).read_text()
        if not text.strip():
            raise SystemExit(f"{pdir / name} is empty")
        prompts[name] = text
        (run_dir / "harness" / "prompts" / name).write_text(text)
    shutil.copyfile(HERE / "check.py", run_dir / "harness" / "check.py")
    shutil.copyfile(HERE / "harvest_sandbox.py", run_dir / "harness" / "harvest_sandbox.py")
    (run_dir / "harness" / "pyshadow" / "sitecustomize.py").write_text("")
    shutil.copyfile(args.tokenizer_json, run_dir / "harness" / "tokenizer.json")
    shutil.copyfile(HERE / "netns_inner.sh", run_dir / "harness" / "netns_inner.sh")
    shutil.copyfile(BRIDGE, run_dir / "harness" / "bridge.py")
    shutil.copyfile(HERE / "sandbox_runner.py", run_dir / "harness" / "sandbox_runner.py")

    driver = arms.load("codex")
    # No resolve(): the venv's bin/python is a symlink to the system python; following it turns the prefix into /usr.
    paths = arms.Paths(agent_runtime=args.agent_python.absolute().parents[1], driver_bin=args.driver_bin.resolve(),
                       runtime=None)
    version = driver.installed_version(paths)
    if version != driver.VERSION:
        raise SystemExit(f"codex {version} != pinned {driver.VERSION}")
    info = get_json(server_root(args.model_base_url) + "/get_server_info")
    models = get_json(args.model_base_url + "/models")
    served = [m["id"] for m in models["data"]]
    if args.model not in served:
        raise SystemExit(f"model {args.model!r} not served (serving {served})")
    if cfg["model_context_window"] > info["context_length"]:
        raise SystemExit(f"model_context_window {cfg['model_context_window']} > server context {info['context_length']}")
    if dp_size is not None and info["dp_size"] != dp_size:
        raise SystemExit(f"--dp-size {dp_size} but the server says dp_size {info['dp_size']}")
    if tito is not None:
        tito["server"] = tito_module("preflight").check_server(server_root(args.model_base_url))
        log(f"tito: ledger instance {tito['server']['instance']} in {tito['server']['ledger_dir']}")
    driver_files = {"model_instructions_file": run_dir / "harness" / "prompts" / "instructions.md",
                    "experimental_compact_prompt_file": run_dir / "harness" / "prompts" / "compact_prompt.md",
                    "tool_output_tokenizer": run_dir / "harness" / "tokenizer.json"}
    driver_config = dict(cfg["driver_config"])
    if "developer_instructions" in driver_config:
        raise SystemExit("developer_instructions comes from prompts/<set>/developer.md, not driver_config")
    driver_config["developer_instructions"] = json.dumps(prompts["developer.md"])
    knobs = arms.Knobs(model=args.model, base_url=args.model_base_url, effort=cfg["effort"],
                       context_window=cfg["model_context_window"], auto_compact=cfg["auto_compact_token_limit"],
                       server_context=info["context_length"],
                       driver_config=tuple(sorted(driver_config.items())),
                       driver_files=tuple((k, str(v)) for k, v in sorted(driver_files.items())))

    watchdog = None
    if cfg["fence"] == "posix":
        for ancestor in [run_dir, *run_dir.parents]:
            if not ancestor.stat().st_mode & 0o001:
                raise SystemExit(f"{ancestor} is not traversable by other users")
        run_dir.chmod(0o711)
        (run_dir / "tasks").chmod(0o711)
        for private in ("results", "driver_env"):
            (run_dir / private).chmod(0o700)
        (run_dir / "proxy").chmod(0o711)     # the sockets live here: task uids can traverse it but not list it
        for p in [run_dir / "harness", *(run_dir / "harness").rglob("*")]:
            p.chmod(0o755 if p.is_dir() else 0o644)
        # When two runs go at once with overlapping uid ranges, one task's kill_user at its end kills the other run's
        # codex with the same uid (10-04 v1b / v2: this interrupted all 13 v1b tasks). Before starting, this uid range
        # must have no processes at all. Group mode: one uid per sample (N copies of one task do not share); several
        # group runs at once need their own uid_base each, otherwise this refuses to run.
        if cfg["view"] == "image" and len(episodes) > SANDBOX_UID_OFFSET:
            raise SystemExit(f"more than {SANDBOX_UID_OFFSET} episodes would overlap the sandbox uids")
        span = {task_user(ep.index, cfg)[1] for ep in episodes}
        if cfg["view"] == "image":
            span |= {sandbox_user(ep.index, cfg)[1] for ep in episodes}
        busy = sorted({uid for _, uid, *_ in posix_fence._procs() if uid in span})
        if busy:
            raise SystemExit(f"uids {busy[:5]}... already have processes: another run uses this uid_base")
        for ep in episodes:
            ensure_user(*task_user(ep.index, cfg))
            if cfg["view"] == "image":
                ensure_user(*sandbox_user(ep.index, cfg))
        watchdog = posix_fence.Watchdog(run_dir / "watchdog.jsonl", game_max_kb=size_kb(cfg["game_memory_max"]),
                                        headroom_kb=size_kb(cfg["watchdog_headroom"]),
                                        cpu_seconds=cfg["watchdog_cpu_seconds"])
        watchdog.tick()
        watchdog.start()
    sandbox_hide = [*image_check.WORLD_WRITABLE, str(run_dir / "tasks")] if cfg["fence"] == "posix" else []
    ctx = Context(cfg=cfg, args=args, run_dir=run_dir, driver=driver, knobs=knobs, paths=paths, prompts=prompts,
                  harness=run_dir / "harness", watchdog=watchdog, cancelled=threading.Event(), deny=deny,
                  require=[str(p) for p in driver_files.values()], sandbox_hide=sandbox_hide,
                  salt_prefix=args.salt_prefix, deadline=deadline,
                  dp_live=[0] * dp_size if dp_size is not None else None, scored=scored)
    return ctx, info, version


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--tasks-dir", type=Path, required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", help="comma-separated task ids")
    ap.add_argument("--agent-python", type=Path, required=True)
    ap.add_argument("--driver-bin", type=Path, required=True)
    ap.add_argument("--model-base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokenizer-json", type=Path, required=True)
    ap.add_argument("--deny", default="", help="colon-separated paths the task users must not read")
    ap.add_argument("--samples", type=int,
                    help="group mode: run the single --only task this many times at once (>= 2); episodes <task>__s<k>")
    ap.add_argument("--salt-prefix", help="config cache_salt episode only (required there): salt = <prefix>/<episode>")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    if "idle" in (cfg.get("budget_extension") or {}):
        raise SystemExit("budget_extension idle needs run_arc2_seq (a run of single solvers grants every extension)")
    group = args.samples is not None
    if group:
        if args.samples < 2:
            raise SystemExit(f"--samples must be >= 2, not {args.samples}")
        if not args.only or "," in args.only:
            raise SystemExit(f"--samples needs --only with exactly one task id, not {args.only!r}")
        if args.limit is not None:
            raise SystemExit("--limit does not go with --samples")
    if cache_salt_field(cfg) is not None and not args.salt_prefix:
        raise SystemExit("config cache_salt episode needs a non-empty --salt-prefix")
    if cache_salt_field(cfg) is None and args.salt_prefix is not None:
        raise SystemExit("--salt-prefix given but config cache_salt is not episode: the salt would never be sent")
    tito = {"config": tito_module("manifest").config_identity(args.config)} if "tito" in cfg else None
    if cfg["fence"] == "posix" and os.geteuid() != 0:
        raise SystemExit("fence posix needs root")
    deny = [p for p in args.deny.split(":") if p]
    if cfg["fence"] == "posix" and not deny:
        raise SystemExit("fence posix needs --deny (the directory holding the answers)")
    tasks = load_tasks(args.tasks_dir, args.limit, args.only.split(",") if args.only else None)
    if cfg["fence"] == "posix" and not any(args.tasks_dir.resolve().is_relative_to(Path(d).resolve()) for d in deny):
        raise SystemExit(f"--tasks-dir {args.tasks_dir} (it holds the test outputs) is not under any --deny path")
    episodes = make_episodes(list(tasks), args.samples)

    ctx, info, version = prepare_run(args, cfg, episodes, deny, tito)
    run_dir, prompts, watchdog = ctx.run_dir, ctx.prompts, ctx.watchdog
    if cfg["view"] == "image":
        sandbox_selftest(ctx, len(episodes))
    # New flags that were not given stay out of STATE.json: classic-mode STATE.json is the same as before the flags
    # were added.
    new_flags = {"samples", "salt_prefix"}
    state_file = {
        "config": cfg, "args": {k: str(v) for k, v in vars(args).items() if not (k in new_flags and v is None)},
        "codex_version": version,
        "prompt_sha256": {n: hashlib.sha256(t.encode()).hexdigest() for n, t in prompts.items()},
        "server_info": {k: info.get(k) for k in ("context_length", "max_running_requests", "served_model_name",
                                                 "default_sampling_params", "preferred_sampling_params",
                                                 "sampling_defaults", "model_path", "version")},
        "tasks": list(tasks), "started": time.strftime("%FT%T%z"),
        **({"tito": tito} if tito is not None else {}),
        **({"episodes": [ep.name for ep in episodes]} if group else {}),
    }
    atomic_write(run_dir / "STATE.json", json.dumps(state_file, indent=1) + "\n")
    solutions = {tid: [t["output"] for t in task["test"]] for tid, task in tasks.items()}
    (run_dir / "results" / "solutions.json").write_text(json.dumps(solutions))
    # Group mode: one thread per sample; the config's concurrency is not used.
    workers = args.samples if group else cfg["concurrency"]
    if group:
        log(f"group: {episodes[0].task} x {args.samples} samples at once (config concurrency ignored), "
            f"budget {cfg['output_token_budget']:,} output tokens, "
            f"window {cfg['model_context_window']}/{cfg['auto_compact_token_limit']}, fence {cfg['fence']}")
    else:
        log(f"{len(tasks)} tasks, concurrency {cfg['concurrency']}, budget {cfg['output_token_budget']:,} output tokens, "
            f"window {cfg['model_context_window']}/{cfg['auto_compact_token_limit']}, fence {cfg['fence']}")
    if ctx.salt_prefix is not None:
        log(f"cache salt: {cache_salt_field(cfg)} = {ctx.salt_prefix}/<episode> in every /responses body")

    # SIGTERM / SIGINT: each session sees cancelled at its next poll (budget_poll_seconds), kills its own task, harvests
    # and writes the result. If the harness is hard-killed (SIGKILL), codex processes are left behind -- in posix mode
    # they are all within the uid_base range.
    def on_signal(signum, frame):
        log(f"signal {signum}: cancelling; running sessions stop at their next poll")
        ctx.cancelled.set()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    started = time.time()
    state = {"done": [], "running": set(), "total": len(episodes)}
    failures: list[dict] = []
    stop_hb = threading.Event()
    threading.Thread(target=heartbeat, args=(stop_hb, ctx, state), daemon=True).start()

    def one(ep: Episode) -> dict:
        if ctx.cancelled.is_set():
            raise RuntimeError("cancelled before start")
        with LOCK:
            state["running"].add(ep.name)
        try:
            # single solver (no deadline, no kinds): if the config has budget_extension, always grant
            gate = ((lambda total: {"granted": True, "reason": "run_arc2: always"})
                    if cfg.get("budget_extension") is not None else None)
            return solve(ctx, ep, tasks[ep.task], extend_gate=gate)
        finally:
            with LOCK:
                state["running"].discard(ep.name)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(one, ep): ep for ep in episodes}
        for future in concurrent.futures.as_completed(futures):
            ep = futures[future]
            label = ep.name
            error = future.exception()
            if error is not None:
                row = {**item_fields(ep), "error": repr(error),
                       "traceback": "".join(traceback.format_exception(error))}
                failures.append(row)
                with (run_dir / "failures.jsonl").open("a") as f:
                    f.write(json.dumps(row) + "\n")
                log(f"{label}: FAILED {error!r}")
                if len(failures) > cfg["max_failures"] or "fence refused" in repr(error):
                    ctx.cancelled.set()
                    for other in futures:
                        other.cancel()
                continue
            result = future.result()
            # Atomic write only after the episode is done (codex finished, answers harvested): the orchestrator takes
            # its existence as the completion signal.
            atomic_write(episode_resources(run_dir, cfg, ep, ctx.salt_prefix)["result"], json.dumps(result) + "\n")
            with LOCK:
                state["done"].append(result)
            log(f"{label}: {result['stop']} score {result['score']:.2f} train "
                f"{sum(v == 'PASS' for v in result['train'])}/{len(result['train'])} "
                f"out {result['usage']['proxy_output_tokens']:,} sessions {len(result['sessions'])} "
                f"{result['seconds']:.0f}s [{len(state['done'])}/{len(episodes)}]")
    stop_hb.set()
    if watchdog is not None:
        watchdog.stop()
        atomic_write(run_dir / "watchdog_summary.json", json.dumps(watchdog.summary(), indent=1) + "\n")
    if group:
        summary = write_group_summary(ctx, episodes[0].task, args.samples, state["done"], failures, started)
    else:
        summary = write_summary(ctx, tasks, state["done"], failures, started)
    log("SUMMARY " + json.dumps(summary))
    if tito is not None:
        # Read the status again: a server restart midway (instance changed) is only recorded, not raised; unreadable
        # raises.
        tito["server"] = tito_module("preflight").check_server_end(tito["server"])
        manifest = tito_module("manifest")
        written = manifest.write(run_dir / manifest.FILE, harness="arc2", config=tito["config"],
                                 server=tito["server"],
                                 items=tito_items(run_dir, state["done"], failures))
        log(f"tito: {manifest.FILE} {written['counts']} instances {written['server']['instances']}")
    if failures:
        raise SystemExit(f"{len(failures)} {'episode' if group else 'task'}(s) failed; see {run_dir / 'failures.jsonl'}")


if __name__ == "__main__":
    main()
