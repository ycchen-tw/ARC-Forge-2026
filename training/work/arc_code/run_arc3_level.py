#!/usr/bin/env python3
"""Harness for ARC-AGI-3 single-level RL episodes. To the orchestrator (work/qwen38fn_rl) it has the same contract as run_arc2.py.

One episode = level K of one game: play from that level's start point until it is cleared or a stop condition comes
first (src/arcforge/arc_code/level_episode.py). Each episode is an independent ``run_arm.py --level-episode``: it starts
its own official gateway (``--start-level`` set to level K), middleware (writes stop files on level clear / action cap)
and jail.sh (no network, sees only the workspace) -- the environment the agent sees is the Kaggle v14 arm itself, plus
one extra passage "start from level K, the episode ends when it is cleared" (prompts/<set>/level.md). Outside it, this
adds:

- **proxy** (arc2/model_proxy.TaskProxy): output budget exact to the token (each request's cap = remaining budget),
  cache salt, route allowlist + codex's session_id required, 503 after a stop file. run_arm's model bridge connects to it
  (``--agent-model-url``). Writes stop file ``budget`` when the budget runs out.
- **reward**: middleware's stop file ``cleared`` (stop files and ledger are on the host side, out of the agent's reach)
  → 1, else 0. A level clear never exceeds the action cap: at the cap, middleware rejects first and writes
  ``action_cap``.
- **results**: ``results/<tid>__s<k>.json`` (written atomically as soon as each episode ends; the orchestrator counts
  slots by it) and the group's ``tito_manifest.json`` (harness arc3, outcome carries score) -- these two are what the
  orchestrator and prepare_rl read. When the group ends (success or not), ``EXIT.json`` {"rc", "finished_unix"} is
  written atomically last: the orchestrator uses it to judge completion when it takes over an in-flight group.

    <agent python> work/arc_code/run_arc3_level.py --config <harness config> --run-dir <group dir> \\
        --tasks-dir <stub dir> --only <tid> --samples 16 --salt-prefix <run>/<session>/<gid> \\
        [--sample-backends <u1>,<u2>,...] --agent-python <agent runtime python> --driver-bin <codex prefix> \\
        --model-base-url http://127.0.0.1:<port>/v1 --model <served name> --tokenizer-json <tokenizer.json>

    <agent python> work/arc_code/run_arc3_level.py --selftest --config ... --run-dir <new dir> \\
        --agent-python ... --driver-bin ...          # anti-cheat positive/negative controls, run before every launch (see selftest())

stub (``<tasks-dir>/<tid>.json``, made by work/arc_code/arc3_levels.py): game, game_id, level, win_levels,
baseline_actions, action_cap, start_hash, game_sha (sha256 of the game directory); optional ``output_token_budget`` =
this level's quota (differs per level so that clear rates land mid-range; ``arc3_levels.py quota``), not given = the
harness config's ``output_token_budget``. The config value is a ceiling: any larger and the context runs into
compaction. The agent can't see the quota (the proxy only truncates at the line), so the 64k prescreen's "tokens at the
moment of clearing" convert directly into the clear rate for any smaller quota.

Prior-context seeds (round 2; stubs made by ``arc3_levels.py seed`` have seven extra ``seed*`` fields, all or none): before
the episode starts, run_arm deep-copies the seed (the real whole-game workspace at the start of level K,
``src/arcforge/arc_code/level_seed.py``) into the workspace, middleware continues recording from the seed record's last
step, and the opening is the handoff message a new session gets when taking over mid-game (prompts ``v14seed``).

Whole-game episodes (harness config ``kind: arc3_game``, RL v4): one episode = one game played from start to end (all
levels cleared, output budget used up, codex dead, or wall), compacting along the way per the arm config (each context
window is one segment of the ledger, prepare_rl ``--allow-compaction``). run_arm gets only ``--stop-dir`` (no
--level-episode). stub (``<tid>.json``, ``kind: game``): game, game_id, win_levels, baseline_actions (one per level),
game_sha, optional output_token_budget. reward = official whole-game score (``arcforge.arc_code.score.game_score``, the
same definition as whole-game eval) ÷ 100, computed from middleware's ledger.

gen-2 single-level episodes (harness config ``kind: arc3_g2seed``, from 2026-10-26): one episode = open level K from a
gen-2 level-change seed (the ``g2seed-20261021`` seed store, the real whole-game workspace before level K's first
action) and play it with the deployed gen-2 harness (g2b) until level clear, action cap, output budget or context cap.
run_arm gets ``--stop-dir``/``--proxy-log`` (the whole-game episode set) plus ``--game-seed <spec>
--episode-action-cap <cap>``: middleware's ``cleared``/``action_cap``/``start_failed`` and the proxy's
``budget``/``context`` are written to the same host-side directory. One sequence per game: the harness config's
``context_token_limit`` must be below g2's window-change point (the arm config's auto_compact_token_limit), so windows
never change within a game; or (RL v7, 2026-10-28) ``window_change: true`` with no context_token_limit: windows change
within the game as deployed, each window is one segment of the ledger (prepare_rl ``--allow-compaction``, as for
whole-game episodes), and the whole game shares one reward. stub (``kind: g2seed``, made by ``arc3_levels.py
g2seeds``): game, game_id, level, win_levels, baseline_actions (this level's), action_cap, game_sha, seed
(level_seed.SeedSpec's ws/sha256/prefix/actions/level/label), optional output_token_budget.
reward = 1[cleared] × min(1, (κ × baseline_actions / actions used on this level)²), κ = the harness config's
``efficiency_free_fraction`` (default 1): a clear is always within the output budget and action cap (middleware
rejects actions as soon as a stop file is written); the squared term only keeps actions from being traded for tokens.
Per-action charge (harness config ``action_token_charge`` C, optional ``action_charge_free_fraction`` f, g2seed only,
off by default): the proxy's remaining budget is further reduced by floor(C × max(0, a − f·b) / b), a = counted actions
in the ledger so far, b = baseline_actions (ActionCharge) -- actions and thinking draw on the same budget, and playing
the baseline number of actions costs C·(1 − f) tokens. reward is unchanged; a game whose budget runs out first because
of the charge ends as ``budget`` as usual, and the stop file and the result's ``action_charge`` record how much was
charged. The harness config's ``output_token_budget_override`` (g2seed) makes every game's budget the config's
output_token_budget (e.g. 64k), ignoring the stub's quota.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import fcntl
import fractions
import hashlib
import http.server
import importlib.util
import json
import math
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
import types
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[1]
SRC = ROOT / "src" / "arcforge"
sys.path.insert(0, str(UNIT / "arc2"))
import model_proxy  # noqa: E402  (work/arc_code/arc2)

SCHEMA_VERSION = 1
KIND = "arc3_level"
#: whole-game episode (RL v4): same harness, same proxy; the stub is one game (GAME_STUB_KEYS)
GAME_KIND = "arc3_game"
#: gen-2 single-level episode (2026-10-26): opens level K from a gen-2 level-change seed (G2_STUB_KEYS)
G2_KIND = "arc3_g2seed"
KINDS = (KIND, GAME_KIND, G2_KIND)
CONFIG_KEYS = {"schema_version", "kind", "fence", "tito", "concurrency", "arm_config", "arm_python",
               "gateway_python", "corpus", "prep", "ffmpeg", "model_context_window", "auto_compact_token_limit",
               "output_token_budget", "episode_wall_seconds", "note"}
#: ``cache_salt``: none (key absent) or episode (the proxy writes ``cache_salt = <salt-prefix>/<episode>`` on every
#: /responses).
#: ``context_token_limit`` (round 2): when one thread's context (input + output) reaches this, the episode ends (stop file
#: ``context``, an uncleared sample, trained on as usual). Must be ≤ codex's actual compaction threshold
#: min(auto_compact_token_limit, 0.9 × window): that way no episode ever compacts.
#: ``serve_context_length``: the rollout server's CTXLEN (launch_b200_stream.sh serve_ctxlen); not given =
#: model_context_window. Deployment (Kaggle v31) is window 150000, served 240000: codex plans and compacts by the window,
#: and the server's extra room keeps a single request (a large tool output, a long summary) from being rejected by
#: sglang at the window's edge.
#: ``efficiency_free_fraction`` (κ, g2seed only, default 1): R = 1[cleared] × min(1, (κ·baseline / actions)²); actions
#: within κ·baseline get full marks. v5 2026-10-26: whole-game act/base is about 1.45× single-level, and the official
#: per-level score caps below 0.933×baseline; κ 0.7 shrinks the single-level free zone to roughly the whole-game cap
#: point (with κ 1, a ≤ b always gives R 1, so drifting from 0.5 to 1.0 within a single level costs nothing).
#: ``action_token_charge`` (C, positive integer) / ``action_charge_free_fraction`` (f, [0, 1], default 0; must come with
#: C): g2seed only; each counted action takes C/b from the output budget, the first f·b actions are free (ActionCharge).
#: v5 2026-10-26: in training the 48k tokens are the binding resource, in deployment it is actions; tokens per action in
#: v5 fell from 4364 to ~950 and RESETs rose with it; the charge makes actions and thinking pay in the same currency.
#: ``output_token_budget_override`` (only true, g2seed only): every game's budget is this config's output_token_budget,
#: ignoring the stub's quota (all 350 g2seed-v5 stubs say 48000). v5 2026-10-26: κ 0.7 made it think longer per action and
#: hit the 48k wall more often, held-out v10 clears −9.5pp; games hitting 48k have a median context of 78k and reach
#: context_token_limit 120k only at ~81k output, so 64k fits.
#: ``seed_note_done`` (only true) / ``packet_level_docs`` (all/latest): g2seed only, passed as is into each game's arm
#: config (run_arm's seed_note_done_config / packet_level_docs_config); the arm config itself must not have them too.
#: 2026-10-27: seeds are snapshotted when note_done passes, yet after loading, level K−1 was waiting for note_done again
#: (400 tokens on average at game start, 1.2% rejected by act do first); the opening packet carries all level files of the
#: first K−1 levels, a median of 40k chars for L08 seeds. packet_level_docs latest is the setting shared by training, eval
#: and deployment (configure_b200's --packet-level-docs writes it into both harnesses).
#: ``window_change`` (only true, g2seed only, mutually exclusive with context_token_limit): RL v7, 300k output per game,
#: windows change as deployed; results and manifest outcome record ``windows``/``compactions``.
#: ``stop_after_first_window`` (only true, must come with window_change; RL v8, 2026-10-29): the game ends at the first
#: window change -- the proxy blocks the first window-1 request (TaskProxy's stop_after_first_window) and the harness
#: writes stop file ``window``. The ledger has only window 0; at the end, the same function as gen-2 SessionStart
#: (gen2/hooks/common.packet) assembles that moment's handoff message from the workspace and saves it as handoff.md in the
#: episode directory, for the note judge (docs/plans/arc3_rl_v8_window_note_judge_20261029.md §1); it also determines the
#: cause of the window change, ``window_cause`` (threshold/early: the model calling new_context itself before the change
#: point is not a valid handoff, R 0, not sent to the judge; window_cause()).
#: ``pace_line`` (only true, g2seed only): likewise passed as is into the arm config (run_arm's pace_line_config: the last
#: line of act's do/status is the output tokens used on this level); the proxy also writes the numbers into the PACE_FILE
#: of this game's workspace (pace_path).
ARM_PASS_KEYS = ("seed_note_done", "packet_level_docs", "pace_line")
OPTIONAL_CONFIG_KEYS = {"cache_salt", "context_token_limit", "serve_context_length", "efficiency_free_fraction",
                        "action_token_charge", "action_charge_free_fraction", "output_token_budget_override",
                        "window_change", "stop_after_first_window", *ARM_PASS_KEYS}
STUB_KEYS = {"schema_version", "game", "game_id", "level", "win_levels", "baseline_actions", "action_cap",
             "start_hash", "game_sha"}
#: prior-context seed (level_seed.py): stub fields, all or none. seed = the seed workspace, seed_sha256 = its tree_sha256,
#: seed_prefix = the replay workspace path hard-coded in the files, seed_actions = action count of the first K−1 levels;
#: label/family/prompts are provenance
SEED_KEYS = {"seed", "seed_sha256", "seed_label", "seed_family", "seed_actions", "seed_prefix", "seed_prompts"}
#: this level's output quota (not given = the harness config's output_token_budget); prior-context seed
OPTIONAL_STUB_KEYS = {"output_token_budget"} | SEED_KEYS
#: whole-game stub (kind game): baseline_actions has one per level (the official whole-game score's denominator needs
#: every level)
GAME_STUB_KEYS = {"schema_version", "kind", "game", "game_id", "win_levels", "baseline_actions", "game_sha"}
#: gen-2 single-level stub (kind g2seed): baseline_actions is this level's (one number); seed is level_seed.SeedSpec's json
G2_STUB_KEYS = {"schema_version", "kind", "game", "game_id", "level", "win_levels", "baseline_actions", "action_cap",
                "game_sha", "seed"}
G2_SEED_KEYS = {"ws", "sha256", "prefix", "actions", "level", "label"}
#: sglang's cache salt field (run_arc2.DEFAULT_CACHE_SALT_FIELD: the fork's radix tree recognises cache_salt)
CACHE_SALT_FIELD = "cache_salt"
#: still not reaped after run_arm ends = a harness bug: allow this long beyond wall
OVERRUN_SECONDS = 900
POLL_SECONDS = 1.0
#: after SIGTERM, run_arm reaps the session scope, scores, writes the manifest; still alive after this long = SIGKILL its
#: whole process tree (this game fails, the group goes on).
#: Must be less than the orchestrator's harness.terminate_grace_seconds (600): on cancel the harness still has time to
#: reap the other games and write EXIT.json
TERM_GRACE_SECONDS = 300
#: per-episode host ports (middleware, the official gateway's host-side bridge): below the ephemeral range (32768+), made
#: mutually exclusive across processes with flock (several group harnesses on the same machine allocate at once).
PORT_RANGE = range(20000, 32000)
PORT_LOCKS = Path(f"/tmp/arc3_level_ports.{os.getuid()}")
LOCK = threading.Lock()


#: stops on which the harness kills codex (the ledger may have one more final turn than codex): same as
#: arcforge.tito.join.RL_KILL_STOPS["arc3"] (checked by test_arc3_level_episode). The result's and manifest outcome's
#: ``harness_killed`` is whether the stop is in this set
KILL_STOPS = frozenset({"budget", "context", "cleared", "won", "action_cap", "wall", "died", "window"})
#: a group still counts as complete with at most this fraction of episodes failed (failed ones are dropped one by one);
#: more = systemic (e.g. 2026-10-24's NameError 32/32): the group exits 1, and the orchestrator quarantines it and
#: counts it toward the circuit breaker
MAX_FAILED_FRACTION = 0.25
#: {"rc": int, "finished_unix": float} written atomically when the group ends (main's finally; the orchestrator's
#: completion check on takeover)
EXIT_FILE = "EXIT.json"


def group_failed(n_failed: int, samples: int) -> bool:
    """Whether the group counts as failed: failed episodes exceed MAX_FAILED_FRACTION (a few broken games only drop
    those games; see the end of run_group)."""
    return n_failed > samples * MAX_FAILED_FRACTION


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def load_src(name: str, path: Path):
    """Load a stdlib module of src/arcforge by file path: the agent runtime doesn't have arcforge installed, and the
    .venv's editable install may point at another worktree (same reason as run_arc2.tito_module)."""
    key = f"arc3_level_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


episode_stops = load_src("level_episode", SRC / "arc_code" / "level_episode.py")


def arc_code_module(name: str):
    """Modules in src/arcforge/arc_code with relative imports (score.py ``from .ledger import``): mount a package that
    has only __path__, then import; same reason as load_src (the agent runtime doesn't have arcforge installed)."""
    package = "arc3_level_arc_code"
    if package not in sys.modules:
        module = types.ModuleType(package)
        module.__path__ = [str(SRC / "arc_code")]
        sys.modules[package] = module
    return importlib.import_module(f"{package}.{name}")
level_seed = load_src("level_seed", SRC / "arc_code" / "level_seed.py")


def tito_module(name: str):
    return load_src(f"tito_{name}", SRC / "tito" / f"{name}.py")


def expand(value: str) -> Path:
    """Paths in the config: ``$VAR`` must be set (raise if not, no default); relative paths are relative to the repo
    root."""
    text = os.path.expandvars(value)
    if "$" in text:
        raise SystemExit(f"config path {value!r}: unset environment variable (got {text!r}); source env/env.sh")
    path = Path(text)
    return path if path.is_absolute() else ROOT / path


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text())
    missing, unknown = CONFIG_KEYS - set(cfg), set(cfg) - CONFIG_KEYS - OPTIONAL_CONFIG_KEYS
    if missing or unknown:
        raise SystemExit(f"{path}: missing {sorted(missing)}, unknown {sorted(unknown)}")
    if cfg["schema_version"] != SCHEMA_VERSION or cfg["kind"] not in KINDS:
        raise SystemExit(f"{path}: schema_version {cfg['schema_version']} / kind {cfg['kind']!r}, "
                         f"want {SCHEMA_VERSION} / one of {KINDS}")
    if cfg["kind"] == GAME_KIND and "context_token_limit" in cfg:
        raise SystemExit(f"{path}: kind {GAME_KIND} compacts (each window is a segment): no context_token_limit")
    if "window_change" in cfg and (cfg["kind"] != G2_KIND or cfg["window_change"] is not True):
        raise SystemExit(f"{path}: window_change must be true and is for kind {G2_KIND} (not {cfg['kind']}, "
                         f"{cfg['window_change']!r})")
    if "stop_after_first_window" in cfg and (cfg["stop_after_first_window"] is not True or "window_change" not in cfg):
        raise SystemExit(f"{path}: stop_after_first_window must be true and goes with window_change (not "
                         f"{cfg['stop_after_first_window']!r}; window_change {cfg.get('window_change')!r})")
    if cfg["kind"] == G2_KIND and ("context_token_limit" in cfg) == ("window_change" in cfg):
        raise SystemExit(f"{path}: kind {G2_KIND} needs exactly one of context_token_limit (one window per episode, below "
                         "the g2 change point = the arm config's auto_compact_token_limit) and window_change true")
    if "efficiency_free_fraction" in cfg:
        kappa = cfg["efficiency_free_fraction"]
        if cfg["kind"] != G2_KIND:
            raise SystemExit(f"{path}: efficiency_free_fraction is for kind {G2_KIND}, not {cfg['kind']}")
        if not (isinstance(kappa, (int, float)) and not isinstance(kappa, bool) and math.isfinite(kappa) and 0 < kappa <= 1):
            raise SystemExit(f"{path}: efficiency_free_fraction {kappa!r} must be a number in (0, 1]")
    if "action_token_charge" in cfg or "action_charge_free_fraction" in cfg:
        if cfg["kind"] != G2_KIND:
            raise SystemExit(f"{path}: action_token_charge / action_charge_free_fraction are for kind {G2_KIND}, "
                             f"not {cfg['kind']}")
        charge = cfg.get("action_token_charge")
        if not (isinstance(charge, int) and not isinstance(charge, bool) and charge > 0):
            raise SystemExit(f"{path}: action_token_charge {charge!r} must be a positive int "
                             "(action_charge_free_fraction goes with it)")
        free = cfg.get("action_charge_free_fraction", 0)
        if not (isinstance(free, (int, float)) and not isinstance(free, bool) and math.isfinite(free) and 0 <= free <= 1):
            raise SystemExit(f"{path}: action_charge_free_fraction {free!r} must be a number in [0, 1]")
    if "output_token_budget_override" in cfg:
        if cfg["kind"] != G2_KIND:
            raise SystemExit(f"{path}: output_token_budget_override is for kind {G2_KIND}, not {cfg['kind']}")
        if cfg["output_token_budget_override"] is not True:
            raise SystemExit(f"{path}: output_token_budget_override must be true (leave it out to use the stubs' quotas), "
                             f"not {cfg['output_token_budget_override']!r}")
    for key in ARM_PASS_KEYS:
        if key in cfg and cfg["kind"] != G2_KIND:
            raise SystemExit(f"{path}: {key} is for kind {G2_KIND}, not {cfg['kind']}")
    if "seed_note_done" in cfg and cfg["seed_note_done"] is not True:
        raise SystemExit(f"{path}: seed_note_done must be true (leave it out to keep the gate), not {cfg['seed_note_done']!r}")
    if "packet_level_docs" in cfg and cfg["packet_level_docs"] not in ("all", "latest"):
        raise SystemExit(f"{path}: packet_level_docs must be all or latest, not {cfg['packet_level_docs']!r}")
    if "pace_line" in cfg and cfg["pace_line"] is not True:
        raise SystemExit(f"{path}: pace_line must be true (leave it out for no pace line), not {cfg['pace_line']!r}")
    if cfg["fence"] != "bwrap":
        raise SystemExit(f"{path}: fence must be bwrap (jail.sh: no network, only the workspace)")
    tito_module("preflight").check_config(cfg["tito"])
    if cfg.get("cache_salt", "none") not in ("none", "episode"):
        raise SystemExit(f"{path}: cache_salt must be none or episode, not {cfg['cache_salt']!r}")
    for key in ("concurrency", "output_token_budget", "episode_wall_seconds"):
        if not (isinstance(cfg[key], int) and cfg[key] > 0):
            raise SystemExit(f"{path}: {key} must be a positive int, not {cfg[key]!r}")
    if not 0 < cfg["auto_compact_token_limit"] < cfg["model_context_window"]:
        raise SystemExit(f"{path}: need 0 < auto_compact_token_limit < model_context_window")
    if "serve_context_length" in cfg:
        served = cfg["serve_context_length"]
        if not (isinstance(served, int) and not isinstance(served, bool) and served >= cfg["model_context_window"]):
            raise SystemExit(f"{path}: serve_context_length {served!r} must be an int >= model_context_window "
                             f"{cfg['model_context_window']}")
    if "context_token_limit" in cfg:
        limit = cfg["context_token_limit"]
        # codex compacts at min(limit, 0.9 x window) (codex-rs protocol/src/openai_models.rs auto_compact_token_limit)
        compacts_at = min(cfg["auto_compact_token_limit"], cfg["model_context_window"] * 9 // 10)
        if not (isinstance(limit, int) and not isinstance(limit, bool) and 0 < limit <= compacts_at):
            raise SystemExit(f"{path}: context_token_limit {limit!r} must be a positive int <= codex's compaction point "
                             f"{compacts_at} (min(auto_compact_token_limit, 0.9 x model_context_window))")
    arm = json.loads(expand(cfg["arm_config"]).read_text())
    if (cfg["kind"] == G2_KIND) != (arm.get("harness") == "gen2"):
        raise SystemExit(f"{path}: kind {cfg['kind']} with arm harness {arm.get('harness')!r}: {G2_KIND} plays the gen-2 "
                         "harness (--game-seed is a gen-2 feature), the other kinds gen-1")
    # run_arm reads these two from the arm config, launch_b200_stream.sh from this one (the server's CTXLEN): they must match
    for key in ("model_context_window", "auto_compact_token_limit"):
        if arm[key] != cfg[key]:
            raise SystemExit(f"{path}: {key} {cfg[key]} != arm config {cfg['arm_config']} {arm[key]}")
    if arm["fence"] != "bwrap" or arm.get("tito") != cfg["tito"]:
        raise SystemExit(f"{cfg['arm_config']}: needs fence bwrap and the same tito as {path}")
    for key in ("game_output_tokens", "tail_resume", "level_episode", "game_episode"):
        if arm.get(key):
            raise SystemExit(f"{cfg['arm_config']}: {key} must not be set for RL episodes "
                             "(the proxy owns the output budget; one episode, no resume pool)")
    clash = [key for key in ARM_PASS_KEYS if key in arm]
    if clash:
        raise SystemExit(f"{cfg['arm_config']}: {clash} come from the harness config ({path}), not the arm config")
    tito_module("preflight").check_codex_config(arm["driver_config"])
    cfg["_arm"] = arm
    return cfg


def load_stub(tasks_dir: Path, tid: str, kind: str = KIND) -> dict:
    path = tasks_dir / f"{tid}.json"
    stub = json.loads(path.read_text())
    if kind == GAME_KIND:
        return check_game_stub(path, stub)
    if kind == G2_KIND:
        return check_g2_stub(path, stub)
    if not STUB_KEYS <= set(stub) <= STUB_KEYS | OPTIONAL_STUB_KEYS:
        raise SystemExit(f"{path}: keys {sorted(stub)}; want {sorted(STUB_KEYS)} (+ optional {sorted(OPTIONAL_STUB_KEYS)})")
    if stub["schema_version"] != SCHEMA_VERSION:
        raise SystemExit(f"{path}: schema_version {stub['schema_version']} != {SCHEMA_VERSION}")
    if not 1 <= stub["level"] <= stub["win_levels"] or stub["action_cap"] < 1:
        raise SystemExit(f"{path}: level {stub['level']} / win_levels {stub['win_levels']} / "
                         f"action_cap {stub['action_cap']}")
    budget = stub.get("output_token_budget")
    if budget is not None and not (isinstance(budget, int) and not isinstance(budget, bool) and budget > 0):
        raise SystemExit(f"{path}: output_token_budget must be a positive int, not {budget!r}")
    seed = SEED_KEYS & set(stub)
    if seed and seed != SEED_KEYS:
        raise SystemExit(f"{path}: seed fields {sorted(seed)}; a seeded stub has all of {sorted(SEED_KEYS)}")
    if seed:
        seed_spec(stub)                              # level >= 2, actions, prefix, sha256 shape (level_seed.SeedSpec)
    return stub


def check_game_stub(path: Path, stub: dict) -> dict:
    """Whole-game stub (harness kind arc3_game): GAME_STUB_KEYS + optional output_token_budget; one positive-int
    baseline per level."""
    if not GAME_STUB_KEYS <= set(stub) <= GAME_STUB_KEYS | {"output_token_budget"}:
        raise SystemExit(f"{path}: keys {sorted(stub)}; a game stub has {sorted(GAME_STUB_KEYS)} (+ output_token_budget)")
    if stub["schema_version"] != SCHEMA_VERSION or stub["kind"] != "game":
        raise SystemExit(f"{path}: schema_version {stub['schema_version']} / kind {stub['kind']!r}, want {SCHEMA_VERSION} / 'game'")
    base = stub["baseline_actions"]
    if not (isinstance(base, list) and len(base) == stub["win_levels"] >= 1
            and all(isinstance(b, int) and not isinstance(b, bool) and b > 0 for b in base)):
        raise SystemExit(f"{path}: baseline_actions {base!r} must be {stub['win_levels']} positive ints (one per level)")
    budget = stub.get("output_token_budget")
    if budget is not None and not (isinstance(budget, int) and not isinstance(budget, bool) and budget > 0):
        raise SystemExit(f"{path}: output_token_budget must be a positive int, not {budget!r}")
    return stub


def check_g2_stub(path: Path, stub: dict) -> dict:
    """gen-2 single-level stub (harness kind arc3_g2seed): G2_STUB_KEYS + optional output_token_budget; the seed is a
    level-change seed for level K."""
    if not G2_STUB_KEYS <= set(stub) <= G2_STUB_KEYS | {"output_token_budget"}:
        raise SystemExit(f"{path}: keys {sorted(stub)}; a g2seed stub has {sorted(G2_STUB_KEYS)} (+ output_token_budget)")
    if stub["schema_version"] != SCHEMA_VERSION or stub["kind"] != "g2seed":
        raise SystemExit(f"{path}: schema_version {stub['schema_version']} / kind {stub['kind']!r}, want {SCHEMA_VERSION} / 'g2seed'")
    if not 2 <= stub["level"] <= stub["win_levels"]:
        raise SystemExit(f"{path}: level {stub['level']} / win_levels {stub['win_levels']} (a level seed starts at L2 or later)")
    for key in ("baseline_actions", "action_cap"):
        value = stub[key]
        if not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
            raise SystemExit(f"{path}: {key} must be a positive int, not {value!r}")
    seed = stub["seed"]
    if not (isinstance(seed, dict) and set(seed) == G2_SEED_KEYS) or seed["level"] != stub["level"]:
        raise SystemExit(f"{path}: seed {seed!r} must be a SeedSpec json ({sorted(G2_SEED_KEYS)}) of level {stub['level']}")
    g2_seed_spec(stub)                                # actions, prefix, sha256 shape (level_seed.SeedSpec)
    budget = stub.get("output_token_budget")
    if budget is not None and not (isinstance(budget, int) and not isinstance(budget, bool) and budget > 0):
        raise SystemExit(f"{path}: output_token_budget must be a positive int, not {budget!r}")
    return stub


def g2_seed_spec(stub: dict):
    """The gen-2 stub's seed → the SeedSpec that run_arm ``--game-seed`` reads."""
    return level_seed.SeedSpec.from_json(stub["seed"])


def is_game(stub: dict) -> bool:
    return stub.get("kind") == "game"


def is_g2(stub: dict) -> bool:
    return stub.get("kind") == "g2seed"


def episode_label(stub: dict) -> str:
    if is_game(stub):
        return stub["game"]
    if is_g2(stub):
        return f"{stub['game']} L{stub['level']} {stub['seed']['label']}"
    return f"{stub['game']} L{stub['level']}"


def seed_spec(stub: dict):
    """The stub's prior-context seed → the SeedSpec that run_arm ``--level-seed`` reads (None without a seed)."""
    if "seed" not in stub:
        return None
    return level_seed.SeedSpec(ws=Path(stub["seed"]), sha256=stub["seed_sha256"], prefix=stub["seed_prefix"],
                               actions=stub["seed_actions"], level=stub["level"], label=stub["seed_label"])


#: the file act's pace line reads (PACE_FILE in src/arcforge/arc_code/act_client.py; not imported here, a test checks
#: the two copies)
PACE_FILE = ".pace.json"


def pace_path(cfg: dict, stub: dict, ep_dir: Path) -> Path | None:
    """Where the proxy writes the pace file when the harness config has pace_line: this game's workspace
    (``games/<game>`` under run_arm ``--run-dir ep_dir/arm``, $HOME inside the jail); otherwise None (no writes)."""
    if not cfg.get("pace_line"):
        return None
    return g2_workspace(stub, ep_dir) / PACE_FILE


def g2_workspace(stub: dict, ep_dir: Path) -> Path:
    """This game's workspace: ``games/<game>`` under run_arm ``--run-dir ep_dir/arm`` ($HOME inside the jail)."""
    return ep_dir / "arm" / "games" / stub["game"]


#: stop_after_first_window: the runner's gen2_budget.json and the hook log saved when the proxy blocks window 1 (the two
#: copies in driver home are deleted when the session is reaped), plus the handoff message assembled at the end, all in
#: the episode directory
G2_BUDGET_COPY = "gen2_budget.json"
G2_HOOK_LOG_COPY = "gen2_hooks.jsonl"
HANDOFF_FILE = "handoff.md"
#: start of the status packet gen-2 SessionStart gives a new window (source compact) (gen2/hooks/common.packet)
HANDOFF_HEADER = "# New context window\n"


def gen2_hooks():
    """The gen-2 hooks' shared module (work/arc_code/gen2/hooks/common.py: status/packet with which SessionStart
    assembles the status packet, and the functions that read history). On import it replaces sys.excepthook with a
    version that writes the hook log ($CODEX_HOME) -- meant for hook subprocesses; the harness is not a hook, so the
    original is put back."""
    hook = sys.excepthook
    module = load_src("gen2_hooks_common", UNIT / "gen2" / "hooks" / "common.py")
    sys.excepthook = hook
    return module


def compact_session_start(ep_dir: Path) -> dict:
    """The SessionStart (source compact) at the start of window 1 in the hook log saved when the proxy blocked window 1
    (G2_HOOK_LOG_COPY); raise unless there is exactly one (neither the window-change cause nor the handoff can be judged)."""
    events = [json.loads(line) for line in (ep_dir / G2_HOOK_LOG_COPY).read_text().splitlines()]
    starts = [e for e in events if e["event"] == "SessionStart" and e["source"] == "compact"]
    if len(starts) != 1:
        raise RuntimeError(f"{ep_dir / G2_HOOK_LOG_COPY}: {len(starts)} SessionStart(compact) events, want exactly the "
                           "window-1 one: cannot tell why the window changed or what it handed over")
    return starts[0]


def window_cause(ep_dir: Path, proxy_log: Path) -> tuple[str, dict]:
    """Why a game with stop ``window`` changed window (the user, 2026-10-29: only a window change at the change point is
    a valid handoff): ``threshold`` = context reached the hook's change point (``limit`` in gen2_budget.json; the model
    called new_context after seeing "window full", or was forced to change by compact_request/codex), ``early`` = the
    model called new_context on its own before that.

    Two pieces of evidence; if either says the point was reached, it is threshold:
    - ``context_full_cleared`` recorded by the SessionStart at the start of window 1 (source compact, exactly one in the
      hook log): it removed the ``.context_full`` that PostToolUse creates when context ≥ limit. This is the hook's own
      judgement, but PostToolUse only runs after tool calls and judges by the previous request's context, so when the
      request that pushed context past limit itself calls new_context, the flag isn't created in time;
    - the maximum context at the end of window-0 responses in the proxy log (usage input + output, the same number the
      hook measures) ≥ limit.
    If the hook log doesn't have exactly one compact SessionStart, or that entry lacks context_full_cleared: undecidable,
    raise. Returns (cause, evidence)."""
    budget = json.loads((ep_dir / G2_BUDGET_COPY).read_text())
    limit = budget["limit"]
    gate = compact_session_start(ep_dir)["context_full_cleared"]
    if not isinstance(gate, bool):
        raise RuntimeError(f"{ep_dir / G2_HOOK_LOG_COPY}: SessionStart context_full_cleared {gate!r} is not a bool")
    peak = None
    for line in proxy_log.read_text().splitlines():
        row = json.loads(line)
        usage = row.get("usage") or {}                  # forbidden / client_disconnected rows have no usage
        sid = row.get("session_id")
        if sid is None or "input_tokens" not in usage or sid.rpartition(":")[2] != "0":
            continue
        end = int(usage["input_tokens"]) + int(usage.get("output_tokens") or 0)
        peak = end if peak is None else max(peak, end)
    reached = gate or (peak is not None and peak >= limit)
    return ("threshold" if reached else "early"), {"limit": limit, "context_full_cleared": gate,
                                                   "window0_context_peak": peak}


def window_handoff(ctx: "Context", ep_dir: Path) -> dict:
    """For a game with stop ``window`` (stop_after_first_window): rebuild the status packet that the SessionStart
    (source compact) at the start of window 1 handed the model, with the same functions as the hook (common.status +
    common.packet), from the workspace at the end -- not a single window-1 request was forwarded, so the workspace hasn't
    changed since the hook ran. Settings such as tail are read from the gen2_budget.json saved when the proxy blocked
    window 1. Written as ``handoff.md``; returns {path (relative to the group directory), sha256, chars}.

    When the settings have a stuck review (``stuck``), the hook picks tail or review for the last part by reading its own
    log (in driver home, already deleted): such a prompt set can't be rebuilt here, raise. ``.context_full`` still there
    = window 1's SessionStart never ran (it removes it first thing): raise too.
    The rebuilt length and notes.md's current length must equal the ``chars``/``notes_chars`` logged by the hook log's
    SessionStart(compact) entry (otherwise the workspace changed after the hook, or the rebuild differs from what the
    hook gave): raise if they differ."""
    hooks = gen2_hooks()
    budget = json.loads((ep_dir / G2_BUDGET_COPY).read_text())
    if budget["stuck"] is not None:
        raise RuntimeError(f"{ep_dir}: the prompt set has a stuck review (stuck.md): the SessionStart hook picks it or the "
                           "tail from its own log, which went with the driver home -- the handoff cannot be rebuilt")
    ws = g2_workspace(ctx.stub, ep_dir)
    st = hooks.status(ws, hooks.history(ws))
    if st["context_full"]:
        raise RuntimeError(f"{ws}: {hooks.CONTEXT_FULL} is still there at the window cut: the window-1 SessionStart hook "
                           "(which removes it first) never ran")
    # packet_level_docs: run_arm writes it into gen2_budget.json only when set; without the key the hook treats it as all
    # (the same line as session_start.py)
    text = hooks.packet(ws, st, "compact", budget["tail"], budget["packet_levels"], budget.get("packet_level_docs", "all"))
    if not text.startswith(HANDOFF_HEADER):
        raise RuntimeError(f"{ep_dir}: the rebuilt handoff starts {text[:60]!r}, not the compact header {HANDOFF_HEADER!r}")
    logged = compact_session_start(ep_dir)
    notes = ws / "notes.md"
    notes_chars = len(notes.read_text(encoding="utf-8", errors="replace")) if notes.exists() else None
    if (len(text), notes_chars) != (logged["chars"], logged["notes_chars"]):
        raise RuntimeError(f"{ep_dir}: the rebuilt handoff has {len(text)} chars and notes.md {notes_chars}, the window-1 "
                           f"SessionStart hook logged {logged['chars']} and {logged['notes_chars']}: the workspace changed "
                           "after the hook or the rebuild differs from what the model was given")
    path = ep_dir / HANDOFF_FILE
    atomic_write(path, text)
    return {"path": str(path.relative_to(ctx.run_dir)), "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "chars": len(text)}


def episode_budget(cfg: dict, stub: dict) -> int:
    """This level's output quota: the stub's if it has one, never above the harness config's ceiling (the context would
    hit compaction). With output_token_budget_override in the config, always the config's value (stub quota ignored)."""
    if cfg.get("output_token_budget_override"):
        return cfg["output_token_budget"]
    budget = stub.get("output_token_budget", cfg["output_token_budget"])
    if budget > cfg["output_token_budget"]:
        raise SystemExit(f"{episode_label(stub)}: output_token_budget {budget} > the harness ceiling "
                         f"{cfg['output_token_budget']}")
    return budget


# --------------------------------------------------------------------------- ports

class PortLease:
    """``n`` host ports that other harness processes can't take while held (one flock file per port). They only count
    if they bind; run_arm has bound them before the release."""

    def __init__(self, n: int) -> None:
        PORT_LOCKS.mkdir(mode=0o700, exist_ok=True)
        self.ports: list[int] = []
        self.fds: list[int] = []
        candidates = list(PORT_RANGE)
        random.shuffle(candidates)
        for port in candidates:
            if len(self.ports) == n:
                break
            fd = os.open(PORT_LOCKS / f"{port}.lock", os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
                continue
            probe = socket.socket()
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                os.close(fd)
                continue
            finally:
                probe.close()
            self.ports.append(port)
            self.fds.append(fd)
        if len(self.ports) != n:
            self.release()
            raise RuntimeError(f"could not lease {n} ports in {PORT_RANGE}")

    def release(self) -> None:
        for fd in self.fds:
            os.close(fd)
        self.fds = []


# --------------------------------------------------------------------------- model server

def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def server_root(base_url: str) -> str:
    if not base_url.endswith("/v1"):
        raise SystemExit(f"model url must end in /v1: {base_url}")
    return base_url[:-3]


def host_port(base_url: str) -> tuple[str, int]:
    url = urllib.parse.urlsplit(base_url)
    if url.hostname != "127.0.0.1" or url.port is None:
        raise SystemExit(f"model url must be http://127.0.0.1:<port>/v1, not {base_url}")
    return url.hostname, url.port


def check_backends(backends: list[str], model: str, context_window: int) -> int:
    """Every replica serves the same model with a long-enough context; returns the server's context length.

    Timeout 120 s: with 96 sessions at full load the replica frontend (event loop) takes 0.7 s to answer even /v1/models,
    so a group of 16 episodes starting at once queues for over 10 s (on 2026-10-19 an RL A/B group exited 1 because of
    this; stream RL hitting it gets a whole-round OrchestratorError)."""
    infos = []
    for backend in backends:
        info = get_json(server_root(backend) + "/get_server_info", timeout=120)
        served = [m["id"] for m in get_json(backend + "/models", timeout=120)["data"]]
        if model not in served:
            raise SystemExit(f"{backend}: model {model!r} not served (serving {served})")
        if context_window > info["context_length"]:
            raise SystemExit(f"{backend}: model_context_window {context_window} > server context {info['context_length']}")
        infos.append(info)
    for backend, info in zip(backends[1:], infos[1:]):
        for key in ("model_path", "context_length"):
            if info[key] != infos[0][key]:
                raise SystemExit(f"{backend}: {key} {info[key]!r} != {infos[0][key]!r} at {backends[0]}")
    return infos[0]["context_length"]


# --------------------------------------------------------------------------- episodes

@dataclasses.dataclass(frozen=True)
class Episode:
    tid: str
    sample: int

    @property
    def name(self) -> str:
        return f"{self.tid}__s{self.sample}"


@dataclasses.dataclass
class Context:
    args: argparse.Namespace
    cfg: dict
    stub: dict
    run_dir: Path
    backends: list[str]
    server_context: int
    cancelled: threading.Event


def ledger_resets(ledger: Path, game_id: str) -> int:
    """Counted RESETs in the ledger (middleware rows with ``action`` RESET and ``counted``; the opening START isn't
    counted)."""
    resets = 0
    for line in ledger.read_text().splitlines():
        row = json.loads(line)
        if row["game"] != game_id:
            raise RuntimeError(f"{ledger}: row for {row['game']}, the episode plays {game_id}")
        resets += row["action"] == "RESET" and bool(row["counted"])
    return resets


def ledger_actions(ledger: Path, game_id: str) -> tuple[int, int, int]:
    """(ledger rows, counted actions, highest levels_completed), from middleware's ledger (host side, the only
    authority). 0 rows = the agent never started the game (START is a row too, not counted as an action)."""
    rows, actions, levels = 0, 0, 0
    for line in ledger.read_text().splitlines():
        row = json.loads(line)
        if row["game"] != game_id:
            raise RuntimeError(f"{ledger}: row for {row['game']}, the episode plays {game_id}")
        rows += 1
        actions += bool(row["counted"])
        levels = max(levels, int(row["levels_completed"] or 0))
    return rows, actions, levels


class ActionCharge:
    """Per-action charge (harness config action_token_charge / action_charge_free_fraction, g2seed): TaskProxy's
    ``charge``. Charged = floor(C × max(0, a − f·b) / b), a = counted actions in the ledger so far (f is computed as a
    fraction of its decimal literal, so 0.7·10 is exactly 7).
    middleware appends the ledger row by row (host side): read incrementally up to the last newline (a half-written row is
    read next time), remembering the position read; a ledger that shrinks or a row of another game fails loudly. The
    proxy's handler threads and the harness's poll ask concurrently, hence the lock."""

    def __init__(self, ledger: Path, game_id: str, token_charge: int, baseline: int, free_fraction: float) -> None:
        if not (token_charge > 0 and baseline > 0 and 0 <= free_fraction <= 1):
            raise ValueError(f"action charge {token_charge} / baseline {baseline} / free fraction {free_fraction}")
        self.ledger, self.game_id, self.token_charge, self.baseline = ledger, game_id, token_charge, baseline
        self.free_fraction = free_fraction
        self.free = fractions.Fraction(repr(free_fraction)) * baseline     # actions not charged, f·b
        self.lock = threading.Lock()
        self.offset = 0
        self.actions = 0

    def tokens(self, actions: int) -> int:
        """Charge for ``actions`` counted actions."""
        return math.floor(self.token_charge * max(fractions.Fraction(0), actions - self.free) / self.baseline)

    def record(self, actions: int) -> dict:
        """Charge record for stop files and results."""
        return {"token_charge": self.token_charge, "free_fraction": self.free_fraction, "baseline_actions": self.baseline,
                "actions": actions, "charged_tokens": self.tokens(actions)}

    def snapshot(self) -> dict:
        with self.lock:
            return self.record(self.actions)

    def __call__(self) -> int:
        with self.lock:
            if self.ledger.exists():                # no ledger yet before the game starts
                size = self.ledger.stat().st_size
                if size < self.offset:
                    raise RuntimeError(f"{self.ledger}: shrank to {size} bytes after {self.offset} were read")
                with self.ledger.open("rb") as f:
                    f.seek(self.offset)
                    data = f.read()
                end = data.rfind(b"\n") + 1
                for line in data[:end].splitlines():
                    row = json.loads(line)
                    if row["game"] != self.game_id:
                        raise RuntimeError(f"{self.ledger}: row for {row['game']}, the episode plays {self.game_id}")
                    self.actions += bool(row["counted"])
                self.offset += end
            return self.tokens(self.actions)


def action_charge(cfg: dict, stub: dict, ledger: Path) -> ActionCharge | None:
    """This game's ActionCharge if the harness config has action_token_charge (load_config has checked keys and
    ranges), otherwise None."""
    if "action_token_charge" not in cfg:
        return None
    return ActionCharge(ledger, stub["game_id"], cfg["action_token_charge"], stub["baseline_actions"],
                        cfg.get("action_charge_free_fraction", 0))


def tokens_at(proxy_log: Path, unix: float) -> int:
    """The proxy's cumulative output tokens at ``unix`` (cum_output_tokens of the last request that finished before
    then)."""
    best = 0
    for line in proxy_log.read_text().splitlines():
        row = json.loads(line)
        if "cum_output_tokens" in row and row["t"] + row.get("seconds", 0.0) <= unix:
            best = max(best, row["cum_output_tokens"])
    return best


def tokenizer_copy(run_dir: Path) -> Path:
    """codex's ``tool_output_tokenizer``: the jail must see it, so use the copy in the group directory (as run_arc2.py
    does)."""
    return run_dir / "harness" / "tokenizer.json"


def arm_config(ctx: Context, ports: list[int]) -> dict:
    """This episode's run_arm config: the Kaggle arm's config, changing only what one game / one level needs."""
    cfg = dict(ctx.cfg["_arm"])
    cfg.update(games=0, concurrency=1, game_seconds=0, wall_seconds=ctx.cfg["episode_wall_seconds"],
               gateway_port=ports[0], official_port=ports[1], keep_driver_home=False, heartbeat_seconds=0,
               tail_resume=False)
    cfg.pop("tail_resume_min_seconds", None)
    # the copy in the group directory (copied by run_group): the original often lives in /dev/shm, and jail.sh's
    # --dev /dev covers read-only mounts under /dev
    cfg["driver_files"] = {**cfg.get("driver_files", {}), "tool_output_tokenizer": str(tokenizer_copy(ctx.run_dir))}
    cfg["note"] = f"[run_arc3_level {episode_label(ctx.stub)}] " + cfg.get("note", "")
    cfg.update({key: ctx.cfg[key] for key in ARM_PASS_KEYS if key in ctx.cfg})
    return cfg


def process_tree(pid: int) -> list[int]:
    """All descendants of ``pid`` (ppid from /proc; ones that exit during the listing are skipped)."""
    children: dict[int, list[int]] = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = stat.read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue                                # exited while listing (ESRCH when it goes between open and read)
        children.setdefault(int(fields[1]), []).append(int(stat.parent.name))
    out, todo = [], [pid]
    while todo:
        for child in children.get(todo.pop(), []):
            out.append(child)
            todo.append(child)
    return out


def kill_tree(pid: int) -> list[int]:
    """SIGKILL ``pid`` and all its descendants; return the pids signalled. SIGSTOP until no new ones show up, then
    SIGKILL them together: killing the parent first gets the children adopted by init and dropped from the tree (the
    middleware / official gateway / bridge that run_arm starts share this harness's process group, so killpg would kill
    other episodes too; codex's outer bwrap is --die-with-parent, so its pid namespace is cleaned up once run_arm dies)."""
    seen: set[int] = set()
    try:
        while True:
            found = ({pid, *process_tree(pid)} - seen)
            if not found:
                break
            seen |= found
            for p in found:
                try:
                    os.kill(p, signal.SIGSTOP)
                except ProcessLookupError:
                    pass                            # exited since it was listed
    finally:                                        # however the listing ends, nothing stays stopped
        for p in seen | {pid}:
            try:
                os.kill(p, signal.SIGKILL)
            except ProcessLookupError:
                pass
    return sorted(seen)


def run_episode(ctx: Context, ep: Episode) -> dict:
    cfg, stub, args = ctx.cfg, ctx.stub, ctx.args
    ep_dir = ctx.run_dir / "episodes" / ep.name
    ep_dir.mkdir(parents=True)                      # an episode directory is used only once
    stop_dir = ep_dir / "stop"
    stop_dir.mkdir(mode=0o700)
    backend = ctx.backends[ep.sample] if args.sample_backends else args.model_base_url
    salt = f"{args.salt_prefix}/{ep.name}" if cfg.get("cache_salt", "none") == "episode" else None
    proxy_log = ctx.run_dir / "proxy" / f"{ep.name}.jsonl"
    lease = PortLease(2)
    charge = action_charge(cfg, stub, ep_dir / "arm" / "ledger.jsonl")    # ledger written by run_arm --run-dir ep_dir/arm
    proxy = model_proxy.TaskProxy(ep.name, episode_budget(cfg, stub), host_port(backend), proxy_log,
                                  ctx.server_context, salt_field=CACHE_SALT_FIELD if salt else None, salt=salt,
                                  routes=model_proxy.CODEX_ROUTES, require_session=True, stop_dir=stop_dir,
                                  context_limit=cfg.get("context_token_limit"), charge=charge,
                                  pace_path=pace_path(cfg, stub, ep_dir),
                                  stop_after_first_window=cfg.get("stop_after_first_window", False))
    server, proxy_port = model_proxy.serve_tcp(proxy)
    began = time.monotonic()
    proc = None
    aborted: list[dict] = []
    closed: list[dict] = []
    try:
        arm = arm_config(ctx, lease.ports)
        atomic_write(ep_dir / "arm_config.json", json.dumps(arm, indent=1, ensure_ascii=False) + "\n")
        argv = [str(expand(cfg["arm_python"])), str(UNIT / "run_arm.py"),
                "--config", str(ep_dir / "arm_config.json"), "--run-dir", str(ep_dir / "arm"),
                "--corpus", str(expand(cfg["corpus"])), "--prep", str(expand(cfg["prep"])),
                "--python", str(expand(cfg["gateway_python"])), "--agent-python", str(args.agent_python),
                "--driver-bin", str(args.driver_bin), "--driver", "codex", "--ffmpeg", str(expand(cfg["ffmpeg"])),
                "--model-base-url", backend, "--agent-model-url", f"http://127.0.0.1:{proxy_port}/v1",
                "--model", args.model, "--games", stub["game"], "--tag", ep.name, "--stop-dir", str(stop_dir)]
        if is_game(stub):                           # whole-game episode: --stop-dir + --proxy-log (run_arm game_episode_config)
            argv += ["--proxy-log", str(proxy_log)]
        elif is_g2(stub):                           # gen-2 single level: whole-game set + the seed and this level's action cap
            atomic_write(ep_dir / "game_seed.json", json.dumps(g2_seed_spec(stub).to_json(), indent=1) + "\n")
            argv += ["--proxy-log", str(proxy_log), "--game-seed", str(ep_dir / "game_seed.json"),
                     "--episode-action-cap", str(stub["action_cap"])]
        else:
            argv += ["--level-episode", str(stub["level"]), "--action-cap", str(stub["action_cap"]),
                     "--expect-start-hash", stub["start_hash"]]
        spec = None if is_g2(stub) else seed_spec(stub)   # g2seed's seed goes through --game-seed (above)
        if spec is not None:
            atomic_write(ep_dir / "level_seed.json", json.dumps(spec.to_json(), indent=1) + "\n")
            argv += ["--level-seed", str(ep_dir / "level_seed.json")]
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        with (ep_dir / "run_arm.log").open("w") as out:
            proc = subprocess.Popen(argv, cwd=ROOT, env=env, stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT)
            terminated: float | None = None        # monotonic time of the SIGTERM
            killed: list[int] = []                  # kill_tree's pids after TERM_GRACE_SECONDS
            while proc.poll() is None:
                if proxy.fatal is not None:
                    raise RuntimeError(f"{ep.name}: proxy fatal: {proxy.fatal}")
                if proxy.exhausted() and episode_stops.BUDGET not in episode_stops.stops(stop_dir):
                    episode_stops.write_stop(stop_dir, episode_stops.BUDGET,
                                             {"output_tokens": proxy.output_tokens,
                                              **({} if charge is None else {"action_charge": charge.snapshot()}),
                                              "unix": time.time()})
                if proxy.context_exhausted() and episode_stops.CONTEXT not in episode_stops.stops(stop_dir):
                    episode_stops.write_stop(stop_dir, episode_stops.CONTEXT,
                                             {"context": proxy.context_peak, "context_cut": proxy.context_cut,
                                              "output_tokens": proxy.output_tokens, "unix": time.time()})
                if proxy.window_exhausted() and episode_stops.WINDOW not in episode_stops.stops(stop_dir):
                    # the handoff message (window_handoff) needs the gen2_budget.json the runner writes for the hook;
                    # driver home is deleted when the session is reaped (keep_driver_home false), and run_arm reaps as
                    # soon as the stop file is written, so save a copy first (likewise the hook's log: window_cause reads
                    # the record of window 1's opening SessionStart, written before window 1's request is sent)
                    driver_home = g2_workspace(stub, ep_dir) / ".driver"
                    shutil.copyfile(driver_home / gen2_hooks().BUDGET, ep_dir / G2_BUDGET_COPY)
                    shutil.copyfile(driver_home / G2_HOOK_LOG_COPY, ep_dir / G2_HOOK_LOG_COPY)
                    episode_stops.write_stop(stop_dir, episode_stops.WINDOW,
                                             {"window_cut": proxy.window_cut, "output_tokens": proxy.output_tokens,
                                              "unix": time.time()})
                overrun = time.monotonic() - began > cfg["episode_wall_seconds"] + OVERRUN_SECONDS
                if (ctx.cancelled.is_set() or overrun) and terminated is None:
                    log(f"{ep.name}: {'cancelled' if ctx.cancelled.is_set() else 'run_arm overran its wall'}; SIGTERM")
                    proc.terminate()                # run_arm reaps the session scope, writes the manifest as usual, exits
                    terminated = time.monotonic()
                elif terminated is not None and time.monotonic() - terminated > TERM_GRACE_SECONDS:
                    killed = kill_tree(proc.pid)
                    log(f"{ep.name}: run_arm still runs {TERM_GRACE_SECONDS}s after SIGTERM; SIGKILL its process tree "
                        f"{killed}")
                    proc.wait()
                    break
                time.sleep(POLL_SECONDS)
        if killed:                                  # failures.jsonl carries it: a sweep can count the hung run_arms
            raise RuntimeError(f"{ep.name}: KILLED_TREE run_arm ignored SIGTERM for {TERM_GRACE_SECONDS}s; SIGKILLed "
                               f"{len(killed)} process(es)")
        if ctx.cancelled.is_set():
            raise RuntimeError(f"{ep.name}: cancelled")
        if proxy.window_exhausted() and episode_stops.WINDOW not in episode_stops.stops(stop_dir):
            # run_arm exited between the proxy's window cut and the next poll: no stop file, no budget / hook log copies
            raise RuntimeError(f"{ep.name}: the proxy cut window {proxy.window_cut} but run_arm exited before the window "
                               "stop was written")
        if proc.returncode != 0:
            raise RuntimeError(f"{ep.name}: run_arm exited {proc.returncode}; see {ep_dir / 'run_arm.log'}")
    finally:
        if proc is not None and proc.poll() is None:
            kill_tree(proc.pid)                     # its middleware / gateway / bridges too, not only run_arm
            proc.wait()
        lease.release()
        # run_arm and codex are both gone: abort what is still generating, cut upstream, and close the sessions left
        # open (a killed codex never closes its session)
        try:
            aborted = proxy.abort_upstreams()
        finally:
            closed = proxy.close_open_sessions()
            server.shutdown()
            server.server_close()
    return finish_episode(ctx, ep, ep_dir, proxy, proxy_log, aborted, closed, began, backend, salt)


def finish_episode(ctx: Context, ep: Episode, ep_dir: Path, proxy, proxy_log: Path, aborted: list[dict],
                   closed: list[dict], began: float, backend: str, salt: str | None) -> dict:
    if is_game(ctx.stub):
        return finish_game_episode(ctx, ep, ep_dir, proxy, proxy_log, aborted, closed, began, backend, salt)
    if is_g2(ctx.stub):
        return finish_g2_episode(ctx, ep, ep_dir, proxy, proxy_log, aborted, closed, began, backend, salt)
    stub = ctx.stub
    stops = episode_stops.stops(ep_dir / "stop")
    if episode_stops.START_FAILED in stops:
        raise RuntimeError(f"{ep.name}: the game did not start at level {stub['level']}: "
                           f"{stops[episode_stops.START_FAILED]}")
    arm_dir = ep_dir / "arm"
    state = json.loads((arm_dir / "ARM_STATE.json").read_text())
    if len(state["plays"]) != 1:
        raise RuntimeError(f"{ep.name}: ARM_STATE has {len(state['plays'])} plays, want 1")
    play = state["plays"][0]
    if play["status"] != "finished":
        raise RuntimeError(f"{ep.name}: run_arm play status {play['status']!r}: {play.get('error')}")
    rows, actions, levels = ledger_actions(arm_dir / "ledger.jsonl", stub["game_id"])
    cleared = episode_stops.CLEARED in stops
    # stop files and ledger must agree: cleared = some action in the ledger pushed levels_completed to K
    if cleared != (levels >= stub["level"]):
        raise RuntimeError(f"{ep.name}: stop files {sorted(stops)} but the ledger reaches levels_completed {levels}")
    if actions > stub["action_cap"]:
        raise RuntimeError(f"{ep.name}: {actions} counted actions > cap {stub['action_cap']} (middleware should refuse)")
    sessions = play["sessions"]
    breach = [s["attempt"] for s in sessions if s["returncode"] == 70]
    if breach:
        # jail.sh 70: the corpus/engine is visible inside the jail. This episode's score is not attributable and must
        # not count as a failed sample
        raise RuntimeError(f"{ep.name}: FENCE BREACH (jail.sh exit 70) in session(s) {breach}")
    if cleared:
        stop = "cleared"
    elif episode_stops.ACTION_CAP in stops:
        stop = "action_cap"
    elif episode_stops.BUDGET in stops:
        stop = "budget"
    elif episode_stops.CONTEXT in stops:
        stop = "context"
    elif episode_stops.DIED in stops:
        stop = "died"                             # codex died (run_arm.level_death): a failure, not resumed
    elif any(s["returncode"] == 124 or s["unit_result"] == "timeout" for s in sessions):
        stop = "wall"
    else:
        stop = "gave_up"                          # the agent stopped by itself, sessions used up (run_arm's give-up rule)
    manifest = json.loads((arm_dir / "tito_manifest.json").read_text())
    if [i["id"] for i in manifest["items"]] != [stub["game"]]:
        raise RuntimeError(f"{ep.name}: run_arm manifest items {[i['id'] for i in manifest['items']]}")
    item = manifest["items"][0]
    overflows = sum(any(k in s for k in ("context_recovery", "stuck_context_limit", "context_error")) for s in sessions)
    result = {"task": ep.name, "tid": ep.tid, "sample": ep.sample, "episode": ep.name, "harness": "arc3",
              "score": 1.0 if cleared else 0.0, "correct": cleared, "stop": stop, "harness_killed": stop in KILL_STOPS,
              "usage": {"proxy_output_tokens": proxy.output_tokens, "proxy_requests": proxy.requests,
                        "proxy_refused": proxy.refused, "proxy_forbidden": proxy.forbidden,
                        "context_peak": proxy.context_peak if proxy.context_limit is not None else None},
              "output_token_budget": episode_budget(ctx.cfg, stub),
              "game": stub["game"], "game_id": stub["game_id"], "level": stub["level"], "actions": actions,
              "action_cap": stub["action_cap"], "baseline_actions": stub["baseline_actions"],
              "tokens_at_clear": tokens_at(proxy_log, stops[episode_stops.CLEARED]["unix"]) if cleared else None,
              "seed": None if "seed" not in stub else {"ws": stub["seed"],
                                                       **{k[len("seed_"):]: stub[k] for k in sorted(SEED_KEYS - {"seed"})}},
              "stops": stops, "context_overflows": overflows,
              "sessions": [{k: s[k] for k in ("attempt", "returncode", "unit_result", "mode", "handle", "seconds",
                                              "actions_after")} for s in sessions],
              "threads": [t["thread"] for t in item["threads"] if t["in_sessions"]], "digests": item["digests"],
              "seconds": round(time.monotonic() - began, 1), "backend": backend,
              "sessions_closed_by_harness": closed, "requests_aborted_by_harness": aborted,
              **({} if salt is None else {"cache_salt": salt})}
    # large files: per-step images/frames (record/current.* and history.jsonl, which the agent reads, are kept), the
    # official per-step recordings. The ledger and TITO are the data.
    # steps/ is created at game start (START is also recorded as a step), so it is absent if the agent never started;
    # recordings/ is created as soon as the official gateway starts
    if rows:
        shutil.rmtree(arm_dir / "games" / stub["game"] / "record" / "steps")
    shutil.rmtree(arm_dir / "recordings")
    atomic_write(ctx.run_dir / "results" / f"{ep.name}.json", json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    log(f"{ep.name}: {stop} score {result['score']:.0f}  actions {actions}/{stub['action_cap']}  "
        f"tokens {proxy.output_tokens}  forbidden {proxy.forbidden}  {result['seconds']:.0f}s")
    return result


def game_progress(ledger: Path, game_id: str, base: list[int]) -> dict:
    """middleware ledger (host side, the only authority) → this game's level clears and official whole-game score.
    Competition mode refuses replays: a game can have only one play."""
    score = arc_code_module("score")
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    if any(row["game"] != game_id for row in rows):
        raise RuntimeError(f"{ledger}: rows for {sorted({row['game'] for row in rows})}, the episode plays {game_id}")
    plays = score.plays_from_ledger(rows).get(game_id, [])
    if len(plays) > 1:
        raise RuntimeError(f"{ledger}: {len(plays)} plays of {game_id} (the gateway refuses replays)")
    play = plays[0] if plays else {"spans": [], "actions": 0, "levels": 0, "won": False}
    clears, done = [], 0                          # unix of the counted action that completed each level
    for row in rows:
        if row.get("counted"):
            while int(row.get("levels_completed") or 0) > done:
                clears.append(row["unix"])
                done += 1
    if done != play["levels"] or len(play["spans"]) != done or done > len(base):
        raise RuntimeError(f"{ledger}: levels {done} / play {play['levels']} / spans {len(play['spans'])} / {len(base)} levels")
    if play["won"] != (done == len(base)):
        raise RuntimeError(f"{ledger}: WIN {play['won']} but {done}/{len(base)} levels")
    return {"rows": len(rows), "actions": play["actions"], "levels": done, "won": play["won"], "spans": play["spans"],
            "level_scores": [score.level_score(a, b) for a, b in zip(play["spans"], base)],
            "game_score": score.game_score(play["spans"], base), "clear_unix": clears}


def finish_game_episode(ctx: Context, ep: Episode, ep_dir: Path, proxy, proxy_log: Path, aborted: list[dict],
                        closed: list[dict], began: float, backend: str, salt: str | None) -> dict:
    """Whole-game episode (kind game): R = official whole-game score ÷ 100. Stop reasons in order: won (all levels
    cleared), budget (the proxy's output budget), died (codex died, written by run_arm), wall, gave_up (sessions used up
    or the agent stopped by itself)."""
    stub = ctx.stub
    stops = episode_stops.stops(ep_dir / "stop")
    unknown = set(stops) - {episode_stops.BUDGET, episode_stops.DIED}
    if unknown:
        raise RuntimeError(f"{ep.name}: stop files {sorted(unknown)} in a whole-game episode (only budget / died)")
    arm_dir = ep_dir / "arm"
    state = json.loads((arm_dir / "ARM_STATE.json").read_text())
    if len(state["plays"]) != 1:
        raise RuntimeError(f"{ep.name}: ARM_STATE has {len(state['plays'])} plays, want 1")
    play = state["plays"][0]
    if play["status"] != "finished":
        raise RuntimeError(f"{ep.name}: run_arm play status {play['status']!r}: {play.get('error')}")
    progress = game_progress(arm_dir / "ledger.jsonl", stub["game_id"], stub["baseline_actions"])
    sessions = play["sessions"]
    breach = [s["attempt"] for s in sessions if s["returncode"] == 70]
    if breach:
        raise RuntimeError(f"{ep.name}: FENCE BREACH (jail.sh exit 70) in session(s) {breach}")
    if progress["won"]:
        stop = "won"
    elif episode_stops.BUDGET in stops:
        stop = "budget"
    elif episode_stops.DIED in stops:
        stop = "died"
    elif any(s["returncode"] == 124 or s["unit_result"] == "timeout" for s in sessions):
        stop = "wall"
    else:
        stop = "gave_up"
    manifest = json.loads((arm_dir / "tito_manifest.json").read_text())
    if [i["id"] for i in manifest["items"]] != [stub["game"]]:
        raise RuntimeError(f"{ep.name}: run_arm manifest items {[i['id'] for i in manifest['items']]}")
    item = manifest["items"][0]
    threads = [t for t in item["threads"] if t["in_sessions"]]
    overflows = sum(any(k in s for k in ("context_recovery", "stuck_context_limit", "context_error")) for s in sessions)
    level_tokens = [tokens_at(proxy_log, unix) for unix in progress["clear_unix"]]
    result = {"task": ep.name, "tid": ep.tid, "sample": ep.sample, "episode": ep.name, "harness": "arc3", "kind": "game",
              "score": progress["game_score"] / 100.0, "correct": progress["won"], "stop": stop,
              "harness_killed": stop in KILL_STOPS,
              "usage": {"proxy_output_tokens": proxy.output_tokens, "proxy_requests": proxy.requests,
                        "proxy_refused": proxy.refused, "proxy_forbidden": proxy.forbidden, "context_peak": None},
              "output_token_budget": episode_budget(ctx.cfg, stub),
              "game": stub["game"], "game_id": stub["game_id"], "level": None, "win_levels": stub["win_levels"],
              "levels": progress["levels"], "game_score": progress["game_score"], "level_spans": progress["spans"],
              "level_scores": progress["level_scores"], "level_tokens": level_tokens,
              "actions": progress["actions"], "action_cap": None, "baseline_actions": sum(stub["baseline_actions"]),
              "tokens_at_clear": level_tokens[-1] if progress["won"] else None, "seed": None,
              "windows": sum(len(t["windows"]) for t in threads),
              "compactions": sum(w["compaction_response_id"] is not None for t in threads for w in t["windows"]),
              "stops": stops, "context_overflows": overflows,
              "sessions": [{k: s[k] for k in ("attempt", "returncode", "unit_result", "mode", "handle", "seconds",
                                              "actions_after")} for s in sessions],
              "threads": [t["thread"] for t in threads], "digests": item["digests"],
              "seconds": round(time.monotonic() - began, 1), "backend": backend,
              "sessions_closed_by_harness": closed, "requests_aborted_by_harness": aborted,
              **({} if salt is None else {"cache_salt": salt})}
    if progress["rows"]:
        shutil.rmtree(arm_dir / "games" / stub["game"] / "record" / "steps")
    shutil.rmtree(arm_dir / "recordings")
    atomic_write(ctx.run_dir / "results" / f"{ep.name}.json", json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    log(f"{ep.name}: {stop} R {result['score']:.3f}  levels {progress['levels']}/{stub['win_levels']}  actions "
        f"{progress['actions']}  tokens {proxy.output_tokens}  windows {result['windows']}  {result['seconds']:.0f}s")
    return result


def finish_g2_episode(ctx: Context, ep: Episode, ep_dir: Path, proxy, proxy_log: Path, aborted: list[dict],
                      closed: list[dict], began: float, backend: str, salt: str | None) -> dict:
    """gen-2 single-level episode (kind g2seed): R = 1[cleared] × min(1, (κ·baseline / actions)²), κ =
    efficiency_free_fraction (default 1). Stop reasons in order: cleared, action_cap, window (stop_after_first_window:
    the first window change), budget, context (the proxy's context cap), died, wall, gave_up. start_failed (the seed
    opened the wrong board) fails this game. The result also records the ledger's RESET count (``resets``) and the
    window-change cause (``window_cause``, non-None only for stop-window games); stop-window games also save the handoff
    message (window_handoff)."""
    stub = ctx.stub
    stops = episode_stops.stops(ep_dir / "stop")
    known = {episode_stops.CLEARED, episode_stops.ACTION_CAP, episode_stops.START_FAILED, episode_stops.BUDGET,
             episode_stops.CONTEXT, episode_stops.DIED, episode_stops.WINDOW}
    if set(stops) - known:
        raise RuntimeError(f"{ep.name}: unknown stop files {sorted(set(stops) - known)}")
    first_window = bool(ctx.cfg.get("stop_after_first_window", False))   # optional key (load_config checks it)
    if episode_stops.WINDOW in stops and not first_window:
        raise RuntimeError(f"{ep.name}: stop file window without the harness config's stop_after_first_window")
    if episode_stops.START_FAILED in stops:
        raise RuntimeError(f"{ep.name}: the seed did not open level {stub['level']}: {stops[episode_stops.START_FAILED]}")
    arm_dir = ep_dir / "arm"
    state = json.loads((arm_dir / "ARM_STATE.json").read_text())
    if len(state["plays"]) != 1:
        raise RuntimeError(f"{ep.name}: ARM_STATE has {len(state['plays'])} plays, want 1")
    play = state["plays"][0]
    if play["status"] != "finished":
        raise RuntimeError(f"{ep.name}: run_arm play status {play['status']!r}: {play.get('error')}")
    # run_arm --game-seed --episode-action-cap always records these two (missing = not this kind of episode's run_arm;
    # the KeyError is loud anyway)
    if state["game_seed"] != stub["seed"]:
        raise RuntimeError(f"{ep.name}: run_arm played seed {state['game_seed']}, the stub's is {stub['seed']}")
    if (state["g2_episode"]["level"], state["g2_episode"]["action_cap"]) != (stub["level"], stub["action_cap"]):
        raise RuntimeError(f"{ep.name}: run_arm g2_episode {state['g2_episode']}, the stub's is level {stub['level']} "
                           f"cap {stub['action_cap']}")
    # the ledger has only this game's rows (middleware continues from the seed's last step, --prior-actions): every
    # counted action is level K's
    rows, actions, levels = ledger_actions(arm_dir / "ledger.jsonl", stub["game_id"])
    cleared = episode_stops.CLEARED in stops
    if cleared != (levels >= stub["level"]):
        raise RuntimeError(f"{ep.name}: stop files {sorted(stops)} but the ledger reaches levels_completed {levels}")
    if levels > stub["level"]:
        raise RuntimeError(f"{ep.name}: the ledger reaches levels_completed {levels} > {stub['level']}: the middleware "
                           "should refuse every action after the clear")
    if actions > stub["action_cap"]:
        raise RuntimeError(f"{ep.name}: {actions} counted actions > cap {stub['action_cap']} (middleware should refuse)")
    if cleared and actions < 1:
        raise RuntimeError(f"{ep.name}: cleared with {actions} counted actions")
    sessions = play["sessions"]
    breach = [s["attempt"] for s in sessions if s["returncode"] == 70]
    if breach:
        raise RuntimeError(f"{ep.name}: FENCE BREACH (jail.sh exit 70) in session(s) {breach}")
    if cleared:
        stop = "cleared"
    elif episode_stops.ACTION_CAP in stops:
        stop = "action_cap"
    elif episode_stops.WINDOW in stops:
        stop = "window"
    elif episode_stops.BUDGET in stops:
        stop = "budget"
    elif episode_stops.CONTEXT in stops:
        stop = "context"
    elif episode_stops.DIED in stops:
        stop = "died"
    elif any(s["returncode"] == 124 or s["unit_result"] == "timeout" for s in sessions):
        stop = "wall"
    else:
        stop = "gave_up"
    resets = ledger_resets(arm_dir / "ledger.jsonl", stub["game_id"])
    kappa = ctx.cfg.get("efficiency_free_fraction", 1.0)   # optional key, default 1 = v5's reward (load_config checks it)
    efficiency = min(1.0, (kappa * stub["baseline_actions"] / actions) ** 2) if cleared else 0.0
    # the charge is computed from the ledger's final action count; the last value the proxy asked for (before the final
    # few actions) can't exceed it -- if it does, that is an ActionCharge bug
    charge = action_charge(ctx.cfg, stub, arm_dir / "ledger.jsonl")
    charged = None if charge is None else {**charge.record(actions), "proxy_last_seen": proxy.charge_seen}
    if charged is not None and charged["proxy_last_seen"] > charged["charged_tokens"]:
        raise RuntimeError(f"{ep.name}: the proxy charged {charged['proxy_last_seen']} tokens, more than the "
                           f"{charged['charged_tokens']} of the ledger's {actions} actions")
    manifest = json.loads((arm_dir / "tito_manifest.json").read_text())
    if [i["id"] for i in manifest["items"]] != [stub["game"]]:
        raise RuntimeError(f"{ep.name}: run_arm manifest items {[i['id'] for i in manifest['items']]}")
    item = manifest["items"][0]
    threads = [t for t in item["threads"] if t["in_sessions"]]
    windows = sum(len(t["windows"]) for t in threads)
    if windows > len(threads) and not ctx.cfg.get("window_change"):
        # one sequence per game: context_token_limit is below the change point. A window change means the setup is wrong,
        # not a sample (with window_change, windows change as deployed)
        raise RuntimeError(f"{ep.name}: {windows} context windows in {len(threads)} thread(s): the episode changed window "
                           f"(context_token_limit must stay under the g2 change point)")
    if first_window:
        # the ledger has only window 0: the proxy blocks the first request of window ≥ 1, so the window 1 that codex
        # remembers has no responses
        late = [(t["thread"], w["window_number"]) for t in threads for w in t["windows"]
                if w["window_number"] >= 1 and (w["response_ids"] or w["compaction_response_id"] is not None)]
        if late:
            raise RuntimeError(f"{ep.name}: stop_after_first_window but windows {late} have responses")
    if (proxy.window_cut is not None) != (episode_stops.WINDOW in stops):
        raise RuntimeError(f"{ep.name}: proxy window_cut {proxy.window_cut} but stop files {sorted(stops)}: the window stop "
                           "and the proxy's cut must go together (stop window implies both)")
    handoff = window_handoff(ctx, ep_dir) if stop == "window" else None
    cause, cause_evidence = window_cause(ep_dir, proxy_log) if stop == "window" else (None, None)
    overflows = sum(any(k in s for k in ("context_recovery", "stuck_context_limit", "context_error")) for s in sessions)
    result = {"task": ep.name, "tid": ep.tid, "sample": ep.sample, "episode": ep.name, "harness": "arc3", "kind": "g2seed",
              "score": efficiency, "correct": cleared, "stop": stop, "harness_killed": stop in KILL_STOPS,
              "usage": {"proxy_output_tokens": proxy.output_tokens, "proxy_requests": proxy.requests,
                        "proxy_refused": proxy.refused, "proxy_forbidden": proxy.forbidden,
                        "context_peak": proxy.context_peak},
              "output_token_budget": episode_budget(ctx.cfg, stub),
              "game": stub["game"], "game_id": stub["game_id"], "level": stub["level"], "actions": actions,
              "action_cap": stub["action_cap"], "baseline_actions": stub["baseline_actions"], "efficiency": efficiency,
              "tokens_at_clear": tokens_at(proxy_log, stops[episode_stops.CLEARED]["unix"]) if cleared else None,
              **({} if charged is None else {"action_charge": charged}),
              **({"windows": windows,
                  "compactions": sum(w["compaction_response_id"] is not None for t in threads for w in t["windows"])}
                 if ctx.cfg.get("window_change") else {}),
              "resets": resets, "window_cause": cause,
              **({} if handoff is None else {"window_cut": stops[episode_stops.WINDOW]["window_cut"], "handoff": handoff,
                                             "window_evidence": cause_evidence}),
              "seed": dict(stub["seed"]), "stops": stops, "context_overflows": overflows,
              "sessions": [{k: s[k] for k in ("attempt", "returncode", "unit_result", "mode", "handle", "seconds",
                                              "actions_after")} for s in sessions],
              "threads": [t["thread"] for t in threads], "digests": item["digests"],
              "seconds": round(time.monotonic() - began, 1), "backend": backend,
              "sessions_closed_by_harness": closed, "requests_aborted_by_harness": aborted,
              **({} if salt is None else {"cache_salt": salt})}
    # record/steps of the first K−1 levels are hard links into the seed store (run_arm link_steps): rmtree only removes
    # the links, the seed store is unaffected
    if rows:
        shutil.rmtree(arm_dir / "games" / stub["game"] / "record" / "steps")
    shutil.rmtree(arm_dir / "recordings")
    atomic_write(ctx.run_dir / "results" / f"{ep.name}.json", json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    log(f"{ep.name}: {stop} R {efficiency:.3f}  actions {actions}/{stub['action_cap']} (baseline {stub['baseline_actions']})  "
        f"tokens {proxy.output_tokens}" + ("" if charged is None else f" + charge {charged['charged_tokens']}") +
        f"  context {proxy.context_peak}" + (f"  windows {windows}" if ctx.cfg.get("window_change") else "") +
        f"  {result['seconds']:.0f}s")
    return result


def tito_items(results: list[dict], failures: list[dict]) -> list[dict]:
    """One item per episode in the group manifest: thread/digest from run_arm's own manifest, outcome replaced with the
    RL fields (join.Manifest.outcome gives a reward only when it sees arc3 has score; prepare_rl needs context_overflows)."""
    items = [{"id": r["task"], "threads": r["threads"], "digests": r["digests"],
              "outcome": {"status": "finished", "score": r["score"], "correct": r["correct"], "stop": r["stop"],
                          "harness_killed": r["harness_killed"],
                          "context_overflows": r["context_overflows"], "wrapup_cut_rid": None,
                          "output_tokens": r["usage"]["proxy_output_tokens"],
                          "output_token_budget": r["output_token_budget"], "game": r["game"],
                          "level": r["level"], "actions": r["actions"], "action_cap": r["action_cap"],
                          "baseline_actions": r["baseline_actions"], "tokens_at_clear": r["tokens_at_clear"],
                          "seed_label": None if r["seed"] is None else r["seed"]["label"],
                          **({"levels": r["levels"], "game_score": r["game_score"], "windows": r["windows"],
                              "compactions": r["compactions"]} if r.get("kind") == "game" else {}),
                          **({"efficiency": r["efficiency"], "resets": r["resets"], "window_cause": r["window_cause"]}
                             if r.get("kind") == "g2seed" else {}),
                          **({"handoff": r["handoff"]} if "handoff" in r else {}),
                          **({"windows": r["windows"], "compactions": r["compactions"]}
                             if r.get("kind") == "g2seed" and "windows" in r else {})}}
             for r in results]
    items += [{"id": f["task"], "threads": [], "digests": [], "outcome": {"status": "failed", "error": f["error"]}}
              for f in failures]
    return sorted(items, key=lambda item: item["id"])


def run_group(args: argparse.Namespace, cfg: dict) -> None:
    if args.samples is None or args.samples < 2 or not args.only or "," in args.only:
        raise SystemExit("run_arc3_level runs one group: --only <one tid> --samples N (N >= 2)")
    tid = args.only
    if (cfg.get("cache_salt", "none") == "episode") != (args.salt_prefix is not None):
        raise SystemExit("cache_salt episode needs --salt-prefix, and only it")
    stub = load_stub(args.tasks_dir, tid, cfg["kind"])
    budget = episode_budget(cfg, stub)
    run_dir = args.run_dir
    run_dir.mkdir(parents=True)                     # a group directory is used only once
    (run_dir / "results").mkdir()
    (run_dir / "proxy").mkdir()
    tokenizer_copy(run_dir).parent.mkdir()
    shutil.copyfile(args.tokenizer_json, tokenizer_copy(run_dir))
    backends = args.sample_backends.split(",") if args.sample_backends else [args.model_base_url]
    if args.sample_backends and (len(backends) != args.samples or backends[0] != args.model_base_url):
        raise SystemExit(f"--sample-backends must give one backend per sample, the first = --model-base-url: {backends}")
    distinct = list(dict.fromkeys(backends))
    server_context = check_backends(distinct, args.model, cfg["model_context_window"])
    preflight = tito_module("preflight")
    servers = [preflight.check_server(server_root(b)) for b in distinct]
    if len({s["instance"] for s in servers}) != len(servers):
        raise SystemExit(f"two backends report the same ledger instance: {[s['instance'] for s in servers]}")
    ctx = Context(args=args, cfg=cfg, stub=stub, run_dir=run_dir, backends=backends,
                  server_context=server_context, cancelled=threading.Event())

    def on_signal(signum, _frame) -> None:
        log(f"signal {signum}: stopping every episode (run_arm closes its sessions and writes its manifest)")
        ctx.cancelled.set()

    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    log(f"group {tid}: {episode_label(stub)}/{stub['win_levels']} x {args.samples}, budget {budget} tokens, "
        f"cap {stub.get('action_cap')} actions, backends {distinct}")
    episodes = [Episode(tid, k) for k in range(args.samples)]
    results, failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=cfg["concurrency"]) as pool:
        futures = {pool.submit(run_episode, ctx, ep): ep for ep in episodes}
        for future in concurrent.futures.as_completed(futures):
            ep = futures[future]
            try:
                results.append(future.result())
            except Exception as error:  # noqa: BLE001 — record every episode failure; the whole group then ends as failed
                traceback.print_exc()
                failures.append({"task": ep.name, "error": repr(error)})
                with (run_dir / "failures.jsonl").open("a") as f:
                    f.write(json.dumps(failures[-1]) + "\n")
    servers = [preflight.check_server_end(s) for s in servers]
    manifest = tito_module("manifest")
    written = manifest.write(run_dir / manifest.FILE, harness="arc3", config=manifest.config_identity(args.config),
                             server=servers[0], **({"servers": servers} if len(servers) > 1 else {}),
                             items=tito_items(results, failures))
    summary = {"tid": tid, "game": stub["game"], "level": stub.get("level"), "samples": args.samples,
               "finished": len(results), "failed": len(failures),
               "reward_mean": round(sum(r["score"] for r in results) / len(results), 4) if results else None,
               "stops": {s: sum(r["stop"] == s for r in results) for s in sorted({r["stop"] for r in results})},
               "forbidden": sum(r["usage"]["proxy_forbidden"] for r in results), "tito": written["counts"]}
    atomic_write(run_dir / "summary.json", json.dumps(summary, indent=1) + "\n")
    log("SUMMARY " + json.dumps(summary))
    if group_failed(len(failures), args.samples):
        raise SystemExit(f"{len(failures)} episode(s) failed (> {MAX_FAILED_FRACTION:.0%} of {args.samples}: systemic, "
                         f"the group fails); see {run_dir / 'failures.jsonl'}")
    if failures:
        # one or two broken games don't sink the whole group: in the manifest they are status failed (no thread, no
        # reward), prepare_rl drops them one by one as harness_failed, the rest trains as usual (2026-10-24 v4's
        # g000000/g000006 were both whole groups with 31/32 good that got quarantined)
        log(f"EPISODE FAILURES {len(failures)}/{args.samples}: kept out of the manifest's threads (prepare drops them as "
            f"harness_failed), the group goes on; see {run_dir / 'failures.jsonl'}")


# --------------------------------------------------------------------------- selftest

PROBE = r'''
import json, os, socket, sys, urllib.error, urllib.request
spec = json.loads(sys.argv[1])
out = {}
try:
    socket.getaddrinfo("github.com", 443)
    out["dns"] = "resolved"
except OSError as e:
    out["dns"] = f"blocked: {e}"
try:
    socket.create_connection(("140.82.112.3", 443), timeout=3).close()
    out["tcp"] = "connected"
except OSError as e:
    out["tcp"] = f"blocked: {e}"
base = "http://127.0.0.1:%s" % spec["model_port"]
out["routes"] = {}
for method, path, body in spec["routes"]:
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            out["routes"][f"{method} {path} {bool(body and body.get('session_id'))}"] = r.status
    except urllib.error.HTTPError as e:
        out["routes"][f"{method} {path} {bool(body and body.get('session_id'))}"] = e.code
out["files"] = {p: os.path.exists(p) for p in spec["files"]}
print("PROBE " + json.dumps(out), flush=True)
'''


class _Recorder(http.server.BaseHTTPRequestHandler):
    """Fake model upstream: records the (method, path) it receives, always answers 200 JSON (non-streaming)."""
    seen: list = []

    def log_message(self, *a):  # noqa: D102
        pass

    def _answer(self):
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        type(self).seen.append((self.command, self.path.split("?")[0]))
        raw = b'{"object":"list","data":[]}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    do_GET = do_POST = _answer


class _EnvStub(http.server.BaseHTTPRequestHandler):
    """Fake middleware: jail.sh's health check hits /api/games (with this game's key)."""
    token = ""

    def log_message(self, *a):  # noqa: D102
        pass

    def do_GET(self):  # noqa: N802
        ok = self.path.startswith("/api/games") and self.headers.get("X-API-Key") == self.token
        raw = b"[]" if ok else b'{"error":"VALIDATION_ERROR"}'
        self.send_response(200 if ok else 401)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


FORBIDDEN_ROUTES = [("GET", "/get_server_info", None), ("POST", "/update_weights_from_disk", {"model_path": "x"}),
                    ("POST", "/flush_cache", {}), ("GET", "/session_ledger/status", None),
                    ("POST", "/abort_request", {"rid": "x"}), ("GET", "/metrics", None),
                    ("POST", "/v1/chat/completions", {"model": "x"}),
                    ("POST", "/v1/responses", {"model": "x", "input": []})]           # no session_id
ALLOWED_ROUTES = [("POST", "/v1/responses", {"model": "x", "input": [], "session_id": "selftest:0"}),
                  ("GET", "/v1/models", None)]


def selftest(args: argparse.Namespace, cfg: dict) -> None:
    """Anti-cheat controls: with **the same jail.sh and the same proxy settings** as an episode, run a probe inside the
    jail (no model, no GPU).

    Positive controls (inside the jail, all must be blocked): external DNS/TCP, sglang admin routes on the model port,
    /responses without session_id, files such as the corpus / repo / harness venv / host-side stop directory. The two
    allowed routes must work, and the fake upstream must receive **only** them.
    Negative controls (the same probe on the host, proxy without allowlist): files visible, admin routes forwarded --
    proving the probe really measures what it is meant to block.
    Raise on any mismatch."""
    arm = cfg["_arm"]
    root = args.run_dir
    root.mkdir(parents=True)
    ws, sockets, stop_dir = root / "ws", root / "sockets", root / "stop"
    for d in (ws, ws / ".driver", sockets, stop_dir):
        d.mkdir(parents=True)
    (ws / "probe.py").write_text(PROBE)
    act = root / "act.py"
    act.write_text("# selftest: jail.sh needs ARCSEC_ACT to be a file\n")
    env_file = root / "driver.env"
    env_file.write_text("")
    token = "selftest-token"
    _EnvStub.token = token
    _Recorder.seen = []
    servers = []

    def serve(handler):
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        servers.append(httpd)
        return httpd.server_address[1]

    upstream_port, env_port = serve(_Recorder), serve(_EnvStub)
    bridge = load_src("bridge", SRC / "arc_code" / "bridge.py")

    def proxy_for(routes, label):
        proxy = model_proxy.TaskProxy(label, 100000, ("127.0.0.1", upstream_port), root / f"proxy_{label}.jsonl",
                                      cfg["model_context_window"], routes=routes,
                                      require_session=routes is not None, stop_dir=stop_dir)
        server, port = model_proxy.serve_tcp(proxy)
        servers.append(server)
        return port

    strict_port, open_port = proxy_for(model_proxy.CODEX_ROUTES, "strict"), proxy_for(None, "open")
    for name, port in (("env", env_port), ("model", strict_port)):
        listener = bridge.listen_unix(sockets / f"{name}.sock")
        threading.Thread(target=bridge.serve, args=(listener, lambda p=port: socket.create_connection(("127.0.0.1", p), 30)),
                         daemon=True).start()
    corpus = expand(cfg["corpus"])
    # the first 7 are always visible on the host (the negative control requires each): game source, corpus metadata
    # (baseline), repo, harness venv, host-side stop directory, proxy log. The rest only have to be invisible in the jail.
    # The prior-context seeds' replay tree (arc3_trace_replay: every seed, the replayed whole-game records and the later
    # levels) must be invisible too
    files = [str(next(corpus.rglob("*.py"))), str(next(corpus.rglob("metadata.json"))), str(ROOT / "README.md"),
             str(ROOT / "work" / "arc_agi3"), str(expand(cfg["gateway_python"])), str(stop_dir),
             str(root / "proxy_strict.jsonl"), "/nfs", str(Path.home() / ".ssh"), str(ROOT / "reference"),
             str(expand("$ARC_RUNS/arc3_trace_replay"))]
    spec = {"model_port": arm["model_port"], "routes": FORBIDDEN_ROUTES + ALLOWED_ROUTES, "files": files}
    jail_env = {"PATH": "/usr/bin:/bin", "ARCSEC_RUNTIME": str(Path(args.agent_python).parents[1]),
                "ARCSEC_ACT": str(act), "ARCSEC_SOCKETS": str(sockets),
                "ARCSEC_BRIDGE": str(SRC / "arc_code" / "bridge.py"), "ARCSEC_DRIVER_BIN": str(args.driver_bin),
                "ARCSEC_DRIVER_HOME": str(ws / ".driver"), "ARCSEC_DRIVER_ENV": str(env_file), "ARCSEC_EXTRA_RO": "",
                "ARCSEC_TOKEN": token, "ARCSEC_PROCESS_DATA_KB": str(6 * 1024 * 1024),
                "ARCSEC_ENV_PORT": str(arm["env_port"]), "ARCSEC_MODEL_PORT": str(arm["model_port"])}
    proc = subprocess.run(["/bin/bash", str(UNIT / "jail.sh"), str(ws), str(args.agent_python), "-I",
                           str(ws / "probe.py"), json.dumps(spec)],
                          env=jail_env, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=180)
    (root / "jail.log").write_text(proc.stdout + proc.stderr)
    if proc.returncode != 0:
        raise SystemExit(f"selftest: jail exited {proc.returncode}; see {root / 'jail.log'}")
    jailed = json.loads(next(l for l in proc.stdout.splitlines() if l.startswith("PROBE "))[6:])
    seen_jailed = list(_Recorder.seen)
    # negative control: on the host, proxy without allowlist
    _Recorder.seen = []
    host = subprocess.run([sys.executable, "-I", "-c", PROBE, json.dumps({**spec, "model_port": open_port})],
                          capture_output=True, text=True, timeout=180)
    control = json.loads(next(l for l in host.stdout.splitlines() if l.startswith("PROBE "))[6:])
    for httpd in servers:
        httpd.shutdown()
    problems = []
    if not jailed["dns"].startswith("blocked") or not jailed["tcp"].startswith("blocked"):
        problems.append(f"network reachable from the jail: dns {jailed['dns']}, tcp {jailed['tcp']}")
    for method, path, body in FORBIDDEN_ROUTES:
        key = f"{method} {path} {bool(body and body.get('session_id'))}"
        if jailed["routes"][key] != 403:
            problems.append(f"{key} -> {jailed['routes'][key]} through the episode proxy, want 403")
    for method, path, body in ALLOWED_ROUTES:
        key = f"{method} {path} {bool(body and body.get('session_id'))}"
        if jailed["routes"][key] != 200:
            problems.append(f"{key} -> {jailed['routes'][key]}, want 200 (the allowed path must work)")
    if sorted(seen_jailed) != sorted((m, p) for m, p, _ in ALLOWED_ROUTES):
        problems.append(f"the upstream saw {seen_jailed}, want only the allowed routes")
    visible = [p for p, there in jailed["files"].items() if there]
    if visible:
        problems.append(f"visible inside the jail: {visible}")
    # negative control: the probe can measure what it is meant to block
    if not all(control["files"][p] for p in files[:7]):
        problems.append(f"control: the probe could not see host files it should: {control['files']}")
    if not any(code == 200 for key, code in control["routes"].items() if "update_weights" in key):
        problems.append(f"control: an open proxy did not forward the admin route: {control['routes']}")
    report = {"jailed": jailed, "upstream_seen_jailed": seen_jailed, "control": control, "problems": problems}
    atomic_write(root / "selftest.json", json.dumps(report, indent=1) + "\n")
    if problems:
        raise SystemExit("selftest FAILED:\n  " + "\n  ".join(problems))
    log(f"selftest passed: {root / 'selftest.json'}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--agent-python", type=Path, required=True)
    p.add_argument("--driver-bin", type=Path, required=True)
    p.add_argument("--selftest", action="store_true", help="anti-cheat positive/negative controls (no model run)")
    p.add_argument("--tasks-dir", type=Path)
    p.add_argument("--only")
    p.add_argument("--samples", type=int)
    p.add_argument("--salt-prefix")
    p.add_argument("--sample-backends")
    p.add_argument("--model-base-url")
    p.add_argument("--model")
    p.add_argument("--tokenizer-json", type=Path)
    args = p.parse_args(argv)
    cfg = load_config(args.config)
    if args.selftest:
        selftest(args, cfg)
        return
    for key in ("tasks_dir", "only", "samples", "model_base_url", "model", "tokenizer_json"):
        if getattr(args, key) is None:
            p.error(f"--{key.replace('_', '-')} is required (not --selftest)")
    rc = 1                                          # any exception other than SystemExit (incl. after a signal stop) gives 1
    try:
        run_group(args, cfg)
        rc = 0
    except SystemExit as error:
        rc = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
        raise
    finally:
        # when the orchestrator takes over an in-flight group, completion is judged only by this file (the process is not
        # its child, so it can't get the exit code); SIGKILL can't write it: process gone and no EXIT.json = crash
        args.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write(args.run_dir / EXIT_FILE, json.dumps({"rc": rc, "finished_unix": time.time()}) + "\n")


if __name__ == "__main__":
    main()
