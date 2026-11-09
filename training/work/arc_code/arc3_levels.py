#!/usr/bin/env python3
"""Task stubs and start-point verification for single-level episodes (run_arc3_level.py). Needs arc_agi's python (harness venv).

stub: ``<out>/<game>_L<k>.json``, read by run_arc3_level.py and the orchestrator's task lists. The start frame's
``start_hash`` is computed by starting the game with **the same official app + the same start_level hook** (Flask test
client, no server); middleware compares it at game start.

    <harness venv>/bin/python work/arc_code/arc3_levels.py stub --corpus $ARC_ARC3_CORPUS \\
        --levels ft09:5,tn36:1 --action-cap-factor 5 --out $ARC_DATA/arc3_levels/public25

verify: from the ``set_level`` start point, replay level k's actions (RESET included) from a real game record; at every
step the last frame must equal the recorded ``record/steps/N/final.txt``, and the level must be cleared on the same step.
Comparing only the start frame is not enough: some games carry state in ``next_level``/``step`` that a direct
``set_level`` misses. Records come from run_arm's game tarballs (``arm-run/games/<game>.tar.gz``, containing record/).
Comparing only the last frame is not strict enough (sc25 L3 passes anyway): the start-point gate for seeds is
``trace_seeds.py gate`` (``_gate``, which also compares frame counts and every frame).

    <harness venv>/bin/python work/arc_code/arc3_levels.py verify --corpus $ARC_ARC3_CORPUS \\
        --levels ft09:5,tn36:1 --games-dir <arm-run/games> --out <report.jsonl>

quota: after the levels are chosen, copy the prescreen stubs to another directory and add each level's output quota
(``output_token_budget``, read by run_arc3_level.py). The prescreen stubs stay untouched and the training conditions are
a separate copy; files already in the directory are not overwritten. Standard library only.

    python3 work/arc_code/arc3_levels.py quota --from $ARC_DATA/arc3_levels/public25 \
        --quotas tn36_L1:32000,ls20_L1:24000 --out $ARC_DATA/arc3_levels/<training stub dir>

recap: change the action cap of a whole stub directory (v3.3: 1.2 × baseline; a clear with an official score below 75
is 0 in the reward, so playing longer only wastes rollouts). Each ``<tid>.json`` gets ``action_cap`` =
``ceil(factor × baseline_actions)`` with the other fields unchanged; ``lists/`` and other files are copied as is;
``lists/recap.json`` records the source and factor (not at the root: configure_b200.py treats every .json at the root
as a stub). If the directory exists, nothing is written. Standard library only.

    python3 work/arc_code/arc3_levels.py recap --from $ARC_DATA/arc3_levels/round2-x2-20261020 --factor 1.2 \
        --out $ARC_DATA/arc3_levels/round2-x2-cap1.2-20261022

seed: round 2's tasks "with prior context" (game, level K, seed). From the trace-replay seed index (``trace_seeds.py
index``), pick the seeds that are index ok **and** pass the start-point gate (``trace_seeds.py gate``), copy the base
stub (made by ``stub``) to ``<out>/<game>_L<kk>_<label>.json`` and add the ``seed*`` fields (the contract of
``src/arcforge/arc_code/level_seed.py``). ``--with-l1`` also copies each game's ``<game>_L1.json`` as is, so the same
tasks directory holds L1 and the seeded levels. Existing files are not overwritten; each run's selection is appended
to ``<out>/seed_selection.jsonl``. Standard library + src/arcforge only.

    python3 work/arc_code/arc3_levels.py seed --index <batch>/index.jsonl --gate <batch>/gate.jsonl \\
        --base-stubs $ARC_DATA/arc3_levels/public25-all --out <stub dir> --families v00 --per-level 1 \\
        --exclude-games sc25 --exclude-seeds <affected_seeds.json> --with-l1

``--families v00,fp8 --per-level 2 --balance-families``: one v00 and one fp8 per level (a level with seeds from only
one family takes both from that family); without ``--balance-families`` seeds are taken in label alphabetical order,
and fp8's labels sort before v00.

lists: stub directory → task lists. ``train.txt`` = the training games' L1 plus seeded levels from level 2 to the
second-to-last; ``e1.txt`` = seeded levels of the training games' last level; ``e2.txt`` = every level of the held-out
games (L1 included); ``heldout.txt`` = e1 ∪ e2; ``lists.json`` records counts and gaps. Standard library only.

    python3 work/arc_code/arc3_levels.py lists --stubs <stub dir> --heldout-games r11l,s5i5,su15,vc33,g50t \\
        --out <stub dir>/lists

start_level.install works only once per process (it patches a class), so every level runs in a subprocess
(``_stub``/``_verify``/``_gate``; ``_gate`` is called by ``trace_seeds.py gate``).
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

UNIT = Path(__file__).resolve().parent
ROOT = UNIT.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from arcforge.arc_code import level_seed  # noqa: E402

STUB_SCHEMA = 1
KEY = "arc3-levels"
#: fields written by ``stub`` (the base stub); ``seed`` adds SEED_KEYS on top
BASE_STUB_KEYS = {"schema_version", "game", "game_id", "level", "win_levels", "baseline_actions", "action_cap",
                  "start_hash", "game_sha"}
SEED_KEYS = {"seed", "seed_sha256", "seed_label", "seed_family", "seed_actions", "seed_prefix", "seed_prompts"}
SEED_ID = re.compile(r"^(?P<label>[^/]+)/(?P<game>[a-z0-9]+)/L(?P<level>\d{2})$")
#: per-step comparison rules of the start-point gate (:func:`gate_step`)
FRAME_RULES = ("exact", "final", "reset")


def parse_levels(text: str) -> list[tuple[str, int]]:
    out = []
    for part in text.split(","):
        game, _, level = part.partition(":")
        if not game or not level.isdigit() or int(level) < 1:
            raise SystemExit(f"--levels wants game:level[,game:level...], got {part!r}")
        out.append((game, int(level)))
    if len(set(out)) != len(out):
        raise SystemExit(f"--levels has duplicates: {text}")
    return out


def game_dir(corpus: Path, game: str) -> Path:
    versions = sorted(p for p in (corpus / game).iterdir() if p.is_dir())
    if len(versions) != 1:
        raise SystemExit(f"{corpus / game}: want exactly one version, found {[v.name for v in versions]}")
    return versions[0]


def open_level(corpus: Path, game: str, level: int):
    """Official app (competition mode, as in run_arm) + start_level hook, with level ``level`` opened. Returns
    (client, game_id, guid, start response)."""
    sys.path.insert(0, str(ROOT / "src"))
    os.environ["ONLY_RESET_LEVELS"] = "true"
    from arcforge.arc_code import start_level
    start_level.install(game, level)
    from arc_agi import Arcade, OperationMode
    from arc_agi.server import create_app
    arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(corpus))
    app, _ = create_app(arcade, competition_mode=True, include_frame_data=True)
    client = app.test_client()
    headers = {"X-API-Key": KEY}
    ids = [g["game_id"] for g in client.get("/api/games", headers=headers).get_json()]
    full = [g for g in ids if g.split("-")[0] == game]
    if len(full) != 1:
        raise SystemExit(f"{game}: {len(full)} matching games in the corpus")
    card = client.post("/api/scorecard/open", json={"tags": [KEY]}, headers=headers).get_json()["card_id"]
    obs = client.post("/api/cmd/RESET", json={"game_id": full[0], "card_id": card}, headers=headers).get_json()
    if obs["levels_completed"] != level - 1 or obs["state"] != "NOT_FINISHED":
        raise SystemExit(f"{game} L{level}: start shows levels_completed {obs['levels_completed']} state {obs['state']}")
    return client, full[0], obs["guid"], obs


def frame_hash(frame) -> str:
    sys.path.insert(0, str(ROOT / "src"))
    from arcforge.arc_code.level_episode import frame_hash as fh
    return fh(frame)


def one_stub(corpus: Path, game: str, level: int, factor: int, out: Path) -> None:
    _, gid, _, obs = open_level(corpus, game, level)
    gdir = game_dir(corpus, game)
    meta = json.loads((gdir / "metadata.json").read_text())
    if meta["game_id"] != gid:
        raise SystemExit(f"{gdir}/metadata.json is {meta['game_id']}, the app serves {gid}")
    baseline = meta["baseline_actions"][level - 1]
    h = hashlib.sha256()
    for name in sorted(p.name for p in gdir.iterdir() if p.is_file()):
        h.update(name.encode()), h.update(b"\0"), h.update((gdir / name).read_bytes()), h.update(b"\0")
    stub = {"schema_version": STUB_SCHEMA, "game": game, "game_id": gid, "level": level,
            "win_levels": obs["win_levels"], "baseline_actions": baseline, "action_cap": factor * baseline,
            "start_hash": frame_hash(obs["frame"]), "game_sha": h.hexdigest()[:16]}
    path = out / f"{game}_L{level}.json"
    if path.exists():
        raise SystemExit(f"{path} exists; stubs are not overwritten")
    path.write_text(json.dumps(stub, indent=1) + "\n")
    print(json.dumps(stub))


def write_quotas(src: Path, spec: str, out: Path) -> None:
    """``spec`` = ``<tid>:<tokens>[,...]``: ``src/<tid>.json`` plus ``output_token_budget``, written to
    ``out/<tid>.json``."""
    quotas = {}
    for part in spec.split(","):
        tid, _, tokens = part.partition(":")
        if not tid or not tokens.isdigit() or int(tokens) < 1 or tid in quotas:
            raise SystemExit(f"--quotas wants <tid>:<tokens>[,...] with distinct tids, got {part!r}")
        quotas[tid] = int(tokens)
    out.mkdir(parents=True, exist_ok=True)
    for tid, tokens in quotas.items():
        stub = json.loads((src / f"{tid}.json").read_text())
        if "output_token_budget" in stub:
            raise SystemExit(f"{src / f'{tid}.json'} already has a quota; copy from the prescreen stubs")
        path = out / f"{tid}.json"
        if path.exists():
            raise SystemExit(f"{path} exists; stubs are not overwritten")
        path.write_text(json.dumps({**stub, "output_token_budget": tokens}, indent=1) + "\n")
        print(f"{path}: {tokens} tokens")


def board(frame) -> list[str]:
    """The response's last frame → 64 hex lines in the format of record's final.txt."""
    return ["".join(f"{v:x}" for v in row) for row in frame[-1]]


def recorded_level(tar: tarfile.TarFile, level: int) -> tuple[list[dict], dict[int, list[str]]]:
    """The action rows sent on level ``level`` in the record (history.jsonl's ``level`` field), and each step's
    final.txt."""
    names = {Path(m.name).relative_to(Path(m.name).parts[0]).as_posix(): m for m in tar.getmembers()
             if len(Path(m.name).parts) > 1}
    history = [json.loads(line) for line in tar.extractfile(names["record/history.jsonl"]).read().splitlines()]
    rows = [r for r in history if r["level"] == level and r["action"] != "START"]
    boards = {}
    for r in rows:
        boards[r["step"]] = tar.extractfile(names[f"record/steps/{r['step']:06d}/final.txt"]).read().decode().split()
    return rows, boards


def one_verify(corpus: Path, game: str, level: int, tarball: Path) -> None:
    client, gid, guid, obs = open_level(corpus, game, level)
    with tarfile.open(tarball) as tar:
        rows, boards = recorded_level(tar, level)
    if not rows or rows[-1]["levels_completed"] < level:
        raise SystemExit(f"{tarball}: level {level} is not cleared in this record")
    headers = {"X-API-Key": KEY}
    mismatches, cleared_at = [], None
    for i, r in enumerate(rows):
        body = {"game_id": gid, "guid": guid, **({"x": r["x"], "y": r["y"]} if "x" in r else {})}
        o = client.post(f"/api/cmd/{r['action']}", json=body, headers=headers).get_json()
        if "frame" not in o:
            raise SystemExit(f"{game} L{level} step {r['step']} {r['action']}: {o}")
        if board(o["frame"]) != boards[r["step"]]:
            mismatches.append(r["step"])
        if cleared_at is None and o["levels_completed"] >= level:
            cleared_at = i
    report = {"game": game, "level": level, "tarball": str(tarball), "actions": len(rows),
              "cleared_at": cleared_at, "recorded_clear_at": len(rows) - 1, "board_mismatch_steps": mismatches,
              "ok": not mismatches and cleared_at == len(rows) - 1}
    print("VERIFY " + json.dumps(report))


def find_tarball(games_dir: Path, game: str, level: int) -> Path | None:
    """The first tarball (by file name) holding a record that clears this level."""
    for tarball in sorted(games_dir.glob(f"{game}-*.tar.gz")):
        with tarfile.open(tarball) as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("record/history.jsonl"))
            last = None
            for line in tar.extractfile(member).read().splitlines():
                row = json.loads(line)
                if row["level"] == level:
                    last = row
            if last is not None and last["levels_completed"] >= level:
                return tarball
    return None


# --------------------------------------------------------------------------- start-point gate (_gate)

def seed_paths(ws: Path, label: str, game: str, level: int) -> tuple[Path, Path]:
    """Seed workspace ``…/replays/<label>/<game>/seeds/L<kk>/ws`` → (the replay's workspace, the replay's full record).
    The replay's workspace (``replays/<label>/<game>/run/games/<game>``) is the path hard-coded in the seed's files (the
    stub's ``seed_prefix``); record is that game's whole-game record (the actions actually played after the seed are in
    it too). Raise if the shape is wrong: both paths are derived from the shape."""
    ws = Path(ws)
    want = ("replays", label, game, "seeds", f"L{level:02d}", "ws")
    if not ws.is_absolute() or os.path.normpath(ws) != str(ws) or ws.parts[-6:] != want:
        raise SystemExit(f"{ws}: not an absolute normalized seed workspace .../{'/'.join(want)}")
    replay_ws = ws.parents[2] / "run" / "games" / game
    return replay_ws, replay_ws / "record"


def _cells(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))


def gate_step(rec: dict, rec_frames: list[str], eng: dict, eng_frames: list[str], *,
              rule: str) -> tuple[list[tuple[str, str]], bool]:
    """One step of the start-point gate (pure function): the replay record's row + its frames vs the engine's response
    to the same action from the ``set_level`` start point. Frames are board text in the format of
    ``record/steps/N/final.txt`` (64 hex lines), one string per frame; a 0-frame step has no board on either side.

    ``rule``: ``exact`` = frame count and every frame identical; ``final`` (games with ``--frames-nondeterministic``;
    lf52's dissolve transition uses an unseeded shuffle) = same frame count and same last frame, intermediate frames may
    differ; ``reset`` (level K's first action is RESET, and the official API returns the previous observation: 2 frames
    of the clearing step in the whole game, 1 frame at the ``set_level`` start) = compare only the last frame. All three
    compare ``levels_completed``, ``state`` and ``available_actions``.

    Returns (mismatching ``(check, reason)`` pairs, whether this step's frame difference was let through by the rule)."""
    if rule not in FRAME_RULES:
        raise ValueError(f"rule {rule!r} not in {FRAME_RULES}")
    bad = []
    for key in ("levels_completed", "state", "available_actions"):
        if rec[key] != eng[key]:
            bad.append((key, f"record {rec[key]!r}, engine {eng[key]!r}"))
    rec_final = rec_frames[-1] if rec_frames else None
    eng_final = eng_frames[-1] if eng_frames else None
    if rec_final != eng_final:
        bad.append(("final", "record has no frame" if rec_final is None else "engine has no frame"
                    if eng_final is None else f"{_cells(rec_final, eng_final)} cells differ"))
    same_count = len(rec_frames) == len(eng_frames)
    differing = [i for i, (a, b) in enumerate(zip(rec_frames, eng_frames)) if a != b]
    if rule != "reset" and not same_count:
        bad.append(("frame_count", f"record {len(rec_frames)}, engine {len(eng_frames)}"))
    if rule == "exact" and same_count:
        inner = [i for i in differing if i != len(rec_frames) - 1]
        if inner:
            bad.append(("frames", f"frames {inner[:8]} of {len(rec_frames)} differ"))
    exact = same_count and not differing
    masked = not exact and not any(check in ("final", "frame_count", "frames") for check, _ in bad)
    return bad, masked


def frames_text(frames) -> list[str]:
    """Frames (the official response's ``frame`` or frames.npz's uint8 (F, H, W)) → one board text per frame (record's
    final.txt format)."""
    import numpy as np
    arr = np.asarray(frames, dtype=np.uint8)
    if arr.size == 0:
        return []
    if arr.ndim != 3:
        raise SystemExit(f"frames have shape {arr.shape}, want (frames, height, width)")
    hexed = np.frombuffer(b"0123456789abcdef", dtype=np.uint8)[arr]
    newline = np.full(arr.shape[:2] + (1,), ord("\n"), dtype=np.uint8)
    return [f.tobytes().decode() for f in np.concatenate([hexed, newline], axis=2)]


def recorded_frames(record: Path, row: dict) -> list[str]:
    """All frames of one record step (frames.npz), also checking that the record is self-consistent: frame count =
    history's ``frames``, final.txt = the last frame (a 0-frame step has no final.txt). An inconsistency means the record
    is broken, not that the seed failed: raise."""
    import numpy as np
    step_dir = record / "steps" / f"{row['step']:06d}"
    with np.load(step_dir / "frames.npz") as data:
        frames = frames_text(data["frames"])
    if len(frames) != row["frames"]:
        raise SystemExit(f"{step_dir}: frames.npz has {len(frames)} frames, history says {row['frames']}")
    final = step_dir / "final.txt"
    if frames and final.read_text() != frames[-1]:
        raise SystemExit(f"{step_dir}: final.txt is not the last frame of frames.npz")
    if not frames and final.exists():
        raise SystemExit(f"{step_dir}: a 0-frame step with a final.txt")
    return frames


def one_gate(corpus: Path, ws: Path, label: str, game: str, level: int, actions: int,
             frames_nondeterministic: bool, record: Path | None = None) -> dict:
    """Start-point gate for one seed. (a) Start: ``open_level``'s last frame = the seed's ``record/current.txt``,
    ``levels_completed`` = K−1 and not finished, ``available_actions``/``win_levels`` = the seed's last row; the seed's
    record itself meets the contract (``level_seed.record_tail``); the seed's history is a prefix of the replayed
    whole-game history. (b) Dynamics: replay from the start point the level-K actions the replay actually played after the
    seed (rows of the whole-game record after the seed's last step whose ``level`` is still K), comparing every step with
    :func:`gate_step`, and the level must be cleared on the same step. ``record`` = the whole-game record directory (for
    seeds snapshotted live by gen-2: ``<run>/games/<game>/record``); if not given, it is derived from the replay seed's
    path shape (:func:`seed_paths`)."""
    if record is None:
        _, record = seed_paths(ws, label, game, level)
    failed: list[dict] = []

    def fail(check: str, reason: str) -> None:
        failed.append({"check": check, "reason": reason})

    seed_lines = [line for line in (ws / "record" / "history.jsonl").read_bytes().split(b"\n") if line.strip()]
    last = json.loads(seed_lines[-1])
    try:
        level_seed.record_tail(ws, level, actions)
    except level_seed.SeedError as error:
        fail("seed_record", str(error))
    current = (ws / "record" / "current.txt").read_text()

    # levels_completed = K−1 and NOT_FINISHED are guaranteed by open_level (a mismatch = the start_level hook is broken,
    # a harness error: raise directly)
    client, gid, guid, obs = open_level(corpus, game, level)
    start = frames_text(obs["frame"])
    if not start or start[-1] != current:
        fail("start_board", "no start frame" if not start else
             f"{_cells(start[-1], current)} cells differ from record/current.txt")
    for key in ("available_actions", "win_levels"):
        if obs[key] != last[key]:
            fail(f"start_{key}", f"start {obs[key]!r}, seed's last row {last[key]!r}")

    full_lines = [line for line in (record / "history.jsonl").read_bytes().split(b"\n") if line.strip()]
    if full_lines[:len(seed_lines)] != seed_lines:
        fail("history_prefix", f"the seed's {len(seed_lines)} history lines are not a prefix of the replay's record")
    after = [row for row in map(json.loads, full_lines) if row["step"] > last["step"]]
    if after and after[0]["step"] != last["step"] + 1:
        raise SystemExit(f"{record}: step {after[0]['step']} follows the seed's last step {last['step']}")
    tail = list(itertools.takewhile(lambda row: row["level"] == level, after))
    if any(row["action"] == "START" for row in tail):
        raise SystemExit(f"{record}: a START row inside level {level}")
    # when the seed's last step is level K−1's clearing action, tail's first row is level K's first action (only it gets
    # the RESET exception)
    first_is_level_start = last["level"] == level - 1

    dyn: dict[str, list] = collections.defaultdict(list)
    excepted = collections.Counter()
    engine_clear, played = None, 0
    for i, row in enumerate(tail):
        body = {"game_id": gid, "guid": guid, **({"x": row["x"], "y": row["y"]} if "x" in row else {})}
        resp = client.post(f"/api/cmd/{row['action']}", json=body, headers={"X-API-Key": KEY})
        o = resp.get_json()
        if resp.status_code != 200 or not isinstance(o, dict) or "frame" not in o:
            fail("dyn_error", f"step {row['step']} {row['action']}: HTTP {resp.status_code} {str(o)[:200]}")
            break
        played += 1
        rule = ("reset" if i == 0 and first_is_level_start and row["action"] == "RESET"
                else "final" if frames_nondeterministic else "exact")
        bad, masked = gate_step(row, recorded_frames(record, row), o, frames_text(o["frame"]), rule=rule)
        if masked:
            excepted[{"reset": "first_reset", "final": "frames_nondeterministic"}[rule]] += 1
        what = row["action"] + (f"({row['x']},{row['y']})" if "x" in row else "")
        for check, reason in bad:
            dyn[check].append(f"step {row['step']} {what}: {reason}")
        if engine_clear is None and o["levels_completed"] >= level:
            engine_clear = row["step"]
    for check, items in dyn.items():
        fail(f"dyn_{check}", f"{len(items)} of {len(tail)} steps; first {items[0]}")
    record_clear = next((row["step"] for row in tail if row["levels_completed"] >= level), None)
    if engine_clear != record_clear:
        fail("dyn_clear_step", f"record clears at step {record_clear}, engine at {engine_clear}")
    return {"seed": f"{label}/{game}/L{level:02d}", "label": label, "game": game, "level": level, "ws": str(ws),
            "pass": not failed, "failed": failed, "dyn_actions": len(tail), "dyn_played": played,
            "excepted": dict(excepted), "record_cleared_at": record_clear, "engine_cleared_at": engine_clear,
            "frames_nondeterministic": frames_nondeterministic}


# --------------------------------------------------------------------------- seed: stubs for seeded levels

def read_jsonl(path: Path) -> list[dict]:
    """Split only on ``\\n`` (JSON strings may contain U+2028 and the like)."""
    lines = path.read_text().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


def exclude_ids(path: Path) -> set[str]:
    """``--exclude-seeds``: the audit's ``affected_seeds.json`` (takes the seeds in ``definite_content`` ∪
    ``likely_content``), a json list, or plain text with one seed id per line (lines starting with ``#`` are comments)."""
    text = path.read_text()
    if path.suffix == ".json":
        data = json.loads(text)
        if isinstance(data, dict):
            if not {"definite_content", "likely_content"} <= set(data):
                raise SystemExit(f"{path}: a json object needs definite_content and likely_content")
            ids = [*data["definite_content"], *data["likely_content"]]
        elif isinstance(data, list):
            ids = data
        else:
            raise SystemExit(f"{path}: want an object (affected_seeds.json) or a list of seed ids")
    else:
        ids = [line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")]
    bad = [i for i in ids if not (isinstance(i, str) and SEED_ID.match(i))]
    if bad:
        raise SystemExit(f"{path}: not seed ids (<label>/<game>/L<kk>): {bad[:5]}")
    return set(ids)


def load_base_stub(base: Path, game: str, level: int) -> dict:
    path = base / f"{game}_L{level}.json"
    stub = json.loads(path.read_text())
    if set(stub) != BASE_STUB_KEYS or stub["game"] != game or stub["level"] != level:
        raise SystemExit(f"{path}: not the base stub of {game} L{level} (keys {sorted(stub)})")
    return stub


def seed_stub(row: dict, base: Path) -> tuple[str, dict]:
    """One seed (an index row) → (tid, stub). stub = base stub + ``seed*``; the seed's current hash and record are
    verified by level_seed."""
    label, game, level = row["label"], row["game"], row["level"]
    tid = f"{game}_L{level:02d}_{label}"
    if "__" in tid or "/" in tid:
        raise SystemExit(f"{row['seed']}: tid {tid!r} must not contain '__' (prepare_rl splits on it) or '/'")
    stub = load_base_stub(base, game, level)
    if stub["win_levels"] != row["win_levels"]:
        raise SystemExit(f"{row['seed']}: win_levels {row['win_levels']} != base stub {stub['win_levels']}")
    ws = Path(row["ws"])
    prefix, _ = seed_paths(ws, label, game, level)
    actions = row["check"]["counted_before_replay"]
    spec = level_seed.SeedSpec(ws=ws, sha256=level_seed.tree_sha256(ws), prefix=str(prefix), actions=actions,
                               level=level, label=label)
    tail = level_seed.check(spec)
    if tail["step"] != actions:
        raise SystemExit(f"{row['seed']}: record tail step {tail['step']} != seed actions {actions}")
    family = row["source"].get("info", {}).get("family")
    if not family:
        raise SystemExit(f"{row['seed']}: the index row has no source.info.family")
    return tid, {**stub, "seed": str(ws), "seed_sha256": spec.sha256, "seed_label": label, "seed_family": family,
                 "seed_actions": actions, "seed_prefix": str(prefix), "seed_prompts": row["source"]["prompts"]}


def balanced(group: list[dict], families: list[str]) -> list[dict]:
    """Interleave one (game, level)'s candidates in ``families`` order: each family's first by label, then each
    family's second, ...; a family that runs out is skipped. ``per_level`` then takes the first N, so N=2 with
    families=v00,fp8 gives one v00 and one fp8 (if only one family has seeds, both come from that family)."""
    by_family = {f: [r for r in group if r["source"]["info"]["family"] == f] for f in families}
    stray = [r["seed"] for r in group if r["source"]["info"]["family"] not in by_family]
    if stray:
        raise SystemExit(f"--balance-families: {stray[:3]} are of a family not in --families {families}")
    order = []
    for i in range(max(len(v) for v in by_family.values())):
        order += [v[i] for v in by_family.values() if i < len(v)]
    return order


def select_seeds(rows: list[dict], gates: dict[str, dict], *, families=None, labels=None, games=None,
                 exclude_games=frozenset(), exclude_seeds=frozenset(), per_level=None,
                 balance_families=False) -> tuple[list[dict], dict]:
    """Pick seeds: in scope (family/label/game), not excluded, index ok, gate passed; ``per_level`` takes the first N
    by label for each (game, level) (``balance_families``: families take turns in ``families`` order, see
    :func:`balanced`). Returns (chosen rows, skipped {reason: [seed ids]}). Raises for an in-scope, index-ok seed with
    no gate result."""
    if balance_families and not (families and len(families) >= 2 and per_level is not None):
        raise SystemExit("--balance-families needs --families with two or more families and --per-level")
    for name, wanted, have in (("family", families, {r["source"]["info"]["family"] for r in rows}),
                               ("label", labels, {r["label"] for r in rows}),
                               ("game", games, {r["game"] for r in rows}),
                               ("excluded game", exclude_games, {r["game"] for r in rows})):
        if wanted and not set(wanted) <= have:
            raise SystemExit(f"{name} {sorted(set(wanted) - have)} not in the index (have {sorted(have)})")
    by_id = {r["seed"]: r for r in rows}
    indexed_labels = {r["label"] for r in rows}
    stale = sorted(s for s in exclude_seeds if SEED_ID.match(s)["label"] in indexed_labels and s not in by_id)
    if stale:
        raise SystemExit(f"--exclude-seeds names seeds of indexed labels that are not in the index: {stale[:5]}")
    skipped: dict[str, list[str]] = collections.defaultdict(list)
    candidates = []
    for r in sorted(rows, key=lambda r: (r["game"], r["level"], r["label"])):
        if ((families and r["source"]["info"]["family"] not in families) or (labels and r["label"] not in labels)
                or (games and r["game"] not in games)):
            continue
        if r["game"] in exclude_games:
            skipped["excluded_game"].append(r["seed"])
        elif r["seed"] in exclude_seeds:
            skipped["excluded_seed"].append(r["seed"])
        elif not r["ok"]:
            skipped["index_not_ok"].append(r["seed"])
        elif r["seed"] not in gates:
            raise SystemExit(f"{r['seed']}: index ok but no gate result (run trace_seeds.py gate first)")
        elif gates[r["seed"]]["ws"] != r["ws"]:
            raise SystemExit(f"{r['seed']}: gate ws {gates[r['seed']]['ws']} != index ws {r['ws']}")
        elif not gates[r["seed"]]["pass"]:
            skipped["gate_failed"].append(r["seed"])
        else:
            candidates.append(r)
    chosen = []
    for _, group in itertools.groupby(candidates, key=lambda r: (r["game"], r["level"])):
        group = list(group)
        if balance_families:
            group = balanced(group, families)
        keep = group if per_level is None else group[:per_level]
        chosen += keep
        skipped["per_level"] += [r["seed"] for r in group[len(keep):]]
    return chosen, dict(skipped)


def write_seed_stubs(a) -> None:
    rows = [r for path in a.index for r in read_jsonl(path)]
    ids = [r["seed"] for r in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"--index files repeat seeds: {[s for s, n in collections.Counter(ids).items() if n > 1][:5]}")
    for r in rows:
        if not r["source"].get("info", {}).get("family"):
            raise SystemExit(f"{r['seed']}: the index row has no source.info.family")
    gates: dict[str, dict] = {}
    for path in a.gate:
        for g in read_jsonl(path):
            if g["seed"] in gates:
                raise SystemExit(f"{path}: {g['seed']} gated twice")
            gates[g["seed"]] = g
    excluded = set().union(*(exclude_ids(p) for p in a.exclude_seeds)) if a.exclude_seeds else set()
    if a.per_level is not None and a.per_level < 1:
        raise SystemExit("--per-level must be >= 1")
    chosen, skipped = select_seeds(rows, gates, families=a.families, labels=a.labels, games=a.games,
                                   exclude_games=a.exclude_games or frozenset(), exclude_seeds=excluded,
                                   per_level=a.per_level, balance_families=a.balance_families)
    stubs, seeds = {}, {}
    for r in chosen:
        tid, stub = seed_stub(r, a.base_stubs)
        if tid in stubs:
            raise SystemExit(f"{r['seed']} and {seeds[tid]} map to the same tid {tid}")
        stubs[tid], seeds[tid] = stub, r["seed"]
    l1 = {}
    if a.with_l1:
        scope = sorted({r["game"] for r in rows
                        if not (a.families and r["source"]["info"]["family"] not in a.families)
                        and not (a.labels and r["label"] not in a.labels)
                        and not (a.games and r["game"] not in a.games)
                        and r["game"] not in (a.exclude_games or ())})
        for game in scope:
            load_base_stub(a.base_stubs, game, 1)
            l1[f"{game}_L1"] = a.base_stubs / f"{game}_L1.json"
    a.out.mkdir(parents=True, exist_ok=True)
    exists = sorted(t for t in [*stubs, *l1] if (a.out / f"{t}.json").exists())
    if exists:
        raise SystemExit(f"{a.out}: {len(exists)} stubs exist ({exists[:5]}); stubs are not overwritten")
    for tid, stub in stubs.items():
        (a.out / f"{tid}.json").write_text(json.dumps(stub, indent=1) + "\n")
    for tid, src in l1.items():
        shutil.copyfile(src, a.out / f"{tid}.json")
    record = {"unix": time.time(), "argv": sys.argv[1:], "stubs": sorted(stubs), "l1": sorted(l1),
              "seeds": seeds,
              "skipped": skipped}
    with (a.out / "seed_selection.jsonl").open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    per_game = collections.defaultdict(list)
    for s in stubs.values():
        per_game[s["game"]].append(s["level"])
    print(f"{len(stubs)} seeded stubs + {len(l1)} L1 stubs -> {a.out}")
    for game in sorted(per_game):
        levels = collections.Counter(per_game[game])
        print(f"  {game}: " + " ".join(f"L{k}" + (f"x{n}" if n > 1 else "") for k, n in sorted(levels.items())))
    for reason, seeds in sorted(skipped.items()):
        print(f"skipped {reason}: {len(seeds)}" + (f" {seeds}" if reason != "per_level" else ""))


# --------------------------------------------------------------------------- lists: task lists

def read_stubs(stubs_dir: Path) -> dict[str, dict]:
    """Every ``*.json`` in the stub directory (tid → stub). File names must follow the tid rules: base stub
    ``<game>_L<k>``, seeded level ``<game>_L<kk>_<label>`` (carrying all SEED_KEYS)."""
    out = {}
    for path in sorted(stubs_dir.glob("*.json")):
        stub = json.loads(path.read_text())
        keys = set(stub) - {"output_token_budget"}     # the quota is added by ``quota``, optional
        if "seed" in stub:
            ok = (keys == BASE_STUB_KEYS | SEED_KEYS
                  and path.stem == f"{stub['game']}_L{stub['level']:02d}_{stub['seed_label']}")
        else:
            ok = keys == BASE_STUB_KEYS and path.stem == f"{stub['game']}_L{stub['level']}"
        if not ok:
            raise SystemExit(f"{path}: not a base or seeded stub named after its game/level/label (keys {sorted(keys)})")
        out[path.stem] = stub
    if not out:
        raise SystemExit(f"{stubs_dir}: no stubs")
    return out


def build_lists(stubs: dict[str, dict], heldout_games: set[str]) -> dict:
    """Set of tid → stub → task lists and gaps. train = training games' L1 (base stub) + seeded levels from level 2 to
    the second-to-last; e1 = seeded levels of the training games' last level; e2 = held-out games' L1 + seeded levels of
    every level; heldout = e1 ∪ e2."""
    win = {}
    for tid, s in stubs.items():
        if win.setdefault(s["game"], s["win_levels"]) != s["win_levels"]:
            raise SystemExit(f"{tid}: win_levels {s['win_levels']} != {win[s['game']]} of other {s['game']} stubs")
    missing = sorted(heldout_games - set(win))
    if missing:
        raise SystemExit(f"held-out games without any stub: {missing}")
    train, e1, e2, unused = [], [], [], []
    for tid, s in sorted(stubs.items()):
        seeded, k, last = "seed" in s, s["level"], s["level"] == win[s["game"]]
        if s["game"] in heldout_games:
            (e2 if seeded or k == 1 else unused).append(tid)
        elif seeded and last:
            e1.append(tid)
        elif (seeded and 2 <= k) or (not seeded and k == 1):
            train.append(tid)
        else:
            unused.append(tid)
    heldout = sorted(set(e1) | set(e2))
    for name, ids in (("train", train), ("e1", e1), ("e2", e2), ("heldout", heldout)):
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{name} has duplicates")
    if any(stubs[t]["game"] in heldout_games for t in train):
        raise SystemExit("a held-out game is in train")
    if any(stubs[t]["level"] == win[stubs[t]["game"]] for t in train):
        raise SystemExit("a last level is in train")
    if set(train) & set(heldout):
        raise SystemExit("train and heldout overlap")
    have = collections.defaultdict(set)
    for s in stubs.values():
        if "seed" in s or s["level"] == 1:
            have[s["game"]].add(s["level"])
    gaps = {"train": {}, "e1": [], "e2": {}}
    for game in sorted(win):
        want = range(1, win[game] + 1) if game in heldout_games else range(1, win[game])
        lack = [k for k in want if k not in have[game]]
        if game in heldout_games:
            if lack:
                gaps["e2"][game] = lack
        else:
            if lack:
                gaps["train"][game] = lack
            if win[game] not in have[game]:
                gaps["e1"].append(game)
    return {"train": train, "e1": e1, "e2": e2, "heldout": heldout, "unused": unused, "gaps": gaps,
            "games": {"train": sorted(set(win) - heldout_games), "heldout": sorted(heldout_games)},
            "win_levels": dict(sorted(win.items()))}


def write_lists(stubs_dir: Path, heldout_games: set[str], out: Path) -> dict:
    stubs = read_stubs(stubs_dir)
    got = build_lists(stubs, heldout_games)
    names = ("train.txt", "e1.txt", "e2.txt", "heldout.txt", "lists.json")
    out.mkdir(parents=True, exist_ok=True)
    exists = [n for n in names if (out / n).exists()]
    if exists:
        raise SystemExit(f"{out}: {exists} exist; lists are not overwritten")
    for key in ("train", "e1", "e2", "heldout"):
        for tid in got[key]:
            if not (stubs_dir / f"{tid}.json").is_file():
                raise SystemExit(f"{tid}: no stub file in {stubs_dir}")
        (out / f"{key}.txt").write_text("".join(t + "\n" for t in got[key]))
    summary = {"stubs_dir": str(stubs_dir), "heldout_games": sorted(heldout_games),
               "counts": {k: len(got[k]) for k in ("train", "e1", "e2", "heldout", "unused")},
               **{k: got[k] for k in ("games", "win_levels", "gaps", "unused")}}
    (out / "lists.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: summary[k] for k in ("counts", "gaps")}, indent=1))
    return summary


def write_game_stubs(corpus: Path, l1_dir: Path, games: list[str], heldout: list[str], budget: int | None,
                     out: Path) -> None:
    """Stubs for whole-game episodes (run_arc3_level.py's arc3_game): ``<game>.json``, with game/game_id/win_levels/
    game_sha copied from ``l1_dir``'s ``<game>_L1.json`` (whose game start was verified) and baseline_actions the whole
    list from the corpus metadata (one per level). ``lists/train.txt`` = games, ``lists/heldout.txt`` = heldout (the
    orchestrator needs a non-empty held-out list; stream doesn't evaluate it inside the loop)."""
    if not heldout or set(heldout) & set(games):
        raise SystemExit(f"held-out games {heldout} must be non-empty and apart from the training games {games}")
    out.mkdir(parents=True, exist_ok=False)
    for game in [*games, *heldout]:
        l1 = json.loads((l1_dir / f"{game}_L1.json").read_text())
        gdir = game_dir(corpus, game)
        meta = json.loads((gdir / "metadata.json").read_text())
        if meta["game_id"] != l1["game_id"]:
            raise SystemExit(f"{gdir}/metadata.json is {meta['game_id']}, {game}_L1.json is {l1['game_id']}")
        h = hashlib.sha256()
        for name in sorted(p.name for p in gdir.iterdir() if p.is_file()):
            h.update(name.encode()), h.update(b"\0"), h.update((gdir / name).read_bytes()), h.update(b"\0")
        if h.hexdigest()[:16] != l1["game_sha"]:
            raise SystemExit(f"{game}: corpus game_sha {h.hexdigest()[:16]} != {game}_L1.json {l1['game_sha']}")
        base = meta["baseline_actions"]
        if len(base) != l1["win_levels"] or base[0] != l1["baseline_actions"]:
            raise SystemExit(f"{game}: metadata baselines {base} vs L1 stub win_levels {l1['win_levels']} / "
                             f"baseline {l1['baseline_actions']}")
        stub = {"schema_version": STUB_SCHEMA, "kind": "game", "game": game, "game_id": l1["game_id"],
                "win_levels": l1["win_levels"], "baseline_actions": base, "game_sha": l1["game_sha"],
                **({} if budget is None else {"output_token_budget": budget})}
        (out / f"{game}.json").write_text(json.dumps(stub, indent=1) + "\n")
    (out / "lists").mkdir()
    (out / "lists" / "train.txt").write_text("".join(f"{g}\n" for g in games))
    (out / "lists" / "heldout.txt").write_text("".join(f"{g}\n" for g in heldout))
    print(f"{len(games)} + {len(heldout)} held-out game stubs -> {out}")



def write_g2seed_stubs(selection: Path, game_stubs: Path, cap_factor: float, budget: int | None, out: Path) -> None:
    """Stubs for gen-2 single-level episodes (run_arc3_level.py's arc3_g2seed). ``selection`` has one level-change seed
    per line: a row of the g2seed seed-store index (seed, game, kind level, level, actions, tree_sha256, prefix, ws, tag)
    plus ``role`` (t training / h held-out). Each one writes ``<game>_L<kk>_<tag>.json``: game/game_id/win_levels/
    game_sha and this level's baseline are copied from the whole-game stub ``<game>.json`` (made by ``games``),
    action_cap = ceil(cap_factor × baseline), seed = level_seed.SeedSpec (ws is the seed unpacked on the local machine;
    tree_sha256 is recomputed here and must equal the index's). ``lists/train.txt``/``lists/heldout.txt`` by role. If
    the directory exists, nothing is written."""
    rows = [json.loads(line) for line in selection.read_text().splitlines() if line.strip()]
    out.mkdir(parents=True, exist_ok=False)
    games: dict[str, dict] = {}
    lists: dict[str, list[str]] = {"t": [], "h": []}
    for row in rows:
        if row["kind"] != "level" or row["role"] not in lists or row["window"] is not None:
            raise SystemExit(f"{row['seed']}: kind {row['kind']} / role {row['role']} / window {row['window']}: want a "
                             "level seed with role t or h")
        game, level = row["game"], row["level"]
        if game not in games:
            games[game] = json.loads((game_stubs / f"{game}.json").read_text())
            if games[game]["kind"] != "game":
                raise SystemExit(f"{game_stubs}/{game}.json is not a whole-game stub")
        g = games[game]
        if not 2 <= level <= g["win_levels"]:
            raise SystemExit(f"{row['seed']}: level {level} of {g['win_levels']}")
        tid = f"{game}_L{level:02d}_{row['tag']}"
        if "__" in tid or "/" in tid or (out / f"{tid}.json").exists():
            raise SystemExit(f"{row['seed']}: tid {tid!r} has '__' or '/', or is taken by another seed")
        ws = Path(row["ws"])
        if level_seed.tree_sha256(ws) != row["tree_sha256"]:
            raise SystemExit(f"{row['seed']}: tree_sha256 of {ws} != the index's {row['tree_sha256']} (incomplete unpack?)")
        spec = level_seed.SeedSpec(ws=ws, sha256=row["tree_sha256"], prefix=row["prefix"], actions=row["actions"],
                                   level=level, label=row["seed"])
        base = g["baseline_actions"][level - 1]
        stub = {"schema_version": STUB_SCHEMA, "kind": "g2seed", "game": game, "game_id": g["game_id"], "level": level,
                "win_levels": g["win_levels"], "baseline_actions": base, "action_cap": math.ceil(cap_factor * base),
                "game_sha": g["game_sha"], "seed": spec.to_json(),
                **({} if budget is None else {"output_token_budget": budget})}
        (out / f"{tid}.json").write_text(json.dumps(stub, indent=1) + "\n")
        lists[row["role"]].append(tid)
    if not lists["t"] or not lists["h"]:
        raise SystemExit(f"need training and held-out seeds: {len(lists['t'])} / {len(lists['h'])}")
    (out / "lists").mkdir()
    (out / "lists" / "train.txt").write_text("".join(f"{t}\n" for t in sorted(lists["t"])))
    (out / "lists" / "heldout.txt").write_text("".join(f"{t}\n" for t in sorted(lists["h"])))
    print(f"{len(lists['t'])} training + {len(lists['h'])} held-out g2seed stubs -> {out}")


def write_recap(src: Path, factor: float, out: Path) -> None:
    """Write each stub of ``src`` to ``out`` with ``action_cap = ceil(factor × baseline_actions)``; non-stub files and
    ``lists/`` are copied as is."""
    if not (math.isfinite(factor) and factor > 0):
        raise SystemExit(f"--factor {factor}: want a finite number > 0")
    if out.exists():
        raise SystemExit(f"{out} exists; stub directories are not overwritten")
    stubs = sorted(src.glob("*.json"))
    if not stubs:
        raise SystemExit(f"{src}: no stubs")
    out.mkdir(parents=True)
    caps = {}
    for path in stubs:
        stub = json.loads(path.read_text())
        base = stub["baseline_actions"]
        if not isinstance(base, int) or isinstance(base, bool) or base < 1:
            raise SystemExit(f"{path}: baseline_actions {base!r}")
        caps[path.stem] = (stub["action_cap"], math.ceil(factor * base))
        (out / path.name).write_text(json.dumps({**stub, "action_cap": caps[path.stem][1]}, indent=1) + "\n")
    for path in sorted(src.iterdir()):
        if path.suffix == ".json" and path.is_file():
            continue
        if path.is_dir():
            shutil.copytree(path, out / path.name)
        else:
            shutil.copy2(path, out / path.name)
    (out / "lists").mkdir(exist_ok=True)
    (out / "lists" / "recap.json").write_text(json.dumps({"from": str(src.resolve()), "factor": factor,
                                                          "stubs": len(caps)}, indent=1) + "\n")
    print(f"{out}: {len(caps)} stubs, action_cap = ceil({factor} x baseline); e.g. "
          + ", ".join(f"{t} {a}->{b}" for t, (a, b) in list(caps.items())[:3]))


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stub")
    s.add_argument("--corpus", type=Path, required=True)
    s.add_argument("--levels", required=True)
    s.add_argument("--action-cap-factor", type=int, required=True)
    s.add_argument("--out", type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--corpus", type=Path, required=True)
    v.add_argument("--levels", required=True)
    v.add_argument("--games-dir", type=Path, required=True, help="run_arm's arm-run/games (one tar.gz per game)")
    v.add_argument("--out", type=Path, required=True, help="one jsonl line per level")
    one = sub.add_parser("_stub")
    one.add_argument("corpus", type=Path), one.add_argument("game"), one.add_argument("level", type=int)
    one.add_argument("factor", type=int), one.add_argument("out", type=Path)
    rc = sub.add_parser("recap")
    rc.add_argument("--from", dest="src", type=Path, required=True, help="source stub directory")
    rc.add_argument("--factor", type=float, required=True, help="action_cap = ceil(factor x baseline_actions)")
    rc.add_argument("--out", type=Path, required=True)
    q = sub.add_parser("quota")
    q.add_argument("--from", dest="src", type=Path, required=True, help="prescreen stub directory (no quota)")
    q.add_argument("--quotas", required=True, help="<tid>:<tokens>[,...]")
    q.add_argument("--out", type=Path, required=True)
    onev = sub.add_parser("_verify")
    onev.add_argument("corpus", type=Path), onev.add_argument("game"), onev.add_argument("level", type=int)
    onev.add_argument("tarball", type=Path)
    oneg = sub.add_parser("_gate", help="start-point gate for one seed (called by trace_seeds.py gate); prints one line "
                          "GATE <json>")
    oneg.add_argument("corpus", type=Path), oneg.add_argument("ws", type=Path), oneg.add_argument("label")
    oneg.add_argument("game"), oneg.add_argument("level", type=int), oneg.add_argument("actions", type=int)
    oneg.add_argument("--frames-nondeterministic", action="store_true")
    oneg.add_argument("--record", type=Path, help="whole-game record directory (seeds snapshotted live by gen-2); if not "
                      "given, derived from the replay seed's path")
    sd = sub.add_parser("seed")
    csv = lambda s: [x for x in s.split(",") if x]  # noqa: E731
    sd.add_argument("--index", type=Path, action="append", required=True, help="trace_seeds' index.jsonl (repeatable)")
    sd.add_argument("--gate", type=Path, action="append", required=True, help="trace_seeds' gate.jsonl (repeatable)")
    sd.add_argument("--base-stubs", type=Path, required=True, help="base stub for each level (made by stub)")
    sd.add_argument("--out", type=Path, required=True)
    sd.add_argument("--families", type=csv, help="source.info.family, comma-separated (v00, fp8)")
    sd.add_argument("--labels", type=csv)
    sd.add_argument("--games", type=csv)
    sd.add_argument("--exclude-games", type=csv)
    sd.add_argument("--exclude-seeds", type=Path, action="append",
                    help="affected_seeds.json (definite_content ∪ likely_content), a json list, or one id per line")
    sd.add_argument("--per-level", type=int, help="at most this many per (game, level), taken in label order")
    sd.add_argument("--balance-families", action="store_true",
                    help="--per-level alternates families in --families order (v00,fp8 with N=2: one each; if only one "
                         "family has seeds, all from it)")
    sd.add_argument("--with-l1", action="store_true", help="copy each in-scope game's <game>_L1.json as is")
    gm = sub.add_parser("games", help="stubs for whole-game episodes (run_arc3_level.py kind arc3_game)")
    gm.add_argument("--corpus", type=Path, required=True)
    gm.add_argument("--l1-stubs", type=Path, required=True, help="stub directory with <game>_L1.json (made by stub, game "
                    "start verified)")
    gm.add_argument("--games", type=csv, required=True)
    gm.add_argument("--heldout-games", type=csv, required=True, help="held-out list (the orchestrator needs it "
                    "non-empty; not trained on)")
    gm.add_argument("--output-token-budget", type=int, help="output quota per game (not given = the harness config's "
                    "output_token_budget)")
    gm.add_argument("--out", type=Path, required=True)
    g2 = sub.add_parser("g2seeds", help="stubs for gen-2 single-level episodes (run_arc3_level.py kind arc3_g2seed)")
    g2.add_argument("--selection", type=Path, required=True,
                    help="jsonl: seed-store index rows + role (t/h), one level-change seed per line")
    g2.add_argument("--game-stubs", type=Path, required=True, help="whole-game stub directory with <game>.json (made "
                    "by games)")
    g2.add_argument("--action-cap-factor", type=float, required=True, help="action_cap = ceil(factor x this level's "
                    "baseline)")
    g2.add_argument("--output-token-budget", type=int, help="output quota per game (not given = the harness config's "
                    "output_token_budget)")
    g2.add_argument("--out", type=Path, required=True)
    ls_ = sub.add_parser("lists")
    ls_.add_argument("--stubs", type=Path, required=True)
    ls_.add_argument("--heldout-games", type=csv, required=True)
    ls_.add_argument("--out", type=Path, required=True)
    a = p.parse_args(argv)
    if a.cmd == "recap":
        write_recap(a.src, a.factor, a.out)
    elif a.cmd == "quota":
        write_quotas(a.src, a.quotas, a.out)
    elif a.cmd == "_gate":
        got = one_gate(a.corpus, a.ws, a.label, a.game, a.level, a.actions, a.frames_nondeterministic, a.record)
        print("GATE " + json.dumps(got, ensure_ascii=False), flush=True)
    elif a.cmd == "seed":
        write_seed_stubs(a)
    elif a.cmd == "lists":
        write_lists(a.stubs, set(a.heldout_games), a.out)
    elif a.cmd == "g2seeds":
        write_g2seed_stubs(a.selection, a.game_stubs, a.action_cap_factor, a.output_token_budget, a.out)
    elif a.cmd == "games":
        write_game_stubs(a.corpus, a.l1_stubs, a.games, a.heldout_games, a.output_token_budget, a.out)
    elif a.cmd == "_stub":
        one_stub(a.corpus, a.game, a.level, a.factor, a.out)
    elif a.cmd == "_verify":
        one_verify(a.corpus, a.game, a.level, a.tarball)
    elif a.cmd == "stub":
        a.out.mkdir(parents=True, exist_ok=True)
        for game, level in parse_levels(a.levels):
            subprocess.run([sys.executable, __file__, "_stub", str(a.corpus), game, str(level),
                            str(a.action_cap_factor), str(a.out)], check=True)
    else:
        lines = []
        for game, level in parse_levels(a.levels):
            tarball = find_tarball(a.games_dir, game, level)
            if tarball is None:
                lines.append({"game": game, "level": level, "ok": None, "reason": "no record clears this level"})
                print(f"{game} L{level}: no record clears this level", flush=True)
                continue
            proc = subprocess.run([sys.executable, __file__, "_verify", str(a.corpus), game, str(level), str(tarball)],
                                  capture_output=True, text=True)
            got = [l for l in proc.stdout.splitlines() if l.startswith("VERIFY ")]
            if proc.returncode != 0 or len(got) != 1:
                raise SystemExit(f"{game} L{level}: verify failed rc={proc.returncode}\n{proc.stdout[-2000:]}"
                                 f"\n{proc.stderr[-2000:]}")
            lines.append(json.loads(got[0][7:]))
            print(json.dumps(lines[-1]), flush=True)
        a.out.write_text("".join(json.dumps(l) + "\n" for l in lines))


if __name__ == "__main__":
    main()
