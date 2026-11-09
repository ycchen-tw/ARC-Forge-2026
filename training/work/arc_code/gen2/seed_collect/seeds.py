#!/usr/bin/env python3
"""gen-2 seeds taken on the spot (``<run>/seeds/<game>/{LKK,Wnnn}``): gating, promotion into the seed store, coverage.

``gate <run dir>...``: only games that have ended (the ARM_STATE play is finished / failed; level K actions after the
  seed must be complete). For each seed:
  - ``tree_sha256`` is still the one from snapshot time, and ``skipped`` is empty (skipped files = snapshot not fully
    identical to the workspace);
  - level change (``LKK``): ``arc3_levels.py _gate`` (harness venv) -- the board / available_actions / win_levels of a
    ``set_level`` opening = the seed's last row, the seed's history is a prefix of the whole game's record, and then the
    level K actions the whole game took after the seed are replayed from the start point and compared step by step
    (lf52's transition frames are nondeterministic, so only the last frame is compared);
  - window change (``Wnnn``): the seed's history is a prefix of the whole game's record, and its last step = meta's
    action count (an opening mid-level would need the in-level actions replayed separately; not verified here).
  Results are appended to ``<run>/seeds/gate.jsonl``; existing ones are skipped. A seed failing is not an error; only
  ``_gate`` itself crashing exits non-zero.
``promote --store <dir> <run dir>...``: hard-links seeds that passed the gate into ``<store>/<game>/<name>-<run tag>/``
  (``ws`` plus a ``meta.json`` with provenance and gate result added), makes the files read-only, and writes
  ``<store>/index.jsonl``. Deleting the run directory later does not affect the store (hard links on the same file
  system).
``dedupe <run dir>...``: a game's snapshots in snapshot-time order, each compared with the previous one: files with the
  same relative path, the same size and mtime_ns, and the same content sha256 become hard links to the previous one's
  file (link under a temp name, then ``os.replace``, atomic). A finished snapshot is never written again, so sharing
  inodes is safe and ``tree_sha256`` is unchanged. Afterwards ``dedup.json`` is placed in the snapshot directory
  (outside ws) and it is skipped from then on. 10-21 s5i5: the agent put a 349 MB ``state/`` in the workspace, and every
  snapshot deep-copied it (13 snapshots in one game = 4.5 GB); collect.py runs this once per round on running runs.
``archive --store <dir> --nfs <dir> --nfs-run-dir <dir>`` (10-22, the user: move to NFS safely): the store's
  not-yet-archived seeds, in batches of ``--batch-seeds``, are hard-linked into ``<store>/.archive/bNNNN/<game>/<seed
  dir>/`` (plus this batch's ``index.jsonl``) and handed to ``nfs_move.py dirs --delete`` (tar.zst, tar -d verify,
  O_DIRECT write, read back and compare sha256, then delete only the temporary hard links) →
  ``<nfs>/store/bNNNN.tar.zst``; the batch's index is placed separately with ``nfs_move.py files`` at
  ``<nfs>/index/bNNNN.jsonl``. Only after verification is one row per seed written to ``<store>/archive.jsonl`` (batch,
  tar, sha256, path inside the tar, whether deleted locally), and then the local **window-change** seeds are deleted
  (level-change seeds stay: collect.py's continuations open from them). Fewer than ``--min-seeds`` seeds: nothing is
  done. Window-change seeds left half-deleted last time (recorded as deleted, directory still there) are deleted first.
  ``--nfs-run-dir`` must be the same as finish.sh's (``nfs_move``'s lock lives there), and it runs under finish.lock:
  one NFS write stream at a time.
``status --root <collection root>``: per game per level, how many level-change seeds passed the gate, how many
  window-change seeds, and why gates failed.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import stat
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
from arcforge.arc_code.level_seed import tree_sha256  # noqa: E402

LEVELS = ROOT / "work/arc_code/arc3_levels.py"
NFS_MOVE = ROOT / "work/tito/collect/nfs_move.py"
FRAMES_NONDETERMINISTIC = {"lf52"}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()] if path.exists() else []


def history(path: Path) -> list[bytes]:
    return [line for line in path.read_bytes().split(b"\n") if line.strip()]


def finished_games(run: Path) -> set[str]:
    state = json.loads((run / "ARM_STATE.json").read_text())
    return {p["game"] for p in state.get("plays", []) if p["status"] in ("finished", "failed")}


def seed_dirs(run: Path):
    for game_dir in sorted(p for p in (run / "seeds").iterdir() if p.is_dir()):
        for seed in sorted(p for p in game_dir.iterdir() if p.is_dir() and not p.name.endswith(".tmp")):
            yield game_dir.name, seed


def gate_one(out: Path, run: Path, game: str, seed: Path, python: str, corpus: str) -> dict:
    meta = json.loads((seed / "meta.json").read_text())
    ws = seed / "ws"
    failed = []
    if tree_sha256(ws) != meta["tree_sha256"]:
        failed.append({"check": "tree_sha256", "reason": "the snapshot changed after it was taken"})
    if meta["skipped"]:
        failed.append({"check": "skipped", "reason": f"{len(meta['skipped'])} path(s) not copied: {meta['skipped'][:3]}"})
    full = history(run / "games" / game / "record" / "history.jsonl")
    mine = history(ws / "record" / "history.jsonl")
    if full[:len(mine)] != mine:
        failed.append({"check": "history_prefix", "reason": "the seed's history is not a prefix of the game's record"})
    if json.loads(mine[-1])["step"] != meta["actions"]:
        failed.append({"check": "actions", "reason": f"last step {json.loads(mine[-1])['step']} != {meta['actions']}"})
    row = {"seed": f"{out.name}/{game}/{seed.name}", "tag": out.name, "game": game, "name": seed.name,
           "kind": meta["kind"], "level": meta["level"], "actions": meta["actions"], "ws": str(ws)}
    if meta["kind"] == "level":
        # when the level-change snapshot was taken (middleware --level-snapshot-at); meta before 10-24 lacks this field,
        # back then always before the first action
        row["at"] = meta.get("at", "first_action")
    past_win = meta["kind"] == "level" and meta["record_last"]["state"] == "WIN"
    if past_win:
        # after beating the game the agent kept sending actions and the old middleware still took "level win+1"
        # (10-22 s092 ar25 L09): there is no such level to replay
        failed.append({"check": "past_win", "reason": f"taken after the game was won (level {meta['level']})"})
    if meta["kind"] == "level" and not past_win:
        cmd = [python, "-B", str(LEVELS), "_gate", corpus, str(ws), out.name, game, str(meta["level"]),
               str(meta["actions"]), "--record", str(run / "games" / game / "record")]
        if game in FRAMES_NONDETERMINISTIC:
            cmd.append("--frames-nondeterministic")
        env = {k: v for k, v in os.environ.items() if k not in ("ONLY_RESET_LEVELS", "PYTHONPATH")}
        env.update(PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(out))
        got = [line[5:] for line in proc.stdout.splitlines() if line.startswith("GATE ")]
        if proc.returncode != 0 or len(got) != 1:
            raise RuntimeError(f"_gate {row['seed']} rc={proc.returncode}: {proc.stderr[-2000:]}")
        g = json.loads(got[0])
        failed += g["failed"]
        row.update({k: g[k] for k in ("dyn_actions", "dyn_played", "excepted", "record_cleared_at",
                                      "engine_cleared_at", "frames_nondeterministic")})
    row.update({"pass": not failed, "failed": failed})
    return row


def cmd_gate(a) -> None:
    errors = []
    for out in a.runs:
        run = out / "run"
        if not (run / "seeds").is_dir():
            print(f"{out.name}: no seeds/", flush=True)
            continue
        path = run / "seeds" / "gate.jsonl"
        done = {r["seed"] for r in read_jsonl(path)}
        finished = finished_games(run)
        corpus = json.loads((run / "ARM_STATE.json").read_text())["corpus"]["path"]
        new = pending = 0
        for game, seed in seed_dirs(run):
            if game not in finished:
                pending += 1
                continue
            if f"{out.name}/{game}/{seed.name}" in done:
                continue
            try:
                row = gate_one(out, run, game, seed, a.python, corpus)
            except RuntimeError as error:
                errors.append(str(error))
                print(f"ERROR {error}", flush=True)
                continue
            with path.open("a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            new += 1
            if not row["pass"]:
                print(f"FAIL {row['seed']}: {[x['check'] for x in row['failed']]}", flush=True)
        print(f"{out.name}: {new} gated now, {len(done)} before, {pending} in games still running", flush=True)
    if errors:
        raise SystemExit(f"_gate failed for {len(errors)} seed(s); rerun to retry")


def source_info(out: Path) -> dict:
    run = out / "run"
    state = json.loads((run / "ARM_STATE.json").read_text())
    cfg = json.loads((out / "config.json").read_text())
    server = json.loads((out / "server_info.json").read_text())        # the /get_server_info launch_codex fetched before starting
    return {"tag": out.name, "run": str(run), "served_name": state.get("model"), "model_path": server["model_path"],
            "weight_version": server["weight_version"], "drafter": server.get("speculative_draft_model_path"),
            "prompts": cfg["prompts"],
            "harness": cfg["harness"], "sampling": json.loads((out / "sampling.json").read_text())["sampling"],
            "git_head": (out / "git_head.txt").read_text().strip(), "game_output_tokens": cfg["game_output_tokens"],
            "auto_compact_token_limit": cfg["auto_compact_token_limit"],
            # which seed a hard-level continuation (run_arm --game-seed) opened from; ordinary runs lack this key (as
            # does ARM_STATE of runs before 10-22)
            "seeded_from": state.get("game_seed")}


def link_tree(src: Path, dst: Path) -> None:
    for dirpath, dirnames, filenames in os.walk(src):
        here = Path(dirpath)
        (dst / here.relative_to(src)).mkdir(parents=True)
        for name in filenames:
            os.link(here / name, dst / here.relative_to(src) / name)
            mode = os.stat(here / name).st_mode
            os.chmod(here / name, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def cmd_archive(a) -> None:
    store = a.store.resolve()
    work = store / ".archive"
    ledger = store / "archive.jsonl"
    for r in read_jsonl(ledger):              # half-deleted last time: already recorded as deleted, NFS already verified
        if r["local_deleted"] and Path(r["local"]).exists():
            shutil.rmtree(r["local"])
            print(f"finished deleting {r['local']}", flush=True)
    done = {r["seed"] for r in read_jsonl(ledger)}
    todo = [r for r in read_jsonl(store / "index.jsonl") if r["seed"] not in done]
    batches = 0
    while len(todo) >= a.min_seeds and batches < a.max_batches:
        part, todo = todo[:a.batch_seeds], todo[a.batch_seeds:]
        n = 1 + max((int(r["batch"][1:]) for r in read_jsonl(ledger)), default=0)
        name = f"b{n:04d}"
        stage = work / name
        if stage.exists() or (work / "index" / f"{name}.jsonl").exists():
            raise SystemExit(f"{stage}: left from an archive that stopped half way; look before rerunning")
        rows = []
        for r in part:
            seed_dir = Path(r["ws"]).parent
            if seed_dir.parent.parent != store or not (seed_dir / "meta.json").is_file():
                raise SystemExit(f"{r['seed']}: {seed_dir} is not a seed directory of {store}")
            rel = f"{name}/{r['game']}/{seed_dir.name}"
            for dirpath, _, filenames in os.walk(seed_dir):
                here = Path(dirpath)
                (work / rel / here.relative_to(seed_dir)).mkdir(parents=True)
                for f in filenames:
                    os.link(here / f, work / rel / here.relative_to(seed_dir) / f)
            rows.append({**r, "archive": f"store/{name}.tar.zst", "path": rel})
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        (stage / "index.jsonl").write_text(text)
        (work / "index").mkdir(parents=True, exist_ok=True)
        (work / "index" / f"{name}.jsonl").write_text(text)
        move = [sys.executable, str(NFS_MOVE), "--run-dir", str(a.nfs_run_dir)]
        print(f"{name}: {len(rows)} seeds -> {a.nfs}/store/{name}.tar.zst", flush=True)
        subprocess.run([*move, "dirs", "--src-root", str(work), "--dst-root", str(a.nfs / "store"), "--delete", name],
                       check=True)
        subprocess.run([*move, "files", "--src-root", str(work), "--dst-root", str(a.nfs), "--delete",
                        f"index/{name}.jsonl"], check=True)
        notes = [line.split("\t") for line in (work / "ARCHIVED_TO_NFS.txt").read_text().splitlines()]
        sha = [x[3] for x in notes if x[1] == f"{name}/"]
        if len(sha) != 1 or not sha[0].startswith("sha256="):
            raise SystemExit(f"{work}/ARCHIVED_TO_NFS.txt: no single verified entry for {name}/: {sha}")
        with ledger.open("a") as f:
            for r in rows:
                f.write(json.dumps({"seed": r["seed"], "kind": r["kind"], "batch": name,
                                    "tar": str(a.nfs / r["archive"]), "tar_sha256": sha[0][7:], "path": r["path"],
                                    "local": str(Path(r["ws"]).parent), "local_deleted": r["kind"] == "window",
                                    "unix": time.time()}) + "\n")
        gone = 0
        for r in rows:
            if r["kind"] == "window":
                shutil.rmtree(Path(r["ws"]).parent)
                gone += 1
        print(f"{name}: archived {len(rows)} seeds (sha256 {sha[0][7:19]}…), deleted {gone} window seeds locally",
              flush=True)
        batches += 1
    print(f"archive: {batches} batch(es), {len(todo)} seeds left for later", flush=True)


def cmd_promote(a) -> None:
    index = a.store / "index.jsonl"
    a.store.mkdir(parents=True, exist_ok=True)
    have = {r["seed"] for r in read_jsonl(index)}
    for out in a.runs:
        gates = [r for r in read_jsonl(out / "run" / "seeds" / "gate.jsonl") if r["pass"]]
        info = source_info(out)
        n = 0
        for g in gates:
            if g["seed"] in have:
                continue
            seed = out / "run" / "seeds" / g["game"] / g["name"]
            dest = a.store / g["game"] / f"{g['name']}-{out.name}"
            link_tree(seed / "ws", dest / "ws")
            meta = {**json.loads((seed / "meta.json").read_text()), "source": info, "gate": g}
            (dest / "meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n")
            os.chmod(dest / "meta.json", 0o444)
            with index.open("a") as f:
                f.write(json.dumps({"seed": g["seed"], "ws": str(dest / "ws"), "game": g["game"], "kind": g["kind"],
                                    "level": g["level"], "actions": g["actions"], "window": meta.get("window"),
                                    "tree_sha256": meta["tree_sha256"], "prefix": meta["prefix"],
                                    "sampling": info["sampling"], "tag": out.name,
                                    # when the level-change snapshot was taken (collect.py --count clean relies on
                                    # it); meta before 10-24 lacks this field = before the first action
                                    **({"at": meta.get("at", "first_action")} if g["kind"] == "level" else {})}) + "\n")
            n += 1
        print(f"{out.name}: promoted {n} of {len(gates)} passing seeds", flush=True)


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dedupe_run(run: Path) -> tuple[int, int]:
    """Returns (number of files linked this time, bytes saved)."""
    linked = saved = 0
    seeds = run / "seeds"
    if not seeds.is_dir():
        return 0, 0
    for game_dir in sorted(p for p in seeds.iterdir() if p.is_dir()):
        snaps = [p for p in game_dir.iterdir() if p.is_dir() and not p.name.endswith(".tmp")
                 and (p / "meta.json").is_file()]
        snaps.sort(key=lambda p: json.loads((p / "meta.json").read_text())["unix"])
        for prev, cur in zip(snaps, snaps[1:]):
            if (cur / "dedup.json").exists():
                continue
            n = b = 0
            for dirpath, _, filenames in os.walk(cur / "ws"):
                here = Path(dirpath)
                for name in filenames:
                    path = here / name
                    old = prev / "ws" / path.relative_to(cur / "ws")
                    try:
                        st, so = os.lstat(path), os.lstat(old)
                    except FileNotFoundError:
                        continue
                    if (st.st_ino == so.st_ino or not stat.S_ISREG(so.st_mode) or st.st_size != so.st_size
                            or st.st_mtime_ns != so.st_mtime_ns or file_sha(path) != file_sha(old)):
                        continue
                    tmp = path.with_name(f".{name}.dedup")
                    os.link(old, tmp)
                    os.replace(tmp, path)
                    n, b = n + 1, b + st.st_size
            (cur / "dedup.json").write_text(json.dumps({"against": prev.name, "linked": n, "bytes": b}) + "\n")
            linked, saved = linked + n, saved + b
    return linked, saved


def cmd_dedupe(a) -> None:
    for out in a.runs:
        n, b = dedupe_run(out / "run")
        print(f"{out.name}: linked {n} files, {b / 2**30:.2f} GiB", flush=True)


def cmd_status(a) -> None:
    rows = []
    for path in sorted(a.root.glob("runs/*/run/seeds/gate.jsonl")):
        rows += read_jsonl(path)
    taken = collections.Counter()
    for meta in a.root.glob("runs/*/run/seeds/*/*/meta.json"):
        taken[json.loads(meta.read_text())["kind"]] += 1
    lv = collections.defaultdict(collections.Counter)
    for r in rows:
        if r["kind"] == "level" and r["pass"]:
            lv[r["game"]][r["level"]] += 1
    print(f"taken: {dict(taken)}; gated {len(rows)}: pass {sum(r['pass'] for r in rows)}, "
          f"window pass {sum(r['pass'] for r in rows if r['kind'] == 'window')}")
    fails = collections.Counter(x["check"] for r in rows if not r["pass"] for x in r["failed"])
    if fails:
        print("fail checks:", dict(fails))
    for game in sorted(lv):
        print(f"  {game}: " + " ".join(f"L{k}x{v}" for k, v in sorted(lv[game].items())))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gate")
    g.add_argument("runs", type=Path, nargs="+", help="launch_codex output directories (containing run/)")
    g.add_argument("--python", default=str(Path(os.environ["ARC_RUNS"]) / "arc_code/harness-venv/bin/python"))
    pr = sub.add_parser("promote")
    pr.add_argument("runs", type=Path, nargs="+")
    pr.add_argument("--store", type=Path, required=True)
    dd = sub.add_parser("dedupe")
    dd.add_argument("runs", type=Path, nargs="+")
    st = sub.add_parser("status")
    st.add_argument("--root", type=Path, required=True)
    ar = sub.add_parser("archive")
    ar.add_argument("--store", type=Path, required=True)
    ar.add_argument("--nfs", type=Path, required=True, help="the collection directory on NFS (holds store/, index/)")
    ar.add_argument("--nfs-run-dir", type=Path, required=True, help="nfs_move.py's --run-dir (the one finish.sh uses)")
    ar.add_argument("--batch-seeds", type=int, default=1000)
    ar.add_argument("--min-seeds", type=int, default=300)
    ar.add_argument("--max-batches", type=int, default=1)
    a = p.parse_args()
    {"gate": cmd_gate, "promote": cmd_promote, "dedupe": cmd_dedupe, "status": cmd_status,
     "archive": cmd_archive}[a.cmd](a)


if __name__ == "__main__":
    main()
