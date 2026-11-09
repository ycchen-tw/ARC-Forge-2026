"""Build $ARC_DATA/arc3_levels/g2seed-v8-20261029 (RL v8: first window only, note judge; plan
docs/plans/arc3_rl_v8_window_note_judge_20261029.md §2). Copied from ../../arc3v7b_20261029/seeds/build_v7b_dir.py; the
games, classes and seed pools are v7b's; only the per-level counts change (v8: hard 5 / wall 3 / medium 2 / easy 0 --
user 10-29 "use the hard and wall levels, mix in some medium").

v7b's docstring follows.

Build $ARC_DATA/arc3_levels/g2seed-v7b-20261029 (RL v7b: 20 training games, from v7@11).

Training games (user 10-29: 20 games with training value; held out 1 hard + 4 others): the 25 public games minus
s5i5 (hard; v7@5 broke its L7 in 4/4 whole-game copies, a known-sensitive held-out probe) and ar25 ft09 lp85 sb26 (every
level ~1.0 at 100k: the least training value).
Classes: v7's 5 games keep v7's classes (lists/classes.json there); the 15 new games use the clean seeds' 100k pass rate
per level (clean_seeds_eval_20261026.jsonl, base, n=5 per seed): easy >= 0.6, hard < 0.25, else medium; a new-game level
with 0 clears over all its seeds at 100k is a wall.
Seeds per level per epoch (the orchestrator runs each id once per epoch): hard 5, medium 3, easy 1, wall 1 -- walls get
one try each (a breakthrough there is signal; 5 x 24 x 300k on a likely all-zero level is not). random.Random(20261029).
v7 games: stubs copied from g2seed-v7-20261028 (300k, seeds verified there), picked from its train + train_unused.
New games: index rows of the clean seeds, ws unpacked from the store archives when not on disk (members + hard-link
targets, tree_sha256 checked), stubs by arc3_levels.py games / g2seeds (action_cap 2 x baseline as v7, budget 300000).
held-out list: 2 seeds of s5i5 L07 (the orchestrator needs a non-empty list; no in-loop held-out runs).

python3 build_v7b_dir.py <repo> <out dir>"""
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

repo, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
sys.path.insert(0, str(repo / "src"))
from arcforge.arc_code import level_seed  # noqa: E402

DATA = Path(os.environ["ARC_DATA"])
STORES = [DATA / "arc3_seeds/g2seed-20261021", Path(os.environ["ARC_NFS"]) / "arc3_seeds/g2seed-20261021"]
V7 = DATA / "arc3_levels/g2seed-v7-20261028"
V7_GAMES = {"g50t", "sk48", "sp80", "su15", "wa30"}
HELD = ["s5i5", "ar25", "ft09", "lp85", "sb26"]
NEW_GAMES = sorted({"bp35", "cd82", "cn04", "dc22", "ka59", "lf52", "ls20", "m0r0", "r11l", "re86", "sc25", "tn36",
                    "tr87", "tu93", "vc33"})
PER = {"hard": 5, "medium": 2, "easy": 0, "wall": 3}
BUDGET, CAP = 300_000, 2.0
key = lambda g, l: f"{g}_L{l:02d}"


def unpack(job):
    """One archive: extract the members of its rows into a private staging dir, check, rename into each row's ws."""
    archive, rows = job
    src = next(s / archive for s in STORES if (s / archive).is_file())
    listing = subprocess.run(["tar", "-I", "zstd", "-tvf", str(src)], check=True, capture_output=True, text=True).stdout
    names, links = [], {}
    for line in listing.splitlines():
        parts = line.split(" link to ")
        name = parts[0].split(maxsplit=5)[5]
        names.append(name)
        if len(parts) == 2:
            links[name] = parts[1]
    members = set()
    for r in rows:
        prefix = r["path"].rstrip("/") + "/"
        mine = [n for n in names if n == r["path"] or n.startswith(prefix)]
        if not mine:
            raise SystemExit(f"{archive}: nothing under {r['path']}")
        members.update(mine)
        members.update(links[n] for n in mine if n in links)
    staging = Path(tempfile.mkdtemp(prefix=".staging-v8-", dir=STORES[0]))
    names_file = staging.with_name(staging.name + ".members")   # a member list too long for argv goes through -T
    names_file.write_text("".join(m + "\n" for m in sorted(members)))
    subprocess.run(["tar", "-I", "zstd", "-xf", str(src), "-C", str(staging), "--no-recursion", "--verbatim-files-from",
                    "-T", str(names_file)], check=True)
    names_file.unlink()
    done = []
    for r in rows:
        s = staging / r["path"]
        s = s / "ws" if (s / "ws").is_dir() and not (s / "notes.md").exists() else s
        if level_seed.tree_sha256(s) != r["tree_sha256"]:
            raise SystemExit(f"{r['seed']}: unpacked tree_sha256 != index")
        ws = Path(r["ws"])
        ws.parent.mkdir(parents=True, exist_ok=True)
        if ws.exists():
            raise SystemExit(f"{ws} appeared meanwhile")
        os.rename(s, ws)
        done.append(r["seed"])
    shutil.rmtree(staging)
    return archive, done


def main():
    clean = [json.loads(l) for l in (repo / "work/arc_code/gen2/seed_collect/results/clean_seeds_eval_20261026.jsonl")
             .read_text().splitlines()]
    index = {}
    for store in STORES:
        for f in sorted((store / "index").glob("*.jsonl")):
            for line in f.read_text().splitlines():
                r = json.loads(line)
                if r["kind"] == "level" and r["window"] is None:
                    index.setdefault(r["seed"], r)
    v7cls = json.loads((V7 / "lists/classes.json").read_text())["levels"]
    by_level = defaultdict(list)
    for c in clean:
        by_level[(c["game"], c["level"])].append(c)
    classes, pools = {}, {}
    for g in sorted(V7_GAMES):
        for lv in sorted(k for k in v7cls if k.startswith(g + "_")):
            classes[lv] = v7cls[lv]
    v7ids = (V7 / "lists/train.txt").read_text().split() + (V7 / "lists/train_unused.txt").read_text().split()
    for t in v7ids:
        pools.setdefault(t.rsplit("_", 1)[0], []).append(t)
    for (g, l), cs in sorted(by_level.items()):
        if g not in NEW_GAMES:
            continue
        n, cl = sum(c["n"] for c in cs), sum(c["cleared"] for c in cs)
        p = cl / n
        classes[key(g, l)] = "wall" if cl == 0 else "easy" if p >= 0.6 else "hard" if p < 0.25 else "medium"
        pools[key(g, l)] = sorted(c["seed"] for c in cs if c["seed"] in index)
    if {lv.split("_")[0] for lv in classes} != V7_GAMES | set(NEW_GAMES) or len(V7_GAMES | set(NEW_GAMES)) != 20:
        raise SystemExit("not the 20 training games")
    rng = random.Random(20261030)
    pick = {}
    for lv in sorted(classes):
        pool = sorted(pools[lv])
        if PER[classes[lv]] == 0:
            continue
        n = min(PER[classes[lv]], len(pool))
        if n < PER[classes[lv]]:
            raise SystemExit(f"{lv}: {len(pool)} seeds, want {PER[classes[lv]]}")
        pick[lv] = sorted(rng.sample(pool, n))
    s5 = sorted(c["seed"] for c in by_level[("s5i5", 7)] if c["seed"] in index)
    held = sorted(rng.sample(s5, 2))
    new_rows = [dict(index[s], role="t") for lv, ss in pick.items() if lv.split("_")[0] in NEW_GAMES for s in ss]
    new_rows += [dict(index[s], role="h") for s in held]
    todo = defaultdict(list)
    for r in new_rows:
        ws = Path(r["ws"])
        if ws.exists():
            if level_seed.tree_sha256(ws) != r["tree_sha256"]:
                raise SystemExit(f"{r['seed']}: on-disk ws {ws} does not hash to the index (partial unpack?)")
        else:
            todo[r["archive"]].append(r)
    print(f"{len(new_rows)} new-game seeds, {sum(map(len, todo.values()))} to unpack from {len(todo)} archives", flush=True)
    with ProcessPoolExecutor(8) as ex:
        for archive, done in ex.map(unpack, sorted(todo.items())):
            print(f"  {archive}: {len(done)} unpacked", flush=True)
    out.mkdir(parents=True, exist_ok=False)
    work = out.parent / f".{out.name}.build"
    work.mkdir()
    (work / "selection.jsonl").write_text("".join(json.dumps(r) + "\n" for r in new_rows))
    py = str(repo / ".venv/bin/python")
    lv_py = str(repo / "work/arc_code/arc3_levels.py")
    subprocess.run([py, lv_py, "games", "--corpus", os.environ["ARC_ARC3_CORPUS"], "--l1-stubs",
                    str(DATA / "arc3_levels/public25"), "--games", ",".join(NEW_GAMES), "--heldout-games", "s5i5",
                    "--out", str(work / "games")], check=True, cwd=repo)
    subprocess.run([py, lv_py, "g2seeds", "--selection", str(work / "selection.jsonl"), "--game-stubs",
                    str(work / "games"), "--action-cap-factor", str(CAP), "--output-token-budget", str(BUDGET),
                    "--out", str(work / "stubs")], check=True, cwd=repo)
    train, heldout = [], (work / "stubs/lists/heldout.txt").read_text().split()
    for p in sorted((work / "stubs").glob("*.json")):
        shutil.copy2(p, out / p.name)
    train += (work / "stubs/lists/train.txt").read_text().split()
    for lv, ss in pick.items():
        if lv.split("_")[0] in V7_GAMES:
            for t in ss:
                doc = json.loads((V7 / f"{t}.json").read_text())
                if doc["output_token_budget"] != BUDGET or doc["action_cap"] != round(CAP * doc["baseline_actions"]):
                    raise SystemExit(f"{t}: budget {doc['output_token_budget']} cap {doc['action_cap']}")
                shutil.copy2(V7 / f"{t}.json", out / f"{t}.json")
                train.append(t)
    lvl = lambda t: t.rsplit("_", 1)[0]
    if Counter(map(lvl, train)) != Counter({lv: len(ss) for lv, ss in pick.items()}) or set(train) & set(heldout):
        raise SystemExit("train list does not match the picks, or overlaps the held-out list")
    (out / "lists").mkdir()
    (out / "lists/train.txt").write_text("".join(t + "\n" for t in sorted(train)))
    (out / "lists/heldout.txt").write_text("".join(t + "\n" for t in heldout))
    cls_out = {lv: ("hard" if c == "wall" else c) for lv, c in classes.items()}
    (out / "lists/classes.json").write_text(json.dumps(
        {"rule": "v7 games: v7's classes; new games: clean seeds' 100k base pass rate per level, easy >= 0.6, hard < "
                 "0.25 (0 clears = wall, counted hard), else medium; v8 seeds per level per epoch hard 5 / medium 2 / "
                 "wall 3 / easy none (random.Random(20261030))", "walls": sorted(k for k, c in classes.items() if c == "wall"),
         "training_games": sorted(V7_GAMES | set(NEW_GAMES)), "heldout_games": HELD,
         "levels": {lv: c for lv, c in sorted(cls_out.items()) if lv in pick}}, indent=1) + "\n")
    shutil.move(str(work), str(out / "lists/build"))
    by = Counter(classes[lvl(t)] for t in train)
    print(f"{len(train)} training ids over {len(pick)} levels: " + ", ".join(f"{c} {by[c]} ({by[c] / len(train):.0%})"
          for c in ("hard", "wall", "medium", "easy")) + f"; held-out {heldout} -> {out}")


if __name__ == "__main__":
    main()
