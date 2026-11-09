"""v7: unpack the clean seeds of sk48 L6 / sp80 L5 / sp80 L6 from the g2seed store and write the selection jsonl.

For each clean seed (clean_seeds_eval_20261026.jsonl) of the three levels: its index row (kind level, window null). If its
ws is not on disk: list the archive, take every member under the row's path plus the hard-link targets those members
point at (the v6 notes: a link's target can sit in a W* window directory of the same archive), extract them into a fresh
staging directory, check level_seed.tree_sha256 against the index, then rename staging/<path> to ws (no overwrite).
Writes selection.jsonl (index row + role: 2 held-out per level, random seed 20261028, the rest t).

python3 extract_new_levels.py <repo> <out dir> <broken ws dir>
"""
import json
import os
import random
import subprocess
import sys
from pathlib import Path

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo / "src"))
from arcforge.arc_code import level_seed  # noqa: E402

STORE = Path(os.environ["ARC_DATA"]) / "arc3_seeds" / "g2seed-20261021"
LEVELS = {("sk48", 6), ("sp80", 5), ("sp80", 6)}
clean = [json.loads(l) for l in (repo / "work/arc_code/gen2/seed_collect/results/clean_seeds_eval_20261026.jsonl")
         .read_text().splitlines()]
want = sorted(r["seed"] for r in clean if (r["game"], r["level"]) in LEVELS)
index = {}
for f in sorted((STORE / "index").glob("*.jsonl")):
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r["kind"] == "level" and r["window"] is None and (r["game"], r["level"]) in LEVELS:
            if r["seed"] in index:
                raise SystemExit(f"{r['seed']} twice in the index")
            index[r["seed"]] = r
missing = [s for s in want if s not in index]
print(f"{len(want)} clean seeds, {len(missing)} not in the index: {missing}")
rows = [index[s] for s in want if s in index]
out.mkdir(parents=True, exist_ok=False)
staging = STORE / f".staging-v7-{os.getpid()}"
staging.mkdir()
by_archive = {}
BROKEN = Path(sys.argv[3])   # on-disk ws whose tree_sha256 is not the index's (partial extractions) move here
for r in rows:
    ws = Path(r["ws"])
    if ws.exists() and level_seed.tree_sha256(ws) != r["tree_sha256"]:
        dst = BROKEN / r["seed"].replace("/", "_")
        print(f"{r['seed']}: on-disk ws sha mismatch, moved to {dst}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        os.rename(ws, dst)
    if not ws.exists():
        by_archive.setdefault(r["archive"], []).append(r)
for archive, rs in sorted(by_archive.items()):
    listing = subprocess.run(["tar", "-I", "zstd", "-tvf", str(STORE / archive)], check=True, capture_output=True,
                             text=True).stdout.splitlines()
    names, links = [], {}
    for line in listing:
        parts = line.split(" link to ")
        name = parts[0].split(maxsplit=5)[5]
        names.append(name)
        if len(parts) == 2:
            links[name] = parts[1]
    members = set()
    for r in rs:
        prefix = r["path"].rstrip("/") + "/"
        mine = [n for n in names if n == r["path"] or n.startswith(prefix)]
        if not mine:
            raise SystemExit(f"{archive}: nothing under {r['path']}")
        members.update(mine)
        members.update(links[n] for n in mine if n in links)
    subprocess.run(["tar", "-I", "zstd", "-xf", str(STORE / archive), "-C", str(staging), "--no-recursion",
                    *sorted(members)], check=True)
    for r in rs:
        src, ws = staging / r["path"], Path(r["ws"])
        got = level_seed.tree_sha256(src / "ws" if (src / "ws").is_dir() and not (src / "notes.md").exists() else src)
        print(f"{r['seed']}: extracted {src} sha {got[:12]} index {r['tree_sha256'][:12]}")
        if got != r["tree_sha256"]:
            raise SystemExit(f"{r['seed']}: tree_sha256 mismatch")
        ws.parent.mkdir(parents=True, exist_ok=True)
        if ws.exists():
            raise SystemExit(f"{ws} appeared meanwhile")
        os.rename(src / "ws" if (src / "ws").is_dir() and not (src / "notes.md").exists() else src, ws)
rng = random.Random(20261028)
sel = []
for lv in sorted(LEVELS):
    rs = sorted((r for r in rows if (r["game"], r["level"]) == lv), key=lambda r: r["seed"])
    held = set(r["seed"] for r in rng.sample(rs, 2))
    for r in rs:
        if level_seed.tree_sha256(Path(r["ws"])) != r["tree_sha256"]:
            raise SystemExit(f"{r['seed']}: on-disk ws sha mismatch")
        sel.append({**r, "role": "h" if r["seed"] in held else "t"})
(out / "selection.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sel))
print(f"selection: {len(sel)} ({sum(r['role'] == 'h' for r in sel)} held-out); staging left at {staging}")
