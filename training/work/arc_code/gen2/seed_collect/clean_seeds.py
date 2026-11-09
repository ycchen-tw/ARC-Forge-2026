#!/usr/bin/env python3
"""Which level-change seeds of the 10-21 round (level-change snapshot taken before the first action of level K) equal the
state at note_done and can be used as clean seeds directly.

When note_done passes, act writes ``.note_gate/L(K-1).done`` (``{"step": the step that cleared the level, "time": when
it was written}``); the snapshot excludes ``.note_gate``, so the time has to be extracted from the collection run's
archive (NFS ``runs/<tag>.tar.zst``). The seed's files were copied with ``shutil.copy2``, so the mtimes are the
originals': if no file outside ``record/`` has an mtime later than that time, no file was changed after note_done.
Deleted files cannot be detected (directory mtimes were created at snapshot time); the output says so plainly.

    python clean_seeds.py <seed store> <extracted note_gate directory> <output jsonl>
      extracted note_gate directory: one subdirectory per run, holding the tar-extracted
      ``*/run/games/<game>/.note_gate/L*.done``
"""

import collections
import glob
import json
import os
import sys
from pathlib import Path

store, gates, out = (Path(a) for a in sys.argv[1:4])
marks = {}
for f in glob.glob(str(gates / "*/*/run/games/*/.note_gate/L*.done")):
    parts = Path(f).parts
    tag, game, level = parts[-6], parts[-3], int(Path(f).stem[1:3])
    marks[(tag, game, level)] = json.loads(Path(f).read_text())

rows, why = [], collections.Counter()
for line in open(store / "index.jsonl"):
    r = json.loads(line)
    if r["kind"] != "level" or r.get("at", "first_action") != "first_action":
        continue
    mark = marks.get((r["tag"], r["game"], r["level"] - 1))
    if mark is None:
        why["no note_done mark"] += 1
        rows.append({"seed": r["seed"], "clean": False, "why": "no note_done mark"})
        continue
    if mark["step"] != r["actions"]:
        raise SystemExit(f"{r['seed']}: note_done mark is for step {mark['step']}, the seed ends at {r['actions']}")
    ws = Path(r["ws"])
    later = []
    for dp, dn, fn in os.walk(ws):
        if Path(dp).relative_to(ws).parts[:1] == ("record",):
            continue
        later += [str((Path(dp) / f).relative_to(ws)) for f in fn if (Path(dp) / f).stat().st_mtime > mark["time"]]
    clean = not later
    why["clean" if clean else "changed after note_done"] += 1
    rows.append({"seed": r["seed"], "game": r["game"], "level": r["level"], "clean": clean,
                 "note_done_time": mark["time"], "changed_after": sorted(later)[:20], "changed_after_n": len(later)})
with open(out, "w") as f:
    for row in rows:
        if row["clean"]:
            f.write(json.dumps(row) + "\n")
print(f"{len(rows)} first-action level seeds: {dict(why)} (deleted files cannot be seen; written: the clean ones -> {out})")
