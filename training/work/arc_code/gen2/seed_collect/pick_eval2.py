#!/usr/bin/env python3
"""Which seeds the second 100k difficulty eval picks (10-24, the user: 10 per level; clean old seeds use the results of
eval-20261023, and any shortfall is run with new clean seeds).

Per level (each (game, level) with clean level-change seeds in the seed store):
1. First take the clean old seeds (confirmed by ``clean_level_seeds.jsonl`` to equal the state at note_done, and each
   run the full 5 times in eval-20261023); with more than 10, take them round-robin by ancestor (ancestor =
   ``collect.ancestor``, the run that first played from the start).
2. Below 10, fill up with new seeds taken at note_done: each time take one from "the rarest ancestor among the seeds
   chosen so far"; within an ancestor, shuffled by seed name (fixed random seed).

    python pick_eval2.py <seed store> <clean_level_seeds.jsonl> <eval_final_seeds.jsonl> <output directory> <runs.jsonl>...
Writes ``selection.jsonl`` (one row per seed per level: seed, source = old_eval / new, ancestor) and ``to_eval.jsonl``
(new ones only, for ``evaluate.py --seeds-file``).
"""

import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect import ancestor, lineage_parents  # noqa: E402

PER_LEVEL = 10


def pick(cands: list[dict], chosen: list[dict], n: int, parent: dict, rng: random.Random) -> list[dict]:
    pool = sorted(cands, key=lambda r: r["seed"])
    rng.shuffle(pool)
    out = []
    seen = collections.Counter(ancestor(r["tag"], parent) for r in chosen)
    while pool and len(out) < n:
        best = min(range(len(pool)), key=lambda i: seen[ancestor(pool[i]["tag"], parent)])
        r = pool.pop(best)
        seen[ancestor(r["tag"], parent)] += 1
        out.append(r)
    return out


def main() -> None:
    store, clean_file, eval_file, out = (Path(a) for a in sys.argv[1:5])
    parent = lineage_parents([Path(a) for a in sys.argv[5:]], [])
    clean = {json.loads(line)["seed"] for line in clean_file.read_text().splitlines() if line.strip()}
    evald = {r["seed"] for r in map(json.loads, eval_file.read_text().splitlines()) if r["n"] == 5}
    by = collections.defaultdict(lambda: {"old_eval": [], "new": []})
    for r in map(json.loads, (store / "index.jsonl").read_text().splitlines()):
        if r["kind"] != "level":
            continue
        if r.get("at") == "note_done":
            by[(r["game"], r["level"])]["new"].append(r)
        elif r["seed"] in clean and r["seed"] in evald:
            by[(r["game"], r["level"])]["old_eval"].append(r)
    rng = random.Random(20261024)
    out.mkdir(parents=True, exist_ok=True)
    rows, short = [], []
    for key in sorted(by):
        old = pick(by[key]["old_eval"], [], PER_LEVEL, parent, rng)
        new = pick(by[key]["new"], old, PER_LEVEL - len(old), parent, rng)
        if len(old) + len(new) < PER_LEVEL:
            short.append(f"{key[0]}:L{key[1]}={len(old) + len(new)}")
        rows += [{"seed": r["seed"], "game": r["game"], "level": r["level"], "source": src,
                  "ancestor": ancestor(r["tag"], parent)} for src, rs in (("old_eval", old), ("new", new)) for r in rs]
    with open(out / "selection.jsonl", "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    with open(out / "to_eval.jsonl", "w") as f:
        f.writelines(json.dumps({"seed": r["seed"]}) + "\n" for r in rows if r["source"] == "new")
    n = collections.Counter(r["source"] for r in rows)
    print(f"{len(by)} levels, {len(rows)} seeds: {dict(n)}; short of {PER_LEVEL}: {short}")


if __name__ == "__main__":
    main()
