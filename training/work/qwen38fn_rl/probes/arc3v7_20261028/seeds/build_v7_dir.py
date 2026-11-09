"""Build $ARC_DATA/arc3_levels/g2seed-v7-20261028 (RL v7).

- stubs: every v6b stub with output_token_budget 80000 -> 300000, plus the new-level stubs (sk48 L06, sp80 L05 / L06,
  generated at 300000); every stub's seed ws must hash to its seed.sha256 (level_seed.tree_sha256), checked here
- classes (v6b QAD-R, groups dispatched at versions 0-4, 80k pass rate per level; coordinator 10-28): easy >= 0.6,
  hard < 0.25 or no v6b data (the three new levels), else medium
- lists/train.txt: hard 9 seeds per level (fewer if the level has fewer), medium 6, easy all -- the orchestrator runs
  each id once per epoch, so this sets the class mix of the dispatched groups (~52 / 31 / 17); random.Random(20261028)
- lists/heldout.txt: v6b's 61 + the 6 new held-out seeds (2 per new level)
- lists/train_unused.txt, lists/classes.json

python3 build_v7_dir.py <repo> <v6b dir> <new stubs dir> <classes txt> <out dir>
"""
import json
import random
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

repo, v6b, new, cls_txt, out = map(Path, sys.argv[1:6])
sys.path.insert(0, str(repo / "src"))
from arcforge.arc_code import level_seed  # noqa: E402

BUDGET = 300_000
PER_LEVEL = {"hard": 9, "medium": 6, "easy": None}
key = lambda t: re.match(r"([a-z0-9]+_L\d\d)_", t).group(1)


def check(path: str) -> tuple[str, bool]:
    stub = json.loads(Path(path).read_text())
    return path, level_seed.tree_sha256(Path(stub["seed"]["ws"])) == stub["seed"]["sha256"]


def main() -> None:
    classes = {}
    for line in cls_txt.read_text().splitlines()[1:]:
        p = line.split()
        if len(p) == 6 and re.fullmatch(r"[a-z0-9]+_L\d\d", p[0]):
            classes[p[0]] = "hard" if p[2] == "-" or float(p[2]) < 0.25 else "easy" if float(p[2]) >= 0.6 else "medium"
    stubs = {}
    for d in (v6b, new):
        for p in sorted(d.glob("*.json")):
            if p.stem in stubs:
                raise SystemExit(f"{p.stem} twice")
            doc = json.loads(p.read_text())
            if doc.get("kind") != "g2seed" or doc.get("output_token_budget") not in (80_000, BUDGET):
                raise SystemExit(f"{p}: kind {doc.get('kind')} budget {doc.get('output_token_budget')}")
            doc["output_token_budget"] = BUDGET
            stubs[p.stem] = (p, doc)
    with ProcessPoolExecutor(8) as ex:
        bad = [p for p, ok in ex.map(check, [str(p) for p, _ in stubs.values()]) if not ok]
    if bad:
        raise SystemExit(f"{len(bad)} stubs whose seed ws does not hash to seed.sha256: {bad[:5]}")
    lists = lambda d, n: (d / "lists" / n).read_text().split()
    train_pool = lists(v6b, "train.txt") + lists(new, "train.txt")
    heldout = lists(v6b, "heldout.txt") + lists(new, "heldout.txt")
    if set(train_pool) & set(heldout) or len(set(train_pool)) != len(train_pool) or not set(train_pool + heldout) <= set(stubs):
        raise SystemExit("train / held-out ids overlap, repeat or have no stub")
    if set(map(key, train_pool + heldout)) != set(classes):
        raise SystemExit(f"levels {set(map(key, train_pool)) ^ set(classes)} not in both the lists and the class table")
    rng = random.Random(20261028)
    train, unused = [], []
    for lv in sorted(classes):
        ids = sorted(t for t in train_pool if key(t) == lv)
        n = PER_LEVEL[classes[lv]]
        pick = set(ids if n is None or len(ids) <= n else rng.sample(ids, n))
        train += [t for t in ids if t in pick]
        unused += [t for t in ids if t not in pick]
    out.mkdir(parents=True, exist_ok=False)
    (out / "lists").mkdir()
    for stem, (_, doc) in sorted(stubs.items()):
        (out / f"{stem}.json").write_text(json.dumps(doc, indent=1) + "\n")
    for name, ids in (("train.txt", train), ("heldout.txt", heldout), ("train_unused.txt", unused)):
        (out / "lists" / name).write_text("".join(t + "\n" for t in ids))
    (out / "lists" / "classes.json").write_text(json.dumps(
        {"rule": "v6b QAD-R groups dispatched at versions 0-4, 80k pass rate per level: easy >= 0.6, hard < 0.25 or no "
                 "v6b data, else medium; train seeds per level: hard 9, medium 6, easy all (random.Random(20261028))",
         "levels": dict(sorted(classes.items()))}, indent=1) + "\n")
    by = {c: [t for t in train if classes[key(t)] == c] for c in PER_LEVEL}
    print(f"{len(stubs)} stubs (all {BUDGET}, seeds verified) -> {out}")
    print(f"train {len(train)}: " + " ".join(f"{c} {len(v)} ({len(v) / len(train):.0%})" for c, v in by.items())
          + f"; held-out {len(heldout)}; unused {len(unused)}")


if __name__ == "__main__":
    main()
