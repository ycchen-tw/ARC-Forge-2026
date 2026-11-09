"""Maintain deploy/sglang.lock.json (src/arcforge/sglang_lock.py): point a deployment target at a sglang-arc-2 commit.

  python lock.py set <target> <fork ref> [--note TEXT]   export the tree (artifacts/sglang_fork/trees/) and lock it
  python lock.py show

The fork checkout is $ARC_SGLANG_FORK (env/local.sh); the commit must be on a pushed branch or tag
(the lock names commits other machines can fetch).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))
import export_tree  # noqa: E402
from arcforge.sglang_lock import FIELDS, LOCK  # noqa: E402

TREES = REPO / "artifacts" / "sglang_fork" / "trees"


def pushed(commit: str) -> bool:
    """Whether the commit is reachable from a branch or tag of the fork's origin (deploy/exp/legacy are tags)."""
    fork = str(export_tree.paths.sglang_fork())
    refs = subprocess.run(["git", "-C", fork, "ls-remote", "--heads", "--tags", "origin"],
                          check=True, capture_output=True, text=True).stdout.split("\n")
    tips = {line.split()[0] for line in refs if line.strip()}
    for tip in tips:
        if subprocess.run(["git", "-C", fork, "merge-base", "--is-ancestor", commit, tip],
                          capture_output=True).returncode == 0:
            return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("set")
    s.add_argument("target")
    s.add_argument("ref")
    s.add_argument("--note", default="")
    sub.add_parser("show")
    a = ap.parse_args()
    lock = json.loads(LOCK.read_text()) if LOCK.exists() else {
        "fork": "ycchen-tw/sglang-arc-2", "upstream": {"tag": "v0.5.20", "commit": "94602c9c2b7cbdb8efd5c52802dac6a1c180089e"},
        "targets": {}}
    if a.cmd == "show":
        for name, t in sorted(lock["targets"].items()):
            print(f"{name:18s} {t['commit'][:12]} {t['digest_symlink'][:12]}/{t['digest_deref'][:12]} "
                  f"files={t['files']} {t.get('note', '')}")
        return
    info = export_tree.export(a.ref, TREES)
    if not pushed(info["commit"]):
        raise SystemExit(f"{a.ref} ({info['commit'][:12]}) is not on any origin branch or tag of the fork: push it first")
    entry = {k: info[k] for k in FIELDS}
    entry |= {"ref": a.ref, "note": a.note}
    lock["targets"][a.target] = entry
    LOCK.parent.mkdir(exist_ok=True)
    LOCK.write_text(json.dumps(lock, indent=2, sort_keys=False) + "\n")
    print(f"{a.target} -> {info['commit'][:12]} ({info['files']} files, {info['source_files']} changed)")


if __name__ == "__main__":
    main()
