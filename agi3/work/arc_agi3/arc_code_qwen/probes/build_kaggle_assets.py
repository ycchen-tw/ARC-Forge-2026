"""Rebuild the offline asset dataset the Kaggle arm mounts.

A competition session has no network, so the three things the arm normally
fetches while staging have to be uploaded once and mounted instead. Each has a
reason it cannot simply be installed there:

  * **codex** is published on npm and its Linux build is a 262 MB vendored
    `x86_64-unknown-linux-musl` binary pulled in as an optional dependency, so
    the whole install prefix has to travel, not a tarball of the wrapper.
  * **arc-code** needs our offline patch applied, and its `data/` directory --
    six complete prior sessions, four of them on these games, one a
    `claude-opus-5` WIN 8/8 -- has to be gone before it goes anywhere an agent
    can read. Two agents replayed those coordinates on 2026-09-16.
  * **bubblewrap** is absent from the image and `apt-get` cannot reach the
    archive. It has to match the image's glibc: this box is 2.39 and the
    competition image is 2.35, so its own /usr/bin/bwrap would not load there.
    The jammy package is taken from the Ubuntu archive instead.

    python3 build_kaggle_assets.py --out "$ARC_RUNS/arc_agi3_code/<tag>/kaggle-assets"
    kaggle datasets create -p <out>/upload --dir-mode zip     # or `version -m`
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UNIT = HERE.parent
PATCH = UNIT.parents[0] / "harness_patches/arc_code_offline.patch"
CODEX_VERSION = "0.154.0"
ARC_CODE_URL = "https://github.com/jerber/arc-code"
ARC_CODE_COMMIT = "6b33c1f"
# jammy, matching the competition image's glibc 2.35.
BWRAP_DEB = ("http://archive.ubuntu.com/ubuntu/pool/main/b/bubblewrap/"
             "bubblewrap_0.6.1-1ubuntu0.1_amd64.deb")
DATASET = "threerabbits/arc-code-kaggle-assets"


def run(argv: list[str], cwd: Path | None = None, stdin=None) -> None:
    done = subprocess.run(argv, cwd=cwd, stdin=stdin, capture_output=True, text=True, timeout=1800)
    if done.returncode:
        raise RuntimeError(f"{argv[0]} exited {done.returncode}: {(done.stdout + done.stderr)[-1500:]}")


def build(out: Path) -> dict:
    upload = out / "upload"
    if upload.exists():
        shutil.rmtree(upload)
    upload.mkdir(parents=True)

    run(["npm", "install", "-g", "--prefix", str(upload / "codex"), f"@openai/codex@{CODEX_VERSION}"])
    native = list((upload / "codex").rglob("vendor/*/bin/codex"))
    if not native:
        raise RuntimeError("codex install has no vendored native binary; offline it would try to fetch one")

    prep = upload / "arc-code-prep"
    run(["git", "clone", "--quiet", ARC_CODE_URL, str(prep)])
    run(["git", "-C", str(prep), "checkout", "--quiet", ARC_CODE_COMMIT])
    with PATCH.open() as handle:
        run(["patch", "-p1", "--forward"], cwd=prep, stdin=handle)
    for path in ("data", ".git"):
        shutil.rmtree(prep / path, ignore_errors=True)
    if (prep / "data").exists():
        raise RuntimeError("upstream data/runs survived; it holds worked solutions for four of these games")

    deb = out / "bubblewrap.deb"
    if not deb.exists():
        with urllib.request.urlopen(BWRAP_DEB, timeout=120) as response:  # noqa: S310 - pinned archive
            deb.write_bytes(response.read())
    extracted = out / "deb"
    shutil.rmtree(extracted, ignore_errors=True)
    run(["dpkg", "-x", str(deb), str(extracted)])
    (upload / "bwrap").mkdir()
    shutil.copy(extracted / "usr/bin/bwrap", upload / "bwrap/bwrap")

    (upload / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET.split("/")[-1].replace("-", " "), "id": DATASET,
         "licenses": [{"name": "other"}]}, indent=1))
    return {"codex_native_bytes": native[0].stat().st_size,
            "arc_code_commit": ARC_CODE_COMMIT,
            "bwrap_bytes": (upload / "bwrap/bwrap").stat().st_size,
            "total_bytes": sum(p.stat().st_size for p in upload.rglob("*") if p.is_file())}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report = build(args.out)
    (args.out / "assets-report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
