#!/usr/bin/env python3
"""Build the one-wheel dataset that gives the Kaggle arm a pip.

The competition image can neither make a pip nor be given one by the organiser:
`python -m venv` exits 1 inside ensurepip (measured 2026-09-17, arm notebook v4),
and the competition's own `arc_agi_3_wheels` holds 31 wheels with no pip among
them. A session has no network, so the only way in is a mount.

This is deliberately its own dataset rather than a new version of
`arc-code-kaggle-assets`: that one is 300 MB of codex, patched arc-code and a
jammy bubblewrap, and v4 proved the copy of it now on Kaggle works. Re-uploading
it to add 1.8 MB would put a known-good mount back on the table for no reason.

    python3 build_pip_dataset.py --out "$ARC_RUNS/arc_agi3_code/<tag>/pip-dataset"
    kaggle datasets create -p <out>/upload          # or `version -m`
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

DATASET = "threerabbits/pip-offline-wheel"
# Pinned by hash: the notebook runs whatever is mounted, so what is mounted has
# to be what was reviewed here.
PIP_VERSION = "26.2.1"
PIP_WHEEL = f"pip-{PIP_VERSION}-py3-none-any.whl"
PIP_SHA256 = "71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e"


def fetch(dest: Path) -> int:
    """Download the pinned wheel, refusing anything that is not it."""
    meta = json.load(urllib.request.urlopen(  # noqa: S310 - pypi over https
        f"https://pypi.org/pypi/pip/{PIP_VERSION}/json", timeout=120))
    urls = [u for u in meta["urls"] if u["filename"] == PIP_WHEEL]
    if len(urls) != 1:
        raise RuntimeError(f"expected exactly one {PIP_WHEEL} on PyPI, found {len(urls)}")
    blob = urllib.request.urlopen(urls[0]["url"], timeout=300).read()  # noqa: S310
    digest = hashlib.sha256(blob).hexdigest()
    if digest != PIP_SHA256:
        raise RuntimeError(f"{PIP_WHEEL} is sha256 {digest}, expected {PIP_SHA256}")
    dest.write_bytes(blob)
    return len(blob)


def build(out: Path) -> dict:
    upload = out / "upload"
    upload.mkdir(parents=True, exist_ok=True)
    size = fetch(upload / PIP_WHEEL)
    (upload / "dataset-metadata.json").write_text(json.dumps(
        {"title": DATASET.split("/")[-1].replace("-", " "), "id": DATASET,
         "licenses": [{"name": "other"}]}, indent=1))
    return {"wheel": PIP_WHEEL, "bytes": size, "sha256": PIP_SHA256, "dataset": DATASET}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    report = build(args.out)
    (args.out / "pip-dataset-report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
