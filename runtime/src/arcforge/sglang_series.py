"""Series arms of the sglang tree: an explicit, ordered patch list instead of the base / hisparse globs.

``arcforge.sglang_tree`` (the digest the Kaggle notebook asserts, and the base / hisparse rows of
``patches/sglang-0.5.20/TREE_DIGEST`` that the G4, L4 and DFlash tooling read) is left as it is. A series arm keeps
its own two files under ``patches/sglang-0.5.20``:

  <arm>.series        one patch path per line, relative to that directory, in apply order ('#' comments)
  TREE_DIGEST.<arm>   one row in TREE_DIGEST's format: ``<arm> <whole-package digest> files=<n> patches=<sha>``,
                      written by ``work/qsa_hisparse/g4_runtime.py series-write --arm <arm>`` on a G4

arc3 = the hs55 v6/v7 tree e93d999a (0001-0032 + experimental-hisparse h01-h09, h12-h22: TREE_DIGEST's hisparse row
at 1f2cf48c) + the arc-code candidates from h40 (experimental-hisparse/candidates/). Its QSA HiSparse manifest is
``arc3.SOURCE_SHA256``. Standard library only.
"""

from __future__ import annotations

import re
from pathlib import Path

#: arm -> (series file, digest file, QSA HiSparse manifest), all under patches/sglang-0.5.20
SERIES = {"arc3": ("arc3.series", "TREE_DIGEST.arc3", "arc3.SOURCE_SHA256")}
ROW = re.compile(r"([a-z0-9]+) ([0-9a-f]{64}) files=(\d+) patches=([0-9a-f]{64})")


def _files(arm: str) -> tuple[str, str, str]:
    if arm not in SERIES:
        raise ValueError(f"{arm!r} is not a series arm ({sorted(SERIES)})")
    return SERIES[arm]


def patch_set(patches_dir: Path, arm: str) -> list[Path]:
    """The patches the series file lists, in order; every entry must be an existing .patch, none twice."""
    path = Path(patches_dir) / _files(arm)[0]
    if not path.is_file():
        raise ValueError(f"no series file {path} for arm {arm!r}")
    out = []
    for n, line in enumerate(path.read_text().splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        p = Path(patches_dir) / line
        if p.suffix != ".patch" or not p.is_file():
            raise ValueError(f"{path}:{n}: {line!r} is not a patch file under {patches_dir}")
        if p in out:
            raise ValueError(f"{path}:{n}: {line!r} is listed twice")
        out.append(p)
    if not out:
        raise ValueError(f"{path} lists no patches")
    return out


def digest_path(patches_dir: Path, arm: str) -> Path:
    return Path(patches_dir) / _files(arm)[1]


def manifest_path(patches_dir: Path, arm: str) -> Path:
    return Path(patches_dir) / _files(arm)[2]


def parse_row(text: str, where: str, arm: str) -> dict:
    """TREE_DIGEST.<arm> text -> {digest, files, patches}; exactly one row, for this arm."""
    _files(arm)
    rows = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        m = ROW.fullmatch(line)
        if m is None or m.group(1) != arm:
            raise ValueError(f"{where}:{n} is not '{arm} <digest> files=<n> patches=<sha>': {line!r}")
        rows.append({"digest": m.group(2), "files": int(m.group(3)), "patches": m.group(4)})
    if len(rows) != 1:
        raise ValueError(f"{where} has {len(rows)} rows, want one {arm} row")
    return rows[0]


def read_row(patches_dir: Path, arm: str) -> dict:
    path = digest_path(patches_dir, arm)
    return parse_row(path.read_text(), str(path), arm)


def row_text(arm: str, digest: str, files: int, patches: str) -> str:
    _files(arm)
    return ("# whole sglang package of the series arm (every file but __pycache__ and _version.py), from\n"
            "# g4_runtime.py series-write; patches= is sha256 over the series' patch files (arcforge.sglang_tree)\n"
            f"{arm} {digest} files={files} patches={patches}\n")
