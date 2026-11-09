"""The single entry point for the four path variables.

Rules: CONVENTIONS.md §Paths. Importing this module never fails; only **calling** a function whose value is
unavailable raises, so "forgot to source env/env.sh" blows up the first time a path is used instead of silently
writing to the wrong place.
"""

from __future__ import annotations

import os
from pathlib import Path

#: repo root. src/arcforge/paths.py -> two levels up.
ROOT = Path(__file__).resolve().parents[2]

_HINT = "run `source env/env.sh` first (see CONVENTIONS.md §Paths)"


def _env_path(name: str, default: Path | None = None) -> Path:
    raw = os.environ.get(name)
    if raw is None:
        if default is None:
            raise RuntimeError(f"{name} is not set. {_HINT}")
        path = default
    else:
        path = Path(raw).expanduser()
    if not path.exists():
        raise RuntimeError(f"{name}={path} does not exist")
    return path


def data() -> Path:
    """Data root. Default `$ARC_ROOT/data` (gitignored)."""
    return _env_path("ARC_DATA", ROOT / "data")


def models() -> Path:
    """Model weights root. No default -- it differs per machine and must be stated explicitly."""
    return _env_path("ARC_MODELS")


def sglang_fork() -> Path:
    """Local checkout of the private sglang fork (ycchen-tw/sglang-arc-2): `$ARC_SGLANG_FORK` (set in the machine file)."""
    return _env_path("ARC_SGLANG_FORK")


def runs() -> Path:
    """Output root. New runs always land in `runs()/<unit>/<YYYYMMDD-tag>/`."""
    return _env_path("ARC_RUNS")


def run_dir(unit: str, tag: str, *, create: bool = False) -> Path:
    """`$ARC_RUNS/<unit>/<tag>/`. With `create=False`, raise if the directory does not exist."""
    path = runs() / unit / tag
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.exists():
        raise RuntimeError(f"{path} does not exist (use create=True to create it)")
    return path
