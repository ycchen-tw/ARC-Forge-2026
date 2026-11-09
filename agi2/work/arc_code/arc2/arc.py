"""This puzzle's grids, already parsed -- use this instead of printing task/task.json.

    from arc import train, test, show, diff, render

    train          list of (input, output) pairs, each a numpy int array (rows x columns)
    test           list of test input arrays
    show(g)        print a grid one row per line, one digit per cell (the cheapest way to look at it)
    diff(a, b)     print where two grids differ (size mismatch, or each differing cell as (row, col): a vs b)
    render(g, path)  save a grid as a PNG (same colours and cell size as task/*.png) for view_image
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TASK = json.loads((_HERE / "task" / "task.json").read_text())
train = [(np.array(p["input"]), np.array(p["output"])) for p in _TASK["train"]]
test = [np.array(t["input"]) for t in _TASK["test"]]

_PALETTE = ((0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0),
            (170, 170, 170), (240, 18, 190), (255, 133, 27), (127, 219, 255), (135, 12, 37))


def show(grid) -> None:
    a = np.asarray(grid)
    print("\n".join("".join(str(int(v)) for v in row) for row in a))


def diff(a, b, limit: int = 40) -> None:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        print(f"size differs: {a.shape} vs {b.shape}")
        return
    cells = np.argwhere(a != b)
    print(f"{len(cells)} of {a.size} cells differ")
    for r, c in cells[:limit]:
        print(f"  ({r},{c}): {a[r, c]} vs {b[r, c]}")
    if len(cells) > limit:
        print(f"  ... {len(cells) - limit} more")


def render(grid, path) -> None:
    from PIL import Image
    a = np.asarray(grid).astype(int)
    c = max(16, -(-256 // max(a.shape)))
    rgb = np.array(_PALETTE, dtype=np.uint8)[a].repeat(c, axis=0).repeat(c, axis=1)
    rgb[::c, :] = 64
    rgb[:, ::c] = 64
    Image.fromarray(rgb).save(path)
