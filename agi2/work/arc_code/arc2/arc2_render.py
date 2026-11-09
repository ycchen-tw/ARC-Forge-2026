"""ARC-AGI-2 grid → PNG. Used for the image attached to codex's first message (``codex exec -i``).

In Qwen3-VL one vision token covers 32×32 px (patch 16, merge 2); images under 256² total pixels get
upscaled. Hence: at least 16 px per cell (a 30×30 task = 480² = 225 tokens, versus ~930 tokens as
text), long side at least 256 px (so small tasks are not bilinearly upscaled into blurred grid lines),
and the whole image padded with background to a multiple of 32 (zero resampling; see pad_to_token in
docs/research/qwen38fn_image_ablation.md).
"""

from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image

#: Standard ARC colors (the official arc-prize viewer).
PALETTE = ("#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
           "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25")
COLOR_NAMES = ("black", "blue", "red", "green", "yellow",
               "grey", "magenta", "orange", "light blue", "maroon")
RGB = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in PALETTE], dtype=np.uint8)
GRID_LINE = np.array([64, 64, 64], dtype=np.uint8)
PAD = np.array([255, 255, 255], dtype=np.uint8)
MIN_CELL = 16
MIN_SIDE = 256
TOKEN_PX = 32


def check_grid(grid) -> np.ndarray:
    """Rectangular, integers 0–9, sides 1–30; raise otherwise."""
    arr = np.asarray(grid)
    if arr.ndim != 2 or not 1 <= arr.shape[0] <= 30 or not 1 <= arr.shape[1] <= 30:
        raise ValueError(f"grid shape {arr.shape} is not 2-D within 1..30")
    if arr.dtype.kind not in "iu" or arr.min() < 0 or arr.max() > 9:
        raise ValueError("grid cells must be integers 0-9")
    return arr.astype(np.uint8)


def cell_px(shape: tuple[int, int]) -> int:
    return max(MIN_CELL, math.ceil(MIN_SIDE / max(shape)))


#: Frame color for wrong cells on the check-result image in image-only mode. White is not in the ARC
#: palette, so it cannot be confused with a cell color.
MARK = np.array([255, 255, 255], dtype=np.uint8)


def render(grid, wrong=None) -> np.ndarray:
    """wrong: boolean array of the grid's shape; True cells get a white frame (image-only mode check
    result)."""
    arr = check_grid(grid)
    c = cell_px(arr.shape)
    img = RGB[arr].repeat(c, axis=0).repeat(c, axis=1)
    if wrong is not None:
        wrong = np.asarray(wrong, dtype=bool)
        if wrong.shape != arr.shape:
            raise ValueError(f"wrong mask {wrong.shape} != grid {arr.shape}")
        t = max(2, c // 6)
        for r, col in np.argwhere(wrong):
            y, x = r * c, col * c
            img[y + 1:y + c, x + 1:x + 1 + t] = MARK
            img[y + 1:y + c, x + c - t:x + c] = MARK
            img[y + 1:y + 1 + t, x + 1:x + c] = MARK
            img[y + c - t:y + c, x + 1:x + c] = MARK
    img[::c, :] = GRID_LINE          # a 1 px grid line along each cell's top and left edge
    img[:, ::c] = GRID_LINE
    h, w = img.shape[:2]
    H, W = -(-h // TOKEN_PX) * TOKEN_PX, -(-w // TOKEN_PX) * TOKEN_PX
    out = np.empty((H, W, 3), dtype=np.uint8)
    out[:] = PAD
    out[:h, :w] = img
    return out


def png(grid, wrong=None) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(render(grid, wrong)).save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def vision_tokens(grid) -> int:
    """Number of vision tokens this image costs the model (excluding marker tokens)."""
    h, w = render(grid).shape[:2]
    return (h // TOKEN_PX) * (w // TOKEN_PX)


def grid_text(grid) -> str:
    """One row per line, one digit per cell, no separators (same format as the agentic_tir prompt)."""
    return "\n".join("".join(str(int(v)) for v in row) for row in check_grid(grid)) + "\n"
