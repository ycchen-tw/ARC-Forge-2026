"""``act_image`` (duckx c0i): markers that ``./act do`` prints at the end of its tool output are replaced by
images before they reach the model. Hooked into ``sampling_proxy.py``'s ``/v1/responses``
(``--image-ref-key`` / ``--image-run-dir``).

    [[arc-image:board:<ref>]]         the board at that step, drawn like record/steps/NNNNNN/final.png (record.png)
    [[arc-image:diff:<ref>:<ref>]]    the later step's board: cells that differ from the earlier step in their new
                                      color, unchanged cells painted the off-board dark blue (Super Duck's diff
                                      image, DIFF_UNCHANGED_RGB); identical boards are replaced by one sentence

Refs are signed by the middleware (``src/arcforge/arc_code/image_ref.py``); the signature is checked here, and
the two refs of a diff must be from the same game with the earlier step first. Only strings of
``function_call_output`` / ``custom_tool_call_output`` (or ``input_text`` parts) are replaced; assistant and
reasoning are untouched: markers the model types itself stay verbatim. The same marker is replaced by the same
bytes every time -- the prefix cache only serves requests that append. If that step's final.txt cannot be read
(a corrupt record), the marker becomes one sentence, printed to stderr and counted, and the request still goes out.
"""

from __future__ import annotations

import base64
import re
import sys
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np

from arcforge.arc_code import image_ref, record

MARKER = re.compile(r"\[\[arc-image:(board|diff):([^\]\s]*)\]\]")
OUTPUT_TYPES = ("function_call_output", "custom_tool_call_output")
#: unchanged cells: Super Duck vision_context.DIFF_UNCHANGED_RGB. Palette 0-5 is a white-to-black grey ramp,
#: so any grey would collide with a cell value.
UNCHANGED_RGB = (16, 24, 48)
UNCHANGED = len(record.RGB)
DIFF_RGB = np.vstack([record.RGB, np.array([UNCHANGED_RGB], dtype=np.uint8)])
DIFF_DARK = (DIFF_RGB.astype(np.uint16) * 3 // 4).astype(np.uint8)


def diff_grid(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Cell values of ``after``; unchanged cells become UNCHANGED (DIFF_RGB's off-board color)."""
    return np.where(before != after, after, UNCHANGED).astype(np.uint8)


def diff_png(before: np.ndarray, after: np.ndarray, size: int) -> bytes:
    return record.encode_png(record.render(diff_grid(before, after), size, DIFF_RGB, DIFF_DARK))


def parse_board(text: str) -> np.ndarray:
    """final.txt (record.board_text: one hex line per row) -> uint8 (H, W). Raises on a bad shape or character."""
    rows = text.splitlines()
    if not rows or len({len(r) for r in rows}) != 1 or not rows[0]:
        raise ValueError(f"not a rectangular board ({len(rows)} rows)")
    return np.array([[int(c, 16) for c in row] for row in rows], dtype=np.uint8)


class ActImages:
    def __init__(self, key: bytes, run_dir: Path, png_size: int, cache_items: int = 8192) -> None:
        if png_size < 64 or png_size % 64:
            raise ValueError(f"png_size {png_size} must be a multiple of 64")
        self.key, self.games, self.png_size = key, run_dir / "games", png_size
        self.cache_items = cache_items
        self.lock = threading.Lock()
        self.cache: OrderedDict[str, dict] = OrderedDict()
        self.counts = {"requests": 0, "requests_with_images": 0, "images": 0, "unchanged_diffs": 0,
                       "missing": 0, "bad_refs": 0}

    def snapshot(self) -> dict:
        with self.lock:
            return {**self.counts, "cached": len(self.cache)}

    def _count(self, name: str, n: int = 1) -> None:
        with self.lock:
            self.counts[name] += n

    def apply(self, payload: dict) -> int:
        """Replace markers in the request's tool outputs with images (payload modified in place); return how many
        substituted images this request carries."""
        n = 0
        items = payload.get("input")
        for item in items if isinstance(items, list) else []:
            if not isinstance(item, dict) or item.get("type") not in OUTPUT_TYPES:
                continue
            output = item.get("output")
            if isinstance(output, str):
                parts = self._split(output)
                if parts is not None:
                    item["output"] = parts
                    n += sum(p["type"] == "input_image" for p in parts)
            elif isinstance(output, list):
                out, hit = [], False
                for part in output:
                    if isinstance(part, dict) and part.get("type") == "input_text" and isinstance(part.get("text"), str):
                        parts = self._split(part["text"])
                        if parts is not None:
                            out += parts
                            n += sum(p["type"] == "input_image" for p in parts)
                            hit = True
                            continue
                    out.append(part)
                if hit:
                    item["output"] = out
        with self.lock:
            self.counts["requests"] += 1
            self.counts["requests_with_images"] += bool(n)
            self.counts["images"] += n
        return n

    def _split(self, text: str) -> list[dict] | None:
        """Text -> content parts. None if there is no recognized marker (left as is); markers that fail
        verification stay as raw text."""
        parts: list[dict] = []
        pos, found = 0, False

        def add_text(s: str) -> None:
            if not s:
                return
            if parts and parts[-1]["type"] == "input_text":
                parts[-1] = {"type": "input_text", "text": parts[-1]["text"] + s}
            else:
                parts.append({"type": "input_text", "text": s})

        for m in MARKER.finditer(text):
            part = self._resolve(m.group(0), m.group(1), m.group(2))
            if part is None:
                continue
            found = True
            add_text(text[pos:m.start()])
            if part["type"] == "input_text":
                add_text(part["text"])
            else:
                parts.append(part)
            pos = m.end()
        if not found:
            return None
        add_text(text[pos:])
        return parts

    def _resolve(self, marker: str, kind: str, body: str) -> dict | None:
        with self.lock:
            hit = self.cache.get(marker)
            if hit is not None:
                self.cache.move_to_end(marker)
                return hit
        refs = body.split(":")
        verified = [image_ref.verify(self.key, r) for r in refs]
        expected = 1 if kind == "board" else 2
        if len(refs) != expected or None in verified \
                or (kind == "diff" and (verified[0][0] != verified[1][0] or verified[0][1] >= verified[1][1])):
            self._count("bad_refs")
            print(f"[act_images] BAD_REF {marker[:200]}", file=sys.stderr, flush=True)
            return None
        try:
            boards = [self._board(game, step) for game, step in verified]
        except (OSError, ValueError) as error:
            self._count("missing")
            print(f"[act_images] MISSING {marker}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            steps = " and ".join(str(step) for _, step in verified)
            part = {"type": "input_text", "text": f"(image unavailable: the record of step {steps} cannot be read)"}
        else:
            if kind == "board":
                part = _image(record.png(boards[0], self.png_size))
            elif boards[0].shape != boards[1].shape:
                part = _image(record.png(boards[1], self.png_size))
            elif np.array_equal(boards[0], boards[1]):
                self._count("unchanged_diffs")
                part = {"type": "input_text",
                        "text": f"(no cell changed between step {verified[0][1]} and step {verified[1][1]})"}
            else:
                part = _image(diff_png(boards[0], boards[1], self.png_size))
        with self.lock:
            self.cache[marker] = part
            while len(self.cache) > self.cache_items:
                self.cache.popitem(last=False)
        return part

    def _board(self, game: str, step: int) -> np.ndarray:
        path = self.games / game / "record" / "steps" / f"{step:06d}" / "final.txt"
        return parse_board(path.read_text())


def _image(png: bytes) -> dict:
    return {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(png).decode()}
