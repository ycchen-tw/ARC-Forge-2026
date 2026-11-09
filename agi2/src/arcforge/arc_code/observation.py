#!/usr/bin/env python3
"""Read recorded pixels without game access; also works as a standalone script.

    python observation.py current --colors 8,e --components
    python observation.py 12:0 --roi 10,20,19,29 --grid
    python observation.py current --compare 12 --colors 8 --cells
    python observation.py 14 --compare 13 --translation 9 30,6,37,14 37,6,44,14

Coordinates are x,y; all bounding boxes are inclusive [x0,y0,x1,y1].
STEP or STEP:final selects final.txt; STEP:INDEX selects an original NPZ frame.
``current`` resolves once through history to an immutable, committed step. A
zero-frame action does not replace the last observed board. No engine, network,
image decoder, or other arcforge module is imported. For solver input use
``grid, source = load_board(Path("record"), "current")`` directly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import deque
from pathlib import Path

import numpy as np

SIDE = 64
MAX_ITEMS = 256
MAX_GRID_CELLS = 1024
DEFAULT_MAX_BYTES = 8192


class ObservationError(ValueError):
    """Invalid or incomplete observation input; never substitute guessed data."""


def _integer(value, name, *, minimum=0, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ObservationError(f"{name} must be an integer")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ObservationError(f"{name} outside {minimum}..{maximum}")
    return value


def _history(root):
    path = root / "history.jsonl"
    rows = {}
    # Read one snapshot: later actions cannot change this query's source step.
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ObservationError(f"history line {lineno} is empty")
        row = json.loads(line)
        if not isinstance(row, dict) or "step" not in row or "frames" not in row:
            raise ObservationError(f"history line {lineno} needs step and frames")
        step = _integer(row["step"], "history step")
        _integer(row["frames"], "history frames")
        if rows and step <= next(reversed(rows)):
            raise ObservationError("history steps must be strictly increasing")
        rows[step] = row
    if not rows:
        raise ObservationError("record has no committed steps")
    return rows


def _selector(selector, frame=None):
    selector = str(selector)
    if selector.count(":") > 1:
        raise ObservationError("source must be current or STEP[:FRAME]")
    parts = selector.split(":")
    if len(parts) == 2 and frame is not None:
        raise ObservationError("use either STEP:FRAME or --frame, not both")
    step = parts[0]
    chosen = parts[1] if len(parts) == 2 else ("final" if frame is None else str(frame))
    if step != "current" and not re.fullmatch(r"[0-9]+", step):
        raise ObservationError("step must be current or a nonnegative integer")
    if chosen != "final" and not re.fullmatch(r"[0-9]+", chosen):
        raise ObservationError("frame must be final or a nonnegative original frame index")
    return step, chosen


def _validate_grid(grid):
    if not isinstance(grid, np.ndarray) or grid.shape != (SIDE, SIDE):
        raise ObservationError("board must be a 64x64 numpy array")
    if grid.dtype != np.uint8 or np.any(grid > 15):
        raise ObservationError("board must be uint8 with cell values 0..15")
    return grid


def _text_grid(path):
    rows = path.read_text(encoding="ascii").splitlines()
    if len(rows) != SIDE or any(not re.fullmatch(r"[0-9a-fA-F]{64}", row) for row in rows):
        raise ObservationError(f"{path.name} must contain exactly 64 rows of 64 hex cells")
    return np.array([[int(c, 16) for c in row] for row in rows], dtype=np.uint8)


def _load(root, selector, history, frame=None):
    step, chosen = _selector(selector, frame)
    latest = next(reversed(history))
    if step == "current":
        available = [s for s, row in history.items() if row["frames"] > 0]
        if not available:
            raise ObservationError("record contains no observed frames")
        step = available[-1]
    else:
        step = int(step)
    if step not in history:
        raise ObservationError(f"step {step} is not committed in history")
    count = history[step]["frames"]
    if count == 0:
        raise ObservationError(f"step {step} returned no frames")
    relative = Path("steps") / f"{step:06d}"
    if chosen == "final":
        relative /= "final.txt"
        grid = _text_grid(root / relative)
        index = count - 1
    else:
        index = int(chosen)
        if index >= count:
            raise ObservationError(f"step {step} has {count} frames; index {index} is out of range")
        relative /= "frames.npz"
        with np.load(root / relative, allow_pickle=False) as archive:
            if "frames" not in archive.files:
                raise ObservationError("NPZ must contain the frames array")
            frames = archive["frames"]
        if frames.shape != (count, SIDE, SIDE) or frames.dtype != np.uint8:
            raise ObservationError("NPZ frames must be uint8 (history.frames,64,64)")
        if np.any(frames > 15):
            raise ObservationError("NPZ frame values outside 0..15")
        grid = frames[index].copy()
    return grid, {
        "step": step,
        "frame": index,
        "frames": count,
        "path": relative.as_posix(),
        "latest_recorded_step": latest,
    }


def load_board(record_dir, selector="current", *, frame=None):
    """Return an exact uint8 array plus its immutable step/frame provenance."""
    root = Path(record_dir)
    return _load(root, selector, _history(root), frame)


def _roi(roi):
    if roi is None:
        return (0, 0, SIDE - 1, SIDE - 1)
    if len(roi) != 4:
        raise ObservationError("ROI needs x0,y0,x1,y1 (inclusive)")
    x0, y0, x1, y1 = tuple(_integer(v, "ROI coordinate", maximum=SIDE - 1) for v in roi)
    if x0 > x1 or y0 > y1:
        raise ObservationError("ROI lower bound exceeds upper bound")
    return x0, y0, x1, y1


def crop_grid(grid, roi):
    """Crop inclusive x,y bounds, preserving ndarray [y,x] indexing."""
    _validate_grid(grid)
    x0, y0, x1, y1 = _roi(roi)
    return grid[y0 : y1 + 1, x0 : x1 + 1].copy()


def _colors(colors):
    if colors is None:
        return tuple(range(16))
    if not colors:
        raise ObservationError("colors must not be empty")
    return tuple(sorted({_integer(v, "color", maximum=15) for v in colors}))


def _bounds(xs, ys):
    if len(xs) == 0:
        return {"bbox": None, "size": [0, 0]}
    x0, x1, y0, y1 = int(min(xs)), int(max(xs)), int(min(ys)), int(max(ys))
    return {"bbox": [x0, y0, x1, y1], "size": [x1 - x0 + 1, y1 - y0 + 1]}


def color_counts(grid, *, roi=None, colors=None):
    """Per-color pixel counts and bounding boxes; absent requested colors count 0."""
    x0, y0, x1, y1 = _roi(roi)
    crop = crop_grid(grid, (x0, y0, x1, y1))
    out = []
    for color in _colors(colors):
        ys, xs = np.nonzero(crop == color)
        if len(xs) or colors is not None:
            out.append({"color": color, "count": len(xs), **_bounds(xs + x0, ys + y0)})
    return out


def components(grid, *, roi=None, colors=None, connectivity=4):
    """Same-color connected components within ROI, in absolute board coordinates.

    A crop may split a larger component. touches_roi_edge reports boundary
    contact, not proof that the component continues outside the crop.
    """
    if connectivity not in (4, 8):
        raise ObservationError("connectivity must be 4 or 8")
    x0, y0, x1, y1 = _roi(roi)
    crop = crop_grid(grid, (x0, y0, x1, y1))
    h, w = crop.shape
    allowed = set(_colors(colors))
    seen = np.zeros((h, w), dtype=bool)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 8:
        directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    out = []
    for y in range(h):
        for x in range(w):
            color = int(crop[y, x])
            if seen[y, x] or color not in allowed:
                continue
            seen[y, x] = True
            queue = deque([(x, y)])
            cells = []
            while queue:
                cx, cy = queue.popleft()
                cells.append((cx + x0, cy + y0))
                for dx, dy in directions:
                    nx, ny = cx + dx, cy + dy
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny, nx] and int(crop[ny, nx]) == color:
                        seen[ny, nx] = True
                        queue.append((nx, ny))
            xs, ys = zip(*cells)
            bounds = _bounds(xs, ys)
            bx0, by0, bx1, by1 = bounds["bbox"]
            out.append(
                {
                    "color": color,
                    "count": len(cells),
                    **bounds,
                    "touches_roi_edge": bx0 == x0 or by0 == y0 or bx1 == x1 or by1 == y1,
                }
            )
    return sorted(out, key=lambda c: (c["color"], -c["count"], c["bbox"][1], c["bbox"][0]))


def _bounded(items, limit):
    return {"total": len(items), "truncated": len(items) > limit, "items": items[:limit]}


def diff_grids(before, after, *, roi=None, colors=None, include_cells=False, limit=64):
    """Exact before→after differences; color filter matches either endpoint."""
    limit = _integer(limit, "limit", minimum=1, maximum=MAX_ITEMS)
    x0, y0, x1, y1 = _roi(roi)
    a, b = crop_grid(before, (x0, y0, x1, y1)), crop_grid(after, (x0, y0, x1, y1))
    changed = a != b
    selected = changed & (np.isin(a, _colors(colors)) | np.isin(b, _colors(colors)))
    ys, xs = np.nonzero(selected)
    pairs, counts = np.unique(np.stack((a[selected], b[selected]), axis=1), axis=0, return_counts=True)
    out = {
        "changed_cells": int(changed.sum()),
        "selected_changed_cells": len(xs),
        **_bounds(xs + x0, ys + y0),
        "transitions": _bounded(
            [{"from": int(u), "to": int(v), "count": int(n)} for (u, v), n in zip(pairs, counts)], limit
        ),
    }
    if include_cells:
        out["cells"] = _bounded([[int(x + x0), int(y + y0), int(a[y, x]), int(b[y, x])] for x, y in zip(xs, ys)], limit)
        out["cell_columns"] = ["x", "y", "before", "after"]
    return out


def _translation_component(grid, color, roi, connectivity):
    region = _roi(roi)
    found = components(grid, roi=region, colors=(color,), connectivity=connectivity)
    report = {"roi": list(region), "component_count": len(found)}
    if len(found) != 1:
        report["reason"] = "no_component" if not found else "multiple_components"
        return report, None
    item = found[0]
    report.update({key: item[key] for key in ("count", "bbox", "size")})
    x0, y0, x1, y1 = region
    # A cropped fragment can look like a translated object. Refuse if any
    # selected pixel connects to the same color outside the requested ROI.
    ys, xs = np.nonzero(crop_grid(grid, region) == color)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 8:
        directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    for x, y in zip(xs + x0, ys + y0):
        if x0 < x < x1 and y0 < y < y1:
            continue
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (0 <= nx < SIDE and 0 <= ny < SIDE
                    and not (x0 <= nx <= x1 and y0 <= ny <= y1)
                    and grid[ny, nx] == color):
                report["reason"] = "roi_cuts_component"
                return report, None
    bx0, by0, bx1, by1 = item["bbox"]
    report["position"] = [bx0, by0]
    mask = grid[by0:by1 + 1, bx0:bx1 + 1] == color
    return report, mask


def compare_translation(before, after, *, color, before_roi, after_roi, connectivity=4):
    """Check translation compatibility of two explicitly selected pixel sets.

    Each ROI must select one complete same-color component. No component
    matching, nearest-object selection, or physical object identity is inferred.
    Even identical translated masks could belong to two different objects.
    ``dx,dy`` are returned only when every selected pixel matches a translation;
    changing area, shape, merging, splitting, or clipping never yields a guessed
    displacement. Component analysis uses the caller's explicit connectivity.
    """
    color = _integer(color, "translation color", maximum=15)
    before_info, before_mask = _translation_component(before, color, before_roi, connectivity)
    after_info, after_mask = _translation_component(after, color, after_roi, connectivity)
    result = {"color": color, "connectivity": connectivity, "object_identity": "not_inferred",
              "before": before_info, "after": after_info}
    if before_mask is None or after_mask is None:
        result["status"] = "ambiguous"
        return result
    if not np.array_equal(before_mask, after_mask):
        result.update(status="not_translation", reason="pixel_mask_changed")
        return result
    result.update(status="translation_compatible",
                  dx=after_info["position"][0] - before_info["position"][0],
                  dy=after_info["position"][1] - before_info["position"][1])
    return result


def inspect_record(
    record_dir,
    selector="current",
    *,
    frame=None,
    roi=None,
    colors=None,
    at=(),
    include_cells=False,
    include_components=False,
    include_grid=False,
    compare=None,
    connectivity=4,
    limit=64,
    translation=None,
):
    """Build a JSON-compatible answer. Lists report explicit truncation and totals.

    Optional translation=(color, before_roi, after_roi) requires compare and
    returns only that relation and its source. It cannot be mixed with ordinary
    ROI/color/point/grid/component/cell queries; query an ordinary diff separately.
    """
    limit = _integer(limit, "limit", minimum=1, maximum=MAX_ITEMS)
    if translation is not None:
        if compare is None:
            raise ObservationError("--translation requires --compare")
        if len(translation) != 3:
            raise ObservationError("translation needs color, before_roi, after_roi")
        if roi is not None or colors is not None or at or include_cells or include_components or include_grid:
            raise ObservationError("--translation has its own color and two ROIs; do not combine it with "
                                   "--roi/--colors/--at/--cells/--components/--grid; query ordinary diff separately")
    root = Path(record_dir)
    history = _history(root)
    grid, source = _load(root, selector, history, frame)
    if translation is not None:
        previous, previous_source = _load(root, compare, history)
        color, before_roi, after_roi = translation
        result = compare_translation(previous, grid, color=color, before_roi=before_roi,
                                     after_roi=after_roi, connectivity=connectivity)
        result["before"]["source"] = previous_source
        result["after"]["source"] = source
        return {"source": source, "coordinates": "x,y", "bbox_convention": "inclusive", "translation": result}
    region, selected_colors = _roi(roi), _colors(colors)
    crop = crop_grid(grid, region)
    x0, y0, x1, y1 = region
    out = {
        "source": source,
        "shape": [SIDE, SIDE],
        "coordinates": "x,y",
        "bbox_convention": "inclusive",
        "roi": list(region),
        "roi_cells": int(crop.size),
        "colors": list(selected_colors),
        "counts": color_counts(grid, roi=region, colors=colors),
    }
    if at:
        if len(at) > limit:
            raise ObservationError(f"at most {limit} point queries are allowed")
        points = []
        for point in at:
            if len(point) != 2:
                raise ObservationError("point needs x,y")
            x, y = (_integer(v, "point coordinate", maximum=SIDE - 1) for v in point)
            if not (x0 <= x <= x1 and y0 <= y <= y1):
                raise ObservationError(f"point ({x},{y}) is outside ROI")
            points.append([x, y, int(grid[y, x])])
        out["at"] = {"columns": ["x", "y", "color"], "items": points}
    if include_cells:
        ys, xs = np.nonzero(np.isin(crop, selected_colors))
        out["cells"] = _bounded([[int(x + x0), int(y + y0), int(crop[y, x])] for x, y in zip(xs, ys)], limit)
        out["cells"]["columns"] = ["x", "y", "color"]
    if include_components:
        out["components"] = {
            "scope": "within_roi",
            "connectivity": connectivity,
            **_bounded(components(grid, roi=region, colors=colors, connectivity=connectivity), limit),
        }
    if include_grid:
        if crop.size > MAX_GRID_CELLS:
            raise ObservationError(f"--grid is limited to {MAX_GRID_CELLS} cells; narrow --roi")
        out["grid"] = {"origin": [x0, y0], "indexing": "values[y-y0][x-x0]", "values": crop.tolist()}
    if compare is not None:
        previous, previous_source = _load(root, compare, history)
        out["diff"] = {
            "before": previous_source,
            "after": source,
            **diff_grids(previous, grid, roi=region, colors=colors, include_cells=include_cells, limit=limit),
        }
    return out


def encode_result(result, max_bytes=DEFAULT_MAX_BYTES):
    max_bytes = _integer(max_bytes, "max_bytes", minimum=1024, maximum=32768)
    text = json.dumps(result, ensure_ascii=True, separators=(",", ":"), allow_nan=False) + "\n"
    if len(text.encode("utf-8")) > max_bytes:
        raise ObservationError(f"answer exceeds {max_bytes} bytes; narrow --roi/--colors or lower --limit")
    return text


def _csv_numbers(text):
    try:
        return tuple(int(v.strip(), 10) for v in text.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated decimal coordinates") from error


def _color_arg(text):
    try:
        values = [int(v, 16) if re.fullmatch(r"[a-fA-F]|0[xX][0-9a-fA-F]+", v) else int(v, 10) for v in text.split(",")]
        return _colors(values)
    except ValueError as error:
        raise argparse.ArgumentTypeError("colors must be 0..15, a..f, or 0x0..0xf") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "source", nargs="?", default="current", help="current or STEP[:FRAME], zero-based original frame"
    )
    parser.add_argument("--record-dir", type=Path, default=Path("record"))
    parser.add_argument("--frame", help="final or original frame index; alternative to STEP:FRAME")
    parser.add_argument("--roi", type=_csv_numbers, help="inclusive x0,y0,x1,y1")
    parser.add_argument("--colors", type=_color_arg, help="comma-separated decimal values or hex a..f")
    parser.add_argument(
        "--at",
        type=_csv_numbers,
        action="append",
        default=[],
        help="exact x,y value; repeatable, independent of color filter",
    )
    parser.add_argument("--cells", action="store_true", help="list selected cells (and changed cells with --compare)")
    parser.add_argument("--components", action="store_true", help="same-color connected components within ROI")
    parser.add_argument("--connectivity", type=int, choices=(4, 8), default=4)
    parser.add_argument("--grid", action="store_true", help="return ROI numeric grid, at most 1024 cells")
    parser.add_argument("--compare", help="before source STEP[:FRAME]; current source is after")
    parser.add_argument("--translation", nargs=3, metavar=("COLOR", "BEFORE_ROI", "AFTER_ROI"),
                        help="with --compare, check one complete same-color component per ROI for exact "
                             "pixel translation; return only this relation, no object identity inferred. "
                             "Do not mix with ordinary query flags (ROIs: x0,y0,x1,y1)")
    parser.add_argument("--limit", type=int, default=64, help="list items per result, 1..256; counts remain exact")
    parser.add_argument("--max-bytes", type=int, default=DEFAULT_MAX_BYTES, help="JSON byte cap, 1024..32768")
    args = parser.parse_args(argv)
    translation = None
    if args.translation is not None:
        try:
            color = _color_arg(args.translation[0])
            if len(color) != 1:
                raise ObservationError("--translation needs exactly one color")
            translation = (color[0], _roi(_csv_numbers(args.translation[1])), _roi(_csv_numbers(args.translation[2])))
        except (ObservationError, argparse.ArgumentTypeError) as error:
            parser.error(str(error))
    try:
        result = inspect_record(
            args.record_dir,
            args.source,
            frame=args.frame,
            roi=args.roi,
            colors=args.colors,
            at=args.at,
            include_cells=args.cells,
            include_components=args.components,
            include_grid=args.grid,
            compare=args.compare,
            connectivity=args.connectivity,
            limit=args.limit,
            translation=translation,
        )
        sys.stdout.write(encode_result(result, args.max_bytes))
    except (ObservationError, OSError, ValueError, KeyError) as error:
        sys.stderr.write(json.dumps({"error": str(error)[:1000]}) + "\n")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
