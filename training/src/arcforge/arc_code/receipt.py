"""`./act do` 結尾的物件級回執（opt-in：config ``act_receipt``）。

整份原始碼原樣塞進 agent 的 act.py（stdlib only，jail 裡沒有 numpy），所以這裡不能 import
arcforge 的任何東西。動機：Codex 每次出手之後平均再叫模型 3 次才出下一手（Super Duck 1.2 次），
多出來的是看圖、自己寫 diff、重推盤面；Duck 每輪直接拿到變化摘要
（docs/research/super_duck_codex_roundtrips_20261008.md）。

回執只講事實，不講玩法：哪些同色 4 連通塊移動／出現／消失／變色（座標 x,y＝欄,列，與簡報一致），
邊框帶（計數器、計時條常在那裡）另計，動畫中變了又變回去的格子，以及完全沒變的步。
"""

import ast
import zipfile
from pathlib import Path

RECEIPT_EDGE = 4          # 離邊框這麼近的格子另計（Duck 的 gameplay_changed 用同一個寬度）
RECEIPT_LARGE = 512       # 比這大的連通塊當成背景／大區域，不逐一列
RECEIPT_MAX_LINES = 12
RECEIPT_SCREEN = 1024     # 邊框帶以內變了這麼多格就當換畫面：逐物件配對在這裡只會配出假的「移動」
RECEIPT_SCENE_LINES = 24  # 新關卡的物件清單最多列幾種
RECEIPT_SCENE_SPOTS = 16  # 每種最多列幾個位置（ft09 一關 13 塊一樣的磚）


def receipt_board(step):
    """step 之後的盤面：record 只在有幀的步寫 final.txt（0 幀的步沒有），所以往回找最近一份，
    跟 record 算 changed_cells 用的 last_final 同一個定義。一份都沒有就是 record 壞了。"""
    for n in range(step, -1, -1):
        path = Path("record/steps/%06d/final.txt" % n)
        if path.is_file():
            return [[int(c, 16) for c in line.strip()] for line in path.read_text().splitlines() if line.strip()]
    raise FileNotFoundError("no record/steps/*/final.txt at or before step %d" % step)


def receipt_frames(step):
    """record/steps/N/frames.npz，不用 numpy：uint8 (F, H, W) → list[list[bytes]]（每幀每列一個 bytes）。"""
    with zipfile.ZipFile("record/steps/%06d/frames.npz" % step) as archive:
        raw = archive.read("frames.npy")
    if raw[:6] != b"\x93NUMPY":
        raise ValueError("frames.npy is not an npy file")
    size = 2 if raw[6] == 1 else 4
    length = int.from_bytes(raw[8:8 + size], "little")
    header = ast.literal_eval(raw[8 + size:8 + size + length].decode("latin1"))
    if header["descr"] not in ("|u1", "<u1") or header["fortran_order"] or len(header["shape"]) != 3:
        raise ValueError("frames.npy is not a C-order uint8 (F, H, W) array: %r" % (header,))
    count, height, width = header["shape"]
    data = raw[8 + size + length:]
    return [[data[(f * height + y) * width:(f * height + y + 1) * width] for y in range(height)]
            for f in range(count)]


def receipt_components(grid, seeds):
    """碰到 seeds 的同色 4 連通塊（整塊，不只變了的格子）。"""
    height, width = len(grid), len(grid[0])
    seen, out = set(), []
    for start in sorted(seeds):
        if start in seen:
            continue
        x0, y0 = start
        color = grid[y0][x0]
        stack, cells = [start], []
        seen.add(start)
        while stack:
            x, y = stack.pop()
            cells.append((x, y))
            for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in seen and grid[ny][nx] == color:
                    seen.add((nx, ny))
                    stack.append((nx, ny))
        out.append((color, frozenset(cells)))
    return out


def receipt_box(cells):
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    return min(xs), min(ys), max(xs), max(ys)


def receipt_shape(cells):
    x0, y0, _, _ = receipt_box(cells)
    return frozenset((x - x0, y - y0) for x, y in cells)


def receipt_extent(cells):
    x0, y0, x1, y1 = receipt_box(cells)
    return "%dx%d (%d cells) at (%d,%d)-(%d,%d)" % (x1 - x0 + 1, y1 - y0 + 1, len(cells), x0, y0, x1, y1)


def receipt_describe(color, cells):
    return "color %x %s" % (color, receipt_extent(cells))


def receipt_objects(before, after, cells):
    """before → after 在 cells 上的變化，以物件描述。回傳（列，{大區域顏色: 淨格數}）。"""
    # 只從變了的格子出發會漏掉沒變的那一半：挖了洞的方塊剩下的環、併進來的鄰居都不含變了的格子。
    # 所以兩邊互相補種子：一邊的小物件裡顏色沒變的格子，也拿去另一邊找它所在的整塊。
    def kept(grid, comps):
        return {p for color, region in comps if len(region) <= RECEIPT_LARGE
                for p in region if grid[p[1]][p[0]] == color}
    old = receipt_components(before, cells)
    new = receipt_components(after, cells | kept(after, old))
    old = receipt_components(before, cells | kept(before, new))
    # 大區域只報淨格數有變的顏色：物件在背景上移動時背景形狀一定變，但淨數是 0，不值一行。
    large = {}
    for color in {c for c, region in old + new if len(region) > RECEIPT_LARGE}:
        net = sum(after[y][x] == color for x, y in cells) - sum(before[y][x] == color for x, y in cells)
        if net:
            large[color] = net
    old = [c for c in old if len(c[1]) <= RECEIPT_LARGE]
    new = [c for c in new if len(c[1]) <= RECEIPT_LARGE]
    lines, used = [], set()
    for color, cells_old in old[:]:
        shape = receipt_shape(cells_old)
        bx, by, _, _ = receipt_box(cells_old)
        best = None
        for i, (color_new, cells_new) in enumerate(new):
            if i in used or color_new != color or len(cells_new) != len(cells_old) \
                    or receipt_shape(cells_new) != shape or cells_new == cells_old:
                continue
            ax, ay, _, _ = receipt_box(cells_new)
            distance = abs(ax - bx) + abs(ay - by)
            if best is None or distance < best[0]:
                best = (distance, i, ax - bx, ay - by)
        if best is not None:
            used.add(best[1])
            old.remove((color, cells_old))
            lines.append("moved: %s by dx=%+d dy=%+d" % (receipt_describe(color, cells_old), best[2], best[3]))
    for color, cells_old in old[:]:
        for i, (color_new, cells_new) in enumerate(new):
            if i not in used and cells_new == cells_old and color_new != color:
                used.add(i)
                old.remove((color, cells_old))
                lines.append("recolored: %s -> color %x" % (receipt_describe(color, cells_old), color_new))
                break
    moved_or_recolored = set(used)
    for color, cells_old in old[:]:
        bx0, by0, bx1, by1 = receipt_box(cells_old)
        for i, (color_new, cells_new) in enumerate(new):
            # 兩塊併成一塊時兩個舊物件都對到同一個新物件，所以這裡只排除已經算成移動／變色的
            if i in moved_or_recolored or color_new != color:
                continue
            ax0, ay0, ax1, ay1 = receipt_box(cells_new)
            if ax0 <= bx1 and bx0 <= ax1 and ay0 <= by1 and by0 <= ay1:
                used.add(i)
                old.remove((color, cells_old))
                if receipt_box(cells_new) == receipt_box(cells_old):
                    lines.append("changed in place: %s, now %d cells (%d cells differ)" % (
                        receipt_describe(color, cells_old), len(cells_new), len(cells_old ^ cells_new)))
                else:
                    lines.append("reshaped: %s -> %s" % (receipt_describe(color, cells_old),
                                                         receipt_extent(cells_new)))
                break
    lines += ["disappeared: %s" % receipt_describe(c, cells) for c, cells in old]
    lines += ["appeared: %s" % receipt_describe(c, cells) for i, (c, cells) in enumerate(new) if i not in used]
    return lines, large


def receipt_transient(step, previous):
    """這一步的動畫裡變了、最後又變回動作前樣子的格子（只有看動畫才知道）。"""
    frames = receipt_frames(step)
    if len(frames) < 2:
        return set()
    prev = [bytes(row) for row in previous]
    final = frames[-1]
    out = set()
    for frame in frames[:-1]:
        for y, row in enumerate(frame):
            if row == prev[y]:
                continue
            for x in range(len(row)):
                if row[x] != prev[y][x] and final[y][x] == prev[y][x]:
                    out.add((x, y))
    return out


def receipt_scene(board):
    """整張盤面的物件清單（新關卡開局用；config ``act_receipt_scene``）：同色同形的歸成一種、列出位置。"""
    height, width = len(board), len(board[0])
    comps = receipt_components(board, {(x, y) for y in range(height) for x in range(width)})
    large = sorted(((len(cells), color) for color, cells in comps if len(cells) > RECEIPT_LARGE), reverse=True)
    kinds = {}
    for color, cells in comps:
        if len(cells) <= RECEIPT_LARGE:
            kinds.setdefault((color, receipt_shape(cells)), []).append(cells)
    order = sorted(kinds.items(), key=lambda kv: (-len(next(iter(kv[1]))), -len(kv[1]), kv[0][0]))
    count = sum(len(members) for members in kinds.values())
    out = ["  %d objects of %d kinds (same colour, 4-connected; x,y = column,row of the top-left corner):"
           % (count, len(kinds))]
    for (color, shape), members in order[:RECEIPT_SCENE_LINES]:
        cells = members[0]
        x0, y0, x1, y1 = receipt_box(cells)
        spots = sorted(receipt_box(m)[:2] for m in members)
        edge = all(receipt_box(m)[0] >= width - RECEIPT_EDGE or receipt_box(m)[1] >= height - RECEIPT_EDGE
                   or receipt_box(m)[2] < RECEIPT_EDGE or receipt_box(m)[3] < RECEIPT_EDGE for m in members)
        out.append("  %scolor %x %dx%d (%d cells) at %s%s%s" % (
            "%dx " % len(members) if len(members) > 1 else "", color, x1 - x0 + 1, y1 - y0 + 1, len(cells),
            " ".join("(%d,%d)" % xy for xy in spots[:RECEIPT_SCENE_SPOTS]),
            " ..." if len(spots) > RECEIPT_SCENE_SPOTS else "",
            " [edge band]" if edge else ""))
    if len(order) > RECEIPT_SCENE_LINES:
        out.append("  ... and %d more kinds" % (len(order) - RECEIPT_SCENE_LINES))
    if large:
        out.append("  large regions: " + ", ".join("color %x (%d cells)" % (c, n) for n, c in large))
    return out


def receipt(first, last, cleared, steps, scene=False):
    """steps：這一批每一步的 (step, frames, changed_cells)。回傳要印的幾行。"""
    if cleared and scene:
        return (["level cleared: the new level's board (record/current.png) has these objects -- "
                 "look at the image too before planning"] + receipt_scene(receipt_board(last)))
    if cleared:
        return ["level cleared: record/current.png is the next level's first board -- look at it before planning"]
    before = receipt_board(first - 1)
    after = receipt_board(last)
    height, width = len(after), len(after[0])
    changed = {(x, y) for y in range(height) for x in range(width) if before[y][x] != after[y][x]}
    edge = {(x, y) for x, y in changed if x < RECEIPT_EDGE or y < RECEIPT_EDGE
            or x >= width - RECEIPT_EDGE or y >= height - RECEIPT_EDGE}
    inner = changed - edge
    span = "step %d" % last if first == last else "steps %d-%d" % (first, last)
    out = ["what changed over %s (board before it vs now; x,y = column,row):" % span]
    if not changed:
        out.append("  nothing: the board is identical to the one before this batch")
    elif len(inner) > RECEIPT_SCREEN:
        x0, y0, x1, y1 = receipt_box(inner)
        out.append("  most of the board changed (%d cells in (%d,%d)-(%d,%d)): a new screen or a reset -- "
                   "look at record/current.png" % (len(changed), x0, y0, x1, y1))
    else:
        lines, large = receipt_objects(before, after, inner)
        if len(lines) > RECEIPT_MAX_LINES:
            extra = len(lines) - RECEIPT_MAX_LINES
            lines = lines[:RECEIPT_MAX_LINES] + ["... and %d more object changes" % extra]
        out += ["  " + line for line in lines]
        for color, net in sorted(large.items()):
            out.append("  large color %x region (over %d cells, e.g. background): %+d cells" % (color, RECEIPT_LARGE, net))
        if edge:
            x0, y0, x1, y1 = receipt_box(edge)
            out.append("  edge band only (within %d cells of the border, often a counter or timer): "
                       "%d cells in (%d,%d)-(%d,%d)" % (RECEIPT_EDGE, len(edge), x0, y0, x1, y1))
        if not inner:
            out.append("  nothing changed away from the edge band")
    still = [s for s, frames, cells in steps if cells == 0 and frames <= 1]
    if still:
        out.append("no visible change at step%s %s" % ("s" if len(still) > 1 else "",
                                                       ", ".join(str(s) for s in still)))
    shown = 0
    for step, frames, cells in steps:
        if frames < 2 or shown >= 4:
            continue
        gone = receipt_transient(step, receipt_board(step - 1))
        if gone:
            shown += 1
            x0, y0, x1, y1 = receipt_box(gone)
            out.append("step %d animated over %d frames: %d cells changed and changed back in (%d,%d)-(%d,%d); "
                       "only its frames show this" % (step, frames, len(gone), x0, y0, x1, y1))
    return out
