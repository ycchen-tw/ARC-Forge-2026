"""recap：從 record/ 讀出「現在在哪」——這一關到目前的每一批（plan、動作、變化格數）、已過的關、
目前盤面的物件（同色同形歸一種）。

動機：c0i-r2 每次 compaction 之後中位要 12 次呼叫才再出手，讀 record/ 重建這一關的狀態最多（678 次）。
stdlib only：act.py（jail 裡沒有 numpy）可以像 receipt.py 一樣把原始碼嵌進去，arcgame（c0L）呼叫
同一份（write_library 寫成 agent_lib/_arcrecap.py），兩條臂看到的 recap 一字不差。函式都帶 ``recap_``
前綴，嵌進 act.py 時不撞名。座標 x,y＝欄,列，與簡報一致。
"""

import json
from pathlib import Path

RECAP_LARGE = 512         # 比這大的連通塊當成背景／大區域（與 receipt.py、arcgame 同）
RECAP_EDGE = 4            # 離邊框這麼近的格子是邊框帶（計數器、計時條常在那裡）
RECAP_SPOTS = 12          # 每種物件最多列幾個位置


def recap_components(grid):
    """同色 4 連通塊：[(color, [(x, y), ...])]，依最上、最左的格子排序。"""
    height, width = len(grid), len(grid[0])
    seen, out = set(), []
    for y0 in range(height):
        for x0 in range(width):
            if (x0, y0) in seen:
                continue
            color = grid[y0][x0]
            stack, cells = [(x0, y0)], []
            seen.add((x0, y0))
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                    if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in seen and grid[ny][nx] == color:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            out.append((color, cells))
    return out


def recap_scene(grid, limit=16):
    """整張盤面的物件依種類（同色同形）分組、各在哪（左上角 x,y），大的種類在前；大區域最後一行。"""
    height, width = len(grid), len(grid[0])
    comps = recap_components(grid)
    kinds = {}
    for color, cells in comps:
        if len(cells) > RECAP_LARGE:
            continue
        x0 = min(c[0] for c in cells)
        y0 = min(c[1] for c in cells)
        shape = frozenset((x - x0, y - y0) for x, y in cells)
        kinds.setdefault((color, shape), []).append(cells)
    order = sorted(kinds.items(), key=lambda kv: (-len(kv[1][0]), -len(kv[1]), kv[0][0]))
    out = ["%d objects of %d kinds (same colour, 4-connected; x,y = column,row of the top-left corner):"
           % (sum(len(m) for _, m in order), len(order))]
    for (color, shape), members in order[:limit]:
        w = max(c[0] for c in shape) + 1
        h = max(c[1] for c in shape) + 1
        boxes = [(min(c[0] for c in m), min(c[1] for c in m), max(c[0] for c in m), max(c[1] for c in m))
                 for m in members]
        spots = sorted(b[:2] for b in boxes)
        edge = all(b[0] >= width - RECAP_EDGE or b[1] >= height - RECAP_EDGE or b[2] < RECAP_EDGE
                   or b[3] < RECAP_EDGE for b in boxes)
        out.append("  %scolor %x %dx%d (%d cells) at %s%s%s" % (
            "%dx " % len(members) if len(members) > 1 else "", color, w, h, len(shape),
            " ".join("(%d,%d)" % xy for xy in spots[:RECAP_SPOTS]), " ..." if len(spots) > RECAP_SPOTS else "",
            " [edge band]" if edge else ""))
    if len(order) > limit:
        out.append("  ... and %d more kinds" % (len(order) - limit))
    large = sorted(((len(cells), color) for color, cells in comps if len(cells) > RECAP_LARGE), reverse=True)
    if large:
        out.append("  large regions: " + ", ".join("color %x (%d cells)" % (c, n) for n, c in large))
    return "\n".join(out)


def recap_token(row):
    if row.get("action") == "ACTION6" and row.get("x") is not None:
        return "ACTION6:%s,%s" % (row["x"], row["y"])
    return str(row.get("action"))


def recap_text(record, status, batches=8):
    """``record``：record/ 目錄；``status``：middleware 的 /api/status（原樣）。回傳要印的文字。"""
    record = Path(record)
    rows = [json.loads(line) for line in (record / "history.jsonl").read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("action") != "START"]
    done = status.get("levels_completed") or 0
    level = done + 1
    here = [r for r in rows if r.get("level") == level]
    out = ["level %d of %s, state=%s, %d actions used in all, %d on this level; available: %s" % (
        level, status.get("win_levels"), status.get("state"), status.get("actions_used", 0), len(here),
        ", ".join("ACTION%d" % a for a in (status.get("available_actions") or [])))]
    cleared, start = [], 0
    for r in rows:
        if (r.get("levels_completed") or 0) > len(cleared):
            cleared.append("L%d at step %d (%d actions)" % (len(cleared) + 1, r["step"], r["step"] - start))
            start = r["step"]
    if cleared:
        out.append("cleared: " + ", ".join(cleared))
    groups = []
    for r in here:
        if r.get("plan") or not groups:
            groups.append({"plan": r.get("plan") or "(no plan)", "rows": []})
        groups[-1]["rows"].append(r)
    if groups:
        out.append("this level's batches%s (steps: plan -> actions [cells changed]):" % (
            ", the last %d of %d" % (batches, len(groups)) if len(groups) > batches else ""))
    for g in groups[-batches:]:
        first, last = g["rows"][0]["step"], g["rows"][-1]["step"]
        acts = " ".join(recap_token(r) for r in g["rows"][:12]) + (" ..." if len(g["rows"]) > 12 else "")
        cells = ",".join("?" if r.get("changed_cells") is None else str(r["changed_cells"]) for r in g["rows"][:12])
        end = "" if g["rows"][-1].get("state") == "NOT_FINISHED" else " -> %s" % g["rows"][-1].get("state")
        out.append("  %s: %s -> %s [%s]%s" % (first if first == last else "%d-%d" % (first, last),
                                              g["plan"][:160], acts, cells, end))
    board = [[int(c, 16) for c in line.strip()] for line in (record / "current.txt").read_text().splitlines()
             if line.strip()]
    out.append(recap_scene(board))
    return "\n".join(out)
