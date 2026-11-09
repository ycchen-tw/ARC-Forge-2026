"""ARC Prize `submission.json` 的落地，naive（`run_submission.py`）與 agentic（`agentic_tir.py`）共用。

官方格式：`{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}`，list 與題目的 `test` 順序對齊。
每一格都必須有兩個 attempt，所以建檔時先用 test input 本身填滿（identity 偶爾是對的，形狀一定合法），
之後每收到一個答案就整份重寫（tmp + rename，被砍在寫到一半不會留下半份 JSON）。

grid 在寫進去之前要驗：非空矩形、每格 0–9 的整數、邊長 ≤ 30。舊版不驗——模型吐出參差列或浮點數會
原封不動寫進檔案，而計分器對壞格子的處置沒有文件保證。不合法的答案**留著 fallback 不動**並記在
`rejected`，呼叫端要印出來。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

MAX_SIDE = 30


def validate_grid(g) -> str | None:
    """合法回 None，否則回一句原因。bool 是 int 的子類別，要明確擋掉。"""
    if not isinstance(g, list) or not g:
        return "not a non-empty list"
    if not all(isinstance(r, list) and r for r in g):
        return "every row must be a non-empty list"
    w = len(g[0])
    if any(len(r) != w for r in g):
        return f"ragged rows ({sorted({len(r) for r in g})})"
    if len(g) > MAX_SIDE or w > MAX_SIDE:
        return f"{len(g)}x{w} exceeds {MAX_SIDE}x{MAX_SIDE}"
    for r in g:
        for v in r:
            if isinstance(v, bool) or not isinstance(v, int) or not 0 <= v <= 9:
                return f"cell {v!r} is not an int in 0..9"
    return None


class SubmissionStore:
    """`{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}`，每次更新就落地。"""

    def __init__(self, challenges: dict, out_path: Path):
        self.out = Path(out_path)
        self.data = {
            tid: [{"attempt_1": t["input"], "attempt_2": t["input"]} for t in task["test"]]
            for tid, task in challenges.items()
        }
        self.filled: set[tuple[str, int, int]] = set()
        self.rejected: list[dict] = []
        self.flush()

    def put(self, task_id: str, test_idx: int, attempt: int, grid) -> bool:
        """寫一格。回傳有沒有真的寫進去（None 或不合法的 grid 都不寫、不動 fallback）。"""
        if grid is None:
            return False
        why = validate_grid(grid)
        if why is not None:
            self.rejected.append({"task_id": task_id, "test_idx": test_idx, "attempt": attempt, "why": why})
            return False
        self.data[task_id][test_idx][f"attempt_{attempt + 1}"] = grid
        self.filled.add((task_id, test_idx, attempt))
        self.flush()
        return True

    def put_attempts(self, task_id: str, test_idx: int, grids: list) -> int:
        """一次給這一格的 attempt 清單（順序就是優先序，只取前兩個）。回傳寫進去的格數。"""
        return sum(int(self.put(task_id, test_idx, a, g)) for a, g in enumerate(grids[:2]))

    def has(self, task_id: str, test_idx: int, attempt: int) -> bool:
        return (task_id, test_idx, attempt) in self.filled

    @property
    def n_cells(self) -> int:
        return sum(len(v) for v in self.data.values())

    def flush(self) -> None:
        tmp = self.out.with_suffix(self.out.suffix + ".tmp")
        tmp.write_text(json.dumps(self.data))
        os.replace(tmp, self.out)

