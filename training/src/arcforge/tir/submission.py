"""Persisting the ARC Prize `submission.json`, shared by naive (`run_submission.py`) and agentic (`agentic_tir.py`).

Official format: `{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}`, the list aligned with the order of the
task's `test`. Every slot must have two attempts, so the file is first filled with the test input itself (identity is
occasionally right, and the shape is always valid), then the whole file is rewritten on every answer received (tmp +
rename, so being killed mid-write never leaves half a JSON).

Grids are validated before being written: non-empty rectangle, integer cells 0–9, sides ≤ 30. The old version did not
validate -- ragged rows or floats from the model went into the file as is, and the scorer's handling of bad grids has no
documented guarantee. An invalid answer **leaves the fallback untouched** and is recorded in `rejected`, which the caller
must print.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

MAX_SIDE = 30


def validate_grid(g) -> str | None:
    """None if valid, otherwise a one-line reason. bool is a subclass of int and must be rejected explicitly."""
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
    """`{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}`, persisted on every update."""

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
        """Write one slot. Returns whether it was actually written (None or an invalid grid writes nothing and leaves the fallback alone)."""
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
        """Give this slot's attempt list at once (order is priority; only the first two are taken). Returns how many were written."""
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
