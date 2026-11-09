"""Compute RHAE from the server-side ledger.

The formula is written against the vendored official ``arc_agi/scorecard.py``, not copied from the old scorer:

``add_level`` (``scorecard.py:146-183``) is called for **every level** -- uncleared levels score 0
**but their weight still counts**. So the denominator of ``to_score`` (``:185-208``) is the weight of all
levels, not of the cleared ones. You cannot see this from ``to_score`` alone; read it together with ``add_level``.

Scoring reads the ledger, not the agent's ``logs.txt``: the ledger is written on the other side of the
socket, out of reach from the jail. So bypassing ``act.py`` does not affect scoring -- which is exactly the
premise for letting direct connections be legitimate.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

from .ledger import LOCAL_GATEWAY


def baselines(corpus: Path) -> dict[str, list[int]]:
    """``baseline_actions`` from ``environment_files/<game>/<version>/metadata.json``.

    This field is **not on the API** (``api.py:56-64`` explicitly excludes it), so RHAE can only be computed
    locally; in the competition rerun the only score is the card the organiser closes.
    """
    out: dict[str, list[int]] = {}
    for path in glob.glob(str(corpus / "*" / "*" / "metadata.json")):
        data = json.loads(Path(path).read_text())
        out[data["game_id"]] = data["baseline_actions"]
    if not out:
        raise RuntimeError(f"no metadata.json under {corpus}")
    return out


def short_index(full_ids) -> dict[str, str]:
    """Map both full and short names to the full name. ``ar25`` and ``ar25-0c556536`` both give ``ar25-0c556536``.

    **Three places must use the same copy**: the gateway resolving the ``game_id`` the agent sends, the launcher
    building the roster, and finalise computing the denominator. In the 2026-09-18 four-arm comparison finalise
    had its own copy that picked by ``glob()`` order, so ``ar25`` resolved to ``e3c63847``, never played -- the
    denominator gained a phantom 0 and lost that game's full score. There should be one copy of the rule;
    writing a second is betting that they agree.

    When a game has several versions, take the lexicographically smallest. **This is arbitrary**, not "pick the
    newest" -- see the corpus section of ``docs/research/arc_code_4arm_20260918.md``.
    """
    ids = sorted(full_ids)
    out: dict[str, str] = {full: full for full in ids}
    for full in ids:
        out.setdefault(full.split("-")[0], full)
    return out


def level_score(actions: int, baseline: int) -> float:
    """``scorecard.py:168-171``, verbatim."""
    if actions <= 0:
        return 0.0
    return min(100.0 * (baseline / actions) ** 2, 115.0)


def game_score(spans: list[int], base: list[int]) -> float:
    """Score of one play of one game.

    ``spans`` is the actions spent on each level **cleared in order**; ``base`` is the baselines of all the game's
    levels. Uncleared levels are not in spans, but their weight is still in the denominator -- that is the else
    branch of ``add_level``.
    """
    n = len(base)
    if n == 0:
        return 0.0
    total_weight = sum(range(1, n + 1))
    done_weight = sum(range(1, len(spans) + 1))
    if done_weight == 0:
        return 0.0
    earned = sum((k + 1) * level_score(spans[k], base[k])
                 for k in range(min(len(spans), n)))
    # min(mean, share of cleared-level weight × 100) -- the official max_score cap.
    return min(earned / total_weight, 100.0 * done_weight / total_weight)


def plays_from_ledger(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split the ledger into "every play of every game".

    A play boundary is where the ``play`` field increments (the gateway increments it on a full reset).
    Competition mode refuses replays, so each game has only one play -- but they are still computed separately
    here, because the official rule is ``max(run.score for run in runs)``, not concatenating the action streams.
    """
    games: dict[str, dict[int, dict[str, Any]]] = {}
    for row in rows:
        if not row.get("counted"):
            continue
        play = games.setdefault(row["game"], {}).setdefault(
            int(row.get("play", 1)),
            {"spans": [], "actions": 0, "mark": 0, "levels": 0,
             "state": None, "won": False})
        play["actions"] += 1
        play["state"] = row.get("state")
        # "Has won" is not "the state at the last moment". After ar25 cleared all eight levels, a resumed session
        # sent one more RESET, so the last row went back to NOT_FINISHED -- reading the last row would miss this win.
        if row.get("state") == "WIN":
            play["won"] = True
        done = int(row.get("levels_completed") or 0)
        while done > play["levels"]:
            play["spans"].append(play["actions"] - play["mark"])
            play["mark"] = play["actions"]
            play["levels"] += 1
    return {game: [plays[k] for k in sorted(plays)] for game, plays in games.items()}


def score_run(rows: list[dict[str, Any]], base: dict[str, list[int]],
              games: list[str] | None = None) -> dict[str, Any]:
    """Score of a round. When ``games`` is given it is the denominator, and untouched games score 0."""
    authorities = {row.get("authority") for row in rows} or {LOCAL_GATEWAY}
    if len(authorities) > 1:
        # A score computed from records of mixed sources is meaningless; better to refuse than silently give a number.
        raise RuntimeError(f"ledger mixes authorities {sorted(authorities)}")
    per_play = plays_from_ledger(rows)
    # `games or ...`: `[]` and None take the same path, and the denominator silently falls back to "games seen in the
    # ledger", which rewards failure (games that never opened are not in the denominator). Measured: 100.0 vs 50.0 on the same ledger.
    if games is not None and len(games) == 0:
        raise RuntimeError("roster is empty: nothing in the denominator; refusing to compute a plausible-looking number")
    names = sorted(per_play) if games is None else games
    # The roster must be full names from the corpus. When a short name (``ar25``) slips in it does **not** blow up on the
    # KeyError below -- `per_play.get()` misses first and takes the "never played" branch's continue,
    # so the whole round gets a plausible-looking score of 0. In the 2026-09-18 four-arm run all four summary.json files
    # were `mean_score: 0.0` with all 25 games marked never played, without a single error message.
    # Block it here: a denominator that does not match the corpus is a bug, not a state to keep computing in.
    unknown = [name for name in names if name not in base]
    if unknown:
        raise RuntimeError(
            f"roster has {len(unknown)}/{len(names)} games not in the corpus: "
            f"{unknown[:3]}{'…' if len(unknown) > 3 else ''}. "
            "Short and full names mixed? Resolve them with score.short_index() before passing them in.")
    out: list[dict[str, Any]] = []
    for game in names:
        plays = per_play.get(game, [])
        if not plays:
            out.append({"game_id": game, "score": 0.0, "levels": 0, "actions": 0,
                        "plays": 0, "state": None, "won": False,
                        "note": "never played"})
            continue
        try:
            b = base[game]
        except KeyError:
            raise RuntimeError(f"{game} is not in the corpus") from None
        scored = [game_score(p["spans"], b) for p in plays]
        best = max(range(len(scored)), key=lambda i: scored[i])
        out.append({
            "game_id": game,
            # Official rule: the score is the best play's, the action count is the sum over all plays
            # (EnvironmentScoreList.score = max(...), .actions = sum(...)).
            "score": scored[best],
            "levels": plays[best]["levels"],
            "actions": sum(p["actions"] for p in plays),
            "plays": len(plays),
            "state": plays[best]["state"],
            "won": plays[best]["won"],
        })
    mean = sum(row["score"] for row in out) / len(out) if out else 0.0
    return {
        "mean_score": round(mean, 4),
        "games_scored": len(out),
        "total_levels": sum(row["levels"] for row in out),
        "total_actions": sum(row["actions"] for row in out),
        "wins": sum(1 for row in out if row.get("won")),
        "source": "server-ledger",
        "authority": next(iter(authorities)),
        "games": out,
    }
