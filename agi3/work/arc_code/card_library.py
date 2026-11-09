"""Game cards: one card per game, for the reviewer to borrow ideas from other games (user 10-31, hs110 v40).

Two sources, one format (``SECTIONS``):

- static: ``work/arc_code/game_cards/<ID>.md``, the 25 public games, written offline from 37-39 played copies each
  and checked against the public rule books (``$ARC_RUNS/agi3_game_cards_20261031``);
- live: during the run a card writer (``review_scheduler.run_card``, prompts ``review_prompts/c1``) turns one game's
  notes and level documents into ``<run-dir>/cards/live/<game>.md`` after it clears levels.

``assemble`` builds a reviewer's ``game_cards/``: every card except those of the reviewed game's base game (the part
of the id before the first ``-``: copies of one public game in our own runs; in the rerun every game is distinct and
nothing else is left out), at most one live card per base game (the one with the most levels cleared), and an
``INDEX.md`` with one line per card.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

UNIT = Path(__file__).resolve().parent
STATIC_DIR = UNIT / "game_cards"
#: in a card's text, in this order
SECTIONS = ("Keywords:", "Looks like:", "## Rules", "## Levels", "## Blind spots", "## Breakthroughs",
            "## Lessons that carry to other games")
CARD_MIN_CHARS = 400
CARD_MAX_CHARS = 14_000
NOT_CARDS = ("README.md", "INDEX.md")
INDEX = "INDEX.md"
#: per card in INDEX.md: the keywords line is cut here
KEYWORDS_MAX_CHARS = 240


def base_id(game: str) -> str:
    """``ka59-c238d34dbb`` and ``KA59`` -> ``ka59``."""
    return game.split("-", 1)[0].lower()


def check_card(text: str) -> str | None:
    """None when ``text`` is a card; otherwise what is wrong with it."""
    body = text.strip()
    if not body.startswith("# "):
        return "the card does not start with a '# ' title line"
    if not CARD_MIN_CHARS <= len(body) <= CARD_MAX_CHARS:
        return f"the card is {len(body)} characters, not {CARD_MIN_CHARS}-{CARD_MAX_CHARS}"
    at = 0
    for section in SECTIONS:
        m = re.compile(r"^" + re.escape(section), re.M).search(body, at)
        if m is None:
            return f"no line starting with {section!r} after the previous section"
        at = m.end()
    return None


def _fields(text: str) -> dict:
    title = text.strip().splitlines()[0][2:].strip()
    kw = re.search(r"^Keywords:\s*(.+)$", text, re.M)
    if kw is None:
        raise ValueError(f"card {title!r} has no Keywords line")
    keywords = kw.group(1).strip()
    if len(keywords) > KEYWORDS_MAX_CHARS:
        keywords = keywords[:KEYWORDS_MAX_CHARS].rsplit(",", 1)[0] + ", ..."
    return {"title": title, "keywords": keywords}


def static_cards(root: Path) -> dict[str, Path]:
    """``{base id: path}`` of the shipped cards; a file that is not a card is a build error, not skipped."""
    out = {}
    for path in sorted(root.glob("*.md")):
        if path.name in NOT_CARDS:
            continue
        problem = check_card(path.read_text(encoding="utf-8"))
        if problem is not None:
            raise ValueError(f"{path}: {problem}")
        out[base_id(path.stem)] = path
    if not out:
        raise FileNotFoundError(f"no game cards in {root}")
    return out


def live_cards(live: Path) -> dict[str, tuple[Path, dict]]:
    """``{base id: (card, meta)}``: for each base game the live card with the most levels cleared (ties: newest)."""
    out: dict[str, tuple[Path, dict]] = {}
    if not live.is_dir():
        return out
    for meta_path in sorted(live.glob("*.json")):
        meta = json.loads(meta_path.read_text())
        card = meta_path.with_suffix(".md")
        if not card.is_file():
            raise FileNotFoundError(f"{meta_path} without its card {card}")
        key = base_id(meta["game"])
        rank = (meta["levels_completed"], meta["written_unix"])
        if key not in out or rank > (out[key][1]["levels_completed"], out[key][1]["written_unix"]):
            out[key] = (card, meta)
    return out


def assemble(dest: Path, game: str, *, static: Path, live: Path) -> dict:
    """``dest`` (must not exist) gets the cards the reviewer of ``game`` may read and their ``INDEX.md``."""
    own = base_id(game)
    dest.mkdir()
    lines = []
    shipped = {k: p for k, p in static_cards(static).items() if k != own}
    for key, path in sorted(shipped.items()):
        text = path.read_text(encoding="utf-8")
        name = f"{key.upper()}.md"
        (dest / name).write_text(text, encoding="utf-8")
        f = _fields(text)
        lines.append(f"- `{name}` | {f['title']} | {f['keywords']} | public game, written from many plays")
    played = {k: v for k, v in live_cards(live).items() if k != own}
    for key, (path, meta) in sorted(played.items()):
        text = path.read_text(encoding="utf-8")
        name = f"run-{meta['game']}.md"
        (dest / name).write_text(text, encoding="utf-8")
        f = _fields(text)
        lines.append(f"- `{name}` | {f['title']} | {f['keywords']} | played in this run: "
                     f"{meta['levels_completed']} of {meta['win_levels']} levels cleared")
    header = ("# Game cards\n\n"
              "One card per game that has been played: what it looked like, its rules, where its player lost "
              "time, what cracked its levels, and lessons for other games. These are other games: an idea from a "
              "card is a hypothesis to test here, never evidence about this game. Read a card with "
              "`cat game_cards/<file>`; search them with `grep -il <word> game_cards/*.md`.\n\n"
              "file | title | keywords | source\n\n")
    (dest / INDEX).write_text(header + "\n".join(lines) + "\n", encoding="utf-8")
    return {"static": len(shipped), "live": len(played), "excluded": own}


def level_stats(rows: list[dict]) -> str:
    """``LEVEL_STATS.md`` for the card writer: per cleared level, minutes, actions, RESETs and GAME_OVERs, from history."""
    out = ["# Level statistics of the cleared levels (from record/history.jsonl)\n",
           "| level | cleared | minutes | actions | RESETs | GAME_OVERs |", "|---|---|---|---|---|---|"]
    start_t, n, resets, overs, level = float(rows[0]["time"]), 0, 0, 0, 1
    done = int(rows[0]["levels_completed"])
    for prev, row in zip(rows, rows[1:]):
        if row["action"] == "RESET":
            resets += 1
        elif row["action"] != "START":
            n += 1
        if row["state"] == "GAME_OVER" and prev["state"] != "GAME_OVER":
            overs += 1
        if int(row["levels_completed"]) > done:
            out.append(f"| {level} | yes | {(float(row['time']) - start_t) / 60:.0f} | {n} | {resets} | {overs} |")
            done, level, start_t, n, resets, overs = int(row["levels_completed"]), level + 1, float(row["time"]), 0, 0, 0
    last = rows[-1]
    # the level in progress is not written (the user, 2026-10-31: a card records only cleared levels)
    out.append(f"\n{done} of {last['win_levels']} levels cleared.\n")
    return "\n".join(out) + "\n"


def publish(live: Path, game: str, text: str, meta: dict) -> Path:
    """Write ``<live>/<game>.md`` and its ``.json`` atomically (the card first: ``live_cards`` keys on the json)."""
    problem = check_card(text)
    if problem is not None:
        raise ValueError(problem)
    live.mkdir(parents=True, exist_ok=True)
    card = live / f"{game}.md"
    for path, data in ((card, text.strip() + "\n"),
                       (live / f"{game}.json", json.dumps({"game": game, **meta, "written_unix": time.time()}))):
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(data, encoding="utf-8")
        os.replace(tmp, path)
    return card
