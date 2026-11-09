"""Generates the agent's ``act.py``: a stdlib-only convenience client that talks to the **middleware**.

Reworked 2026-09-18. The previous act.py faced a gateway that mimicked the official wire, so it had
to manage the guid and the cookie itself; both went wrong (lost guid -> permanent 400; cookie not
saved -> two games wrecked for six hours each). Now the middleware holds the guid for the agent, does
the one opening RESET on its behalf, blocks illegal actions and rate-limits -- the agent side is left
with only "send actions, record boards".

Three things remain deliberate:

1. **stdlib only.** The jail has no harness venv, nor should it.
2. **No action-budget check, no replays.** The authority on the former is the official card; the
   latter is refused by the official gateway.
3. **The agent may skip it.** The middleware's interface is HTTP; writing your own client is entirely
   legitimate and scores the same -- because the score does not come from here at all.

``logs.txt`` is the agent's memory (upstream docstring: *"written by code, read by code"*),
not a ledger. The agent ``cat``-ing this file to learn the interface is **expected behaviour**.

``RECORD_TEMPLATE`` is the version used when the middleware keeps the record (``record.py``): the
middleware writes the boards into ``record/``, act no longer writes ``logs.txt``, and ``--plan`` goes
into the record with the batch's first action. ``TEMPLATE`` is not touched by a single character --
arms without a record get an act.py bit-identical to before.

Since 2026-09-28 (prompts v12) ``RECORD_TEMPLATE`` is a **pure tool**: all English, only usage and
facts, no play guidance (the reminder to write levels/Lnn.md was removed; the briefing has it). The old
docstring was in Chinese and still said "writing your own client is entirely legitimate" with an API
list attached -- in v11, 16/34 games read it at the start, in direct conflict with the developer rule
(only use ./act). Point 3 above now holds only for ``TEMPLATE``.
"""

from __future__ import annotations

from pathlib import Path

#: The generated file. Deliberately something readable, copyable and bypassable.
#: Model-facing (written into the agent's workspace), kept verbatim as deployed, including its Chinese docstring
#: (usage, the server's HTTP interface, and "writing your own client is entirely legitimate") and comments.
TEMPLATE = '''#!/usr/bin/env python3
"""ARC-AGI-3 的便利 actuator。stdlib only。

你不一定要用這個檔案。環境是一個一般的 HTTP 服務（$ARC_BASE_URL），你手上有
$ARC_API_KEY，自己寫 client 直接打它完全合法，分數一樣算。這個檔案只是把
「送一個動作、把盤面寫進 logs.txt」包起來。

    ./act init                 # 開局（免費），初始盤面進 logs.txt
    ./act do ACTION1 ACTION3 ACTION6:24,20 --plan "what this batch tests"
    ./act status               # 免費：關卡、已用動作、目前可用的動作
    ./act board                # 免費：印目前盤面

你的 key 只對應這一局，所以 game_id 不用給。伺服器替你記著開局用的識別碼，
你不需要知道它。

伺服器的介面（你自己寫 client 時用得到）：
    GET  /api/status                        免費，回 {state, levels_completed,
                                            win_levels, available_actions,
                                            actions_used, frame}
    POST /api/cmd/RESET                     還沒開局 = 開局（免費）；
                                            開局後 = 重置**這一關**（算 1 動作）；
                                            state 是 GAME_OVER 時就是用它繼續
    POST /api/cmd/ACTION1..ACTION7          body: {} 或 {"x":.., "y":..}（ACTION6）
    每個請求都帶 header  X-API-Key: $ARC_API_KEY
    不在 available_actions 裡的動作會被擋在伺服器前面（400，不花動作）。
    太快會拿到 429 + Retry-After；等一下再送。
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("ARC_BASE_URL", "http://127.0.0.1:8731").rstrip("/")
API_KEY = os.environ.get("ARC_API_KEY", "")
LOG = "logs.txt"
STATE = "state.json"
SEP = "=" * 80
COORD_ACTION = "ACTION6"
USER_AGENT = "arc-code-act/3"
TERMINAL = ("WIN", "GAME_OVER")

PREAMBLE = """logs.txt — every action you sent and the board it produced.
Format: a line of = , a header (`action N TOKEN state=... level=...`), then the
board as 64 lines of 64 hex digits (one digit per cell, the digit is the cell's
value 0-15). If an action produced several frames they are printed in order as
[anim i/n] ... [final]; the last one is the board you now stand on.
Read this file with code (grep/sed/python), not by eye.
colors: 0=White 1=Off-White 2=Light-Gray 3=Gray 4=Off-Black 5=Black
        6=Magenta 7=Light-Magenta 8=Red 9=Blue a=Light-Blue b=Yellow
        c=Orange d=Maroon e=Green f=Purple
"""


class ActError(SystemExit):
    pass


def _request(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("X-API-Key", API_KEY)
    req.add_header("User-Agent", USER_AGENT)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            if exc.code == 429 and attempt < 3:
                # 伺服器的速率限制。它告訴你要等多久，照做就好。
                wait = float(exc.headers.get("Retry-After") or 1)
                time.sleep(min(wait, 10.0))
                continue
            raise ActError("%s %s -> %s: %s" % (method, path, exc.code, detail)) from None
        except urllib.error.URLError as exc:
            raise ActError("%s %s -> cannot reach %s: %s" % (method, path, BASE_URL, exc.reason)) from None
    raise ActError("%s %s -> still rate-limited after retries" % (method, path))


def get(path):
    return _request("GET", path)


def post(path, payload):
    return _request("POST", path, payload)


def load(ws):
    try:
        return json.loads((ws / STATE).read_text())
    except FileNotFoundError:
        return {"actions_used": 0, "levels": 0}


def save(ws, state):
    (ws / STATE).write_text(json.dumps(state, indent=2))


def grid(layer):
    # 值夾在 0..15：超過會變成兩個字元，把整列往右推，而你是逐字元解析的。
    return "\\n".join("".join("%x" % max(0, min(15, int(v))) for v in row) for row in layer)


def board(frame):
    """所有 frame 層依序印出。多層 = 這個動作有動畫，最後一層是你現在站的盤面。"""
    if not frame:
        return ""
    if len(frame) == 1:
        return grid(frame[0])
    parts = []
    for i, layer in enumerate(frame[:-1], start=1):
        parts.append("[anim %d/%d]" % (i, len(frame) - 1))
        parts.append(grid(layer))
    parts.append("[final]")
    parts.append(grid(frame[-1]))
    return "\\n".join(parts)


def append(ws, header, obs):
    with (ws / LOG).open("a") as handle:
        handle.write("%s\\n%s\\n%s\\n" % (SEP, header, board(obs.get("frame") or [])))


def describe(obs):
    return "state=%s level=%s/%s available=%s" % (
        obs.get("state"), obs.get("levels_completed"), obs.get("win_levels"),
        ["ACTION%d" % a for a in (obs.get("available_actions") or [])])


def cmd_init(ws, args):
    if not (ws / LOG).exists():
        (ws / LOG).write_text(PREAMBLE + "\\n")
    status = get("/api/status")
    if status.get("started"):
        if status.get("state") == "GAME_OVER":
            # 6h 實跑 tu93：這裡以前印「already open, not restarting」，方向剛好相反，
            # agent 因此自己寫了 client 繞過。GAME_OVER 之後該做的是重置這一關。
            print("game is over (GAME_OVER) after %d actions. It is still open: run\\n"
                  "  ./act do RESET\\n"
                  "to restart the current level (costs one action). Read logs.txt first."
                  % status.get("actions_used", 0))
            return
        print("game already open: %s (%d actions used so far). "
              "Not restarting — read logs.txt." % (describe(status), status.get("actions_used", 0)))
        return
    obs = post("/api/cmd/RESET", {})
    state = {"game_id": obs.get("game_id"), "actions_used": 0,
             "levels": obs.get("levels_completed") or 0}
    save(ws, state)
    append(ws, "start %s %s" % (obs.get("game_id"), describe(obs)), obs)
    print("started %s: %s" % (obs.get("game_id"), describe(obs)))


def cmd_do(ws, args):
    state = load(ws)
    before_levels = state.get("levels", 0)
    if args.plan:
        with (ws / LOG).open("a") as handle:
            handle.write("%s\\nplan %s\\n" % (SEP, args.plan))
    for index, token in enumerate(args.actions):
        name, _, coords = token.partition(":")
        payload = {}
        if name == COORD_ACTION:
            if not coords:
                raise ActError("%s needs coordinates, e.g. %s:24,20" % (name, name))
            x, _, y = coords.partition(",")
            payload = {"x": int(x), "y": int(y)}
        obs = post("/api/cmd/%s" % name, payload)
        state["actions_used"] = state.get("actions_used", 0) + 1
        state["levels"] = obs.get("levels_completed") or 0
        save(ws, state)
        append(ws, "action %d %s %s" % (state["actions_used"], token, describe(obs)), obs)
        print("%s -> %s" % (token, describe(obs)))
        if obs.get("state") in TERMINAL:
            break
        if state["levels"] > before_levels:
            dropped = len(args.actions) - index - 1
            if dropped:
                print("level cleared -- dropped the remaining %d action(s); "
                      "the board you planned against is gone" % dropped)
            break


def cmd_status(ws, args):
    status = get("/api/status")
    status.pop("frame", None)
    print(json.dumps(status, indent=2))


def cmd_board(ws, args):
    status = get("/api/status")
    if not status.get("started"):
        raise ActError("game not started yet — run `./act init`")
    print(board(status.get("frame") or []))


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="act", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init", help="open the game (free); initial board -> logs.txt")
    p_init.add_argument("game", nargs="?", help="ignored: your key already names the game")
    p_init.set_defaults(func=cmd_init)
    p_do = sub.add_parser("do", help="play actions in order; stops when a level clears")
    p_do.add_argument("actions", nargs="+", metavar="ACTION")
    p_do.add_argument("--plan", default="", help="what this batch is testing; recorded in logs.txt")
    p_do.set_defaults(func=cmd_do)
    p_status = sub.add_parser("status", help="free: where you are and what you can do")
    p_status.set_defaults(func=cmd_status)
    p_board = sub.add_parser("board", help="free: print the current board")
    p_board.set_defaults(func=cmd_board)

    args = parser.parse_args(argv)
    if not API_KEY:
        raise ActError("ARC_API_KEY is not set")
    args.func(Path.cwd(), args)


if __name__ == "__main__":
    main()
'''

RECORD_TEMPLATE = '''#!/usr/bin/env python3
"""Play this game. Standard library only.

    ./act init                   open the game (free); the first board is record/steps/000000
    ./act do ACTION1 ACTION6:24,20 --plan "..."
                                 play actions in order; the plan is saved in record/history.jsonl
    ./act status                 free: state, levels, actions used, available actions
    ./act board                  free: print the current board

The server writes the result of every action to record/ (see record/README.md).
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = os.environ.get("ARC_BASE_URL", "http://127.0.0.1:8731").rstrip("/")
API_KEY = os.environ.get("ARC_API_KEY", "")
COORD_ACTION = "ACTION6"
USER_AGENT = "arc-code-act/3"
TERMINAL = ("WIN", "GAME_OVER")


class ActError(SystemExit):
    pass


def _request(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(BASE_URL + path, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/json")
    req.add_header("X-API-Key", API_KEY)
    req.add_header("User-Agent", USER_AGENT)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.loads(response.read() or b"{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            if exc.code == 429 and attempt < 3:
                wait = float(exc.headers.get("Retry-After") or 1)
                time.sleep(min(wait, 10.0))
                continue
            raise ActError("%s %s -> %s: %s" % (method, path, exc.code, detail)) from None
        except urllib.error.URLError as exc:
            raise ActError("%s %s -> cannot reach %s: %s" % (method, path, BASE_URL, exc.reason)) from None
    raise ActError("%s %s -> still rate-limited after retries" % (method, path))


def get(path):
    return _request("GET", path)


def post(path, payload):
    return _request("POST", path, payload)


def grid(layer):
    return "\\n".join("".join("%x" % int(v) for v in row) for row in layer)


def describe(obs):
    return "state=%s level=%s/%s available=%s" % (
        obs.get("state"), obs.get("levels_completed"), obs.get("win_levels"),
        ["ACTION%d" % a for a in (obs.get("available_actions") or [])])


def recorded(obs):
    rec = obs.get("record")
    if not rec:
        return ""
    if "error" in rec:
        return " | RECORD FAILED for step %s: %s" % (rec.get("step"), rec["error"])
    text = " | step %d: %d frame(s), %s cell(s) changed -> %s/" % (
        rec["step"], rec["frames"], "?" if rec.get("changed_cells") is None else rec["changed_cells"],
        rec["dir"])
    if rec["frames"] >= 2:
        if "video_error" in rec:
            text += " (no animation: %s)" % rec["video_error"]
        elif rec.get("video_frames", rec["frames"]) < rec["frames"]:
            text += " (animation.mp4 shows %d of the %d frames, see animation.json)" % (rec["video_frames"], rec["frames"])
        else:
            text += " (animation.mp4)"
    return text


def cmd_init(ws, args):
    status = get("/api/status")
    if status.get("started"):
        if status.get("state") == "GAME_OVER":
            print("game is open and over (state=GAME_OVER, %d actions used); only RESET is accepted: "
                  "`./act do RESET` restarts the current level (one action)" % status.get("actions_used", 0))
            return
        print("game already open: %s (%d actions used so far); nothing changed"
              % (describe(status), status.get("actions_used", 0)))
        return
    obs = post("/api/cmd/RESET", {})
    print("started %s: %s%s" % (obs.get("game_id"), describe(obs), recorded(obs)))


def cmd_do(ws, args):
    before_levels = get("/api/status").get("levels_completed") or 0
    for index, token in enumerate(args.actions):
        name, _, coords = token.partition(":")
        payload = {}
        if name == COORD_ACTION:
            if not coords:
                raise ActError("%s needs coordinates, e.g. %s:24,20" % (name, name))
            x, _, y = coords.partition(",")
            payload = {"x": int(x), "y": int(y)}
        if index == 0 and args.plan:
            payload["plan"] = args.plan
        obs = post("/api/cmd/%s" % name, payload)
        print("%s -> %s%s" % (token, describe(obs), recorded(obs)))
        cleared = (obs.get("levels_completed") or 0) > before_levels
        if obs.get("state") in TERMINAL or cleared:
            dropped = len(args.actions) - index - 1
            if dropped:
                print("%s -- dropped the remaining %d action(s), not sent"
                      % ("level cleared" if cleared else "state=%s" % obs.get("state"), dropped))
            break


def cmd_status(ws, args):
    status = get("/api/status")
    status.pop("frame", None)
    print(json.dumps(status, indent=2))


def cmd_board(ws, args):
    status = get("/api/status")
    if not status.get("started"):
        raise ActError("game not started yet -- run `./act init`")
    frame = status.get("frame") or []
    print(grid(frame[-1]) if frame else "")


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(prog="act", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_init = sub.add_parser("init", help="open the game (free); initial board -> record/steps/000000")
    p_init.add_argument("game", nargs="?", help="ignored: your key already names the game")
    p_init.set_defaults(func=cmd_init)
    p_do = sub.add_parser("do", help="play actions in order; stops when a level clears")
    p_do.add_argument("actions", nargs="+", metavar="ACTION")
    p_do.add_argument("--plan", default="", help="what this batch is testing; recorded in record/history.jsonl")
    p_do.set_defaults(func=cmd_do)
    p_status = sub.add_parser("status", help="free: where you are and what you can do")
    p_status.set_defaults(func=cmd_status)
    p_board = sub.add_parser("board", help="free: print the current board")
    p_board.set_defaults(func=cmd_board)

    args = parser.parse_args(argv)
    if not API_KEY:
        raise ActError("ARC_API_KEY is not set")
    args.func(Path.cwd(), args)


if __name__ == "__main__":
    main()
'''

ALLOWED_IMPORTS = {
    "argparse", "json", "os", "sys", "time",
    "urllib.error", "urllib.request", "pathlib",
    "ast", "zipfile",                      # act_receipt: read frames.npz (without numpy)
}
#: Any of these showing up is a bug -- they are exactly why a venv gets mounted.
FORBIDDEN = ("requests", "pydantic", "arcengine", "arc_agi")


def observation_source() -> str:
    """Exact source copied into an opt-in observation-tools workspace."""
    return Path(__file__).with_name("observation.py").read_text(encoding="utf-8")


def receipt_source() -> str:
    """``receipt.py`` minus its module docstring, put into act.py verbatim."""
    text = Path(__file__).with_name("receipt.py").read_text(encoding="utf-8")
    body = text.split('"""', 2)[2]
    return "\n# ---- act_receipt (src/arcforge/arc_code/receipt.py) ----" + body


def with_receipt(text: str, scene: bool = False) -> str:
    """When ``./act do`` finishes, print this batch's object-level changes (config ``act_receipt``). The actions themselves are bit-identical."""
    marker = "def cmd_do(ws, args):\n"
    line = '        print("%s -> %s%s" % (token, describe(obs), recorded(obs)))\n'
    if text.count(marker) != 1 or text.count(line) != 1:
        raise ValueError("record actuator cmd_do changed; update the receipt hook")
    text = text.replace(line, line + "        _receipt_note(obs)\n", 1)
    # Model-facing (written into act.py), kept verbatim as deployed; the Chinese comment in _receipt_note says:
    # "act_receipt always comes with record and the middleware returns record on every step; a broken step already printed RECORD FAILED above".
    hook = receipt_source() + '''

RECEIPT_STEPS = []
RECEIPT_SCENE = %s


def _receipt_note(obs):
    # act_receipt 一定配 record，middleware 每一步都回 record；寫壞的步上一行已經印了 RECORD FAILED
    rec = obs["record"]
    if "error" not in rec:
        RECEIPT_STEPS.append((rec["step"], rec["frames"], rec["changed_cells"], obs["levels_completed"]))


def cmd_do(ws, args):
    del RECEIPT_STEPS[:]
    before = get("/api/status").get("levels_completed") or 0
    _cmd_do_actions(ws, args)
    if not RECEIPT_STEPS:
        return
    cleared = RECEIPT_STEPS[-1][3] > before
    steps = [(s, f, c) for s, f, c, _ in RECEIPT_STEPS]
    try:
        lines = receipt(steps[0][0], steps[-1][0], cleared, steps, RECEIPT_SCENE)
    except (OSError, ValueError, KeyError, IndexError) as exc:
        lines = ["(change summary unavailable: %s: %s)" % (type(exc).__name__, exc)]
    print("\\n".join(lines))


'''
    hook = hook.replace("RECEIPT_SCENE = %s", "RECEIPT_SCENE = %r" % scene)
    return text.replace(marker, hook + "def _cmd_do_actions(ws, args):\n", 1)


# Model-facing (written into act.py), kept verbatim as deployed; its Chinese comment says: "act_image: the model proxy swaps the
# trailing markers for this step's board image and this batch's change image (work/arc_code/act_images.py)".
IMAGE_HOOK = '''_cmd_do_batch = cmd_do


def cmd_do(ws, args):
    # act_image：model proxy 把結尾的標記換成這一步的盤面圖與這一批的變化圖（work/arc_code/act_images.py）
    before = get("/api/status")
    try:
        _cmd_do_batch(ws, args)
    finally:
        _print_images(before)


def _print_images(before):
    try:
        after = get("/api/status")
    except ActError as exc:
        print("(board images unavailable: %s)" % exc)
        return
    if not after.get("started"):
        return
    first, last = before.get("image_ref"), after.get("image_ref")
    if not last:
        print("(board images unavailable: the server sent no image reference)")
        return
    if first == last:
        return
    step = last.split("/")[1]
    print("Board now (step %s, the same picture as record/current.png):" % step)
    print("[[arc-image:board:%s]]" % last)
    cleared = (after.get("levels_completed") or 0) > (before.get("levels_completed") or 0)
    if first and not cleared:
        print("Changes in this batch (step %s -> %s): changed cells in their new colour, "
              "unchanged cells dark navy (not a game colour):" % (first.split("/")[1], step))
        print("[[arc-image:diff:%s:%s]]" % (first, last))


'''


def with_images(text: str) -> str:
    """At the end of ``./act do``, print the markers for the board after this batch and its change image (config ``act_image``). The actions themselves are bit-identical."""
    marker = "def main(argv=None):\n"
    if text.count(marker) != 1 or text.count("def cmd_do(ws, args):\n") != 1:
        raise ValueError("record actuator changed; update the image hook")
    return text.replace(marker, IMAGE_HOOK + marker, 1)


LIBRARY_DOC = ("    ./act do ACTION1 ACTION6:24,20 --plan \"...\"\n"
               "                                 play actions in order; the plan is saved in record/history.jsonl\n")
LIBRARY_DOC_NEW = ("    actions are sent from Python: from arcgame import act -- see\n"
                   "                                 python3 -c 'import arcgame; help(arcgame)'\n")
# Model-facing (written into act.py), kept verbatim as deployed; its Chinese comment says: "act_library (duckx c0L): actions go
# only through Python's arcgame (src/arcforge/arc_code/arcgame.py)".
LIBRARY_HOOK = '''def cmd_do(ws, args):
    # act_library（duckx c0L）：出手只走 Python 的 arcgame（src/arcforge/arc_code/arcgame.py）
    raise ActError("./act do is not available in this game: actions are sent from Python with the arcgame "
                   "library, e.g.\\n  python3 -c 'from arcgame import act; print(act(\\"ACTION1\\"))'\\n"
                   "Its documentation: python3 -c 'import arcgame; help(arcgame)'")


'''


def with_library(text: str) -> str:
    """Replace ``./act do`` with an error pointing to arcgame (config ``act_library``); init/status/board unchanged."""
    marker = "def main(argv=None):\n"
    help_old = 'p_do = sub.add_parser("do", help="play actions in order; stops when a level clears")'
    help_new = 'p_do = sub.add_parser("do", help="not available here: actions are sent from Python (arcgame)")'
    if text.count(marker) != 1 or text.count(LIBRARY_DOC) != 1 or text.count(help_old) != 1:
        raise ValueError("record actuator changed; update the library hook")
    text = text.replace(LIBRARY_DOC, LIBRARY_DOC_NEW, 1).replace(help_old, help_new, 1)
    return text.replace(marker, LIBRARY_HOOK + marker, 1)


# Model-facing (written into act.py), kept verbatim as deployed; its Chinese comment says: "act_recap: recap.py (embedded above)
# reads record/; with act_image on, status carries image_ref, so the current board's image is attached too".
RECAP_HOOK = '''

def cmd_recap(ws, args):
    # act_recap：recap.py（嵌在上面）讀 record/；開了 act_image 時 status 帶 image_ref，順便附目前盤面的圖
    status = get("/api/status")
    if not status.get("started"):
        raise ActError("game not started yet -- run `./act init`")
    print(recap_text(ws / "record", status))
    ref = status.get("image_ref")
    if ref:
        print("Board now (step %s, the same picture as record/current.png):" % ref.split("/")[1])
        print("[[arc-image:board:%s]]" % ref)


'''


def with_recap(text: str) -> str:
    """``./act recap`` (config ``act_recap``): for free, print this level's batches so far, the levels cleared and the
    objects on the current board (recap.py, the same copy as arcgame.recap()). All other commands bit-identical."""
    doc = "    ./act board                  free: print the current board\n"
    parse = "    args = parser.parse_args(argv)\n"
    marker = "def main(argv=None):\n"
    if any(text.count(s) != 1 for s in (doc, parse, marker)):
        raise ValueError("record actuator changed; update the recap hook")
    body = recap_source().split('"""', 2)[2]
    text = text.replace(doc, doc + "    ./act recap                  free: where you are -- this level's batches, the levels "
                        "cleared, the objects on the board\n", 1)
    text = text.replace(parse, '    p_recap = sub.add_parser("recap", help="free: where you are, from the record")\n'
                        "    p_recap.set_defaults(func=cmd_recap)\n" + parse, 1)
    return text.replace(marker, "# ---- act_recap (src/arcforge/arc_code/recap.py) ----" + body + RECAP_HOOK + marker, 1)


def render(record: bool = False, observation_tools: bool = False, receipt: bool = False, scene: bool = False,
           images: bool = False, library: bool = False, recap: bool = False) -> str:
    if images and not record:
        raise ValueError("act_image requires the middleware record")
    if recap:
        if not record or library:
            raise ValueError("act_recap goes with the middleware record and ./act (not act_library: arcgame has recap())")
        base = with_recap(_render(record, observation_tools, receipt, scene))
        return with_images(base) if images else base
    if library:
        # Images are printed by arcgame when each acting process exits; act.py has no do, so IMAGE_HOOK is not attached
        if not (record and images) or observation_tools or receipt or scene:
            raise ValueError("act_library goes with record + act_image only")
        return with_library(_render(record, False, False, False))
    if not images:
        return _render(record, observation_tools, receipt, scene)
    return with_images(_render(record, observation_tools, receipt, scene))


def _render(record: bool, observation_tools: bool, receipt: bool, scene: bool) -> str:
    if receipt and not record:
        raise ValueError("act_receipt requires the middleware record")
    if scene and not receipt:
        raise ValueError("act_receipt_scene requires act_receipt")
    if not observation_tools:
        base = RECORD_TEMPLATE if record else TEMPLATE
        return with_receipt(base, scene) if receipt else base
    if not record:
        raise ValueError("observation_tools requires the middleware record")
    text = RECORD_TEMPLATE.replace(
        "    ./act init                   open the game",
        "    ./act inspect current --help read-only queries over recorded grids\n"
        "    ./act init                   open the game",
        1,
    )
    before = "def main(argv=None):\n    import argparse\n"
    after = """def main(argv=None):
    import argparse

    query_argv = list(sys.argv[1:] if argv is None else argv)
    if query_argv[:1] == ["inspect"]:
        reader = Path.cwd() / "observation.py"
        os.execv(sys.executable, [sys.executable, str(reader), *query_argv[1:]])
"""
    if text.count(before) != 1:
        raise ValueError("record actuator main changed; update the observation dispatcher")
    text = text.replace(before, after, 1)
    return with_receipt(text, scene) if receipt else text


def write(path: Path, record: bool = False, observation_tools: bool = False, receipt: bool = False,
          scene: bool = False, images: bool = False, library: bool = False, recap: bool = False) -> Path:
    path.write_text(render(record, observation_tools, receipt, scene, images, library, recap), encoding="utf-8")
    path.chmod(0o755)
    if observation_tools:
        reader = path.with_name("observation.py")
        reader.write_text(observation_source(), encoding="utf-8")
        reader.chmod(0o644)
    return path


#: Frozen versions of arcgame (config ``arcgame``). Same rule as the prompt sets: a version that must be reproducible is frozen into a file,
#: not edited in place; when not given, the current ``arcgame.py``.
#:   c0L-r1: the copy duckx c0L-r1 actually mounted into the jail (started 2026-10-11 20:37 UTC, commit 43d40d4a; 800k 69.99),
#:            md5 b3f48c741bd8a175b162de35dc8b3516, untouched byte for byte. Only after it did arcgame.py gain recap() / show(),
#:            reading record/ from $HOME, and Step used as a string.
ARCGAME_FROZEN = {"c0L-r1": "arcgame_c0l_r1.py"}


def library_source(driver_comm: str, version: str | None = None) -> str:
    """``arcgame.py`` (or the frozen version named by ``version``) verbatim, with only the driver's process name replaced
    (markers are written to the stdout of the shell it starts)."""
    if version is not None and version not in ARCGAME_FROZEN:
        raise ValueError(f"no frozen arcgame {version!r}; known: {sorted(ARCGAME_FROZEN)}")
    name = "arcgame.py" if version is None else ARCGAME_FROZEN[version]
    text = Path(__file__).with_name(name).read_text(encoding="utf-8")
    line = 'DRIVER_COMM = "codex"\n'
    if text.count(line) != 1 or not driver_comm.isidentifier():
        raise ValueError(f"cannot set DRIVER_COMM={driver_comm!r} in arcgame.py")
    return text.replace(line, f'DRIVER_COMM = "{driver_comm}"\n', 1)


def recap_source() -> str:
    """``recap.py`` verbatim (stdlib only): arcgame's ``_arcrecap``; the same copy is embedded into act.py."""
    return Path(__file__).with_name("recap.py").read_text(encoding="utf-8")


def write_library(directory: Path, driver_comm: str, version: str | None = None) -> Path:
    """Inside the jail PYTHONPATH points to ``directory`` (bwrap read-only mount; under posix root-owned 0755):
    ``from arcgame import act``; ``_arcrecap.py`` is the recap.py that arcgame.recap() uses (the frozen c0L-r1 does not use it)."""
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o755)
    path = directory / "arcgame.py"
    path.write_text(library_source(driver_comm, version), encoding="utf-8")
    path.chmod(0o644)
    helper = directory / "_arcrecap.py"
    helper.write_text(recap_source(), encoding="utf-8")
    helper.chmod(0o644)
    return path
