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
    "signal", "pwd",                       # note_gate: stop this game's other processes, recognise the posix game user
    "subprocess",                          # toolkit act-side view: run the agent's lib/view.py in a subprocess
    "select",                              # toolkit act-side map: wait for the view tracker subprocess's answer (time-limited)
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


#: ``act_expect`` (RSI batch 1, item A, prompts v14ra): prediction check for ``./act do``. docs/plans/rsi_harness_batch1.md
EXPECT_DOC = ("    ./act do ACTION1 ACTION6:24,20 --plan \"...\"\n"
              "                                 play actions in order; the plan is saved in record/history.jsonl\n")
EXPECT_DOC_NEW = EXPECT_DOC + (
    "                                 --expect-change: stop at the first action that changes nothing\n"
    "                                 --expect-boards F: stop at the first board unlike the prediction in F\n"
    "                                 (a JSON list, one predicted board per action: 64 strings of hex\n"
    "                                 digits with ? for cells not predicted, or 64 lists of ints with -1;\n"
    "                                 null skips that action); the last line says what ran and why it stopped\n")
EXPECT_ARGS_OLD = '    p_do.add_argument("--plan", default="", help="what this batch is testing; recorded in record/history.jsonl")\n'
EXPECT_ARGS_NEW = EXPECT_ARGS_OLD + (
    '    p_do.add_argument("--expect-change", action="store_true",\n'
    '                      help="stop at the first action that changes no cell")\n'
    '    p_do.add_argument("--expect-boards", metavar="FILE",\n'
    '                      help="JSON list of predicted boards, one per action; stop at the first that differs")\n')
EXPECT_DO = r"""def _expected_boards(path, n):
    try:
        boards = json.loads(Path(path).read_text())
    except (OSError, ValueError) as exc:
        raise ActError("--expect-boards %s: %s" % (path, exc))
    if not isinstance(boards, list) or len(boards) != n:
        raise ActError("--expect-boards %s: needs a JSON list with one entry per action (%d), got %s"
                       % (path, n, "%d entries" % len(boards) if isinstance(boards, list) else type(boards).__name__))
    out = []
    for i, board in enumerate(boards):
        if board is None:
            out.append(None)
            continue
        if not isinstance(board, list) or not board:
            raise ActError("--expect-boards entry %d: a board is a list of rows (or null)" % i)
        rows = []
        for row in board:
            if isinstance(row, str):
                if any(c not in "0123456789abcdefABCDEF?" for c in row):
                    raise ActError("--expect-boards entry %d: string rows hold hex digits and ? only" % i)
                rows.append([-1 if c == "?" else int(c, 16) for c in row])
            elif isinstance(row, list) and all(isinstance(v, int) and not isinstance(v, bool) for v in row):
                rows.append(row)
            else:
                raise ActError("--expect-boards entry %d: each row is a hex string or a list of ints" % i)
        out.append(rows)
    return out


def _board_misses(expected, actual):
    if len(expected) != len(actual) or any(len(e) != len(a) for e, a in zip(expected, actual)):
        return None
    return [(x, y, e, a) for y, (erow, arow) in enumerate(zip(expected, actual))
            for x, (e, a) in enumerate(zip(erow, arow)) if e != -1 and e != a]


def _misses_text(misses):
    xs = [m[0] for m in misses]
    ys = [m[1] for m in misses]
    first = ", ".join("(%d,%d) expected %x got %x" % m for m in misses[:6])
    return "%d cell(s) in x %d-%d, y %d-%d; %s%s" % (
        len(misses), min(xs), max(xs), min(ys), max(ys), first, ", ..." if len(misses) > 6 else "")


def cmd_do(ws, args):
    expected = _expected_boards(args.expect_boards, len(args.actions)) if args.expect_boards else None
    before_levels = get("/api/status").get("levels_completed") or 0
    executed, no_change, stopped = 0, [], ""
    try:
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
            executed += 1
            print("%s -> %s%s" % (token, describe(obs), recorded(obs)))
            rec = obs.get("record") or {}
            if rec.get("changed_cells") == 0:
                no_change.append("%s(%s)" % (rec["step"], token))
            cleared = (obs.get("levels_completed") or 0) > before_levels
            if obs.get("state") in TERMINAL or cleared:
                stopped = "level cleared" if cleared else "state=%s" % obs.get("state")
                break
            if args.expect_change and rec.get("changed_cells") == 0:
                stopped = "--expect-change: step %s (%s) changed nothing" % (rec["step"], token)
                break
            if expected is not None and expected[index] is not None:
                frames = obs.get("frame")
                if not frames:
                    raise ActError("the server sent no board for %s; --expect-boards cannot check it" % token)
                misses = _board_misses(expected[index], frames[-1])
                if misses is None:
                    stopped = "--expect-boards: entry %d is not %dx%d like the board" % (
                        index, len(frames[-1][0]), len(frames[-1]))
                    break
                if misses:
                    stopped = "--expect-boards: step %s (%s) differs from the prediction in %s" % (
                        rec.get("step", "?"), token, _misses_text(misses))
                    break
    except ActError as exc:
        stopped = "error: %s" % exc
        print(exc)
    left = len(args.actions) - executed
    print("executed %d/%d; no-change: %s; stopped: %s%s" % (
        executed, len(args.actions), " ".join(no_change) or "none", stopped or "no",
        " -- %d action(s) not sent" % left if left else ""))
    if stopped.startswith("error: "):
        sys.stdout.flush()
        raise SystemExit(1)
"""


def with_expect(text: str) -> str:
    """``./act do --expect-change / --expect-boards``, with the summary line always last (config ``act_expect``).
    Without the flags the actions sent are bit-identical; only that last line is added."""
    start = text.find("def cmd_do(ws, args):\n")
    end = text.find("\n\ndef cmd_status(ws, args):\n")
    if (start < 0 or end < 0 or text.count("def cmd_do(ws, args):\n") != 1 or text.count(EXPECT_DOC) != 1
            or text.count(EXPECT_ARGS_OLD) != 1):
        raise ValueError("record actuator cmd_do changed; update the expect hook")
    text = text[:start] + EXPECT_DO.rstrip("\n") + text[end:]
    return text.replace(EXPECT_DOC, EXPECT_DOC_NEW, 1).replace(EXPECT_ARGS_OLD, EXPECT_ARGS_NEW, 1)


#: ``act_digest`` (RSI round 3, prompts v14re; needs act_expect): after every action ``./act do`` prints a ``what changed:`` line --
#: which objects moved (by how much), appeared, disappeared, changed shape, or that only cells on the board's edge changed (mostly a
#: counter bar); ``--expect-change`` also treats "only the edge changed" as unchanged. On levels where it was stuck, the model used
#: batch tests containing refused / no-effect actions to rule out rules (RSI round 3, 15 sets read in full).
DIGEST_FUNCS = r"""EDGE = 2          # boards of 32+ cells a side: the outer EDGE rows and columns hold counters and bars
DIGEST_ITEMS = 4


def _objects(board, bg, skip):
    h, w = len(board), len(board[0])
    seen = [[False] * w for _ in range(h)]
    out = set()
    for y in range(h):
        for x in range(w):
            v = board[y][x]
            if seen[y][x] or v == bg or (x, y) in skip:
                continue
            seen[y][x] = True
            stack, cells = [(x, y)], []
            while stack:
                cx, cy = stack.pop()
                cells.append((cx, cy))
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if (0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and board[ny][nx] == v
                            and (nx, ny) not in skip):
                        seen[ny][nx] = True
                        stack.append((nx, ny))
            out.add((v, frozenset(cells)))
    return out


def _where(cells):
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    x = "x%d" % min(xs) if min(xs) == max(xs) else "x%d-%d" % (min(xs), max(xs))
    y = "y%d" % min(ys) if min(ys) == max(ys) else "y%d-%d" % (min(ys), max(ys))
    return x + " " + y


def _shape(cells):
    x0 = min(c[0] for c in cells)
    y0 = min(c[1] for c in cells)
    return frozenset((cx - x0, cy - y0) for cx, cy in cells), x0, y0


def _part(kind, v, cells, changed, verb):
    # an object most of whose cells did not change was split off or joined, not gone or new
    k = len(cells & changed)
    if 2 * k < len(cells):
        return (3, -len(cells), "%x (%d cells) at %s: %d of its cells changed" % (v, len(cells), _where(cells), k))
    return (kind, -len(cells), "%x (%d cells) %s at %s" % (v, len(cells), verb, _where(cells)) if verb == "appeared"
            else "%x (%d cells) at %s %s" % (v, len(cells), _where(cells), verb))


def _what_changed(a, b):
    # (one line on what the action changed between boards a and b, whether only edge cells changed)
    if a is None or b is None:
        return "no board to compare", False
    if len(a) != len(b) or any(len(ra) != len(rb) for ra, rb in zip(a, b)):
        return "the board changed size (%dx%d -> %dx%d)" % (len(a[0]), len(a), len(b[0]), len(b)), False
    h, w = len(a), len(a[0])
    changed = [(x, y) for y in range(h) for x in range(w) if a[y][x] != b[y][x]]
    if not changed:
        return "nothing on the board", False
    edge = set()
    if h >= 32 and w >= 32:
        edge = {(x, y) for y in range(h) for x in range(w)
                if y < EDGE or y >= h - EDGE or x < EDGE or x >= w - EDGE}
    inner = [c for c in changed if c not in edge]
    n_edge = len(changed) - len(inner)
    if not inner:
        return "only %d cell(s) on the board's edge (a counter or bar?), nothing else" % n_edge, True
    counts = {}
    for row in a:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    bg = max(counts, key=counts.get)
    changed_set = set(inner)
    gone = _objects(a, bg, edge) - _objects(b, bg, edge)
    new = _objects(b, bg, edge) - _objects(a, bg, edge)
    items = []          # (kind: 0 moved, 1 appeared, 2 gone, 3 reshaped; -size; text)
    for v, cells in sorted(gone, key=lambda o: -len(o[1])):
        shape, x0, y0 = _shape(cells)
        match = next((o for o in new if o[0] == v and _shape(o[1])[0] == shape), None)
        if match is None:
            match = next((o for o in new if o[0] == v and o[1] & cells), None)
            if match is not None:
                items.append((3, -len(cells), "%x (%d cells) at %s now %d cells at %s" % (
                    v, len(cells), _where(cells), len(match[1]), _where(match[1]))))
                new.discard(match)
                continue
            match = next((o for o in new if o[0] == v and len(o[1]) == len(cells)), None)
            if match is None:
                items.append(_part(2, v, cells, changed_set, "gone"))
                continue
        _, x1, y1 = _shape(match[1])
        items.append((0, -len(cells), "%x (%d cells) at %s moved %+d,%+d%s" % (
            v, len(cells), _where(cells), x1 - x0, y1 - y0, "" if _shape(match[1])[0] == shape else " (new shape)")))
        new.discard(match)
    for v, cells in new:
        items.append(_part(1, v, cells, changed_set, "appeared"))
    items.sort()
    text = "%d cell(s): " % len(changed) + "; ".join(t for _, _, t in items[:DIGEST_ITEMS])
    if len(items) > DIGEST_ITEMS:
        text += "; +%d more" % (len(items) - DIGEST_ITEMS)
    if not items:
        text += "cells recoloured to the background colour %x or within one object" % bg
    if n_edge:
        text += " (+%d edge cell(s))" % n_edge
    return text, False


"""
DIGEST_EDITS = (
    ('    before_levels = get("/api/status").get("levels_completed") or 0\n'
     '    executed, no_change, stopped = 0, [], ""\n',
     '    status = get("/api/status")\n'
     '    before_levels = status.get("levels_completed") or 0\n'
     '    prev = (status.get("frame") or [None])[-1]\n'
     '    executed, no_change, stopped = 0, [], ""\n'),
    ('            rec = obs.get("record") or {}\n'
     '            if rec.get("changed_cells") == 0:\n'
     '                no_change.append("%s(%s)" % (rec["step"], token))\n',
     '            rec = obs.get("record") or {}\n'
     '            cur = (obs.get("frame") or [None])[-1]\n'
     '            what, edge_only = _what_changed(prev, cur)\n'
     '            prev = cur\n'
     '            print("    what changed: " + what)\n'
     '            if rec.get("changed_cells") == 0 or edge_only:\n'
     '                no_change.append("%s(%s%s)" % (rec["step"], token, ",edge-only" if edge_only else ""))\n'),
    ('            if args.expect_change and rec.get("changed_cells") == 0:\n'
     '                stopped = "--expect-change: step %s (%s) changed nothing" % (rec["step"], token)\n',
     '            if args.expect_change and (rec.get("changed_cells") == 0 or edge_only):\n'
     '                stopped = "--expect-change: step %s (%s) changed nothing%s" % (\n'
     '                    rec["step"], token, " but cells on the board\'s edge" if edge_only else "")\n'),
    ("                                 --expect-change: stop at the first action that changes nothing\n",
     "                                 --expect-change: stop at the first action that changes nothing\n"
     "                                 (or nothing but cells on the board's edge); every action also\n"
     "                                 prints a `what changed:` line: objects moved, gone, appeared, reshaped\n"),
)


def with_digest(text: str) -> str:
    """``act_digest``: stacks only on top of ``with_expect``; every anchor must occur exactly once."""
    marker = "def cmd_do(ws, args):\n"
    if text.count(marker) != 1 or "--expect-boards" not in text:
        raise ValueError("act_digest goes on the act_expect actuator")
    for old, new in DIGEST_EDITS:
        if text.count(old) != 1:
            raise ValueError("act_expect cmd_do changed; update the digest hook: %r" % old[:60])
        text = text.replace(old, new, 1)
    return text.replace(marker, DIGEST_FUNCS + marker, 1)


#: ``notes_budget`` (RSI batch 1, item B, prompts v14rb): when notes.md exceeds the cap, warn after every ./act command and remind to tidy it up on level clear.
NOTES_MAIN_OLD = "    args.func(Path.cwd(), args)\n"
NOTES_MAIN_NEW = ("    before = (get(\"/api/status\").get(\"levels_completed\") or 0) if args.command == \"do\" else None\n"
                  "    args.func(Path.cwd(), args)\n"
                  "    _notes_check(Path.cwd(), before)\n")
NOTES_HOOK = r"""NOTES_LIMIT = %d


def _notes_check(ws, before):
    if before is not None:
        after = get("/api/status").get("levels_completed") or 0
        if after > before:
            print("level %%d cleared: before playing on, write levels/L%%02d.md, move this level's notes there, "
                  "and leave notes.md with the cross-level points and a fresh section for the new level"
                  %% (after, after))
    notes = ws / "notes.md"
    if notes.is_file():
        size = len(notes.read_text(errors="replace"))
        if size > NOTES_LIMIT:
            print("notes.md is %%d characters, over %%d: move details into levels/ and keep notes.md "
                  "to what you need to read after a compaction" %% (size, NOTES_LIMIT))


"""


#: Enabled together with act_expect (Kaggle's v14r): ``./act do`` runs the notes check itself before the summary line, so the summary
#: line is still last; other commands still check when main wraps up.
NOTES_MAIN_WITH_EXPECT = ("    args.func(Path.cwd(), args)\n"
                          "    if args.command != \"do\":\n"
                          "        _notes_check(Path.cwd(), None)\n")
EXPECT_SUMMARY = "    left = len(args.actions) - executed\n"


def with_notes_budget(text: str, limit: int) -> str:
    """notes.md length warning and level-clear reminder (config ``notes_budget``). The actions themselves are bit-identical.
    When ``text`` already carries act_expect, the ``./act do`` check goes before the summary line."""
    marker = "def main(argv=None):\n"
    if text.count(marker) != 1 or text.count(NOTES_MAIN_OLD) != 1:
        raise ValueError("record actuator main changed; update the notes hook")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
        raise ValueError(f"notes_budget must be a positive int, not {limit!r}")
    if "--expect-boards" in text:
        if text.count(EXPECT_SUMMARY) != 1:
            raise ValueError("expect cmd_do changed; update where the notes check goes")
        text = text.replace(EXPECT_SUMMARY, "    _notes_check(ws, before_levels)\n" + EXPECT_SUMMARY, 1)
        text = text.replace(NOTES_MAIN_OLD, NOTES_MAIN_WITH_EXPECT, 1)
    else:
        text = text.replace(NOTES_MAIN_OLD, NOTES_MAIN_NEW, 1)
    return text.replace(marker, NOTES_HOOK % limit + marker, 1)


#: ``note_gate`` (prompt fragment prompts/_features/note_gate): on level clear the next level is withheld at first -- act prints the frames of
#: how this level ended and stops this game's other processes; after that ``./act do`` / ``./act board`` refuse until the agent has updated
#: levels/Lnn.md and notes.md and run ``./act note_done``.
#: Only in act: record / middleware are unchanged (the next level's board is still written to record/, act just does not volunteer it); the gate
#: state is derived from record/history.jsonl (written by the server, the agent cannot change it) and the workspace's .note_gate/. It can be
#: bypassed (calling the API directly, forging markers), which shows afterwards in the history and the marker times.
#:
#: Processes are stopped only inside a fence: under posix it is a user dedicated to this game (the kernel only lets it signal the same user);
#: under bwrap it is its own pid namespace with a freshly mounted /proc (PID 1 is bwrap). If no fence is recognised, raise and send not a
#: single signal -- reaching this code on the host as the host user would kill things across the whole machine. Kept alive: codex (the inode of
#: ARC_DRIVER_EXE) and processes it started from the same executable, bwrap's bridge, PID 1, and this command itself
#: (the subtree under codex that act runs in: if the tail of ``./act do ... | tail`` is killed, the level-clear message is lost).
NOTE_GATE_DOC = "    ./act board                  free: print the current board\n"
NOTE_GATE_DOC_NEW = NOTE_GATE_DOC + (
    "    ./act note_done              after a level clears: checks that levels/LNN.md and notes.md were\n"
    "                                 updated since, then starts the next level\n")
NOTE_GATE_PARSE = "    p_board.set_defaults(func=cmd_board)\n"
NOTE_GATE_PARSE_NEW = NOTE_GATE_PARSE + (
    '    p_note = sub.add_parser("note_done", help="after a level clears: check the notes, then start the next level")\n'
    "    p_note.set_defaults(func=cmd_note_done)\n")
#: This game's workspace (both fences set HOME to it). note_gate brings it in; the first-generation toolkit act (hs110 v38) has no note_gate
#: and brings it separately.
WORKSPACE_FUNC = r'''def _workspace():
    # both fences set HOME to this game's workspace; record/ in it is written by the server
    home = os.environ.get("HOME", "")
    if not home or not (Path(home) / "record").is_dir():
        raise ActError("HOME (%r) is not this game's workspace: it has no record/" % home)
    return Path(home)
'''
NOTE_GATE_HOOK = r'''# ---- note_gate: when a level clears, the next level waits for ./act note_done ----
GATE_DIR = ".note_gate"
PROC = Path("/proc")
STOP_LIMIT = 64          # more candidates than this means something is wrong: stop none of them
STOP_GRACE = 2.0         # seconds between SIGTERM and SIGKILL


''' + WORKSPACE_FUNC + r'''

def _last_clear(ws):
    # (level, step, time, frames) of the latest clear that did not win the game, from the server's record
    path = ws / "record" / "history.jsonl"
    if not path.is_file():
        return None
    last, done = None, 0
    with open(path) as handle:
        for line in handle:
            if not line.endswith("\n"):
                break                         # the server is still writing this line
            row = json.loads(line)
            now = row.get("levels_completed") or 0
            if now > done and row.get("state") != "WIN":
                last = (now, row["step"], row["time"], row.get("frames") or 1)
            done = max(done, now)
    return last


def _mark(ws, level):
    return ws / GATE_DIR / ("L%02d.done" % level)


def _waiting(ws):
    clear = _last_clear(ws)
    if clear is None:
        return None
    mark = _mark(ws, clear[0])
    if mark.is_file() and json.loads(mark.read_text()).get("step") == clear[1]:
        return None
    return clear


def _waiting_text(clear):
    return ("level %d is cleared and the next level waits for `./act note_done`: update levels/L%02d.md and "
            "notes.md, then run it. No action was sent." % (clear[0], clear[0]))


def _cleared_text(clear):
    level, step, _, frames = clear
    lines = ["Level %d cleared." % level]
    if frames >= 2:
        last = frames - 2                     # the final frame is already the next level's first board
        lines.append("How it ended: record/steps/%06d/frame_000.png%s -- look at it." % (
            step, "" if last == 0 else " to frame_%03d.png" % last))
    lines.append("record/current.png and this step's final.png already show the next level; it can wait. First "
                 "update levels/L%02d.md and notes.md with what this level taught you, then run `./act note_done`: "
                 "it starts the next level. Until then `./act do` and `./act board` are refused and no action is "
                 "spent." % level)
    return "\n".join(lines)


def _ppid(proc, pid):
    raw = (proc / str(pid) / "stat").read_text()
    return int(raw[raw.rindex(")") + 2:].split()[1])


def _uid(proc, pid):
    for line in (proc / str(pid) / "status").read_text().splitlines():
        if line.startswith("Uid:"):
            return int(line.split()[1])
    raise ActError("%s/%d/status has no Uid line" % (proc, pid))


def _home(proc, pid):
    for item in (proc / str(pid) / "environ").read_bytes().split(b"\0"):
        if item.startswith(b"HOME="):
            return item[5:].decode(errors="replace")
    return None


def _cmdline(proc, pid):
    return (proc / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()


def _fenced(proc, me, uid):
    # raise unless this process is inside a game fence, where it can reach only this game's processes
    fence = os.environ.get("ARC_FENCE")
    if fence == "bwrap":
        init = (proc / "1" / "comm").read_text().strip()
        if init != "bwrap" or os.readlink(proc / "1" / "ns" / "pid") != os.readlink(proc / str(me) / "ns" / "pid"):
            raise ActError("ARC_FENCE=bwrap but %s is not the jail's own (pid 1 is %r)" % (proc, init))
    elif fence == "posix":
        import pwd
        user = os.environ.get("ARC_GAME_USER", "")
        if uid == 0 or not user.startswith("arc-") or pwd.getpwnam(user).pw_uid != uid:
            raise ActError("ARC_FENCE=posix but this process (uid %d) is not the game user %r" % (uid, user))
    else:
        raise ActError("ARC_FENCE is %r: not inside a game fence" % fence)


def _same_exe(proc, pid, want):
    try:
        st = os.stat(proc / str(pid) / "exe")
    except PermissionError:
        return False                          # another user's process (posix: runuser, root)
    return (st.st_dev, st.st_ino) == (want.st_dev, want.st_ino)


def _processes(proc):
    # {pid: ppid} of every process that still exists
    out = {}
    for entry in proc.iterdir():
        if entry.name.isdigit():
            try:
                out[int(entry.name)] = _ppid(proc, int(entry.name))
            except (FileNotFoundError, ProcessLookupError):
                continue                      # it exited while we looked
    return out


def _protected(proc, me, uid):
    # pid 1, the driver and what it runs from its own executable, the bridges, and this command's whole tree
    exe = os.environ.get("ARC_DRIVER_EXE")
    if not exe:
        raise ActError("ARC_DRIVER_EXE is not set")
    want = os.stat(exe)
    parents = _processes(proc)
    chain, pid = [], me
    while not _same_exe(proc, pid, want):
        chain.append(pid)
        pid = parents.get(pid, 0)
        if pid <= 1 or len(chain) > 64:
            raise ActError("the driver (%s) is not among this command's ancestors" % exe)
    driver, top = pid, chain[-1]
    # the driver is the top of this game's processes: its parent is bwrap's pid 1 or another user's (posix: runuser)
    above = parents.get(driver, 0)
    if above != 1 and _uid(proc, above) == uid:
        raise ActError("the driver candidate %d (%s) runs under another process of this game (%d): not the driver"
                       % (driver, exe, above))
    keep = {1, me, driver}
    children = {}
    for child, parent in parents.items():
        children.setdefault(parent, []).append(child)
    todo = [top]
    while todo:
        pid = todo.pop()
        keep.add(pid)
        todo.extend(children.get(pid, []))
    for child in children.get(driver, []):
        try:
            if _same_exe(proc, child, want) or b"/bridge.py" in (proc / str(child) / "cmdline").read_bytes().split(b"\0"):
                keep.add(child)
        except (FileNotFoundError, ProcessLookupError):
            continue
    return keep


def _candidates(proc, uid, ws, keep):
    out = []
    for pid in sorted(_processes(proc)):
        if pid in keep:
            continue
        try:
            if _uid(proc, pid) != uid or _home(proc, pid) != str(ws):
                continue
            out.append((pid, _cmdline(proc, pid)))
        except (FileNotFoundError, ProcessLookupError):
            continue                          # it exited while we looked
        except PermissionError:
            continue                          # its environment is not readable: not ours to judge
    return out


def _stop(ws, at, proc, me, uid, send, wait):
    import signal
    _fenced(proc, me, uid)
    keep = _protected(proc, me, uid)
    stopped = {}
    for sig, pause in ((signal.SIGTERM, STOP_GRACE), (signal.SIGKILL, 0.5)):
        targets = _candidates(proc, uid, ws, keep)
        if len(targets) > STOP_LIMIT:
            raise ActError("%d processes look like this game's, more than %d: none was stopped (%s)"
                           % (len(targets), STOP_LIMIT, "; ".join(c[:60] for _, c in targets[:5])))
        for pid, cmd in targets:
            try:
                send(pid, sig)
            except ProcessLookupError:
                continue
            stopped.setdefault(pid, cmd)
        if targets:
            wait(pause)
    left = _candidates(proc, uid, ws, keep)
    (ws / GATE_DIR).mkdir(exist_ok=True)
    with open(ws / GATE_DIR / "stops.jsonl", "a") as handle:
        handle.write(json.dumps({"time": time.time(), "at": at, "stopped": [[p, c[:200]] for p, c in stopped.items()],
                                 "left": [[p, c[:200]] for p, c in left]}) + "\n")
    if left:
        raise ActError("could not stop %d process(es): %s" % (len(left), "; ".join(c[:60] for _, c in left)))
    return sorted(stopped.items())


def _stop_others(ws, at):
    sys.stdout.flush()
    stopped = _stop(ws, at, PROC, os.getpid(), os.getuid(), os.kill, time.sleep)
    if stopped:
        names = "; ".join("%s" % c[:60] for _, c in stopped)
        print("Stopped the other %d process(es) of this game: %s" % (len(stopped), names[:400]))


def cmd_do(ws, args):
    ws = _workspace()
    waiting = _waiting(ws)
    if waiting is not None:
        raise ActError("refused: " + _waiting_text(waiting))
    _cmd_do_batch(ws, args)
    cleared = _waiting(ws)
    if cleared is not None:
        print(_cleared_text(cleared))
        _stop_others(ws, "clear L%02d" % cleared[0])


def cmd_board(ws, args):
    waiting = _waiting(_workspace())
    if waiting is not None:
        raise ActError("refused: " + _waiting_text(waiting))
    _cmd_board_open(ws, args)


def cmd_note_done(ws, args):
    ws = _workspace()
    waiting = _waiting(ws)
    if waiting is None:
        print("nothing to do: no cleared level is waiting")
        return
    level, step, when, _ = waiting
    stale = []
    for name in ("levels/L%02d.md" % level, "notes.md"):
        path = ws / name
        if not path.is_file():
            stale.append("%s does not exist" % name)
        elif path.stat().st_mtime <= when:
            stale.append("%s has not changed since level %d cleared" % (name, level))
    if stale:
        raise ActError("not yet: %s. Update them, then run `./act note_done` again." % "; ".join(stale))
    _mark(ws, level).parent.mkdir(exist_ok=True)
    _mark(ws, level).write_text(json.dumps({"step": step, "time": time.time()}))
    print("Level %d starts now. Its first board is record/current.png (also record/steps/%06d/final.png): "
          "look at it before you plan." % (level + 1, step))
    _stop_others(ws, "note_done L%02d" % level)


'''


#: gen-2 (work/arc_code/gen2): the harness opens the game before the session starts (run_arm's open_game), so act has no init to describe;
#: ``init`` stays as a no-op alias (model habit), and a game that is not open is a harness bug that fails loudly. The GAME_OVER hint moved
#: from init into do's output.
GEN2_DOC = "    ./act init                   open the game (free); the first board is record/steps/000000\n"
GEN2_INIT_OLD = """def cmd_init(ws, args):
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
"""
GEN2_INIT_NEW = """def cmd_init(ws, args):
    status = get("/api/status")
    if not status.get("started"):
        raise ActError("the game is not open: the harness opens it before your session starts (harness bug)")
    print("the game is already open (the harness opened it): %s; %d actions used. Nothing changed."
          % (describe(status), status.get("actions_used", 0)))
"""
GEN2_DO_LINE = '        print("%s -> %s%s" % (token, describe(obs), recorded(obs)))\n'
GEN2_DO_NEW = GEN2_DO_LINE + (
    '        if obs.get("state") == "GAME_OVER":\n'
    '            print("game over on this level: only RESET is accepted now; `./act do RESET` restarts the level "\n'
    '                  "(one action)")\n')
GEN2_BOARD_OLD = '        raise ActError("game not started yet -- run `./act init`")\n'
GEN2_BOARD_NEW = '        raise ActError("the game is not open: the harness opens it before your session starts (harness bug)")\n'
GEN2_INIT_HELP_OLD = 'p_init = sub.add_parser("init", help="open the game (free); initial board -> record/steps/000000")'
GEN2_INIT_HELP_NEW = 'p_init = sub.add_parser("init", help="nothing to do: the harness already opened the game")'
#: Window full (gen-2's PostToolUse hook creates ``.context_full`` in the workspace, the next window's SessionStart removes it; the file
#: name matches CONTEXT_FULL in work/arc_code/gen2/hooks/common.py): ``do`` sends no action at all. ``_workspace`` belongs to note_gate,
#: which gen-2 always stacks.
#: 2026-10-21 analysis: in 58% of force-switched windows the model still acted after its last note write, and the results never made it
#: into the notes (docs/plans/arc3_harness_gen2_20261020.md).
GEN2_DO_HEAD_OLD = 'def cmd_do(ws, args):\n    before_levels = get("/api/status").get("levels_completed") or 0\n'
GEN2_DO_HEAD_NEW = '''# ---- gen-2: when the context window is full, no action is sent until the next window ----
CONTEXT_FULL = ".context_full"


def _refuse_when_context_full():
    if (_workspace() / CONTEXT_FULL).exists():
        raise ActError("refused: this context window is full. Bring your notes up to date, then call "
                       "`new_context`; the next window can send this. No action was sent.")


def cmd_do(ws, args):
    _refuse_when_context_full()
    before_levels = get("/api/status").get("levels_completed") or 0
'''


#: gen-2 seed collection with level-change snapshots taken at note_done (middleware ``--level-snapshot-at note_done``): once note_done passes
#: and this game's other processes are stopped, ask the middleware to snapshot the workspace as the next level's ``LKK`` (the next level has
#: no actions yet at that point). Prints nothing: the output the model sees is the same as in arms that do not collect; a failed snapshot
#: still raises from post (the middleware also writes ERRORS.jsonl and collection stops).
NOTE_DONE_TAIL = '    _stop_others(ws, "note_done L%02d" % level)\n'
NOTE_DONE_TAIL_SNAPSHOT = NOTE_DONE_TAIL + '    post("/api/snapshot", {"level_done": level})\n'


def with_note_done_snapshot(text: str) -> str:
    if text.count(NOTE_DONE_TAIL) != 1:
        raise ValueError("note_done changed; update where the level snapshot request goes")
    return text.replace(NOTE_DONE_TAIL, NOTE_DONE_TAIL_SNAPSHOT, 1)


#: pace line (2026-10-27, gen-2 RL single-level episodes, run_arm's ``pace_line``): the last line of ``./act do`` (one action or a batch) and
#: of ``./act status`` is the output tokens used on this level. The numbers come from ``.pace.json``, which the harness writes in the workspace
#: ($HOME) -- run_arc3_level's proxy writes ``{"level_output_tokens": X, "level_output_budget": Y}`` at the start and end of every request
#: (X is the proxy's cumulative output count, excluding the per-action deduction; Y is this episode's output budget).
#: Read and validated up front; a missing or malformed file is a harness bug: raise and send no action at all. Printed on the last line:
#: agents often pipe act's output into ``tail``.
#: Read once, and the same value printed when the command finishes: no request is in flight while a command runs (codex executes tool
#: calls between requests). The file name must match run_arc3_level's PACE_FILE and seed_snapshot.EXCLUDED (a test checks all three).
PACE_FILE = ".pace.json"
PACE_HOOK = r'''# ---- pace: ./act do and ./act status end with this level's output tokens (the harness writes .pace.json) ----
PACE_FILE = ".pace.json"
PACE_KEYS = ("level_output_tokens", "level_output_budget")


def _pace():
    path = _workspace() / PACE_FILE
    try:
        pace = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        raise ActError("cannot read %s (harness bug): %s. Nothing was sent." % (PACE_FILE, exc))
    if (not isinstance(pace, dict) or sorted(pace) != sorted(PACE_KEYS)
            or any(type(pace[key]) is not int for key in PACE_KEYS)
            or pace["level_output_tokens"] < 0 or pace["level_output_budget"] <= 0):
        raise ActError("%s is not {level_output_tokens: int >= 0, level_output_budget: int > 0} (harness bug): %r. "
                       "Nothing was sent." % (PACE_FILE, pace))
    return "This level: {:,} output tokens used of {:,}.".format(pace["level_output_tokens"],
                                                                pace["level_output_budget"])


def cmd_do(ws, args):
    pace = _pace()
    _cmd_do_unpaced(ws, args)
    print(pace)


def cmd_status(ws, args):
    pace = _pace()
    _cmd_status_unpaced(ws, args)
    print(pace)


'''


def with_pace(text: str) -> str:
    """``pace``: see PACE_HOOK above. Stacks on gen-2's act (one more wrapper around note_gate's cmd_do)."""
    marker = "def main(argv=None):\n"
    do, status = "def cmd_do(ws, args):\n", "def cmd_status(ws, args):\n"
    if any(text.count(s) != 1 for s in (marker, do, status)) or PACE_HOOK.count('PACE_FILE = "%s"\n' % PACE_FILE) != 1:
        raise ValueError("gen2 actuator changed; update the pace hook")
    text = text.replace(do, "def _cmd_do_unpaced(ws, args):\n", 1)
    text = text.replace(status, "def _cmd_status_unpaced(ws, args):\n", 1)
    return text.replace(marker, PACE_HOOK + marker, 1)


def with_gen2(text: str) -> str:
    pairs = ((GEN2_DOC, ""), (GEN2_INIT_OLD, GEN2_INIT_NEW), (GEN2_DO_LINE, GEN2_DO_NEW),
             (GEN2_BOARD_OLD, GEN2_BOARD_NEW), (GEN2_INIT_HELP_OLD, GEN2_INIT_HELP_NEW),
             (GEN2_DO_HEAD_OLD, GEN2_DO_HEAD_NEW))
    for old, new in pairs:
        if text.count(old) != 1:
            raise ValueError("record actuator changed; update the gen2 hook: %r" % old[:60])
        text = text.replace(old, new, 1)
    return text


#: Click feedback (config ``gen2_click_info``, 2026-10-26; on the first-generation pure record actuator it is ``act_click_info``, 2026-10-27 hs110 v35):
#: under every ACTION6 line, print one line about the clicked cell -- its colour, and the cell count and extent of the same-colour region (4-connected)
#: it is in; a click on background reports the nearest non-background cell within CLICK_NEAR cells; an unchanged board is reported too. In v32g4 / v34g4,
#: 8.6% / 5.3% of clicks changed nothing (other actions 2.1%), and a click one cell off went unnoticed until the image was looked at. Reads the board
#: from before the click (the previous step's last frame; a 0-frame step carries over the one before).
CLICK_FUNCS = r'''CLICK_NEAR = 3          # how far from a background click to look for the nearest other cell


def _region(board, x, y):
    h, w = len(board), len(board[0])
    v = board[y][x]
    seen = {(x, y)}
    stack = [(x, y)]
    while stack:
        cx, cy = stack.pop()
        for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
            if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in seen and board[ny][nx] == v:
                seen.add((nx, ny))
                stack.append((nx, ny))
    return seen


def _span(cells):
    xs = [c[0] for c in cells]
    ys = [c[1] for c in cells]
    x = "x%d" % min(xs) if min(xs) == max(xs) else "x%d-%d" % (min(xs), max(xs))
    y = "y%d" % min(ys) if min(ys) == max(ys) else "y%d-%d" % (min(ys), max(ys))
    return x + " " + y


def _click_text(board, x, y):
    # one line on the cell an ACTION6 hits, read from the board just before the click
    where = "click %d,%d" % (x, y)
    if board is None:
        return where + ": no board to read it on (the last step returned no frame)"
    h, w = len(board), len(board[0])
    if not (0 <= x < w and 0 <= y < h):
        return where + ": outside the %dx%d board" % (w, h)
    counts = {}
    for row in board:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    bg = max(counts, key=counts.get)
    v = board[y][x]
    cells = _region(board, x, y)
    if v != bg:
        return "%s: color %x, a region of %d cell(s) at %s" % (where, v, len(cells), _span(cells))
    text = "%s: background color %x" % (where, bg)
    if 2 * len(cells) < counts[bg]:
        text += " (an enclosed area of %d cell(s) at %s)" % (len(cells), _span(cells))
    near = [(max(abs(dx), abs(dy)), abs(dx) + abs(dy), dy, dx)
            for dy in range(-CLICK_NEAR, CLICK_NEAR + 1) for dx in range(-CLICK_NEAR, CLICK_NEAR + 1)
            if 0 <= x + dx < w and 0 <= y + dy < h and board[y + dy][x + dx] != bg]
    if not near:
        return text + "; nothing else within %d cells" % CLICK_NEAR
    _, _, dy, dx = min(near)
    other = _region(board, x + dx, y + dy)
    return text + "; nearest other cell %d,%d (%+d,%+d): color %x, a region of %d cell(s) at %s" % (
        x + dx, y + dy, dx, dy, board[y + dy][x + dx], len(other), _span(other))


'''
CLICK_EDITS = (
    ("                                 play actions in order; the plan is saved in record/history.jsonl\n",
     "                                 play actions in order; the plan is saved in record/history.jsonl\n"
     "                                 every ACTION6 also prints the cell it hit: color and region\n"),
    ('    before_levels = get("/api/status").get("levels_completed") or 0\n',
     '    status = get("/api/status")\n'
     '    before_levels = status.get("levels_completed") or 0\n'
     '    board = (status.get("frame") or [None])[-1]\n'),
    ('        obs = post("/api/cmd/%s" % name, payload)\n'
     '        print("%s -> %s%s" % (token, describe(obs), recorded(obs)))\n',
     '        clicked = _click_text(board, payload["x"], payload["y"]) if name == COORD_ACTION else None\n'
     '        obs = post("/api/cmd/%s" % name, payload)\n'
     '        print("%s -> %s%s" % (token, describe(obs), recorded(obs)))\n'
     '        board = (obs.get("frame") or [board])[-1]\n'
     '        if clicked is not None:\n'
     '            same = (obs.get("record") or {}).get("changed_cells") == 0\n'
     '            print("    " + clicked + ("; the board did not change" if same else ""))\n'),
    ("def cmd_init(ws, args):\n", CLICK_FUNCS + "def cmd_init(ws, args):\n"),
)


def with_click_info(text: str) -> str:
    """``gen2_click_info`` / ``act_click_info``: stacks after with_gen2 and before with_note_gate (cmd_do not yet renamed)."""
    for old, new in CLICK_EDITS:
        if text.count(old) != 1:
            raise ValueError("record actuator changed; update the click info hook: %r" % old[:60])
        text = text.replace(old, new, 1)
    return text


#: The act side of the gen-2 toolkit (``act`` in gen2/toolkit/<version>/MANIFEST.json, 2026-10-27 toolkit loop): the model reads output pushed to it but
#: rarely calls tools on its own (precedent: the first-generation receipt / inspect in docs/research), so ``./act do`` prints these itself. Stacks after
#: with_click_info (uses its ``board``) and before with_note_gate.
#:   anim  -- steps that returned 2+ frames: how many frames, how many differ, where cells changed midway, how many cells changed and changed back
#:            (invisible on the final board).
#:   shift -- when a step changes more than a fifth of the cells, check whether the new board is the old one shifted as a whole (map scrolling); if so,
#:            report the offset.
#:   view  -- if the workspace's ``lib/view.py`` has ``describe(before, after)``, use it at the end of every batch (that did not clear the level) to
#:            describe the batch's changes: how objects are defined is up to the model (the user, 2026-10-27). Runs in a subprocess with a time limit
#:            and a length cap; an error is printed as one line, as is.
#:   map2 -- (tk4, replaces map) the fixed mode of the same viewtrack.py: map/LNN.txt uses fixed world coordinates (top-left of this level's first
#:            screen = (0, 0), rulers labelled in world x / y), an out-of-view cell whose last-seen colour differs from its first-seen one is marked
#:            '?' (stale); it also writes map/LNN.npz for lib/arc.py's load_map() and each step's viewport position to .act_view/camera.jsonl for
#:            cam(step); answers carry the scroll amount, on a scroll step the state line first moves the old state's coordinates along before
#:            comparing (whatever only moves with the camera is not a change), and "state unchanged but cells changed" is computed on aligned boards.
#:            2026-10-27 round 2/3 study: bp35's map row numbers were renumbered on every scroll ("the map's coordinate system shifted"), stale pixels
#:            were left behind ("STALE bubble pixels"), and there was no loader (6 scripts reading the map); on 43/45 scroll steps the state line
#:            was just the scroll itself.
#:   movers -- (tk4, on top of events) events' key-frame line becomes "who moves how at which frame": the small colours whose cells changed in
#:            this step (not the board's most common colour, at most EV_SMALL cells in every frame), listed in order of when they start changing,
#:            with frame range, cell counts and bounding boxes before/after, and the frame where they appear / vanish; the file name comes last.
#:            2026-10-27 round 2 study: the PNG named on the key-frame line was opened only 3/117 times, and the model wrote 79 frame-by-frame
#:            colour-search loops across 13 games; study E: tu93's cause of death (who moves first) and the program order of sb26 / tn36 exist
#:            only in the frames. lib/arc.py's movers(step) uses the same algorithm.
TOOLKIT_ACT = ("anim", "shift", "view", "events", "map", "map2", "state", "digest", "movers")
MOVERS_FUNCS = r'''EV_MOVERS = 4          # most movers named on the key-frame line
MV_EDGE = 2            # the outer rows / columns (counters, step bars) are left out
MV_CHARS = 160         # most characters said about one colour


def _boxes(board, usual):
    # colour -> [cells, x0, y0, x1, y1] of every colour but the usual one, away from the edge band
    out = {}
    h, w = len(board), len(board[0])
    for y, row in enumerate(board):
        if y < MV_EDGE or y >= h - MV_EDGE:
            continue
        for x, v in enumerate(row):
            if v == usual or x < MV_EDGE or x >= w - MV_EDGE:
                continue
            b = out.get(v)
            if b is None:
                out[v] = [1, x, y, x, y]
            else:
                b[0] += 1
                b[1], b[2], b[3], b[4] = min(b[1], x), min(b[2], y), max(b[3], x), max(b[4], y)
    return out


def _pieces(board, v):
    # the 4-connected pieces of colour v away from the edge band
    h, w = len(board), len(board[0])
    todo = {(x, y) for y in range(MV_EDGE, h - MV_EDGE) for x in range(MV_EDGE, w - MV_EDGE) if board[y][x] == v}
    n = 0
    while todo:
        n += 1
        stack = [todo.pop()]
        while stack:
            x, y = stack.pop()
            for p in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if p in todo:
                    todo.discard(p)
                    stack.append(p)
    return n


def _box_text(b):
    return "%d cell(s) %dx%d at (%d,%d)" % (b[0], b[3] - b[1] + 1, b[4] - b[2] + 1, b[1], b[2])


def _movers(before, seq, limit=EV_MOVERS):
    """What moved in a step: one text per small colour whose cells changed away from the edge band (frame k is
    seq[k]; before is the board before the step), its track cut into runs -- moved as a whole, changed shape or
    size, appeared, gone -- in order. Colours that moved come first, then those that appeared or went, then the
    rest, each by the frame it started. [] when nothing small changed."""
    usual = max(_counts(before).items(), key=lambda kv: kv[1])[0]
    boards = [before] + list(seq)
    series = [_boxes(f, usual) for f in boards]
    colours = {v for boxes in series for v in boxes}
    found = []
    for v in colours:
        if max(boxes[v][0] for boxes in series if v in boxes) > EV_SMALL:
            continue
        track = [boxes.get(v) for boxes in series]
        runs = []                                  # [kind, first frame, last frame, box before, box after]
        for k in range(1, len(track)):
            a, b = track[k - 1], track[k]
            if a == b:
                continue
            if a is None:
                kind = "appeared"
            elif b is None:
                kind = "gone"
            elif a[0] == b[0] and (a[3] - a[1], a[4] - a[2]) == (b[3] - b[1], b[4] - b[2]) \
                    and _pieces(boards[k - 1], v) == 1 and _pieces(boards[k], v) == 1:
                kind = "moved"           # one piece, same size, somewhere else: the same thing moved
            elif _pieces(boards[k - 1], v) > 1 or _pieces(boards[k], v) > 1:
                kind = "region"          # several pieces of one colour: which went where is not known
            else:
                kind = "changed"
            if runs and runs[-1][0] == kind and kind in ("moved", "changed", "region"):
                runs[-1][2], runs[-1][4] = k - 1, b
            else:
                runs.append([kind, k - 1, k - 1, a, b])
        if not runs:
            continue
        parts = []
        for kind, a, b, old, new in runs[:3]:
            when = _frange(a, b)
            if kind == "moved":
                parts.append("moved (%d,%d)->(%d,%d) (%s)" % (old[1], old[2], new[1], new[2], when))
            elif kind == "changed":
                parts.append("changed to %s (%s)" % (_box_text(new), when))
            elif kind == "region":
                parts.append("its pieces changed, now %s in %d piece(s) (%s)" % (
                    _box_text(new), _pieces(boards[b + 1], v), when))
            elif kind == "appeared":
                parts.append("appeared at frame %d (%s)" % (a, _box_text(new)))
            else:
                parts.append("gone at frame %d" % a)
        if len(runs) > 3:
            parts.append("...")
        if track[0] is not None and track[-1] == track[0]:
            parts.append("then back as before")
        k0 = 0 if track[0] is not None else runs[0][1] + 1
        first, n0 = track[k0], _pieces(boards[k0], v)
        head = "colour %x (%s)" % (v, _box_text(first) if n0 == 1 else "%d cell(s) in %d pieces within (%d,%d)-(%d,%d)"
                                   % (first[0], n0, first[1], first[2], first[3], first[4]))
        kinds = {r[0] for r in runs}
        rank = 2 if track[-1] == track[0] else 0 if "moved" in kinds else 1 if kinds & {"appeared", "gone"} else 3
        text = head + ": " + ", ".join(parts)
        found.append((rank, runs[0][1], v, text if len(text) <= MV_CHARS else text[:MV_CHARS - 4] + " ..."))
    found.sort()
    texts = [t for *_, t in found[:limit]]
    if len(found) > limit:
        texts.append("+%d more colour(s)" % (len(found) - limit))
    return texts


'''
MOVERS_KEY_LINE = ('''                lines.append("%s%d frames, %d distinct; key frames: %s" % (
                    "the first %s animation on this level: " % action if first_anim and not cleared else "",
                    len(frames), distinct, ", ".join("frame_%03d.png" % k for k in keys)))
''', '''                try:
                    moving = _movers(before, seq)
                except Exception as exc:          # a game nobody has seen must not stop ./act do: say it, go on
                    moving = ["(movers failed: %s: %s)" % (type(exc).__name__, str(exc)[:120])]
                lines.append("%s%d frames, %d distinct%s; key frames: %s" % (
                    "the first %s animation on this level: " % action if first_anim and not cleared else "",
                    len(frames), distinct, ": " + "; ".join(moving) if moving else "",
                    ", ".join("frame_%03d.png" % k for k in keys)))
''')
TOOLKIT_ACT_FUNCS = r'''SHIFT_MAX = 16         # largest view shift looked for, in cells, each way
VIEW_TIMEOUT = 20      # seconds lib/view.py gets per batch
VIEW_CHARS = 1500      # most characters of its answer shown


def _anim_text(frames):
    # one line on an action that returned several frames: what happened between the first and the last
    n = len(frames)
    if any(len(f) != len(frames[0]) or len(f[0]) != len(frames[0][0]) for f in frames):
        return "animation: %d frames; the board size changes during it -- look at the frames" % n
    distinct = 1 + sum(1 for i in range(1, n) if frames[i] != frames[i - 1])
    first, last = frames[0], frames[-1]
    moving, back = [], 0
    for y in range(len(first)):
        for x in range(len(first[0])):
            v = first[y][x]
            if any(f[y][x] != v for f in frames[1:]):
                moving.append((x, y))
                back += last[y][x] == v
    text = "animation: %d frames, %d distinct" % (n, distinct)
    if moving:
        text += "; cells change at %s" % _span(moving)
    if back:
        text += "; %d cell(s) changed and changed back, so the final board does not show it" % back
    return text + " -- look at the frames"


def _shift_text(a, b):
    # when much of the board changed: is the new board the old one moved as a whole (the view scrolled)?
    h, w = len(a), len(a[0])
    if len(b) != h or len(b[0]) != w:
        return None
    if sum(1 for y in range(h) for x in range(w) if a[y][x] != b[y][x]) * 5 < h * w:
        return None
    counts = {}
    for row in b:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    bg = max(counts, key=counts.get)
    cells = [(x, y) for y in range(1, h - 1, 2) for x in range(1, w - 1, 2) if b[y][x] != bg]
    if len(cells) < 30:
        return None

    def agree(dx, dy):
        hit = tot = 0
        for x, y in cells:
            sx, sy = x - dx, y - dy
            if 0 <= sx < w and 0 <= sy < h:
                tot += 1
                hit += a[sy][sx] == b[y][x]
        return (hit / tot, tot) if tot * 3 >= len(cells) else (0.0, tot)
    still = agree(0, 0)[0]
    best = max(((agree(dx, dy)[0], dx, dy) for dy in range(-SHIFT_MAX, SHIFT_MAX + 1)
                for dx in range(-SHIFT_MAX, SHIFT_MAX + 1) if dx or dy), key=lambda r: r[0])
    if best[0] < 0.85 or best[0] < still + 0.3:
        return None
    return ("view shift: the new board is the previous one moved by %+d,%+d (%d%% of its non-background cells "
            "match) -- the map probably scrolled; things did not all move on their own, and what came in at the "
            "edge is new" % (best[1], best[2], round(100 * best[0])))


def _step_notes(before, frames):
    notes = []
    if ACT_ANIM and len(frames) >= 2:
        notes.append(_anim_text(frames))
    if ACT_SHIFT and before is not None and frames:
        shift = _shift_text(before, frames[-1])
        if shift:
            notes.append(shift)
    return notes


def _view(before, after):
    # lib/view.py's describe(before, after) on the batch's first and last boards, if the agent wrote one
    import subprocess
    ws = _workspace()
    if before is None or after is None or not (ws / "lib" / "view.py").is_file():
        return
    code = ("import json, sys\nsys.path.insert(0, %r)\nimport numpy as np\nimport view\n"
            "a, b = (np.array(g, dtype=np.uint8) for g in json.load(sys.stdin))\n"
            "out = view.describe(a, b)\nprint('' if out is None else out)\n" % str(ws / "lib"))
    try:
        done = subprocess.run([sys.executable, "-c", code], input=json.dumps([before, after]), capture_output=True,
                              text=True, timeout=VIEW_TIMEOUT, cwd=str(ws))
    except subprocess.TimeoutExpired:
        print("lib/view.py: describe() took over %d s, nothing shown" % VIEW_TIMEOUT)
        return
    if done.returncode != 0:
        last = (done.stderr.strip().splitlines() or ["(no message)"])[-1]
        print("lib/view.py failed: %s" % last[:300])
        return
    out = done.stdout.strip()
    if out:
        print("lib/view.py describe(first board, last board) of this batch:")
        print(out[:VIEW_CHARS] + (" ... (cut at %d characters)" % VIEW_CHARS if len(out) > VIEW_CHARS else ""))


'''


#:   events -- (tk2, replaces anim) printed only on steps where something happens: back on this level's first board (no RESET sent: walked back to
#:            the start or restarted by itself = lost a life; differing cells are still listed, that is where ls20's lives are; only reported once 3
#:            actions have been sent since the level start / RESET, since in the smoke test bp35 taking one step left and one right was taken for a
#:            restart), which cells differ from the first board after a RESET, the whole screen turning one colour, colours that vanish or appear on
#:            the GAME_OVER / level-clear / back-to-start step, cells that changed and changed back (invisible on the final board), each action's
#:            first animation on this level, and the key-frame file name; on GAME_OVER it lists how many actions were sent since the level start
#:            (or the last RESET) and how many were the same one. Always relative to the board before the step. 2026-10-27 study E:
#:            tk1's anim used frame 0 as the baseline, missed ls20's life loss and m0r0's automatic restart on hitting something, printed on every
#:            step with 1-4 sentence patterns, and got skipped; over v34's 37,210 steps this version prints on 4.4% of steps.
EVENTS_FUNCS = r'''EV_FLASH = 0.9         # one colour over this share of a frame (not the board's usual colour): a full-screen flash
EV_RESTART = 16        # the board within this many cells of the level's first board: the level is back at its start
EV_RESTART_AFTER = 3   # ... said only after this many actions since the level's start or RESET (not for probing a
                       # move and its opposite)
EV_RESET_NEAR = 64     # after a RESET, cells that differ from the level's first board are listed up to this many
EV_SMALL = 400         # a colour of at most this many cells is reported when it vanishes or appears
EV_BACK = 50           # cells that changed and came back during one step: reported from this many on
EV_KEYS = 4            # most key frames named


def _frange(a, b):
    return "frame %d" % a if a == b else "frames %d-%d" % (a, b)


def _counts(board):
    counts = {}
    for row in board:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    return counts


def _differ(a, b):
    return [(x, y) for y in range(len(a)) for x in range(len(a[0])) if a[y][x] != b[y][x]]


def _same_size(*boards):
    return all(len(b) == len(boards[0]) and len(b[0]) == len(boards[0][0]) for b in boards)


def _except(cells, old, new):
    pairs = sorted({(old[y][x], new[y][x]) for x, y in cells})
    return " except %d cell(s) at %s (%s%s)" % (len(cells), _span(cells), ", ".join("%x->%x" % p for p in pairs[:3]),
                                               ", ..." if len(pairs) > 3 else "")


def _key_frames(seq, before):
    # the last frame of each run of transitions that change one area
    ends, box = [], None
    prev = before
    for k, f in enumerate(seq):
        cells = _differ(prev, f)
        prev = f
        if not cells:
            box = None
            continue
        xs = [c[0] for c in cells]
        ys = [c[1] for c in cells]
        new = (min(xs), min(ys), max(xs), max(ys))
        if box is None or new[0] > box[2] + 2 or box[0] > new[2] + 2 or new[1] > box[3] + 2 or box[1] > new[3] + 2:
            ends.append(k)
        else:
            ends[-1] = k
        box = new
    if len(ends) > EV_KEYS:
        ends = [ends[round(i * (len(ends) - 1) / (EV_KEYS - 1))] for i in range(EV_KEYS)]
    return ends


def _events(before, frames, start, cleared, action, state, since, first_anim, slid):
    """Facts about one step, relative to the board before it; [] when nothing stands out.
    start: (step, board) of this level's first board; since: (step, [(step, action), ...]) -- the level's start or its
    last RESET, and the actions sent after it (before this one); first_anim: no earlier step of this action on this
    level returned several frames; slid: the view tracker saw the picture slide (its line says it all)."""
    lines = []
    restarted = False
    if not frames or before is None or not _same_size(before, *frames):
        return lines
    final = frames[-1]
    seq = frames[:-1] if cleared else frames
    first_step, first = start
    if not cleared and _same_size(first, final):
        off = _differ(first, final)
        if action == "RESET":
            if len(off) <= EV_RESET_NEAR:
                lines.append("RESET: the board is this level's first board (step %d)%s" % (
                    first_step, _except(off, first, final) if off else " again"))
        elif len(off) <= EV_RESTART and len(_differ(first, before)) >= len(off) + 16 \
                and len(since[1]) >= EV_RESTART_AFTER:
            restarted = True
            lines.append("the board is this level's first board (step %d) again%s, without a RESET: you walked back "
                         "to the start, or the level restarted (a lost life?)" % (
                             first_step, _except(off, first, final) if off else ""))
    flash = []
    if len(seq) >= 2 or cleared:
        usual = max(_counts(before).items(), key=lambda kv: kv[1])[0]
        for k, f in enumerate(seq):
            v, n = max(_counts(f).items(), key=lambda kv: kv[1])
            if v != usual and n >= EV_FLASH * len(f) * len(f[0]):
                flash.append((k, v))
        if flash:
            lines.append("%s: the whole screen is colour %x" % (_frange(flash[0][0], flash[-1][0]), flash[0][1]))
    # colours that vanish or appear (on a clear: up to the old level's last frame)
    end = seq[-1] if seq else final
    if action != "RESET" and (state == "GAME_OVER" or cleared or restarted):
        was, now = _counts(before), _counts(end)
        for v in sorted(was):
            if v not in now and was[v] <= EV_SMALL:
                cells = [(x, y) for y in range(len(before)) for x in range(len(before[0])) if before[y][x] == v]
                gone = next((k for k in range(len(seq)) if all(v not in row for row in seq[k])), len(seq) - 1)
                lines.append("colour %x (%d cell(s) at %s before) is gone%s" % (
                    v, was[v], _span(cells), " from frame %d on" % gone if len(seq) >= 2 else ""))
        for v in sorted(now):
            if v not in was and now[v] <= EV_SMALL and not cleared:
                cells = [(x, y) for y in range(len(end)) for x in range(len(end[0])) if end[y][x] == v]
                lines.append("colour %x appeared: %d cell(s) at %s" % (v, now[v], _span(cells)))
    if len(seq) >= 2 and (not slid or state == "GAME_OVER"):
        moved = set()
        prev = before
        for f in seq:
            moved.update(_differ(prev, f))
            prev = f
        back = [(x, y) for x, y in moved if end[y][x] == before[y][x]]
        distinct = 1 + sum(1 for k in range(1, len(seq)) if seq[k] != seq[k - 1])
        if len(back) >= EV_BACK and not flash and (first_anim or lines or state == "GAME_OVER"):
            lines.append("%d cell(s) at %s changed during the step and are back to how they were: only the frames "
                         "show it" % (len(back), _span(back)))
        if len(seq) >= 3 and (lines or state == "GAME_OVER" or (first_anim and distinct >= 3)):
            keys = _key_frames(seq, before)
            if keys:
                lines.append("%s%d frames, %d distinct; key frames: %s" % (
                    "the first %s animation on this level: " % action if first_anim and not cleared else "",
                    len(frames), distinct, ", ".join("frame_%03d.png" % k for k in keys)))
    if state == "GAME_OVER":
        origin, sent = since
        same = [s for s, a in sent if a == action]
        lines.append("GAME_OVER on %s: %d action(s) since step %d (this level's start or its last RESET), "
                     "%d of them %s%s" % (action, len(sent), origin, len(same), action,
                                          " (steps %s)" % ", ".join(str(s) for s in same[-6:]) if same else ""))
    return lines


def _level_context():
    # from record/: this level's first board, the actions since its start or last RESET, the actions that animated
    record = _workspace() / "record"
    level, done = {"start": 0, "since": (0, []), "animated": set()}, 0
    for line in (record / "history.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["step"] == 0:
            continue
        if row["levels_completed"] > done:
            done = row["levels_completed"]
            level = {"start": row["step"], "since": (row["step"], []), "animated": set()}
        else:
            _level_update(level, row["step"], row["action"], row["frames"])
    final = record / "steps" / ("%06d" % level["start"]) / "final.txt"
    level["first"] = [[int(ch, 16) for ch in line] for line in final.read_text().split()]
    return level


def _level_update(level, step, action, frames):
    if action == "RESET":
        level["since"] = (step, [])
    else:
        level["since"][1].append((step, action))
    if frames >= 2:
        level["animated"].add(action)


'''


#:   map  -- (tk2, replaces shift) src/arcforge/arc_code/viewtrack.py as a subprocess of act (needs numpy; started once per batch, one question
#:            and answer per step): follows the animation step by step to compute the whole screen's translation (map scrolling), reports which way
#:            it scrolled, how coordinates convert, which rows are new; stitches what this level has shown into map/LNN.txt; state is kept in
#:            .act_view/ (recomputed from record when missing or mismatched). Clicks in the same batch after the screen has moved are not sent
#:            (in lf52 43% of such clicks had no effect, elsewhere 11%). 2026-10-27 study F: of the 25 titles only bp35 and lf52 scroll; tk1's shift
#:            caught only 32%; over 20 games / 7,819 steps of those two titles this version got the viewport position right on 7,807 steps, marked 12
#:            as unknown and 0 wrong, with 0 false positives on the other 23 titles.
MAP_FUNCS = r'''MAP_TIMEOUT = 60       # seconds the view tracker gets to start (it may replay the level) or to answer a step


class _Map:
    # the view tracker (numpy) in a child process: one JSON request per step, one JSON answer per line
    def __init__(self):
        import subprocess
        self.proc = subprocess.Popen([sys.executable, "-c", VIEW_TRACKER, str(_workspace())], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.failed = False
        self._answer()

    def _answer(self):
        import select
        if not select.select([self.proc.stdout], [], [], MAP_TIMEOUT)[0]:
            self.proc.kill()
            return self._fail("no answer in %d s" % MAP_TIMEOUT)
        line = self.proc.stdout.readline()
        if not line:
            self.proc.wait()
            last = (self.proc.stderr.read().strip().splitlines() or ["(no message)"])[-1]
            return self._fail(last[:300])
        return json.loads(line)

    def _fail(self, why):
        if not self.failed:
            print("    map: the view tracker failed: %s; no view lines for the rest of this call" % why)
        self.failed = True
        return None

    def step(self, step, action, cleared):
        if self.failed:
            return [], False
        self.proc.stdin.write(json.dumps({"step": step, "action": action, "cleared": cleared}) + "\n")
        self.proc.stdin.flush()
        answer = self._answer()
        return (answer["lines"], answer["moved"]) if answer else ([], False)

    def close(self):
        if not self.failed:
            self.proc.stdin.write(json.dumps({"end": True}) + "\n")
            self.proc.stdin.flush()
            self._answer()


'''


MAP2_FUNCS = MAP_FUNCS.replace('VIEW_TRACKER, str(_workspace())]', 'VIEW_TRACKER, str(_workspace()), "fixed"]', 1) \
    .replace('            return [], False\n', '            return [], False, None\n', 1) \
    .replace('        return (answer["lines"], answer["moved"]) if answer else ([], False)\n',
             '        return (answer["lines"], answer["moved"], answer["shift"] and (answer["shift"], answer["pos"])) '
             'if answer else ([], False, None)\n', 1)
if MAP2_FUNCS.count('"fixed"') != 1 or MAP2_FUNCS.count("False, None") != 2:
    raise ValueError("MAP_FUNCS changed; update MAP2_FUNCS")


#:   state -- (tk2, replaces view) the model's own object definition: if the workspace's ``lib/view.py`` has ``state(board[, level]) -> dict``,
#:            every batch starts a subprocess (the model's code needs numpy; act uses only the standard library), every step's board is sent in,
#:            and the harness compares consecutive states and prints one line (which keys changed, what lists gained or lost); it also reports cells
#:            inside the border changing while state did not (the model's detector missed something) and state changing while the board did not;
#:            on level clear it runs once on the new level's first board and prints the result (where a changed definition breaks). Errors print
#:            step, level and line number, once per batch. 2026-10-27 study B: tk1's describe(a, b) only compared a batch's first and last board,
#:            0/110 games had existing code that fit it, and no game in round 1 wrote one; 61% of games already had a parser returning a dict /
#:            list, which plugs into this interface in 1-3 lines.
STATE_CODE = r'''import inspect
import json
import sys
import traceback

import numpy as np

WS = sys.argv[1]
sys.path.insert(0, WS + "/lib")


def plain(v):
    if isinstance(v, dict):
        return {str(k): plain(x) for k, x in v.items()}
    if isinstance(v, (set, frozenset)):
        return sorted((plain(x) for x in v), key=repr)
    if isinstance(v, (list, tuple)):
        return [plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return plain(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return repr(v)


def where(exc):
    tb = traceback.extract_tb(exc.__traceback__)
    mine = [f for f in tb if f.filename.startswith(WS + "/lib/")]
    f = (mine or tb or [None])[-1]
    return "" if f is None else " (%s:%d in %s)" % (f.filename[len(WS) + 1:] if f.filename.startswith(WS) else
                                                      f.filename, f.lineno, f.name)


try:
    import view
    fn = view.state
    takes_level = len(inspect.signature(fn).parameters) >= 2
except Exception as exc:
    print(json.dumps({"fatal": "%s: %s%s" % (type(exc).__name__, str(exc)[:200], where(exc))}), flush=True)
    sys.exit(0)
print(json.dumps({"ready": True}), flush=True)
for line in sys.stdin:
    ask = json.loads(line)
    board = np.array(ask["board"], dtype=np.uint8)
    try:
        s = plain(fn(board, ask["level"]) if takes_level else fn(board))
        if not isinstance(s, dict):
            s = {"objects" if isinstance(s, list) else "state": s}
        print(json.dumps({"state": s}), flush=True)
    except Exception as exc:
        print(json.dumps({"error": "%s: %s%s" % (type(exc).__name__, str(exc)[:200], where(exc))}), flush=True)
'''
STATE_FUNCS = r'''STATE_TIMEOUT = 10     # seconds lib/view.py gets to start, then for each board
STATE_CHARS = 160      # most characters of one view line
SLIDE_CHARS = 240      # ... of one on a slide step (map2), which also says what moved only with the window
SLIDE_TILE_MAX = 8     # the largest tile size (cells) a slide state line tries as the agent's unit
STATE_START_CHARS = 300
STATE_EDGE = 4         # cells this close to the border (counters, timers) do not count as "cells changed"


class _State:
    # lib/view.py's state(board, level) in a child process, one per `./act do`: one board in, one state out
    def __init__(self):
        import subprocess
        self.proc, self.errors = None, 0
        if not (_workspace() / "lib" / "view.py").is_file():
            return
        self.proc = subprocess.Popen([sys.executable, "-c", STATE_CODE, str(_workspace())], cwd=str(_workspace()),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        answer = self._answer()
        if answer is not None and "fatal" in answer:
            print("    view: lib/view.py cannot be used: %s" % answer["fatal"])
            self.proc = None

    def _answer(self):
        import select
        if not select.select([self.proc.stdout], [], [], STATE_TIMEOUT)[0]:
            self.proc.kill()
            print("    view: lib/view.py took over %d s; no view for the rest of this call" % STATE_TIMEOUT)
            self.proc = None
            return None
        line = self.proc.stdout.readline()
        if not line:
            self.proc.wait()
            last = (self.proc.stderr.read().strip().splitlines() or ["(no message)"])[-1]
            print("    view: lib/view.py stopped: %s; no view for the rest of this call" % last[:300])
            self.proc = None
            return None
        return json.loads(line)

    def ask(self, board, level, where):
        """this board's state, or None (after one line saying why, the first time in this call)"""
        if self.proc is None or board is None:
            return None
        self.proc.stdin.write(json.dumps({"board": board, "level": level}) + "\n")
        self.proc.stdin.flush()
        answer = self._answer()
        if answer is None:
            return None
        if "error" in answer:
            self.errors += 1
            if self.errors == 1:
                print("    view: lib/view.py failed on %s (level %d): %s" % (where, level, answer["error"]))
            return None
        return answer["state"]

    def close(self):
        if self.proc is not None:
            self.proc.stdin.close()
            self.proc.wait()
            if self.errors > 1:
                print("    view: lib/view.py failed on %d more step(s) in this call" % (self.errors - 1))


def _short(v):
    text = json.dumps(v, separators=(",", ":"))
    return text if len(text) <= 50 else text[:47] + "..."


def _change(a, b):
    # old->new; two long values are shown from where they first differ, so the change is in the line
    ta, tb = json.dumps(a, separators=(",", ":")), json.dumps(b, separators=(",", ":"))
    if len(ta) <= 50 and len(tb) <= 50:
        return ta + "->" + tb
    i = next((k for k in range(min(len(ta), len(tb))) if ta[k] != tb[k]), min(len(ta), len(tb)))
    j = max(i - 10, 0)

    def cut(text):
        return ("..." if j else "") + text[j:i + 30] + ("..." if len(text) > i + 30 else "")
    return cut(ta) + "->" + cut(tb)


def _flat(state, prefix=""):
    # nested dicts as dotted keys: {"shapes": {"d": {"center": [1, 2]}}} -> {"shapes.d.center": [1, 2]}
    out = {}
    for k, v in state.items():
        if isinstance(v, dict) and v:
            out.update(_flat(v, prefix + str(k) + "."))
        else:
            out[prefix + str(k)] = v
    return out


def _point(v):
    # a short list of plain values, like a position (x, y): one value, not a collection of things
    return len(v) <= 4 and all(not isinstance(x, (list, dict)) for x in v)


SLIDE_KEEP = ("size", "shape", "dim", "wh", "count", "len", "num", "width", "height")   # keys that hold no position


def _slid(v, dx, dy):
    # v as it reads once every position in it follows the picture's slide (dx, dy): a pair of ints (x, y) or a box of
    # four (x0, y0, x1, y1) in a list, or a dict's "x" and "y" -- except under a key named like a size or a count
    def num(x):
        return isinstance(x, int) and not isinstance(x, bool)
    if isinstance(v, dict):
        out = {k: x if any(w in str(k).lower() for w in SLIDE_KEEP) else _slid(x, dx, dy) for k, x in v.items()}
        if num(v.get("x")) and num(v.get("y")):
            out["x"], out["y"] = v["x"] + dx, v["y"] + dy
        return out
    if isinstance(v, list):
        if len(v) == 2 and all(num(x) for x in v):
            return [v[0] + dx, v[1] + dy]
        if len(v) == 4 and all(num(x) for x in v):
            return [v[0] + dx, v[1] + dy, v[2] + dx, v[3] + dy]
        return [_slid(x, dx, dy) for x in v]
    return v


def _slide_scales(dx, dy):
    # the units the agent's state may count in: board cells, or tiles of s cells that the slide is a whole number of
    return [1] + [k for k in range(2, SLIDE_TILE_MAX + 1) if (dx or dy) and dx % k == 0 and dy % k == 0]


def _slide_key(a, moved, b):
    # one key on a slide: a its old value, moved the same moved along with the window (in the agent's units), b the
    # new value -> (moved-with-the-window count, stayed-put flag, leftover text or None). A list of things is matched
    # item by item; the rest is said as counts, not guessed at.
    if isinstance(a, list) and isinstance(b, list) and not (_point(a) and _point(b)):
        left = {}
        for old_item, moved_item in zip(a, moved):
            k = json.dumps(moved_item, sort_keys=True)
            left.setdefault(k, []).append(old_item != moved_item)
        along = came = 0
        for v in b:
            k = json.dumps(v, sort_keys=True)
            if left.get(k):
                along += left[k].pop()
            else:
                came += 1
        gone = sum(len(v) for v in left.values())
        return along, False, (None if not gone and not came else
                              "%d left the view or changed, %d came into view or changed" % (gone, came))
    if a == b:
        return 0, moved != a, None
    if moved == b:
        return 1, False, None
    return 0, False, _change(a, b)


def _slide_line(old, new, before, after, slide):
    # the picture slid by (dx, dy) cells. The agent's state is in its own units (board cells, or tiles: bp35's agents
    # counted 6-cell tiles, and adding the slide in cells made 79 of 84 such lines nonsense in round 4), so nothing is
    # converted: each key is checked against the old value moved along with the window, in the unit that explains the
    # most, and what is left is said as the agent's own values or as counts
    (dx, dy), _ = (slide, None) if isinstance(slide[0], int) else slide
    h, w = len(after), len(after[0])
    flat_old, flat_new = _flat(old), _flat(new)
    keys = sorted(set(flat_old) | set(flat_new), key=str)
    best = None
    for scale in _slide_scales(dx, dy):
        flat_moved = _flat(_slid(old, dx // scale, dy // scale))
        rows = {k: _slide_key(flat_old[k], flat_moved[k], flat_new[k]) for k in keys if k in flat_old and k in flat_new}
        along = sum(r[0] for r in rows.values())
        if best is None or along > best[1]:
            best = (scale, along, rows)
    scale, along, rows = best
    said = ["view: the picture slid (%+d, %+d)" % (dx, dy)]
    if along:
        said.append("%d thing(s) in your state moved with the window%s" % (
            along, "" if scale == 1 else " (your state seems to count tiles of %d cells)" % scale))
    stayed = [k for k, r in rows.items() if r[1]]
    if stayed:
        said.append("stayed put on the board: %s%s (a HUD value, or something the camera follows)"
                    % (", ".join(stayed[:3]), " +%d" % (len(stayed) - 3) if len(stayed) > 3 else ""))
    # single values (a player, a door) before the counts of lists of things, which are long and less telling
    rest = sorted(["%s: %s" % (k, r[2]) for k, r in rows.items() if r[2]], key=lambda t: " left the view or " in t)
    rest += ["%s %s->%s" % (k, "(none)" if k not in flat_old else _short(flat_old[k]),
                            "(none)" if k not in flat_new else _short(flat_new[k]))
             for k in keys if (k in flat_old) != (k in flat_new)]
    if rest:
        said.append("besides that, in your own values: " + "; ".join(rest[:4])
                    + ("; +%d more key(s)" % (len(rest) - 4) if len(rest) > 4 else ""))
    else:
        if not along and not stayed:
            said.append("your state did not change")
        inner = [(x, y) for y in range(STATE_EDGE, h - STATE_EDGE) for x in range(STATE_EDGE, w - STATE_EDGE)
                 if 0 <= y - dy < h and 0 <= x - dx < w and before[y - dy][x - dx] != after[y][x]]
        if inner:
            said.append("%d cell(s) changed in %s once the slide is undone" % (len(inner), _span(inner)))
    line = said[0] + ": " + "; ".join(said[1:])
    return line if len(line) <= SLIDE_CHARS else line[:SLIDE_CHARS - 4] + " ..."


def _state_line(old, new, before, after, shift=None):
    # one line on how the agent's own state changed, checked against the board; shift: the picture slid (map2)
    h, w = len(after), len(after[0])
    if shift is not None:
        try:
            return _slide_line(old, new, before, after, shift)
        except Exception as exc:                  # a game nobody has seen must not stop ./act do: say it, go on
            return "view: the picture slid; comparing your state failed: %s: %s" % (type(exc).__name__,
                                                                                    str(exc)[:120])
    if old == new:
        inner = [(x, y) for y in range(STATE_EDGE, h - STATE_EDGE) for x in range(STATE_EDGE, w - STATE_EDGE)
                 if before[y][x] != after[y][x]]
        if inner:
            return "view: your state did not change, but %d cell(s) changed in %s" % (len(inner), _span(inner))
        return None
    if before == after:
        return "view: your state changed but the board did not -- lib/view.py depends on something besides the board"
    parts = _parts(_flat(old), _flat(new))
    line = "view: " + "; ".join(parts[:4]) + ("; +%d more key(s)" % (len(parts) - 4) if len(parts) > 4 else "")
    return line if len(line) <= STATE_CHARS else line[:STATE_CHARS - 4] + " ..."


def _parts(old, new):
    # old, new: flat states; one "key change" text per key that differs
    parts = []
    for k in sorted(set(old) | set(new), key=str):
        a, b = old.get(k), new.get(k)
        if a == b:
            continue
        if isinstance(a, list) and isinstance(b, list) and not (_point(a) and _point(b)):
            count = {}
            for v in a:
                count[json.dumps(v, sort_keys=True)] = count.get(json.dumps(v, sort_keys=True), 0) - 1
            for v in b:
                count[json.dumps(v, sort_keys=True)] = count.get(json.dumps(v, sort_keys=True), 0) + 1
            gone = [json.loads(v) for v, n in count.items() for _ in range(-n)]
            came = [json.loads(v) for v, n in count.items() for _ in range(n)]
            items = ["-" + _short(v) for v in gone[:3]] + ["+" + _short(v) for v in came[:3]]
            more = len(gone) + len(came) - len(items)
            parts.append("%s %s%s" % (k, " ".join(items) or "reordered", " (+%d)" % more if more > 0 else ""))
        elif k in old and k in new:
            parts.append("%s %s" % (k, _change(a, b)))
        else:
            parts.append("%s %s->%s" % (k, "(none)" if k not in old else _short(a), "(none)" if k not in new
                                        else _short(b)))
    return parts


'''


#:   digest -- (tk3) reprinted at the end of the batch: the hints from this batch's earlier steps (click changed nothing, events, map, state) and the
#:            change in colour counts of the two edge rows / columns (counters, timer bars). 2026-10-27 round 1 traces: 70% of ./act do calls are
#:            piped into tail / head, and 40% of per-step lines went unseen; the model read counters like b[63] by hand 381 times. Stacks after the
#:            other act features: replaces the prints in cmd_do that emit per-step hints with _say.
CALL_DIGEST_FUNCS = r'''DIGEST_LINES = 10      # most notes repeated at the end of `./act do`
NOTES = []             # (step, note) printed under the steps of this call


def _say(obs, note):
    print("    " + note)
    NOTES.append(((obs.get("record") or {}).get("step"), note))


def _edges(a, b):
    # colour counts that changed in the two outer rows / columns on each side (counters, timers)
    h, w = len(b), len(b[0])
    sides = (("top rows 0-1", [(x, y) for y in (0, 1) for x in range(w)]),
             ("bottom rows %d-%d" % (h - 2, h - 1), [(x, y) for y in (h - 2, h - 1) for x in range(w)]),
             ("left columns 0-1", [(x, y) for x in (0, 1) for y in range(h)]),
             ("right columns %d-%d" % (w - 2, w - 1), [(x, y) for x in (w - 2, w - 1) for y in range(h)]))
    out = []
    for name, cells in sides:
        before, after = {}, {}
        for x, y in cells:
            before[a[y][x]] = before.get(a[y][x], 0) + 1
            after[b[y][x]] = after.get(b[y][x], 0) + 1
        moved = ["colour %x %d->%d" % (c, before.get(c, 0), after.get(c, 0)) for c in sorted(set(before) | set(after))
                 if before.get(c, 0) != after.get(c, 0)]
        if moved:
            out.append("%s: %s" % (name, ", ".join(moved)))
    return out


def _digest(first, last, sent, asked):
    # the end of `./act do`, so that its last lines hold what matters even when the output is cut (codex keeps the
    # head and the tail of a long output; a `| tail` keeps only the tail): every note of the call, the counters,
    # and where the game is now -- always the very last line
    if not sent:
        return
    steps = [s for s, _ in sent]
    notes = [(s, n) for s, n in NOTES if not n.startswith("click ") or n.endswith("the board did not change")]
    edges = _edges(first, last) if first is not None and last is not None and len(first) == len(last) \
        and len(first[0]) == len(last[0]) else []
    if not notes and not edges and len(sent) == asked:
        return
    print("this call, step%s %s:" % ("" if len(steps) == 1 else "s", "%d" % steps[0] if len(steps) == 1
                                     else "%d-%d" % (steps[0], steps[-1])))
    if len(notes) > DIGEST_LINES:
        print("  (%d earlier note(s) above)" % (len(notes) - DIGEST_LINES))
    for s, n in notes[-DIGEST_LINES:]:
        print("  step %s: %s" % (s, n if len(n) <= 200 else n[:196] + " ..."))
    for line in edges:
        print("  %s (counters?)" % line)
    if len(sent) < asked:
        print("  sent %d of the %d action(s) asked; why is above" % (len(sent), asked))
    print("now, after step %s: %s" % (steps[-1], sent[-1][1]))


'''
#: digest replaces these per-step prints in cmd_do with _say (kept, then printed again at the end of the batch)
CALL_DIGEST_SAYS = (('print("    " + clicked + (', '_say(obs, clicked + ('),
               ('print("    " + note)', '_say(obs, note)'),
               ('print("    " + line)', '_say(obs, line)'))


def view_tracker_source() -> str:
    """viewtrack.py verbatim; act runs it with ``python -c``."""
    text = Path(__file__).with_name("viewtrack.py").read_text(encoding="utf-8")
    if "'''" in text:
        raise ValueError("viewtrack.py must not contain triple single quotes: act embeds it in r'''...'''")
    return text


#: The first-generation toolkit act's do before the game is open (hs110 v39): the first generation opens the game with ./act init; when do came
#: first, the middleware used to open the game along with the first action, but the toolkit needs record/history.jsonl before sending actions and
#: it does not exist yet then (v38 notes: not hit in G4's 110 games, not fixed). Open the game first the way init does (a RESET before the game is
#: open costs no action); a leading RESET was meant to open the game anyway, so swallow it.
GEN1_DO_OPEN = ('def cmd_do(ws, args):\n    status = get("/api/status")\n',
                '    if not status.get("started"):\n'
                '        obs = post("/api/cmd/RESET", {})\n'
                '        print("started %s: %s%s" % (obs.get("game_id"), describe(obs), recorded(obs)))\n'
                '        if args.actions and args.actions[0] == "RESET":\n'
                '            args.actions = args.actions[1:]\n'
                '        status = get("/api/status")\n')


def with_toolkit_act(text: str, features: tuple[str, ...], gen1: bool = False) -> str:
    """The toolkit's act side: stacks after with_click_info and before with_note_gate. ``features`` is a subset of TOOLKIT_ACT.
    ``gen1`` (2026-10-30 hs110 v38: v35's first-generation arm + tk5): stacks on the first generation's pure record + click_info actuator.
    The first-generation do lacks the GAME_OVER hint with_gen2 adds, so the state line is attached in front of the line that follows it in
    gen-2 (``cleared = ...``) instead, keeping the same order as gen-2."""
    unknown = set(features) - set(TOOLKIT_ACT)
    if unknown or not features or len(set(features)) != len(features):
        raise ValueError(f"toolkit act features must be distinct names from {TOOLKIT_ACT}, got {features}")
    if "anim" in features and "events" in features:
        raise ValueError("toolkit act: events replaces anim; pick one")
    if len({"shift", "map", "map2"} & set(features)) > 1:
        raise ValueError("toolkit act: map and map2 replace shift; pick one of shift, map, map2")
    if "view" in features and "state" in features:
        raise ValueError("toolkit act: state replaces view; pick one")
    if "movers" in features and "events" not in features:
        raise ValueError("toolkit act: movers rewrites the events' key-frame line; it needs events")
    consts = "ACT_ANIM = %r\nACT_SHIFT = %r\n" % ("anim" in features, "shift" in features)
    if gen1:
        # The toolkit's functions use _workspace(); in gen-2 with_note_gate brings it (stacked later), the first generation has no note_gate
        if "def _workspace():" in text:
            raise ValueError("gen-1 toolkit act: the actuator already defines _workspace")
        consts = WORKSPACE_FUNC + "\n\n" + consts
    edits = [
        ("def cmd_init(ws, args):\n", consts + TOOLKIT_ACT_FUNCS + "def cmd_init(ws, args):\n"),
        ('    board = (status.get("frame") or [None])[-1]\n',
         '    board = (status.get("frame") or [None])[-1]\n    first_board = board\n'),
        ('        board = (obs.get("frame") or [board])[-1]\n',
         '        before, board = board, (obs.get("frame") or [board])[-1]\n'),
        ('            print("    " + clicked + ("; the board did not change" if same else ""))\n',
         '            print("    " + clicked + ("; the board did not change" if same else ""))\n'
         '        for note in _step_notes(before, obs.get("frame") or []):\n'
         '            print("    " + note)\n'),
    ]
    notes = '        for note in _step_notes(before, obs.get("frame") or []):\n            print("    " + note)\n'
    if "events" in features:
        events_funcs = EVENTS_FUNCS
        if "movers" in features:
            if EVENTS_FUNCS.count(MOVERS_KEY_LINE[0]) != 1:
                raise ValueError("EVENTS_FUNCS changed; update MOVERS_KEY_LINE")
            events_funcs = MOVERS_FUNCS + EVENTS_FUNCS.replace(*MOVERS_KEY_LINE)
        edits += [
            ("def cmd_init(ws, args):\n", events_funcs + "def cmd_init(ws, args):\n"),
            ('    first_board = board\n', '    first_board = board\n    level = _level_context()\n'),
            (notes, notes +
             '        frames = obs.get("frame") or []\n'
             '        for note in _events(before, frames, (level["start"], level["first"]),\n'
             '                            (obs.get("levels_completed") or 0) > before_levels, name, obs.get("state"),\n'
             '                            level["since"], name not in level["animated"], %s):\n'
             '            print("    " + note)\n'
             '        _level_update(level, obs["record"]["step"], name, len(frames))\n'
             % ("moved" if "map" in features or "map2" in features else "False")),
        ]
    if "map" in features:
        edits += [
            ("def cmd_init(ws, args):\n", "VIEW_TRACKER = r'''" + view_tracker_source() + "'''\n\n\n" + MAP_FUNCS
             + "def cmd_init(ws, args):\n"),
            ('    first_board = board\n', '    first_board = board\n    view_map, view_moved = _Map(), None\n'),
            ('        payload = {}\n        if name == COORD_ACTION:\n',
             '        payload = {}\n'
             '        if name == COORD_ACTION and view_moved is not None:\n'
             '            print("stopped before %s: the view slid at step %d in this call, so a click planned on the "\n'
             '                  "board before that lands somewhere else -- look at the board again, then click. "\n'
             '                  "Dropped the remaining %d action(s), not sent" % (token, view_moved,\n'
             '                                                                   len(args.actions) - index))\n'
             '            break\n'
             '        if name == COORD_ACTION:\n'),
            (notes,
             '        lines, moved = view_map.step(obs["record"]["step"], name,\n'
             '                                     (obs.get("levels_completed") or 0) > before_levels)\n'
             '        for note in lines:\n'
             '            print("    " + note)\n'
             '        if moved:\n'
             '            view_moved = obs["record"]["step"]\n' + notes),
        ]
    if "map2" in features:
        edits += [
            ("def cmd_init(ws, args):\n", "VIEW_TRACKER = r'''" + view_tracker_source() + "'''\n\n\n" + MAP2_FUNCS
             + "def cmd_init(ws, args):\n"),
            ('    first_board = board\n', '    first_board = board\n    view_map, view_moved = _Map(), None\n'),
            ('        payload = {}\n        if name == COORD_ACTION:\n',
             '        payload = {}\n'
             '        if name == COORD_ACTION and view_moved is not None:\n'
             '            print("stopped before %s: the view slid at step %d in this call, so a click planned on the "\n'
             '                  "board before that lands somewhere else -- look at the board again, then click. "\n'
             '                  "Dropped the remaining %d action(s), not sent" % (token, view_moved,\n'
             '                                                                   len(args.actions) - index))\n'
             '            break\n'
             '        if name == COORD_ACTION:\n'),
            (notes,
             '        lines, moved, view_shift = view_map.step(obs["record"]["step"], name,\n'
             '                                                 (obs.get("levels_completed") or 0) > before_levels)\n'
             '        for note in lines:\n'
             '            print("    " + note)\n'
             '        if moved:\n'
             '            view_moved = obs["record"]["step"]\n' + notes),
        ]
    if "view" in features:
        edits.append(('            break\n\n\ndef cmd_status(ws, args):\n',
                      '            break\n    if not cleared:\n        _view(first_board, board)\n\n\n'
                      'def cmd_status(ws, args):\n'))
    if "state" in features:
        over = ('        cleared = (obs.get("levels_completed") or 0) > before_levels\n' if gen1 else
                '        if obs.get("state") == "GAME_OVER":\n            print("game over on this level')
        edits += [
            ("def cmd_init(ws, args):\n", "STATE_CODE = r\'\'\'" + STATE_CODE + "\'\'\'\n" + STATE_FUNCS
             + "def cmd_init(ws, args):\n"),
            ('    first_board = board\n', '    first_board = board\n'
             '    own = _State()\n'
             '    own_state = own.ask(board, before_levels + 1, "the board before this call")\n'),
            (over,
             '        done = obs.get("levels_completed") or 0\n'
             '        if done > before_levels:\n'
             '            own_state = own.ask(board, done + 1, "the first board of level %d" % (done + 1))\n'
             '            if own_state is not None:\n'
             '                shown = json.dumps(own_state, separators=(",", ":"))\n'
             '                print("    view: level %d starts -- your state: %s" % (done + 1, shown if len(shown) <= '
             'STATE_START_CHARS\n'
             '                      else shown[:STATE_START_CHARS - 4] + " ..."))\n'
             '        else:\n'
             '            now = own.ask(board, done + 1, "step %d" % obs["record"]["step"])\n'
             '            line = _state_line(own_state, now, before, board' + (", view_shift" if "map2" in features else "")
             + ') if None not in (own_state, now) else None\n'
             '            if line:\n'
             '                print("    " + line)\n'
             '            own_state = now\n' + over),
        ]
    # after view's edit: these go at the end of cmd_do
    if "map" in features or "map2" in features:
        edits.append(('\n\n\ndef cmd_status(ws, args):\n', '\n    view_map.close()\n\n\ndef cmd_status(ws, args):\n'))
    if "state" in features:
        edits.append(('\n\n\ndef cmd_status(ws, args):\n', '\n    own.close()\n\n\ndef cmd_status(ws, args):\n'))
    if "digest" in features:
        shown = '        print("%s -> %s%s" % (token, describe(obs), recorded(obs)))\n'
        edits += [
            ("def cmd_init(ws, args):\n", CALL_DIGEST_FUNCS + "def cmd_init(ws, args):\n"),
            ('    first_board = board\n', '    first_board = board\n    sent = []\n'),
            (shown, shown + '        sent.append(((obs.get("record") or {}).get("step"), "%s -> %s%s" % (token, describe(obs), '
                            'recorded(obs))))\n'),
            ('\n\n\ndef cmd_status(ws, args):\n',
             '\n    _digest(first_board, board, sent, len(args.actions))\n\n\ndef cmd_status(ws, args):\n'),
        ]
    if gen1:
        edits.append((GEN1_DO_OPEN[0], GEN1_DO_OPEN[0] + GEN1_DO_OPEN[1]))
    for old, new in edits:
        if text.count(old) != 1:
            raise ValueError("record actuator changed; update the toolkit act hook: %r" % old[:60])
        text = text.replace(old, new, 1)
    if "digest" in features:
        start, end = text.index("def cmd_do(ws, args):\n"), text.index("def cmd_status(ws, args):\n")
        body = text[start:end]
        for old, new in CALL_DIGEST_SAYS:
            body = body.replace(old, new)
        if "print(\"    \" + " in body:
            raise ValueError("toolkit act digest: a per-step note in cmd_do is still a plain print")
        text = text[:start] + body + text[end:]
    return text


def with_note_gate(text: str) -> str:
    """``note_gate`` (config): see NOTE_GATE_HOOK above. Stacks only on the pure record actuator (v14 arms)."""
    marker = "def main(argv=None):\n"
    do, board = "def cmd_do(ws, args):\n", "def cmd_board(ws, args):\n"
    if any(text.count(s) != 1 for s in (marker, do, board, NOTE_GATE_DOC, NOTE_GATE_PARSE)):
        raise ValueError("record actuator changed; update the note_gate hook")
    text = text.replace(do, "def _cmd_do_batch(ws, args):\n", 1).replace(board, "def _cmd_board_open(ws, args):\n", 1)
    text = text.replace(NOTE_GATE_DOC, NOTE_GATE_DOC_NEW, 1).replace(NOTE_GATE_PARSE, NOTE_GATE_PARSE_NEW, 1)
    return text.replace(marker, NOTE_GATE_HOOK + marker, 1)


def render(record: bool = False, observation_tools: bool = False, receipt: bool = False, scene: bool = False,
           images: bool = False, library: bool = False, recap: bool = False, *, expect: bool = False,
           notes_budget: int | None = None, digest: bool = False, note_gate: bool = False,
           gen2: bool = False, note_done_snapshot: bool = False, click_info: bool = False,
           toolkit_act: tuple[str, ...] = (), pace: bool = False) -> str:
    if toolkit_act and not click_info:
        raise ValueError("toolkit act features go on top of click_info")
    if toolkit_act and not gen2 and note_gate:
        raise ValueError("gen-1 toolkit act goes on the plain record + click_info actuator (hs110 v38), not note_gate")
    if note_done_snapshot and not gen2:
        raise ValueError("note_done_snapshot is a gen-2 seed collection feature")
    if click_info and (not record or observation_tools or receipt or scene or images or library or recap or expect
                       or notes_budget is not None or digest):
        raise ValueError("click_info goes on the plain record actuator (gen-1 v14 arm, note_gate, or gen2)")
    if pace and not gen2:
        raise ValueError("pace is a gen-2 feature (the RL harness writes .pace.json for the gen-2 act)")
    if gen2:
        # gen-2: pure record actuator + harness opens the game + note_gate; the first generation's other opt-ins are not stacked
        if (not record or not note_gate or observation_tools or receipt or scene or images or library or recap
                or expect or notes_budget is not None or digest):
            raise ValueError("gen2 is the plain record actuator with note_gate and nothing else")
        text = with_gen2(RECORD_TEMPLATE)
        text = with_click_info(text) if click_info else text
        text = with_note_gate(with_toolkit_act(text, toolkit_act) if toolkit_act else text)
        text = with_pace(text) if pace else text
        return with_note_done_snapshot(text) if note_done_snapshot else text
    if note_gate:
        if (not record or observation_tools or receipt or scene or images or library or recap or expect
                or notes_budget is not None or digest):
            raise ValueError("note_gate goes with the plain record actuator (the v14 arm)")
        return with_note_gate(with_click_info(RECORD_TEMPLATE) if click_info else RECORD_TEMPLATE)
    if click_info:
        text = with_click_info(RECORD_TEMPLATE)
        return with_toolkit_act(text, toolkit_act, gen1=True) if toolkit_act else text
    if digest and not expect:
        raise ValueError("act_digest goes on the act_expect actuator")
    if expect or notes_budget is not None:
        # RSI batch 1 is only for v14 arms (pure record actuator); stacking with other hooks first needs thought about whose output is last
        if not record or observation_tools or receipt or scene or images or library or recap:
            raise ValueError("act_expect / notes_budget go with the plain record actuator (the v14 arm)")
        text = with_expect(RECORD_TEMPLATE) if expect else RECORD_TEMPLATE
        text = with_digest(text) if digest else text
        return with_notes_budget(text, notes_budget) if notes_budget is not None else text
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
          scene: bool = False, images: bool = False, library: bool = False, recap: bool = False, *,
          expect: bool = False, notes_budget: int | None = None, digest: bool = False,
          note_gate: bool = False, gen2: bool = False, note_done_snapshot: bool = False,
          click_info: bool = False, toolkit_act: tuple[str, ...] = (), pace: bool = False) -> Path:
    path.write_text(render(record, observation_tools, receipt, scene, images, library, recap,
                           expect=expect, notes_budget=notes_budget, digest=digest, note_gate=note_gate,
                           gen2=gen2, note_done_snapshot=note_done_snapshot, click_info=click_info,
                           toolkit_act=toolkit_act, pace=pace),
                    encoding="utf-8")
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
