"""Stop: when the model wants to end without having won, call it back to keep playing (same process, same turn).

codex has no loop cap: if ``CAP`` consecutive Stops pass with no new action between them, let it go, and the runner
continues with a new session as before.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

CAP = 3

inp = common.read_input()
ws = Path(inp["cwd"])
st = common.status(ws, common.history(ws))
state_path = common.codex_home() / "gen2_stop.json"
prev = json.loads(state_path.read_text()) if state_path.exists() else {"actions": -1, "blocks": 0}
if st["state"] == "WIN":
    common.log("Stop", decision="allow", why="won", **st)
    sys.exit(0)
blocks = prev["blocks"] + 1 if st["actions"] == prev["actions"] else 1
if blocks > CAP:
    state_path.write_text(json.dumps({"actions": st["actions"], "blocks": 0}))
    common.log("Stop", decision="allow", why=f"{CAP} stops in a row without an action", **st)
    sys.exit(0)
state_path.write_text(json.dumps({"actions": st["actions"], "blocks": blocks}))
#: Telling it to keep playing when the window is full (act refuses do) would be contradictory: tell it to wrap up
#: instead (where() already says the window is full and what to do)
reason = ("You stopped, but the game is not won. " + common.where(st)
          + "\nNobody will read a report or answer a question. "
          + ("Finish this window as above." if st["context_full"]
             else "Keep playing: decide your next action and take it."))
common.log("Stop", decision="block", blocks=blocks, stop_hook_active=inp.get("stop_hook_active"), **st)
print(json.dumps({"decision": "block", "reason": reason}))
