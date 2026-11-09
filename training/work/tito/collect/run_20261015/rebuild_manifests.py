#!/usr/bin/env python3
"""10-15 03:00Z shutdown: arc3 rounds opened before 10-15 00:25Z run the old run_arm (no SIGTERM wrap-up) and write no
tito_manifest.json after being stopped with SIGINT. This rebuilds them offline:

- Finished games: the digest files in ARM_STATE.plays (same as run_arm.tito_items).
- Games still running at shutdown (status=running in plays; sessions killed after SIGINT are added to sessions): re-extract
  a driver_digest.rebuilt.jsonl from that game's driver home (ws/.driver, keep_driver_home=true) with digest() from
  arms/codex.py; threads = the sessions' handles ∪ the thread_ids of usage rows in the digest; outcome.status=interrupted.
- server: ARM_STATE.tito.server, with server_root replaced by the replica holding the ledger (c0L's server_root is the
  sampling proxy that has already stopped), then read status once with preflight.check_server_end (the server must still be up).

    PYTHONPATH=src venv-fn-train/bin/python rebuild_manifests.py <run dir>...
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]  # repo root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "work/arc_code"))
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location("run_arm", ROOT / "work/arc_code/run_arm.py")
run_arm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_arm)
from arms import codex  # noqa: E402
from arcforge import paths  # noqa: E402
from arcforge.tito import manifest as tito_manifest, preflight as tito_preflight  # noqa: E402

C = paths.runs() / "tito/20261015-fp8-dflash-collect"
LEDGER_PORT = {C / "gate/ledger": 30811, C / "replicas/r1/ledger": 30812,
               C / "replicas/r2/ledger": 30813, C / "replicas/r3/ledger": 30814}


def rebuild(out: Path) -> dict:
    run_dir = out / "run"
    if (run_dir / tito_manifest.FILE).exists():
        raise SystemExit(f"{run_dir}: manifest already exists")
    state = json.loads((run_dir / "ARM_STATE.json").read_text())
    if not (state["status"] == "failed" and "KeyboardInterrupt" in str(state.get("error"))):
        raise SystemExit(f"{run_dir}: ARM_STATE status {state['status']!r} error {state.get('error')!r}, "
                         "expected the SIGINT KeyboardInterrupt")
    plays = {}
    for p in state["plays"]:
        plays[p["game"]] = p                        # several records for one game: take the last (continued play)
    if sorted(plays) != sorted(state["games"]):
        raise SystemExit(f"{run_dir}: plays {sorted(plays)} != games {sorted(state['games'])}")
    ledger_path = run_dir / "ledger.jsonl"
    items, rebuilt = [], []
    for game in state["games"]:
        play = plays[game]
        ws = run_dir / "games" / game
        if play["status"] == "running":
            home = ws / ".driver"
            if not home.is_dir():
                raise SystemExit(f"{home}: no driver home")
            rows = list(codex.digest(home))
            f = ws / "driver_digest.rebuilt.jsonl"
            f.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
            digests = sorted(ws.glob("driver_digest*.jsonl"))
            handles = [x["handle"] for x in play["sessions"] if x["handle"] is not None]
            threads = list(handles)
            for r in rows:
                if r["kind"] == "usage" and r.get("thread_id") and r["thread_id"] not in threads:
                    threads.append(r["thread_id"])
            dig = {r["thread_id"] for r in rows if r["kind"] == "usage"}
            if not set(handles) <= dig:
                raise SystemExit(f"{run_dir}/{game}: session handles {set(handles) - dig} not in the rebuilt digest")
            status = "interrupted"
            rebuilt.append(game)
        elif "digest" in play:
            digests = [ws / name for name in play["digest"]["files"]]
            threads = [s["handle"] for s in play["sessions"] if s["handle"] is not None]
            status = play["status"]
        elif play["status"] == "failed":
            digests, threads, status = [], [s["handle"] for s in play["sessions"] if s["handle"]], "failed"
        else:
            raise SystemExit(f"{run_dir}/{game}: play status {play['status']!r} without digest")
        won, actions, levels = run_arm._progress(ledger_path, game)
        items.append({"id": game, "threads": threads, "digests": digests,
                      "outcome": {"status": status, "levels_completed": levels, "won": won, "actions": actions,
                                  **({"error": play["error"]} if "error" in play else {})}})
    server = dict(state["tito"]["server"])
    port = LEDGER_PORT[Path(server["ledger_dir"])]
    server["server_root"] = f"http://127.0.0.1:{port}"
    server = tito_preflight.check_server_end(server)
    written = tito_manifest.write(run_dir / tito_manifest.FILE, harness="arc3", config=state["tito"]["config"],
                                  server=server, items=items)
    (out / "MANIFEST_REBUILT.txt").write_text(
        "tito_manifest.json rebuilt offline 2026-10-15 03:0xZ (rebuild_manifests.py) after the collection stop: "
        "old run_arm (launched before 00:25Z) has no SIGTERM ending, so it was stopped with SIGINT. "
        f"Games still running at the stop ({len(rebuilt)}: {', '.join(rebuilt)}) got outcome.status=interrupted, "
        "their response ids from driver_digest.rebuilt.jsonl (codex digest() over the kept driver home). "
        f"status_end read from replica :{port} directly.\n")
    with (out / "launcher.log").open("a") as f:
        f.write(f"ARM_DONE {out.name} (manifest rebuilt offline)\n")
    return {"run": out.name, "counts": written["counts"], "interrupted": len(rebuilt)}


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(rebuild(Path(d))), flush=True)
