"""The official gateway, started as is -- ``arc_agi.server.create_app()``, not a single line rewritten.

Before 2026-09-18 this was our own rewrite of the 15 routes (``gateway.py`` / ``server.py``).
The differential probe (``work/arc_code/probes/conformance_diff.py``) measured 12 divergences, four of
which could not be found by reading the source: the ``full_reset`` response field had the opposite value,
the close response leaked ``api_key``, play was still allowed after the levels were done, and for a game
not yet started with a fake guid the official server opens the game while we refused. One more was a
cookie check I invented, which the official server never performs, and it wrecked two games for six hours each.

**Running the official app makes all of these disappear by construction.** (The only addition: ``--step-timeout``'s
step_guard, wrapped around the outside of the wrapper without changing semantics.) Game semantics and scoring locally and in the Kaggle rerun
are therefore bit-identical: "a game can build its environment only once", "levels can be reset, the game
cannot", "a RESET with ``_action_count == 0`` is not sent to the engine" are all already part of the
official competition mode.

Per-game keys, illegal-action filtering, rate limiting -- things the official server lacks -- live in
``middleware.py``, not here. The whole value of this file is that it **adds nothing**.

    <python with arc_agi and flask> -m arcforge.arc_code.official \\
        --corpus <.../environment_files> --port 18801 \\
        --recordings-dir <run>/recordings --scorecard-out <run>/scorecard_official.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--recordings-dir", type=Path,
                   help="official per-action recordings (including full_reset, frame, available_actions)")
    p.add_argument("--scorecard-out", type=Path,
                   help="on close, write the whole official card here -- the only authoritative score")
    p.add_argument("--no-competition-mode", dest="competition_mode",
                   action="store_false")
    p.add_argument("--step-timeout", type=float,
                   help="interrupt a game's step / reset / load after this many seconds and disable that game (step_guard.py); unset if omitted")
    p.add_argument("--step-timeout-log", type=Path,
                   help="write timeout events here (jsonl); give together with --step-timeout")
    p.add_argument("--start-level", metavar="GAME:K",
                   help="single-level episode: GAME starts at level K (1-based), RESET only returns to the current level (start_level.py)")
    a = p.parse_args(argv)
    if (a.step_timeout is None) != (a.step_timeout_log is None):
        raise SystemExit("--step-timeout and --step-timeout-log must be given together")

    if a.corpus.name != "environment_files":
        raise SystemExit(f"corpus must be named environment_files: {a.corpus}")

    # Deferred imports: these two exist only in the harness venv, and the module itself must be importable without them.
    from arc_agi import Arcade, OperationMode
    from arc_agi.server import create_app

    if a.step_timeout is not None:
        # Official game semantics and scoring untouched; only wrap a timeout around LocalEnvironmentWrapper (untrusted community games can hang the whole gateway)
        from arcforge.arc_code import step_guard
        step_guard.install(a.step_timeout, a.step_timeout_log)

    if a.start_level is not None:
        # Before create_app: api.py reads ONLY_RESET_LEVELS at construction (gateway_jail.sh clears the environment, so it can only be set here)
        import os
        from arcforge.arc_code import start_level
        os.environ["ONLY_RESET_LEVELS"] = "true"
        start_level.install(*start_level.parse(a.start_level))

    if a.recordings_dir:
        a.recordings_dir.mkdir(parents=True, exist_ok=True)
    arcade = Arcade(
        operation_mode=OperationMode.OFFLINE,
        environments_dir=str(a.corpus),
        recordings_dir=str(a.recordings_dir) if a.recordings_dir else None,
    )

    def on_close(card) -> None:
        if a.scorecard_out is None:
            return
        # Same handling as the official api.py:246: do not expose api_key.
        card.api_key = None
        payload = card.model_dump(mode="json", exclude_none=True)
        a.scorecard_out.write_text(json.dumps(payload, indent=2))
        print(f"official scorecard -> {a.scorecard_out}", file=sys.stderr, flush=True)

    app, _api = create_app(
        arcade,
        competition_mode=a.competition_mode,
        save_all_recordings=a.recordings_dir is not None,
        include_frame_data=True,
        on_scorecard_close=on_close,
    )
    # werkzeug prints one line per request; the middleware already rate-limits, so a 3.5-million-line storm cannot recur.
    logging.getLogger("werkzeug").setLevel(logging.INFO)
    print(f"official gateway on {a.host}:{a.port} "
          f"competition_mode={a.competition_mode} "
          f"games={len(arcade.available_environments)} "
          f"step_timeout={a.step_timeout} start_level={a.start_level}", file=sys.stderr, flush=True)
    app.run(host=a.host, port=a.port, threaded=True, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
