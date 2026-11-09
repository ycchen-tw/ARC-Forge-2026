"""Kaggle entry point: codex solves the ARC-AGI-2 hidden test (or the public eval in Save & Run) and writes submission.json.

    <python> work/arc_code/arc2/kaggle_run.py \\
      --config work/arc_code/arc2/configs/kaggle-l4-seq-c150k.json \\
      --challenges PATH [--solutions PATH] [--limit N] \\
      --out DIR --submission-out /kaggle/working/submission.json --archive-out /kaggle/working/arc2-run.tar.gz \\
      --model-base-url http://127.0.0.1:PORT/v1 --model SERVED_NAME \\
      --dp-size 2 --concurrency N --deadline-unix TS [--filler-guard 7200 --filler-min-left 3600] \\
      --codex-dataset DIR --agent-python PATH --tokenizer PATH

Must run as root (the config is fence posix: one useradd-created uid per task, privileges dropped with runuser). In order:

  1. Validate arguments and config (concurrency comes only from --concurrency; the config's is null)
  2. Copy codex (0.154.0 + patches 0001–0007) from --codex-dataset to <out>/codexbin/bin/codex, cross-checked three
     ways: the pinned sha256, the dataset's own SHA256SUMS, ``--version``
  3. **Close /kaggle** (chmod 700; Kaggle's /kaggle/input is a read-only mount where chmod gives EROFS; closing the
     parent directory is the approach tested to work -- AGI-3 image 2026-09-17): verify as nobody that --challenges /
     --solutions are unreadable. So --out must be outside /kaggle (e.g. /tmp/arc2run); codex, tokenizer and harness all
     live under --out. On exit (exceptions included) /kaggle's permissions are restored
  4. Verify the agent python has numpy + Pillow (as nobody, with the PATH the agent will get; after closing /kaggle)
  5. run_arc2_seq.run (since 10-12: per task two independent solvers s0, s2 + one refiner, ``--solvers-first``: s0 > s2 >
     refiner, no stage barrier; each episode picks the currently least busy DP. Since 10-13 also filler re-refiners: they
     use only slots the main line cannot use; while the main line is still running they need --filler-guard left before
     the deadline, after it is done --filler-min-left; the submission is refine_exp and its vote
     refine_vote): before starting, write an all-placeholder
     submission, rewritten after every finished episode; once --deadline-unix arrives, start no new episodes, kill the
     running ones and harvest their answers, write the final one
  6. Pack <out>/run into one tar.gz at --archive-out (Kaggle output costs one request per file; small files are slow).
     Without --solutions (competition rerun: output not visible) nothing is packed -- a tar of 720 workspaces would only
     eat time after the deadline

Configuration errors raise before the first task starts (exit ≠ 0). Once running, individual task failures are only
recorded; the submission is still written, exit 0 (exceeding the config's max_failures cancels the whole run and the
remaining tasks get placeholders); the ``ARC2_KAGGLE FAILURES`` line is printed.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_arc2  # noqa: E402
import run_arc2_seq  # noqa: E402

#: codex/vendor/x86_64-unknown-linux-musl/bin/codex of threerabbits/arcforge-codex-0154-p8 (0.154.0 + 0001–0008).
#: p8 = p7 + 0008 server-reasoning-included (features.server_reasoning_included); 0001's content changed too (video
#: tokens now read from the mp4 header; file name unchanged). p7 (62ee15a0…) does not accept
#: features.server_reasoning_included (--strict-config rejects it).
CODEX_SHA256 = "3b4dc03403070f03c822ff776a980e9afb1e76b7a3d79cb2547ef6f14b68a882"
CODEX_PATCHES = ("0001-view-video-tool.patch", "0002-sglang-session-release.patch",
                 "0003-compaction-reuse-prefix.patch", "0004-external-compact-request.patch",
                 "0005-pre-compact-reminder.patch", "0006-compaction-drops-reminder.patch",
                 "0007-real-token-truncation.patch", "0008-server-reasoning-included.patch")
KAGGLE = Path("/kaggle")


def say(*parts) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] ARC2_KAGGLE", *parts, flush=True)


def sha256_of(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_codex(dataset: Path, prefix: Path) -> dict:
    """Copy the codex binary from the mounted dataset to prefix/bin/codex (the mount point is not executable / gets
    closed later).

    Three-way check: pin, the dataset's own SHA256SUMS, ``--version``. Just "some 0.154.0" also runs, but silently
    lacks session release (0002), the midway reminder (0005) and real token truncation (0007), so PATCHES is checked
    entry by entry too."""
    if not (dataset / "codex" / "SHA256SUMS").is_file():
        raise SystemExit(f"--codex-dataset {dataset} has no codex/SHA256SUMS (give the dataset root)")
    natives = sorted((dataset / "codex").glob("vendor/*/bin/codex"))
    if len(natives) != 1:
        raise SystemExit(f"expected one codex/vendor/*/bin/codex under {dataset}, found {natives}")
    target = prefix / "bin" / "codex"
    target.parent.mkdir(parents=True)
    shutil.copyfile(natives[0], target)
    target.chmod(0o755)
    for path in (prefix, prefix / "bin"):
        path.chmod(0o755)
    digest = sha256_of(target)
    if digest != CODEX_SHA256:
        raise SystemExit(f"codex is {digest}, kaggle_run pins {CODEX_SHA256}")
    recorded = (dataset / "codex" / "SHA256SUMS").read_text().split()[0]
    if recorded != digest:
        raise SystemExit(f"the dataset's SHA256SUMS says {recorded}, the file is {digest}")
    patches = (dataset / "codex" / "PATCHES").read_text()
    missing = [p for p in CODEX_PATCHES if p not in patches]
    if missing:
        raise SystemExit(f"codex PATCHES does not list {missing}")
    version = subprocess.run([str(target), "--version"], capture_output=True, text=True, timeout=120,
                             env={"PATH": "/usr/bin:/bin"}, check=True).stdout.strip()
    if "0.154.0" not in version:
        raise SystemExit(f"codex --version said {version!r}, not 0.154.0")
    return {"path": str(target), "sha256": digest, "version": version, "from": str(natives[0])}


def check_agent_python(agent_python: Path, as_nobody: bool) -> dict:
    """Both what the agent gets when it types ``python3`` (the first PATH entry is the agent python's prefix/bin) and
    the agent python the harness collects answers with must import numpy and PIL. Under the posix fence this is
    verified as nobody (only what is readable counts)."""
    prefix = agent_python.absolute().parents[1]
    path = f"{prefix}/bin:/usr/bin:/bin"
    probe = "import numpy, PIL, sys; print(sys.executable, sys.version.split()[0], numpy.__version__, PIL.__version__)"
    out = {}
    for name, argv in (("agent_python", [str(agent_python), "-I", "-c", probe]), ("python3_on_path", ["python3", "-I", "-c", probe])):
        cmd = ["env", "-i", f"PATH={path}", "HOME=/tmp", *argv]
        if as_nobody:
            cmd = ["runuser", "-u", "nobody", "--", *cmd]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd="/")
        if done.returncode != 0:
            raise SystemExit(f"{name} ({' '.join(argv[:1])}, PATH={path}) cannot import numpy+PIL: {done.stderr[-600:]}")
        out[name] = done.stdout.strip()
    return out


def close_kaggle(hidden: list[Path]) -> int:
    """chmod 700 /kaggle, then verify as nobody that every hidden path is unreadable. Returns the original mode (for
    restoring)."""
    before = KAGGLE.stat().st_mode & 0o7777
    KAGGLE.chmod(0o700)
    for path in hidden:
        readable = subprocess.run(["runuser", "-u", "nobody", "--", "test", "-r", str(path)],
                                  capture_output=True).returncode == 0
        if readable:
            KAGGLE.chmod(before)
            raise SystemExit(f"closing {KAGGLE} did not hide {path} from nobody")
    return before


def archive(run_dir: Path, out: Path) -> int:
    tmp = out.with_name(f".{out.name}.tmp")

    def regular(info: tarfile.TarInfo):
        # With network access codex clones the plugins repo into each task's .driver/.tmp/plugins (Runpod measured
        # 89 MB per task, 1 GB for a 20-task archive); unrelated to solving, so it is not packed, lest 240 tasks blow
        # up /kaggle/working.
        if "/.driver/.tmp/plugins" in f"/{info.name}":
            return None
        return info if (info.isfile() or info.isdir() or info.issym()) else None   # sockets are not packed

    with tarfile.open(tmp, "w:gz", compresslevel=1) as tar:
        tar.add(run_dir, arcname=run_dir.name, filter=regular)
    os.replace(tmp, out)
    return out.stat().st_size


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--challenges", type=Path, required=True, help="{task_id: {train, test:[{input}]}}")
    ap.add_argument("--solutions", type=Path, help="{task_id: [grid per test]}: score the run (Save & Run)")
    ap.add_argument("--limit", type=int, help="first N tasks only (rehearsal; never in the scored rerun)")
    ap.add_argument("--out", type=Path, required=True, help="new directory outside /kaggle")
    ap.add_argument("--submission-out", type=Path, required=True)
    ap.add_argument("--archive-out", type=Path, required=True, help="tar.gz of the run directory, written at the end")
    ap.add_argument("--model-base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dp-size", type=int, required=True)
    ap.add_argument("--concurrency", type=int, required=True)
    ap.add_argument("--deadline-unix", type=float, required=True,
                    help="no new task after this; running tasks are stopped and harvested")
    # Defaults from the 10-13 scheduling simulation (actual tokens of two b200 runs, 720 episodes; time per step ∝ 44.5 +
    # rows per DP): guarding 120 minutes, at all three throughputs (C96 1,200 / 1,300 / 1,400 tok/s) the main-line
    # episodes cut by the deadline equal those with no filler at all; guarding 60 minutes cuts more
    # Not given: 7200 / 3600 divided by the config's time_scale (G4 simulation; Kaggle has no such key = real time;
    # run_harness.sh also passes them explicitly)
    ap.add_argument("--filler-guard", type=float,
                    help="while main episodes remain, re-refiners (filler) start only with this many seconds left "
                         "(default 7200 / the config's time_scale)")
    ap.add_argument("--filler-min-left", type=float,
                    help="once the main line is done, re-refiners start only with this many seconds left "
                         "(default 3600 / the config's time_scale)")
    ap.add_argument("--codex-dataset", type=Path, required=True, help="root of the mounted codex dataset")
    ap.add_argument("--agent-python", type=Path, required=True, help="python with numpy and Pillow for the agent")
    ap.add_argument("--tokenizer", type=Path, required=True, help="the served model's tokenizer.json")
    args = ap.parse_args(argv)

    began = time.time()

    # The builder sends SIGTERM when the server dies. run_arc2_seq.run installs its "cancel, harvest" handler only once
    # running; before that the default action exits at once and finally does not run -- /kaggle would stay at 700.
    # Install one that raises first, so the finally below still restores it.
    def early_signal(signum, frame):
        raise SystemExit(f"signal {signum} before the run started")
    signal.signal(signal.SIGTERM, early_signal)
    signal.signal(signal.SIGINT, early_signal)
    cfg = run_arc2.load_config(args.config, concurrency=args.concurrency)
    k = run_arc2_seq.time_scale(cfg)
    # x39: the config may give filler_guard_seconds / filler_min_left_seconds (only short Save & Run test configs do, so
    # the re-refiner gets to run within a few tens of minutes; production configs have neither = 7200 / 3600)
    args.filler_guard = cfg.get("filler_guard_seconds", 7200.0) / k if args.filler_guard is None else args.filler_guard
    args.filler_min_left = (cfg.get("filler_min_left_seconds", 3600.0) / k if args.filler_min_left is None
                            else args.filler_min_left)
    posix = cfg["fence"] == "posix"
    if cfg["network_namespace"]:
        raise SystemExit("Kaggle has no user namespaces (unshare --user is EPERM): network_namespace must be false")
    if posix and os.geteuid() != 0:
        raise SystemExit("fence posix needs root")
    if args.deadline_unix <= began:
        raise SystemExit(f"--deadline-unix {args.deadline_unix:.0f} is not in the future (now {began:.0f})")
    for path in (args.challenges, args.tokenizer, args.agent_python, *([args.solutions] if args.solutions else [])):
        if not path.is_file():
            raise SystemExit(f"{path} is not a file")
    out = args.out.absolute()
    if posix and out.is_relative_to(KAGGLE):
        raise SystemExit(f"--out {out} is under {KAGGLE}, which is closed to the task users")
    if not args.submission_out.parent.is_dir() or not args.archive_out.parent.is_dir():
        raise SystemExit("the parent directories of --submission-out and --archive-out must exist")
    tasks, scored = run_arc2.load_challenges(args.challenges, args.solutions, args.limit)
    say(f"{len(tasks)} tasks from {args.challenges} ({'scored with ' + str(args.solutions) if scored else 'unscored'}), "
        f"concurrency {cfg['concurrency']}, dp {args.dp_size}, deadline in {(args.deadline_unix - began) / 60:.1f} min")
    del tasks                           # run_arc2_seq reads them again itself (same function); this only validates up front

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))   # per task: a listen socket, log, upstream + client connections
    say(f"RLIMIT_NOFILE {soft} -> {hard}")

    out.mkdir(parents=True)
    out.chmod(0o711)
    codex = stage_codex(args.codex_dataset, out / "codexbin")
    say("CODEX", json.dumps(codex))

    mode_before = None
    deny: list[str] = []
    if posix:
        hidden = [args.challenges.absolute(), *([args.solutions.absolute()] if args.solutions else [])]
        outside = [p for p in hidden if not p.is_relative_to(KAGGLE)]
        if outside:
            raise SystemExit(f"{outside} are outside {KAGGLE}: closing it would not hide them")
        mode_before = close_kaggle(hidden)
        deny = [str(KAGGLE / "input"), *map(str, hidden)]
        say(f"CLOSED {KAGGLE} (was {oct(mode_before)}): nobody cannot read {[str(p) for p in hidden]}")
    try:
        # Verify only after closing /kaggle: if the agent python is under /kaggle, this fails here rather than at the
        # first task's fence.
        pythons = check_agent_python(args.agent_python, as_nobody=posix)
        say("AGENT_PYTHON", json.dumps(pythons))
        seq_argv = ["--config", str(args.config), "--run-dir", str(out / "run"), "--challenges", str(args.challenges),
                    *(["--solutions", str(args.solutions)] if args.solutions else []),
                    *(["--limit", str(args.limit)] if args.limit is not None else []),
                    "--agent-python", str(args.agent_python), "--driver-bin", str(out / "codexbin"),
                    "--model-base-url", args.model_base_url, "--model", args.model,
                    "--tokenizer-json", str(args.tokenizer), "--deny", ":".join(deny),
                    "--concurrency", str(args.concurrency), "--dp-size", str(args.dp_size),
                    "--deadline-unix", str(args.deadline_unix), "--solvers-first",
                    "--arms", "ind,refine_exp,rerefine", "--filler-guard", str(args.filler_guard),
                    "--filler-min-left", str(args.filler_min_left),
                    "--submission-out", str(args.submission_out), "--submit-arm", "refine_vote",
                    # Save & Run (with --solutions) never reaches the production extra_solvers threshold in 80 minutes:
                    # use the Save & Run variant if there is one
                    *(["--extra-solvers-key", "extra_solvers_saverun"]
                      if args.solutions is not None and "extra_solvers_saverun" in cfg else [])]
        say("SEQ_ARGV", json.dumps(seq_argv))
        summary, failures, _ = run_arc2_seq.run(seq_argv, record={
            **{k: str(v) for k, v in vars(args).items()}, "codex": codex, "agent_pythons": pythons,
            "kaggle_closed": mode_before is not None})
    finally:
        if mode_before is not None:
            KAGGLE.chmod(mode_before)
            say(f"RESTORED {KAGGLE} to {oct(mode_before)}")
    if scored:
        size = archive(out / "run", args.archive_out)
        say(f"ARCHIVE {args.archive_out} {size / 2**20:.1f} MiB")
    else:
        say("ARCHIVE skipped: unscored competition rerun (its output is not visible)")
    if failures:
        say(f"FAILURES {len(failures)}: {[f['task'] for f in failures]} (see {out / 'run' / 'failures.jsonl'})")
    say(f"END answered {summary['answered_test_pairs']} of {summary['test_pairs']} test pairs, "
        f"episodes done {summary['episodes_done']} of {summary['episodes']}, "
        f"not started {len(summary['episodes_not_started'])}, fillers {summary.get('fillers')}, "
        f"deadline_hit {summary['deadline_hit']}, "
        f"close failures {summary['close_session_failures']}, score {summary.get('score_submission')}, "
        f"submission {args.submission_out}, {(time.time() - began) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
