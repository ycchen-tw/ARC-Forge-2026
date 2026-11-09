"""The only cell that runs anything. Everything above it only defines names.

**Save & Run and the scored rerun take the same code path.** The launcher is
always in upstream mode: in the rerun the upstream is the organiser's
`gateway:8001`; in Save & Run it is a stand-in -- the official `create_app()`,
competition mode on, over the public corpus oversampled to the rerun's size --
started here. The old arm's Save & Run took a corpus branch the rerun never
takes (journal 2026-09-18: "what needs verifying, Save & Run cannot verify");
this one rehearses the path that is scored.
"""

rerun = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").lower() in ("1", "true")
state["rerun"] = rerun
save()
stage("arm_files", unpack_arm)
if not rerun:
    # Kaggle validates the *selected version's* Save & Run output at submit time
    # and refuses a version that never wrote submission.parquet. In the rerun the
    # gateway writes the real one and a placeholder would overwrite the score.
    import pandas as pd
    placeholder = WORK / "submission.parquet"
    pd.DataFrame([{"row_id": "1_0", "game_id": "1", "end_of_game": True, "score": 1}],
                 columns=["end_of_game", "game_id", "row_id", "score"]).to_parquet(placeholder, index=False)
    state["submission"] = {"placeholder": str(placeholder), "bytes": placeholder.stat().st_size}
    save()
    say("SUBMISSION", json.dumps(state["submission"]))


def close_parent(corpus_src, restore=True):
    """Hide the corpus by closing the directory its read-only mount sits in.

    Asked as the user who would be reading -- root passes every mode check.
    Measured on this image 2026-09-17: every chmod under /kaggle/input is EROFS,
    Landlock is ABI 0, user namespaces are EPERM; closing /kaggle is what works.
    """
    parent = pathlib.Path("/kaggle")
    before = parent.stat().st_mode & 0o7777
    report = {"path": str(parent), "mode_before": oct(before)}
    try:
        parent.chmod(0o700)
    except OSError as error:
        return {**report, "closes": False, "outcome": f"{type(error).__name__}: {error.strerror}"}
    readable = subprocess.run(["runuser", "-u", "nobody", "--", "test", "-r", str(corpus_src)],
                              capture_output=True).returncode == 0
    report.update(outcome="chmod returned", readable_by_nobody=readable, closes=not readable)
    if restore:
        parent.chmod(before)
        report["mode_restored"] = oct(parent.stat().st_mode & 0o7777)
    return report


def oversample(corpus, target):
    """Distinct ids for the same public games, so Save & Run is rerun-sized.

    The copy's version directory gets a **prefix** tag (`c2<hash>`), never a
    suffix: the official gateway decides "this game already has an environment"
    with `key.startswith(game_id)` (arc_agi/scorecard.py has_environment), so a
    suffixed copy `ar25-0c556536x2` that starts first makes `ar25-0c556536`
    unstartable for the rest of the run -- measured on PC 2026-09-25.
    """
    seeds = sorted(v for g in sorted(corpus.iterdir()) if g.is_dir()
                   for v in sorted(g.iterdir()) if v.is_dir())
    made = []
    while len(seeds) + len(made) < target:
        seed = seeds[len(made) % len(seeds)]
        copy = seed.with_name(f"c{len(made) // len(seeds) + 2}{seed.name}")
        shutil.copytree(seed, copy)
        meta = copy / "metadata.json"
        spec = json.loads(meta.read_text())
        spec["game_id"] = f"{copy.parent.name}-{copy.name}"
        meta.write_text(json.dumps(spec, indent=1))
        made.append(spec["game_id"])
    ids = sorted(f"{g.name}-{v.name}" for g in corpus.iterdir() if g.is_dir() for v in g.iterdir() if v.is_dir())
    clash = [(a, b) for a in ids for b in ids if a != b and b.startswith(a)]
    if clash:
        raise RuntimeError(f"oversampled ids where one is a prefix of another: {clash[:3]}")
    return {"seeds": len(seeds), "added": len(made), "total": len(ids), "sample": made[:3]}


#: Named Save & Run sets (notebook key ``oversample_set``; ``oversample_to`` keeps the old round-robin). Each public
#: game is played ``per_game`` times, the ``extra`` games once more. "110v2" (user 2026-10-29): ``oversample_to`` 110
#: gave the fifth copy to the first ten games by name; here it goes to the ten hardest -- the ten lowest mean official
#: scores over the G4 9 h reruns v32g4, v33g4, v34g4, v36g4 and v37g4 (bp35 21, lf52 31, sk48 38, sp80 53, s5i5 59,
#: su15 66, wa30 71, dc22 75, g50t 85, tn36 89; next is sc25 91). Only Save & Run and the G4 replica play it: the
#: scored rerun plays the organiser's games.
OVERSAMPLE_SETS = {
    "110v2": {"games": 25, "per_game": 4, "total": 110,
              "extra": ("bp35", "lf52", "sk48", "sp80", "s5i5", "su15", "wa30", "dc22", "g50t", "tn36")},
}


def oversample_set(corpus, name):
    """Make the named set of OVERSAMPLE_SETS out of the public corpus, with oversample()'s prefix tags
    (``c2<hash>``, ``c3<hash>``, ...). Refuses a corpus that is not the set's games with one version each."""
    if name not in OVERSAMPLE_SETS:
        raise RuntimeError(f"unknown oversample set {name!r}; known: {sorted(OVERSAMPLE_SETS)}")
    spec = OVERSAMPLE_SETS[name]
    games = sorted(g for g in corpus.iterdir() if g.is_dir())
    missing = sorted(set(spec["extra"]) - {g.name for g in games})
    if len(games) != spec["games"] or missing:
        raise RuntimeError(f"oversample set {name}: corpus has {len(games)} games (want {spec['games']}), "
                           f"extra games missing: {missing}")
    made = []
    for game in games:
        versions = sorted(v for v in game.iterdir() if v.is_dir())
        if len(versions) != 1:
            raise RuntimeError(f"oversample set {name}: {game.name} has {len(versions)} versions, want 1")
        seed = versions[0]
        for k in range(2, spec["per_game"] + (game.name in spec["extra"]) + 1):
            copy = seed.with_name(f"c{k}{seed.name}")
            shutil.copytree(seed, copy)
            meta = copy / "metadata.json"
            data = json.loads(meta.read_text())
            data["game_id"] = f"{game.name}-{copy.name}"
            meta.write_text(json.dumps(data, indent=1))
            made.append(data["game_id"])
    ids = sorted(f"{g.name}-{v.name}" for g in corpus.iterdir() if g.is_dir() for v in g.iterdir() if v.is_dir())
    clash = [(a, b) for a in ids for b in ids if a != b and b.startswith(a)]
    if clash:
        raise RuntimeError(f"oversampled ids where one is a prefix of another: {clash[:3]}")
    if len(ids) != spec["total"]:
        raise RuntimeError(f"oversample set {name}: made {len(ids)} games, want {spec['total']}")
    return {"set": name, "seeds": len(games), "added": len(made), "total": len(ids), "sample": made[:3]}


def pack_games(games, out):
    """One ``<game>.tar.gz`` per workspace (its record/, notes, scripts, codex rollouts).

    Kaggle serves the output one file per request, about 0.3 s each, and the game
    record writes several small files per step: the 30-min record probe
    (2026-09-26) left tens of thousands and took an hour to download. gzip level 1:
    the PNG/MP4/NPZ are compressed already, the rollouts are text.
    """
    out.mkdir(parents=True, exist_ok=True)

    def skip(info):
        return None if not (info.isfile() or info.isdir() or info.issym()) else info

    def pack(ws):
        with tarfile.open(out / f"{ws.name}.tar.gz", "w:gz", compresslevel=1) as tar:
            tar.add(ws, arcname=ws.name, filter=skip)

    dirs = sorted(d for d in games.iterdir() if d.is_dir()) if games.is_dir() else []
    with ThreadPoolExecutor(8) as pool:
        list(pool.map(pack, dirs))
    return len(dirs)


def start_standin(harness_python, corpus, port):
    """The official gateway, unchanged, as Save & Run's stand-in for gateway:8001."""
    log = (WORK / "standin.log").open("w")
    proc = subprocess.Popen([harness_python, "-m", "arcforge.arc_code.official",
                             "--corpus", str(corpus), "--port", str(port), "--host", "127.0.0.1",
                             "--scorecard-out", str(WORK / "standin-scorecard.json")],
                            # The kernel's env carries MPLBACKEND=module://matplotlib_inline…,
                            # which matplotlib (imported by arc_agi) rejects outside Jupyter:
                            # v1 of this notebook died here, 6 minutes in.
                            env=dict(os.environ, PYTHONPATH=str(CODE / "src"), MPLBACKEND="Agg"),
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    begin = time.monotonic()
    while time.monotonic() - begin < 300:
        if proc.poll() is not None:
            raise RuntimeError("stand-in gateway exited: " + (WORK / "standin.log").read_text()[-1500:])
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/healthcheck", timeout=3):
                return proc
        except Exception:
            time.sleep(1)
    raise RuntimeError("stand-in gateway never answered /api/healthcheck")


# Where the agent's world lives: outside /kaggle, which the fence closes.
AGENT.mkdir(parents=True, exist_ok=True)
AGENT.chmod(0o711)
state["agent_root"] = str(AGENT)
tools = stage("tools", stage_tools)
model = stage("model", find_model)
prefetching = start_prefetch(model)
runtime = stage("runtime", unpack_runtime)
state["runtime_provenance"] = stage("provenance", lambda: assert_runtime_provenance(runtime))
state["runtime_overlay"] = stage(
    "overlay", lambda: apply_runtime_overlay(runtime, CFG["runtime_overlay"], CFG["codex_dataset"]))
if CFG.get("sglang_tree"):
    state["sglang_tree"] = stage("sglang_tree", lambda: install_sglang_tree(runtime))
if CFG.get("int3"):
    state["humming"] = stage("humming", install_humming)
wheels = mounted("arc_agi_3_wheels")
corpus_src = mounted("environment_files")
state["mounts"] = {"wheels": str(wheels), "corpus": str(corpus_src)}
save()
venvs = stage("venv", lambda: make_venvs(wheels))
# The corpus copy is the answer key: root only. The launcher reads it for the
# baselines (Save & Run's cross-check) and the stand-in serves it.
corpus = WORK / "environment_files"
if not rerun:
    if not corpus.exists():
        shutil.copytree(corpus_src, corpus)
    corpus.chmod(0o700)
    if CFG.get("oversample_to") and CFG.get("oversample_set"):
        raise RuntimeError("notebook config sets both oversample_to and oversample_set")
    if CFG.get("oversample_to"):
        state["oversample"] = oversample(corpus, CFG["oversample_to"])
        say("OVERSAMPLE", json.dumps(state["oversample"]))
    elif CFG.get("oversample_set"):
        state["oversample"] = oversample_set(corpus, CFG["oversample_set"])
        say("OVERSAMPLE", json.dumps(state["oversample"]))
state["prefetch"] = stage("prefetch", lambda: (PENDING.clear(), join_prefetch(prefetching))[1])
cuda = stage("cuda", lambda: cuda_report(runtime_interpreter(), runtime_env(runtime, os.environ.copy())))
state["rewarm"] = stage("rewarm", lambda: rewarm(model))
server, base, startup, env = stage(
    "server", lambda: start_server(runtime, model, os.environ.copy(), cuda["arch_list"]))[:4]
state["startup_s"] = startup
save()


def snapshot_metrics(name):
    """sglang's /metrics as text into WORK: two of them bracket the arm, so the generated tokens per second
    over the whole arm (prefill stalls included) come from the server's own counters, as on the G4."""
    with urllib.request.urlopen(base + "/metrics", timeout=30) as response:
        (WORK / name).write_bytes(response.read())
    return {"file": name, "at_s": round(time.monotonic() - T0, 1)}


state["metrics_start"] = snapshot_metrics("metrics-start.txt")


class ServerDeathReport(threading.Thread):
    """Report-only: no restart (v11 showed a restart does not finish on Kaggle). When sglang exits during the
    arm, print why -- the tail of its log and the container's memory -- because a Save & Run's live log is
    the only place that is readable while it runs, and the rest of the run is lost either way."""

    def __init__(self, proc):
        super().__init__(daemon=True, name="server-death-report")
        self.proc, self.halt, self.event = proc, threading.Event(), None

    def run(self):
        while not self.halt.wait(10):
            code = self.proc.poll()
            if code is None:
                continue
            self.event = {"exit": code, "at_s": round(time.monotonic() - T0), "memory": cgroup_now()}
            say("SERVER DIED", json.dumps(self.event))
            print((WORK / "server.log").read_text(errors="replace")[-4000:], flush=True)
            return

    def stop(self):
        self.halt.set()
        self.join(timeout=30)


def cgroup_memory() -> dict:
    """This container's memory as the kernel accounts it, in GiB."""
    rel = Path("/proc/self/cgroup").read_text().strip().splitlines()[-1].split(":", 2)[2]
    base = Path("/sys/fs/cgroup") / rel.lstrip("/")
    stat = dict(line.split() for line in (base / "memory.stat").read_text().splitlines())
    out = {k: round(int(stat[k]) / 2**30, 1) for k in ("anon", "shmem", "file", "file_dirty", "file_writeback")}
    out["current"] = round(int((base / "memory.current").read_text()) / 2**30, 1)
    out["max"] = (base / "memory.max").read_text().strip()
    return out


def copy_checkpoint(source: Path, dest: Path, workers: int, chunk: int = 64 << 20,
                    flush_every: int = 1 << 30, report_every: int = 10 << 30) -> dict:
    """The checkpoint on the local disk, for a server restart to load from.

    A restart during the arm cannot load from /kaggle/input: the v11 probe v4
    (2026-09-27) read the checkpoint twice at 9.5 and 27 GB/s right before the
    restart, and the loader still took 14 minutes over its 6th shard of 20 --
    with the arm's own memory in the cgroup the 106 GB no longer fits beside the
    server's host memory, so the mmap loader refaults it off NFS. A refault off
    the local disk is not an NFS round trip.

    v5 copied with shutil on 16 threads and the session died without output a few
    minutes in (the stand-in's 1 s start took 194 s beside it). So now: few
    threads, and every GiB written is synced and dropped from the cache, source
    and copy alike, so dirty pages stay bounded; every 10 GiB prints the cgroup's
    memory and the disk, so the next death says whether it was memory or disk.
    """
    begin = time.monotonic()
    files = sorted(p for p in source.rglob("*") if p.is_file())
    total = sum(p.stat().st_size for p in files)
    free = shutil.disk_usage(dest.parent).free
    if free < total + (20 << 30):
        raise RuntimeError(f"{dest.parent} has {free / 2**30:.0f} GiB free for a {total / 2**30:.0f} GiB copy")
    dest.mkdir(mode=0o700)
    lock, done = threading.Lock(), {"bytes": 0, "reported": 0}

    def one(path):
        target = dest / path.relative_to(source)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        src = os.open(path, os.O_RDONLY)
        dst = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            offset = flushed = 0
            while True:
                block = os.read(src, chunk)
                if not block:
                    break
                view = memoryview(block)
                while view:
                    view = view[os.write(dst, view):]
                offset += len(block)
                if offset - flushed >= flush_every:
                    os.fdatasync(dst)
                    os.posix_fadvise(dst, flushed, offset - flushed, os.POSIX_FADV_DONTNEED)
                    os.posix_fadvise(src, flushed, offset - flushed, os.POSIX_FADV_DONTNEED)
                    flushed = offset
                with lock:
                    done["bytes"] += len(block)
                    if done["bytes"] - done["reported"] >= report_every:
                        done["reported"] = done["bytes"]
                        say("MODEL COPY", f"{done['bytes'] / 2**30:.0f}/{total / 2**30:.0f} GiB in "
                            f"{time.monotonic() - begin:.0f} s; memory", json.dumps(cgroup_memory()),
                            f"disk free {shutil.disk_usage(dest).free / 2**30:.0f} GiB")
            os.fsync(dst)
            os.posix_fadvise(dst, 0, 0, os.POSIX_FADV_DONTNEED)
            os.posix_fadvise(src, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(src)
            os.close(dst)
        if offset != path.stat().st_size:
            raise RuntimeError(f"short copy: {target} {offset} of {path.stat().st_size}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(one, p) for p in files]):
            future.result()
    seconds = time.monotonic() - begin
    return {"dest": str(dest), "files": len(files), "bytes": total, "seconds": round(seconds, 1),
            "GB_per_second": round(total / seconds / 1e9, 3), "memory_after": cgroup_memory()}


local_copy = {}


def run_copy():
    try:
        say("MODEL COPY", "memory before", json.dumps(cgroup_memory()))
        local_copy["report"] = copy_checkpoint(Path(model), Path(CFG["restart_model_copy"]),
                                               CFG.get("model_copy_workers", 2))
        say("MODEL COPY", json.dumps(local_copy["report"]))
    except BaseException as error:  # noqa: BLE001 -- printed now; a restart raises it again
        local_copy["error"] = f"{type(error).__name__}: {error}"[:600]
        say("MODEL COPY FAILED", local_copy["error"])


copying = None
if CFG.get("restart_model_copy"):
    copying = threading.Thread(target=run_copy, daemon=True, name="model-copy")
    copying.start()
    say("MODEL COPY", "started to", CFG["restart_model_copy"])

arm_config = json.loads((UNIT / "configs" / (CFG["rerun_arm_config"] if rerun
                                             else CFG["saverun_arm_config"])).read_text())
served_context = (state.get("server_info") or {}).get("context_length")
if served_context != CFG["context_length"]:
    raise RuntimeError(f"served context {served_context} != notebook config {CFG['context_length']}")
if not arm_config["auto_compact_token_limit"] < arm_config["model_context_window"] <= served_context:
    raise RuntimeError(f"need compact < codex window <= served context: "
                       f"{arm_config['auto_compact_token_limit']} / {arm_config['model_context_window']} / {served_context}")
if CFG.get("codex_tokenizer_sha256"):
    # codex 0007: tool output is cut at tool_output_token_limit *real* tokens of the served
    # model. The game uid cannot reach /kaggle once the parent is closed, so the agent tree
    # (AGENT is 0711) gets a pinned copy, handed to codex through driver_files.
    tokenizer = AGENT / "tokenizer.json"
    shutil.copyfile(pathlib.Path(model) / "tokenizer.json", tokenizer)
    tokenizer.chmod(0o644)
    digest = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
    if digest != CFG["codex_tokenizer_sha256"]:
        raise RuntimeError(f"tokenizer.json sha256 {digest} != notebook config {CFG['codex_tokenizer_sha256']}")
    if "tool_output_tokenizer" in (arm_config.get("driver_files") or {}):
        raise RuntimeError("arm config already sets driver_files.tool_output_tokenizer")
    arm_config.setdefault("driver_files", {})["tool_output_tokenizer"] = str(tokenizer)
    say("TOKENIZER", str(tokenizer), digest[:16])

standin = None
if rerun:
    upstream = CFG["gateway_url"]
    # The same resolution every scored arc-code submission used: the organiser's
    # key if the environment carries one, else the SDK's documented default.
    # The official server does not authenticate; the key only has to be consistent.
    upstream_key = os.environ.get("ARC_API_KEY") or "test-key-123"
    state["upstream_key_source"] = "ARC_API_KEY" if os.environ.get("ARC_API_KEY") else "default test-key-123"
else:
    standin = stage("standin", lambda: start_standin(venvs["harness"], corpus, CFG["standin_port"]))
    upstream = f"http://127.0.0.1:{CFG['standin_port']}"
    upstream_key = "test-key-123"
key_file = WORK / "upstream_key"
key_file.write_text(upstream_key + "\n")
key_file.chmod(0o600)
state["upstream"] = upstream
save()

# Kaggle stops the session nine hours after it started -- before this cell began
# -- and a killed session commits nothing. Five minutes at each end.
# kaggle_session_seconds overrides the nine hours: a timeout test only (does the scored rerun really stop at 9 h?).
KAGGLE_SESSION = CFG.get("kaggle_session_seconds", 32400)
FRONT = 300
BACK = 300
budget = int(KAGGLE_SESSION - FRONT - BACK - (time.monotonic() - T0))
wall = min(arm_config["wall_seconds"], budget)
state["budget"] = {"spent_s": round(time.monotonic() - T0, 1), "playable_s": budget,
                   "configured_wall_s": arm_config["wall_seconds"], "wall_s": wall}
say("BUDGET", json.dumps(state["budget"]))
if wall < 1200:
    raise RuntimeError("less than 20 minutes of playable wall left: " + json.dumps(state["budget"]))
arm_config_path = WORK / "arm-config.json"
arm_config_path.write_text(json.dumps(arm_config, indent=1))
state["arm_config"] = arm_config
save()

# The empty sitecustomize the fenced sessions import instead of the image's.
pyshadow = AGENT / "pyshadow"
pyshadow.mkdir(exist_ok=True)
(pyshadow / "sitecustomize.py").write_text("# Deliberately empty: shadows the host's sitecustomize.\n")
pyshadow.chmod(0o755)
(pyshadow / "sitecustomize.py").chmod(0o644)


def start_model_proxy(spec, image_key, png_size):
    """work/arc_code/sampling_proxy.py between codex and sglang (config model_proxy; arm act_image).

    act_image / act_library end a command's output with signed [[arc-image:...]] markers; the proxy swaps them
    for the PNGs of the game record (act_images.py) before the request reaches sglang, the same bytes every
    time, so the QSA session prefix still resumes. It writes config model_proxy.sampling into every request:
    v19 wrote what the server uses anyway when codex sends none (ResponsesRequest 0.7 / 1.0, top_k unset -> -1
    under --sampling-defaults openai), v30 writes 1.0 / 0.95 / top_k 20 with no act_image or reviewer at all.
    Every other path (/close_session, /metrics, /health, /get_server_info) is relayed byte for byte.
    """
    sets = [a for k, v in sorted(spec["sampling"].items()) for a in ("--set", f"{k}={json.dumps(v)}")]
    argv = [venvs["harness"], "-u", str(UNIT / "sampling_proxy.py"), "--listen", str(spec["port"]),
            "--upstream", base, *sets, "--manifest", str(WORK / "sampling_proxy.json"),
            "--usage-log", str(WORK / "proxy-usage.jsonl")]
    if image_key is not None:
        argv += ["--image-ref-key", str(image_key), "--image-run-dir", str(RUN), "--image-png-size", str(png_size)]
    if arm_config.get("reviewer"):
        # the read-only reviewer (review_scheduler.py): the report lands in review/Rnn.md and the proxy appends a
        # one-line notice to the player's next tool output; run_arm refuses to start without this proxy's marker
        argv += ["--review-run-dir", str(RUN)]
    log = (WORK / "sampling_proxy.log").open("a")
    proc = subprocess.Popen(argv, env=dict(os.environ, PYTHONPATH=str(CODE / "src"), MPLBACKEND="Agg"),
                            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    url = f"http://127.0.0.1:{spec['port']}"
    begin = time.monotonic()
    while time.monotonic() - begin < 120:
        if proc.poll() is not None:
            raise RuntimeError("model proxy exited: " + (WORK / "sampling_proxy.log").read_text()[-1500:])
        try:
            # relayed to sglang's /health: a 200 here is the proxy and the server both answering
            with urllib.request.urlopen(url + "/health", timeout=5) as response:
                if response.status == 200:
                    return proc, url
        except Exception:
            time.sleep(1)
    raise RuntimeError("model proxy never relayed /health")


class ProxyKeeper(threading.Thread):
    """Every model request of every game goes through the proxy: dead, it is all 110 games stalled until the
    wall ends. It holds nothing a restart loses but its counters and its image cache, so start it again with
    the same arguments -- loudly, at most max_restarts times."""

    def __init__(self, proc, restart, max_restarts):
        super().__init__(daemon=True, name="proxy-keeper")
        self.proc, self.restart, self.max_restarts = proc, restart, max_restarts
        self.halt, self.events = threading.Event(), []

    def run(self):
        while not self.halt.wait(5):
            code = self.proc.poll()
            if code is None:
                continue
            event = {"exit": code, "at_s": round(time.monotonic() - T0)}
            self.events.append(event)
            say("MODEL PROXY DIED", json.dumps(event))
            print((WORK / "sampling_proxy.log").read_text(errors="replace")[-3000:], flush=True)
            if len(self.events) > self.max_restarts:
                say("MODEL PROXY", f"no restarts left ({self.max_restarts}); the games stall from here")
                return
            try:
                self.proc = self.restart()[0]
                event["restarted"] = True
                say("MODEL PROXY", "restarted, pid", self.proc.pid)
            except BaseException as error:  # noqa: BLE001 -- printed; the next poll sees the dead process again
                event["restart_error"] = f"{type(error).__name__}: {error}"[:300]
                say("MODEL PROXY RESTART FAILED", event["restart_error"])

    def stop(self):
        self.halt.set()
        self.join(timeout=30)


class WeightSwitch(threading.Thread):
    """Config ``weight_switch`` (hs110 v39, user 10-31): the games start on the served package and, ``after_s`` into
    the arm, the same server takes an RL version in place -- pause_generation(in_place) -> update_weights_from_disk
    (keep_identity, no cache flush) -> continue_generation; the paused requests go on with the new weights over their
    cached KV (G4 10-30: 24 streams through an 11.5 s update, the int3 runtime printing ARC_INT3_DELTA_LOADED).

    The scored rerun never stops for it. A check that fails before the pause leaves the base serving. A load that fails
    after it started writing poisons the server (every request aborted, generate refused until an update succeeds) and
    is tried once more: a merged export rewrites every tensor the failed load could have touched. Every step prints a
    WEIGHT SWITCH line; ``failed`` makes Save & Run and the G4 replica end in an error once the evidence is saved."""

    def __init__(self, spec, export, after_s, server_url, server_log):
        super().__init__(daemon=True, name="weight-switch")
        self.spec, self.export, self.after_s = spec, export, after_s
        self.server_url, self.server_log = server_url, server_log
        self.halt, self.began, self.failed = threading.Event(), None, None
        self.report = {"after_s": after_s, "version": spec["version"], "dir": export["dir"], "events": []}

    def event(self, name, **fields):
        entry = {"event": name, "at_s": round(time.monotonic() - self.began, 1), **fields}
        self.report["events"].append(entry)
        say("WEIGHT SWITCH", json.dumps(entry)[:1500])

    def call(self, path, body=None, timeout=60):
        request = urllib.request.Request(self.server_url + path, method="GET" if body is None else "POST",
                                         data=None if body is None else json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")

    def info(self):
        served = self.call("/model_info")
        return {k: served.get(k) for k in ("weight_version", "weights_source", "poisoned", "weights_poisoned")}

    def read_export(self):
        """The pinned bytes, read whole. Right before the pause this also puts them in the page cache, so the update
        reads memory instead of the mount while every game waits."""
        began = time.monotonic()
        got = sha256_of(pathlib.Path(self.export["dir"]) / "model.safetensors")
        if got != self.spec["sha256"]:
            raise RuntimeError(f"model.safetensors is {got}, the config pins {self.spec['sha256']}")
        return round(time.monotonic() - began, 1)

    def load(self):
        out, began = {}, time.monotonic()
        out["pause"] = self.call("/pause_generation", {"mode": "in_place"})
        out["pause_s"] = round(time.monotonic() - began, 2)
        try:
            out["update"] = self.call("/update_weights_from_disk", {
                "model_path": self.export["dir"], "keep_identity": True, "flush_cache": False,
                "weight_version": self.spec["version"]}, timeout=1800)
        finally:
            out["update_s"] = round(time.monotonic() - began - out["pause_s"], 2)
            for attempt in range(1, 6):                                 # never leave the games paused
                try:
                    out["continue"] = self.call("/continue_generation", {})
                    break
                except BaseException as error:  # noqa: BLE001 -- retried; the fifth failure propagates
                    if attempt == 5:
                        raise
                    say("WEIGHT SWITCH", f"continue_generation failed ({type(error).__name__}: {error}); again in 10 s")
                    time.sleep(10)
            out["total_s"] = round(time.monotonic() - began, 2)
        return out

    def run(self):
        self.began = time.monotonic()
        try:
            self.event("armed", in_s=self.after_s, read_s=self.read_export())
            if self.halt.wait(max(0.0, self.began + self.after_s - time.monotonic())):
                self.event("not reached", why="the arm ended first")
                return
            before = self.info()
            if before["weight_version"] != "default" or before["poisoned"]:
                raise RuntimeError(f"the server is not on the base it switches from: {before}")
            self.event("start", read_s=self.read_export(), before=before)
            delta_line = f"ARC_INT3_DELTA_LOADED tensors={self.export['tensors']}"
            seen = self.server_log.read_text(errors="replace").count(delta_line)
            out = self.load()
            if not out["update"].get("success") and out["update"].get("weights_poisoned"):
                self.event("poisoned, loading again", first=out)
                out = self.load()
            after = self.info()
            loaded = self.server_log.read_text(errors="replace").count(delta_line) - seen
            ok = (out["update"].get("success") is True and not after["poisoned"]
                  and after["weight_version"] == self.spec["version"] and loaded >= 1)
            self.event("done" if ok else "FAILED", **out, after=after, delta_lines=loaded)
            if not ok:
                self.failed = f"update {out['update']}, then {after}, {loaded} '{delta_line}' line(s)"
        except BaseException as error:  # noqa: BLE001 -- printed and kept; the games go on with what is served
            self.failed = f"{type(error).__name__}: {error}"[:600]
            self.event("FAILED", error=self.failed)

    def stop(self):
        self.halt.set()
        self.join(timeout=60)


model_base = base
image_key = proxy = proxy_keeper = None
if arm_config.get("reviewer") and not CFG.get("model_proxy"):
    raise RuntimeError("the arm config has a reviewer: its notices reach the players through model_proxy")
png_size = None
if CFG.get("model_proxy"):
    if not (arm_config.get("act_image") or arm_config.get("reviewer")) and \
            CFG["model_proxy"]["sampling"] == {"temperature": 0.7, "top_p": 1.0, "top_k": -1}:
        raise RuntimeError("model_proxy without act_image or a reviewer is only for sampling other than the server's own")
    if arm_config.get("act_image"):
        # middleware signs the image refs, the proxy checks them; root only (/kaggle is closed during the arm)
        image_key = WORK / "image_ref.key"
        image_key.write_text(os.urandom(32).hex() + "\n")
        image_key.chmod(0o600)
        png_size = arm_config["record"]["png_size"]
    proxy, model_base = stage("model_proxy", lambda: start_model_proxy(CFG["model_proxy"], image_key, png_size))
    state["model_proxy"] = {"url": model_base, "pid": proxy.pid, "png_size": png_size,
                            "sampling": CFG["model_proxy"]["sampling"]}
    say("MODEL PROXY", json.dumps(state["model_proxy"]))
    save()
elif arm_config.get("act_image"):
    raise RuntimeError("the arm config turns act_image on: its markers need model_proxy to become images")

argv = [venvs["harness"], "-u", str(UNIT / "run_arm.py"),
        "--config", str(arm_config_path), "--run-dir", str(RUN),
        "--upstream", upstream, "--upstream-key-file", str(key_file),
        "--prep", tools["prep"], "--python", venvs["harness"], "--agent-python", venvs["agent"],
        "--driver-bin", tools["prefix"], "--driver", "codex",
        "--model-base-url", model_base + "/v1", "--model", CFG["served_model"],
        "--wall-seconds", str(wall),
        "--fence-deny", ":".join(str(p) for p in (corpus, corpus_src.parent)),
        "--pyshadow", str(pyshadow), "--tag", CFG.get("kernel_slug", ""),
        # the heartbeat counts QSA session decisions, thinking-budget closes and host KV from it
        "--server-log", str(WORK / "server.log")]
if image_key is not None:
    argv += ["--image-ref-key", str(image_key)]
if not rerun:
    # Save & Run: the launcher cross-checks the stand-in's card against its own
    # ledger RHAE, which needs the baselines.
    argv += ["--corpus", str(corpus)]
# No runtime PYTHONPATH: run_arm is stdlib + arcforge, and the sglang tree on
# sys.path could shadow it.
arm_env = dict(os.environ, PYTHONPATH="", MPLBACKEND="Agg")
weight_switch = None
if CFG.get("weight_switch"):
    # keyed by the arm config this run plays, so Save & Run switches inside its short wall and the G4 replica (which
    # plays the rerun config on the Save & Run path) at the rerun's time
    switch_after = CFG["weight_switch"]["after_s"][CFG["rerun_arm_config"] if rerun else CFG["saverun_arm_config"]]
    if switch_after >= wall:
        raise RuntimeError(f"weight switch at {switch_after} s is past the arm's wall of {wall} s")
    weight_switch = WeightSwitch(CFG["weight_switch"], state["weight_switch_export"], switch_after, base,
                                 WORK / "server.log")
    say("WEIGHT SWITCH", json.dumps({"to": CFG["weight_switch"]["version"], "after_s": switch_after,
                                     "dir": state["weight_switch_export"]["dir"]}))

# The last thing before the arm: from here no unprivileged uid can traverse into
# /kaggle at all. fence_posix.sh checks that, per game, as the game's own user.
closing = close_parent(corpus_src, restore=False)
state["closing"] = closing
say("CLOSED", json.dumps(closing))
if not closing["closes"]:
    raise RuntimeError("closing /kaggle did not hide the corpus: " + json.dumps(closing))
save()
#: launcher lines worth showing live in the Kaggle log. Everything goes to arm.log;
#: these also go to the notebook's stdout, which is the only thing readable while
#: the session runs (the old arm's log was silent for the whole nine hours).
LIVE = ("HEARTBEAT", "WATCHDOG", "schedule:", "scorecard", "posix fence", "arcgame:", "prompts:", "WIN", "FENCE", "OOM", "CLOSE",
        "official scorecard", "拋出", "放棄", "unit result", "Traceback", "Error", "===", "mean ",
        "MODEL ERROR", "model server", "FINAL CALL", "REVIEW")


class ServerWatchdog(threading.Thread):
    """sglang dead -> start it again with the same arguments and the same log checks.

    The scheduler turns any exception (a CUDA OOM in the vision encoder is the one we
    know of: the ViT runs a whole video at once, outside the static pool) into
    SIGQUIT for the whole server, and nothing restarted it: with the model gone the
    agents failed until their games were retired. codex waits out a refused
    connection on its own (unbounded_connection_retries), so a server that comes
    back within a few minutes costs a pause, not the run. The dead server's log is
    kept as server.<n>.log and its tail printed -- in a Save & Run it is the cause.
    """

    def __init__(self, proc, restart, max_restarts, kill_after=None):
        super().__init__(daemon=True, name="server-watchdog")
        self.proc, self.restart, self.max_restarts = proc, restart, max_restarts
        self.kill_after = kill_after          # Save & Run test: kill the server once, this many s in
        self.smi = shutil.which("nvidia-smi")
        if self.smi is None:
            raise RuntimeError("the server watchdog needs nvidia-smi to see the GPU freed before a restart")
        self.events = []
        self.halt = threading.Event()

    def run(self):
        began = time.monotonic()
        up_since = began
        while not self.halt.wait(5):
            if self.kill_after is not None and time.monotonic() - began >= self.kill_after:
                say("SERVER TEST KILL", f"killing sglang {self.kill_after} s into the arm, as configured")
                os.killpg(self.proc.pid, 9)
                self.kill_after = None
            code = self.proc.poll()
            if code is None:
                continue
            number = len(self.events) + 1
            tail = (WORK / "server.log").read_text(errors="replace")[-2500:]
            (WORK / "server.log").rename(WORK / f"server.{number}.log")
            event = {"n": number, "exit": code, "uptime_s": round(time.monotonic() - up_since),
                     "log": f"server.{number}.log"}
            say("SERVER DIED", json.dumps(event))
            print(tail, flush=True)
            if number > self.max_restarts:
                event["restarted"] = False
                self.events.append(event)
                say("SERVER NOT RESTARTED", f"{self.max_restarts} restarts used")
                return
            try:
                os.killpg(self.proc.pid, 9)
            except ProcessLookupError:
                pass
            say("SERVER RESTART", f"attempt {number} of {self.max_restarts}")
            try:
                event["gpu_used_mib"] = self.wait_gpu_free()
                proc, _, seconds, _ = self.restart()
            except BaseException as error:  # noqa: BLE001 -- recorded and printed; the arm sees a dead server
                event["restart_error"] = f"{type(error).__name__}: {error}"[:600]
                self.events.append(event)
                say("SERVER RESTART FAILED", event["restart_error"])
                return
            self.proc, up_since = proc, time.monotonic()
            event["restart_s"] = seconds
            self.events.append(event)
            say("SERVER RESTARTED", json.dumps(event))

    def wait_gpu_free(self, limit_mib=2048, timeout=180):
        """The dead scheduler's memory goes back when its processes are gone; loading
        the model before that is an OOM at load time."""
        deadline = time.monotonic() + timeout
        while True:
            out = subprocess.run([self.smi, "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, check=True).stdout
            used = int(out.split()[0])
            if used < limit_mib:
                return used
            if time.monotonic() > deadline:
                raise RuntimeError(f"GPU still has {used} MiB in use {timeout} s after the server died")
            time.sleep(3)

    def stop(self):
        self.halt.set()
        self.join(timeout=30)


def run_live(argv, timeout, env, log):
    """run(), but the launcher's stdout is read as it is written.

    The timeout is a timer of its own, not a check between lines: a launcher that
    hangs *silently* never yields another line, and a check in the read loop would
    then never run -- the session would sit there until Kaggle's own limit and
    commit nothing. On expiry the launcher's whole process group is killed.
    """
    begin = time.monotonic()
    expired = threading.Event()
    with (WORK / log).open("w") as sink:
        proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, start_new_session=True)

        def expire():
            expired.set()
            print(f"[{time.strftime('%H:%M:%S')}] ARM TIMEOUT after {timeout} s: killing the launcher", flush=True)
            os.killpg(proc.pid, 9)

        timer = threading.Timer(timeout, expire)
        timer.daemon = True
        timer.start()
        try:
            for line in proc.stdout:
                sink.write(line)
                sink.flush()
                if any(key in line for key in LIVE):
                    print(line.rstrip()[:600], flush=True)
            code = proc.wait()
        finally:
            timer.cancel()
    if expired.is_set():
        raise RuntimeError(f"arm exceeded its {timeout} s timeout")
    if code:
        raise RuntimeError(f"arm exited {code}: " + (WORK / log).read_text(errors="replace")[-3000:])
    return {"seconds": round(time.monotonic() - begin, 1)}


def restart_server():
    """What the first start did, in the same order: the checkpoint read twice, then
    sglang -- but from the local copy (copy_checkpoint), which it waits for.

    v2 of the v11 probe restarted straight off /kaggle/input and sat at shard 0/20
    for 20 minutes; v4 read it twice first and still crawled (see copy_checkpoint).
    The first read here is the local disk's speed, measured, not assumed.
    """
    if copying is None:
        raise RuntimeError("no restart_model_copy configured: a restart off /kaggle/input "
                           "does not finish (v11 probe v2/v4)")
    say("SERVER RESTART", "waiting for the model copy")
    copying.join()
    if "error" in local_copy:
        raise RuntimeError("model copy failed: " + local_copy["error"])
    local = local_copy["report"]["dest"]
    for label in ("disk", "warm"):
        began = time.monotonic()
        manifest = prefetch_checkpoint(Path(local), CFG.get("prefetch_workers", 16))
        say("SERVER RESTART", f"checkpoint read ({label}) in {time.monotonic() - began:.0f} s,",
            f"{manifest['GB_per_second']:.2f} GB/s")
    say("SERVER RESTART", "starting sglang on", local)
    return start_server(runtime, local, os.environ.copy(), cuda["arch_list"])


# The watchdog is a probe, not part of a submission (2026-09-27): its restart
# does not finish on Kaggle -- off /kaggle/input the loader crawls, a local copy
# trips the session's disk limit -- and the gateway closes an idle scorecard after
# 15 minutes anyway. Without "server_watchdog" the arm runs as v10 did.
watchdog = None
if CFG.get("server_watchdog"):
    watchdog = ServerWatchdog(
        server, restart_server,
        max_restarts=CFG["server_max_restarts"],
        # The test kill is for Save & Run only; a scored rerun never kills its own server.
        kill_after=None if rerun else CFG.get("test_kill_server_after_s"))
    watchdog.start()
elif CFG.get("test_kill_server_after_s") or CFG.get("restart_model_copy") or "server_max_restarts" in CFG:
    raise RuntimeError("test kill / model copy / server_max_restarts configured without server_watchdog")
death_report = None
if watchdog is None:
    death_report = ServerDeathReport(server)
    death_report.start()
if proxy is not None:
    proxy_keeper = ProxyKeeper(proxy, lambda: start_model_proxy(CFG["model_proxy"], image_key, png_size),
                               CFG["model_proxy"]["max_restarts"])
    proxy_keeper.start()
if weight_switch is not None:
    weight_switch.start()
try:
    stage("arm", lambda: run_live(argv, wall + BACK, arm_env, "arm.log"))
finally:
    if weight_switch is not None:
        weight_switch.stop()
        state["weight_switch"] = {**weight_switch.report, "failed": weight_switch.failed}
    if watchdog is not None:
        watchdog.stop()
        server = watchdog.proc
        state["server_restarts"] = watchdog.events
    if death_report is not None:
        death_report.stop()
        state["server_death"] = death_report.event
    if proxy_keeper is not None:
        proxy_keeper.stop()
        proxy = proxy_keeper.proc
        state["model_proxy"]["deaths"] = proxy_keeper.events
    if server.poll() is None:
        try:
            state["metrics_end"] = snapshot_metrics("metrics-end.txt")
        except BaseException as error:  # noqa: BLE001 -- teardown: recorded and printed; the evidence below matters more
            state["metrics_end"] = {"error": f"{type(error).__name__}: {error}"[:300]}
            say("METRICS END FAILED", state["metrics_end"]["error"])
    state["memory_at_end"] = cgroup_now()
    if copying is not None:
        copying.join()
    state["model_copy"] = local_copy
    for proc in (server, standin, proxy):
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, 15)
    # /kaggle is reopened for the teardown: the evidence goes back into WORK.
    pathlib.Path("/kaggle").chmod(int(closing["mode_before"], 8))
    arm_state = RUN / "ARM_STATE.json"
    if arm_state.exists():
        state["arm"] = json.loads(arm_state.read_text())

    def not_copyable(directory, names):
        # Unix sockets and fifos cannot be copied ([Errno 6]); losing the whole
        # run directory to that is losing the evidence the run exists to produce.
        # The game workspaces are packed separately (pack_games). A reviewer's or card writer's snapshot
        # (reviews/<game>/Rnn/ws, cards/<game>/Cnn/ws) hardlinks the game record: copied out it is a second copy of
        # every frame -- its record/ stays behind, and in the rerun (nobody sees the output) the whole snapshot does;
        # Rnn/REVIEW.md, Cnn/CARD.md, result.json and cards/live/ go. Copied out, a card writer's record/ is a full copy
        # of the game's frames per card (v40 had it, 10-31). A codex home with internet clones ~100 MB of plugins into
        # .driver/.tmp (G4 replica; on Kaggle it is a lock file): not evidence, left behind too.
        here = pathlib.Path(directory)

        def snapshot_ws(path):
            return path.name == "ws" and path.parent.parent.parent in (RUN / "reviews", RUN / "cards")

        snapshot = snapshot_ws(here)
        driver_home = here.name == ".driver" and snapshot_ws(here.parent)
        return [n for n in names
                if stat.S_ISSOCK(os.lstat(os.path.join(directory, n)).st_mode)
                or stat.S_ISFIFO(os.lstat(os.path.join(directory, n)).st_mode)
                or here == RUN / "games"
                or snapshot and (rerun or n == "record")
                or driver_home and n == ".tmp"]

    try:
        if RUN.exists():
            shutil.copytree(RUN, WORK / "arm-run", symlinks=True, dirs_exist_ok=True,
                            ignore=not_copyable, ignore_dangling_symlinks=True)
            # Nobody sees the rerun's output; packing 110 workspaces there only eats
            # into the margin between the arm and the session's end.
            packed = 0 if rerun else pack_games(RUN / "games", WORK / "arm-run" / "games")
            state["arm_run_materialised"] = {"packed_games": packed, "bytes": sum(
                f.lstat().st_size for f in (WORK / "arm-run").rglob("*") if f.is_file() and not f.is_symlink())}
    except BaseException as error:
        state["arm_run_materialised"] = {"error": f"{type(error).__name__}: {error}"[:600]}
    # The per-game tokens and the competition key are not evidence, and v3's
    # output carried both. Dropped from the committed copy.
    for secret in ("gateway/tokens", "gateway/upstream_key"):
        (WORK / "arm-run" / secret).unlink(missing_ok=True)
    say("MATERIALISE", json.dumps(state.get("arm_run_materialised")))
    text = (WORK / "arm.log").read_text(errors="replace") if (WORK / "arm.log").exists() else ""
    if text.strip():
        say("ARM_TAIL", text[-6000:])
    dropped = {}
    for name in ("runtime", "environment_files"):
        path = WORK / name
        if path.is_dir() and not path.is_symlink():
            try:
                dropped[name] = sum(f.lstat().st_size for f in path.rglob("*") if f.is_file() and not f.is_symlink())
                shutil.rmtree(path)
            except BaseException as error:
                dropped[name] = f"{type(error).__name__}: {error}"[:300]
    for name in ("upstream_key", "image_ref.key"):
        (WORK / name).unlink(missing_ok=True)
    state["dropped_from_working"] = dropped
    try:
        if AGENT.is_dir():
            shutil.rmtree(AGENT)
        state["dropped_agent_root"] = str(AGENT)
    except BaseException as error:
        state["dropped_agent_root"] = f"{type(error).__name__}: {error}"[:300]
    state["complete"] = True
    save()
say("DONE", disk())
final = (WORK / "arm-run" / "FINAL.txt")
if final.exists():
    say("FINAL\n" + final.read_text())
arm_state = state.get("arm") or {}
say("RESULT", json.dumps({k: arm_state.get(k) for k in ("status", "error", "schedule", "watchdog")})[:4000])
if weight_switch is not None and weight_switch.failed and not rerun:
    # a rehearsal (Save & Run, the G4 replica) whose switch failed is not a version to submit; the rerun plays on
    raise RuntimeError("weight switch failed: " + weight_switch.failed)
