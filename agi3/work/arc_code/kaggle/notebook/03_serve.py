"""Bring up sglang on the pinned runtime, and prove it is serving what was asked for.

Forked verbatim on 2026-09-25 from work/arc_agi3/arc_code_qwen/notebook/03_serve.py:
the runtime, the rewarm and the startup checks are the parts of the old arm that
Kaggle has already proven, and nothing about the new arm changes how sglang starts.
"""


def runtime_interpreter():
    """The pinned runtime is a cp312 site-packages tree, not a venv, so it needs a
    system interpreter of exactly that version. Kaggle's image python is 3.12,
    but it is not necessarily at /usr/bin/python3.12.
    """
    interpreter = "/usr/bin/python3.12" if pathlib.Path("/usr/bin/python3.12").exists() else sys.executable
    if subprocess.run([interpreter, "-c", "import sys; raise SystemExit(sys.version_info[:2] != (3, 12))"]).returncode:
        raise RuntimeError(f"no python3.12 for the cp312 runtime; tried {interpreter}")
    return interpreter


def runtime_env(runtime, env, arch_list=None):
    """The environment this runtime is served with everywhere else, not a new guess.

    v5 built its own env here holding nothing but a PYTHONPATH, and sglang died
    inside flashinfer's JIT with "FlashInfer requires GPUs with sm75 or higher"
    on an sm120 card. That message is what an *empty* TARGET_CUDA_ARCHS prints:
    the set is filled from the CUDA toolchain, and nothing here was pointing at
    one. The archive ships its own (manifest `layout.cuda` = package/cuda-13.1),
    which is exactly what work/qwen38fn_eval/qsa_sm120_pipeline.py's
    runtime_environment() -- the env that has actually served this tree on this
    image -- puts on PATH. Mirroring it is the point; the tools it asserts on are
    the ones flashinfer shells out to.
    """
    root = pathlib.Path(runtime)
    site = root / "lib/python3.12/site-packages"
    env = dict(env,
               CUDA_HOME=str(root / "cuda-13.1"),
               CUDA_VISIBLE_DEVICES="0",
               PATH=os.pathsep.join([str(root / "cuda-13.1/bin"), str(root / "bin"), env["PATH"]]),
               PYTHONPATH=os.pathsep.join([str(root / "python"), str(site),
                                           str(site / "nvidia_cutlass_dsl/dsl_packages")]),
               MAX_JOBS="2", FLASHINFER_NVCC_THREADS="1", MPLBACKEND="Agg")
    # flashinfer decides what to compile for from TARGET_CUDA_ARCHS, and left to
    # itself it derives that through a CUDA-version probe that can fail quietly:
    # every failure inside CompilationContext.__init__ is swallowed into a
    # warning and leaves the set *empty*, which check_cuda_arch then reports as
    # "requires GPUs with sm75 or higher" -- the v5 death, on an sm120 card.
    # A value carrying an arch suffix is taken as-is, so passing the one measured
    # off this GPU skips that path entirely instead of hoping it works.
    if arch_list:
        env["FLASHINFER_CUDA_ARCH_LIST"] = arch_list
    missing = [tool for tool in ("nvcc", "ninja", "g++")
               if shutil.which(tool, path=env["PATH"]) is None]
    if missing:
        raise RuntimeError(f"the pinned runtime cannot compile: {missing} not on {env['PATH']}")
    return env


CUDA_REPORT = r"""
import json, os
report = {}
def note(name, fn):
    try:
        report[name] = fn()
    except BaseException as error:
        report[name] = f"{type(error).__name__}: {error}"
import torch
note("torch", lambda: torch.__version__)
note("torch_cuda", lambda: torch.version.cuda)
note("device_count", lambda: torch.cuda.device_count())
note("capability", lambda: list(torch.cuda.get_device_capability(0)))
report["env_arch_list"] = {name: os.environ.get(name) for name in
                           ("FLASHINFER_CUDA_ARCH_LIST", "TORCH_CUDA_ARCH_LIST")}
from flashinfer.jit.cpp_ext import get_cuda_path, get_cuda_version
note("cuda_path", get_cuda_path)
note("cuda_version", lambda: str(get_cuda_version()))
from flashinfer.compilation_context import CompilationContext
# Its own normalisation, not a copy of it: the sm12.x suffix rule lives there and
# changes with the CUDA version it detects.
note("arch_list", lambda: "%d.%s" % CompilationContext._normalize_cuda_arch(
    *torch.cuda.get_device_capability(0)))
note("target_archs", lambda: sorted(f"{a}.{b}" for a, b in CompilationContext().TARGET_CUDA_ARCHS))
print(json.dumps(report))
"""


def eligible(target_archs):
    """flashinfer's own rule, applied where the numbers are still readable.

    check_cuda_arch accepts sm75 and up, so an empty list is not the only way to
    fail it: a list holding nothing but older architectures -- which is what an
    inherited FLASHINFER_CUDA_ARCH_LIST would produce -- fails it too, and prints
    the same sentence about sm75.
    """
    if not isinstance(target_archs, list):
        return False
    for arch in target_archs:
        major, _, minor = str(arch).partition(".")
        if major.isdigit() and (int(major) >= 8 or (major == "7" and minor[:1].isdigit()
                                                    and int(minor[:1]) >= 5)):
            return True
    return False


def cuda_report(interpreter, env):
    """Measure what flashinfer will decide, before sglang spends ten minutes on it.

    Every field is recorded rather than asserted, except the one that decides:
    an empty arch list is the v5 failure, and it is worth naming here -- where
    the numbers that produced it are still in hand -- instead of one stage later
    as a sentence about sm75.
    """
    done = subprocess.run([interpreter, "-c", CUDA_REPORT], capture_output=True, text=True,
                          timeout=600, env=env)
    if done.returncode:
        raise RuntimeError(f"cuda probe exited {done.returncode}: {(done.stdout + done.stderr)[-1200:]}")
    report = json.loads(done.stdout.strip().splitlines()[-1])
    # One rule, flashinfer's own, applied to the value actually being pinned.
    # Every field here is produced by a helper that turns failures into strings,
    # so "is a string" proves nothing -- "would satisfy check_cuda_arch" does.
    if not eligible([report.get("arch_list")]):
        raise RuntimeError("no usable architecture to pin for flashinfer, which is the v5 death "
                           "dressed as a message about sm75: " + json.dumps(report))
    return report


def assert_runtime_provenance(runtime):
    """The patch stack is already in the archive; this is where that is checked.

    The old line applied patches here, on Kaggle, which meant "one patch did not
    apply" was discovered in the one place nobody can look -- and it happened:
    an installer's output piped into `head` gave `patch` a SIGPIPE and left the
    stack half-applied. package_runtime_sm120_0520.sh now runs the marker gate
    on the b200 and writes runtime-provenance.json beside the tree, so all that
    is left here is to refuse an archive that is not the one that was gated.

    Hashes, not markers: a marker says a patch was applied, a hash says the file
    is the file. The archive could still be an older build of the same stack.
    """
    root = pathlib.Path(runtime)
    record_path = root / "runtime-provenance.json"
    if not record_path.exists():
        raise RuntimeError(f"{record_path} is missing: this archive was not built by "
                           "package_runtime_sm120_0520.sh, so nothing knows what is in it")
    record = json.loads(record_path.read_text())
    expected = CFG.get("runtime_source_sha256") or {}
    if not expected:
        raise RuntimeError("the config carries no runtime_source_sha256, so the archive "
                           "cannot be pinned; add it from the packager's provenance")
    actual = record.get("source_sha256") or {}
    drifted = sorted(name for name, digest in expected.items() if actual.get(name) != digest)
    if drifted:
        raise RuntimeError(f"the mounted runtime is not the one this notebook was built "
                           f"against; {len(drifted)} files differ: {drifted[:6]}")
    missing = sorted(set(CFG.get("runtime_patches") or []) - set(record.get("patches") or {}))
    if missing:
        raise RuntimeError(f"the archive was built without {missing}")
    say("RUNTIME", json.dumps({"base_ref": record.get("base_ref"),
                               "arc_forge_commit": record.get("arc_forge_commit"),
                               "patches": len(record.get("patches") or {}),
                               "pinned_files": len(expected),
                               "aot": (record.get("aot") or {}).get("modules")}))
    return {"base_ref": record.get("base_ref"),
            "arc_forge_commit": record.get("arc_forge_commit"),
            "patches": sorted(record.get("patches") or {}),
            "aot_bytes": (record.get("aot") or {}).get("bytes")}


def apply_runtime_overlay(runtime, entries, dataset):
    """Replace pinned runtime files with pinned files shipped in `dataset`.

    How one sglang file changes without repackaging the multi-GB runtime: the
    arm uses it for the rewritten 0018 (serving_responses.py), which the runtime
    dataset does not have yet. Runs after assert_runtime_provenance(), so the
    archive underneath is still the one runtime_source_sha256 pins -- that check
    is about the archive and is left exactly as it was. Every entry is pinned
    three ways (the file it replaces, the shipped file, the file afterwards),
    `dest` must be the copy Python imports -- the first match along the
    runtime's PYTHONPATH -- and every mismatch raises.
    """
    if not entries:
        return "none"
    root = pathlib.Path(runtime)
    shipped_root = mounted(dataset.split("/")[-1])
    search = [pathlib.Path(p) for p in runtime_env(runtime, os.environ.copy())["PYTHONPATH"].split(os.pathsep)]
    applied = []
    for entry in entries:
        dest, source = root / entry["dest"], shipped_root / entry["source"]
        pinned = CFG["runtime_source_sha256"].get(entry["dest"])
        if pinned != entry["replaces_sha256"]:
            raise RuntimeError(f"overlay {entry['dest']} replaces {entry['replaces_sha256']}, but "
                               f"runtime_source_sha256 pins {pinned}")
        module = pathlib.Path(entry["dest"]).relative_to("python")
        imported = next((d / module for d in search if (d / module).is_file()), None)
        if imported != dest:
            raise RuntimeError(f"Python would import {module} from {imported}, not the overlay target {dest}")
        before = sha256_of(dest)
        if before != entry["replaces_sha256"]:
            raise RuntimeError(f"{dest} is {before}, the overlay expects to replace {entry['replaces_sha256']}")
        shipped = sha256_of(source)
        if shipped != entry["source_sha256"]:
            raise RuntimeError(f"{source} is {shipped}, the config pins {entry['source_sha256']}")
        shutil.copyfile(source, dest)
        after = sha256_of(dest)
        if after != entry["result_sha256"]:
            raise RuntimeError(f"{dest} is {after} after the copy, the config pins {entry['result_sha256']}")
        stale = sorted((dest.parent / "__pycache__").glob(dest.stem + ".*.pyc"))
        for pyc in stale:
            pyc.unlink()
        applied.append({"dest": entry["dest"], "source": entry["source"], "before": before, "after": after,
                        "stale_pyc_removed": [p.name for p in stale]})
        say("OVERLAY", entry["dest"], before[:12], "->", after[:12], f"{len(stale)} stale pyc removed")
    return applied


#: serve_e2e.sh's FORBIDDEN list: test-only switches (determinism, debug checks, skipped I/O). Any of them in a
#: served environment changes what is measured without saying so.
TEST_ONLY_ENV = ("ARC_HISPARSE_QSA_SYNC_STAGING", "ARC_QSA_TOPK_EXACT", "ARC_HC_MIX_DETERMINISTIC",
                 "SGLANG_FLASHINFER_MOE_FUSED_FINALIZE", "ARC_QSA_TOPK_SORT", "ARC_HISPARSE_QSA_DEBUG",
                 "SGLANG_DEBUG_HISPARSE_SKIP_IO")

#: distributions() walks sys.path in order and also yields the image's own copies further down (Kaggle's torch
#: 2.10.0 behind the runtime's 2.13.0), so keep the first of each name: that is the one import resolves.
TREE_PROBE = r"""
import importlib.metadata as m, importlib.util as u, json
want = {"torch", "triton", "flashinfer-python", "flashinfer-jit-cache", "sglang-kernel", "sgl-kernel", "tilelang"}
versions = {}
for d in m.distributions():
    if d.metadata["Name"] in want:
        versions.setdefault(d.metadata["Name"], d.version)
print(json.dumps({"origin": u.find_spec("sglang").origin, "versions": versions}))
"""


def install_sglang_tree(runtime):
    """Replace the archive's sglang source with the HiSparse tree, and prove the server will import that tree.

    The runtime archive stays the one v12 serves (assert_runtime_provenance has just checked it): its
    site-packages, the flashinfer AOT set and the CUDA toolchain are what the G4 runs used too. Only its
    python/sglang (0001-0028 and the old 0018) goes, replaced by v0.5.20 + 0001-0029 + experimental-hisparse
    h*, built and asserted on the PC (work/qsa_hisparse/probes/pack_kaggle_tree.py) -- the G4 fast path does
    the same (g4_runtime_install.sh). Pinned three ways: the archive's sha256, the whole-package digest after
    extraction (patches/sglang-0.5.20/TREE_DIGEST, arcforge.sglang_tree), and the import origin as the runtime
    interpreter sees it. The manifest h05 checks at startup comes from the same dataset, pinned too.
    """
    spec = CFG["sglang_tree"]
    root = pathlib.Path(runtime)
    archive = mounted_glob(spec["archive"])
    manifest = archive.parent / spec["manifest"]
    for path, pinned in ((archive, spec["archive_sha256"]), (manifest, spec["manifest_sha256"])):
        got = sha256_of(path)
        if got != pinned:
            raise RuntimeError(f"{path} is {got}, the config pins {pinned}")
    zstd = WORK / "zstd"
    if not zstd.is_file():
        raise RuntimeError(f"no {zstd}: unpack_runtime lays it down, and it has to run first")
    plain = WORK / "sglang-tree.tar"
    run(["bash", "-c", f"{zstd} -q -d -c {archive} > {plain}"], 600,
        env=dict(os.environ, LD_LIBRARY_PATH=f"{WORK}:" + os.environ.get("LD_LIBRARY_PATH", "")))
    with tarfile.open(plain) as tar:
        members = tar.getmembers()
        outside = [m.name for m in members if m.name != "python/sglang" and not m.name.startswith("python/sglang/")]
        if outside:
            raise RuntimeError(f"the tree archive has members outside python/sglang/: {outside[:5]}")
        shutil.rmtree(root / "python")
        tar.extractall(root, filter="data")
    plain.unlink()
    if str(CODE / "src") not in sys.path:
        sys.path.insert(0, str(CODE / "src"))
    from arcforge.sglang_tree import manifest_digest, package_manifest
    entries = package_manifest(root / "python" / "sglang")
    digest = manifest_digest(entries)
    if (digest, len(entries)) != (spec["tree_digest"], spec["tree_files"]):
        raise RuntimeError(f"the extracted tree is {digest} ({len(entries)} files), the config pins "
                           f"{spec['tree_digest']} ({spec['tree_files']} files)")
    done = subprocess.run([runtime_interpreter(), "-c", TREE_PROBE], capture_output=True, text=True, timeout=300,
                          env=runtime_env(runtime, os.environ.copy()))
    if done.returncode:
        raise RuntimeError(f"tree probe exited {done.returncode}: {(done.stdout + done.stderr)[-1200:]}")
    probe = json.loads(done.stdout.strip().splitlines()[-1])
    expected = root / "python" / "sglang" / "__init__.py"
    if pathlib.Path(probe["origin"]) != expected:
        raise RuntimeError(f"the runtime interpreter imports sglang from {probe['origin']}, not {expected}")
    report = {"digest": digest, "files": len(entries), "archive": str(archive), "manifest": str(manifest),
              "members": len(members), "versions": probe["versions"]}
    say("SGLANG TREE", json.dumps(report))
    return report


def cgroup_now():
    """The container's memory as the kernel accounts it (GiB), with the headroom posix_fence.py watches:
    memory.max - anon - shmem. Page cache is left out on purpose -- the kernel reclaims it before it kills
    anything, and the notebook keeps the checkpoint in it deliberately (rewarm). Pinned host memory (the PLE
    table, the HiSparse host pool) is shmem. Without a ceiling, MemAvailable is the headroom."""
    rel = Path("/proc/self/cgroup").read_text().strip().splitlines()[-1].split(":", 2)[2]
    base = Path("/sys/fs/cgroup") / rel.lstrip("/")
    stat = dict(line.split() for line in (base / "memory.stat").read_text().splitlines())
    gib = {k: int(stat[k]) / 2**30 for k in ("anon", "shmem", "file")}
    ceiling = (base / "memory.max").read_text().strip()
    if ceiling == "max":
        headroom = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                        if line.startswith("MemAvailable:")) / 2**20
    else:
        headroom = int(ceiling) / 2**30 - gib["anon"] - gib["shmem"]
    return {**{k: round(v, 1) for k, v in gib.items()}, "headroom": round(headroom, 1), "max": ceiling}


class MemorySampler(threading.Thread):
    """cgroup memory every `every` s until stopped: the low point of the headroom and the peaks, e.g. while
    the server starts (a cold JIT cache compiles then, next to the checkpoint in the page cache)."""

    def __init__(self, every=2.0):
        super().__init__(daemon=True, name="memory-sampler")
        self.every, self.halt, self.samples = every, threading.Event(), []

    def run(self):
        while True:
            self.samples.append(cgroup_now())
            if self.halt.wait(self.every):
                return

    def stop(self):
        self.halt.set()
        self.join(timeout=30)
        if not self.samples:
            raise RuntimeError("the memory sampler took no sample")
        low = min(self.samples, key=lambda s: s["headroom"])
        return {"samples": len(self.samples), "headroom_min": low["headroom"], "at_headroom_min": low,
                **{f"{k}_max": max(s[k] for s in self.samples) for k in ("anon", "shmem", "file")}}


def jit_cache_dirs(env):
    """Where anything compiled at startup lands: the cache directories the env names, and the defaults under
    $HOME that the runtime also writes (tilelang, triton, the CUDA cache)."""
    home = pathlib.Path(env.get("HOME") or os.path.expanduser("~"))
    named = [env[k] for k in ("SGLANG_CACHE_DIR", "SGLANG_JIT_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH",
                              "FLASHINFER_WORKSPACE_BASE") if env.get(k)]
    return sorted({pathlib.Path(p) for p in named}
                  | {home / ".cache", home / ".triton", home / ".nv", home / ".tilelang"})


def cache_files(dirs):
    return {str(p) for d in dirs if d.is_dir() for p in d.rglob("*") if p.is_file()}


def hisparse_startup_checks(text):
    """What the HiSparse server log must say beyond the static require_log markers: the tree h05 verified
    (the manifest's sha256), and the sizing R, C, L this config asks for. B (hot blocks per request) is read,
    reported against the G4 run, and not enforced: it follows from the VRAM left after the weights, which is
    a measurement, not a contract; h17 refuses B < K x top-k by itself."""
    spec = CFG["sglang_tree"]
    args = CFG["server_args"]
    logical = json.loads(args[args.index("--hisparse-config") + 1])["qsa_logical_tokens"]
    for marker in (f"ARC_HISPARSE_QSA_TREE={spec['manifest_sha256']}",
                   f"R={CFG['max_requests']}, C={CFG['context_length']} -> logical {logical} tokens"):
        if marker not in text:
            near = [line[-300:] for line in text.splitlines() if "QSA HiSparse sizing" in line or
                    "ARC_HISPARSE_QSA_TREE" in line][:4]
            raise RuntimeError(f"server log never said {marker!r}; nearby: {near or 'none'}")
    sizing = re.search(r"QSA HiSparse sizing: budget ([\d.]+) GiB, R=(\d+), C=(\d+) -> logical (\d+) tokens, "
                       r"prefill area (\d+) tokens, B=(\d+) blocks", text)
    if sizing is None:
        raise RuntimeError("the server log has the sizing marker but no parsable sizing line")
    keep = ("QSA HiSparse host pool:", "QSA HiSparse GDN snapshots:", "QSA HiSparse session host budget:",
            "QSA HiSparse draft pool:", "available_gpu_mem=")
    report = {"budget_gib": float(sizing[1]), "R": int(sizing[2]), "C": int(sizing[3]), "L": int(sizing[4]),
              "prefill_area": int(sizing[5]), "B": int(sizing[6]), "B_g4_c55": CFG.get("expect_hot_blocks"),
              "lines": [line[line.index(k):][:240] for line in text.splitlines() for k in keep if k in line][:8]}
    say("HISPARSE", json.dumps(report))
    return report


def prefetch_weights(model):
    """prefetch_checkpoint over what the loader reads: the checkpoint, and the weight patch (02_stage weight_patch)
    whose tensors it swaps in between the checkpoint's own."""
    began = time.monotonic()
    dirs = [Path(model), *([Path(state["model_patch"]["dir"])] if CFG.get("model_patch") else [])]
    total = sum(prefetch_checkpoint(d, CFG.get("prefetch_workers", 16))["bytes"] for d in dirs)
    return {"bytes": total, "GB_per_second": total / (time.monotonic() - began) / 1e9, "dirs": [str(d) for d in dirs]}


def start_prefetch(model):
    """Warm the checkpoint's page cache while the runtime is still unpacking.

    sglang's loader mmaps the shards, and mmap is only fast on a warm cache --
    that is the pairing `vista-sequential-nvfp4/config.json` records as "mmap
    loader follows verified page-cache warming". Reading them here, on 16
    threads, overlaps the cold NFS read with stages that are waiting on CPU and
    disk anyway. It is not sufficient on its own: see rewarm().
    """
    outcome = {}

    def work():
        begin = time.monotonic()
        try:
            outcome["manifest"] = prefetch_weights(model)
        except BaseException as error:  # a cold cache is slower, not fatal
            outcome["error"] = f"{type(error).__name__}: {error}"
        outcome["seconds"] = round(time.monotonic() - begin, 1)

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    say("PREFETCH", "started on", model, disk())
    handle = (thread, outcome)
    PENDING.append(lambda: join_prefetch(handle))
    return handle


def join_prefetch(handle):
    thread, outcome = handle
    thread.join()
    manifest = outcome.get("manifest") or {}
    report = {"seconds": outcome.get("seconds"), "error": outcome.get("error"),
              "bytes": manifest.get("bytes"), "GB_per_second": manifest.get("GB_per_second")}
    say("PREFETCH", json.dumps(report))
    if outcome.get("error"):
        # Named, not raised: the loader will simply read it cold instead.
        state["prefetch_failed"] = outcome["error"]
    return report


def rewarm(model):
    """Read the checkpoint a second time, right before the server, so it survives the load.

    One read leaves every page on the inactive LRU list, and the server's own
    ~75 GiB of host memory pushes the 175 GiB cgroup to its limit while the
    loader runs -- so the kernel reclaims exactly those pages and the mmap
    loader refaults them off NFS. Measured 2026-09-24 (startup probe v1): the
    arm's single prefetch, 100% resident at launch, still refaulted 106.9 GB
    and took 1,174 s in Load weight; a second read, which promotes the pages
    to the active list, brought it to 31 s with 1.1 GB refaulted. The second
    read itself costs ~4 s because it is served from the cache it protects.

    It is a warm read only if it is fast. A slow one means the first pass was
    already evicted; that is recorded and printed, not raised -- the server
    still starts, just at the old speed.
    """
    begin = time.monotonic()
    manifest = prefetch_weights(model)
    report = {"seconds": round(time.monotonic() - begin, 1),
              "GB_per_second": round(manifest["GB_per_second"], 3)}
    if report["GB_per_second"] < 5:
        state["rewarm_cold"] = report
        say("REWARM", "COLD: the second read went to NFS, so the first was already evicted "
            "and the load will be slow:", json.dumps(report))
    else:
        say("REWARM", json.dumps(report))
    return report


def dflash_env():
    """ARC_DFLASH_* for the MTP5-DFlash drafter (sglang-arc-2 speculative/arc_dflash), resolved on the mounted
    dataset: the export directory, its keep set and GEMM2 calibration, each pinned by sha256 (the export by its
    config.json; the worker itself refuses an incomplete export). A dense export (`ffn_type: dense_shared`,
    arc-forge convert_dense.py) has no experts to keep or calibrate: only its config is pinned, and its
    config.json must say dense_shared; plan B's plain / injected-context export (`ffn_type: dense`) likewise.
    An optional draft vocabulary (draft_vocab*) restricts the drafter's head to
    a pinned id list. The worker replaces the native MTP draft, so the checkpoint's MTP weights are not loaded and
    there is no draft KV pool."""
    spec = CFG["dflash"]
    dense = spec.get("ffn_type") in ("dense_shared", "dense")
    root = mounted(spec["dataset"].split("/")[-1])
    export = root / spec["export_dir"] if spec["export_dir"] else root
    files = {"config": (export / "config.json", spec["config_sha256"])}
    if not dense:
        files |= {"keep": (root / spec["keep"], spec["keep_sha256"]),
                  "calib": (root / spec["calib"], spec["calib_sha256"])}
    for name, (path, pinned) in files.items():
        got = sha256_of(path)
        if got != pinned:
            raise RuntimeError(f"dflash {name} {path} is {got}, the config pins {pinned}")
    ffn = json.loads((export / "config.json").read_text()).get("ffn_type", "moe")
    if (ffn != spec["ffn_type"]) if dense else ffn in ("dense_shared", "dense"):
        raise RuntimeError(f"dflash export {export} has ffn_type {ffn}, the notebook config says {spec.get('ffn_type')}")
    out = {"ARC_DFLASH_DRAFTER": str(export)}
    if not dense:
        out |= {"ARC_DFLASH_KEEP": str(files["keep"][0]), "ARC_DFLASH_CALIB": str(files["calib"][0])}
    if spec.get("draft_vocab"):
        # the drafter's head over a fixed id list (sglang-arc-2 heads.SubsetHead), pinned by sha256
        path = mounted(spec["draft_vocab_dataset"].split("/")[-1]) / spec["draft_vocab"]
        got = sha256_of(path)
        if got != spec["draft_vocab_sha256"]:
            raise RuntimeError(f"dflash draft vocab {path} is {got}, the config pins {spec['draft_vocab_sha256']}")
        out["ARC_DFLASH_DRAFT_VOCAB"] = str(path)
    state["dflash"] = out
    say("DFLASH", json.dumps(out))
    return out


def install_humming():
    """The Humming wheel the packed INT3 experts run on (config ``int3``), laid out as a PYTHONPATH overlay.

    The runtime archive has no humming-kernels; G4 lays the same 0.1.16 wheel beside it
    (work/qwen38fn_eval/g4_provision_int3_pair.py). Unzipped, not pip-installed, so the .dist-info the INT3
    plugin reads its version from comes along. /kaggle/input is read-only and NVRTC refuses the bundled
    native tools without the exec bit, so the overlay lives under WORK and they are chmod'ed there.
    """
    spec = CFG["int3"]
    wheel = mounted_glob(spec["humming_wheel"])
    got = sha256_of(wheel)
    if got != spec["humming_wheel_sha256"]:
        raise RuntimeError(f"{wheel} is {got}, the config pins {spec['humming_wheel_sha256']}")
    overlay = WORK / "humming-overlay"
    if overlay.exists():
        shutil.rmtree(overlay)
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(overlay)
    native = sorted((overlay / "humming/_native/x86_64").glob("*"))
    if not native:
        raise RuntimeError(f"the humming wheel laid no native tools under {overlay}/humming/_native/x86_64")
    for path in native:
        path.chmod(0o755)
    dist = sorted(overlay.glob("humming_kernels-*.dist-info"))
    if [d.name for d in dist] != [f"humming_kernels-{spec['humming_version']}.dist-info"]:
        raise RuntimeError(f"humming overlay dist-info {[d.name for d in dist]} != {spec['humming_version']}")
    out = {"overlay": str(overlay), "wheel": str(wheel), "native": len(native)}
    say("HUMMING", json.dumps(out))
    return out


def start_server(runtime, model, env, arch_list):
    env = runtime_env(runtime, env, arch_list)
    env.update({k: str(v) for k, v in CFG.get("server_env", {}).items()})
    if CFG.get("int3"):
        if "humming" not in state:
            raise RuntimeError("the config serves packed INT3 experts but install_humming never ran")
        overlay = state["humming"]["overlay"]
        env["PYTHONPATH"] = os.pathsep.join([overlay, env["PYTHONPATH"]])
        env["ARC_HUMMING_DIR"] = overlay
        if CFG["int3"].get("tuning"):
            table = CODE / CFG["int3"]["tuning"]
            got = sha256_of(table)
            if got != CFG["int3"]["tuning_sha256"]:
                raise RuntimeError(f"{table} is {got}, the config pins {CFG['int3']['tuning_sha256']}")
            env["ARC_INT3_TUNING"] = str(table)
            say("INT3 TUNING", str(table))
    if CFG.get("model_patch"):
        if not CFG.get("int3") or "model_patch" not in state:
            raise RuntimeError("model_patch is swapped in by the int3 runtime, after 02_stage weight_patch checked it")
        env["ARC_INT3_WEIGHT_PATCH"] = state["model_patch"]["dir"]
        say("WEIGHT PATCH", env["ARC_INT3_WEIGHT_PATCH"])
    if CFG.get("sglang_tree"):
        if "sglang_tree" not in state:
            raise RuntimeError("the config asks for the HiSparse tree but install_sglang_tree never ran")
        env["ARC_HISPARSE_QSA_MANIFEST"] = state["sglang_tree"]["manifest"]
    if CFG.get("dflash"):
        env.update(dflash_env())
    leaked = [k for k in TEST_ONLY_ENV if k in env]
    if leaked:
        raise RuntimeError(f"test-only switches in the server environment: {leaked}")
    # Record-only: the switches the config hands sglang (none secret). Opt-in features that log nothing when on
    # (h20's ARC_GDN_FOLD_IN_VERIFY only refuses a start without its prerequisites) show up here.
    state["server_env"] = {k: env[k] for k in CFG.get("server_env", {})}
    say("SERVER ENV", json.dumps(state["server_env"]))
    caches = jit_cache_dirs(env)
    for d in caches:
        d.mkdir(parents=True, exist_ok=True)
    for key in ("SGLANG_CACHE_DIR", "SGLANG_JIT_CACHE_DIR", "TRITON_CACHE_DIR", "CUDA_CACHE_PATH"):
        if env.get(key):
            pathlib.Path(env[key]).chmod(0o700)  # root's; no game uid has any business there
    cached_before = cache_files(caches)
    interpreter = runtime_interpreter()
    # Packed INT3: the launcher audits the checkpoint headers, installs the arc_int3 loader in this process and
    # (through its sitecustomize) in every spawned worker, then runs sglang.launch_server with the same argv.
    entry = ([str(CODE / "work/qwen38fn_eval/int3_deploy_runtime.py"), "launch"] if CFG.get("int3")
             else ["-m", "sglang.launch_server"])
    argv = [interpreter, *entry, "--model-path", str(model),
            "--served-model-name", CFG["served_model"], "--host", "127.0.0.1",
            "--port", str(CFG["port"]), "--context-length", str(CFG["context_length"]),
            "--max-running-requests", str(CFG["max_requests"]),
            "--mem-fraction-static", str(CFG["mem_fraction"]), *CFG.get("server_args", [])]
    if CFG.get("max_total_tokens"):
        argv += ["--max-total-tokens", str(CFG["max_total_tokens"])]
    log = (WORK / "server.log").open("w")
    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, env=env)
    base = f"http://127.0.0.1:{CFG['port']}"
    begin = time.monotonic()
    memory = MemorySampler()
    memory.start()
    try:
        while time.monotonic() - begin < CFG["startup_timeout"]:
            if proc.poll() is not None:
                raise RuntimeError("server exited: " + (WORK / "server.log").read_text()[-1500:])
            try:
                with urllib.request.urlopen(base + "/health", timeout=3) as response:
                    if response.status == 200:
                        break
            except Exception:
                time.sleep(3)
        else:
            raise RuntimeError("startup deadline")
    finally:
        state["startup_memory"] = memory.stop()
        compiled = sorted(cache_files(caches) - cached_before)
        state["startup_compiled"] = {
            "new_files": len(compiled), "dirs": [str(d) for d in caches],
            "sgl_jit": sorted({pathlib.Path(f).parent.name for f in compiled if "sgl_kernel_jit" in f})[:60],
            "sample": compiled[:30]}
        save()
        say("STARTUP", json.dumps({"seconds": round(time.monotonic() - begin, 1),
                                   "memory": state["startup_memory"],
                                   "compiled_files": state["startup_compiled"]["new_files"]}))
    # codex 0.154 removed wire_api=chat, so /v1/responses is the only wire left --
    # and sglang turns that endpoint off on an init failure with one WARNING line
    # rather than refusing to start. Offline, its optional downloads are a live
    # candidate for exactly that.
    text = (WORK / "server.log").read_text(errors="replace")
    bad = [line for line in text.splitlines() if "Responses API (/v1/responses) disabled" in line]
    if bad:
        raise RuntimeError("sglang disabled the Responses API: " + bad[0][-300:])
    # A served NVFP4 pool is a claim about the startup log, not about the flags
    # that were passed: sglang will fall back and start quite happily. That
    # check used to be imported from install_qsa_nvfp4_kaggle; with the patch
    # stack baked into the archive there is no installer to import, so the
    # markers live in the config beside every other startup-log claim.
    if CFG.get("kv_cache_dtype") == "nvfp4" and not CFG.get("require_log"):
        raise RuntimeError("kv_cache_dtype is nvfp4 but the config asks for no log "
                           "markers, so nothing would notice the pool falling back")
    # Each of these has a silent fallback behind it: QSA can drop back to its
    # fa2 default, and an `auto` MoE runner can land on trtllm-gen, which has no
    # SM120 cubins. Neither says so; both change what was measured.
    for marker in CFG.get("require_log", []):
        if marker not in text:
            # A marker can be absent for two very different reasons: the
            # fallback really happened, or the string moved. 0.5.20 switched to
            # logging `server_args=<resolved_dict()>`, which turned
            # `moe_runner_backend='x'` into `'moe_runner_backend': 'x'` -- and a
            # gate that fails with nothing but the string it wanted costs a
            # whole run to tell those two apart. So bring the evidence.
            stem = max(marker.replace("'", " ").replace(":", " ").split(), key=len)
            near = [line[-300:] for line in text.splitlines() if stem in line][:4]
            raise RuntimeError(f"server log never said {marker!r}; lines mentioning "
                               f"{stem!r}: {near or 'none'}")
    for marker in CFG.get("refuse_log", []):
        if marker in text:
            raise RuntimeError(f"server log said {marker!r}")
    if CFG.get("sglang_tree"):
        state["hisparse_sizing"] = hisparse_startup_checks(text)
        after = cgroup_now()
        state["memory_after_start"] = after
        if after["headroom"] < CFG["min_headroom_after_start_gib"]:
            raise RuntimeError(f"cgroup headroom {after['headroom']} GiB after start < "
                               f"{CFG['min_headroom_after_start_gib']}: something holds memory the plan "
                               f"does not count: {json.dumps(after)}")
    with urllib.request.urlopen(base + "/get_server_info", timeout=30) as response:
        info = json.loads(response.read())
    state["server_info"] = {k: info.get(k) for k in
                            ("max_total_num_tokens", "context_length", "max_running_requests",
                             "kv_cache_dtype", "page_size", "attention_backend")}
    save()
    say("ALLOC", json.dumps(state["server_info"]))
    return proc, base, round(time.monotonic() - begin, 1), env
