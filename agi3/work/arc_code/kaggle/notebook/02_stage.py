"""Lay down the agent's world: codex, the runtime, the two venvs.

Forked on 2026-09-25 from work/arc_agi3/arc_code_qwen/notebook/02_stage.py. What
changed is what the new arm needs: no arc-code checkout to patch (the new arm
ships its own act client), and **two** venvs instead of one -- the harness venv
holds arc_agi for the stand-in gateway, the agent venv holds only what an agent
may use. The old arm handed the agent the harness venv, and 19 of 25 games on
b200 read the actuator's source out of it.
"""

def shipped_pip():
    """pip has to be mounted, because this image can neither make one nor be given one.

    `python -m venv` runs ensurepip, and on this image ensurepip exits 1 --
    measured 2026-09-17. The competition's own arc_agi_3_wheels holds 31 wheels
    and pip is not among them. So one is shipped in its own dataset.
    """
    return mounted_glob("pip-*-py3-none-any.whl")


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(16 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stage_patched_codex(dataset, pinned, target):
    """The self-built codex (0.154.0 + patches/codex-0.154.0), copied off its mount and pinned.

    Checked three ways -- the config's pin, the dataset's own SHA256SUMS, and
    `--version` -- because a codex that is merely *a* 0.154.0 would run fine and
    silently lack view_video, the session release and the compaction prefix reuse.
    """
    root = mounted(dataset.split("/")[-1])
    native = only(root / "codex", "vendor/*/bin/codex")
    target = pathlib.Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(native, target)
    target.chmod(0o755)
    digest = sha256_of(target)
    if digest != pinned:
        raise RuntimeError(f"mounted codex is {digest}, the config pins {pinned}")
    recorded = (root / "codex/SHA256SUMS").read_text().split()[0]
    if recorded != digest:
        raise RuntimeError(f"the dataset's own SHA256SUMS says {recorded}, the file is {digest}")
    version = run([str(target), "--version"], 120).strip()
    if "0.154.0" not in version:
        raise RuntimeError(f"codex --version said {version!r}, not 0.154.0")
    return {"path": str(target), "sha256": digest, "version": version,
            "patches": (root / "codex/PATCHES").read_text().strip().splitlines()}


def stage_tools():
    """codex into AGENT/codexbin (the launcher's --driver-bin), and an arc-code prep.

    The prep is only *inspected* by the new arm (manifest.verify_prep records its
    commit and pins); it plays through its own act client. So it is copied as it
    comes, with no patches applied.
    """
    prefix = AGENT / "codexbin"
    codex = stage_patched_codex(CFG["codex_dataset"], CFG["codex_sha256"], prefix / "bin/codex")
    prep = AGENT / "arc-code-prep"
    if not prep.exists():
        shutil.copytree(mounted("arc-code-prep"), prep)
    for path in [AGENT, prefix, prefix / "bin"]:
        path.chmod(0o711 if path == AGENT else 0o755)
    return {**codex, "prefix": str(prefix), "prep": str(prep)}


def unpack_runtime():
    root = WORK / "runtime"
    if (root / "package").exists():
        return str(root / "package")
    archive = mounted_glob("*sm120-cp312.tar.zst")
    zstd = mounted_glob("zstd")
    root.mkdir(parents=True, exist_ok=True)
    # That dataset ships liblz4/liblzma/libz beside the binary because the
    # image does not have all three; copying only `zstd` gives a loader error.
    for lib in zstd.parent.iterdir():
        shutil.copy(lib, WORK / lib.name)
    (WORK / "zstd").chmod(0o755)
    run(["bash", "-c", f"{WORK}/zstd -d -c {archive} | tar -x -C {root}"], 1800,
        env=dict(os.environ, LD_LIBRARY_PATH=f"{WORK}:" + os.environ.get("LD_LIBRARY_PATH", "")))
    return str(root / "package")


def find_model():
    """Locate the weights by what they contain, not by a guessed mount path."""
    hint = CFG.get("model_hint", "")
    found = []
    for parent, dirs, files in os.walk("/kaggle/input", followlinks=True):
        if "config.json" in files and any(f.endswith(".safetensors") for f in files):
            found.append(parent)
            dirs[:] = []
    if not found:
        raise RuntimeError("no checkpoint under /kaggle/input: "
                           + ", ".join(sorted(p.name for p in pathlib.Path("/kaggle/input").iterdir())))
    matching = [p for p in found if hint in p] or found
    if len(matching) > 1:
        raise RuntimeError(f"{len(matching)} checkpoints match {hint!r}: {matching}")
    # Model versions that share config.json (QAD v2 of the gEAN320 INT3 package) differ in deploy_manifest.json,
    # which lists every shard's sha256: pinning it pins the weights without hashing 89 GB here.
    pinned = CFG.get("model_manifest_sha256")
    if pinned is not None:
        got = sha256_of(pathlib.Path(matching[0]) / "deploy_manifest.json")
        if got != pinned:
            raise RuntimeError(f"{matching[0]}/deploy_manifest.json is {got}, not the pinned {pinned}")
    if CFG.get("model_overlay"):
        return overlay_model(pathlib.Path(matching[0]), CFG["model_overlay"], OVERLAY_MODEL_DIR)
    if CFG.get("model_patch"):
        state["model_patch"] = weight_patch(CFG["model_patch"])
    if CFG.get("weight_switch"):
        state["weight_switch_export"] = weight_switch_export(CFG["weight_switch"])
    return matching[0]


def weight_switch_export(spec):
    """Config ``weight_switch``: the RL version 04_main's WeightSwitch loads in place mid-run, a model mounted beside
    the package holding one merged export -- model.safetensors and its merged_manifest.json, no config.json, so
    find_model never takes it for a checkpoint. Checked here by names and size against the manifest; WeightSwitch
    hashes model.safetensors against the pinned sha256 when the arm starts and again right before it switches."""
    root = only(mounted(spec["model"].split("/")[1]), "merged_manifest.json").parent
    names = sorted(p.name for p in root.iterdir())
    if names != ["merged_manifest.json", "model.safetensors"]:
        raise RuntimeError(f"{root} holds {names}, not exactly merged_manifest.json and model.safetensors")
    manifest = json.loads((root / "merged_manifest.json").read_text())
    size = (root / "model.safetensors").stat().st_size
    if size != manifest["bytes"] or manifest["tensors"] != len(manifest["keys"]):
        raise RuntimeError(f"{root}/model.safetensors is {size} bytes; merged_manifest.json says {manifest['bytes']} "
                           f"bytes, {manifest['tensors']} tensors, {len(manifest['keys'])} keys")
    out = {"dir": str(root), "tensors": manifest["tensors"], "bytes": size,
           "manifest_weight_version": manifest["weight_version"]}
    say("WEIGHT SWITCH EXPORT", json.dumps(out))
    return out


def weight_patch(spec):
    """Config ``model_patch``: the tensors an RL version changed (work/qwen38fn_rl/rl_weight_patch.py), a model mounted
    beside the unchanged package. start_server points the int3 runtime at it (ARC_INT3_WEIGHT_PATCH), which swaps each
    tensor in at load and checks it against patch.json's sha256; the config requires its log line. Kaggle stores
    ~3.4 GB instead of the overlay's ~21 GB of rewritten shards, and nothing is written here. Raises unless patch.json
    is the pinned one, was made against the mounted package's manifest, and names exactly the mounted patch files at
    their sizes (sizes, not sha256: the runtime hashes every tensor it swaps in)."""
    root = only(mounted(spec["model"].split("/")[1]), "patch.json").parent
    got = sha256_of(root / "patch.json")
    if got != spec["patch_sha256"]:
        raise RuntimeError(f"{root}/patch.json is {got}, the config pins {spec['patch_sha256']}")
    patch = json.loads((root / "patch.json").read_text())
    if patch["base_manifest_sha256"] != CFG["model_manifest_sha256"]:
        raise RuntimeError(f"the patch was made against {patch['base_manifest_sha256']}, "
                           f"the mounted package is {CFG['model_manifest_sha256']}")
    got = sha256_of(root / "deploy_manifest.json")
    if got != patch["deploy_manifest_sha256"]:
        raise RuntimeError(f"patch deploy_manifest.json is {got}, patch.json says {patch['deploy_manifest_sha256']}")
    files = sorted(p.name for p in root.glob("*.safetensors"))
    if files != sorted(patch["files"]):
        raise RuntimeError(f"mounted patch files {files} != patch.json {sorted(patch['files'])}")
    for name, entry in sorted(patch["files"].items()):
        if (root / name).stat().st_size != entry["bytes"]:
            raise RuntimeError(f"{root / name} is {(root / name).stat().st_size} bytes, patch.json says {entry['bytes']}")
    out = {"dir": str(root), "tensors": len(patch["tensors"]), "bytes": sum(e["bytes"] for e in patch["files"].values()),
           "deploy_manifest_sha256": patch["deploy_manifest_sha256"]}
    say("MODEL PATCH", json.dumps(out))
    return out


#: config ``model_overlay``: the served checkpoint, a directory of symlinks (outside /kaggle/working, which is output)
OVERLAY_MODEL_DIR = pathlib.Path("/tmp/arc-model-overlay")


def overlay_model(base, spec, dest):
    """The mounted package ``base`` (pinned by model_manifest_sha256) with some shards replaced by an overlay's
    (work/qwen38fn_rl/rl_overlay_package.py: the shards an RL version rewrote, its deploy_manifest.json and
    overlay.json), from a dataset or a model (overlay_root). Kaggle stores ~20 GB instead of a second 89 GB package; the
    served directory is symlinks, so no large write on a machine that vanishes past 80-90 GiB of writes. Raises unless
    the overlay's manifest is the pinned one and was made against this base, it differs from the base manifest in
    exactly the overlay's shards, and each of those has its manifest's size. Sizes, not sha256: v30 hashed them here
    and the cold read (~31 MB/s) took 649 s of the rerun's play time; the bytes are pinned by the overlay manifest's
    sha256 (checked here) and the immutable dataset / model version the kernel mounts."""
    root = overlay_root(spec)
    overlay = json.loads((root / "overlay.json").read_text())
    if overlay["base_manifest_sha256"] != CFG["model_manifest_sha256"]:
        raise RuntimeError(f"the overlay was made against {overlay['base_manifest_sha256']}, "
                           f"the mounted package is {CFG['model_manifest_sha256']}")
    got = sha256_of(root / "deploy_manifest.json")
    if not got == spec["manifest_sha256"] == overlay["deploy_manifest_sha256"]:
        raise RuntimeError(f"overlay deploy_manifest.json is {got}, pinned {spec['manifest_sha256']}, "
                           f"overlay.json says {overlay['deploy_manifest_sha256']}")
    base_files = json.loads((base / "deploy_manifest.json").read_text())["files"]
    new_files = json.loads((root / "deploy_manifest.json").read_text())["files"]
    replaced = {name for name, entry in new_files.items() if base_files.get(name) != entry}
    if set(new_files) != set(base_files) or replaced != set(overlay["files"]):
        raise RuntimeError(f"overlay manifest: files {sorted(set(new_files) ^ set(base_files))} differ in name, "
                           f"replaced {sorted(replaced)} vs overlay.json {sorted(overlay['files'])}")
    for name in sorted(replaced):
        path = root / name
        if path.stat().st_size != new_files[name]["bytes"]:
            raise RuntimeError(f"{path} is {path.stat().st_size} bytes, the overlay manifest says "
                               f"{new_files[name]['bytes']}")
    links = {entry.name: entry for entry in base.iterdir()}
    links.update({name: root / name for name in [*sorted(replaced), "deploy_manifest.json"]})
    dest.mkdir(parents=True)
    for name, target in sorted(links.items()):
        (dest / name).symlink_to(target)
    say("MODEL OVERLAY", f"{dest}: {len(links)} links, {len(replaced)} shards from {root}")
    return str(dest)


def overlay_root(spec):
    """Where overlay.json is: a dataset (``dataset``) mounts its files at its root, a model (``model``, the handle
    owner/slug/framework/variation/version) a framework/variation/version deeper."""
    if "dataset" in spec:
        return mounted(spec["dataset"].split("/")[-1])
    return only(mounted(spec["model"].split("/")[1]), "overlay.json").parent


def make_venv(name, wheels, packages, check, extra_links=()):
    """A venv without ensurepip, filled from the competition wheels with the pip we mounted.

    --without-pip is the whole point rather than a fallback: asking for a pip is
    what killed the old arm's v4. A wheel is a zip with pip/__main__.py inside it,
    so `python <wheel>/pip` runs pip without installing it; running it with the
    *venv's* interpreter is what makes it install into the venv.
    """
    venv = AGENT / name
    base = "/usr/bin/python3.12" if pathlib.Path("/usr/bin/python3.12").exists() else sys.executable
    state.setdefault("venv_base", base)
    if not (venv / "bin/python").exists():
        run([base, "-m", "venv", "--without-pip", str(venv)], 300)
    pip = shipped_pip()
    state["pip_wheel"] = pip.name
    save()
    links = [a for d in (wheels, *extra_links) for a in ("--find-links", str(d))]
    run([str(venv / "bin/python"), f"{pip}/pip", "install", "--no-index", *links,
         "--disable-pip-version-check", "-q", *packages], 900)
    run([str(venv / "bin/python"), "-c", check], 120, env=dict(os.environ, MPLBACKEND="Agg"))
    return str(venv / "bin/python")


def stage_agent_wheels(spec):
    """Config ``agent_wheels``: wheels the competition's arc_agi_3_wheels lacks (scipy), from our own dataset.

    Each file is found by name, checked against the config's sha256 and copied to one
    directory that make_venv adds as a second --find-links; the agent venv's other
    packages still come from the competition's wheels. RL's jail runtime
    ($ARC_RUNS/arc_code/agent-runtime-scipy) is the same wheel files built the same way.
    """
    dest = AGENT / "agent-wheels"
    dest.mkdir(parents=True, exist_ok=True)
    for name, want in spec["files"].items():
        source = mounted_glob(name)
        got = sha256_of(source)
        if got != want:
            raise RuntimeError(f"{source} is {got}, the config pins {want}")
        shutil.copyfile(source, dest / name)
    state["agent_wheels"] = dict(spec["files"])
    save()
    return dest


def make_venvs(wheels):
    """harness: arc_agi + flask, for the stand-in gateway (root only).
    agent: numpy, requests, pillow, matplotlib (+ config ``agent_wheels``, e.g. scipy) -- and provably
    no arcengine, because fence_posix.sh refuses a session whose runtime can import it."""
    harness = make_venv("harness-venv", wheels, ["arc-agi==0.9.8", "flask"],
                        "from arc_agi import Arcade; from arc_agi.server import create_app")
    packages, links, check = ["numpy", "requests", "pillow", "matplotlib"], [], "import numpy, requests, PIL, matplotlib; "
    if CFG.get("agent_wheels"):
        spec = CFG["agent_wheels"]
        links = [stage_agent_wheels(spec)]
        packages += spec["packages"]
        check += "".join(f"import {p.split('==')[0]}; assert {p.split('==')[0]}.__version__ == {p.split('==')[1]!r}; "
                         for p in spec["packages"]) + spec["check"] + "; "
    agent = make_venv("agent-venv", wheels, packages,
                      check + "import importlib.util as u; raise SystemExit(u.find_spec('arcengine') is not None)", links)
    # The agent venv is read by every game's uid; the harness venv by root only.
    run(["chmod", "-R", "a+rX,go-w", str(AGENT / "agent-venv")], 300)
    (AGENT / "harness-venv").chmod(0o700)
    return {"harness": harness, "agent": agent}
