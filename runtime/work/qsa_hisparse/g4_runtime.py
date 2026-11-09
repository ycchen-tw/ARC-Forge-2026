"""G4 runtime identity: what a Colab G4 serves from, asserted, and a fingerprint every run records.

The VM's runtime is described by one record the bootstrap writes, /content/rt/RUNTIME.json:
  kind "kaggle-dataset"  the pinned Kaggle runtime (g4_runtime_spec.json) at root, with main's sglang source laid
                         over it per arm (the dataset's own sglang source is deleted), one venv per arm whose .pth
                         puts the arm's source, the runtime's site-packages and the cutlass DSL packages on
                         sys.path (what the Kaggle notebook puts on PYTHONPATH)
  kind "source-build"    the slow path: build_sglang_0520_sm120.sh per arm (no dataset, versions only recorded)
Every command reads that record; there is no default runtime.

  identity     assert the runtime against the spec (dataset, archive, provenance, python, package versions,
               module locations, flashinfer AOT files, the nvcc sglang's JIT resolves) -- run with the arm's python
  tree         the whole installed sglang package (every file but __pycache__ and the generated _version.py)
               against patches/sglang-0.5.20/TREE_DIGEST; with --pristine, list extra / missing / changed files
  tree-write   write TREE_DIGEST from a pristine v0.5.20 checkout + the repo's patches (base, hisparse)
  series-write write TREE_DIGEST.<arm> of a series arm (arc3: patches/sglang-0.5.20/arc3.series; arcforge.sglang_series)
  fingerprint  identity + tree + caches as JSON (runs and gates record it; e2e_metrics.py compare checks it)
  cache-snapshot / cache-check
               file list of the JIT caches before a start, and what appeared by /health: new flashinfer JIT
               modules that AOT should have covered, and (REQUIRE_WARM) any new file at all

  python g4_runtime.py identity --arm hisparse            # with /content/v0520-hs/bin/python
  python g4_runtime.py tree --arm hisparse [--pristine /content/rt/sglang-v0520]
  python g4_runtime.py tree-write --pristine <v0.5.20 checkout> --out patches/sglang-0.5.20/TREE_DIGEST
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

#: sys.path as the interpreter set it up (.pth order), before any import below can append to it (torch,
#: tilelang and the cutlass DSL do)
SYS_PATH_AT_START = tuple(sys.path)
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))  # the tree digest is shared with the Kaggle notebook (arcforge.sglang_tree)
from arcforge import sglang_tree  # noqa: E402
from arcforge import sglang_series  # noqa: E402  series arms (arc3): own series / digest / manifest files
from arcforge import sglang_lock  # noqa: E402  fork arms: a deploy/sglang.lock.json target installed from its tarball

PATCHES = REPO / "patches" / "sglang-0.5.20"
SPEC = HERE / "g4_runtime_spec.json"
TREE_DIGEST = PATCHES / "TREE_DIGEST"
RECORD = Path("/content/rt/RUNTIME.json")
ARMS = ("base", "hisparse")
#: every arm a VM can install and check: ARMS (TREE_DIGEST, globbed patches) + the series arms (arcforge.sglang_series:
#: <arm>.series, TREE_DIGEST.<arm>, written by series-write) + the sglang-arc-2 lock targets (deploy/sglang.lock.json,
#: installed from their tarballs); tree-write still writes ARMS only
ALL_ARMS = ARMS + tuple(sglang_series.SERIES) + tuple(sglang_lock.read_lock()["targets"])
GENERATED = sglang_tree.GENERATED
#: distributions / modules every runtime has; a source build records them, the Kaggle runtime asserts them
CORE_DISTS = ("torch", "triton", "flashinfer-python", "sglang-kernel", "tilelang", "apache-tvm-ffi", "sglang")
CORE_MODULES = ("torch", "triton", "flashinfer", "sgl_kernel", "tilelang", "tvm_ffi")
#: what only the Kaggle runtime ships (flashinfer AOT set, FA2); its modules must come from its site-packages
KAGGLE_DISTS = ("flash-attn", "nvidia-cutlass-dsl")
KAGGLE_MODULES = ("flashinfer_jit_cache", "flash_attn")


def fail(problems: list[str], what: str) -> None:
    if problems:
        raise SystemExit(f"{what}: {len(problems)} problem(s)\n  " + "\n  ".join(problems))


sha256_file = sglang_tree.sha256_file


def load_record(path: Path = RECORD) -> dict:
    if not path.is_file():
        raise SystemExit(f"{path} is missing: this VM was not brought up by g4_bootstrap.sh")
    rec = json.loads(path.read_text())
    for key in ("kind", "arms"):
        if key not in rec:
            raise SystemExit(f"{path} has no {key!r}")
    if rec["kind"] not in ("kaggle-dataset", "source-build"):
        raise SystemExit(f"{path}: unknown kind {rec['kind']!r}")
    return rec


def arm_record(rec: dict, arm: str) -> dict:
    if arm not in rec["arms"]:
        raise SystemExit(f"RUNTIME.json has no arm {arm!r} (has {sorted(rec['arms'])})")
    return rec["arms"][arm]


# ---------------------------------------------------------------- the whole sglang package
# The digest itself lives in arcforge.sglang_tree (the Kaggle notebook asserts the same one); these wrappers keep
# this CLI's contract: a bad input is a SystemExit with the reason.

def package_manifest(package_root: Path) -> dict[str, str]:
    """relpath -> sha256 over every file under the sglang package directory but caches and generated files."""
    try:
        return sglang_tree.package_manifest(package_root)
    except ValueError as error:
        raise SystemExit(str(error)) from error


manifest_digest = sglang_tree.manifest_digest


def patch_set(arm: str) -> list[Path]:
    try:
        if arm in sglang_series.SERIES:
            return sglang_series.patch_set(PATCHES, arm)
        return sglang_tree.patch_set(PATCHES, arm)
    except ValueError as error:
        raise SystemExit(str(error)) from error


def patches_digest(arm: str) -> str:
    return sglang_tree.patches_digest(patch_set(arm))


def read_tree_digest(path: Path = TREE_DIGEST) -> dict[str, dict]:
    try:
        return sglang_tree.read_tree_digest(path)
    except ValueError as error:
        raise SystemExit(str(error)) from error


def tree_check(package_root: Path, arm: str, pristine: Path | None) -> dict:
    if arm in sglang_lock.read_lock()["targets"]:  # a sglang-arc-2 tree (g4_runtime_install.sh G4_TARGETS)
        try:
            t = sglang_lock.check_tree(arm, package_root)
        except RuntimeError as error:
            raise SystemExit(str(error)) from error
        # check_tree accepted the tree under one of the lock's two digest conventions; report the git-tree one
        return {"arm": arm, "lock_target": arm, "commit": t["commit"], "digest": t["digest_symlink"], "files": t["files"]}
    if arm in sglang_series.SERIES:
        try:
            want = sglang_series.read_row(PATCHES, arm)
        except (OSError, ValueError) as error:
            raise SystemExit(f"{arm}: {error} (write it with `g4_runtime.py series-write --arm {arm} ...`)") from error
    else:
        want = read_tree_digest()[arm]
    if patches_digest(arm) != want["patches"]:
        raise SystemExit(f"TREE_DIGEST was written for another set of {arm} patches: regenerate it with "
                         "`python work/qsa_hisparse/g4_runtime.py tree-write --pristine <v0.5.20 checkout>`")
    got = package_manifest(package_root)
    digest = manifest_digest(got)
    if digest == want["digest"] and len(got) == want["files"]:
        return {"arm": arm, "digest": digest, "files": len(got)}
    problems = [f"digest {digest[:12]} ({len(got)} files) != TREE_DIGEST {want['digest'][:12]} ({want['files']} files)"]
    if pristine is not None:  # rebuild main's tree to name the files that differ
        with tempfile.TemporaryDirectory() as tmp:
            ref = package_manifest(build_tree(pristine, arm, Path(tmp) / arm))
        problems += [f"extra    {r}" for r in sorted(set(got) - set(ref))]
        problems += [f"missing  {r}" for r in sorted(set(ref) - set(got))]
        problems += [f"changed  {r}" for r in sorted(set(got) & set(ref)) if got[r] != ref[r]]
        if manifest_digest(ref) != want["digest"]:
            problems.append(f"the rebuilt reference itself is {manifest_digest(ref)[:12]}, not TREE_DIGEST's")
    else:
        problems.append("pass --pristine <v0.5.20 checkout> to list the files that differ")
    fail(problems, f"installed {arm} sglang package != main's tree")
    raise AssertionError("unreachable")


def build_tree(pristine: Path, arm: str, dest: Path) -> Path:
    """dest <- pristine v0.5.20 checkout + the arm's patches (git apply, in order)."""
    subprocess.run(["git", "-c", "advice.detachedHead=false", "clone", "-q", "--no-hardlinks", str(pristine),
                    str(dest)], check=True)
    ref = re.search(r"^REF=([0-9a-f]{40})", (PATCHES / "BASE").read_text(), re.M).group(1)
    head = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()
    if head != ref:
        raise SystemExit(f"{pristine} HEAD is {head}, BASE says {ref}")
    for p in patch_set(arm):
        subprocess.run(["git", "-C", str(dest), "apply", str(p)], check=True)
    return dest / "python" / "sglang"


def tree_write(pristine: Path, out: Path) -> None:
    lines = ["# whole sglang package per arm (every file but __pycache__ and _version.py), from g4_runtime.py tree-write;",
             "# patches= is sha256 over the arm's patch files: a patch change without a rewrite fails the CPU test"]
    with tempfile.TemporaryDirectory() as tmp:
        for arm in ARMS:
            pkg = build_tree(pristine, arm, Path(tmp) / arm)
            entries = package_manifest(pkg)
            lines.append(f"{arm} {manifest_digest(entries)} files={len(entries)} patches={patches_digest(arm)}")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[2:]))


def series_write(pristine: Path, arm: str) -> None:
    """TREE_DIGEST.<arm> for a series arm, built from a pristine v0.5.20 checkout + its series (TREE_DIGEST itself is
    not touched)."""
    with tempfile.TemporaryDirectory() as tmp:
        pkg = build_tree(pristine, arm, Path(tmp) / arm)
        entries = package_manifest(pkg)
        text = sglang_series.row_text(arm, manifest_digest(entries), len(entries), patches_digest(arm))
    sglang_series.digest_path(PATCHES, arm).write_text(text)
    print(text.splitlines()[-1])


# ---------------------------------------------------------------- runtime identity (run with the arm's python)

def observe(rec: dict, arm: str) -> dict:
    """What this interpreter actually imports and which nvcc sglang's JIT would call."""
    a = arm_record(rec, arm)
    kaggle = rec["kind"] == "kaggle-dataset"
    obs = {"python": sys.version.split()[0], "prefix": sys.prefix, "versions": {}, "modules": {}}
    for dist in CORE_DISTS + (KAGGLE_DISTS if kaggle else ()):
        obs["versions"][dist] = importlib.metadata.version(dist)
    for name in CORE_MODULES + (KAGGLE_MODULES if kaggle else ()) + ("sglang",):
        mod = importlib.import_module(name)
        obs["modules"][name] = str(Path(mod.__file__).resolve())
    if kaggle:
        import flashinfer_jit_cache
        obs["versions"]["flashinfer-jit-cache"] = flashinfer_jit_cache.__version__
    import sglang
    obs["sglang_version"] = sglang.__version__
    from flashinfer.jit import env as fi_env
    obs["flashinfer_aot_dir"] = str(Path(fi_env.FLASHINFER_AOT_DIR).resolve())
    from sglang.kernels.jit.utils.compile import toolchain
    nvcc = Path(toolchain.device_compiler_path())
    obs["cuda_home_env"] = os.environ.get("CUDA_HOME")
    obs["jit_nvcc"] = str(nvcc.resolve())
    done = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, check=True)
    obs["nvcc_release"] = re.search(r"release [\d.]+, V([\d.]+)", done.stdout).group(1)
    obs["arm_cuda_home"] = str(Path(a["cuda_home"]).resolve())
    if "test_tools" in rec:  # g4_testtools.sh: gate tooling, an explicit addition after the runtime on sys.path
        tools = rec["test_tools"]
        obs["test_tools"] = {"dir": tools["dir"], "last_on_sys_path": SYS_PATH_AT_START[-1] == tools["dir"],
                             "versions": {n: importlib.metadata.version(n) for n in tools["pins"]},
                             "modules": {n: str(Path(importlib.import_module(n).__file__).resolve())
                                         for n in tools["pins"]}}
    return obs


def identity(rec: dict, arm: str) -> dict:
    a = arm_record(rec, arm)
    obs = observe(rec, arm)
    p: list[str] = []
    if Path(sys.prefix).resolve() != Path(a["venv"]).resolve():
        p.append(f"running under {sys.prefix}, not the {arm} venv {a['venv']}")
    if obs["cuda_home_env"] is None or Path(obs["cuda_home_env"]).resolve() != Path(a["cuda_home"]).resolve():
        p.append(f"CUDA_HOME={obs['cuda_home_env']!r}, the {arm} arm's is {a['cuda_home']}")
    want_nvcc = (Path(a["cuda_home"]) / "bin" / "nvcc").resolve()
    if Path(obs["jit_nvcc"]) != want_nvcc:
        p.append(f"sglang's JIT would call {obs['jit_nvcc']}, not {want_nvcc}")
    src_init = (Path(a["src"]) / "python" / "sglang" / "__init__.py").resolve()
    if Path(obs["modules"]["sglang"]) != src_init:
        p.append(f"sglang imports from {obs['modules']['sglang']}, not the arm's source {src_init}")
    if obs["sglang_version"] != "0.5.20":
        p.append(f"sglang.__version__ {obs['sglang_version']}")
    if rec["kind"] == "kaggle-dataset":
        spec = json.loads(SPEC.read_text())
        root = Path(rec["root"])
        site = (root / spec["layout"]["site_packages"]).resolve()
        for key in ("dataset", "version", "archive", "archive_sha256"):
            if rec["dataset"][key] != spec[key]:
                p.append(f"dataset {key} {rec['dataset'][key]!r} != spec {spec[key]!r}")
        prov = root / "runtime-provenance.json"
        if sha256_file(prov) != spec["provenance_sha256"]:
            p.append(f"{prov} sha256 {sha256_file(prov)[:12]} != spec {spec['provenance_sha256'][:12]}")
        record = json.loads(prov.read_text())
        if record["kaggle_dataset"] != spec["dataset"]:
            p.append(f"provenance kaggle_dataset {record['kaggle_dataset']!r}")
        if (root / spec["layout"]["sglang_source"]).exists():
            p.append(f"the dataset's own sglang source {root / spec['layout']['sglang_source']} still exists")
        if (site / "sglang").exists():
            p.append(f"an sglang package inside the runtime site-packages: {site / 'sglang'}")
        if obs["python"] != spec["python"]:
            p.append(f"python {obs['python']} != spec {spec['python']}")
        for dist, want in spec["versions"].items():
            if obs["versions"].get(dist) != want:
                p.append(f"{dist} {obs['versions'].get(dist)} != spec {want}")
        for name in CORE_MODULES + KAGGLE_MODULES:
            if not Path(obs["modules"][name]).is_relative_to(site):
                p.append(f"{name} imports from {obs['modules'][name]}, outside the runtime {site}")
        aot = (site / "flashinfer_jit_cache" / "jit_cache").resolve()
        if Path(obs["flashinfer_aot_dir"]) != aot:
            p.append(f"flashinfer AOT dir {obs['flashinfer_aot_dir']} != {aot}")
        for rel, meta in record["aot"]["files"].items():
            path = site / rel
            if not path.is_file() or path.stat().st_size != meta["bytes"] or sha256_file(path) != meta["sha256"]:
                p.append(f"AOT file {rel} differs from the provenance")
        if obs["nvcc_release"] != spec["nvcc"]:
            p.append(f"nvcc {obs['nvcc_release']} != spec {spec['nvcc']}")
        if "test_tools" in rec:
            tt = obs["test_tools"]
            if rec["test_tools"]["pins"] != spec["test_tools"]:
                p.append(f"RUNTIME.json test_tools {rec['test_tools']['pins']} != spec {spec['test_tools']}")
            if tt["versions"] != spec["test_tools"]:
                p.append(f"test tools import as {tt['versions']}, spec {spec['test_tools']}")
            if not tt["last_on_sys_path"]:
                p.append(f"{tt['dir']} is not the last sys.path entry: it could shadow the runtime")
            for name, path in tt["modules"].items():
                if not Path(path).is_relative_to(Path(tt["dir"]).resolve()):
                    p.append(f"{name} imports from {path}, not {tt['dir']}")
    fail(p, f"{arm} runtime identity")
    return obs


# ---------------------------------------------------------------- caches

def cache_files(dirs: list[Path]) -> dict[str, int]:
    out = {}
    for d in dirs:
        if not d.is_dir():
            raise SystemExit(f"cache dir {d} does not exist (create it before the snapshot)")
        for path in d.rglob("*"):
            if path.is_file() and not path.is_symlink():
                out[str(path)] = path.stat().st_size
    return out


def cache_digest(dirs: list[Path]) -> dict:
    files = cache_files(dirs)
    h = hashlib.sha256()
    for name in sorted(files):
        h.update(f"{name}\0{files[name]}\0{sha256_file(Path(name))}\n".encode())
    return {"dirs": [str(d) for d in dirs], "files": len(files), "bytes": sum(files.values()), "digest": h.hexdigest()}


def aot_modules(rec: dict) -> list[str]:
    """Module names the runtime's flashinfer_jit_cache ships (empty for a source build: it has no AOT set)."""
    if rec["kind"] != "kaggle-dataset":
        return []
    prov = json.loads((Path(rec["root"]) / "runtime-provenance.json").read_text())
    return sorted(Path(m).parent.name for m in prov["aot"]["modules"])


#: what a new cache file says was built. sglang's JIT loader logs nothing at INFO for a fresh build (only a
#: debug-level "Rebuilding ... changed" for a stale one), so these file patterns are the only record of it.
BUILD_PATTERNS = {
    "sgl_jit": re.compile(r"/jit/[^/]+/([^/]+)/build-[^/]+/deps-[^/]+/\1\.so$"),
    "triton": re.compile(r"/triton/[^/]+/([^/]+)\.cubin$"),
    "flashinfer_jit": re.compile(r"/cached_ops/([^/]+)/\1\.so$"),
    "tilelang": re.compile(r"/\.tilelang/.*?/([^/]+)$"),
}


def cache_check(snapshot: Path, aot: list[str]) -> dict:
    """What the caches gained since the snapshot: every new file, and what was built (module / kernel names)."""
    snap = json.loads(snapshot.read_text())
    before = set(snap["files"])
    new = sorted(set(cache_files([Path(d) for d in snap["dirs"]])) - before)
    builds = {kind: sorted({m.group(1) for f in new for m in [pat.search(f)] if m})
              for kind, pat in BUILD_PATTERNS.items()}
    jit = sorted({m.group(1) for f in new for m in [re.search(r"/cached_ops/([^/]+)/", f)] if m})
    return {"dirs": snap["dirs"], "new_files": len(new), "new_sample": new[:20], "builds": builds,
            "flashinfer_jit_modules": jit, "aot_modules_rebuilt": sorted(set(jit) & set(aot))}


# ---------------------------------------------------------------- fingerprint

def runtime_identity(rec: dict, arm: str) -> dict:
    """The part of a fingerprint that decides whether two runs are comparable (no caches, no timestamps)."""
    a = arm_record(rec, arm)
    obs = observe(rec, arm)
    ident = {"kind": rec["kind"], "arm": arm, "python": obs["python"], "versions": obs["versions"],
             "nvcc": obs["nvcc_release"],
             "sglang_tree": manifest_digest(package_manifest(Path(a["src"]) / "python" / "sglang"))}
    if rec["kind"] == "kaggle-dataset":
        ident["dataset"] = f"{rec['dataset']['dataset']}@v{rec['dataset']['version']}"
        ident["archive_sha256"] = rec["dataset"]["archive_sha256"]
        ident["provenance_sha256"] = sha256_file(Path(rec["root"]) / "runtime-provenance.json")
    if arm == "hisparse":
        man = PATCHES / "experimental-hisparse" / "SOURCE_SHA256"
        ident["hisparse_source_tree"] = hashlib.sha256(man.read_bytes()).hexdigest()
    if "test_tools" in obs:
        ident["test_tools"] = obs["test_tools"]["versions"]
    return ident


def fingerprint(rec: dict, arm: str, caches: list[Path]) -> dict:
    return {"identity": runtime_identity(rec, arm), "cache": cache_digest(caches)}


def diff_identity(a: dict, b: dict, prefix: str = "") -> list[str]:
    out = []
    for key in sorted(set(a) | set(b)):
        x, y = a.get(key, "<absent>"), b.get(key, "<absent>")
        if isinstance(x, dict) and isinstance(y, dict):
            out += diff_identity(x, y, f"{prefix}{key}.")
        elif x != y:
            out.append(f"{prefix}{key}: {x} | {y}")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("identity")
    s.add_argument("--arm", choices=ALL_ARMS, required=True)
    s.add_argument("--record", type=Path, default=RECORD)
    s = sub.add_parser("tree")
    s.add_argument("--arm", choices=ALL_ARMS, required=True)
    s.add_argument("--record", type=Path, default=RECORD)
    s.add_argument("--package-root", type=Path, help="default: the arm's source from RUNTIME.json")
    s.add_argument("--pristine", type=Path, help="v0.5.20 checkout: on a mismatch, list extra/missing/changed files")
    s = sub.add_parser("tree-write")
    s.add_argument("--pristine", type=Path, required=True)
    s.add_argument("--out", type=Path, default=TREE_DIGEST)
    s = sub.add_parser("series-write", help="TREE_DIGEST.<arm> for a series arm (arc3)")
    s.add_argument("--arm", choices=tuple(sglang_series.SERIES), required=True)
    s.add_argument("--pristine", type=Path, required=True)
    s = sub.add_parser("fingerprint")
    s.add_argument("--arm", choices=ALL_ARMS, required=True)
    s.add_argument("--record", type=Path, default=RECORD)
    s.add_argument("--cache", type=Path, action="append", required=True)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("ident", help="the identity part of the fingerprint only (per-tree cache sets use it)")
    s.add_argument("--arm", choices=ALL_ARMS, required=True)
    s.add_argument("--record", type=Path, default=RECORD)
    s.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("ident-compare", help="exit 1 listing every field where two identity files differ")
    s.add_argument("want", type=Path)
    s.add_argument("got", type=Path)
    s = sub.add_parser("cache-snapshot")
    s.add_argument("--cache", type=Path, action="append", required=True)
    s.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    if a.cmd == "identity":
        print(json.dumps(identity(load_record(a.record), a.arm)))
        print(f"IDENTITY_OK {a.arm}")
    elif a.cmd == "tree":
        root = a.package_root or Path(arm_record(load_record(a.record), a.arm)["src"]) / "python" / "sglang"
        r = tree_check(root, a.arm, a.pristine)
        print(f"TREE_OK {a.arm} {r['digest']} files={r['files']}")
    elif a.cmd == "tree-write":
        tree_write(a.pristine, a.out)
    elif a.cmd == "series-write":
        series_write(a.pristine, a.arm)
    elif a.cmd == "fingerprint":
        fp = fingerprint(load_record(a.record), a.arm, a.cache)
        a.out.write_text(json.dumps(fp, indent=1) + "\n")
        print(json.dumps(fp["identity"]))
    elif a.cmd == "ident":
        ident = runtime_identity(load_record(a.record), a.arm)
        a.out.write_text(json.dumps(ident, indent=1, sort_keys=True) + "\n")
    elif a.cmd == "ident-compare":
        fail(diff_identity(json.loads(a.want.read_text()), json.loads(a.got.read_text())),
             f"runtime identity {a.want} != {a.got}")
        print("IDENT_SAME")
    elif a.cmd == "cache-snapshot":
        a.out.write_text(json.dumps({"dirs": [str(d) for d in a.cache], "files": cache_files(a.cache)}) + "\n")


if __name__ == "__main__":
    main()
