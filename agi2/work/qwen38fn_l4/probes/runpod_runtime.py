"""Pinned Kaggle runtimes on a fresh remote: fetch, install, fingerprint, and SM89 environment.

No GPU work during install. The baseline reads only literal asset/patch metadata from the
hash-pinned notebook; no notebook or embedded game harness is executed. All paths are explicit.
"""
from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from arcforge import sglang_tree  # noqa: E402

NOTEBOOK_SHA = "4738fa3a764817442f94049823a9ee24a21ef3e205d78eeb25ce0cd5b1b84ef2"
BASE_ARCHIVE = "sglang-qwen4-73a255206f-sm120-cp312.tar.zst"
BASE_SHA = "e7c03ad7300669fab9d96f9d70080671b0df198f7ab40b99e0adff9162309481"
NEW_ARCHIVE = "sglang-0520-94602c9-sm120-cp312.tar.zst"
NEW_SHA = "cbf427cb3677313cf060ec4e4c4e53d2cf7a9dbf06ac884ce31e93447345c1b3"
NEW_BYTES = 3496285060
NEW_PROVENANCE_SHA = "812fd475d9056e54c107272d93960c8aa2770c8e4945b476846811e30b908680"
NEW_REF = "94602c9c2b7cbdb8efd5c52802dac6a1c180089e"
sha256_file = sglang_tree.sha256_file


def write_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def notebook_metadata(path: Path, expected_sha: str = NOTEBOOK_SHA) -> dict:
    if sha256_file(path) != expected_sha:
        raise RuntimeError("notebook SHA256 mismatch; use the audited Kaggle source")
    wanted = {"MANIFEST", "PATCH_ORDER", "PATCH_SHA256", "PATCH_PAYLOADS"}
    values = {}
    for cell in json.loads(path.read_text())["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = cell["source"]
        source = "".join(source) if isinstance(source, list) else source
        if not re.search(r"^(MANIFEST|PATCH_ORDER|PATCH_SHA256|PATCH_PAYLOADS)\s*=", source, re.MULTILINE):
            continue  # skip notebook magics and all embedded harness cells
        for node in ast.parse(source).body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if isinstance(target, ast.Name) and target.id in wanted:
                    if target.id in values:
                        raise RuntimeError(f"duplicate notebook metadata: {target.id}")
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != wanted:
        raise RuntimeError(f"missing notebook metadata: {wanted - set(values)}")
    if values["MANIFEST"]["runtime_name"] != BASE_ARCHIVE or values["MANIFEST"]["runtime_sha256"] != BASE_SHA:
        raise RuntimeError("notebook runtime identity drift")
    order = values["PATCH_ORDER"]
    if len(order) != 9 or not order[0].startswith("0036-") or not order[-1].startswith("0044-"):
        raise RuntimeError("notebook patch order drift")
    if set(order) != set(values["PATCH_SHA256"]) or set(order) != set(values["PATCH_PAYLOADS"]):
        raise RuntimeError("notebook patch metadata differs")
    for name in order:
        safe_relative(name)
        payload_sha = hashlib.sha256(values["PATCH_PAYLOADS"][name].encode()).hexdigest()
        if payload_sha != values["PATCH_SHA256"][name]:
            raise RuntimeError(f"patch payload SHA mismatch: {name}")
    return values


def safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or path == PurePosixPath(".") or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"unsafe manifest path: {name!r}")
    return path


def signed_manifest(key: Path, source: str, model: bool, version: int | None, out: Path) -> None:
    """Use an explicitly authorized credential file; output contains expiring URLs only."""
    credential = json.loads(key.read_text())
    auth = base64.b64encode(f"{credential['username']}:{credential['key']}".encode()).decode()
    headers = {"Authorization": f"Basic {auth}", "User-Agent": "arc-forge-runpod"}
    api = "https://www.kaggle.com/api/v1"
    if model:
        listing = f"{api}/models/{source}/files"
        list_key = "files"
    else:
        listing = f"{api}/datasets/list/{source}"
        list_key = "datasetFiles"
    token = None
    files = []
    while True:
        params = {"pageSize": 200}
        if version:
            params["datasetVersionNumber"] = version
        if token:
            params["pageToken"] = token
        request = urllib.request.Request(listing + "?" + urllib.parse.urlencode(params), headers=headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.load(response)
        files.extend(body[list_key])
        token = body.get("nextPageToken")
        if not token:
            break
    if not files:
        raise RuntimeError("Kaggle asset has no files")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, hdrs, newurl):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    rows = []
    for row in sorted(files, key=lambda r: r["name"]):
        name = row["name"]
        safe_relative(name)
        size = row["size"] if model else row["totalBytes"]
        if not isinstance(size, int) or size <= 0:
            raise RuntimeError(f"invalid Kaggle file size: {name}")
        quoted = urllib.parse.quote(name, safe="")
        url = f"{api}/models/{source}/download/{quoted}" if model else f"{api}/datasets/download/{source}/{quoted}"
        if version:
            url += "?" + urllib.parse.urlencode({"datasetVersionNumber": version})
        request = urllib.request.Request(url, headers=headers)
        try:
            with opener.open(request, timeout=60) as response:
                raise RuntimeError(f"expected signed redirect for {name}, got HTTP {response.status}")
        except urllib.error.HTTPError as exc:
            if exc.code != 302 or not exc.headers.get("Location"):
                raise RuntimeError(f"Kaggle signed redirect failed: {name}, HTTP {exc.code}") from None
            signed_url = exc.headers["Location"]
        if not signed_url.startswith("https://"):
            raise RuntimeError("Kaggle returned non-HTTPS signed URL")
        rows.append({"name": name, "size": size, "url": signed_url})
        print(f"SIGN_OK {name} {size}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.fchmod(fd, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump({"files": rows}, handle)


def fetch_manifest(manifest: Path, dest: Path, workers: int) -> None:
    """Resume signed URLs; never include URLs or their query strings in output/errors."""
    rows = json.loads(manifest.read_text())["files"]
    if not rows or workers < 1:
        raise ValueError("empty manifest or invalid worker count")
    names = set()
    for row in rows:
        safe_relative(row["name"])
        if row["name"] in names or not isinstance(row["size"], int) or row["size"] <= 0:
            raise ValueError("duplicate manifest name or invalid size")
        if not row["url"].startswith("https://"):
            raise ValueError("manifest requires HTTPS URLs")
        names.add(row["name"])
    dest.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        path = dest / row["name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        # Resolve existing symlinks too: a reused destination cannot escape its root.
        if not path.resolve().is_relative_to(dest.resolve()):
            raise ValueError(f"manifest destination escapes root: {row['name']}")
        for attempt in range(6):
            have = path.stat().st_size if path.exists() else 0
            if have == row["size"]:
                return row["name"], have
            if have > row["size"]:
                raise RuntimeError(f"oversized existing asset: {row['name']}")
            headers = {"Range": f"bytes={have}-"} if have else {}
            try:
                req = urllib.request.Request(row["url"], headers=headers)
                with urllib.request.urlopen(req, timeout=120) as response:
                    if response.status not in (200, 206):
                        raise RuntimeError("unexpected HTTP status")
                    if response.status == 206:
                        content_range = response.headers.get("Content-Range", "")
                        if not content_range.startswith(f"bytes {have}-"):
                            raise RuntimeError("invalid download range")
                    elif have:
                        have = 0  # server ignored Range: replace the incomplete file
                    with path.open("ab" if have else "wb") as out:
                        shutil.copyfileobj(response, out, 8 << 20)
                if path.stat().st_size == row["size"]:
                    return row["name"], row["size"]
            except (OSError, urllib.error.URLError, RuntimeError) as exc:
                # Exception messages can contain bearer URLs, so print only the type.
                print(f"retry {row['name']} {attempt + 1}/6: {type(exc).__name__}", flush=True)
            if attempt < 5:
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"download failed or size mismatch: {row['name']}")

    with ThreadPoolExecutor(workers) as pool:
        for name, size in pool.map(fetch, sorted(rows, key=lambda r: -r["size"])):
            print(f"FETCH_OK {name} {size}", flush=True)
    write_json(dest / "download_sizes.json", {"files": [{"name": r["name"], "size": r["size"]} for r in rows]})
    manifest.unlink()


# model reference -> config.json SHA256: the checkpoints the L4 notebooks may serve (the REAP320 NVFP4 base and,
# l4opt 10-10, the gEAN320 packed INT3 GPTQ deployment; 10-15 its Joint QAD v2 run1 step-147 values as version 2, which
# keeps version 1's config.json -- so the config names a family, and the pinned target_sha256 tells the versions apart)
MODEL_IDENTITIES = {
    "threerabbits/qwen38fn-nvfp4-reap320-arccode/transformers/default/1":
        "655d013b7d28371b4199f2ae309402ebf4c72701ac6ee6b3f4c054dafba4f8b7",
    "threerabbits/qwen38fn-gean320-int3-gptq/transformers/default/1":
        "8b2a29dbe7f7922238b447d689bc574de9b34ef7d3c1e16adae7ea92d4b5c27a",
    "threerabbits/qwen38fn-gean320-int3-gptq/transformers/default/2":
        "8b2a29dbe7f7922238b447d689bc574de9b34ef7d3c1e16adae7ea92d4b5c27a",
    # 10-24: version 3 = student-trajectory QAD R (qadmix-R-v1000); config.json identical to v1 / v2
    "threerabbits/qwen38fn-gean320-int3-gptq/transformers/default/3":
        "8b2a29dbe7f7922238b447d689bc574de9b34ef7d3c1e16adae7ea92d4b5c27a",
}


def verify_model(model: Path, metadata: dict) -> dict:
    """Sizes and hashes of metadata["MANIFEST"]; metadata["IDENTITY"] (default REAP320) names the checkpoint, one of
    MODEL_IDENTITIES, and a pinned config.json SHA that MODEL_IDENTITIES knows must be the one it records for that checkpoint."""
    manifest = metadata["MANIFEST"]
    identity = metadata.get("IDENTITY", "threerabbits/qwen38fn-nvfp4-reap320-arccode/transformers/default/1")
    if identity not in MODEL_IDENTITIES:
        raise RuntimeError(f"model identity {identity!r} is none of the audited checkpoints (MODEL_IDENTITIES)")
    pinned = manifest["target_sha256"].get("config.json")
    if pinned in MODEL_IDENTITIES.values() and pinned != MODEL_IDENTITIES[identity]:
        raise RuntimeError(f"pinned config.json {pinned} is not {identity}'s {MODEL_IDENTITIES[identity]}")
    for name, size in manifest["target_sizes"].items():
        path = model / safe_relative(name)
        if not path.is_file() or path.stat().st_size != size:
            raise RuntimeError(f"model missing or wrong size: {name}")
    for name, expected in manifest["target_sha256"].items():
        if sha256_file(model / name) != expected:
            raise RuntimeError(f"model SHA mismatch: {name}")
    return {"model": str(model.resolve()), "identity": identity,
            "files_sized": len(manifest["target_sizes"]), "files_hashed": len(manifest["target_sha256"]),
            "target_sizes": manifest["target_sizes"], "target_sha256": manifest["target_sha256"]}


def apply_patch(tree: Path, patch: Path, allow_applied: bool = False) -> None:
    args = ["git", "apply", "--whitespace=nowarn"]
    check = subprocess.run([*args, "--check", str(patch.resolve())], cwd=tree, capture_output=True, text=True)
    if check.returncode:
        reverse = subprocess.run([*args, "--reverse", "--check", str(patch.resolve())], cwd=tree, capture_output=True)
        if allow_applied and reverse.returncode == 0:
            return
        raise RuntimeError(f"patch source drift: {patch.name}\n{check.stderr}")
    subprocess.run([*args, str(patch.resolve())], cwd=tree, check=True)


def apply_mtp_topology(package: Path) -> None:
    """Exact semantic backport from the audited notebook's cell 13."""
    source = package / "python/sglang/srt/configs/model_config.py"
    text = source.read_text()
    marker = 'self.hf_config.architectures[0] = "Qwen4ExpForCausalLMMTP"'
    if marker not in text:
        anchor = '        if is_draft_model and self.hf_config.architectures[0] == "Qwen3MoeForCausalLM":\n'
        if text.count(anchor) != 1:
            raise RuntimeError("Qwen3-MoE draft anchor drifted")
        block = '''        if (
            is_draft_model
            and self.hf_config.architectures[0]
            == "Qwen4ExpForConditionalGeneration"
        ):
            # The target and draft initially share the HF config. Isolate the
            # draft before reducing its topology to the one-layer MTP head.
            from copy import deepcopy

            self.hf_config = deepcopy(self.hf_config)
            self.hf_text_config = get_hf_text_config(self.hf_config)
            self.hf_config.architectures[0] = "Qwen4ExpForCausalLMMTP"
            text_config = self.hf_text_config
            text_config.num_nextn_predict_layers = 1
            text_config.num_hidden_layers = 1
            text_config.layer_types = ["full_attention"]
            text_config.full_attention_interval = 1

'''
        source.write_text(text.replace(anchor, block + anchor))
    patched = source.read_text()
    if marker not in patched or 'text_config.num_hidden_layers = 1' not in patched:
        raise RuntimeError("MTP draft topology fix missing")
    compile(patched, str(source), "exec")


def source_identity(source: Path) -> dict:
    entries = sglang_tree.package_manifest(source / "sglang")
    return {"source": str(source.resolve()), "digest": sglang_tree.manifest_digest(entries), "files": len(entries)}


def check_runtime(root: Path) -> dict:
    rec = json.loads((root / "RUNTIME.json").read_text())
    if rec["arch"] != "8.9":
        raise RuntimeError("runtime arch is not SM89")
    for arm, identity in rec["arms"].items():
        actual = source_identity(Path(identity["source"]))
        if actual != {k: identity[k] for k in actual}:
            raise RuntimeError(f"runtime source drift: {arm}")
    return rec


def extract_archive(archive: Path, root: Path, expected_sha: str) -> None:
    if sha256_file(archive) != expected_sha:
        raise RuntimeError("runtime archive SHA mismatch")
    root.mkdir(parents=True, exist_ok=True)
    # Shared Runpod volumes reject restoring the packager's UID/GID even as container root.
    subprocess.run(["tar", "--no-same-owner", "--zstd", "-xf", str(archive.resolve()),
                    "-C", str(root.resolve())], check=True)
    if not (root / "package/python/sglang").is_dir():
        raise RuntimeError("runtime archive layout mismatch")


def install_baseline(archive: Path, notebook: Path, root: Path) -> dict:
    metadata = notebook_metadata(notebook)
    if (root / "RUNTIME.json").exists():
        rec = check_runtime(root)
        if rec["archive_sha256"] != BASE_SHA or rec["notebook_sha256"] != NOTEBOOK_SHA:
            raise RuntimeError("existing baseline identity mismatch")
        return rec
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"incomplete install exists; inspect it before retrying: {root}")
    extract_archive(archive, root, BASE_SHA)
    package = root / "package"
    patch_dir = root / "notebook-patches"
    patch_dir.mkdir()
    for name in metadata["PATCH_ORDER"]:
        patch = patch_dir / name
        patch.write_text(metadata["PATCH_PAYLOADS"][name])
        if name.startswith("0044-"):
            apply_mtp_topology(package)
        else:
            apply_patch(package, patch, allow_applied=True)
    rec = {"kind": "notebook-baseline", "arch": "8.9", "package": str(package.resolve()),
           "archive_sha256": BASE_SHA, "notebook_sha256": NOTEBOOK_SHA,
           "patch_sha256": metadata["PATCH_SHA256"], "target_manifest": metadata["MANIFEST"],
           "arms": {"baseline": source_identity(package / "python")}}
    write_json(root / "RUNTIME.json", rec)
    return check_runtime(root)


def install_0520(archive: Path, root: Path, patch_dir: Path) -> dict:
    """Independent base/hisparse source trees; TREE_DIGEST must match the current patch set."""
    expected = sglang_tree.read_tree_digest(patch_dir / "TREE_DIGEST")
    sets = {arm: sglang_tree.patch_set(patch_dir, arm) for arm in ("base", "hisparse")}
    for arm, patches in sets.items():
        if sglang_tree.patches_digest(patches) != expected[arm]["patches"]:
            raise RuntimeError(f"{arm}: stale TREE_DIGEST for current patches")
    if (root / "RUNTIME.json").exists():
        rec = check_runtime(root)
        if rec["archive_sha256"] != NEW_SHA:
            raise RuntimeError("existing 0520 archive identity mismatch")
        for arm in sets:
            if rec["arms"][arm]["digest"] != expected[arm]["digest"]:
                raise RuntimeError(f"existing {arm} tree differs from current patches; use a new runtime dir")
        return rec
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"incomplete install exists; inspect it before retrying: {root}")
    if archive.stat().st_size != NEW_BYTES:
        raise RuntimeError("0520 runtime archive size mismatch")
    extract_archive(archive, root, NEW_SHA)
    package = root / "package"
    if sha256_file(package / "runtime-provenance.json") != NEW_PROVENANCE_SHA:
        raise RuntimeError("0520 runtime provenance SHA mismatch")
    pristine = root / "pristine"
    subprocess.run(["git", "clone", "--depth", "1", "--branch", "v0.5.20",
                    "https://github.com/sgl-project/sglang", str(pristine)], check=True)
    ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=pristine, text=True).strip()
    if ref != NEW_REF:
        raise RuntimeError("upstream v0.5.20 ref mismatch")
    arms = {}
    for arm, patches in sets.items():
        tree = root / arm
        subprocess.run(["git", "clone", "--local", str(pristine), str(tree)], check=True)
        for patch in patches:
            apply_patch(tree, patch)
        (tree / "python/sglang/_version.py").write_text('__version__ = "0.5.20"\n__version_tuple__ = ("0", "5", "20")\n')
        identity = source_identity(tree / "python")
        if identity["digest"] != expected[arm]["digest"] or identity["files"] != expected[arm]["files"]:
            raise RuntimeError(f"{arm}: installed source differs from TREE_DIGEST")
        arms[arm] = {**identity, "patches_digest": expected[arm]["patches"]}
    rec = {"kind": "0520-source-overlay", "arch": "8.9", "package": str(package.resolve()),
           "archive_sha256": NEW_SHA, "provenance_sha256": NEW_PROVENANCE_SHA,
           "base_ref": NEW_REF, "arms": arms, "distributed_gpu_validation": "pending"}
    write_json(root / "RUNTIME.json", rec)
    return check_runtime(root)


def overlay_runtime(source_root: Path, root: Path, patch: Path, patch_dir: Path) -> dict:
    """Preserve a prior runtime; copy only its HiSparse source and reuse pinned binary libraries."""
    parent = check_runtime(source_root)
    all_patches = sglang_tree.patch_set(patch_dir, "hisparse")
    if not all_patches or all_patches[-1].resolve() != patch.resolve():
        raise RuntimeError("overlay patch must be the final canonical HiSparse patch")
    if sglang_tree.patches_digest(all_patches[:-1]) != parent["arms"]["hisparse"]["patches_digest"]:
        raise RuntimeError("prior runtime differs from the canonical patch prefix")
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("overlay requires a fresh directory; preserve failed candidates")
    old_tree = Path(parent["arms"]["hisparse"]["source"]).parent
    if root.resolve().is_relative_to(old_tree.resolve()):
        raise RuntimeError("overlay cannot be placed inside its source tree")
    root.mkdir(parents=True, exist_ok=True)
    tree = root / "hisparse"
    shutil.copytree(old_tree, tree, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    apply_patch(tree, patch)
    record = copy.deepcopy(parent)
    record["parent_runtime"] = str(source_root.resolve())
    record["overlay_patch"] = {"name": patch.name, "sha256": sha256_file(patch)}
    record["arms"]["hisparse"] = {**source_identity(tree / "python"),
                                    "patches_digest": sglang_tree.patches_digest(all_patches)}
    write_json(root / "RUNTIME.json", record)
    return check_runtime(root)


def runtime_env(root: Path, arm: str, python: str | None = None) -> tuple[str, dict, dict]:
    rec = check_runtime(root)
    identity = rec["arms"][arm]
    package = Path(rec["package"])
    site = package / "lib/python3.12/site-packages"
    if not (site / "torch/__init__.py").is_file() and (site / "site-packages/torch/__init__.py").is_file():
        site /= "site-packages"
    if not (site / "torch/__init__.py").is_file():
        raise RuntimeError("bundled Torch layout missing")
    pybin = python or os.environ.get("RUNPOD_PYTHON") or shutil.which("python3.12")
    if not pybin:
        raise RuntimeError("Python 3.12 required; set RUNPOD_PYTHON to its executable")
    version = subprocess.check_output([pybin, "-c", "import sys; print('%s.%s'%sys.version_info[:2])"], text=True).strip()
    if version != "3.12":
        raise RuntimeError(f"runtime needs Python 3.12, got {version}")
    cuda = package / "cuda-13.1"
    compat = os.environ.get("CUDA_COMPAT_DIR")
    candidates = ([Path(compat) / "libcuda.so.1"] if compat else []) + [
        Path("/usr/lib/x86_64-linux-gnu/libcuda.so.1"), Path("/usr/local/nvidia/lib64/libcuda.so.1")]
    if compat and not candidates[0].is_file():
        raise RuntimeError("CUDA_COMPAT_DIR has no libcuda.so.1")
    drivers = [p for p in candidates if p.is_file()]
    if not drivers:
        raise RuntimeError("libcuda.so.1 missing")
    shim = root / "libcuda-link"
    shim.mkdir(exist_ok=True)
    link = shim / "libcuda.so"
    desired = drivers[0].resolve()
    if link.is_symlink() and link.resolve() != desired:
        link.unlink()
    if not link.exists():
        link.symlink_to(desired)
    env = {**os.environ, "CUDA_HOME": str(cuda), "CUDA_VISIBLE_DEVICES": "0,1,2,3",
           "PATH": f"{cuda}/bin:{package}/bin:{os.environ.get('PATH', '')}",
           "PYTHONPATH": f"{identity['source']}:{site}:{site}/nvidia_cutlass_dsl/dsl_packages",
           "LIBRARY_PATH": f"{shim}:{os.environ.get('LIBRARY_PATH', '')}",
           "FLASHINFER_CUDA_ARCH_LIST": "8.9", "TORCH_CUDA_ARCH_LIST": "8.9",
           "MAX_JOBS": "4", "FLASHINFER_NVCC_THREADS": "1", "HF_HUB_OFFLINE": "1",
           "TRANSFORMERS_OFFLINE": "1", "DO_NOT_TRACK": "1", "MPLBACKEND": "Agg",
           "SGLANG_RUST_SERVER": "0", "SGLANG_RUST_BUILD_MODE": "never", "SGLANG_QWEN4_MNNVL_HC": "0",
           "PYTORCH_ALLOC_CONF": "expandable_segments:True"}
    if compat:
        env["LD_LIBRARY_PATH"] = f"{compat}:{os.environ.get('LD_LIBRARY_PATH', '')}"
    if arm == "baseline":
        env.update(SGLANG_QSA_DIRECT_DECODE="1", SGLANG_QSA_DIRECT_MIN_ROWS="1", ARC_QSA_FP4_FUSED="1")
    return pybin, env, rec


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("fetch")
    fetch.add_argument("--manifest", type=Path, required=True)
    fetch.add_argument("--dest", type=Path, required=True)
    fetch.add_argument("--workers", type=int, default=8)
    signed = commands.add_parser("manifest")
    signed.add_argument("--kaggle-key", type=Path, required=True)
    source = signed.add_mutually_exclusive_group(required=True)
    source.add_argument("--model")
    source.add_argument("--dataset")
    signed.add_argument("--version", type=int)
    signed.add_argument("--out", type=Path, required=True)
    model = commands.add_parser("verify-model")
    model.add_argument("--notebook", type=Path, required=True)
    model.add_argument("--model", type=Path, required=True)
    for command in ("install-baseline", "install-0520"):
        p = commands.add_parser(command)
        p.add_argument("--archive", type=Path, required=True)
        p.add_argument("--root", type=Path, required=True)
        if command == "install-baseline":
            p.add_argument("--notebook", type=Path, required=True)
        else:
            p.add_argument("--patch-dir", type=Path, default=REPO / "patches/sglang-0.5.20")
    for command in ("check", "env"):
        p = commands.add_parser(command)
        p.add_argument("--root", type=Path, required=True)
        if command == "env":
            p.add_argument("--arm", required=True, choices=("baseline", "base", "hisparse"))
            p.add_argument("--python")
    overlay = commands.add_parser("overlay")
    overlay.add_argument("--source-runtime", type=Path, required=True)
    overlay.add_argument("--root", type=Path, required=True)
    overlay.add_argument("--patch", type=Path, required=True)
    overlay.add_argument("--patch-dir", type=Path, default=REPO / "patches/sglang-0.5.20")
    args = parser.parse_args()
    if args.command == "fetch":
        fetch_manifest(args.manifest, args.dest, args.workers)
    elif args.command == "manifest":
        if args.model and args.version:
            parser.error("--version applies only to datasets")
        signed_manifest(args.kaggle_key, args.model or args.dataset, bool(args.model), args.version, args.out)
    elif args.command == "verify-model":
        print(json.dumps(verify_model(args.model, notebook_metadata(args.notebook)), indent=2))
    elif args.command == "install-baseline":
        print(json.dumps(install_baseline(args.archive, args.notebook, args.root), indent=2))
    elif args.command == "install-0520":
        print(json.dumps(install_0520(args.archive, args.root, args.patch_dir), indent=2))
    elif args.command == "check":
        print(json.dumps(check_runtime(args.root), indent=2))
    elif args.command == "overlay":
        print(json.dumps(overlay_runtime(args.source_runtime, args.root, args.patch, args.patch_dir), indent=2))
    else:
        pybin, env, _ = runtime_env(args.root, args.arm, args.python)
        for name in ("CUDA_HOME", "CUDA_VISIBLE_DEVICES", "PATH", "PYTHONPATH", "LIBRARY_PATH", "LD_LIBRARY_PATH",
                     "FLASHINFER_CUDA_ARCH_LIST", "TORCH_CUDA_ARCH_LIST", "MAX_JOBS", "FLASHINFER_NVCC_THREADS",
                     "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "DO_NOT_TRACK", "MPLBACKEND", "SGLANG_RUST_SERVER",
                     "SGLANG_RUST_BUILD_MODE", "SGLANG_QWEN4_MNNVL_HC", "PYTORCH_ALLOC_CONF",
                     "SGLANG_QSA_DIRECT_DECODE", "SGLANG_QSA_DIRECT_MIN_ROWS", "ARC_QSA_FP4_FUSED"):
            if name in env:
                print(f"export {name}={shlex.quote(env[name])}")
        print(f"export PYBIN={shlex.quote(pybin)}")


if __name__ == "__main__":
    main()
