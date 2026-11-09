"""Stage a frozen h28 private Kaggle source asset; never publish or submit."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
from arcforge import sglang_tree  # noqa: E402

REF = "c3485dae"
DIGEST = "b9cbd753acb18f0a40802b198f7d64fd92a48d5bc2117c2918bcff2b9b8ebb22"
MANIFEST = "a80d8d481679e25c17c4f9c3089c9d77812ef5a676eaab371541df2187de7f84"
FAST_DIGEST = "df5847fd1fb73f62c09e5432c7a263d85ea97832594a0836880dd3f02e252b28"
FAST_MANIFEST = "3e71105a9ebcb21dee2c030c65b5b24a94c8aeca825d2b381bc0290c8251c710"
LAUNCHER = "work/qwen38fn_l4/probes/runpod_hisparse.py"
SHA = sglang_tree.sha256_file


def archive(paths, target):
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        plain = Path(tmp) / "data.tar"
        with tarfile.open(plain, "w", format=tarfile.PAX_FORMAT) as tar:
            for path, name in paths:
                info = tar.gettarinfo(str(path), arcname=name)
                if not info.isfile():
                    raise ValueError(f"asset only contains regular files: {name}")
                info.mtime = info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mode = 0o644
                with path.open("rb") as source:
                    tar.addfile(info, source)
        subprocess.run(["zstd", "-q", "-10", "-o", str(target), str(plain)], check=True)
    return {"archive": target.name, "sha256": SHA(target), "bytes": target.stat().st_size}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--candidate-dir", type=Path, help="explicit isolated candidate directory; default remains frozen h28/h30")
    p.add_argument("--candidate-id", choices=("r7-r4", "r8-r2", "r9", "r9-r2", "arc-main", "arc-dense", "arc-dense-det", "arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv"), default="r7-r4",
                   help="explicit registry ID for candidate-dir; R7 compatibility default")
    p.add_argument("--expect-tree", help="required explicit full source tree SHA with --candidate-dir")
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--model-pin", type=Path,
                   help="model identity JSON (model_source, target_sizes, target_sha256) for a checkpoint other than the "
                        "reference notebook's REAP320, e.g. work/qwen38fn_l4/kaggle/model_gean320_int3_hf.json")
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--zstd-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source-dataset", default="threerabbits/qwen38fn-l4-hisparse-h28",
                   help="owned private dataset; use a new slug to preserve the running v5 asset")
    a = p.parse_args()
    if a.candidate_dir is None and a.candidate_id != "r7-r4":
        raise ValueError("--candidate-id requires --candidate-dir")
    if a.candidate_dir is not None and a.candidate_id in ("r7-r4", "r8-r2", "r9"):
        raise ValueError("MTP candidate has known draft-probability relay defect; use r9-r2+h39")
    if not re.fullmatch(r"threerabbits/[a-z0-9-]+", a.source_dataset):
        raise ValueError("source dataset must be an explicit owned private ref")
    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        raise FileExistsError("refusing to overwrite a nonempty source asset")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    entries = sglang_tree.package_manifest(a.freeze / "sglang")
    selected_digest = sglang_tree.manifest_digest(entries)
    selected_manifest = SHA(a.source_manifest)
    if a.candidate_dir is None and (len(entries) != 4901 or (selected_digest, selected_manifest) not in {
            (DIGEST, MANIFEST), (FAST_DIGEST, FAST_MANIFEST)}):
        raise ValueError("source must be frozen h28 or its startup-only h30 candidate")
    source_paths = [(x, "python/sglang/" + x.relative_to(a.freeze / "sglang").as_posix())
                    for x in sorted((a.freeze / "sglang").rglob("*"))
                    if x.is_file() and "__pycache__" not in x.parts and x.suffix != ".pyc"]
    candidate = None
    if a.candidate_dir is not None:
        import runtime_setup
        sys.path.insert(0, str(REPO / "work/qwen38fn_l4/probes"))
        from stage_source_candidates import verify_archive
        candidate = {"id": a.candidate_id, "expect_tree": a.expect_tree}
        source_name = f"sglang-candidate-{selected_digest[:12]}.tar.gz"
        for key, name in (("record", "CANDIDATE.json"), ("package_manifest", "PACKAGE_SHA256.json"),
                          ("cpu_validation", "ARCHIVE_VALIDATION.json")):
            path = a.candidate_dir / name
            candidate[key] = {"file": name, "sha256": SHA(path)}
        runtime_setup.candidate_identity({"candidate": candidate, "source": {
            "digest": selected_digest, "files": len(entries), "manifest_sha256": selected_manifest,
            "archive": source_name, "sha256": SHA(a.candidate_dir / "source.tar.gz")}})
        if a.source_dataset == "threerabbits/qwen38fn-l4-hisparse-h28":
            raise ValueError("candidate requires a separate source dataset slug")
        record = json.loads((a.candidate_dir / "CANDIDATE.json").read_text())
        expected = json.loads((a.candidate_dir / "PACKAGE_SHA256.json").read_text())
        if expected != entries or record["tree_sha256"] != a.expect_tree or record["source_manifest_sha256"] != selected_manifest:
            raise ValueError("candidate source differs from composition manifests")
        original = a.candidate_dir / "source.tar.gz"
        with tempfile.TemporaryDirectory(prefix="l4-kaggle-candidate-") as directory:
            runtime_setup.extract_archive(original, Path(directory), prefix="sglang")
        validation = verify_archive(original, expected)
        if validation != json.loads((a.candidate_dir / "ARCHIVE_VALIDATION.json").read_text()) or SHA(original) != record["source_archive_sha256"]:
            raise ValueError("candidate archive differs from CPU roundtrip receipt")
        # Kaggle auto-extracts .tar.gz uploads; runtime_setup reads the suffixed name (bytes/SHA unchanged).
        dataset_file = source_name + runtime_setup.KAGGLE_NOEXTRACT_SUFFIX
        shutil.copyfile(original, a.out_dir / dataset_file)
        source = {"archive": source_name, "dataset_file": dataset_file, "sha256": SHA(original),
                  "bytes": original.stat().st_size}
        for key in ("record", "package_manifest", "cpu_validation"):
            name = candidate[key]["file"]
            shutil.copyfile(a.candidate_dir / name, a.out_dir / name)
    else:
        if a.expect_tree is not None:
            raise ValueError("--expect-tree requires --candidate-dir")
        source = archive(source_paths, a.out_dir / f"sglang-h28-{selected_digest[:12]}.tar.zst")
    shutil.copyfile(a.source_manifest, a.out_dir / "SOURCE_SHA256")
    source.update(manifest="SOURCE_SHA256", manifest_sha256=selected_manifest, digest=selected_digest, files=len(entries),
                  base_digest=DIGEST, startup_cache_drop=selected_digest == FAST_DIGEST or candidate is not None)
    with tempfile.TemporaryDirectory(dir=a.out_dir) as tmp:
        stage = Path(tmp)
        paths = []
        probes = ("runpod_hisparse.py", "runpod_baseline.py", "runpod_runtime.py", "runpod_batch.py",
                  "runpod_vision.py", "runpod_correctness.py", "runpod_stop_server.py", "runpod_capacity.py")
        if candidate is not None:
            probes += ("runpod_mtp.py", "runpod_capacity_mtp.py", "runpod_cpu_affinity.py")
        for name in probes:
            rel = "work/qwen38fn_l4/probes/" + name
            dest = stage / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes((REPO / rel).read_bytes() if candidate is not None else
                             subprocess.check_output(["git", "show", REF + ":" + rel], cwd=REPO))
            paths.append((dest, "repo/" + rel))
        for rel in ("src/arcforge/__init__.py", "src/arcforge/paths.py", "src/arcforge/sglang_tree.py",
                    "src/arcforge/checkpoint_prefetch.py",
                    "work/qwen38fn_l4/kaggle/compiled_cache.py",
                    "work/qwen38fn_l4/kaggle/run_capacity_phase.py",
                    "work/qwen38fn_l4/kaggle/live_status.py",
                    "work/qwen38fn_l4/kaggle/run_submission.py"):
            paths.append((REPO / rel, "repo/" + rel))
        if candidate is not None:
            # v15: hostcomm hook + its sitecustomize injector (enabled only by Profile.hostcomm).
            for rel in ("work/qwen38fn_l4/probes/hostcomm/arc_hostcomm_hook.py",
                        "work/qwen38fn_l4/probes/hostcomm/hostcomm.py",
                        "work/qwen38fn_l4/probes/hostcomm/hostcomm.cu",
                        "work/qwen38fn_l4/kaggle/hc_inject/sitecustomize.py",
                        # l4opt: ARC_HC_MIX_W8 / ARC_HC_W8A16 Triton kernel (ARC_SKINNY_DIR)
                        "work/qwen38fn_l4/probes/skinny_gemm/w8a16.py",
                        "work/qwen38fn_l4/probes/skinny_gemm/skinny_gemm.py",
                        # l4opt: ARC_LMHEAD_TRITON verify lm_head GEMM
                        "work/qwen38fn_l4/probes/skinny_gemm/lm_head_gemm.py"):
                paths.append((REPO / rel, "repo/" + rel))
            # l4opt: DFlash draft vocabularies (builder draft_vocab, ARC_DFLASH_DRAFT_VOCAB) and target output vocabularies
            # (builder target_vocab, ARC_TARGET_VOCAB), every list in the checkout so a preset can switch lists without a
            # packer change
            paths += [(x, "repo/" + x.relative_to(REPO).as_posix())
                      for pattern in ("draft_vocab_*.json", "target_vocab_*.json")
                      for x in sorted((REPO / "work/qwen38fn_l4/kaggle").glob(pattern))]
        paths += [(x, "repo/" + x.relative_to(REPO).as_posix())
                  for x in sorted((REPO / "src/arcforge/tir").glob("*"))
                  if x.is_file() and x.suffix in (".py", ".json")]
        frozen_arc2 = REPO / "work/qwen38fn_l4/kaggle/frozen_arc2"
        paths += [(x, "repo/" + x.relative_to(REPO).as_posix()) for x in sorted(frozen_arc2.rglob("*"))
                  if x.is_file() and x.suffix in (".py", ".json") and "__pycache__" not in x.parts]
        paths.append((a.reference, "repo/reference/qwen38fn-nvfp4-mtp-replayssm-probe.ipynb"))
        launcher_sha = SHA(stage / LAUNCHER)
        payload = archive(sorted(paths, key=lambda v: v[1]), a.out_dir / ("arcforge-candidate-payload.tar.zst" if candidate is not None else "arcforge-h28-payload.tar.zst"))
        payload.update(launcher=LAUNCHER, launcher_sha256=launcher_sha)
        if candidate is not None:
            payload["files_sha256"] = {name.removeprefix("repo/"): SHA(path) for path, name in paths}
            payload["candidate_code_scope"] = "working tree files individually hash pinned"
            payload["candidate_code_git"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    setup = REPO / "work/qwen38fn_l4/kaggle/runtime_setup.py"
    shutil.copyfile(setup, a.out_dir / "runtime_setup.py")
    prefetch = REPO / "src/arcforge/checkpoint_prefetch.py"
    shutil.copyfile(prefetch, a.out_dir / "checkpoint_prefetch.py")
    reference_record = json.loads(a.reference.read_text())
    config_source = reference_record["cells"][1]["source"]
    config_source = "".join(config_source) if isinstance(config_source, list) else config_source
    config_node = next(node for node in ast.parse(config_source).body if isinstance(node, ast.ClassDef) and node.name == "Config")
    original_config = {node.target.id: ast.literal_eval(node.value) for node in config_node.body if isinstance(node, ast.AnnAssign)}
    original_manifest = json.loads((frozen_arc2 / "ORIGINAL_MANIFEST.json").read_text())
    if original_manifest["notebook_sha256"] != SHA(a.reference):
        raise ValueError("original ARC2 harness was extracted from another notebook")
    for name, entry in original_manifest["files"].items():
        if SHA(frozen_arc2 / name) != entry["sha256"]:
            raise ValueError(f"original ARC2 source changed: {name}")
    sys.path.insert(0, str(REPO / "work/qwen38fn_l4/probes"))
    import runpod_runtime
    model = runpod_runtime.notebook_metadata(a.reference)["MANIFEST"]
    model_source = "threerabbits/qwen38fn-nvfp4-reap320-arccode/Transformers/default/1"
    if a.model_pin is not None:
        pin = json.loads(a.model_pin.read_text())
        model, model_source = pin, pin["model_source"]
        identity = "/".join(part.lower() if i == 2 else part for i, part in enumerate(model_source.split("/")))
        if identity not in runpod_runtime.MODEL_IDENTITIES:
            raise ValueError(f"{a.model_pin}: {model_source} is none of MODEL_IDENTITIES")
        if runpod_runtime.MODEL_IDENTITIES[identity] != pin["target_sha256"]["config.json"]:
            raise ValueError(f"{a.model_pin}: config.json SHA is not {model_source}'s in MODEL_IDENTITIES")
    manifest = dict(schema_version=1, source_dataset=a.source_dataset,
        runtime_dataset="threerabbits/qwen38fn-sglang-0520-sm120-cp312",
        zstd_dataset="threerabbits/zstd-x86-64-kaggle",
        model_source=model_source,
        runtime_setup={"file": "runtime_setup.py", "sha256": SHA(setup)}, payload=payload, source=source,
        checkpoint_prefetch={"file": "checkpoint_prefetch.py", "sha256": SHA(prefetch), "bytes": prefetch.stat().st_size},
        compiled_cache_helper={"file": "work/qwen38fn_l4/kaggle/compiled_cache.py",
            "sha256": SHA(REPO / "work/qwen38fn_l4/kaggle/compiled_cache.py")},
        capacity_phase={"file": "work/qwen38fn_l4/kaggle/run_capacity_phase.py",
            "sha256": SHA(REPO / "work/qwen38fn_l4/kaggle/run_capacity_phase.py"),
            "probe_sha256": SHA(REPO / "work/qwen38fn_l4/probes/runpod_capacity.py")},
        live_status={"file": "work/qwen38fn_l4/kaggle/live_status.py",
            "sha256": SHA(REPO / "work/qwen38fn_l4/kaggle/live_status.py")},
        original_arc2={"root": "work/qwen38fn_l4/kaggle/frozen_arc2", "notebook_sha256": SHA(a.reference),
                       "manifest_sha256": SHA(frozen_arc2 / "ORIGINAL_MANIFEST.json"), "config": original_config},
        runtime_archive={"file": runpod_runtime.NEW_ARCHIVE, "sha256": runpod_runtime.NEW_SHA,
            "bytes": runpod_runtime.NEW_BYTES, "provenance_sha256": runpod_runtime.NEW_PROVENANCE_SHA},
        model_identity={k: model[k] for k in ("target_sizes", "target_sha256")},
        zstd_binary={"file": "zstd", "sha256": SHA(a.zstd_dir / "zstd"),
                     "bytes": (a.zstd_dir / "zstd").stat().st_size})
    if candidate is not None:
        manifest["candidate"] = candidate
        manifest["gpu_validation"] = "pending; local packaging is not promotion or Kaggle validation"
        if candidate["id"] in ("arc-main", "arc-dense", "arc-dense-det", "arc-dense-vocab", "arc-dense-vocab-tp4", "arc-dense-vocab-tp4-shv"):  # sglang-arc-2 fork exports (stage_fork_candidate.py)
            record = json.loads((a.candidate_dir / "CANDIDATE.json").read_text())
            manifest["candidate_validation"] = {"cpu_source_archive": "passed", "fork_commit": record["fork_commit"],
                "lock_target": record["lock_target"], "gpu_full_model": "Runpod/Vast L4 (work/sglang_fork/README.md)",
                "kaggle": "pending", "solver_quality": "pending", "private_only": True}
        if candidate["id"] == "r9-r2":
            manifest["candidate_validation"] = {"cpu_source_archive": "passed", "gpu_full_model": "pending",
                "kaggle": "pending", "solver_quality": "pending", "private_only": True, "h37_included": False}
    filename = "candidate-kaggle-manifest.json" if candidate is not None else "h28-kaggle-manifest.json"
    (a.out_dir / filename).write_text(json.dumps(manifest, indent=2) + "\n")
    (a.out_dir / "dataset-metadata.json").write_text(json.dumps({
        "id": manifest["source_dataset"], "title": f"Qwen38FN L4 HiSparse {a.candidate_id} MTP candidate" if candidate is not None else "Qwen38FN L4 HiSparse h28", "isPrivate": True,
        "licenses": [{"name": "apache-2.0"}], "description": f"Isolated {a.candidate_id} MTP candidate, CPU source/archive identity validated; model, performance and Kaggle validation pending." if candidate is not None else "GPU-validated h28 base, startup-only h30 candidate, original ARC2 harness and ARC3 checkpoint warming; Kaggle GPU validation pending."}, indent=2) + "\n")
    print(json.dumps({"stage": str(a.out_dir), "source_digest": selected_digest, "payload": payload,
                      "runtime_setup_sha256": SHA(setup)}, indent=2))


if __name__ == "__main__":
    main()
