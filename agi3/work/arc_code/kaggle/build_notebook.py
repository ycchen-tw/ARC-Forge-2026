#!/usr/bin/env python3
"""Emit the Kaggle notebook that runs the new arc-code arm on the ARC-AGI-3 machine.

Forked on 2026-09-25 from work/arc_agi3/arc_code_qwen/build_kaggle_arm.py, which
builds the old arm and stays untouched until this one has scored. Same inputs,
all mounts (the session has no network):

    /kaggle/input/arc-code-kaggle-assets       arc-code prep (inspected, not played through)
    /kaggle/input/arcforge-codex-0154-patched  the self-built codex, the runtime overlay
    /kaggle/input/<pip dataset>                a pip: this image cannot make one
    /kaggle/input/<runtime dataset>            the pinned sm_120 sglang tree
    /kaggle/input/<model>                      weights
    /kaggle/input/competitions/...             arc_agi wheels and environment_files

The arm is carried inside the notebook with its repo layout -- work/arc_code/ and
src/arcforge/ -- in one deterministic gzipped tar, so what ran is what is in git
at this commit.

    python work/arc_code/kaggle/build_notebook.py --out <dir>
"""

from __future__ import annotations

import argparse
import ast
import base64
import gzip
import hashlib
import io
import json
import pprint
import re
import subprocess
import sys
import tarfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CELLS = HERE / "notebook"
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
from arcforge import sglang_tree  # noqa: E402  the same digest the notebook asserts on Kaggle
from arcforge import sglang_series  # noqa: E402  series arms (arc3): their own digest row and manifest
from arcforge import sglang_lock  # noqa: E402  sglang-arc-2 trees: deploy/sglang.lock.json targets

RUNTIME_DATASET = "threerabbits/qwen38fn-sglang-0520-sm120-cp312"
PIP_DATASET = "threerabbits/pip-offline-wheel"
ZSTD_DATASET = "threerabbits/zstd-x86-64-kaggle"
ASSETS_DATASET = "threerabbits/arc-code-kaggle-assets"
COMPETITION = "arc-prize-2026-arc-agi-3"
#: the image every arm has run on (pulled from arc-code-newarm-v12-hs55, 2026-10-04): a new kernel slug otherwise gets
#: Kaggle's latest image (Python 3.13 in 2026-10) and the cp312 runtime refuses to start
DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
PREFETCH_SOURCE = "work/qwen38fn_eval/sm120_runtime.py"
PREFETCH_CELL_BEFORE = "03_serve.py"
INT3_TUNING_DIR = "work/qwen38fn_eval/int3_tuning"
#: int3_deploy_runtime switches that print "<KEY>=<value>" at launch; a config that sets one must require that line
INT3_SWITCHES = ("ARC_INT3_RUNNER_REUSE", "ARC_HOST_POOL_PRIVATE", "ARC_PINNED_HOST_REGISTER")


def carried() -> list[str]:
    """Repo-relative paths of everything the arm needs at run time. Globbed, then
    checked: a file the launcher imports but the bundle lacks is a build error."""
    unit = REPO_ROOT / "work/arc_code"
    names = ["src/arcforge/__init__.py", "src/arcforge/sglang_tree.py",
             *sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / "src/arcforge/arc_code").glob("*.py")),
             *(f"work/arc_code/{n}" for n in ("run_arm.py", "posix_fence.py", "fence_posix.sh",
                                               "jail.sh", "prompts.py")),
             # the model proxy (config model_proxy) and what it imports at start / for act_image
             *(f"work/arc_code/{n}" for n in ("sampling_proxy.py", "compaction_tail.py", "reasoning_trim.py",
                                               "act_images.py")),
             # the read-only reviewer (config reviewer): run_arm imports these, the proxy review_notice.py
             *(f"work/arc_code/{n}" for n in ("reviewer.py", "review_scheduler.py", "review_notice.py")),
             *sorted(str(p.relative_to(REPO_ROOT)) for p in (unit / "review_prompts").rglob("*.md")),
             # game cards (config cards): the card library and the static cards the reviewer gets
             "work/arc_code/card_library.py",
             *sorted(str(p.relative_to(REPO_ROOT)) for p in (unit / "game_cards").glob("*.md")),
             # packed INT3 (config ``int3``): 03_serve launches through these
             "work/qwen38fn_eval/int3_deploy_runtime.py", "work/qwen38fn_eval/int3_deploy_patch/sitecustomize.py",
             # Humming schedule tables (int3.tuning -> ARC_INT3_TUNING)
             *sorted(str(p.relative_to(REPO_ROOT)) for p in (REPO_ROOT / INT3_TUNING_DIR).glob("*.json")),
             *sorted(str(p.relative_to(REPO_ROOT)) for p in (unit / "arms").glob("*.py")),
             # harness gen-2 (config harness gen2): codex runs these hooks inside the fence as the game user
             *sorted(str(p.relative_to(REPO_ROOT)) for p in (unit / "gen2" / "hooks").glob("*.py")),
             # gen-2 toolkits (config gen2_toolkit): run_arm reads MANIFEST.json and copies the files into the workspace
             *sorted(str(p.relative_to(REPO_ROOT)) for p in (unit / "gen2" / "toolkit").rglob("*")
                     if p.is_file() and "__pycache__" not in p.parts),
             *sorted(str(p.relative_to(REPO_ROOT)) for p in (unit / "prompts").rglob("*.md")),
             *sorted(str(p.relative_to(REPO_ROOT)) for p in (unit / "driver_files").rglob("*.md")),
             *sorted(str(p.relative_to(REPO_ROOT)) for p in (unit / "configs").glob("kaggle-*.json"))]
    missing = [n for n in names if not (REPO_ROOT / n).is_file()]
    if missing:
        raise SystemExit(f"carried files missing: {missing}")
    return names


def bundle(names: list[str]) -> bytes:
    """Deterministic: mtime/uid/owner zeroed and gzip's own mtime too, so an
    unchanged tree builds an identical notebook."""
    def normalise(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.mtime, info.uid, info.gid = 0, 0, 0
        info.uname = info.gname = ""
        info.mode = 0o755 if info.mode & 0o100 else 0o644
        return info

    raw = io.BytesIO()
    with tarfile.open(fileobj=raw, mode="w") as archive:
        for name in names:
            archive.add(REPO_ROOT / name, arcname=name, filter=normalise)
    return gzip.compress(raw.getvalue(), 9, mtime=0)


def provenance(names: list[str]) -> dict:
    def git(*argv):
        done = subprocess.run(["git", "-C", str(REPO_ROOT), *argv], capture_output=True, text=True)
        return done.stdout.strip() if done.returncode == 0 else None

    status = git("status", "--porcelain", "--", "work/arc_code", "src/arcforge")
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": None if status is None else bool(status),
            "carried": {n: hashlib.sha256((REPO_ROOT / n).read_bytes()).hexdigest()[:16] for n in names}}


def prefetch_source() -> str:
    """The 16-thread page-cache warm, lifted verbatim (it *is* the original)."""
    text = (REPO_ROOT / PREFETCH_SOURCE).read_text()
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name == "prefetch_checkpoint")
    return ast.get_source_segment(text, node)


def cells(config: dict) -> list[tuple[str, str]]:
    names = carried()
    build = provenance(names)
    files = sorted(CELLS.glob("[0-9][0-9]_*.py"))
    overview = "\n".join([
        f"# {config['kernel_slug']}",
        "",
        "The new arc-code arm (work/arc_code) on the ARC-AGI-3 machine. Save & Run plays a stand-in",
        "official gateway through the same upstream path the scored rerun takes to `gateway:8001`.",
        "",
        f"- built from `{build['commit']}`" + (" (working tree dirty)" if build["dirty"] else ""),
        f"- checkpoint `{config['model_source']}`",
        *([f"- overlay `{config['model_overlay'].get('dataset') or config['model_overlay']['model']}`: "
           f"deploy_manifest.json "
           f"`{config['model_overlay']['manifest_sha256'][:16]}` over the checkpoint's "
           f"`{config['model_manifest_sha256'][:16]}`"] if config.get("model_overlay") else []),
        *([f"- weight patch `{config['model_patch']['model']}`: patch.json `{config['model_patch']['patch_sha256'][:16]}` "
           f"over the checkpoint's `{config['model_manifest_sha256'][:16]}`"] if config.get("model_patch") else []),
        *([f"- weight switch to `{config['weight_switch']['version']}` (`{config['weight_switch']['model']}`, "
           f"model.safetensors `{config['weight_switch']['sha256'][:16]}`) in place, "
           + ", ".join(f"{s} s into `{arm}`" for arm, s in config["weight_switch"]["after_s"].items())]
          if config.get("weight_switch") else []),
        f"- codex `{config['codex_dataset']}` sha256 `{config['codex_sha256'][:16]}`",
        f"- served as `{config['served_model']}`: context {config['context_length']}, "
        f"{config['max_requests']} concurrent, KV {config.get('kv_cache_dtype', 'auto')}",
        f"- arm configs: rerun `{config['rerun_arm_config']}`, Save & Run `{config['saverun_arm_config']}`",
        *([f"- sglang tree `{config['sglang_tree']['dataset']}`: whole package "
           f"`{config['sglang_tree']['tree_digest'][:12]}` ({config['sglang_tree']['tree_files']} files) laid over "
           f"the runtime archive, HiSparse manifest `{config['sglang_tree']['manifest_sha256'][:12]}`"]
          if config.get("sglang_tree") else []),
    ])
    generated = "\n".join([
        '"""What the build decided: the configuration, the carried code, and where both came from."""', "",
        "BUILD = " + pprint.pformat(build, width=100, sort_dicts=False), "",
        "CFG = " + pprint.pformat(config, width=100, sort_dicts=False), "",
        'PAYLOAD = "' + base64.b64encode(bundle(names)).decode() + '"', ""])
    built = [("markdown", overview), ("code", generated)]
    for path in files:
        if path.name == PREFETCH_CELL_BEFORE:
            built.append(("code", f'"""The page-cache warm, lifted verbatim from {PREFETCH_SOURCE}."""\n\n\n'
                                  + prefetch_source() + "\n"))
        if path.name == "04_main.py" and config.get("saverun_placeholder_only"):
            built.append(("code", placeholder_main(path.read_text())))
        else:
            built.append(("code", path.read_text()))
    return built


def placeholder_main(main_source: str) -> str:
    """The main cell of a submit-only version: the scored rerun executes 04_main.py unchanged (exec'd in the
    notebook's globals, as the cell itself would run); Save & Run only writes the placeholder submission.parquet
    Kaggle needs on the selected version, then ends -- nothing is served or played, so it validates nothing."""
    return "\n".join([
        '"""Submit-only version (saverun_placeholder_only): the rerun runs 04_main.py as is; Save & Run writes the',
        'placeholder submission.parquet and stops."""',
        "MAIN_SOURCE = " + repr(main_source),
        'if os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").lower() in ("1", "true"):',
        '    exec(compile(MAIN_SOURCE, "04_main.py", "exec"), globals())',
        "else:",
        "    import pandas as pd",
        '    placeholder = WORK / "submission.parquet"',
        '    pd.DataFrame([{"row_id": "1_0", "game_id": "1", "end_of_game": True, "score": 1}],',
        '                 columns=["end_of_game", "game_id", "row_id", "score"]).to_parquet(placeholder, index=False)',
        '    say("SUBMISSION", json.dumps({"placeholder_only": str(placeholder), "bytes": placeholder.stat().st_size}))',
        "",
    ])


def check(config: dict) -> None:
    """Refuse a notebook that would fail on Kaggle for a reason visible here.

    A HiSparse config's tree pins are checked against the commit that made its tree
    (``sglang_tree.tree_commit``, read with git show), which must be in this commit's history. They are not
    compared with this commit's patches: the patch stack is shared with other lines (L4, DFlash) and moves on
    without the tree a deployment pins, so a build here carries this commit's arm against the pinned tree.
    """
    need = ("kernel_slug", "served_model", "context_length", "max_requests", "runtime_source_sha256",
            "codex_dataset", "codex_sha256", "model_source", "gateway_url", "standin_port",
            "rerun_arm_config", "saverun_arm_config")
    missing = [k for k in need if not config.get(k)]
    if missing:
        raise SystemExit(f"notebook config lacks {missing}")
    if config["codex_dataset"] not in (config.get("extra_dataset_sources") or []):
        raise SystemExit(f"codex_dataset {config['codex_dataset']} is not in extra_dataset_sources")
    for key in ("rerun_arm_config", "saverun_arm_config"):
        arm = json.loads((REPO_ROOT / "work/arc_code/configs" / config[key]).read_text())
        if arm["fence"] != "posix":
            raise SystemExit(f"{config[key]}: Kaggle has no namespaces; fence must be posix")
        if "tito" in arm:
            # run_arm imports arcforge.tito only for such a config, and carried() does not bring it along;
            # the rerun serves with replay but no ledger dir, which the tito preflight refuses anyway.
            raise SystemExit(f"{config[key]}: tito (session ledger) is not carried into the notebook")
        if not arm["auto_compact_token_limit"] < arm["model_context_window"] <= config["context_length"]:
            raise SystemExit(f"{config[key]}: need compact < codex window <= served context")
        if arm["concurrency"] > config["max_requests"]:
            raise SystemExit(f"{config[key]}: concurrency {arm['concurrency']} > max_requests")
    check_oversample(config)
    if "kaggle_session_seconds" in config:
        # 04_main's session length (default 32400, Kaggle's nine hours); anything else is a timeout test and says so
        session = config["kaggle_session_seconds"]
        if not isinstance(session, int) or session <= 0:
            raise SystemExit(f"kaggle_session_seconds must be a positive int, not {session!r}")
        if session != 32400 and "timeout test" not in config.get("note", ""):
            raise SystemExit("kaggle_session_seconds other than 32400 is a timeout test: the note must say 'timeout test'")
    if config.get("sglang_tree"):
        check_sglang_tree(config)
    if config.get("dflash"):
        check_dflash(config)
    if config.get("int3"):
        check_int3(config)
    if "ARC_THINK_BUDGET" in config.get("server_env", {}):
        check_think_budget(config)
    check_model_proxy(config)
    check_reviewer(config)
    if config.get("model_overlay"):
        check_model_overlay(config)
    if config.get("model_patch"):
        check_model_patch(config)
    if config.get("weight_switch"):
        check_weight_switch(config)
    if config.get("agent_wheels"):
        check_agent_wheels(config)


def oversample_sets() -> dict:
    """04_main's OVERSAMPLE_SETS, read from the cell source (cells are text here, not modules)."""
    tree = ast.parse((CELLS / "04_main.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["OVERSAMPLE_SETS"]:
            return ast.literal_eval(node.value)
    raise SystemExit("04_main.py defines no OVERSAMPLE_SETS")


def check_oversample(config: dict) -> None:
    """Save & Run's game set: ``oversample_to`` (round-robin to N) or ``oversample_set`` (a named set), not both."""
    if config.get("oversample_to") and config.get("oversample_set"):
        raise SystemExit("set oversample_to or oversample_set, not both")
    name = config.get("oversample_set")
    if name is not None and name not in oversample_sets():
        raise SystemExit(f"oversample_set {name!r} is not in 04_main.OVERSAMPLE_SETS {sorted(oversample_sets())}")


def check_agent_wheels(config: dict) -> None:
    """``agent_wheels`` (02_stage stage_agent_wheels): wheels the competition's arc_agi_3_wheels lacks, from our own
    dataset (in extra_dataset_sources), each pinned by sha256; ``packages`` are ``name==version`` pins with a wheel
    among ``files``, installed into the agent venv beside the four base packages; ``check`` is Python run in that venv."""
    spec = config["agent_wheels"]
    if set(spec) != {"dataset", "files", "packages", "check"}:
        raise SystemExit(f"agent_wheels keys {sorted(spec)} != ['check', 'dataset', 'files', 'packages']")
    if spec["dataset"] not in (config.get("extra_dataset_sources") or []):
        raise SystemExit(f"agent_wheels dataset {spec['dataset']} is not in extra_dataset_sources")
    if not spec["files"] or not all(n.endswith(".whl") and re.fullmatch(r"[0-9a-f]{64}", h)
                                    for n, h in spec["files"].items()):
        raise SystemExit(f"agent_wheels files must map wheel names to sha256: {spec['files']}")
    for pin in spec["packages"]:
        name, _, version = pin.partition("==")
        if not (name and version) or not any(f.startswith(f"{name}-{version}-") for f in spec["files"]):
            raise SystemExit(f"agent_wheels package {pin!r}: needs name==version with its wheel in files")
    if not spec["packages"] or not isinstance(spec["check"], str) or not spec["check"].strip():
        raise SystemExit("agent_wheels needs packages and a check")


def check_model_overlay(config: dict) -> None:
    """``model_overlay`` (02_stage overlay_model): a dataset (``dataset``, in extra_dataset_sources) or a model
    (``model``, a version handle; mounted through model_sources) holding the shards an RL version rewrote
    (work/qwen38fn_rl/rl_overlay_package.py), laid over the mounted package as symlinks. The package must be pinned
    (the overlay names the manifest it was made against) and so must the overlay's own deploy_manifest.json."""
    spec = config["model_overlay"]
    if sorted(spec) not in (["dataset", "manifest_sha256"], ["manifest_sha256", "model"]):
        raise SystemExit(f"model_overlay keys are dataset or model, and manifest_sha256, not {sorted(spec)}")
    if "dataset" in spec and spec["dataset"] not in (config.get("extra_dataset_sources") or []):
        raise SystemExit(f"model_overlay dataset {spec['dataset']} is not in extra_dataset_sources")
    if "model" in spec and not re.fullmatch(r"[^/]+/[^/]+/[^/]+/[^/]+/\d+", spec["model"]):
        raise SystemExit(f"model_overlay model {spec['model']!r} is not owner/slug/framework/variation/version")
    for key, value in (("model_manifest_sha256", config.get("model_manifest_sha256")),
                       ("model_overlay.manifest_sha256", spec["manifest_sha256"])):
        if not (isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)):
            raise SystemExit(f"{key} must be a sha256, not {value!r}")


def check_model_patch(config: dict) -> None:
    """``model_patch`` (02_stage weight_patch): a model (``model``, a version handle mounted through model_sources) holding
    the tensors an RL version changed (work/qwen38fn_rl/rl_weight_patch.py), pinned by its patch.json (``patch_sha256``),
    which names the package manifest it was made against -- so that must be pinned too. The int3 runtime swaps the
    tensors in at load (ARC_INT3_WEIGHT_PATCH), so the config serves int3, and require_log holds its line with the
    tensor count; an overlay and a patch together would be two answers to which weights are served."""
    spec = config["model_patch"]
    if sorted(spec) != ["model", "patch_sha256"]:
        raise SystemExit(f"model_patch keys are model and patch_sha256, not {sorted(spec)}")
    if not re.fullmatch(r"[^/]+/[^/]+/[^/]+/[^/]+/\d+", spec["model"]):
        raise SystemExit(f"model_patch model {spec['model']!r} is not owner/slug/framework/variation/version")
    for key, value in (("model_manifest_sha256", config.get("model_manifest_sha256")),
                       ("model_patch.patch_sha256", spec["patch_sha256"])):
        if not (isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)):
            raise SystemExit(f"{key} must be a sha256, not {value!r}")
    if config.get("model_overlay"):
        raise SystemExit("model_patch and model_overlay together: serve one of them")
    if not config.get("int3"):
        raise SystemExit("model_patch is applied by the int3 deploy runtime; the config does not serve int3")
    if not any(re.fullmatch(r"ARC_INT3_WEIGHT_PATCH_LOADED tensors=\d+", m) for m in config.get("require_log", [])):
        raise SystemExit("model_patch needs 'ARC_INT3_WEIGHT_PATCH_LOADED tensors=<n>' in require_log")


def check_weight_switch(config: dict) -> None:
    """``weight_switch`` (04_main WeightSwitch, hs110 v39): ``model`` (a version handle mounted through model_sources)
    holds one merged export, its model.safetensors pinned by ``sha256``, loaded in place as weight version ``version``
    ``after_s[<arm config>]`` seconds into the arm. Both arm configs need an entry inside their wall, so Save & Run
    switches too. The load goes through the int3 runtime's delta path with the DFlash drafter (the only draft that
    keep_identity takes) over the package itself: a patch or an overlay would be a second answer to what it replaces."""
    spec = config["weight_switch"]
    if sorted(spec) != ["after_s", "model", "sha256", "version"]:
        raise SystemExit(f"weight_switch keys are after_s, model, sha256 and version, not {sorted(spec)}")
    if not re.fullmatch(r"[^/]+/[^/]+/[^/]+/[^/]+/\d+", spec["model"]):
        raise SystemExit(f"weight_switch model {spec['model']!r} is not owner/slug/framework/variation/version")
    if not (isinstance(spec["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", spec["sha256"])):
        raise SystemExit(f"weight_switch.sha256 must be a sha256, not {spec['sha256']!r}")
    if not isinstance(spec["version"], str) or spec["version"] in ("", "default"):
        raise SystemExit(f"weight_switch.version {spec['version']!r}: name the version (the server starts as 'default')")
    arms = {config["rerun_arm_config"], config["saverun_arm_config"]}
    if not isinstance(spec["after_s"], dict) or set(spec["after_s"]) != arms:
        raise SystemExit(f"weight_switch.after_s needs exactly the two arm configs {sorted(arms)}")
    for name, after in spec["after_s"].items():
        wall = json.loads((REPO_ROOT / "work/arc_code/configs" / name).read_text())["wall_seconds"]
        if not (isinstance(after, int) and not isinstance(after, bool) and 0 < after < wall):
            raise SystemExit(f"weight_switch.after_s[{name}] = {after!r}: a whole number of seconds inside its wall {wall}")
    if not config.get("int3") or not config.get("dflash"):
        raise SystemExit("weight_switch loads through the int3 runtime's delta path with the DFlash drafter")
    if config.get("model_patch") or config.get("model_overlay"):
        raise SystemExit("weight_switch goes over the package itself, not a model_patch or model_overlay")


def check_reviewer(config: dict) -> None:
    """An arm config with a ``reviewer`` block (work/arc_code/review_scheduler.py): the block passes run_arm's own
    check, and the notices reach the players only through the model proxy (04_main gives it --review-run-dir)."""
    for key in ("rerun_arm_config", "saverun_arm_config"):
        arm = json.loads((REPO_ROOT / "work/arc_code/configs" / config[key]).read_text())
        if "reviewer" not in arm:
            continue
        if not config.get("model_proxy"):
            raise SystemExit(f"{config[key]}: a reviewer needs model_proxy (its notices go through the proxy)")
        sys.path.insert(0, str(REPO_ROOT / "work/arc_code"))
        import review_scheduler
        review_scheduler.check_spec(arm["reviewer"], arm, "codex")


#: what sglang's ResponsesRequest samples with when codex sends nothing (--sampling-defaults openai keeps the model's
#: generation_config out); 04_main refuses a proxy that only writes these, as check_model_proxy does
SERVER_SAMPLING = {"temperature": 0.7, "top_p": 1.0, "top_k": -1}

#: ports the notebook and the arm already use on the host loopback (04_main, the arm configs' defaults)
TAKEN_PORTS = ("port", "standin_port")


def check_model_proxy(config: dict) -> None:
    """``model_proxy`` (04_main start_model_proxy): sampling_proxy.py between codex and sglang, which turns the
    act_image markers into the game record's PNGs and carries the reviewer's notices. It goes with either in both
    arm configs: markers without the proxy reach the model as text. Without either it only writes the sampling into
    every request (codex sends none), so it has to write something other than the server's own default
    (SERVER_SAMPLING), or it is a hop for nothing."""
    spec = config.get("model_proxy")
    arms = {key: json.loads((REPO_ROOT / "work/arc_code/configs" / config[key]).read_text())
            for key in ("rerun_arm_config", "saverun_arm_config")}
    images = {key: bool(arm.get("act_image")) for key, arm in arms.items()}
    if len(set(images.values())) != 1:
        raise SystemExit(f"act_image differs between the arm configs: {images}")
    reviewers = {key: "reviewer" in arm for key, arm in arms.items()}
    if len(set(reviewers.values())) != 1:
        raise SystemExit(f"a reviewer in one arm config only: {reviewers}")
    needed = images["rerun_arm_config"] or reviewers["rerun_arm_config"]
    if needed and not spec:
        raise SystemExit("model_proxy missing for act_image / a reviewer in the arm configs")
    if spec and not needed and spec.get("sampling") == SERVER_SAMPLING:
        raise SystemExit("model_proxy without act_image / a reviewer and with the server's own sampling: a hop for nothing")
    if not spec:
        return
    if set(spec) != {"port", "sampling", "max_restarts"}:
        raise SystemExit(f"model_proxy keys are port, sampling, max_restarts, not {sorted(spec)}")
    if not isinstance(spec["port"], int) or spec["port"] in [config[k] for k in TAKEN_PORTS] \
            or any(spec["port"] in (arm["gateway_port"], arm["env_port"], arm["official_port"], arm["model_port"])
                   for arm in arms.values()):
        raise SystemExit(f"model_proxy port {spec['port']!r} is not a free int port")
    if not spec["sampling"] or not isinstance(spec["sampling"], dict):
        raise SystemExit("model_proxy sampling must name what it writes into every request")
    if not isinstance(spec["max_restarts"], int) or spec["max_restarts"] < 0:
        raise SystemExit(f"model_proxy max_restarts {spec['max_restarts']!r} must be a non-negative int")
    sizes = {arm["record"]["png_size"] for arm in arms.values()}
    if len(sizes) != 1:
        raise SystemExit(f"record png_size differs between the arm configs: {sizes}")


def check_think_budget(config: dict) -> None:
    """ARC_THINK_BUDGET is read by sglang-arc-2's scheduler (speculative/arc_think_budget.py), which logs
    "ARC think budget: <N> tokens (...)" at start; a tree without that module ignores the variable without a word,
    so the config must require the line."""
    value = config["server_env"]["ARC_THINK_BUDGET"]
    if not (isinstance(value, str) and value.isdigit() and int(value) >= 1):
        raise SystemExit(f"ARC_THINK_BUDGET {value!r} must be a positive integer string")
    line = f"ARC think budget: {int(value)} tokens"
    if line not in config.get("require_log", []):
        raise SystemExit(f"ARC_THINK_BUDGET={value} needs the '{line}' log marker")


def git_show(commit: str, rel: str) -> bytes:
    done = subprocess.run(["git", "-C", str(REPO_ROOT), "show", f"{commit}:{rel}"], capture_output=True)
    if done.returncode != 0:
        raise SystemExit(f"git show {commit[:12]}:{rel} failed: {done.stderr.decode().strip()}")
    return done.stdout


#: sglang_tree.arm: absent = hisparse (every notebook up to hs55 v7: TREE_DIGEST's hisparse row and
#: experimental-hisparse/SOURCE_SHA256); a series arm (arcforge.sglang_series, arc3) pins TREE_DIGEST.<arm> and its
#: own manifest
TREE_ARMS = ("hisparse",) + tuple(sglang_series.SERIES)


def check_sglang_tree_commit(config: dict) -> None:
    """The tree pins are what the digest row of the tree's arm (``sglang_tree.arm``, default hisparse: TREE_DIGEST's
    hisparse row; a series arm: TREE_DIGEST.<arm>) and that arm's manifest held at ``sglang_tree.tree_commit``, the
    commit whose patches made the tree in the dataset.

    That commit must be in this commit's history: one on an unmerged branch stops resolving once the branch
    is gone (the dataset's own TREE json can name such a commit; record the main commit with the same
    patches/ tree instead).
    """
    tree = config["sglang_tree"]
    commit = tree["tree_commit"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit(f"sglang_tree tree_commit {commit!r} is not a full commit sha")
    ancestor = subprocess.run(["git", "-C", str(REPO_ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
                              capture_output=True, text=True)
    if ancestor.returncode != 0:
        raise SystemExit(f"sglang_tree tree_commit {commit[:12]} is not in this commit's history: "
                         f"{ancestor.stderr.strip() or 'not an ancestor of HEAD'}")
    patches = "patches/sglang-0.5.20"
    arm = tree.get("arm", "hisparse")
    if arm not in TREE_ARMS:
        raise SystemExit(f"sglang_tree arm {arm!r} is not one of {list(TREE_ARMS)}")
    if arm == "hisparse":
        row = sglang_tree.parse_tree_digest(git_show(commit, f"{patches}/TREE_DIGEST").decode(),
                                            f"{commit[:12]}:{patches}/TREE_DIGEST")["hisparse"]
        manifest_rel = "experimental-hisparse/SOURCE_SHA256"
    else:
        digest_rel = sglang_series.digest_path(Path(), arm).name
        try:
            row = sglang_series.parse_row(git_show(commit, f"{patches}/{digest_rel}").decode(),
                                          f"{commit[:12]}:{patches}/{digest_rel}", arm)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        manifest_rel = sglang_series.manifest_path(Path(), arm).name
    if (tree["tree_digest"], tree["tree_files"]) != (row["digest"], row["files"]):
        raise SystemExit(f"sglang_tree pins {tree['tree_digest'][:12]} ({tree['tree_files']} files), TREE_DIGEST "
                         f"at its tree_commit {commit[:12]} is {row['digest'][:12]} ({row['files']})")
    manifest = hashlib.sha256(git_show(commit, f"{patches}/{manifest_rel}")).hexdigest()
    if tree["manifest_sha256"] != manifest:
        raise SystemExit(f"sglang_tree manifest {tree['manifest_sha256'][:12]} != {manifest_rel} at its "
                         f"tree_commit {commit[:12]} ({manifest[:12]})")


def check_sglang_tree(config: dict) -> None:
    """A HiSparse notebook is complete and sizes L for the arm it serves (internal consistency only)."""
    tree = config["sglang_tree"]
    need = ("dataset", "archive", "archive_sha256", "tree_digest", "tree_files", "manifest", "manifest_sha256")
    missing = [k for k in need if not tree.get(k)]
    if missing:
        raise SystemExit(f"sglang_tree lacks {missing}")
    if bool(tree.get("tree_commit")) == bool(tree.get("lock_target")):
        raise SystemExit("sglang_tree pins its tree by exactly one of tree_commit (a patch-built tree) or "
                         "lock_target (a sglang-arc-2 tree, deploy/sglang.lock.json)")
    if tree["dataset"] not in (config.get("extra_dataset_sources") or []):
        raise SystemExit(f"sglang_tree dataset {tree['dataset']} is not in extra_dataset_sources")
    if config["runtime_overlay"]:
        raise SystemExit("runtime_overlay must be [] with sglang_tree: the tree carries 0018 and an overlay "
                         "would replace a file inside the pinned tree")
    named = tree["commit"][:12] if tree.get("lock_target") else tree["tree_digest"][:12]
    if not tree["archive"].endswith(f"{named}.tar.zst"):
        raise SystemExit(f"sglang_tree archive {tree['archive']} is not named for its tree {named}")
    args = config["server_args"]
    if "--enable-hisparse" not in args or "--hisparse-config" not in args:
        raise SystemExit("sglang_tree without --enable-hisparse / --hisparse-config in server_args")
    logical = json.loads(args[args.index("--hisparse-config") + 1])["qsa_logical_tokens"]
    if not config.get("min_headroom_after_start_gib"):
        raise SystemExit("a HiSparse notebook needs min_headroom_after_start_gib (the post-start RAM floor)")
    for key in ("rerun_arm_config", "saverun_arm_config"):
        arm = json.loads((REPO_ROOT / "work/arc_code/configs" / config[key]).read_text())
        # L is R x per-request budget; an arm below R leaves L unused, and each request must reach compaction.
        # dynamic_context (run_arm): with all R games running the compaction point is min_tokens, and it only grows
        # as games end, to safety x L / games running; L is the server's whole pool, so one run per server.
        if arm["concurrency"] != config["max_requests"]:
            raise SystemExit(f"{config[key]}: concurrency {arm['concurrency']} != max_requests {config['max_requests']}")
        dynamic = arm.get("dynamic_context")
        if dynamic is not None and dynamic["pool_share"] != 1:
            raise SystemExit(f"{config[key]}: dynamic_context.pool_share must be 1 (one run per Kaggle server)")
        compaction = arm["auto_compact_token_limit"] if dynamic is None else dynamic["min_tokens"]
        if logical // config["max_requests"] < compaction:
            raise SystemExit(f"L/R = {logical // config['max_requests']} < compaction at {compaction} ({config[key]})")
    if tree.get("lock_target"):
        check_sglang_tree_lock(config)
    else:
        check_sglang_tree_commit(config)


def check_sglang_tree_lock(config: dict) -> None:
    """A sglang-arc-2 tree (work/sglang_fork/pack_kaggle_fork_tree.py): the pins are the lock target's -- the fork
    commit, its whole-package digest (symlink convention, as the notebook computes it after extraction), file
    count and h05 manifest."""
    tree = config["sglang_tree"]
    try:
        t = sglang_lock.target(tree["lock_target"])
    except KeyError as error:
        raise SystemExit(str(error)) from error
    pins = {"commit": (tree.get("commit"), t["commit"]), "tree_digest": (tree["tree_digest"], t["digest_symlink"]),
            "tree_files": (tree["tree_files"], t["files"]),
            "manifest_sha256": (tree["manifest_sha256"], t["source_manifest_sha256"])}
    wrong = {k: v for k, v in pins.items() if v[0] != v[1]}
    if wrong:
        raise SystemExit(f"sglang_tree differs from lock target {tree['lock_target']}: {wrong}")


def check_dflash(config: dict) -> None:
    """The MTP5-DFlash drafter (sglang-arc-2 speculative/arc_dflash) in place of the native MTP draft: its dataset
    is mounted, its export, keep set and calibration (MoE only; a dense export has neither) are pinned by sha256,
    the tree has the worker, and the server runs NEXTN K-1 / K, top-k 1, with rejection sampling (the output stays
    target-distributed)."""
    d = config["dflash"]
    if d.get("ffn_type", "moe") not in ("moe", "dense_shared", "dense"):
        raise SystemExit(f"dflash ffn_type {d['ffn_type']!r}: moe (default), dense_shared or dense (plan B)")
    moe_only = ("keep", "keep_sha256", "calib", "calib_sha256")
    need = ("dataset", "export_dir", "config_sha256")
    if d.get("ffn_type") == "dense":
        check_plain_dflash(config)
    if d.get("ffn_type") in ("dense_shared", "dense"):
        stray = [k for k in moe_only if k in d]
        if stray:
            raise SystemExit(f"a dense dflash export has no keep set or calibration: drop {stray}")
    else:
        need += moe_only
    missing = [k for k in need if k not in d or d[k] is None]
    if missing:
        raise SystemExit(f"dflash lacks {missing}")
    if d["dataset"] not in (config.get("extra_dataset_sources") or []):
        raise SystemExit(f"dflash dataset {d['dataset']} is not in extra_dataset_sources")
    vocab = [k for k in ("draft_vocab_dataset", "draft_vocab", "draft_vocab_sha256") if d.get(k)]
    if vocab and len(vocab) != 3:
        raise SystemExit(f"dflash draft vocab needs draft_vocab_dataset, draft_vocab and draft_vocab_sha256 (got {vocab})")
    if vocab and d["draft_vocab_dataset"] not in (config.get("extra_dataset_sources") or []):
        raise SystemExit(f"dflash draft vocab dataset {d['draft_vocab_dataset']} is not in extra_dataset_sources")
    if not (config.get("sglang_tree") or {}).get("lock_target"):
        raise SystemExit("dflash needs a sglang-arc-2 tree (sglang_tree.lock_target): the worker lives in the fork")
    args = config["server_args"]
    val = lambda flag: args[args.index(flag) + 1] if flag in args else None  # noqa: E731
    k = int(val("--speculative-num-draft-tokens") or 0)
    if val("--speculative-algorithm") != "NEXTN" or not 2 <= k <= 8 or int(val("--speculative-num-steps") or 0) != k - 1 \
            or val("--speculative-eagle-topk") != "1" or "--speculative-use-rejection-sampling" not in args:
        raise SystemExit("dflash needs --speculative-algorithm NEXTN, --speculative-num-steps K-1, "
                         "--speculative-num-draft-tokens K (2..8), --speculative-eagle-topk 1 and rejection sampling")
    # the paths are resolved on the mounted dataset by 03_serve.dflash_env; behaviour switches (ARC_DFLASH_SPARSE_Q, ...)
    # are plain server_env
    resolved = {"ARC_DFLASH_DRAFTER", "ARC_DFLASH_KEEP", "ARC_DFLASH_CALIB", "ARC_DFLASH_DRAFT_VOCAB"}
    if resolved & set(config.get("server_env", {})):
        raise SystemExit(f"{sorted(resolved)} come from the dflash block (resolved on the mounted dataset), not server_env")


#: sglang-arc-2 commits whose tree serves plan B's plain / injected-context drafter (ffn_type dense) on SM120:
#: all-merge-1020, verified on G4 in the v28 configuration with the 8-layer drafter (journal 2026-10-20)
PLAIN_DFLASH_COMMITS = ("a5c67c41cd22933a153ff82d8b6c5a6ca6f458fb",)
#: what the plain drafter logs at start in place of the D-line drafter's own lines
PLAIN_DFLASH_LOG = ("residual plain (PlainDecoder", "context inject (pool conv boundaries none",
                    "DFlash exact prefill: a prompt's window from its last 1024 positions (window 1024: injected context)")
DLINE_DFLASH_LOG = ("dense experts: ", "'dense 10240'", "DFlash exact prefill: a prompt's window from its last 5135")


def check_plain_dflash(config: dict) -> None:
    """Plan B's 8-layer drafter (ffn_type dense): a tree that serves it, and its start lines required in place of the
    D-line drafter's (which it does not print, so a config carrying them would stop every run at the start check)."""
    commit = (config.get("sglang_tree") or {}).get("commit")
    if commit not in PLAIN_DFLASH_COMMITS:
        raise SystemExit(f"dflash ffn_type dense needs a plan B tree {PLAIN_DFLASH_COMMITS}, not {commit}")
    markers = config.get("require_log", [])
    missing = [m for m in PLAIN_DFLASH_LOG if m not in markers]
    stale = [m for m in markers if m.startswith(DLINE_DFLASH_LOG)]
    if missing or stale:
        raise SystemExit(f"plan B drafter require_log: add {missing}, drop the D-line lines {stale}")


def check_int3(config: dict) -> None:
    """Packed INT3 experts (work/qwen38fn_eval/int3_deploy_runtime.py, Humming uint3) served the way G4 served them
    (work/qwen38fn_eval/g4_serve_int3.py): the Humming wheel is mounted and pinned, the MoE runner is humming on
    TP1, and the two switches G4 drops are not set -- ARC_ONLINE_FP8 (the dense weights already carry FP8 scales)
    and ARC_PLE_NVFP4 (refused by the fork next to ARC_PLE_INT5)."""
    spec = config["int3"]
    missing = [k for k in ("humming_dataset", "humming_wheel", "humming_wheel_sha256", "humming_version")
               if not spec.get(k)]
    if missing:
        raise SystemExit(f"int3 lacks {missing}")
    # 0.1.18 starts only on a fork tree whose Humming runner aligns w2 to its own block M (int3_deploy_runtime
    # checks INDEXED_W2_OWN_ALIGNMENT at load: fork c61e351b7 or later, e.g. lock g4-humming0118-1021)
    if spec["humming_version"] not in ("0.1.12", "0.1.16", "0.1.18"):
        raise SystemExit(f"int3 humming_version {spec['humming_version']}: the INT3 plugin reviews 0.1.12 / 0.1.16 / "
                         f"0.1.18")
    if spec["humming_dataset"] not in (config.get("extra_dataset_sources") or []):
        raise SystemExit(f"int3 humming_dataset {spec['humming_dataset']} is not in extra_dataset_sources")
    if not (config.get("sglang_tree") or {}).get("lock_target"):
        raise SystemExit("int3 needs a sglang-arc-2 tree (sglang_tree.lock_target)")
    args = config["server_args"]
    val = lambda flag: args[args.index(flag) + 1] if flag in args else None  # noqa: E731
    if val("--moe-runner-backend") != "humming" or val("--tp") != "1" or val("--dtype") != "bfloat16":
        raise SystemExit("int3 needs --moe-runner-backend humming, --tp 1 and --dtype bfloat16")
    stray = sorted({"ARC_ONLINE_FP8", "ARC_PLE_NVFP4", "ARC_INT3_DEPLOY"} & set(config.get("server_env", {})))
    if stray:
        raise SystemExit(f"int3: drop {stray} from server_env")
    act = config.get("server_env", {}).get("ARC_INT3_ACT", "bf16")
    if act not in ("bf16", "fp8"):
        raise SystemExit(f"int3: ARC_INT3_ACT {act!r} is bf16 (W3A16) or fp8 (W3A8)")
    if not any(m.startswith("ARC_INT3_DIRECT_LOADED ") and m.endswith(f" act={act}") for m in config.get("require_log", [])) \
            and act != "bf16":
        raise SystemExit(f"int3: ARC_INT3_ACT={act} needs the 'ARC_INT3_DIRECT_LOADED ... act={act}' log marker")
    env, markers = config.get("server_env", {}), config.get("require_log", [])
    # the table path is resolved under the carried code on Kaggle (03_serve), like the dflash paths
    if "ARC_INT3_TUNING" in env:
        raise SystemExit("int3: ARC_INT3_TUNING comes from int3.tuning (a carried table), not server_env")
    if spec.get("tuning"):
        tuning = Path(spec["tuning"])
        if tuning.parent != Path(INT3_TUNING_DIR) or not (REPO_ROOT / tuning).is_file():
            raise SystemExit(f"int3 tuning {tuning} is not a table under {INT3_TUNING_DIR}")
        got = hashlib.sha256((REPO_ROOT / tuning).read_bytes()).hexdigest()
        if got != spec.get("tuning_sha256"):
            raise SystemExit(f"int3 tuning {tuning} is {got}, the config pins {spec.get('tuning_sha256')}")
        table = json.loads((REPO_ROOT / tuning).read_text())
        if (table["humming"], table["act"]) != (spec["humming_version"], act):
            raise SystemExit(f"int3 tuning {tuning} is for humming {table['humming']} act={table['act']}, "
                             f"the config runs humming {spec['humming_version']} act={act}")
        for sub in sorted(table["sublayers"]):
            if not any(m.startswith(f"ARC_INT3_TUNING {sub}: ") for m in markers):
                raise SystemExit(f"int3 tuning needs the 'ARC_INT3_TUNING {sub}: ...' log marker")
    elif "tuning_sha256" in spec:
        raise SystemExit("int3 tuning_sha256 without int3.tuning")
    for key in INT3_SWITCHES:
        if key in env:
            if env[key] not in ("0", "1"):
                raise SystemExit(f"int3: {key} {env[key]!r} is 0 or 1")
            line = f"{key}={env[key]}"
            if not any(m == line or m.startswith(line + " ") for m in markers):
                raise SystemExit(f"int3: {key} in server_env needs the '{line}' log marker")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=HERE / "notebook-reap320-0520.json")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    check(config)
    write(config, args.out)
    return 0


def write(config: dict, out: Path) -> None:
    """The notebook and its kernel metadata, unchecked: main checks the config first and is the only way to
    build one to push."""
    out.mkdir(parents=True, exist_ok=True)
    built = cells(config)
    (out / "arm.ipynb").write_text(json.dumps({
        "cells": [{"cell_type": kind, "metadata": {},
                   **({"execution_count": None, "outputs": []} if kind == "code" else {}),
                   "source": source.splitlines(keepends=True)}
                  for kind, source in built],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5}, indent=1))
    (out / "kernel-metadata.json").write_text(json.dumps({
        "id": config["kernel_slug"],
        "title": config["kernel_slug"].split("/")[-1].replace("-", " "),
        "code_file": "arm.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_internet": False,
        # PascalCase and only here; in the notebook's own metadata it is ignored
        # and the session silently lands on a P100.
        "machine_shape": "NvidiaRtxPro6000",
        "competition_sources": [COMPETITION],
        "dataset_sources": [ASSETS_DATASET, PIP_DATASET, RUNTIME_DATASET, ZSTD_DATASET,
                            *(config.get("extra_dataset_sources") or [])],
        "model_sources": [config["model_source"],
                          *([config["model_overlay"]["model"]] if "model" in (config.get("model_overlay") or {})
                            else []),
                          *([config["model_patch"]["model"]] if config.get("model_patch") else []),
                          *([config["weight_switch"]["model"]] if config.get("weight_switch") else [])],
        "docker_image": DOCKER_IMAGE,
    }, indent=1))
    total = sum(len(source) for _, source in built)
    print(f"wrote {out}/arm.ipynb: {len(built)} cells, {total} chars, "
          f"{len(carried())} carried files; build {provenance(carried())['commit'][:7]}"
          + (" DIRTY" if provenance(carried())["dirty"] else ""))


if __name__ == "__main__":
    raise SystemExit(main())
