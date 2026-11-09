"""Pin a local parent checkpoint and prepare a full-state robust-v2 continuation.

No GPU work and no parent mutation. Only an explicit immutable checkpoint is
accepted. Hard links preserve its bytes when the parent's retention removes it.
"""

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict
from pathlib import Path

from local_storage import require_local
from robust_objective import RobustConfig, VERSION


def sha256(path):
    path = require_local(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_parent(config):
    objective = config["objective"]
    if (objective["name"] != "acceptance" or objective["temperature"] != 0.7
            or objective["cov_weight"] != 0.3 or not objective["joint"]
            or objective["opd_weight"] or objective["accept_weight"] != 1):
        raise ValueError("robust v2 requires the unchanged offline acceptance recipe")
    if config["smoke"] or config["step_schedule"] or config["rewarm_steps"] or not config["total_anchors"]:
        raise ValueError("robust v2 requires an existing production anchor schedule, without LR rewarm")
    if "robust_objective" in config or "robust_transition_sha256" in config:
        raise ValueError("parent is already robust; resume its child instead of resetting the ramp")


def validate_config(config, spec):
    expected = dict(spec["parent_config"])
    expected["robust_objective"] = spec["robust_objective"]
    # The digest binds the transition file, whose parent config must remain exact.
    actual = dict(config)
    actual.pop("robust_transition_sha256")
    if actual != expected:
        changed = sorted(k for k in set(actual) | set(expected) if actual.get(k) != expected.get(k))
        raise ValueError(f"objective transition changes protected configuration fields: {changed}")


def load_spec(path):
    path = require_local(path)
    spec = json.loads(path.read_text())
    if spec["version"] != VERSION or spec["robust_objective"] != asdict(RobustConfig()):
        raise ValueError("unsupported robust recipe")
    validate_parent(spec["parent_config"])
    if spec["ramp_anchors"] != 128 * spec["parent_config"]["step_anchors"]:
        raise ValueError("transition ramp differs from the fixed v2 recipe")
    start, total = spec["start_anchors"], spec["parent_config"]["total_anchors"]
    if not 0 <= start < total or total - start < spec["ramp_anchors"]:
        raise ValueError("parent has insufficient original budget remaining for the objective ramp")
    return spec, sha256(path)


def mix_at(spec, anchors_done):
    if anchors_done < spec["start_anchors"]:
        raise ValueError("restored anchor counter precedes the objective transition")
    return min(1.0, (anchors_done - spec["start_anchors"]) / spec["ramp_anchors"])


def validate_restore(saved, spec, digest):
    if "robust_transition_sha256" in saved:
        if saved["robust_transition_sha256"] != digest:
            raise ValueError("checkpoint belongs to another objective transition")
        if any(saved.get(k) != v for k, v in checkpoint_state(spec, digest).items()):
            raise ValueError("checkpoint objective version or ramp metadata changed")
    elif (saved["step"] != spec["parent_step"] or saved["anchors_done"] != spec["start_anchors"]):
        raise ValueError("initial transition must restore the pinned parent's exact counters")
    mix_at(spec, saved["anchors_done"])


def checkpoint_state(spec, digest):
    return {"robust_transition_sha256": digest, "robust_version": VERSION,
            "transition_start_anchors": spec["start_anchors"], "ramp_anchors": spec["ramp_anchors"]}


def validate_launch(spec, env):
    """Reject recipe drift before pipeline.sh starts any capture servers."""
    config, launch = spec["parent_config"], spec["parent_launch"]
    for key, field, cast in (("STEP_ANCHORS", "step_anchors", int), ("MICRO_ANCHORS", "micro_anchors", int),
                             ("JOBS_PER_RANK", "jobs_per_rank", int), ("LR_SCALE", "lr_scale", float),
                             ("NEW_LR_MULT", "new_lr_mult", float), ("REDUCER_LR", "reducer_lr", float)):
        if cast(env[key]) != config[field]:
            raise ValueError(f"robust launch changes {field}")
    maximum = int(env["MAX_RANK_TOKENS"]) if env["MAX_RANK_TOKENS"] else None
    if maximum != config["max_rank_tokens"]:
        raise ValueError("robust launch changes max_rank_tokens")
    if (float(env["TEMPERATURE"]) != .7 or float(env["COV_WEIGHT"]) != .3
            or float(env["OFFLINE_ACCEPT_WEIGHT"]) != 1 or env["WARM_START"] or env["OPD_ROOT"]
            or env["STEP_SCHEDULE"] != "0" or env["RESUME"] != "1" or env["SMOKE"] != "0"):
        raise ValueError("robust launch changes the offline objective or schedule")
    trainers, captures = env["TRAINER_GPUS"].split(","), env["CAPTURE_GPUS"].split(",")
    all_gpus = trainers + captures
    if len(set(all_gpus)) != len(all_gpus) or not all(g.isdigit() for g in all_gpus):
        raise ValueError("GPU allocation must contain distinct numeric devices")
    if len(trainers) != config["world_size"] or len(captures) != len(launch["capture_gpus"].split(",")):
        raise ValueError("robust launch changes trainer/capture topology")
    if str(require_local(env["INIT"])) != config["checkpoint"]:
        raise ValueError("robust launch changes frozen/init tensors")
    plan = require_local(env["PLAN"])
    if sha256(plan / "summary.json") != config["dataset_manifest_sha256"]:
        raise ValueError("robust launch changes data-plan summary")
    if sha256(plan / "plan.jsonl") != launch["plan_sha256"]:
        raise ValueError("robust launch changes data-plan jobs")
    profile = json.loads(require_local(plan / "plan_meta.json").read_text())["target_profile"]["path"]
    if sha256(profile) != spec["target_profile_sha256"]:
        raise ValueError("robust launch changes capture profile")
    # A hard NFS mount can make a process unkillable. Never fall back to it.
    require_local(env["CAPTURE_MODEL_PATH"])
    require_local(env["HELDOUT_FROM"])


def prepare(parent, out):
    parent, out = require_local(parent), require_local(out)
    if not parent.name.startswith("step-") or parent.name.endswith(".partial"):
        raise ValueError("give an explicit completed step directory, not latest or a partial checkpoint")
    if out.exists():
        raise FileExistsError(out)
    config = json.loads(require_local(parent.parent / "run-config.json").read_text())
    validate_parent(config)
    init_config = json.loads(require_local(Path(config["checkpoint"]) / "config.json").read_text())
    if init_config.get("attention_sink", False):
        raise ValueError("the v2 plan continues the non-sink baseline")
    manifest = json.loads(require_local(parent / "manifest.json").read_text())
    if manifest["format"] != "torch-zstd-v1" or not manifest["replicated"]:
        raise ValueError("transition requires the replicated fast-engine checkpoint")
    if manifest["world_size"] != config["world_size"]:
        raise ValueError("checkpoint topology does not match parent run config")
    if manifest["source_init"] != config["checkpoint"]:
        raise ValueError("checkpoint source initialization differs from parent config")
    if sha256(manifest["dataset_manifest"]) != config["dataset_manifest_sha256"]:
        raise ValueError("checkpoint data plan differs from parent config")
    counters = None
    for line in require_local(parent.parent / "metrics.jsonl").read_text().splitlines():
        # The checkpoint is immutable, but an active parent's final metrics line may be incomplete.
        if not line.endswith("}"):
            continue
        row = json.loads(line)
        if row["step"] == manifest["step"]:
            counters = row
            break
    if counters is None:
        raise ValueError("parent metrics do not contain the checkpoint's exact anchor counter")
    launch = json.loads(require_local(parent.parent.parent / "launch.json").read_text())
    spec = {"version": VERSION, "robust_objective": asdict(RobustConfig()),
            "parent_config": config, "parent_step": manifest["step"],
            "start_anchors": counters["anchors_done"], "ramp_anchors": 128 * config["step_anchors"],
            "parent_git_commit": launch["git_commit"], "parent_checkpoint": str(parent),
            "parent_launch": launch, "target_profile_sha256": sha256(launch["target_profile"]),
            "checkpoint_hashes": {}}
    if config["total_anchors"] - spec["start_anchors"] < spec["ramp_anchors"]:
        raise ValueError("not enough original training budget remains")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.parent.stat().st_dev != parent.stat().st_dev:
        raise ValueError("pinning requires the same local filesystem; stage a verified copy explicitly first")
    if shutil.disk_usage(out.parent).free < 24 * 2**30:
        raise RuntimeError("less than 24 GiB of local disk reserve")
    out.mkdir()
    pin = out / "parent-checkpoint"
    pin.mkdir()
    # Pin before hashing: the active parent's next checkpoint may unlink the old directory.
    for name in ("replicated.pt.zst", "manifest.json", "consumed.jsonl"):
        os.link(require_local(parent / name), pin / name)
    for name in ("replicated.pt.zst", "manifest.json", "consumed.jsonl"):
        spec["checkpoint_hashes"][name] = sha256(pin / name)
    transition = out / "objective-transition.json"
    transition.write_text(json.dumps(spec, indent=2) + "\n")
    spec, digest = load_spec(transition)
    child = config | {"robust_objective": spec["robust_objective"], "robust_transition_sha256": digest}
    validate_config(child, spec)
    (out / "run-config.json").write_text(json.dumps(child, indent=2) + "\n")
    (out / "latest").write_text("parent-checkpoint\n")
    (out / "PAUSED.json").write_text(json.dumps({"status": "prepared_not_started", "automatic_start": False,
                                              "required": "fresh robust GPU and full-state transition gates"}, indent=2))
    return transition


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-launch", type=Path)
    parser.add_argument("--parent-checkpoint", type=Path)
    parser.add_argument("--out", type=Path, help="new child trainer directory")
    args = parser.parse_args()
    if args.check_launch is not None:
        validate_launch(load_spec(args.check_launch)[0], os.environ)
        print("robust launch recipe validated")
        return
    if args.parent_checkpoint is None or args.out is None:
        parser.error("give --parent-checkpoint and --out, or --check-launch")
    print(prepare(args.parent_checkpoint, args.out))


if __name__ == "__main__":
    main()
