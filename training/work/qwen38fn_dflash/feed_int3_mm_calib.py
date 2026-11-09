"""Replay a bounded prefix of the existing MM calibration plan; verify then discard GR.

Uses the original calib split and sequence order, never reselects a pruning mask.
Scheduler PIDs are explicit owned children of int3_mm_calib_runtime.py. All media,
ledger and recorded configs must be staged locally; no source NFS access is needed.
The complete receipt requires every replay check and every rank's Hessian dump.
"""
import argparse
import hashlib
import json
import re
import shutil
import time
import uuid
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def local(path):
    path = Path(path).resolve(strict=True)
    if path == Path("/nfs") or Path("/nfs") in path.parents:
        raise RuntimeError(f"stage locally first: {path}")
    return path


def load_selection(plan, target_tokens):
    if target_tokens <= 0:
        raise ValueError("target tokens must be positive")
    meta = json.loads((plan.parent / "plan_meta.json").read_text())
    if digest(plan) != meta["plan_sha256"]:
        raise RuntimeError("plan SHA256 differs from its metadata")
    jobs = [json.loads(line) for line in plan.read_text().splitlines()]
    if len(jobs) != meta["jobs"] or [j["index"] for j in jobs] != list(range(len(jobs))):
        raise RuntimeError("plan job count / indices differ from metadata")
    chosen, total = [], 0
    for job in jobs:
        if job["split"] != "calib":
            continue
        if job["pool"] not in ("training1000", "nvarc_training") or job["n_videos"]:
            raise RuntimeError("this driver accepts only the original clean ARC2 calib split")
        if job["num_tokens"] <= 0:
            raise RuntimeError("invalid segment token count")
        chosen.append(job)
        total += job["num_tokens"]
        if total >= target_tokens:
            break
    if total < target_tokens:
        raise RuntimeError("plan cannot satisfy calibration token budget")
    if not sum(j["n_images"] for j in chosen):
        raise RuntimeError("selected calibration prefix contains no images")
    return meta, chosen


def check_pids(pids, launcher_pid):
    if not pids or len(set(pids)) != len(pids):
        raise RuntimeError("scheduler PIDs must be nonempty and distinct")
    for pid in pids:
        proc = Path("/proc") / str(pid)
        status = dict(line.split(":", 1) for line in (proc / "status").read_text().splitlines() if ":" in line)
        cmd = (proc / "cmdline").read_bytes()
        if int(status["PPid"].strip()) != launcher_pid:
            raise RuntimeError(f"PID {pid} is not a child of the owned launcher {launcher_pid}")
        if "scheduler" not in status["Name"] and b"scheduler" not in cmd:
            raise RuntimeError(f"PID {pid} is not a scheduler")


def check_dump(hess, server, pids):
    lo, hi = map(int, server["layers"].split(":"))
    markers = [hess / f"hess_rank{r}_done.json" for r in range(server["n_ranks"])]
    if not all(p.is_file() for p in markers):
        return None
    done = [json.loads(p.read_text()) for p in markers]
    if {d["pid"] for d in done} != set(pids):
        raise RuntimeError("Hessian completion PIDs differ from the armed schedulers")
    for rank, rec in enumerate(done):
        for key in ("run_id", "mask_sha256", "mtp_mask_sha256"):
            if rec[key] != server[key]:
                raise RuntimeError(f"rank {rank}: Hessian {key} differs from this server")
        if rec["rank"] != rank or rec["layer_window"] != [lo, hi] or rec["layers"] != list(range(lo, hi)):
            raise RuntimeError(f"rank {rank}: wrong Hessian rank / layer window")
        if not rec["complete"] or rec["armed"]:
            raise RuntimeError(f"rank {rank}: incomplete / not frozen Hessian collection")
        if rec["src"] != server["model_path"] or rec["n_experts"] != server["n_experts"]:
            raise RuntimeError(f"rank {rank}: Hessian source / expert count differs from server")
        if rec["n_local"] * server["n_ranks"] != server["n_experts"]:
            raise RuntimeError(f"rank {rank}: Hessian EP layout differs from server")
        if set(rec["coverage"]) != {str(layer) for layer in range(lo, hi)} or any(
            cov["n_hit"] <= 0 for cov in rec["coverage"].values()
        ):
            raise RuntimeError(f"rank {rank}: no calibration samples for a layer")
        expected = {f"hess_rank{rank}_L{layer:02d}.pt" for layer in range(lo, hi)}
        expected.add(f"hess_rank{rank}_meta.json")
        if set(rec["files"]) != expected or not all((hess / name).stat().st_size > 0 for name in expected):
            raise RuntimeError(f"rank {rank}: incomplete Hessian file list")
    return done


def check_hook_receipts(work, n_tokens, n_ranks, n_layers):
    expected = {str(layer): {"gateup": 1, "down": 1} for layer in range(n_layers)}
    summary = []
    for rank in range(n_ranks):
        files = sorted((work / "hesshooks" / f"rank{rank}").glob("*.json"))
        position = 0
        for path in files:
            rec = json.loads(path.read_text())
            if rec["rank"] != rank or rec["prefix"] != position or rec["tokens"] <= 0 or rec["calls"] != expected:
                raise RuntimeError(f"rank {rank}: incomplete, duplicate or misaligned native Hessian hook receipt")
            position += rec["tokens"]
        if position != n_tokens:
            raise RuntimeError(f"rank {rank}: Hessian hook chunks do not cover all {n_tokens} prompt tokens")
        summary.append({"rank": rank, "chunks": len(files), "tokens": position, "once_per_layer": True})
    return summary


def write_control(root, run_id, action):
    token = uuid.uuid4().hex
    tmp = root / "hessian_control.partial"
    tmp.write_text(json.dumps({"run_id": run_id, "action": action, "token": token}) + "\n")
    tmp.replace(root / "hessian_control.json")
    return token


def check_hessian_status(root, server, pids, token, tokens, *, armed):
    lo, hi = map(int, server["layers"].split(":"))
    rows = [json.loads((root / "hessian_status" / f"rank{rank}.json").read_text())
            for rank in range(server["n_ranks"])]
    if {row["pid"] for row in rows} != set(pids):
        raise RuntimeError("actual Hessian status comes from different schedulers")
    for rank, row in enumerate(rows):
        if (row["rank"] != rank or row["run_id"] != server["run_id"] or row["token"] != token or
                row["armed"] != armed or set(row["banks"]) != {str(layer) for layer in range(lo, hi)}):
            raise RuntimeError("actual Hessian control epoch / layer set / armed status differs")
        for banks in row["banks"].values():
            if set(banks) != {"gateup", "down"} or any(bank["n_hit"] <= 0 for bank in banks.values()):
                raise RuntimeError("actual Hessian matrices are missing or contain no calibration samples")
    for layer in range(lo, hi):
        for bank in ("gateup", "down"):
            total = sum(row["banks"][str(layer)][bank]["nsamp_sum"] for row in rows)
            if total != 10 * tokens:
                raise RuntimeError(f"actual H samples layer {layer} {bank}: {total} != 10 * {tokens}; "
                                   "collection skipped, duplicated, capped or pulse-contaminated")
    return rows


def run(args):
    import capture_tito as ct
    import capture_tito_runtime as rt
    import requests

    from arcforge.tito.ledger import Vocab, ids_sha256, load_instance

    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
        raise ValueError("run id must contain only letters, numbers, underscore and hyphen")
    plan, model, ledger, configs = map(local, (args.plan, args.model_dir, args.ledger_root, args.recorded_config_dir))
    root, hess = map(local, (args.capture_root, args.hess_dir))
    meta, jobs = load_selection(plan, args.target_tokens)
    server = json.loads((root / "mm_calib_server.json").read_text())
    if server["run_id"] != args.run_id or server["n_ranks"] != len(args.scheduler_pids):
        raise RuntimeError("server run identity / rank count differs from driver")
    if Path(server["model_path"]).resolve() != model or digest(model / "config.json") != server["config_sha256"]:
        raise RuntimeError("server is not the staged calibration model")
    if not all(server["isolation"].values()) or (root / "active.json").exists():
        raise RuntimeError("server is not isolated or another job is active")
    if server["hessian_bridge"] != "native-triton-file-control-v2" or server["max_nsamp"] != 2147483647:
        raise RuntimeError("file-controlled, uncapped Hessian calibration server required")
    if list(hess.glob("hess_rank*")):
        raise RuntimeError("Hessian output already contains a previous dump")
    check_pids(args.scheduler_pids, server["launcher_pid"])
    max_len, _ = ct.check_server(args.url, server["served_model_name"])
    vocab = Vocab(model)
    spec = ct.ModelSpec(model, vocab)
    target_spec = rt.input_spec(json.loads((model / "config.json").read_text()))
    instances, verified_media = {}, set()
    recording_policies = {}
    used = {j["instance"] for j in jobs}
    for rec in meta["instances"]:
        if rec["name"] not in used:
            continue
        path = local(ledger / rec["name"])
        if digest(path / "server.json") != rec["server_json_sha256"]:
            raise RuntimeError(f"{path}: recording server digest differs from the plan")
        recorded = json.loads((path / "server.json").read_text())
        recording_policies[rec["name"]] = {"ple_mm_placeholder_ids":
            "--ple-mm-placeholder-ids" in recorded["launch_command"].split()}
        want = recorded["model_files_sha256"]
        for name in rt.INPUT_FILES:
            if digest(model / name) != want[name]:
                raise RuntimeError(f"target {name} differs from recording")
        raw_config = local(configs / (want["config.json"] + ".json"))
        if digest(raw_config) != want["config.json"] or rt.input_spec(json.loads(raw_config.read_text())) != target_spec:
            raise RuntimeError("target input config differs from the recording")
        instances[rec["name"]] = load_instance(path, vocab)
    args.out.mkdir(parents=True, exist_ok=False)
    receipt = {"complete": False, "run_id": args.run_id, "plan_sha256": meta["plan_sha256"],
               "selection": jobs, "server": server, "scheduler_pids": args.scheduler_pids,
               "recording_input_policies": recording_policies,
               "target_tokens": args.target_tokens, "tainted": False, "requests": []}
    (args.out / "selected.json").write_text(json.dumps(receipt, indent=1) + "\n")
    arm_token = write_control(root, args.run_id, "arm")
    replayed_tokens = 0
    for job in jobs:
        seg = instances[job["instance"]].segments[job["seg"]]
        if ids_sha256(seg.ids) != job["ids_sha256"] or len(seg) != job["num_tokens"]:
            raise RuntimeError("selected segment ids differ from the original plan")
        reason = ct.skip_reason(seg, max_len)
        if reason is not None:
            raise RuntimeError(f"selected segment cannot replay: {reason}")
        inst = instances[job["instance"]]
        for media in seg.media:
            key = (inst.name, media.sha256)
            if key not in verified_media:
                if digest(local(inst.media_path(media.sha256))) != media.sha256:
                    raise RuntimeError("recorded media bytes fail SHA256")
                verified_media.add(key)
        ids, items, body = ct.segment_request(inst, seg, vocab, spec, "high")
        job_id = f"int3-mm-{args.run_id}-{job['index']:09d}"
        work = ct.open_job(root, job_id)
        body.update(rid=job_id, sampling_params={"max_new_tokens": 1, "temperature": 0})
        began = time.monotonic()
        info = ct.run_request(args.url, root, {"job_id": job_id, "num_tokens": len(ids)}, body)
        if info.get("cached_tokens", 0) != 0:
            raise RuntimeError("cached prefill would omit calibration tokens")
        cap = ct.assemble(work / "parts", len(ids))
        ct.check_capture(ids, items, cap, spec)
        hook_receipts = check_hook_receipts(work, len(ids), server["n_ranks"], server["n_layers"])
        replayed_tokens += len(ids)
        actual_hess = check_hessian_status(root, server, args.scheduler_pids, arm_token, replayed_tokens, armed=True)
        if not receipt["requests"]:
            (args.out / "first_segment_hessian_gate.json").write_text(json.dumps(actual_hess, indent=1) + "\n")
        actual_policy = bool(cap["ple_placeholder_ids"][0])
        if actual_policy != server["ple_mm_placeholder_ids"]:
            raise RuntimeError("captured media ids differ from the requested server input policy")
        rec = {**job, "seconds": time.monotonic() - began, "prompt_tokens": info["prompt_tokens"],
               "cached_tokens": info.get("cached_tokens", 0), "token_mrope_gr_verified": True,
               "native_hessian_hooks": hook_receipts,
               "actual_hessian_total_tokens_verified": replayed_tokens,
               "ple_placeholder_ids": actual_policy,
               "ple_policy_changed_from_recording": actual_policy !=
               recording_policies[job["instance"]]["ple_mm_placeholder_ids"]}
        del cap
        shutil.rmtree(work)
        receipt["requests"].append(rec)
        with (args.out / "requests.jsonl").open("a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"verified {len(receipt['requests'])}/{len(jobs)}: {job['key']} {len(ids)} tokens", flush=True)
    if (root / "active.json").exists():
        raise RuntimeError("active capture remains before dump pulse")
    check_pids(args.scheduler_pids, server["launcher_pid"])
    dump_token = write_control(root, args.run_id, "freeze_dump")
    pulse = requests.post(args.url.rstrip("/") + "/generate", json={"input_ids": [1] * 4096,
                          "sampling_params": {"max_new_tokens": 1, "temperature": 0}}, timeout=args.timeout)
    pulse.raise_for_status()
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        dumped = check_dump(hess, server, args.scheduler_pids)
        if dumped is not None:
            statuses = [json.loads((root / "hessian_status" / f"rank{rank}.json").read_text())
                        for rank in range(server["n_ranks"])]
            if any(row["token"] != dump_token for row in statuses):
                time.sleep(3)  # done marker precedes the wrapper's final status write
                continue
            final_hess = check_hessian_status(root, server, args.scheduler_pids, dump_token, replayed_tokens, armed=False)
            receipt.update(complete=True, dump=dumped, total_tokens=sum(j["num_tokens"] for j in jobs),
                           actual_hessian_status=final_hess, file_control=True, dump_pulse_tokens=4096,
                           total_images=sum(j["n_images"] for j in jobs), total_videos=0)
            tmp = args.out / "complete.partial"
            tmp.write_text(json.dumps(receipt, indent=1) + "\n")
            tmp.rename(args.out / "complete.json")
            print(f"complete {receipt['total_tokens']} tokens; all rank/layer dumps verified", flush=True)
            return
        time.sleep(3)
    raise RuntimeError("Hessian dump did not complete before timeout")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "ledger-root", "model-dir", "recorded-config-dir", "capture-root", "hess-dir", "out"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--url", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--scheduler-pids", required=True, nargs="+", type=int)
    p.add_argument("--target-tokens", type=int, default=2_000_000)
    p.add_argument("--timeout", type=float, default=3600)
    run(p.parse_args())


if __name__ == "__main__":
    main()
