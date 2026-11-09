"""Collect a fresh FP8 base baseline with the run's heldout tasks and matched 32k harness."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from orchestrator import Orchestrator, OrchestratorError, atomic_write, cfg_path, http_ok, load_config
from heldout_offline import ledger_status, server_identity


def condition_fingerprint(orch: Orchestrator, samples: int) -> str:
    condition = {"harness": json.loads(cfg_path(orch.cfg["harness"]["config"]).read_text()),
                 "heldout": orch.pools["heldout"], "samples": samples,
                 "tokenizer_dir": orch.cfg["tokenizer_dir"]}
    return hashlib.sha256(json.dumps(condition, sort_keys=True).encode()).hexdigest()


def collect(orch: Orchestrator, samples: int) -> dict:
    """Collect one fresh base condition; caller owns cancellation and the run lock."""
    cfg, out = orch.cfg, orch.root / "baseline-b32k"
    if out.exists():
        raise OrchestratorError(f"{out} exists; baseline output is never reused")
    if not http_ok(orch.server_root + "/health"):
        raise OrchestratorError("baseline server is not healthy")
    info = server_identity(orch, cfg["rollout"]["served_name"], "baseline")
    if str(info["weight_version"]) != "0" or info["weights_source"] != info["model_path"]:
        raise OrchestratorError("baseline requires base weights at version 0")
    before = ledger_status(orch, None, "baseline start")
    condition = condition_fingerprint(orch, samples)
    orch.served_name = cfg["rollout"]["served_name"]
    out.mkdir()
    (out / "logs").mkdir()
    orch.log_file = (out / "baseline.log").open("a")
    try:
        result = orch.sample_runs(out, out / "logs", "", Path(cfg["tasks"]["heldout"]["dir"]),
                                  orch.pools["heldout"], samples, timeout=14400, what="matched32k baseline")
        after = ledger_status(orch, before["instance"], "baseline end")
        final = server_identity(orch, orch.served_name, "baseline end")
        if final["weight_version"] != info["weight_version"] or final["weights_source"] != info["weights_source"]:
            raise OrchestratorError("baseline weights changed during evaluation")
        if condition_fingerprint(orch, samples) != condition:
            raise OrchestratorError("baseline condition changed during collection")
        doc = {**result, "condition_sha256": condition, "model_info": info,
               "ledger_before": before, "ledger_after": after}
        atomic_write(out / "summary.json", json.dumps(doc, indent=1) + "\n")
        return doc
    finally:
        orch.log_file.close()
        orch.log_file = None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:30571/v1")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.samples < 1 or not args.base_url.endswith("/v1"):
        ap.error("samples must be positive, base-url must end in /v1")
    cfg = load_config(args.config)
    cfg["rollout"]["base_url"] = args.base_url
    cfg["harness"]["config"] = str(Path(cfg["run_root"]) / "harness-eval.json")
    orch = Orchestrator(cfg)
    orch.static_checks(salted=False)
    out = orch.root / "baseline-b32k"
    if out.exists():
        raise OrchestratorError(f"{out} exists; baseline output is never reused")
    if args.check:
        print(f"baseline: {len(orch.pools['heldout'])} tasks x {args.samples} samples, out {out}")
        return
    collect(orch, args.samples)


if __name__ == "__main__":
    main()
