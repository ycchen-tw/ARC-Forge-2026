"""The pool's resume evidence for fast_gates.py: a pipeline.sh SMOKE=1 run stopped mid-way (trainer/STOP) and resumed
(RESUME=1) must have trained every train job of its plan exactly once, ended drained, and shown held-out evaluations.

    python probes/pool_resume_check.py --plan <plan dir> --run <$ARC_RUNS/qwen38fn_dflash/<tag>> --objective acceptance \
        --out <evidence.json>

Reads the run's final checkpoint (trainer/latest: its consumed.jsonl is the whole run's list), run-result.json (the
resumed segment's start_step is the stop's checkpoint) and eval.jsonl.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_plan import load_plan  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True, type=Path)
    p.add_argument("--run", required=True, type=Path)
    p.add_argument("--objective", required=True)
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    _, jobs = load_plan(args.plan / "plan.jsonl")
    planned = Counter((j["key"], j["pass"]) for j in jobs if j["split"] == "train")
    trainer = args.run / "trainer"
    latest = trainer / (trainer / "latest").read_text().strip()
    consumed = [json.loads(line) for line in (latest / "consumed.jsonl").read_text().splitlines()]
    got = Counter((c["key"], c["pass"]) for c in consumed)
    result = json.loads((trainer / "run-result.json").read_text())
    evals = [json.loads(line) for line in (trainer / "eval.jsonl").read_text().splitlines()]
    config = json.loads((trainer / "run-config.json").read_text())
    evidence = {
        "objective": config["objective"]["name"],
        "planned_train_jobs": sum(planned.values()),
        "consumed_entries": len(consumed),
        "missing": sorted(map(list, planned - got))[:20],
        "extra_or_repeated": sorted(map(list, got - planned))[:20],
        "resumed_from_step": result["start_step"],
        "consumed_before_resume": sum(c["step"] <= result["start_step"] for c in consumed),
        "final_step": result["steps"],
        "latest_checkpoint": latest.name,
        "drained": result["drained"],
        "evaluations": [e["step"] for e in evals],
        # run-config.json names these only when set (train.py): a baseline run has none of them
        "prefix_gate": config.get("prefix_gate"),
        "lr_schedule": config.get("lr_schedule", "cosine"),
        "cov_weight_final": config.get("cov_weight_final"),
    }
    gated = True
    if evidence["prefix_gate"] is not None:
        # The gate must have been in every step's loss: its metrics on each record, no non-finite tail.
        steps = [json.loads(line) for line in (trainer / "metrics.jsonl").read_text().splitlines()]
        evidence["gate_steps"] = sum("gate_weight_mean" in r for r in steps)
        evidence["gate_nonfinite"] = sum(r["gate_nonfinite"] for r in steps if "gate_weight_mean" in r)
        evidence["gate_weight_mean_last"] = steps[-1]["gate_weight_mean"] if steps else None
        gated = bool(steps) and evidence["gate_steps"] == len(steps) and evidence["gate_nonfinite"] == 0
    evidence["passed"] = bool(
        gated and evidence["objective"] == args.objective
        and not evidence["missing"] and not evidence["extra_or_repeated"]
        and 0 < result["start_step"] < result["steps"] and 0 < evidence["consumed_before_resume"] < len(consumed)
        and result["drained"] and len(evals) >= 3
    )
    with args.out.open("x") as stream:
        json.dump(evidence, stream, indent=2)
    print(json.dumps(evidence, indent=2))
    if not evidence["passed"]:
        raise SystemExit("resume check failed")


if __name__ == "__main__":
    main()
