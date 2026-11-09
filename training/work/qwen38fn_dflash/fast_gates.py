"""Production authorization for the fast engine's pool training (train.py --pool without --smoke): gates.py's role for
the reference engine's queue epoch, bound to the same source hashes (supervise.validate_source).

    python fast_gates.py --features <capture job dir> --resume-evidence <pool_resume_check.json> --objective acceptance \
        --out <gate.json>

Runs every test of the unit -- the CPU contracts, and with --features on a GPU the fast engine against the reference
(test_fast_train_gpu.py: clean trace, proposals, loss and gradients, both objectives' head terms) and the head kernel
-- and requires the pool resume evidence (probes/pool_resume_check.py: a run stopped mid-way and resumed from its
checkpoint consumed every planned job, the in-flight ones twice and nothing else). The gate is per objective.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True, type=Path, help="a capture job dir with >= 512 anchors in its tail")
    p.add_argument("--resume-evidence", required=True, type=Path)
    p.add_argument("--objective", required=True, choices=("dpace", "acceptance"))
    p.add_argument("--transition-evidence", type=Path,
                   help="robust-v2 multi-rank full-state transition gate evidence")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()
    evidence = json.loads(args.resume_evidence.read_text())
    if not evidence["passed"] or evidence["objective"] != args.objective:
        raise RuntimeError(f"pool resume evidence does not pass for {args.objective}: {evidence}")
    unit = Path(__file__).resolve().parent
    robust_version = None
    if args.transition_evidence is not None:
        from robust_objective import VERSION
        from supervise import validate_source

        transition = json.loads(args.transition_evidence.read_text())
        if args.objective != "acceptance" or transition["robust_version"] != VERSION:
            raise ValueError("transition gate is for robust-v2 acceptance")
        required = ("passed", "parent_state_restored", "child_state_restored", "rng_restored", "consumed_restored",
                    "ramp_not_reset", "continued_parameters_and_adam_bitwise_equal")
        if not all(transition[k] is True for k in required) or transition["world_size"] < 2:
            raise ValueError("full-state multi-rank objective transition did not pass")
        validate_source(unit, transition)
        probe = unit / "probes" / "robust_transition_gate.py"
        if hashlib.sha256(probe.read_bytes()).hexdigest() != transition["probe_sha256"]:
            raise ValueError("transition probe changed after its evidence")
        robust_version = VERSION
    # The repo's own src first: the venv's editable arcforge may be another checkout's (capture tests import it).
    src = str(unit.parents[1] / "src")
    env = dict(os.environ, DFLASH_FEATURES=str(args.features.resolve()),
               PYTHONPATH=os.pathsep.join(filter(None, (src, os.environ.get("PYTHONPATH")))))
    tests = sorted(unit.glob("test_*.py"))
    # -rs lists skips: a GPU test that skipped (no GPU, no features) must fail the gate, not pass it silently.
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-rs", "-p", "no:logging", *map(str, tests)],
                         cwd=unit, env=env, capture_output=True, text=True)
    args.out.with_suffix(".pytest.log").write_text(run.stdout + run.stderr)
    print(run.stdout[-4000:])
    if run.returncode != 0:
        raise RuntimeError("tests failed")
    skipped = [line for line in run.stdout.splitlines() if line.startswith("SKIPPED")]
    gpu_skips = [line for line in skipped if "_gpu.py" in line]
    if gpu_skips:
        raise RuntimeError(f"GPU equivalence tests skipped: {gpu_skips}")
    result = {
        "production_ready": True,
        "engine": "fast",
        "objective": args.objective,
        "compute_dtype": "bfloat16",
        "attention_backend": "flex",
        "source_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in unit.glob("*.py")},
        "features": str(args.features.resolve()),
        "resume_evidence": str(args.resume_evidence.resolve()),
        "tests_summary": run.stdout.strip().splitlines()[-1],
        "skipped": skipped,
        "robust_version": robust_version,
        "transition_evidence": str(args.transition_evidence) if args.transition_evidence else None,
        # the objective the resume evidence trained with: train.py lets a prefix-gate run start only on its own
        "prefix_gate_version": evidence["prefix_gate"]["version"] if evidence["prefix_gate"] is not None else None,
        "prefix_gate": evidence["prefix_gate"],
    }
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "source_hashes"}, indent=2))


if __name__ == "__main__":
    main()
