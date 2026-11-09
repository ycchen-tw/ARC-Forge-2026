#!/usr/bin/env python3
"""Reproducible inventory of exactly 1,000 pinned ARC-AGI-2 training IDs.

Missing directories stay missing. Stored seed.json verification is never accepted
as completion: PASS requires a current hash-bound audit of all pairs and 200+
inputs across RNG 0, 1, 17, 20261003. No seed or official data is changed.
Use --semantic-reports for the release gate, which also requires a current
successful replay of retained independent construction/counterexample tests.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
BANK = ROOT / "work/seed_bank"
DATASET_REVISION = "f3283f727488ad98fe575ea6a5ac981e4a188e49"
EXPECTED_COUNT = 1000


def _load_audit():
    spec = importlib.util.spec_from_file_location("seed_bank_inventory_audit", Path(__file__).with_name("audit.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def official_tasks(dataset_root, expected_count=EXPECTED_COUNT, expected_revision=DATASET_REVISION):
    """Check the data pin, the list, and the complete ID set; tests may pass a smaller fixture count."""
    root = Path(dataset_root)
    directory = root / "data/training"
    tasks = {path.stem: path for path in sorted(directory.glob("*.json"))}
    if len(tasks) != expected_count:
        raise ValueError(f"training split must contain exactly {expected_count} tasks; found {len(tasks)}")
    if any(not re.fullmatch(r"[0-9a-f]{8}", tid) for tid in tasks):
        raise ValueError("training task filenames must be eight lowercase hexadecimal characters")
    manifest = (root / "data/training.txt").read_text(encoding="utf-8").splitlines()
    if len(manifest) != expected_count or len(set(manifest)) != expected_count or set(manifest) != set(tasks):
        raise ValueError("training.txt does not match the complete training JSON ID set")
    if expected_revision is not None:
        commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        if commit != expected_revision:
            raise ValueError(f"dataset revision mismatch: expected {expected_revision}, got {commit}")
        diff = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all",
                               "--", "data/training", "data/training.txt"],
                              capture_output=True, text=True, check=True)
        if diff.stdout:
            raise ValueError("official training data differs from the pinned clean checkout")
    return tasks


def report_status(report, task_id, source_hashes, task_hash, pair_count, checksums, finite_cardinality=None,
                  finite_identity_members=0):
    """Old results, reduced checks, and PASSes from other splits all count only as unverified."""
    if not isinstance(report, dict):
        return "unverified", ["audit report must be an object"]
    matches = (report.get("task_id") == task_id and report.get("kind") == "training"
               and report.get("source_sha256") == source_hashes
               and report.get("official_task_sha256") == task_hash
               and report.get("validator_sha256") == checksums)
    if not matches:
        return "unverified", ["audit identity, source, task or validator hashes are stale/mismatched"]
    if report.get("status") in ("FAIL", "ERROR", "TIMEOUT"):
        return "fail", report.get("errors") or [f"audit status: {report['status']}"]
    samples = report.get("samples_per_rng")
    rng = report.get("rng_seeds")
    finite = report.get("finite_domain")
    finite_ok = (finite_cardinality is not None and isinstance(finite, dict)
                 and finite.get("cardinality") == finite_cardinality
                 and finite.get("exhaustive_checked") == finite_cardinality
                 and finite.get("enumeration_replays") == finite_cardinality
                 and finite.get("official_inputs_in_domain") is True
                 and finite.get("generated_inputs_in_domain") is True
                 and finite.get("identity_members") == finite_identity_members
                 and finite.get("identity_members_supported_by_official") is True)
    if finite_cardinality is not None and not finite_ok:
        return "unverified", ["finite-domain seed lacks complete exhaustive validation"]
    diversity_ok = (isinstance(report.get("unique_inputs"), int) and report["unique_inputs"] >= 28
                    and isinstance(report.get("unique_first_30_by_rng"), dict)
                    and all(isinstance(report["unique_first_30_by_rng"].get(str(seed)), int)
                            and report["unique_first_30_by_rng"][str(seed)] >= 28
                            for seed in audit.DEFAULT_RNG_SEEDS))
    if (report.get("status") != "PASS" or report.get("errors") != []
            or report.get("official_pairs_total") != pair_count
            or report.get("official_pairs_passed") != pair_count
            or not isinstance(samples, int) or isinstance(samples, bool) or samples < 50
            or rng != audit.DEFAULT_RNG_SEEDS
            or report.get("generated_checked") != samples * 4
            or report.get("generator_rng_replays") != samples * 4
            or not (finite_ok or diversity_ok)):
        return "unverified", ["no complete passing audit of all pairs, four RNGs and at least 200 inputs"]
    return "pass", []


def semantic_report_status(report, task_id, source_hashes, task_hash, *, root=ROOT):
    if not isinstance(report, dict):
        return "unverified", ["semantic report must be an object"]
    replay = Path(__file__).with_name("replay_training_semantics.py")
    if (report.get("task_id") != task_id or report.get("source_sha256") != source_hashes
            or report.get("official_task_sha256") != task_hash
            or report.get("replay_sha256") != audit.sha256(replay)):
        return "unverified", ["semantic report identity, source, task or replay hashes are stale/mismatched"]
    external = report.get("external_check_sha256")
    if not isinstance(external, dict) or not external:
        return "unverified", ["semantic report has no retained check source hashes"]
    root = Path(root).resolve()
    for relative, checksum in external.items():
        if not isinstance(relative, str) or Path(relative).is_absolute():
            return "unverified", ["semantic check path is not repository-relative"]
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file() or audit.sha256(path) != checksum:
            return "unverified", ["retained semantic check is missing or changed"]
    checks = report.get("checks")
    if report.get("status") != "PASS":
        return "fail", [report.get("error", "independent semantic replay failed")]
    if (not isinstance(checks, list) or not checks or
            any(not isinstance(check, dict) or check.get("returncode") != 0 for check in checks)):
        return "unverified", ["semantic PASS lacks successful retained checks"]
    return "pass", []


def build_inventory(dataset_root, bank=BANK, reports=None, *, semantic_reports=None, expected_count=EXPECTED_COUNT,
                    expected_revision=DATASET_REVISION):
    tasks = official_tasks(dataset_root, expected_count, expected_revision)
    directory = Path(bank) / "seeds/training"
    present = {p.name for p in directory.iterdir() if p.is_dir()} if directory.exists() else set()
    unexpected = sorted(present - set(tasks))
    if unexpected:
        raise ValueError(f"unknown IDs in training seed directory: {unexpected}")
    checksums = audit.validator_hashes()
    rows, task_hashes = [], {}
    for task_id, task_path in tasks.items():
        task_bytes = task_path.read_bytes()
        task_hash = hashlib.sha256(task_bytes).hexdigest()
        task_hashes[task_id] = task_hash
        task = json.loads(task_bytes)
        errors = audit.validator.task_errors(task)
        if errors:
            raise ValueError(f"invalid official task {task_id}: {errors}")
        pair_count = sum(len(task[split]) for split in ("train", "test"))
        seed_dir = directory / task_id
        source_hashes = audit.hashes(seed_dir)
        row = {"task_id": task_id, "status": "missing", "official_task_sha256": task_hash,
               "official_pairs_total": pair_count, "source_sha256": source_hashes}
        if not seed_dir.is_dir():
            row["reasons"] = ["seed directory absent"]
        else:
            missing = [name for name in audit.REQUIRED_FILES if name not in source_hashes]
            if missing:
                row.update(status="fail", reasons=[f"missing required files: {', '.join(missing)}"])
            else:
                try:
                    seed = json.loads((seed_dir / "seed.json").read_text(encoding="utf-8"))
                    errors = audit.validator.validate_schema(seed, task, "training", task_id)
                except (OSError, ValueError, TypeError) as exc:
                    errors = [f"invalid seed.json: {type(exc).__name__}: {exc}"]
                if errors:
                    row.update(status="fail", reasons=errors)
                else:
                    report_path = None if reports is None else Path(reports) / f"{task_id}.json"
                    if report_path is None or not report_path.is_file():
                        row.update(status="unverified", reasons=["no external audit report; stored verification is not evidence"])
                    else:
                        row["audit_report_sha256"] = audit.sha256(report_path)
                        try:
                            report = json.loads(report_path.read_text(encoding="utf-8"))
                            cardinality = seed.get("finite_domain", {}).get("cardinality")
                            identities = seed.get("finite_domain", {}).get("identity_members", 0)
                            status, reasons = report_status(report, task_id, source_hashes, task_hash,
                                                            pair_count, checksums, cardinality, identities)
                        except (OSError, ValueError, TypeError) as exc:
                            status, reasons = "unverified", [f"invalid audit report: {type(exc).__name__}: {exc}"]
                        row.update(status=status, reasons=reasons)
                    if row["status"] == "pass" and semantic_reports is not None:
                        semantic_path = Path(semantic_reports) / f"{task_id}.json"
                        if not semantic_path.is_file():
                            row.update(status="unverified", reasons=["no external semantic replay report"])
                        else:
                            row["semantic_report_sha256"] = audit.sha256(semantic_path)
                            try:
                                semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
                                status, reasons = semantic_report_status(semantic, task_id, source_hashes, task_hash)
                            except (OSError, ValueError, TypeError) as exc:
                                status, reasons = "unverified", [f"invalid semantic report: {type(exc).__name__}: {exc}"]
                            row.update(status=status, reasons=reasons)
                    rule_status = seed.get("rule_status")
                    if isinstance(rule_status, str) and rule_status.startswith("unresolved"):
                        row["semantic_rule_status"] = rule_status
                        row.update(status="fail", reasons=["rule is explicitly unresolved; executable agreement cannot accept it"])
        rows.append(row)
    counts = Counter(row["status"] for row in rows)
    counts = {status: counts[status] for status in ("missing", "pass", "fail", "unverified")}
    return {
        "schema_version": 1, "split": "training", "expected_count": expected_count,
        "task_count": len(rows), "dataset_revision": expected_revision,
        "official_ids_sha256": digest(sorted(tasks)), "official_snapshot_sha256": digest(task_hashes),
        "seed_snapshot_sha256": digest({row["task_id"]: row["source_sha256"] for row in rows}),
        "validator_sha256": checksums, "inventory_sha256": audit.sha256(Path(__file__)),
        "counts": counts, "all_executable_checks_passed": counts["pass"] == expected_count,
        "semantic_replay_required": semantic_reports is not None,
        "semantic_completeness": "retained independent construction/counterexample checks required; text-only reconstruction remains unmeasured"
            if semantic_reports is not None else "not established by executable checks", "tasks": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="ARC-AGI-2 checkout, defaults to $ARC_DATA/ARC-AGI-2")
    parser.add_argument("--bank", type=Path, default=BANK)
    parser.add_argument("--reports", type=Path, help="per-task JSON files from audit.py --kind training")
    parser.add_argument("--semantic-reports", type=Path, help="release gate: per-task replay_training_semantics.py reports")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        root = audit.paths.data() / "ARC-AGI-2" if args.dataset_root is None else args.dataset_root
        result = build_inventory(root, args.bank, args.reports, semantic_reports=args.semantic_reports)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"split": "training", "task_count": result["task_count"], "counts": result["counts"],
                      "official_snapshot_sha256": result["official_snapshot_sha256"]}, sort_keys=True))
    return 0 if result["all_executable_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
