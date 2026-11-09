#!/usr/bin/env python3
"""Verify a seed's full schema, the pair references it actually makes, and the executable rule.

By default the result is written back to seed.json; --no-write only checks. training/evaluation/synth are located separately.
Full stress checks use tools/audit.py; PASS does not mean semantic completeness has been proven.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
TAXONOMY = ROOT / "work/concept_extract/taxonomy.yaml"
REQ_TOP = ("task_id", "split", "origin", "rule", "concepts", "concepts_free",
           "difficulty", "provenance", "verification")
REQ_RULE = ("intent", "input", "entities", "procedure", "output_size", "decided_by")
PAIR_REF = re.compile(r"(?<![A-Za-z_])(train|test)\s*(-?\d+)(?:\s*[-–—~～至到]\s*(\d+))?(?!\d|\.\d)")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def valid_grid(grid):
    try:
        grid = np.asarray(grid)
        return bool(grid.ndim == 2 and all(1 <= n <= 30 for n in grid.shape)
                    and np.issubdtype(grid.dtype, np.integer)
                    and grid.min() >= 0 and grid.max() <= 9)
    except (TypeError, ValueError):
        return False


def taxonomy_ids():
    tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    if not isinstance(tax, dict) or not isinstance(tax.get("concepts"), list):
        raise ValueError("taxonomy.yaml must contain a concepts list")
    ids = [entry["id"] for entry in tax["concepts"]]
    if not ids or any(not isinstance(item, str) for item in ids) or len(set(ids)) != len(ids):
        raise ValueError("taxonomy ids must be nonempty unique strings")
    return set(ids)


def task_errors(task):
    if not isinstance(task, dict):
        return ["task must be an object"]
    errors = []
    for split in ("train", "test"):
        pairs = task.get(split)
        if not isinstance(pairs, list) or not pairs:
            errors.append(f"task.{split} must be a nonempty list")
            continue
        for index, pair in enumerate(pairs):
            if not isinstance(pair, dict):
                errors.append(f"task.{split}[{index}] must be an object")
                continue
            for name in ("input", "output"):
                if name not in pair or not valid_grid(pair[name]):
                    errors.append(f"task.{split}[{index}].{name} must be a legal ARC grid")
    return errors


def validate_schema(seed, task=None, expected_split=None, expected_task_id=None, tax_ids=None):
    """Return all structural errors; does not run seed code and does not patch missing fields."""
    if not isinstance(seed, dict):
        return ["seed.json must be an object"]
    errors = [f"seed.json missing field: {name}" for name in REQ_TOP if name not in seed]
    tax_ids = taxonomy_ids() if tax_ids is None else tax_ids

    def text(value, label):
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{label} must be a nonempty string")
            return False
        if value.strip().lower() in {"todo", "tbd", "placeholder", "待補", "待撰寫"}:
            errors.append(f"{label} is a placeholder")
            return False
        return True

    def strings(value, label, nonempty=False):
        if not isinstance(value, list) or (nonempty and not value):
            errors.append(f"{label} must be a {'nonempty ' if nonempty else ''}list")
            return []
        for i, item in enumerate(value):
            text(item, f"{label}[{i}]")
        return [item for item in value if isinstance(item, str)]

    for name in ("task_id", "origin"):
        text(seed.get(name), name)
    split = seed.get("split")
    if split not in ("training", "evaluation", "synth"):
        errors.append("split must be training, evaluation, or synth")
    if split in ("training", "evaluation") and seed.get("origin") not in ("v1_train", "v1_eval", "v2_new"):
        errors.append("official origin must be v1_train, v1_eval, or v2_new")
    if expected_split is not None and split != expected_split:
        errors.append(f"split mismatch: expected {expected_split!r}, got {split!r}")
    if expected_task_id is not None and seed.get("task_id") != expected_task_id:
        errors.append(f"task_id mismatch: expected {expected_task_id!r}, got {seed.get('task_id')!r}")
    if split in ("training", "evaluation") and not re.fullmatch(r"[0-9a-f]{8}", str(seed.get("task_id", ""))):
        errors.append("official task_id must be eight lowercase hexadecimal characters")
    for name in ("difficulty", "provenance", "verification"):
        if not isinstance(seed.get(name), dict):
            errors.append(f"{name} must be an object")
    difficulty = seed.get("difficulty")
    if isinstance(difficulty, dict):
        count = difficulty.get("n_pass_models")
        if "n_pass_models" not in difficulty or (count is not None and
                (not isinstance(count, int) or isinstance(count, bool) or count < 0)):
            errors.append("difficulty.n_pass_models must be a nonnegative integer or null")
    provenance = seed.get("provenance")
    if isinstance(provenance, dict):
        text(provenance.get("author"), "provenance.author")
        if split in ("training", "evaluation"):
            strings(provenance.get("nl_sources"), "provenance.nl_sources", nonempty=True)
            # Independently authored rules legitimately have no reused code.
            # Keep the explicit list required, but do not invent a code source.
            strings(provenance.get("code_sources"), "provenance.code_sources")
            text(provenance.get("notes"), "provenance.notes")
    if "context_required" in seed and not isinstance(seed["context_required"], bool):
        errors.append("context_required must be a boolean")
    if "finite_domain" in seed:
        domain = seed["finite_domain"]
        if not isinstance(domain, dict):
            errors.append("finite_domain must be an object")
        else:
            count = domain.get("cardinality")
            if not isinstance(count, int) or isinstance(count, bool) or not 1 <= count <= 4096:
                errors.append("finite_domain.cardinality must be an integer from 1 to 4096")
            identities = domain.get("identity_members", 0)
            if (not isinstance(identities, int) or isinstance(identities, bool) or identities < 0
                    or isinstance(count, int) and identities >= count):
                errors.append("finite_domain.identity_members must be a nonnegative integer smaller than cardinality")
            text(domain.get("rationale"), "finite_domain.rationale")
    concepts = strings(seed.get("concepts"), "concepts", nonempty=True)
    strings(seed.get("concepts_free"), "concepts_free")
    for concept in concepts:
        if concept not in tax_ids:
            errors.append(f"concepts contains unknown taxonomy id: {concept}")
    rule = seed.get("rule")
    if not isinstance(rule, dict):
        errors.append("rule must be an object")
        return errors
    errors.extend(f"rule missing field: {name}" for name in REQ_RULE if name not in rule)
    if text(rule.get("intent"), "rule.intent") and not re.search(r"[一-鿿]", rule["intent"]):
        errors.append("rule.intent must contain Chinese")
    text(rule.get("output_size"), "rule.output_size")
    inp = rule.get("input")
    if not isinstance(inp, dict):
        errors.append("rule.input must be an object")
    else:
        for name in ("invariants", "varies", "distractors"):
            entries = strings(inp.get(name), f"rule.input.{name}", nonempty=True)
            if name == "invariants" and any(not re.search(r"[一-鿿]", item) for item in entries):
                errors.append("rule.input.invariants must contain Chinese in every entry")
    entities = rule.get("entities")
    if not isinstance(entities, list):
        errors.append("rule.entities must be a list")
    else:
        if "object_detection" in concepts and not entities:
            errors.append("object_detection requires nonempty entities")
        for i, entity in enumerate(entities):
            if not isinstance(entity, dict):
                errors.append(f"rule.entities[{i}] must be an object")
                continue
            for name in ("name", "detect", "role"):
                text(entity.get(name), f"rule.entities[{i}].{name}")
    procedure = rule.get("procedure")
    if not isinstance(procedure, list) or not procedure:
        errors.append("rule.procedure must be a nonempty list")
    else:
        for i, step in enumerate(procedure):
            if not isinstance(step, dict):
                errors.append(f"rule.procedure[{i}] must be an object")
                continue
            op = step.get("op")
            if not isinstance(op, str) or op not in tax_ids:
                errors.append(f"rule.procedure[{i}].op is not a taxonomy id: {op!r}")
            elif op not in concepts:
                errors.append(f"rule.procedure uses {op} but concepts omits it")
            text(step.get("text"), f"rule.procedure[{i}].text")
    decided_by = rule.get("decided_by")
    if not isinstance(decided_by, dict):
        errors.append("rule.decided_by must be an object")
    else:
        strings(decided_by.get("undecided"), "rule.decided_by.undecided")
        decided = decided_by.get("decided")
        if not isinstance(decided, list) or not decided:
            errors.append("rule.decided_by.decided must be a nonempty list")
        else:
            for i, claim in enumerate(decided):
                label = f"rule.decided_by.decided[{i}]"
                if not isinstance(claim, dict):
                    errors.append(f"{label} must be an object")
                    continue
                text(claim.get("claim"), f"{label}.claim")
                evidence = claim.get("evidence")
                if not text(evidence, f"{label}.evidence"):
                    continue
                refs = PAIR_REF.findall(evidence)
                if not refs:
                    errors.append(f"{label}.evidence must cite a pair: train i or test j")
                for pair_split, first, last in refs:
                    start, stop = int(first), int(last or first)
                    if start < 0 or stop < start:
                        errors.append(f"{label}.evidence has invalid pair range: {pair_split} {first}–{last or first}")
                    elif task is not None and (not isinstance(task, dict)
                            or not isinstance(task.get(pair_split), list) or stop >= len(task[pair_split])):
                        errors.append(f"{label}.evidence cites absent pair: {pair_split} {stop}")
    if task is not None:
        errors.extend(task_errors(task))
    return errors


def main(seed_dir, *, write=True):
    directory = Path(seed_dir).resolve()
    seed_path = directory / "seed.json"
    try:
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
        kind = directory.parent.name if directory.parent.name in ("training", "evaluation", "synth") else (
            seed.get("split") if isinstance(seed, dict) else None)
        errors = validate_schema(seed, expected_split=kind, expected_task_id=directory.name)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as exc:
        errors = [f"cannot validate seed: {type(exc).__name__}: {exc}"]
    if errors:
        print(json.dumps({"status": "FAIL", "errors": errors}, ensure_ascii=False, indent=1))
        return 1
    audit = load(Path(__file__).parent / "tools/audit.py", "seed_bank_audit")
    result = audit.audit_one(directory.name, 30, [0], kind, seed_dir=directory)
    # The 30-input quick check does not pose as the full audit with four RNGs and 200 inputs.
    verification = {
        "transform_pairs": f"{result.get('official_pairs_passed', result.get('task_pairs_passed', 0))}/"
                           f"{result.get('official_pairs_total', result.get('task_pairs_total', 0))}",
        "check_input_on_originals": result.get("check_input_on_originals", False),
        "gen_valid": f"{result.get('generated_valid', 0)}/30",
        "gen_check_input": f"{result.get('generated_check_input', 0)}/30",
        "gen_transform_runs": f"{result.get('generated_transform_runs', 0)}/30",
        "gen_output_differs": f"{result.get('generated_output_differs', 0)}/30",
        "generated_checked": result.get("generated_checked", 0),
        "gen_unique": f"{result.get('unique_inputs', 0)}/30",
        "rng_seeds": [0], "samples_per_rng": 30,
        "nl_complete": None, "status": result["status"], "errors": result["errors"],
    }
    if write:
        seed["verification"] = verification
        seed_path.write_text(json.dumps(seed, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=1))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("seed_dir", type=Path)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    sys.exit(main(args.seed_dir, write=not args.no_write))
