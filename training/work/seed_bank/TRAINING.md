# training seeds: sources and rerun

The deliverable for each official ID is in `seeds/training/<id>/`: Traditional Chinese rules, an executable transform,
a procedural generator / predicate, and the retained independent checks. The assignment lists are in `batches/20261003-training-astra*.json`;
scheduling, source coverage, or an existing directory do not equal acceptance.

## Pinned sources

| Source | commit | Use |
|---|---|---|
| [ARC-AGI-2](https://github.com/arcprize/ARC-AGI-2) | `f3283f727488ad98fe575ea6a5ac981e4a188e49` | the 1,000 training tasks, untouched |
| [ARC-AGI-1](https://github.com/fchollet/ARC-AGI) | `399030444e0ab0cc8b4e199870fb20b863846f34` | cross-checks 391 v1_train and 376 v1_eval; the other 233 are v2_new |
| [ARC-GEN](https://github.com/google/ARC-GEN) | `a15cbdb44c776610aeeb9f487a06af875d3d0878` | 891 overlapping generators, Apache-2.0 |
| [solution atlas](https://github.com/yhay81/arc-agi-2-solution-atlas) | `7f5940a7ffc1c491b9f92a2bf5a78052b7510bc8` | candidate rules, Apache-2.0 |
| [re-arc](https://github.com/michaelhodel/re-arc) | `e5b7f1d06362a76f9d3b8c25154ff1fafca897ce` | cross-source for 391 overlapping tasks, MIT |

Full license texts in licenses/. Per-task provenance states what was reused / rewritten and the ambiguities.
The unmodified Atlas rules pass all official pairs on only 998/1,000 tasks; the sources' own parameter special cases,
corrected answers, and fixed coordinate tables are not treated as rule evidence. Original training snapshot SHA256:
`70a3710675c700717cb5b2d76908c3a068d379e21a74f11b81ca4c386d665201`.

## Rerunning acceptance

Put the official checkout at `$ARC_DATA/ARC-AGI-2` and check out the commit in the table above;
this repo's environment needs NumPy, PyYAML, pytest. Third-party clones are only for re-extraction; running the seeds does not depend on them.

```bash
source env/env.sh
RUN="$ARC_RUNS/seed_bank/20261003-training"
.venv/bin/python work/seed_bank/tools/audit.py --kind training --workers 5 --out "$RUN/audit"
.venv/bin/python work/seed_bank/tools/replay_training_semantics.py --workers 5 --out "$RUN/semantics"
.venv/bin/python work/seed_bank/tools/training_inventory.py \
  --reports "$RUN/audit" --semantic-reports "$RUN/semantics" --out "$RUN/inventory.json"
.venv/bin/python work/seed_bank/build_viewer.py --split training --samples 3
.venv/bin/python -m pytest tests/test_training_seed_*.py tests/test_training_source_staging.py
```

A non-zero exit from audit keeps the per-task FAILs; the next two verification commands should still be run to get a complete inventory.
Semantic replay forces Python assertions on, so an external `PYTHONOPTIMIZE` cannot switch the tests off.
inventory checks per task the official pin / ID, source and verifier hashes, all original pairs, 200 inputs / four RNGs,
generator replay, and independent-check hashes; an old PASS after a code change only counts as unverified.
Small finite families are verified over their whole parameter domain instead, must still cover all original tasks and run 200 inputs; diversity must not be padded with unrelated noise.

## Limits of interpretation

- Exact match on the official pairs is a hard gate; dossiers of unexplained differences are kept, official cells are never altered to score.
- `context_required` may only learn templates from the train pairs first; fitting on test is forbidden.
- The generation range deliberately excludes inputs without evidence yet; `decided_by.undecided` states the boundary.
- Comparison against constructed answers and counterexample exclusion do not mean the Chinese text alone suffices for a blind rebuild; `nl_complete` stays null.
- The lineage of this training batch must not pass as new evaluation tasks; the existing evaluation120 and its verification results are kept separate.

## Parent quality grading (2026-10-04)

```bash
OUT="$ARC_RUNS/seed_bank/20261004-training-arcgen-crosscheck"
ls -d work/seed_bank/seeds/training/*/ | xargs -P 20 -I{} sh -c \
  't=$(basename {}); .venv/bin/python work/seed_bank/tools/crosscheck_arcgen.py {} -n 100 --rng-seed 7 > '"$OUT"'/$t.json'
.venv/bin/python work/seed_bank/tools/seed_quality.py "$OUT" --out work/seed_bank/results/training_seed_quality.json
```

ARC-GEN's `generate()` draws input and output together, so it can serve as an independent reference answer; some generators' rejection loops never end,
so the tool adds a 3-second timeout per sample and records `gen_timeout`. Results are leads, not verdicts: a mismatch needs a human to confirm whether the seed or ARC-GEN is at fault.
