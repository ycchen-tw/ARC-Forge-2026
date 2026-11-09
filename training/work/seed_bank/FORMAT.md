# Seed format specification (v0, 2026-09-05)

One directory per task, `seeds/<split>/<task_id>/`, with three files: `seed.json`, `transform.py`, `generate_input.py`.
All text is written in **Traditional Chinese** (the example values below are translated for readability). Code comments in Chinese, identifiers in English.

## seed.json

```json
{
  "task_id": "bf45cf4b",
  "split": "evaluation",
  "origin": "v2_new",
  "rule": {
    "intent": "One plain-language sentence: use the single-colour mask as a blueprint and place one copy of the multicolour tile on each mask cell.",
    "input": {
      "invariants": ["a property that holds for every input, one per entry, written as a predicate", "..."],
      "varies":     ["quantities that change across pairs and their ranges; this is the parameter table of generate_input", "..."],
      "distractors":["things that may appear but are unrelated to the rule; write \"none\" if there are none"]
    },
    "entities": [
      {"name": "tile", "detect": "how to find it in the grid", "role": "template"}
    ],
    "procedure": [
      {"op": "background_identification", "text": "Take the most frequent colour as the background."},
      {"op": "object_detection",          "text": "Take 8-connected components of non-background cells; there should be two."}
    ],
    "output_size": "same size / fixed HxW / formula",
    "decided_by": {
      "decided":   [{"claim": "background cells inside the mask frame map to an all-background block", "evidence": "train 2 and test 0"}],
      "undecided": ["how to handle a tile that itself contains background-coloured holes; all four tiles in the original task are solid"]
    }
  },
  "concepts": ["background_identification", "object_detection", "tiling"],
  "concepts_free": ["concepts missing from the taxonomy but necessary; empty array if none"],
  "difficulty": {"n_pass_models": 19},
  "provenance": {
    "nl_sources": ["concept_extract", "rulecode"],
    "code_sources": ["rulecode"],
    "author": "agent-<name>",
    "notes": "important judgements made while writing, e.g. which reference code was adopted and what was changed"
  },
  "verification": {}
}
```

Field rules:

- `origin`: `v1_train`, `v1_eval`, or `v2_new`, checked against the source lists; `split` is the current official training/evaluation split.
- `intent` is one sentence written for humans; no technical details.
- `invariants` are properties that **hold for every input**, written as checkable sentences; `check_input` must implement them.
  Conditions that "the original task does not actually require, tightened only so the generator avoids undecided cases" get the suffix "(generator constraint)",
  so a reader of the rule can tell invariants of the rule itself from safety margins of the sampling range.
- Each `varies` entry says "what changes, and the range seen in the original task"; generate_input's sampling ranges follow it.
- `entities` may be an empty array for purely geometric tasks; **required** when `concepts` contains `object_detection`.
- `procedure` has one `op` per step, which **must be** one of the 73 ids in `work/concept_extract/taxonomy.yaml`;
  tie-breaks and edge cases go in that step's `text`, not in separate fields. This is the only authoritative statement of the transform; do not write a separate summary.
- Every `decided_by.decided` entry must cite a pair (`train i` / `test j`); entries without a citation are not allowed.
  `undecided` lists situations that do not occur in the original task but the rule might encounter; generate_input must **avoid generating** them.
  The evidence file's `public_eval.wrong_clusters` are wrong answers other models actually submitted; if you **can explain** which rule produced one, write it as a decided entry (claim "not X"); if not, skip it -- do not force it.
- `concepts` may only contain taxonomy ids; every op used in `procedure` must appear in `concepts`.
- `verification` is left as an empty object; `verify.py` fills it.
- `provenance.code_sources` may be an empty array, meaning an independent implementation; do not fabricate reused sources.

## transform.py

```python
import numpy as np
def transform(grid: np.ndarray) -> np.ndarray: ...
```

- Deterministic; only `numpy` and the standard library; no file reads, no print.
- The input may be a list of lists; call `np.array(grid)` first.
- Must pass **all** train and test pairs of the original task.

If the task requires learning an arbitrary template from the task's own examples (e.g. the shared base image of `269e22fb`),
add `context_required: true` and implement `fit(train_pairs)`; the verifier first builds in-task state from train only,
then verifies all pairs. Storing the original task's answer matrices in the program or building templates from test is forbidden.
`generate_input` may sample new crops of the learned template and must state the scope of such an in-task conditional family explicitly.

## generate_input.py

```python
import numpy as np
def generate_input(rng: np.random.Generator, size_hint: int | None = None) -> np.ndarray: ...
def check_input(grid: np.ndarray) -> bool: ...
```

- Use only the `rng` passed in; the global `np.random.*` and the `random` module are not allowed. Helpers in the same directory may be reused with `from transform import ...`.
- `check_input` must return True for every input of the original task and for every input it generates itself; it is the program version of `invariants`.
- Generated grids must be 1..30 per side with colours 0..9, at least 28 of 30 mutually distinct; `transform` must run on them and the output must differ from the input.
- `size_hint` may be ignored, but keep the parameter.
- Do not generate situations listed in `undecided`. On generation failure, raise; do not return a degenerate grid.

### Small finite families

If the rule's faithful parameter domain is too small, do not add unfounded colours or shapes just to reach 28/30 unique.
Instead declare `finite_domain: {"cardinality": N, "rationale": "exact parameter domain and derivation of the count"}`
(N ≤ 4096) and implement a parameterless `enumerate_inputs()` that constructs every input from the parameter combinations one by one;
enumerating the original task's grids is not allowed. The verifier then replaces the sampled-uniqueness threshold with checks of validity, uniqueness, the predicate, reproducibility, and the transform over all N;
the original task and all 200 samples must belong to this domain. This does not prove semantic completeness.
If the original task contains valid inputs that are not transformed, the exact count can be declared as `finite_domain.identity_members`;
the full enumeration keeps them and checks the count against the original task's evidence; the 200 random generations still draw only from the non-identity subdomain.

## Verification

```bash
.venv/bin/python work/seed_bank/verify.py work/seed_bank/seeds/evaluation/<task_id>
```

This is a quick check (30 inputs, one RNG) that writes back `verification`; add `--no-write` for a read-only check.
Formal training acceptance must additionally run all original pairs, four RNGs with 200 inputs in total, and same-RNG replay,
then replay the independently constructed answers and semantic counterexamples; [how to rerun](TRAINING.md). Merely having `verification: PASS`
or passing pairs from a candidate source does not count as done. Unexplained differences from the official tasks must keep FAIL with the difference coordinates.
