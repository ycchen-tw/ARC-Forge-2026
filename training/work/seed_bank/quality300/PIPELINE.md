# Pipeline and reproducibility

## What this tree reproduces

`python -B tools/replay.py` needs Python 3.12 and the standard library. It verifies the file manifest, all saved train/test outputs, input nonmutation, generator/transform determinism, legal ARC grids, local Python import closure, and latent-oracle equality where generate_case is supplied. Twenty fixed mechanical seeds per task are reused from the original validator's deterministic sequence. This is an implementation replay, not a new model evaluation or a new random blind pool. Use `--samples 1` for a quicker check or `--output result.json` to save a report outside the manifest-bound files.

The task runtime and task.json are copied byte-for-byte. Metadata is curated to remove workflow identities, machine locations and service bindings. The original archive hashes bind the unabridged evidence; the package manifest binds the curated files. Sanitized metadata hashes are not passed off as the historical hashes used at acceptance.

The production `runtime_imports.py` binds the static transitive local imports of transform.py and generate_input.py, excluding unrelated rendering/authoring utilities. Its original eager loader uses a temporary package scope, restores same-named caller modules and the search path even after import failure, and does not support local imports deferred until a later function call. Static closure does not discover dynamic imports or external runtime data. The exact original closure module is included at tools/runtime_imports.py for reference. This lean replay isolates each task in its own Python process and keeps that task's package directory available during execution; it does not recreate the omitted original blind harness.

## Generate an input and reference answer

Each task exposes `generate_input(integer_seed)` and `transform(grid)`. For example, from the collection root:

```sh
cd packages/q001
python -B -c "import json; from generate_input import generate_input; from transform import transform; x=generate_input(12345); print(json.dumps({'input': x, 'output': transform(x)}))"
```

Copy the complete task directory when using it elsewhere. The same task and integer seed reproduce the same input. Some late generators reserve small integer seeds for exact teaching scenes; DOMAIN.md and the generator describe those cases. Different seeds can produce the same grid, so seed inequality alone is not evidence of a fresh input. `generate_case(seed)`, when present, also supplies a latent-scene answer for consistency checking. Generated reference answers are not independently inferred answers and confer no new blind-test credit.

## How acceptance was obtained

1. Inspect at least two original source mechanisms, including one eligible evaluation seed from the pinned snapshot. Record exact source hashes, available code, missing-code limitations and any deliberate domain adaptations. Three or more parents are allowed when they contribute naturally.
2. Write a coherent visual rule, normally at most two meaningful operations. Provide three to five teaching pairs and two heldouts, varying the rule's meaningful regimes. Implement plausible rivals and require an actual teaching witness against each registered rival. A finite rival list is not an exhaustive uniqueness proof.
3. Run reference replay, generator/latent checks, nonmutation, source identity checks, concept comparison and teaching review. Inspect relevant boundary cases and source-causal contrasts. Author/reviewer sample counts vary by task and are not represented as one universal 600-case study.
4. Freeze the current artifacts. A fresh context receives only anonymous training grids and held-out inputs, then returns exact answers, a short rule summary and an independently written program. Small batches may share a context; there is not a distinct context for every task.
5. Freeze that program before drawing and saving twenty new generator inputs. Exclude shown and recorded prior input hashes; require twenty distinct grids and at least ten classes after zero-padding, global nonzero color renaming and eight rigid symmetries are removed. Finish selecting the pool before scoring. Both heldouts and all twenty program outputs must match exactly.
6. Compare the inferred rule with the intended domain. The summaries distinguish complete source-code review from earlier summary-and-mechanism comparison. An equivalent simpler algorithm is allowed. A material counterexample can cause a revision or rejection even after 2/2 and 20/20 sampled passes. Preserve prior scores and versions; changed examples or runtime code require fresh bound evidence.

The implemented minimum is one fresh grid-only context plus its frozen program on twenty cases. The early two-context policy was a proposal, not a completed requirement for all tasks. IMPLEMENTED_POLICY.json records the implemented minimum. Failed current-version attempts cannot be ignored by repeatedly sampling solvers until a pass appears.

## What has deliberately been omitted

The lean package excludes raw blind payloads/responses, frozen inferred programs and transfer pools, retired candidates, full original source snapshots, source-inspection dumps, review images, large author/semantic reports and the original acceptance harness. Consequently, `pipeline/validate.py`, `accept.py` and strict frozen-program evidence replay are not commands provided by this tree. See AUDIT_ARCHIVES.md for exact full-archive filenames and hashes. Their omission is a packaging choice, not deletion of unfavorable evidence.

The gallery keeps the existing offline interface and exact grids, with only its embedded metadata sanitized. The included Node smoke test verifies navigation, filters, hide/reveal toggles and package links. It is a DOM-contract test, not a full browser layout or human usability study.

## Limitations that remain

- No claim of human difficulty calibration, universal rule identifiability, global novelty, out-of-distribution generalization or full-domain mathematical correctness.
- Source-byte verification is not a fresh official-original replay. Missing source code and genuine fresh replays remain distinguished per parent.
- new_design and quality_rework are separate categories; parameter-only variations are not additional concepts. Reworks identify the earlier concept and the concrete improvement.
- q091 had a tool-aborted twenty-case pool whose inputs/results were lost before a score was observed or counted. It was subsequently retested using a fresh context and a pre-saved pool. Disjointness from the lost inputs cannot be proved.
- q031's historical pool included a shown input; that evidence was superseded by disjoint sweeps. Replaying those histories never creates new independent evidence.
- Static local import inspection does not discover disguised dynamic imports or arbitrary external runtime data. This package rejects declared unsupported imports and direct runtime file I/O; that is a portability check, not a security sandbox.

## Rebuilding this source package

Run `python -B tools/build_source_package.py --help`. Supply the validated accepted-only release directory, its receipt and archive manifest, and a new destination/name. The builder checks the complete release manifest, exact300 acceptance, passing gates, original archive bytes, runtime hashes, portable replay and the gallery contract before exporting deterministic ordinary ZIP files smaller than 14,000,000 bytes each. All parts merge into one common directory. It performs no external writes.

Preview mode is only for local construction tests. It requires NOT_FOR_PUBLICATION in the output name, creates a warning file and gallery banner, and reports its actual task count. It must never be substituted for the final 300-task export.
