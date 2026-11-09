# Version Validation for the 25 Public Games (2026-10-20)

Chinese source

This correction covers the 25 public games and 183 levels after the 2026-04-14 update.
Final per-game action counts, handoff tests, and the evidence index are in STATUS.
Reading rules, targeted runtime tests, and full-game completion are three different forms of evidence; older experiments that were not rerun are labeled historical records.
The subsequent per-game rules review is in [REVIEW](REVIEW.md); shared definitions of controls, RESET, and experimental starting points are in [COMMON](COMMON.md).

## Sources and versions

| Data | Purpose and checks |
|---|---|
| `$ARC_DATA/arc3_games/official/versions.json` | Selects the 25 `current` IDs and supplies source hashes and baseline provenance |
| `official/api_source/<id>-<version>/<id>.py` | Official public source actually executed; each file's SHA256 is checked against the list above |
| `official/kaggle_competition/environment_files/` | Competition source and metadata; the version is the first eight characters of the source SHA256 |
| Comparison of the two source forms | After removing only the two internal hooks `_get_hidden_state` and `_get_valid_actions` from the competition version, ASTs match for all 25 games; file-header comments do not participate in the AST |
| Official `arcengine 0.9.3` | Pinned using distribution metadata, with SHA256 hashes saved for seven Python files; the module's own `__version__` is not the package distribution version |
| `reference/arc-agi-games/`, commit `41b87fe1ea8d9819a44eea35172ffe28d6c5ffe6` | March historical snapshot, used only to explain differences in older documents and versions |

The API and competition versions are not byte-identical; the first eight characters of an API file's hash cannot be treated as its version ID.
This comparison froze both source forms, metadata, engine, solver, and harness; official game source was not modified.
The baseline is the per-level human action count in current metadata, not the solver's action budget.

The 15 version changes in the official [changelog](https://docs.arcprize.org/changelog) and the
[scoring methodology](https://docs.arcprize.org/methodology) were checked again on 2026-10-20;
the per-level human upper median has been used since 4/14, and baselines also changed for the ten games whose source versions did not change.

## Scope of correction

| Category | Games | Conclusion |
|---|---|---|
| Source version unchanged | BP35, CD82, FT09, G50T, LF52, LP85, LS20, SB26, TR87, WA30 | Baselines updated; official engine semantics still require rechecking |
| Main mechanics retained, with reobfuscation and some data/compositing differences | AR25, KA59, M0R0, RE86, SC25, SP80, SU15, TN36, VC33 | Source names, level tables, and invisible tiles were checked game by game; these cannot broadly be called “identical word for word” |
| Clickability changed | SK48 | Some bases in L3/L4 lost `sys_click`; current L4 is completed using only the initial horizontal arm and directional keys |
| Display/target data changed | DC22, R11L, S5I5, TU93 | Occlusion, same-color decoys, and camera compositing order can change the solver's observations |
| Major mechanics changes | CN04 | Five levels became six; two connector types, per-level budgets, and stacked variants; the full text was rewritten for the new version |

The following are content corrections, not merely version-label changes:

- **LS20**: official L5/L6/L7 do contain 1/3/1 movement stations; the old document incorrectly classified them as dead code using the reconstructed engine.
- **SC25**: official hit testing accepts invisible sensing regions; the rune board's actual click intervals were added.
- **R11L**: true-target collision regions include invisible `-2` interiors, not just the drawn cross; the names used to exclude false targets are not on screen.
- **DC22**: the game's custom collisions/underfoot checks exclude `-2`, but its key probe accepts it; the official generic Sprite collision mask cannot be applied directly.
- **SK48**: L4 no longer allows selecting an arm in another direction; the old switching solution does not apply.
- **S5I5/TU93**: custom Cameras draw in sprite-list order; the old reconstructed engine sorted first, potentially changing occlusion. TU93's exit can be recovered from an intermediate frame of the same action.
- **SB26/TN36/SP80**: respectively corrected outdated hit-testing concerns, the exception in the first level's default instruction tape, and the retry timing where loss occurs only on the fifth water release.

## Reproducible validation

`verify_current.py` creates a new output directory on every run:

```bash
source env/env.sh
: "${PY:?Specify a Python with the required dependencies}"
: "${ARCENGINE_SITE:?Specify the official arcengine 0.9.3 site-packages}"
PYTHONPATH="$ARC_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" \
  work/arc_agi3/runtime/pixel/probes/verify_current.py --workers 4
```

Output root: `$ARC_RUNS/arc_agi3/20261020-public25-current/`.
Each `frozen/manifest.json` records game/metadata/engine/solver hashes;
`results/<id>.json` preserves per-level action counts and complete action sequences for the main game and three handoffs, with stdout alongside.
`--frozen <path>` verifies hashes and uses that snapshot's solver, without silently substituting the latest working directory.
After updating a solver, create a separate snapshot; a PASS from an old snapshot is not a result for the new code.

## Validation limits

- Solvers forbid RESET, are checked by import/AST contracts, and receive only frames, public state, and available actions; development probes may read source and internal objects, and are stored separately.
- Standard handoffs test only 25%/50%/75% of the completion route. Dense cut points, stacked states, and alternative routes are recorded separately; this still does not exhaust all reachable states.
- R11L false targets can have the same color and shape as true targets; the current solver uses pairing inference from the set of visible colors, so it cannot claim that one frame always distinguishes them.
- The SK48 planner still overapproximates the clickability of paired bases; neither this round's L4 completion nor the additional tests sends those clicks.
- S5I5's representative directions for hidden pivots are used only under the independent conditions checked; cases requiring the true direction raise an error.
- The original “24/25” was obtained on March source plus the reconstructed engine; the old S5I5 solver fails even on the first level of the official engine. The instructional `replica/` was not comprehensively rewritten in this round either.
- Dead-code notes, old local shortest paths, and random cross-checks retained in the rule pages do not automatically become current official runtime evidence because this round achieved full-game PASS.
