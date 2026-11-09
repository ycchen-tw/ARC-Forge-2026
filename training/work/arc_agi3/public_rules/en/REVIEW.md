# Review of the 25 Game Rule Documents (2026-10-20)

Chinese source

**Full-text review of all 25/25 pages is complete.** Five groups compared each game against its fixed-version control flow and all 183 levels of data;
an additional reviewer read every page for contradictions and reference usability.
This round checked the accuracy of rule statements, using source code, normal actions, and explicitly labeled state-injection probes as evidence.

For every game, the review covered controls and invalid inputs, costs/animation, win/loss conditions and ordering, visible/hidden state,
per-level data, Undo/resets/persistence across levels, and whether older versions, the reconstructed engine, or solver inference had been mixed into current rules.
Each page ends with “Sources and checks,” listing functions and line numbers; shared definitions are collected in [COMMON](COMMON.md).

## Per-game corrections and additions

| Game | Focus of this round |
|---|---|
| [AR25](games/ar25.md) | Undo neither refunds costs nor restores selection; hitting a boundary and failing to move on the wrong axis follow different win-check paths |
| [BP35](games/bp35.md) | Water-level parity also depends on clicks/falling; the antigravity click window spans 12 rows; gems take priority on the final action |
| [CD82](games/cd82.md) | An ordinary large-brush action takes 15 frames; action 100 loses first; removed the unproved global-optimality claim |
| [CN04](games/cn04.md) | Clicking to change variants does not clamp back inside the boundary; ACTION5 does; connector counts are separate for each true type |
| [DC22](games/dc22.md) | Directional actions check for victory before charging; clicks also recheck the tile underfoot; corrected RESET and bridge-count descriptions |
| [FT09](games/ft09.md) | L6 affects only the clicked tile and the tile immediately above, not recursively the whole chain; the final allowed action may still complete the level |
| [G50T](games/g50t.md) | Switching bodies clears the recording but does not reset everything: toggle doors and spent budget may persist; skipping a command does not imply that resynchronization is forever impossible |
| [KA59](games/ka59.md) | Victory is checked for each target frame; collision rollback is local, so downstream blocks already moved may stay moved; added explosion/ice-slide ordering |
| [LF52](games/lf52.md) | Updating the front cart first lets the whole line advance; added Undo/HUD, budget, and repeated scrolling details; offscreen vehicles in L10 can move into the viewport |
| [LP85](games/lp85.md) | Ring lengths/target counts for every level; L8's short rings are exact prefixes of 2/6/7 cells, not uniformly “half a ring” |
| [LS20](games/ls20.md) | Internal resource 0 remains alive; only negative values lose a life; completing the target on the same action takes priority; distinguishes charged steps from API actions |
| [M0R0](games/m0r0.md) | Action 150 executes; action 151 loses first; door state may temporarily persist after a trap returns players to the start |
| [R11L](games/r11l.md) | Only the first 59 clicks are processed; all true targets must be satisfied simultaneously; the solver's true/false-target inference is separately labeled as a non-rule |
| [RE86](games/re86.md) | Coverage uses the topmost pixel after compositing; a wrong color can cover a correct one; transformations do not always preserve the center position |
| [S5I5](games/s5i5.md) | A hand that reaches its point may temporarily leave and return; collision rollback still costs resources; the historical route is no longer claimed to be unique |
| [SB26](games/sb26.md) | The full 6×6 hit region includes an invisible `-2` outer border; added the different outcomes of placement versus execution when energy reaches zero |
| [SC25](games/sc25.md) | Spells are reassigned each level; hints and orientation have separate persistence rules; spending exactly the limit remains valid, and only exceeding it loses |
| [SK48](games/sk48.md) | Undo does not restore arm selection or remaining action budget; added final-action success priority and fields that persist after a failed shift |
| [SP80](games/sp80.md) | Deducts an action first and loses immediately at zero, even if the display changes afterward; added the hidden retry count and HUD position after rotation |
| [SU15](games/su15.md) | The target check precedes the normal charge; a fatal penalty does not automatically Undo; one frame cannot reconstruct penalty, Undo, or animation history |
| [TN36](games/tn36.md) | A socket requires bbox overlap and matching dimensions; some instructions still execute after death; added checkpoint/budget ordering |
| [TR87](games/tr87.md) | Corrected screen divisions for each level; victory compares only the target prefix, and length alone cannot uniquely identify rewrite flags |
| [TU93](games/tu93.md) | The mimic formula applies only to the sequence of successful moves after waking; occlusion still requires history; hitting a wall only spends a step and does not update enemies, queues, or waking |
| [VC33](games/vc33.md) | A zero-length piston can extend again; it is not permanently dead; gap units are native pixels |
| [WA30](games/wa30.md) | Invalid movement/empty interaction still advances robots; held-object state cannot always be determined from one frame; removed the claim that player delivery of the final crate is always faster |

## Usage limits and provenance

- The corrections above are incorporated into the individual game pages; reading the review report first is unnecessary to obtain the correct rules.
- Conditions established directly from source and branches tested by probes are recorded separately. Tests with injected boundary states do not establish normal reachability.
- Historical shortest paths, uniqueness, and all offscreen/multiple-object-overlap geometry have not been comprehensively re-proved; these are now described as local routes or carry explicit limits.
- Full-game completion also does not establish whether one frame uniquely reconstructs state; hidden fields and solver heuristics are labeled separately.

Per-game receipts include document/game SHA256 hashes, review scope, before/after corrections, source line numbers, methods, and unresolved items.
The compact index is review.json; full receipts, per-level inventories, and probe outputs are in
`$ARC_RUNS/arc_agi3/20261020-rules-reference-review/group-{a,b,c,d,e}/`.
Probe source is in probes; shared RESET tests are described in [COMMON](COMMON.md).
