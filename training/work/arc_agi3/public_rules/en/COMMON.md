# Common Semantics for Reading Rules and Designing Experiments

[Chinese source](../COMMON.md)

Fixed scope: the 25 game versions in this directory and official `arcengine 0.9.3`.
The following describes the engine interface; each game's custom counters, Undo, collisions, and reset effects are still defined by its own page.
Source hashes are in the [source receipt](../../runtime/pixel/results/20261020-current/summary.json); per-game review is in [REVIEW](REVIEW.md).

## Three meanings of “one step”

| Term | Precise meaning |
|---|---|
| API action | One call to `perform_action`. While the game is nonterminal, a non-RESET action increments the engine's `_action_count` **before** entering `step()`; entering a new level resets it to zero. A non-RESET action sent after WIN/GAME_OVER does not enter `step()` or increment the count |
| In-game resource | The game's own energy, action bar, mistakes, or lives. One action may spend several units, some operations may be free, or Undo may refund resources; this is not the API action count |
| Animation frame/internal tick | One action can call `step()` repeatedly and render multiple frames until `complete_action()`; these calls do not further increment the engine action count |

Thus, “limit 60” cannot simply be written as “60 operations are allowed”: R11L loses before entering the action logic on action 60,
whereas LS20 remains alive when its custom resource reaches 0. Check the comparison, where the deduction occurs, and the order of win and loss checks.
The individual pages now state the first rejected action or resource threshold; the baseline is a human reference, not a game budget.

The action sets in these documents come from each game's `available_actions`; RESET is explained separately. Sending unexposed actions directly to the underlying engine,
even if the code accepts them, counts only as an implementation probe, not part of the normal control interface.

## Official display color IDs

The table gives official `arc_agi.rendering.COLOR_MAP` color IDs and display names, rather than the ARC-AGI-2 0–9 palette; IDs are unchanged. Source identifiers and separately labeled historical renderers are not renamed by this table.

| Color ID | Display color | Color ID | Display color |
|---|---|---|---|
| 0 | White `#FFFFFF` | 8 | Red `#F93C31` |
| 1 | Light gray / off-white `#CCCCCC` | 9 | Blue `#1E93FF` |
| 2 | Gray `#999999` | 10 | Light blue `#88D8F1` |
| 3 | Dark gray `#666666` | 11 | Yellow `#FFDC00` |
| 4 | Very dark gray / near-black `#333333` | 12 | Orange `#FF851B` |
| 5 | Black `#000000` | 13 | Maroon `#921231` |
| 6 | Magenta `#E53AA3` | 14 | Green `#4FCC30` |
| 7 | Pink `#FF7BCC` | 15 | Purple `#A356D6` |

## Display and true state

- An official action result contains a set of frames; keeping only the final frame is a client choice. Hints or occluded objects may appear only in intermediate frames.
- Output is a 64×64 color-code image; native level/camera dimensions may be smaller, with integer scaling and centered padding.
  `ACTION6(x,y)` uses **display coordinates**; a sprite's `(x,y)` usually uses native world coordinates, while board-cell indices are another layer. Source positions in a per-level table cannot be used directly as click coordinates.
- Invisible, noncolliding, and unclickable are three different properties. The generic Camera draws only nonnegative pixels, but generic Sprite collisions and `get_sprite_at` may include `-2`.
  DC22 has custom collision logic that excludes negative values. Also inspect visibility, interaction, blocking, tags, and the game's own branches.
- Source names, undisplayed connector types, and complete enemy memory are not single-frame observations. These documents may explain those true rules;
  solver inference rules must be labeled separately, and a workable heuristic must not be presented as the game's decision rule.

## RESET, level entry, and independent experiments

The following are the official engine's default RESET branches; none of these 25 games overrides these shared methods:

| Operation/condition | Engine effect |
|---|---|
| Send RESET after a non-RESET action in the current level | Clone the level's clean layout and rerun `on_set_level`; preserve the completed-level count |
| Send RESET when the current level's `_action_count == 0`, or after WIN | `full_reset`: clone all levels, zero the completed-level count, and return to level 1 |
| `ONLY_RESET_LEVELS=true` and not yet WIN | Reset the current level even when the count is 0 |
| Call `set_level(k)` directly | Change the index, zero the engine count, adjust the camera, and call `on_set_level`; **do not clone a clean layout or set the completed-level count** |
| Create a new game instance | Run the constructor again; this differs from RESET on the same instance |

Therefore, pressing RESET immediately after entering the next level, before taking an action, returns to level 1 by default; pressing RESET again after a level reset may also return to level 1.
`full_reset` still reuses the same Python game instance and does not automatically clear every custom field;
which fields persist across levels/resets depends on that game's `on_set_level` and RESET branches.
For example, SC25 reassigns its spell set each level, but hint selection and orientation have separate persistence rules; G50T's body-switch key is not the engine's RESET.

For isolated level experiments, state whether the starting point comes from natural completion, a level jump on the same instance, a current-level RESET, or a new instance.
`Runtime.reset()` in `pixel/official_runtime.py` **creates a new game**, so it cannot substitute for testing RESET sent midway through play.

## Reading the evidence

Each page's “Sources and checks” section points to functions/line numbers in a fixed version; source-derived conclusions are separated from probes that actually send actions.
Some boundary probes inject counters, positions, or states. They establish only the behavior of that branch, **not that the state is reachable from a normal start**.
Shortest paths, random cross-checks, and “unique solution” claims from older versions/the reconstructed engine do not become current evidence through this round's completion results.

Engine references: `set_level:148`, `perform_action:189`, `_set_action:269`,
`handle_reset:305`, `full_reset:318`, and `level_reset:326` in `base_game.py`; camera scaling/coordinates in
`camera.py:190/270/318`; generic hit testing/collisions in `level.py:189` and `sprites.py:490`.
The [common probe](../probes/common_engine.py) completes CD82 normally through entry to L2, then tests boundary/mid-level/consecutive RESETs,
and separately confirms that level jumps do not update the completed-level count. Results are stored at
`$ARC_RUNS/arc_agi3/20261020-rules-reference-review/common-engine-reviewed.json`.

See the [independent document playtest](../PLAYTEST.md) for the 2026-10-22 corrections, per-game coverage, and untested claims.
