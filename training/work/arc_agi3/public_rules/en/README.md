# Rules for the 25 Public ARC-AGI-3 Games

Translated from the Chinese originals · [Translation sources and hashes](SOURCE.json)

**Second document playtest, 2026-10-22**: another 25 independent Sol players operated from the documents. They cleared 24 games and 180/183 levels; BP35 reached level 7, leaving levels 8–9 untested. The round-2 report (not in this release) covers claim coverage, amendments, and limits. Round 1 cleared 182/183 levels, leaving KA59's final level unfinished. Different players ran the two rounds, so this is not a causal comparison of document revisions.

**Version correction, 2026-10-20**: covers the 25 games and 183 levels after the 2026-04-14 update.
Each page describes controls, win/loss conditions, level-specific mechanics, and observation limits; the full text has been reviewed against the current source.
Before designing experiments, read [common semantics](COMMON.md) (actions, costs, animation, coordinates, RESET, and level jumps),
then the individual game pages. The [per-game review](REVIEW.md) lists this round's corrections; each game page ends with source functions and line numbers.

Game/engine versions and hashes are in [version validation](VALIDATION.md). Source interpretation, normal actions, and injected boundary probes are labeled separately.
The pixel validation table records full-game completion evidence on official `arcengine 0.9.3`; it does not exhaust all reachable states.
Results from older versions or the reconstructed engine remain in the historical snapshot (Chinese), and are not current rules.

## Quick reference: 25 games

| ID | One-line description | Actions | Levels | Total human baseline actions (current definition) |
|---|---|---|---|---|
| [AR25](games/ar25.md) | Move mirrors and blocks; use reflections (up to 4 images) to cover every target | 1-7 **all enabled** | 8 | 748 |
| [BP35](games/bp35.md) | Vertical cave falling puzzle; gravity defaults upward, and clicking blocks changes the terrain | 3,4,6,7 | 9 | 651 |
| [CD82](games/cd82.md) | Painting machine: reproduce the target pattern with half-canvas/triangle brushes | 1-6 | 6 | 171 |
| [CN04](games/cn04.md) | Match connectors: move and rotate shapes to pair connectors of the same type (the two types look identical) | 1-6 | 6 | 789 |
| [DC22](games/dc22.md) | Alter terrain to navigate a maze; button groups toggle bridges, portals, and cranes | 1-4,6 | 6 | 1228 |
| [FT09](games/ft09.md) | Light-grid logic: clue tiles specify colors in the eight surrounding directions | **6 only** | 6 | 208 |
| [G50T](games/g50t.md) | Record clones: turn a traveled path into a clone, rewind, switch bodies, then cooperate | 1-5 | 7 | 879 |
| [KA59](games/ka59.md) | **Billiard-style** block placement: the striker stays still, the struck block flies away; adds ice and cyclic bombs | 1-4,6 | 7 | 730 |
| [LF52](games/lf52.md) | Peg solitaire plus rail-cart logistics: transport pegs between regions, then jump-capture down to the required remaining count | 1-4,6,7 | 10 | 1339 |
| [LP85](games/lp85.md) | Ring-rotation puzzle: rotate rings to move blocks into target cells | **6 only** | 8 | 388 |
| [LS20](games/ls20.md) | Transformation maze: change shape, color, and orientation to enter the target tile | 1-4 | 7 | 776 |
| [M0R0](games/m0r0.md) | Mirrored two-player maze: break symmetry by hitting walls so the two players can meet | 1-6 | 6 | 1107 |
| [R11L](games/r11l.md) | Center-of-mass spider: move only the feet; the body follows their center of mass | **6 only** | 6 | 233 |
| [RE86](games/re86.md) | Move wireframe shapes (cross/X/diamond/square) to cover blueprint marks; shapes can be recolored and transformed | 1-5 | 8 | 1255 |
| [S5I5](games/s5i5.md) | Control articulated arms with extension sliders and rotation knobs; linkages form a tree | **6 only** | 8 | 638 |
| [SB26](games/sb26.md) | Visual programming: call subroutines by color name (recursion supported) | 5,6,7 | 8 | 213 |
| [SC25](games/sc25.md) | Cast spells through a maze: draw patterns on a 3×3 rune board to trigger scaling, teleportation, or fireballs | 1-4,6 | 6 | 350 |
| [SK48](games/sk48.md) | Extend and shift mechanical arms so the currently pressed color sequence matches a fixed template | 1-4,6,7 | 8 | 1070 |
| [SP80](games/sp80.md) | Guide water into cups: arrange barriers, then release water for simulation | 1-6 | 6 | 518 |
| [SU15](games/su15.md) | Absorb and merge (2048/Suika Game), with enemies that eat fruit | 6,7 | 9 | 361 |
| [TN36](games/tn36.md) | Binary visual programming: flip bits to encode instruction values 0-63, then execute | **6 only** | 7 | 317 |
| [TR87](games/tr87.md) | Symbol rewriting: use grammar rules to rewrite the source string into the target string | 1-4 | 6 | 414 |
| [TU93](games/tu93.md) | Maze escape with three enemy types: tracking, patrolling, and mimicking | 1-4 | 9 | 462 |
| [VC33](games/vc33.md) | Schedule extending/retracting platforms to deliver passengers to matching-color destinations | **6 only** | 7 | 447 |
| [WA30](games/wa30.md) | Grab and drag crates into zones, with cooperative and disruptive autonomous robots | 1-5 | 9 | 1843 |

## Versions and scoring

- 15 games changed version: CN04 substantially changed its level count and mechanics; some SK48 bases are no longer clickable; DC22, R11L, S5I5, and TU93 have visual/data differences. The other nine retain their mechanics but have newly obfuscated names.
- The other ten games retain their source versions, but **human baselines changed for all 25 games**. The table sums the per-level baselines in current metadata; these totals cannot be compared directly with the old definition.
- The baseline uses each level's human upper median. RHAE first computes `(baseline / actions)²` per level, capped at 1.15, then weights by level number and limits the game total according to completion. The action ratios here are not RHAE scores.

Official references: [4/14 update](https://docs.arcprize.org/changelog), [scoring methodology](https://docs.arcprize.org/methodology).
