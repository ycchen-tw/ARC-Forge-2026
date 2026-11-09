> English translation of `README_zh.md`, added for this release. The other files in this folder are the audit as delivered; their sha256 values are in `manifest.json`.

# true_hard.csv: review results for 549 tasks

- Review scope: 549 tasks; another 9 restricted116 tasks excluded per the user's instruction.
- Confirmed that a reasonable alternative rule changes the test: 11 tasks.
- Still to be quarantined / re-reviewed: 12 tasks.
- Bounded search found no counterexample: 526 tasks; this is not the same as proven unambiguous or hard.

## How to use
Filter with audit_549.csv first. For evidence, see audit_549.json, counterexample_adjudication.json and evidence/. confirmed_training_ambiguity tasks should first get added demonstrations and then be re-verified; uncertain should not be treated as a safe pass. no_counterexample_found only means this bounded search found no counterexample.

## Whether the training examples are sufficient
This review issues no global uniqueness or sufficiency guarantee for any task. confirmed_insufficient means a reasonable train-fitting rule divergence was found that affects the test. coverage_gap_no_test_divergence means some conditions are not discriminated but the existing test still comes out the same; these should not all be treated as bad tasks. bounded_support_only is bounded support; not_assessed means the evidence is still insufficient. See each task's coverage_depth, concrete exclusion evidence, remaining hypotheses and added examples.

## Confirmed counterexamples and directions for added examples

### ind_0052
- Alternative rule: following the visible chain of junctions, first one item per station, then fill the stations in chain order; capacity, resource conservation and image rendering unchanged.
- Training gap: the five train pairs have only 3 or 4 resources; across three stations at most one is left after the first round, so nothing discriminates the allocation order of a multi-item remainder.
- Suggested added example: at least 5 resources and two stations that can still take several items; e.g. capacities [4,1,4], chain middle→right→left, reference allocation [2,1,2], alternative [1,1,3].

### ind_0074
- Alternative rule: keep obstacle-aware shortest-path distance, container order and capacity; tie-breaking among equidistant particles becomes rightmost first, then bottommost, instead of topmost first, then leftmost.
- Training gap: train does have equidistant ties at the capacity boundary, so it is not tie-free; but none of the relevant ties separates these two fixed orders.
- Suggested added example: two equidistant particles where only one fits, and the upper one is not also the more rightward one; (12,16) and (19,17) in the existing test index1 are a separating example.

### ind_0085
- Alternative rule: same column filling, overflow and 8-connected grouping; the whole gray group turns cyan only with a straight diagonal of at least 4 cells, instead of at least 3.
- Training gap: in train, the longest diagonal of colored groups is 4 or 6 and of uncolored groups 1 or 2; the case of exactly 3 is absent.
- Suggested added example: a gray 8-connected group whose longest straight diagonal is exactly 3, with none of length 4; state explicitly whether it turns cyan.

### ind_0205
- Alternative rule: the relay cost of each road is the arithmetic mean, and the road with the smaller mean is chosen; cost decoding, paths and coloring otherwise identical.
- Training gap: in train the route rankings by sum and by mean agree, even though the number of relays is not always the same.
- Suggested added example: two roads ranked oppositely by sum and by mean; e.g. upper road 4 stations summing to 9, lower road 2 stations summing to 6: sum picks the lower, mean 2.25<3 picks the upper.

### ind_0289
- Alternative rule: among all translation-only exact covers, pick the minimum by (total Chebyshev distance, total Manhattan distance, fixed row-major); the anchor is still the bbox center rounded down.
- Training gap: for the two cover solutions in each of the five train pairs, the L1 and L∞ rankings agree; the trade-off between the two distances appears only in the test.
- Suggested added example: two cover solutions with opposite distance rankings, e.g. L1/L∞ of 12/9 and 14/8, showing which one is intended.

### ind_0318
- Alternative rule: enumerate the 24 total orders, take the minimum by (maximum modified cells over the three tables, total modified cells, fixed header-order), then draw the overlapping rectangles in that order.
- Training gap: train has no observed combination that separates "sacrifice one table to lower the total error" from "spread the error evenly".
- Suggested added example: a competing solution with smaller total error but one very bad table, e.g. (2,2,6) vs (4,4,4), drawn with the intended total order.

### ind_0371
- Alternative rule: D4 shape equivalence unchanged; the two successors enter the stable partition refinement as an unordered multiset rather than as ordered left/right channels.
- Training gap: same-shape states in train do not reveal whether swapping two different successor classes should break equivalence; the ordered and unordered relation tables are identical.
- Suggested added example: two same-shape states with successors (A,B) / (B,A), A and B distinguishable; state explicitly whether they are equivalent.

### ind_0459
- Alternative rule: a same-color strip is treated as a setting update of that color's period, and only the last one is kept; the order of the other initial writes and the cellular rule unchanged.
- Training gap: train does have repeated colors, but the effects of the dropped earlier strips are all masked by later writes; it does not show whether an effect unique to an earlier same-color strip survives.
- Suggested added example: an earlier period of a repeated-color strip produces a pixel in some column that neither later strip overwrites; e.g. 2×3, 5×2, 2×5, wide enough for the difference to show.

### ind_0821
- Alternative rule: still gray out all symmetric donor objects, but the receiver's copy count becomes the number of red donor columns with at least one qualifying object, instead of the total number of qualifying objects.
- Training gap: in each of the six train pairs every red donor column has at most one qualifying symmetric object, so the object total exactly equals the number of active columns.
- Suggested added example: at least one red donor column with two qualifying symmetric objects, keeping enough empty receiver slots; the existing test0 with qualifying counts [2,1,0] separates 3 objects from 2 columns.

### q2_n551
- Alternative rule: the intersection across child cards is judged on translation-normalized geometry, ignoring color; remove all copies of the common shapes and promote one unit of each pigment present into that color's coordinate slot.
- Training gap: in every train family the common geometry appears in a single color only; nothing shows whether same-shape copies in another color also count as common.
- Suggested added example: a shape present in color A in every child, plus a same-shape copy in color B in only some children; the output shows whether B stays or is removed and promoted as well.

### q2_n560
- Alternative rule: list all D4 transforms consistent with the before/after stencil and apply them to the query; each cell keeps the strict-majority color and is filled with 0 when there is no strict majority, instead of keeping only unanimous cells.
- Training gap: no train cell has "a strict non-black majority that is not unanimous". Note in particular that train index3 already has 4 cells at 3:1, but the majority there is always 0, so both rules still fill 0.
- Suggested added example: a train pair with 4 feasible transforms and a cell with votes=[a,a,a,b], a≠0, b≠a; if unanimity is intended the output should be 0. E.g. test1(0,0)=[1,0,1,1].

## Difficulty and test quality, listed separately
The core test input of cp_098 can be recovered from training; q2_n343 reuses a geometric template. The tests of q2_n350, n367, n493, n508 and others can be passed by some simplified rules that training already rules out; that is a problem of the test's discriminative power, not the same kind of train ambiguity. Some tasks are only moderately structured; historical models' token use or failures are not measurements of human difficulty.

## Method and limitations
Hypotheses were first formed by reading the demonstrations; many tasks still needed the author's rule as a reference to rebuild and test. Saved outputs were checked with independent implementations and a bounded set of natural alternative rules was searched; no model was retrained and the original task pool was not modified.

Pinned CSV commit: 910601e01466dd2f4498cf8c031423142241c52f. quality200, composition and size use the corresponding pinned PR versions; every task has source_ref/source_path. These sources were not compared bit by bit against the payload of node1's 10/27 run; the conclusions apply to the fixtures the report specifies.

The rerun programs and evidence are kept in evidence/ under their original relative dirs; some programs keep their execution-time paths, so set the locations from the adjacent manifest / input parameters before rerunning. A matching rebuild of the reference alone does not imply that the rule is unique.
