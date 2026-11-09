# eval120fix: the ARC-AGI-2 public evaluation with defects fixed

The 120 evaluation tasks of upstream `arcprize/ARC-AGI-2` f3283f7 (2025-05-15, still HEAD as of 2026-10-25), with
confirmed defects changed cell by cell. `build.py` writes them to `$ARC_DATA/arc2_eval120fix/evaluation/` (run_arc2's
`--tasks-dir`), with `MANIFEST.json` next to it.

- [`fixes.json`](fixes.json): 6 tasks, 29 cells; each task is first checked against the upstream sha256, then each cell
  against its original value, before changing anything.
  - Wrong train cells, 4 tasks (the test answer was right all along; a faithful program cannot pass that pair):
    581f7754, b6f77b65, a6f40cea match upstream fix PRs #52 / #61 / #64 cell for cell;
    e12f9a14 was never reported upstream; the added beam equals the output rebuilt by seed_bank.
  - Cosmetic test-side defects, 2 tasks (input and output changed together; the answer of a program that copies the
    input is unchanged): 1ae2feb7 (PR #40), 53fb4810 (PR #50).
  - Reviewed but not changed (7ed72f31, 291dc1e1, 9bbf930d, 5dbc8537): reasons in `reviewed_not_changed`.
- [`alternates.json`](alternates.json): 5 tasks, 6 "other answers the train pairs cannot rule out", used only for the
  lenient score: 7b80bb43 (2), 800d221b, c7f57c3e, 4e34c42c test 0, 88e364bc test 0 (the PR #62 version).
- `split.json`: dev / holdout, 60 tasks each, stratified by per-task score over the 14 rounds of 10-23
  QAD. Inside the loop only dev failures are read; holdout is only scored.
- `score.py`: strict (the fixed official answers, equal to the run's summary score) / lenient (with
  alternates), per round or as a paired bootstrap of two arms.
- `views.py`: transposes / rotates / flips whole tasks into another `--tasks-dir`
  (`$ARC_DATA/arc2_eval120fix_<view>/`); score.py maps the answers back to the original orientation.

Basis: the 10-24 hard20 review (the 4 wrong-cell tasks re-verified cell by cell on 10-25) and upstream issues / PRs. The
other 99 tasks were not reviewed by hand; passing the seed_bank rebuild does not mean error-free (the seeds of b6f77b65
and a6f40cea only passed with rules specially made for the wrong cells).
