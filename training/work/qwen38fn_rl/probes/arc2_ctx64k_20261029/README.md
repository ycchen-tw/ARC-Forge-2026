# Task pool and settings for arc2-r4 (context 64k, all tests correct only) (2026-10-29, node1)

One-off probe, not maintained. The user's spec of 10-29: from base (QAD-R INT3), 60 random eval120fix tasks + 30 random true_hard
tasks after review, output 40k / context 64k, reward = 1 only if every test is correct, group 24 × 24 groups, lr 1e-4, tp1x3 with
128 concurrency per server, W3A16, T1.0 / top_p 0.95 / top_k 20, tolerances from the template, no held-out, ambiguous tasks not
trained on.

| File | What it does |
|---|---|
| `build_pool.py` | builds the 90-task pool (seed 20261029) into `$ARC_DATA/arc2_pools/ctx64k90v2-20261029`, with `.train.txt` / `.selection.json` next to it |
| `audit549/` | the user-provided review of the 549 true_hard tasks (zip as-is; the csv's sha256 matches `manifest.json`), plus `README_en.md`, an English translation of `README_zh.md` |
| `env/arc2-r4.env` | launcher environment (`source` it, then run `launch_arc2_largegrid_int3qat.sh`) |
| `watch-r4.sh` | watchdog: unit, OOM / Traceback in the server logs, stalls with all three servers at 0 running, retract, KV, quarantine, zombie / routing M / D_cross, disk space and NUMA node 1; `bash watch-r4.sh <RL_RUN> rl-arc2r4 <worktree>` |
| `watch.py` | group statistics segmented by dispatch version (all tests correct, tokens, budget hits, also split by eval120fix / true_hard) and per-batch reward / tokens / time; `python3 watch.py <RL_RUN> [segment width]` |

Tasks are excluded only by the written per-task review verdicts (the rest are drawn at random):
- eval120fix minus 7 tasks: the 5 in `alternates.json`, plus abc82100 and faa9f03d, judged test-ambiguous in the 10-28 follow-up review.
- true_hard minus 23 tasks: the 11 the review judged `confirmed_training_ambiguity` and the 12 judged `uncertain`.
- The first version of the pool also dropped 60 tasks by "≥2 samples submit the same wrong answer that passes all train pairs";
  the user rejected that: the rule also drops genuinely hard tasks where the model often converges on another reading. That version
  was stopped right after its first group was dispatched and discarded.

Settings:
- harness [`harness-b40k-c64k-v3y-alltests-group.json`](../../configs/harness-b40k-c64k-v3y-alltests-group.json): wrap-up starts when output reaches 36k or context reaches 58k, followed by another 4k, so the output cap is 40k and context about 62k plus the tool output during wrap-up. `score_mode all_tests`.
- template [`arc2-ctx64k-tp1x3-int3qat.json`](../../configs/arc2-ctx64k-tp1x3-int3qat.json): no held-out. configure always requires a held-out task list, so it gets a nominal one, the 60 eval120fix tasks not used for training (`ctx64k90v2-20261029.untrained_eval120fix.txt`), which never actually runs.
